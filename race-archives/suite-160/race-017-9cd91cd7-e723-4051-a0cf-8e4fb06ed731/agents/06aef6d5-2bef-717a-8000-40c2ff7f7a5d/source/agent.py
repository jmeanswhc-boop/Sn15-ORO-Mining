"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 36             # model calls of the main loop, and of the whole episode: passing episodes stay under these,
ALL_CALLS = 46              # and past them the 300 s limit ends the episode with no order at all
SPEND_SOFT = 0.055
SPEND_HARD = 0.068         # a race run is capped near $0.055 a task on average; the costliest episodes are nearly all failures
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 320
ROW_CHARS = 160             # a search row's text in the conversation: enough to choose what to open, and it is resent every turn
OBS_LIMIT = 15000
HISTORY_LIMIT = 100000      # characters of conversation before old results shrink: long ones slow each call toward the 300 s limit
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 14             # further search pages Kyo reads itself for a cheapest, most or least pick
SWEEP_PAGES = 7             # further pages of one search Kyo reads itself under a pick rule (all within the paging reach)
SWEEP_ROWS = 30             # buyable rows of such a search listed for a cheapest pick, cheapest first
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
EARLY_TRAILS = 3            # chain reads while the agent works (after a page view), so a chain's value reaches the ledger
COMPARE_FIRST = 5           # listings sent side by side when the store declares no limit; a declared limit starts at half (stores cap below it)
WIDEN_EACH = 3              # listings of each item whose attributes the question planner should have seen
FOLLOW_STEPS = 14           # most steps Kyo spends following the pages requirements point at itself
FOLLOW_ROUNDS = 10          # rounds (a search, a next page or a further side-by-side read) Kyo spends on the links of those chains
LIST_FROM = 3               # messages a world must allow before the first question names several properties
LIST_KEYS = 8               # properties of one item a message names: the shopper takes up one of them, nearly always the first
                            # one that is on their mind, so a long list wastes no message on properties nobody holds a view on
PLAN_EACH = 6               # properties per item the planner ranks; the item's most carried keys follow them in the list
PROBE_LEFT = 24             # steps that must be left for Kyo's one test of the store's keyword search (one step)
SEARCH_PLANS = 3            # light-model search plans per episode at most
FOLLOW_ROWS = 6             # likely rows of one search result Kyo reads while looking for a chain's link
LONG_STEPS = 70             # a stated step limit this long: after LATE_PLAIN main-loop turns, the light model's turns go
PLAIN_AFTER = 24            # model calls in an episode after which the main loop's light-model turns go without hidden reasoning
LATE_PLAIN = 26             # without hidden reasoning (several times faster), so the long many-item tasks finish in time;
                            # earlier turns keep it (without it the model gives up on searches far more often)

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A keyword search that returns no rows usually means one of its words appears on no listing: drop words (keep the product noun) or browse its category instead; an empty result never shows that the store lacks the item. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Kyo learns this store's keyword search from its results (how many of a query's words it reads, whether every word must match, filters it ignores, how deep paging goes) and reports it after searches, with a search plan of the best next queries: follow those notes and the plan when you search.
- Some details (what other products an item suits, codes it carries) may appear only in a listing's description, which a keyword search may not read: browse its category and read the rows' descriptions.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page settles the value. The order counts only when every page of that chain was opened. The requirement's own relation word says what that value means for your product: for the same or a matching value, buy that value; for a value unlike or different from it, any other value fits (only listings sharing that exact value are out, so never end without a purchase because of it alone); for a value that suits it, goes well with it or is a compatible one, buy the partner value the last page names for its own value (the page may name it outright or state the store's rule of paired values: look up the page's own value there), never the page's own value. The ledger shows the value Kyo read once the chain is done; from then on that chain's pages count as opened, so never search for them or open them again.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- A requirement that holds only in one case depends on what another requested item turns out to have: choose that other item first, read its record, and apply the requirement only when its case holds (when it does not, the requirement asks nothing). A list of the models a product works with can cover a series by a pattern (a letter such as x standing for any digit) and name exceptions: it covers a model the pattern matches unless that model is named as an exception.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing. When the ledger says the closest listings are to be put forward if nothing meets everything, Kyo names them in that ending itself.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing in the request's own relation words, copied as written and never swapped for "same as" or any other paraphrase ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product, "fits": the number (counting from 1 in this list) of the other requested item this product has to fit or work with by the request, else 0}. Never put the order's own budget, stock or quantity into musts; a price limit the request sets on one item alone is a must of that item ("price: at most 40"). A requirement that holds only in one case, depending on what another item of the order turns out to have (unless, except when, only if, in case), is ONE must of the product it is about that keeps its case in the request's words ("lid colour: green, except when the kettle's body is steel"); never drop the case and never make it a must of the other item. A must naming the model of a vehicle or device it must suit keeps that model code. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item for every item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule, nor is a rule about which listing to put forward when nothing fits.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A must about another item of this same order (one that has to fit or suit whatever gets bought) is no lookup.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"if_nothing_fits": "closest" when the request wants the nearest listing put forward when nothing meets everything, "decline" when it wants the task ended without buying, null when it says nothing.
"closest_rule": the request's own rule for choosing that closest listing, in at most 25 words, else "".
"keyword_search": how to search the store by free words: {"tool": the exact tool, "words": its parameter that takes the words, "fixed": the other arguments that tool needs for a search by words, such as a mode (exact allowed values), else {}}, or null when no tool searches by words.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (when two single-listing tools exist, the other one: it shows the listing's page or full record, such as its description, notes, price and stock; null when "open" is the only single-listing tool), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You rank the properties a shopping agent asks a shopper about. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A message names several properties of one item, best first, and the shopper answers about one of them, nearly always the first of them that is on their mind, so the likeliest properties go first.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key, how many different values they show (over the few listings read), and example values. An unwritten preference is nearly always on one of the listed keys of its item.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them. Name it by the item's listed key it points at: a hint about who will wear it points at gender; a hint about looks points at colour, unless the sentence is about a part of the look that another listed key names (its shape, pattern or finish).
2. Then the listed keys, ranked by how many of the item's listings carry the key: most unwritten preferences are on the keys nearly all of the item's listings carry, even when the few listings read show a single value for it; a key seen on only one or two of them rarely hides one.
Skip a property the request already settles for that item, a second name for a property already chosen (keys that differ only in spelling or by a prefix are one property), a key that lists purchase variants (a mix of sizes, colours and pack counts, often called options), identifiers, brand, model, condition, warranty and other logistics.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
For each item choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": false, "unclear": false, "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question that names one property, when the shopper said they do not mind about it; never for a question that names several properties.
- "crowded": true when an answer wants a single property per question, else false. "unclear": true when an answer says the shopper did not follow the question and asks for it in plainer words, else false.
- An answer saying the shopper does not mind, has no view or will take whatever is chosen states no need, whatever it adds about their day, home or habits.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted, copied exactly as listed above"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; a requirement on price is judged on the listing's own price; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value; a rule of suiting or going well with another listing's value asks for the partner value that listing's page names for its own (the value the ledger shows), never its own value; a requirement that holds only in one case is judged with the record of the cart line of the other item it depends on, and asks nothing when that case does not hold; a statement of the models a product works with can cover a series by a pattern (a letter such as x standing for any digit) and name exceptions: it covers a model the pattern matches unless that model is named as an exception. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; a requirement on price is judged on the listing's own price; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; a rule of suiting or going well with another listing's value asks for the partner value that listing's page names for its own (the value the ledger shows), never its own value; a requirement that holds only in one case is judged with the record of the cart line or chosen listing of the other item it depends on: when that case does not hold, the requirement asks nothing; a statement of the models a product works with can cover a series by a pattern (a letter such as x standing for any digit) and name exceptions: it covers a model the pattern matches unless that model is named as an exception; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

FITS = """A shopping agent has just read a requirement's value from another listing's page. Check the OPENED listing records below against every requirement in force, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": {"item number": [ids of listings that appear to meet every requirement of that item, best by the pick rule first, at most 3]}}"""

CLOSEST = """A shopping agent found no sure fit for everything and is about to put forward the listings that come nearest instead. For each requested item, choose from the records below the listing of that kind of product that leaves the smallest number of requirements unmet (being in stock is a requirement too, and so is the budget, judged for each item on its own: its price times its quantity against the whole budget), with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: choose one only when no OPENED record of that kind exists. Among listings that miss equally many, take the cheapest (a rule that also accepts a dearer one up to some margin is met by the cheapest too), unless the request's own rule for the closest says otherwise. A line already in the cart stays unless a record clearly misses fewer.
Return ONE JSON object and nothing else: {"lines": [{"item": item number, "id": listing id, "misses": how many requirements it misses}], "budget_blocks": true when the records show, for every requested item, a listing that meets all of that item's requirements apart from its price, and yet no order of such listings stays inside the budget (the budget alone is what stops a fitting order), else false}, one line per item that any record can serve."""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read). A link is the listing whose own record shows the values its description gives: another listing of the same shop or kind with other values is not that link, and what its page says must not be followed. The request shows the relation each requirement uses in the shopper's own words, and those words decide it wherever the requirement below words it otherwise.
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement's number below, "links": [the listing ids along this chain so far, in order, from the records below, stopping at the first one whose page is not OPENED], "unmatched": [the values in the description of the last link that its own record does not show; a link with any such value is the wrong listing and does not belong in "links"], "done": true when the last link is an OPENED page that states the property's value itself and does not point the reader at a further listing for that property (a page that does is a middle link, whatever value of its own it shows), else false, "about": when not done, how the requirement or the last page describes the next listing to find, in at most 20 words, else "", "find": when not done, a short keyword query that would find that next listing (its shop's name first, then the kind of product, then a few distinctive values from how it is described; never the name of the property being looked up), else "", "own": the property's value that last page states for its own listing, or "", "partner": the value that page pairs with its own value (named outright, or through a rule of paired values the page states, looked up by its own value), or "", "value": what the product being bought must have, read by the requirement's relation word, by its meaning in the shopper's dialect: the own value for the same or a matching value, and for wordings that allow only that value ("nothing but that one"); "anything but " and the own value for a value unlike or different from it; the partner for a value that suits it, goes well with it or is a compatible one (the own value when the page names no partner); "" when not done}]}"""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

POINT = """A shopping agent is about to check out the cart lines below. A listing's own text can send the reader to another listing to learn more about it (what else it suits, its full details, a companion part), naming that other listing by a code, a number or a title (the models or devices it works with are not such listings). For each cart line whose record does that, give a short keyword query that finds the listing it names: the code exactly as written when there is one, else a few of its distinctive words. Return ONE JSON object and nothing else: {"find": [{"line": the cart line's listing id, "query": "..."}]}, with [] when no record sends the reader on."""

SEARCHPLAN = """You plan the next searches of a shopping agent in one online store. The store's keyword search follows its own rules, learned from the searches run so far (STORE SEARCH below); obey them exactly. For each requested item whose fitting listings are not all found yet, give up to three keyword queries to run next, best first. Build each from the item's requirements and the words the store's own titles use for that kind of product, keep it within the number of words the store reads, and put the most telling words first: a required value with the product word, or a shop's name with the kind of product when a requirement names a shop. Never repeat a query already run. Use the words the titles shown use for that kind of product: a requirement's figure or unit seldom appears in a title, so leave figures out unless titles carry them. Browse the category only when paging reaches all its listings; a larger category shows only part of itself in a fixed order, so cover it with several narrow queries instead (each word the titles use for that kind of product, alone or with one required value). When the pick is the cheapest, choose queries narrow enough that every match fits on the pages the store lets the agent reach, so the cheapest fitting listing cannot hide beyond them. Return ONE JSON object and nothing else: {"items": [{"item": item number, "queries": ["..."], "browse": true or false, "note": "at most 25 words"}]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null, "title": the key, inside a row, of the listing's title or name, or null}"""


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = text.find("{"), text.rfind("}")
    if 0 <= left < right:
        part = text[left:right + 1]
        # the second try forgives a comma before a closing bracket
        for attempt in (part, re.sub(r",\s*([}\]])", r"\1", part)):
            try:
                got = json.loads(attempt)
                return got if isinstance(got, dict) else None
            except ValueError:
                continue
    return None


def _words(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return True


def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema; list what cannot be fixed."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and not any(_fits_kind(value, k) for k in kinds):
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                value = int(number) if kind == "integer" or number.is_integer() else number
                break
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
                break
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
                break
            if kind == "array" and not isinstance(value, list) and value is not None:
                value = [value]
                break
        if not any(_fits_kind(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f"{where}: allowed values are {options}")
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        value = value[: schema["maxLength"]]
    if isinstance(value, str) and isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [_shape(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            for name in [n for n in value if n not in props]:
                faults.append(f"{where}.{name} is not an allowed argument")
        for name in schema.get("required") or []:
            if name not in value:
                faults.append(f"{where}.{name} is required")
        value = {k: _shape(v, props.get(k), f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
    return value


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def _id_like(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_id_like(v, "id" in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ("id" in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether a list in the result holds a record carrying this id (a cart listing still showing the line)."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,3}")


def _scalar(text: str) -> Any:
    """A field's text as the value it writes: none, true/false, a whole or decimal number, else the text."""
    low = text.strip()
    if low in ("None", "null", "NULL", ""):
        return None
    if low in ("True", "true"):
        return True
    if low in ("False", "false"):
        return False
    if re.fullmatch(r"-?\d+", low) and len(low) < 16:
        return int(low)
    if re.fullmatch(r"-?\d+\.\d+", low) and len(low) < 20:
        return float(low)
    return low


def _text_parts(text: str, sep: str, mark: str, known: set[str] | None) -> list[list[str]] | None:
    """A line cut into [key, value] parts at `sep`, each part starting "key<mark>value"; a piece that does not start with
    a key (a name of a few words, one word unless its neighbours share that name) belongs to the part before it."""
    parts: list[list[str]] = []
    for piece in text.split(sep):
        key, found, value = piece.partition(mark)
        key = key.strip()
        if found and _KEY.fullmatch(key) and (not known or key in known or " " not in key or not parts):
            parts.append([key, value])
        elif parts:
            parts[-1][1] += sep + piece
        else:
            return None
    return parts


def _text_record(text: Any, known: set[str] | None = None) -> dict[str, Any] | None:
    """A listing written as one line of "key: value" parts joined by " | " (three parts at least, an id among them): its
    record; a part holding "k=v; k=v" pairs becomes a nested record of those."""
    if not isinstance(text, str) or text.count(" | ") < 2 or ": " not in text:
        return None
    parts = _text_parts(text, " | ", ": ", known)
    if not parts or len(parts) < 3 or len({k for k, _ in parts}) < len(parts):
        return None
    out: dict[str, Any] = {}
    for key, value in parts:
        inner = _text_parts(value, "; ", "=", None) if "=" in value else None
        if inner and len(inner) >= 2:
            out[key] = {k: _scalar(v) for k, v in inner}
        else:
            out[key] = _scalar(value)
    if not any(_id_like(v) for v in out.values() if isinstance(v, str)):
        return None
    return out


def _decoded(node: Any, known: set[str], depth: int = 0) -> Any:
    """A store reply with every listing written as text turned into its record. In a list of such lines only the keys
    most of them share count as keys (a title may itself hold " | " or ": "); those keys are kept for single lines later."""
    if depth > 6:
        return node
    if isinstance(node, dict):
        return {k: _decoded(v, known, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        texts = [v for v in node if isinstance(v, str)]
        if len(texts) >= 3:
            counts: dict[str, int] = {}
            for text in texts:
                for key, _ in _text_parts(text, " | ", ": ", None) or []:
                    counts[key] = counts.get(key, 0) + 1
            shared = {k for k, n in counts.items() if n * 2 >= len(texts)}
            if len(shared) >= 3:
                known.update(shared)
        out = []
        for value in node:
            record = _text_record(value, known or None) if isinstance(value, str) else None
            out.append(record if record is not None else _decoded(value, known, depth + 1))
        return out
    if isinstance(node, str):
        record = _text_record(node, known or None)
        return record if record is not None else node
    return node



_FIT_CODE = re.compile(r"\b([A-Z][A-Za-z]{1,7})[- ]?(\d{2,6})\b")
_FIT_WILD = re.compile(r"\b([A-Z][A-Za-z]{1,7})[- ]?(\d{1,5})([xX]{1,4})\b")
_FIT_SERIES = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,6}\b")
_FIT_LONE = re.compile(r"\b[A-Z]{1,5}\d{1,3}\b")


def _fit_text(node: Any, depth: int = 0) -> str:
    """Every string a record holds, joined (keys left out)."""
    if depth > 4:
        return ""
    if isinstance(node, dict):
        return " | ".join(_fit_text(v, depth + 1) for v in node.values())
    if isinstance(node, list):
        return " | ".join(_fit_text(v, depth + 1) for v in node[:40])
    return node if isinstance(node, str) else ""


def _fit_codes(text: str) -> set[tuple[str, str]]:
    return {(a.upper(), b) for a, b in _FIT_CODE.findall(text)}


def _fit_covers(wild: tuple[str, str, int], code: tuple[str, str]) -> bool:
    return code[0] == wild[0] and code[1].startswith(wild[1]) and len(code[1]) == len(wild[1]) + wild[2]


def _fit_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;!?])\s+|\s\|\s|\n", text) if s.strip()]


def _part_reading(title: str, text: str) -> dict[str, Any]:
    """What a part's text says it fits: codes named, patterns, codes a pattern leaves out, other parts whose page holds
    the rest of its list, and codes its text gives to other parts."""
    own = _fit_codes(title)
    prefixes = {c[0] for c in own}
    named: set[tuple[str, str]] = set()
    wild: list[tuple[str, str, int]] = []
    left: set[tuple[str, str]] = set()
    points: set[tuple[str, str]] = set()
    gives: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for sentence in _fit_sentences(text):
        codes = _fit_codes(sentence) - own
        found = [(a.upper(), d, len(x)) for a, d, x in _FIT_WILD.findall(sentence)]
        others = {c for c in codes if c[0] in prefixes}
        devices = codes - others
        if others:
            if devices or found:
                for other in others:
                    gives.setdefault(other, set()).update(devices)
            else:
                points |= others
            continue
        for code in devices:
            (left if any(_fit_covers(w, code) for w in found) else named).add(code)
        wild += found
    series = set(_FIT_SERIES.findall(title))
    return {"own": own, "named": named, "wild": wild, "left": left, "points": points, "gives": gives, "series": series}


def _fit_says(device_text: str, part: dict[str, Any]) -> bool | None:
    """True when the part's reading fits the device's text, False when it names models and none is the device's, None
    when the part's text names no model at all (nothing read)."""
    codes = _fit_codes(device_text)
    if codes & part["named"]:
        return True
    if any(_fit_covers(w, c) for w in part["wild"] for c in codes if c not in part["left"]):
        return True
    lone = set(_FIT_LONE.findall(device_text))
    if part["series"] & lone:
        return True
    if part["named"] or part["wild"] or part["series"]:
        return False
    return None


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.fits = 0                       # the item (1-based) this product has to fit or work with, 0 for none


class Kyo:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get("max_steps"), float):
            steps = int(self.view["max_steps"])
        self.max_steps = steps if steps and steps > 0 else 30
        self.plain_loop = False
        self.reasoning_refused = False
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        self.credit_fails = 0               # calls in a row refused for credit after their waits
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        self.near_ids: list[str] = []
        self.pointed: set[str] = set()
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.ref_early: dict[str, str] = {}
        self.note_flags: dict[str, bool] = {}
        self.next_find: dict[str, str] = {}
        self.follow_reads = 0
        self.closest_tries = 0
        self.nothing_fits = False
        self.followed: set[str] = set()
        self.empty_filters: list[tuple[str, str, str]] = []
        self.blind: dict[tuple[str, str], str] = {}
        self.filters_ok: set[tuple[str, str, str]] = set()
        self.trails = 0
        self.trail_mark: tuple[frozenset[str], tuple[str, ...]] | None = None  # pages opened and requirements at the last early read
        self.early_trails = 0
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        # this store's keyword search, learned from its results (first pages: words, other arguments, total, rows)
        self.search_log: list[dict[str, Any]] = []
        self.lead_read = 0                  # a query's word at this position was shown to change what a search returns
        self.lead_cap: int | None = None    # the words after this position were shown to change nothing
        self.word_rule = ""                 # "all": a further word narrowed the matches; "any": it widened them
        self.ignored_args: set[str] = set()  # arguments a keyword search was shown to ignore
        self.reach: int | None = None       # rows paging reaches, when fewer than a search reports
        self.search_known: tuple[str, ...] = ()
        self.probed = False
        self.plans = 0
        self.queries_run: list[tuple[str, ...]] = []
        self.cat_sizes: dict[str, int] = {}
        self.over_reach = 0                 # the largest total a search reported beyond the rows paging reaches
        self.page_kind: dict[str, tuple[str, int]] = {}  # per search tool: "number" or "offset" paging, rows on its first page
        self.reach_told = False
        self.paged_obs: list[Any] = []
        self.text_keys: set[str] = set()     # field names a store's text-written listings share
        self.hidden_rows: set[str] = set()  # rows a keyword search returned though their own text shows none of its words read
        self.hidden_told: set[str] = set()

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Tools that show a listing's page: the offer view when the store has one apart from the attributes view
        (an attributes list alone is not the page), else the single-listing view; and the side-by-side view."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the side-by-side view shows attributes: True or False once seen, None before."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or, not tried yet, the store has a separate
        attributes view (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              plain: bool = False) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, plain)
                if message is not None:
                    self.credit_fails = 0
                    return message
                if status == 400 and plain and not self.reasoning_refused:
                    # the route refuses the reasoning setting: the same call goes again without it, and later ones skip it
                    self.reasoning_refused = True
                    continue
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                # one refusal can be the route's, not the money's: only three calls in a row refused end the model work
                self.credit_fails += 1
                if self.credit_fails >= 3:
                    self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   plain: bool = False) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if plain and model == LIGHT and not self.reasoning_refused:
            body["reasoning"] = {"enabled": False}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for number, model in enumerate(models):
            if number and model == HEAVY and self.cost >= SPEND_SOFT:
                # past the soft spend line the heavy model is no fallback: on a long prompt one such call cost as much as a
                # whole episode's light-model turns, and the episodes that spend this much rarely end in a paid order
                return None
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def open_cart_pages(self) -> None:
        """Every cart listing's page opened with the page view before the checkout (a listing read only through the
        attributes view or a search row is not opened), one step each, the checkout's own step kept."""
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        if not tool or not target:
            return
        closed = [i for i in self.cart_ids() if i not in self.opened and i not in self.gone]
        self.opening = True
        try:
            for ident in closed:
                if self.left() < 2 or self.finished:
                    break
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.offer_seen[ident] = self.step
                self.observe(tool, reply.get("observation"))
        finally:
            self.opening = False
        if closed:
            self.trace.append({"kind": "cart_pages", "ids": closed})

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if not getattr(self, "opening", False) and not self.finished and self.is_buy(name, args):
            self.open_cart_pages()
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                if "observation" in data:
                    data["observation"] = _decoded(data["observation"], self.text_keys)
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that): those the answer
                    # shows, since a refused call answers without them
                    asked = [v.strip() for value in args.values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
                    shown = _dump(seen)
                    self.opened.update(v for v in asked if v in shown)
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "finaliz" in low:
                    self.finished = True
                    return None
                found = re.search(r"expected (\d+)", body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                if "reuse" in low or "different action" in low:
                    self.step += 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    # ---------- reading the request ----------
    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
            self.items[-1].fits = _count(entry.get("fits")) or 0
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        limit = _count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get("ask"):
            self.msg_limit = 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        self.plain_loop = self.max_steps >= LONG_STEPS
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # the messages the steps at the order take; a step at an ending without a purchase gets what the interview
            # leaves (with few messages, a held-back one cost the second question in a fifth of race 13's tasks)
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (read along the pages it points at, by its relation: {self.ref_values[m]})" if m in self.ref_values
                          else f" (read so far along the pages it points at, by its relation: {self.ref_early[m]})" if m in self.ref_early else "")
                     for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        pick = brief.get("pick")
        if pick == "cheapest":
            parts.append("pick: the cheapest listing that fits (lowest price)")
        elif pick in ("most", "least"):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace("  ", " "))
        else:
            parts.append("pick: any listing that fits everything")
        if brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        for wish in _items(brief.get("wishes")):
            parts.append(f"wish (tie-breaker only): {wish}")
        if self.fit_links():
            parts += self.fit_cart()
        for rule in _items(brief.get("rules")):
            parts.append(f"rule from the request: {rule}")
        if self.closest_mode():
            rule = str(brief.get("closest_rule") or "").strip()
            parts.append("if nothing meets everything: end without a purchase; Kyo names the closest listing of each item (the one missing the fewest requirements, stock and budget count) in that ending" + (f" ({rule})" if rule else ""))
        elif brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        rules = self.search_facts()
        if rules:
            parts.append("this store's keyword search (learned from its results): " + "; ".join(rules))
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def fit_links(self) -> list[tuple[int, int]]:
        """(part item, device item) pairs the request ties by fit, 1-based."""
        return [(n, item.fits) for n, item in enumerate(self.items, 1) if 1 <= item.fits <= len(self.items) and item.fits != n]

    def of_item(self, ident: str, number: int) -> bool:
        """The listing's category is the best match among the requested items for item `number`."""
        cat_key = self.category_key()
        record = self.full.get(ident) or self.rows.get(ident) or {}
        label = str(record.get(cat_key) or "") if cat_key else ""
        if not label:
            return False
        scores = [self.shelf_match(label, item) for item in self.items]
        return scores[number - 1] > 0 and scores[number - 1] == max(scores)

    def full_title(self, ident: str) -> str:
        """The listing's whole title (its row's title key, else its page's)."""
        keys = [shape["title"] for shape in self.shapes.values() if shape and shape.get("title")]
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            title = next((str(record[k]) for k in keys if isinstance(record.get(k), str) and record[k].strip()), "")
            if title:
                return title
        return ""

    def fit_text_of(self, ident: str) -> str:
        return _fit_text(self.rows.get(ident) or {}) + " | " + _fit_text(self.full.get(ident) or {})

    def fit_readings(self, part_no: int) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
        """Each seen listing of the part item read for the models it fits, with the lists other parts' pages give it
        and the lists of the parts its page sends the reader to; and, per part, the codes of parts still to be opened."""
        ids = [i for i in self.seen_order if self.of_item(i, part_no)]
        readings = {i: _part_reading(self.full_title(i), self.fit_text_of(i)) for i in ids}
        owner = {code: i for i, r in readings.items() for code in r["own"]}
        merged: dict[str, dict[str, Any]] = {}
        unread: dict[str, list[str]] = {}
        for i, r in readings.items():
            m = dict(r)
            m["named"] = set(r["named"])
            m["wild"] = list(r["wild"])
            for code in r["points"]:
                j = owner.get(code)
                if j and j != i and (readings[j]["named"] or readings[j]["wild"]):
                    m["named"] |= readings[j]["named"]
                    m["wild"] += readings[j]["wild"]
                else:
                    unread.setdefault(i, []).append(f"{code[0]} {code[1]}")
            merged[i] = m
        for r in readings.values():
            for code, devices in r["gives"].items():
                j = owner.get(code)
                if j in merged:
                    merged[j]["named"] |= devices
        return merged, unread

    def fit_table(self) -> str:
        """For each fit the request asks for: the seen devices with the seen parts whose texts say they fit them."""
        out: list[str] = []
        cheapest = self.brief.get("pick") == "cheapest"
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        for part_no, dev_no in self.fit_links():
            readings, unread = self.fit_readings(part_no)
            if not readings:
                continue
            parts = [i for i in readings if self.buyable(i)]
            devices = [i for i in self.seen_order if self.of_item(i, dev_no) and self.buyable(i)]
            if cheapest:
                parts.sort(key=price)
                devices.sort(key=price)
            rows: list[str] = []
            seen_codes: set[tuple[str, str]] = set()
            for d in devices:
                text = self.fit_text_of(d)
                codes = _fit_codes(text)
                if not codes:
                    continue
                seen_codes |= codes
                good = [p for p in parts if _fit_says(text, readings[p]) is True]
                shown = ", ".join(f"{p} ({price(p):g})" if price(p) != float("inf") else p for p in good[:5])
                rows.append(f"{d}" + (f" ({price(d):g})" if price(d) != float("inf") else "") + ": " + (f"parts that fit it: {shown}" + (f" and {len(good) - 5} more" if len(good) > 5 else "") if good else "no part read so far fits it"))
                if len(rows) >= 6:
                    break
            lonely = []
            for p in parts:
                r = readings[p]
                if (r["named"] or r["wild"]) and not any(_fit_covers(w, c) for w in r["wild"] for c in seen_codes if c not in r["left"]) and not (r["named"] & seen_codes):
                    names = [f"{a} {b}" for a, b in sorted(r["named"])][:3] + [f"{a} {d}{'x' * n}" for a, d, n in r["wild"]][:2]
                    lonely.append(f"{p} fits {', '.join(names)}")
            mute = [p for p in parts if _fit_says("", readings[p]) is None and p in self.full][:4]
            text = (f"[fits read from model codes: item {part_no} has to fit item {dev_no}; a part fits a device when its text names one of the device's"
                    " model codes (a device's page may give other names for the same model), a code pattern ending in x's that covers it without"
                    " naming it again to leave it out, or the part series that begins the part's code]")
            if rows:
                text += "\n" + "\n".join(rows)
            if lonely:
                text += "\nparts whose models no device seen so far carries: " + " | ".join(lonely[:4])
            pointed = [f"{p} (rest of its list on the page of part {', '.join(codes)})" for p, codes in unread.items() if p in parts][:3]
            if pointed:
                text += "\nparts whose page sends you to another part's page for more models (open that part to read them): " + " | ".join(pointed)
            if mute:
                text += "\nopened parts whose text names no model: " + ", ".join(mute)
            out.append(text)
        return "\n".join(out)

    def fit_cart(self) -> list[str]:
        """For each fit the request asks for: whether each cart line of the part item fits the cart's device line."""
        lines: list[str] = []
        cart = self.cart_ids()
        for part_no, dev_no in self.fit_links():
            devices = [i for i in cart if self.of_item(i, dev_no)]
            parts = [i for i in cart if self.of_item(i, part_no)]
            if not devices or not parts:
                continue
            readings, _ = self.fit_readings(part_no)
            for p in parts:
                for d in devices:
                    says = _fit_says(self.fit_text_of(d), readings.get(p) or _part_reading("", ""))
                    if says is None or not _fit_codes(self.fit_text_of(d)):
                        continue
                    lines.append(f"fit read from model codes: cart line {p} {'fits' if says else 'does NOT fit'} cart device {d}")
        return lines

    def status(self) -> str:
        left = self.left()
        text = f"[Kyo] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += (" FINAL STEPS: check out now; Kyo carts the closest listing of any item still without a line." if self.closest_mode() else
                     " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).")
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif left <= len(self.items) + 7 and self.buys == 0 and self.cart_lines and self.lines_missing() == 0:
            text += " Every item has a cart line: check out now, so a change the store announces at the checkout still leaves steps to act on."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.buys == 0 and left > self.tail() + 6:
            questions = 0 if self.asked else max(0, self.msgs_free())
            text += (f"\n[step plan] {left} steps left: " + (f"{questions} go to the interviewer's questions, " if questions else "")
                     + f"keep {self.tail() + len(self.items) + 3} for the cart, the checkout and a change the store may announce at the checkout; the rest is for finding and checking the items.")
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.ref_early]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page settles the value, read by the requirement's relation word] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.buys == 0 and self.fit_links():
            fits = self.fit_table()
            if fits:
                text += "\n" + fits
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(_id_like(r.get(self.id_key), True) for r in rows):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    # a flag of the listing's own record turning from true to false is read as "can no longer be bought"
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)

        walk(observation, 0)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get("enum") or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)

        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str], int]]:
        """(category, key name, listings carrying it, sample values, how many different values), most carried first per
        category."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            group["keys"] = self.fold_spellings(group["keys"])
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]], len(set(slot["per"].values()))))
        return table

    def fold_spellings(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Keys that are one property under several spellings (a prefix, a longer or run-together name) become one key.
        Two keys join when no listing carrying both shows them differing and either one's words hold the other's (with a
        listing or a value in common) or they share a word over three or more listings that carry both. A joined key
        counts every listing carrying any of its spellings and goes by the spelling most listings use (fewer words on a
        tie): the name a shopper knows it by."""
        def bare(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", text)

        names = list(keys)
        head = {name: name for name in names}

        def top(name: str) -> str:
            while head[name] != name:
                name = head[name]
            return name

        for at, one in enumerate(names):
            for two in names[at + 1:]:
                w1, w2 = _stem(one), _stem(two)
                if not w1 & w2:
                    continue
                p1, p2 = keys[one]["per"], keys[two]["per"]
                common = [i for i in p1 if i in p2]
                if any(bare(p1[i]) != bare(p2[i]) for i in common):
                    continue
                nested = w1 <= w2 or w2 <= w1
                alike = bool({bare(v) for v in p1.values()} & {bare(v) for v in p2.values()})
                if (nested and (common or alike)) or len(common) >= 3:
                    head[top(two)] = top(one)
        bundles: dict[str, list[str]] = {}
        for name in names:
            bundles.setdefault(top(name), []).append(name)
        out: dict[str, dict[str, Any]] = {}
        for members in bundles.values():
            members.sort(key=lambda n: (-keys[n]["n"], len(n.split()), n))
            per: dict[str, str] = {}
            values: list[str] = []
            for name in members:
                for ident, text in keys[name]["per"].items():
                    per.setdefault(ident, text)
                values += [v for v in keys[name]["values"] if v not in values]
            out[members[0]] = {"n": len(per), "values": values, "per": per}
        return out

    def shelf_match(self, label: str, item: Item) -> int:
        """3 when a store category label is the item's own shelf or name, 2 when all its words are among the item's, 1
        when they share a word of four letters or more, 0 otherwise."""
        theirs = _stem(label)
        mine = _stem(item.name) | _stem(item.shelf or "")
        if not theirs:
            return 0
        if _words(label) == _words(item.name) or (item.shelf and _words(label) == _words(item.shelf)):
            return 3
        long = {w for w in mine if len(w) >= 4}
        return 2 if theirs <= mine else 1 if any(len(w) >= 4 and w in long for w in theirs) else 0

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                elif isinstance(value, dict):
                    single = self.ident(value)
                    if single:
                        found.append((single, value))
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def note_view(self, name: str, observation: Any) -> None:
        """Which fields each viewing tool shows, and which a listing has shown since the last change: a view whose
        fields were all shown already is not run again."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    # ---------- result pages ----------
    def unwrapped(self, observation: Any) -> Any:
        """A result that came with notes beside it: the part holding the rows."""
        if isinstance(observation, dict) and not any(isinstance(v, list) and v and all(isinstance(r, dict) for r in v) for v in observation.values()):
            inner = [v for v in observation.values() if isinstance(v, dict) and any(isinstance(x, list) for x in v.values())]
            if len(inner) == 1:
                return inner[0]
        return observation

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        observation = self.unwrapped(observation)
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first),
                 "title": named(got.get("title"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        observation = self.unwrapped(observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        keep = 12 if pages <= 3 else 12 + 2 * (len(self.items) - 1)
        if pages > SWEEP_PAGES or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < keep or (self.reach and count > self.reach):
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.learn_search(name, call, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen, ROW_CHARS)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        if texts and pick == "cheapest" and pages > 3 and self.price_key:
            return self.price_list([observation] + self.paged_obs, len(texts), count)
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def price_list(self, observations: list[Any], pages: int, count: int) -> str:
        """Every buyable row of a search Kyo paged through, cheapest first, with its title: under a cheapest pick only the
        cheap end matters, and one short list costs less to resend than every page."""
        found: list[str] = []
        for seen in observations:
            found += [i for i, _ in self.records_in(seen) if i not in found]
        rows = sorted((self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None)
        lines = []
        for price, ident in rows[:SWEEP_ROWS]:
            title = (self.titles_of([ident], 1) or [""])[0][:80]
            lines.append(f"{ident} | {price:g} | {title}")
        rest = len(rows) - len(lines)
        return (f"\n[Kyo read the other {pages} page{'s' if pages > 1 else ''} of this search itself, one step a page: {len(found)} of its {count} matches seen."
                f" Buyable rows of every page, cheapest first:]\n" + "\n".join(lines)
                + (f"\n({rest} dearer buyable rows left out)" if rest > 0 else "") + (f"\n({len(found) - len(rows)} rows not buyable)" if len(found) > len(rows) else ""))

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        plan = self.plan_search()
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        where = (" This store's search also matches words found only on a listing's page, so rows whose titles lack the words searched can fit, and rows come in no price order."
                 if len(self.hidden_rows) >= 2 else " Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order.")
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest." + where +
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out." + plan)

    def label_hint(self, args: dict[str, Any]) -> str:
        """A search that returned no rows: the category labels result rows have shown, since a filter takes a label
        only as the rows spell it (nothing when the search already used such a label)."""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        counts: dict[str, int] = {}
        for ident in self.seen_order:
            label = (self.rows.get(ident) or {}).get(cat_key)
            if isinstance(label, str) and label.strip():
                counts[label] = counts.get(label, 0) + 1
        if not counts or any(isinstance(v, str) and v in counts for v in args.values()):
            return ""
        # the requested items' own labels first
        order: list[str] = []
        for ids in self.item_groups():
            label = (self.rows.get(ids[0]) or {}).get(cat_key) if ids else None
            if isinstance(label, str) and label in counts and label not in order:
                order.append(label)
        order += [label for label, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if label not in order]
        return ("\n[Kyo] No rows. A filter matches a value only as result rows spell it; category labels the rows have shown: "
                + ", ".join(repr(label) for label in order[:8]) + ".")

    def row_keys_of(self, value: str) -> set[str]:
        """The record keys under which listings seen so far show this exact text value."""
        want = " ".join(value.lower().split())
        return {k for row in self.rows.values() for k, v in row.items() if isinstance(v, str) and " ".join(v.lower().split()) == want}

    def learn_blind(self) -> None:
        """A search parameter that found nothing for a value listings turn out to show (a shop name, say) does not read
        the field they show it under: later calls with such values are not sent."""
        for name, param, value in list(self.empty_filters):
            keys = self.row_keys_of(value)
            if keys:
                self.blind[(name, param)] = sorted(keys)[0]
                self.empty_filters.remove((name, param, value))

    def blind_call(self, name: str, args: dict[str, Any]) -> str:
        for param, value in args.items():
            key = self.blind.get((name, param))
            if key and isinstance(value, str) and value.strip():
                keys = self.row_keys_of(value)
                if not keys or keys == {key}:
                    return (f"Not run (no step used): {param} found nothing earlier for a value that listings show as their {key!r}, so this search does not read"
                            f" that field. Find such a listing with a keyword search (its {key}, kind and distinctive values).")
        return ""

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps: side by
        side when that view shows attributes (one step for several), else one by one."""
        todo = [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]
        many, joiner = self.compare_target(), self.roles.get("compare")
        if len(todo) >= 2 and many and joiner and self.compare_facts() is True:
            texts = []
            room = self.compare_room()
            for start in range(0, len(todo), room):
                if self.left() <= keep or self.finished:
                    break
                chunk = todo[start:start + room]
                faults: list[str] = []
                args = _shape({many: chunk}, self.schemas.get(joiner), "args", faults)
                if faults:
                    break
                reply = self.act(joiner, args)
                if reply is None:
                    break
                seen = reply.get("observation")
                self.observe(joiner, seen)
                self.note_view(joiner, seen)
                texts.append(", ".join(chunk) + ": " + self.show(seen)[:4000])
            if texts:
                self.trace.append({"kind": "fill_facts", "n": len(todo), "joined": True})
            return texts
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in todo:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few, cheapest first) so the last check can weigh them against the cart's choice."""
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        # several at once when the side-by-side view shows attributes: one step reads them all
        joined = bool(self.compare_target() and self.roles.get("compare") and self.compare_facts() is True)
        per, limit = (3, self.compare_room()) if joined else (2, CHEAPER_OPENS)
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:per] if i not in picks]
        return self.fill_facts(picks[:limit], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}" + self.after_change())
        return meaning

    def widen(self) -> None:
        """Open more listings side by side (one step) so the key counts are not read off two or three listings."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        # a side-by-side view without attributes shows the planner no keys: the attributes view one by one instead
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        room = self.compare_room()
        # turns between the items, each up to WIDEN_EACH listings read in all, a call at most per item (one when one will do)
        lanes = []
        for ids in self.item_groups():
            read = sum(1 for i in ids if i in self.full)
            lanes.append([i for i in ids if i not in self.full][: max(0, WIDEN_EACH - read)])
        queue = [lane[k] for k in range(WIDEN_EACH) for lane in lanes if k < len(lane)]
        for start in range(0, min(len(queue), room * max(1, len(lanes))), room):
            picks = queue[start:start + room]
            if len(picks) < 2 or self.left() <= 12:
                return
            faults: list[str] = []
            args = _shape({target: picks}, schema, "args", faults)
            if faults:
                return
            before = len(self.full)
            reply = self.act(tool, args)
            if reply is None:
                return
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
                self.compare_ok = max(self.compare_ok, len(rows))
            else:
                self.compare_bad = min(self.compare_bad or 999, len(picks))
            self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                # the side-by-side view shows no attributes here: read them singly instead
                self.widen_singly()
                return

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        # turns between the item groups, up to two each; with a cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            # a single item gets more of its listings read: the questions list its keys, and keys from two cards are few
            lanes.append(pool[: max(0, min(4 if len(self.items) == 1 else 2, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[: min(6, max(4, 2 * len(lanes)))], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        # listings whose attributes have been read (a page opened along a chain may show none)
        opened = sum(1 for record in self.full.values() if _fact_block(record))
        # an item none of whose listings was read has no property to ask about: wait for one while half the steps remain
        if len(self.items) > 1 and self.step < self.max_steps // 2 and self.left() > self.msgs_free() + self.tail() + 8:
            table = self.key_table()
            if any(not self.item_keys(number, table) for number in range(1, len(self.items) + 1)):
                return False
        if opened >= 3 * len(self.items) + 1:
            return True
        # without an attribute record for each item the planner has no keys to choose from: wait while steps allow
        return self.step >= max(8, self.max_steps // 4) and (opened >= len(self.items) or self.left() <= self.msgs_free() + self.tail() + 8)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("ask")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values, kinds) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}, {kinds} different): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(PLAN_EACH)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        queues: dict[int, list[str]] = {number: [] for number in range(1, len(self.items) + 1)}
        order: list[int] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per pick: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            self.queue_add(queues[number], label)
            if number not in order:
                order.append(number)
        for number in queues:
            # the item's most carried keys the plan did not name follow its picks, so a list reaches past them
            for name in self.item_keys(number, table):
                self.queue_add(queues[number], name)
            if queues[number] and number not in order:
                order.append(number)
        hinted = [n for n in order if any(isinstance(p, dict) and str(p.get("hint") or "").strip() and min(max(1, _count(p.get("item") or 1) or 1), len(self.items)) == n
                                          for p in _items(plan.get("ask")))]
        order = hinted + [n for n in order if n not in hinted]
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queues": _clip(queues, 60)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        # each item's order as it stood after its last revealed preference, and the answers since that revealed nothing
        base = {number: list(queue) for number, queue in queues.items()}
        frees = {number: 0 for number in queues}
        # a list the shopper refuses costs one message: with few to spend, one property a message from the start
        single = self.msg_limit < LIST_FROM
        last: int | None = None
        while self.msgs_free() > 0 and not self.finished:
            live = [n for n in order if queues[n]]
            if not live:
                break
            # take turns between the items; an item whose answer already revealed a preference waits for the others
            pool = [n for n in live if n not in found] or live
            begin = order.index(last) + 1 if last in order else 0
            number = next(order[(begin + k) % len(order)] for k in range(len(order)) if order[(begin + k) % len(order)] in pool)
            labels = queues[number][: 1 if single else LIST_KEYS]
            batch = self.send([(number, labels[0], self.question(number, labels))])
            if not batch:
                break
            last = number
            sent += batch
            # one answer: the light model reads it (the heavy one when the light one fails)
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            named = [(min(max(1, _count(e.get("item")) or 1), len(self.items)), str(e.get("need") or "").split(":")[0].strip())
                     for e in _items(heard.get("needs")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
            if named:
                for n, prop in named:
                    found.add(n)
                    queue = queues.get(n) or []
                    at = next((i for i, lab in enumerate(queue) if self.same_property(lab, prop)), None)
                    if n == number and (at is None or at >= len(labels)):
                        # the answer names a property the message did not list by that name: the listed one sharing a
                        # word with it, else the whole list just sent goes to the back (never the same message twice)
                        words = {w for w in _stem(prop) if len(w) >= 3}
                        near = next((i for i, lab in enumerate(labels) if words & _stem(lab)), None)
                        if near is None:
                            rest = [lab for lab in queue if lab not in labels and not self.same_property(lab, prop)]
                            queues[n] = rest + [lab for lab in labels if not self.same_property(lab, prop)]
                            base[n], frees[n] = list(queues[n]), 0
                            continue
                        at = near
                    if at is not None and n == number and at < len(labels):
                        # the properties named before the answered one held no view: the next list starts after it
                        queues[n] = queue[at + 1:] + queue[:at]
                    else:
                        queues[n] = [lab for lab in queue if not self.same_property(lab, prop)]
                    base[n], frees[n] = list(queues[n]), 0
            elif heard.get("crowded") is True:
                # this shopper takes one property per message: the rest of the questions name one each
                single = True
            elif single or len(labels) == 1:
                queues[number] = queues[number][1:]
            else:
                # the first property on the shopper's mind was one they do not mind (or one they could not place),
                # somewhere in the list: the next list starts from the far end, the one after that from the middle, then a
                # quarter of the way in
                frees[number] += 1
                if not heard.get("unclear") and labels:
                    base[number] = [lab for lab in base[number] if lab != labels[0]]
                whole, size = base[number], len(base[number])
                cut = size // 2 if frees[number] == 2 else size // 4
                queues[number] = whole[::-1] if frees[number] == 1 else whole[cut:] + whole[:cut]
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        if told:
            summary += self.plan_search()
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            followed = self.follow()
            if followed:
                summary += "\n" + followed
            pointed = [need for need in pointed if need in self.unresolved()]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def question(self, number: int, labels: list[str]) -> str:
        names = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
        return f"What {names} would you like for the {self.items[number - 1].name}?"

    def same_property(self, one: str, two: str) -> bool:
        a, b = _stem(one), _stem(two)
        return bool(a) and bool(b) and (a <= b or b <= a)

    def queue_add(self, queue: list[str], label: str) -> None:
        if all(not self.same_property(label, other) for other in queue):
            queue.append(label)

    def item_keys(self, number: int, table: list[tuple[str, str, int, list[str], int]]) -> list[str]:
        """The keys of the category that fits an item best, those carried by two or more listings first, each by how many
        listings carry it; a property the request or an answer settles is left out."""
        item = self.items[number - 1]
        settled = [_stem(m.split(":")[0]) for m in item.musts + item.told]
        fits = {label: self.shelf_match(label, item) for label, _, _, _, _ in table}
        best = max(fits.values(), default=0)
        if not best and len(self.items) > 1:
            return []
        varied: list[str] = []
        rest: list[str] = []
        for label, name, carried, _, kinds in table:
            if best and fits[label] != best:
                continue
            if any(k and (k <= _stem(name) or _stem(name) <= k) for k in settled):
                continue
            (varied if carried >= 2 else rest).append(name)
        return varied + rest

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.message_args(text)
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get("observation")
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary."""
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = (GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                      + "\n\nRECORD (messages and step calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                # what a reply to the gate's own call asks for (earlier answers are the interviewer's to read)
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a
            # message is about the final cart, so its line must come after the cart last changed
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Kyo: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
                break
            if self.left() < 2 or (name == self.roles.get("ask") and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name], "args", faults)
            if faults:
                break
            if name == self.roles.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        return " ".join(found)[:800] if found else _dump(observation)[:400]

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not self.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        self.note_flags = {"dropped": False, "price": got.get("price_moved") is True, "budget": False}
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
            self.note_flags["budget"] = True
            changed.append(f"budget now {budget}")
        added: set[str] = set()
        drops = [e for e in _items(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in _items(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(":")[0]) == _words(target.split(":")[0])][:1]
            if not hit:
                # named in other words ("collar type" for "collar"): the requirement whose property shares its words
                gone = _stem(target.split(":")[0])
                hit = [m for m in live if _names(gone, _stem(m.split(":")[0])) or _names(_stem(m.split(":")[0]), gone)][:1]
            item.dropped.extend(hit or [target])
            self.note_flags["dropped"] = True
            changed.append(f"dropped {target}")
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def after_change(self) -> str:
        """What the change just read means for the picks: a wider fitting set or a moved price under a cheapest, most or
        least pick, and a cart left above a lowered budget."""
        flags, self.note_flags = self.note_flags, {}
        pick = self.brief.get("pick")
        out = []
        if flags.get("dropped") and pick in ("cheapest", "most", "least"):
            out.append(f"A requirement was dropped, so listings that failed only on it fit now and one of them may win the {pick} pick: look at them again before checking out.")
        if flags.get("price") and pick == "cheapest":
            out.append("A price moved: re-read the cart line's price; another fitting listing may be cheaper now.")
        lines = {i: q for i, q in self.cart_lines.items() if q > 0}
        prices = [self.price_of(i) for i in lines]
        if flags.get("budget") and lines and None not in prices and self.budget is not None and self.brief.get("budget_scope") != "each":
            total = sum(p * q for p, q in zip(prices, lines.values()))
            if total > self.budget:
                out.append(f"The cart now totals {total:g}, above the new budget of {self.budget:g}: replace lines with cheaper fitting listings.")
        return (" " + " ".join(out)) if out else ""

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and ident not in ids:
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        needs = self.needs_now()
        if not ids or not needs:
            return ""
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f"(item {number}) " if many else ""
                if hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == "cheapest":
            # listings whose attributes were read first, each group cheapest first
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        # past the given spend line the check still runs, on the light model, until the hard line
        light = self.cost > limit
        if key in self.checked or (light and self.cost > SPEND_HARD + 0.01) or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [LIGHT] if light else [HEAVY, LIGHT], 1500)
        self.trace.append({"kind": "check", "cart": ids, "light": light, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def pending_refs(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def trail(self, early: bool = False, settle: bool = False) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout. `early` (while the agent works, or before an ending without
        a purchase): only reads the opened pages, so a finished chain's value reaches the ledger in time."""
        pending = self.pending_refs()
        mark = (frozenset(self.opened), tuple(pending))
        if early and pending and mark == self.trail_mark:
            # nothing opened since the last read of these chains: that read still holds
            return "", False
        if early:
            self.trail_mark = mark
        # an early read is provisional (more pages may open): it shows in the ledger, and the checkout read settles it
        found = self.ref_early if early and not settle else self.ref_values
        known = dict(found)
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 + self.early_trails + self.follow_reads and self.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 400))[:2200])
                elif not early:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= (30 if early else 60):
                    break
            prompt = TRAIL + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(f"{n}. {need}" for n, need in enumerate(pending, 1))
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            # room for the light model's whole answer: a reply cut at its limit went to the heavy one at ten times the cost
            got = self.ask_json(prompt, [LIGHT] if early else [LIGHT, HEAVY], 3500) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                number = _count(chain.get("need"))
                need = (pending[number - 1] if number and 1 <= number <= len(pending) else
                        next((p for p in pending if p == str(chain.get("need") or "").strip()), None)) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = _items(chain.get("links"))
                ids = [str((l.get("id") if isinstance(l, dict) else l) or "").strip() for l in links]
                if str(chain.get("find") or "").strip():
                    self.next_find[need] = str(chain["find"]).strip()
                # the value stands once the chain's last page is open (a link left without its id is not a missing page)
                if chain.get("done") is True and links and not _items(chain.get("unmatched")) and ids[-1] in self.opened and all(i in self.opened for i in ids if i) and str(chain.get("value") or "").strip():
                    found[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = str(chain.get("about") or "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in found]
            if early or not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if early:
            new = {k: v for k, v in found.items() if known.get(k) != v}
            if new:
                text += "[Kyo read the chain of pages: the product needs] " + "; ".join(f"{k} -> {v}" for k, v in new.items()) + "\n"
            return text, False
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def unresolved(self) -> list[str]:
        """Requirements pointing at another listing whose chain has given no value yet (not even a provisional one)."""
        return [need for need in self.pending_refs() if need not in self.ref_early]

    def query_words(self, text: str) -> list[str]:
        """A free-text query's words as stores split them: lower case, cut at spaces and punctuation, one-letter words and
        repeats left out."""
        out: list[str] = []
        text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", str(text).lower())
        for word in re.split(r"[\s(),/:;|!?\"]+", text):
            word = word.strip("'-")
            if len(word) >= 2 and word not in out:
                out.append(word)
        return out

    def keyword_text(self, name: str, args: dict[str, Any]) -> tuple[str, str] | None:
        """(parameter, words) when this call searches by free words: the brief's keyword search with its fixed arguments."""
        spec = self.brief.get("keyword_search") if isinstance(self.brief.get("keyword_search"), dict) else {}
        if not spec or spec.get("tool") != name:
            return None
        param = spec.get("words")
        if not isinstance(param, str) or not isinstance(args.get(param), str) or not args[param].strip():
            return None
        fixed = spec.get("fixed") if isinstance(spec.get("fixed"), dict) else {}
        if any(k in args and str(args[k]).lower() != str(v).lower() for k, v in fixed.items()):
            return None
        return param, args[param]

    def read_words(self) -> int | None:
        """How many of a query's words the store reads at most, once the results have shown it."""
        return self.lead_cap

    def search_facts(self) -> list[str]:
        out = []
        cap = self.lead_cap
        if cap is not None and self.lead_read >= cap:
            out.append(f"a query is read only up to its first {cap} word{'s' if cap > 1 else ''} (later words change nothing), so put the {cap} most telling words first")
        elif cap is not None:
            out.append(f"words after the first {cap} of a query change nothing")
        elif self.lead_read >= 2:
            out.append(f"at least the first {self.lead_read} words of a query are read")
        if self.word_rule == "all":
            out.append("every word read has to match (a further word narrows the matches)")
        elif self.word_rule == "any":
            out.append("a listing matching any one word read is returned (a further word widens the matches, the best matches come first)")
        if self.ignored_args:
            out.append("it ignores " + ", ".join(sorted(self.ignored_args)) + " (its rows come from any category)")
        if self.reach:
            out.append(f"paging reaches only its first {self.reach} rows whatever total it reports, so narrow a search until its matches fit in them")
        if len(self.hidden_rows) >= 2:
            out.append("it also matches words found only on a listing's page (its attributes), so a row whose title lacks the words searched can still carry them: read such rows' pages before ruling them out")
        return out

    def learn_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Learn this store's keyword search from one more result. Two first pages of the same search that differ only
        after their first k words show the rule: the same rows say the words after k are not read, other rows say the
        word after k is; a longer query that extends a shorter one shows whether a further word narrows the matches
        (every word must match) or widens them (any word will do). Rows from other categories under a category argument
        show that argument ignored, and a last page short of the reported total shows how deep paging reaches. Returns
        what is new for the agent."""
        shape = self.read_shape(name, observation) or {}
        page_arg = shape.get("next_arg")
        rows = [i for i, _ in self.records_in(observation)]
        plan = self.paging(name, observation)
        offset = args.get(page_arg) if page_arg else None
        later = page_arg and offset not in (None, 0, "", "0", 1, "1")
        if plan and not later and isinstance(plan[3], int) and not isinstance(plan[3], bool):
            # the first page tells how the search pages: the next page asked by its number (2) or by the offset after its rows
            self.page_kind[name] = ("number" if plan[3] == 2 and plan[1] != 2 else "offset" if plan[3] == plan[1] else "", plan[1])
        kind, size = self.page_kind.get(name, ("", 0))
        if plan and plan[3] is None and kind and isinstance(offset, int) and not isinstance(offset, bool):
            deepest = (offset - 1) * size + plan[1] if kind == "number" else offset + plan[1]
            if deepest < plan[0]:
                self.reach = deepest
        top = (((self.schemas.get(name) or {}).get("properties") or {}).get(page_arg) or {}).get("maximum") if page_arg else None
        if plan and not later and self.reach is None and isinstance(top, int) and not isinstance(top, bool)                 and isinstance(plan[3], int) and not isinstance(plan[3], bool):
            # the paging parameter's stated maximum: an offset (the next page starts after the rows shown) or a page number
            deepest = top + plan[1] if plan[3] == plan[1] else top * plan[1] if plan[3] == 2 else None
            if deepest and deepest < plan[0]:
                self.reach = deepest
        if plan and not later and self.reach and plan[0] > self.reach:
            self.over_reach = max(self.over_reach, plan[0])
        got = self.keyword_text(name, args)
        if not got:
            # a search by category: the number of listings it reports, when every row shown carries that label
            cat_key = self.category_key()
            if plan and cat_key and rows:
                labels = {str(self.rows[i].get(cat_key) or "").strip().lower() for i in rows if i in self.rows}
                for key, value in args.items():
                    if key != page_arg and isinstance(value, str) and labels == {value.strip().lower()}:
                        self.cat_sizes[value.strip().lower()] = plan[0]
            return self.search_news()
        param, text = got
        words = self.query_words(text)
        if not words:
            return ""
        read = words[: self.lead_cap or None]
        for ident in rows:
            if self.row_hides(ident, read):
                self.hidden_rows.add(ident)
        if later:
            return self.search_news()
        self.queries_run.append(tuple(words))
        total = plan[0] if plan else len(rows)
        others = tuple(sorted((k, _dump(v)) for k, v in args.items() if k not in (param, page_arg)))
        cat_key = self.category_key()
        for key, value in args.items():
            # a category argument the rows do not keep to (fewer than half of three or more rows under that label,
            # which other rows have shown): ignored by a search by words
            if key in (param, page_arg) or not isinstance(value, str) or not cat_key or len(rows) < 3:
                continue
            label = value.strip().lower()
            labels = {str(self.rows[i].get(cat_key) or "").strip().lower() for i in self.seen_order}
            mine = [str(self.rows[i].get(cat_key) or "").strip().lower() for i in rows if i in self.rows]
            if label in labels and mine and sum(1 for m in mine if m == label) * 2 < len(mine):
                self.ignored_args.add(key)
        for old in self.search_log:
            if old["tool"] != name or old["others"] != others or old["words"] == words:
                continue
            k = 0
            while k < min(len(words), len(old["words"])) and words[k] == old["words"][k]:
                k += 1
            if old["total"] != total or old["rows"] != rows:
                self.lead_read = max(self.lead_read, k + 1)
                if k == min(len(words), len(old["words"])):
                    longer, shorter = (total, old["total"]) if len(words) > len(old["words"]) else (old["total"], total)
                    if longer != shorter:
                        self.word_rule = "all" if longer < shorter else "any"
            elif k >= 1 and len(rows) >= 3 and max(len(words), len(old["words"])) > k:
                self.lead_cap = k if self.lead_cap is None else min(self.lead_cap, k)
        if self.lead_cap is not None and self.lead_read > self.lead_cap:
            # the results disagree (a coincidence on one side): nothing is claimed
            self.lead_cap = None
        self.search_log.append({"tool": name, "others": others, "words": words, "total": total, "rows": rows})
        news = self.search_news()
        cap = self.read_words()
        if cap and len(words) > cap:
            news += f"\n[Kyo] This store read only {' '.join(words[:cap])!r} of that query; {' '.join(words[cap:])!r} changed nothing."
        return news

    def row_hides(self, ident: str, words: list[str]) -> bool:
        """A row returned by a keyword search whose own text shows none of the words read (a word counts as shown when its
        stem appears): the store matched it on something only its page carries."""
        row = self.rows.get(ident)
        if not isinstance(row, dict) or not words:
            return False
        text = " ".join(str(v) for v in row.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)).lower()
        return not any(word[: max(3, len(word) - 3)] in text for word in words)

    def hidden_note(self, observations: list[Any]) -> str:
        """After a keyword search: its rows the store matched by words they do not show, cheapest first, unread."""
        if len(self.hidden_rows) < 2:
            return ""
        found: list[str] = []
        for seen in observations:
            found += [i for i, _ in self.records_in(seen) if i in self.hidden_rows and i not in found]
        found = [i for i in found if i not in self.full and i not in self.hidden_told and self.buyable(i) and not self.sold_out(i)]
        if not found:
            return ""
        found.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        self.hidden_told.update(found[:4])
        return ("\n[Kyo] The store matched these rows to the words searched though their rows show none of them, so their pages may carry"
                " the values asked for (cheapest first): " + ", ".join(f"{i} ({self.price_of(i):g})" if self.price_of(i) is not None else i for i in found[:4]) + ".")

    def hidden_leads(self) -> list[str]:
        out = [i for i in self.hidden_rows if i not in self.full and i not in self.opened and self.buyable(i) and not self.sold_out(i)]
        return sorted(out, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))

    def search_news(self) -> str:
        facts = tuple(self.search_facts())
        if not facts or facts == self.search_known:
            return ""
        self.search_known = facts
        self.trace.append({"kind": "search_rules", "facts": list(facts)})
        return "\n[Kyo learned this store's keyword search] " + "; ".join(facts) + "."

    def probe_search(self, name: str, args: dict[str, Any]) -> str:
        """Once, with steps to spare: the agent's keyword query of three or more words run again with its first two
        words only (one step). The same rows show the store reads no further; other rows show whether a further word
        narrows the matches or widens them."""
        got = self.keyword_text(name, args)
        if self.probed or not got or self.lead_cap is not None or self.word_rule or self.left() < PROBE_LEFT + self.tail():
            return ""
        param, text = got
        words = self.query_words(text)
        shape = self.shapes.get(name) or {}
        if len(words) < 3 or (shape.get("next_arg") and args.get(shape["next_arg"]) not in (None, 0, "", "0", 1, "1")):
            return ""
        self.probed = True
        faults: list[str] = []
        call = _shape({**args, param: " ".join(words[:2])}, self.schemas.get(name), "args", faults)
        if faults:
            return ""
        reply = self.act(name, call)
        if reply is None or self.finished:
            return ""
        seen = reply.get("observation")
        self.observe(name, seen)
        news = self.learn_search(name, call, seen)
        self.trace.append({"kind": "search_probe", "query": " ".join(words[:2]), "facts": list(self.search_facts())})
        shown = "" if self.read_words() == 2 else "\nIts rows: " + self.show(seen, ROW_CHARS)
        return (f"\n[Kyo tried {' '.join(words[:2])!r} (one step) to learn how this store reads a query]" + news + shown)

    def titles_of(self, ids: list[str], limit: int) -> list[str]:
        """A few listings' titles, by the row key the result shape names as the title."""
        keys = [shape["title"] for shape in self.shapes.values() if shape and shape.get("title")]
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            title = next((str(row[k]) for k in keys if isinstance(row.get(k), str) and row[k].strip()), "")
            if title:
                out.append(title[:90])
            if len(out) >= limit:
                break
        return out

    def plan_search(self) -> str:
        """A light-model plan of the next keyword queries per item, within the store's learned search rules."""
        facts = self.search_facts()
        if not facts or self.plans >= SEARCH_PLANS or self.cost > SPEND_SOFT or self.finished or not self.keyword_call("x"):
            return ""
        self.plans += 1
        runs = []
        for entry in self.search_log[-14:]:
            runs.append(f"{' '.join(entry['words'])!r}: {entry['total']} matches, {len(entry['rows'])} rows shown")
        cat_key = self.category_key()
        counts: dict[str, int] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "").strip() if cat_key else ""
            if label:
                counts[label] = counts.get(label, 0) + 1
        groups = self.item_groups()
        prompt = (SEARCHPLAN + "\n\nREQUEST:\n" + self.request + "\n\nITEMS, REQUIREMENTS AND RULES:\n" + self.ledger()
                  + "\n\nSTORE SEARCH:\n" + "\n".join("- " + f for f in facts)
                  + "\n\nQUERIES RUN:\n" + ("\n".join(runs) or "none")
                  + "\n\nCATEGORY LABELS ON ROWS SEEN (rows seen):\n" + (", ".join(f"{k} ({v})" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12]) or "none")
                  + "\n\nLISTINGS IN A CATEGORY (as a search by category reported):\n" + (", ".join(f"{k}: {v}" for k, v in self.cat_sizes.items()) or "none seen")
                  + "\n\nTITLES SEEN:\n" + "\n".join("- " + t for ids in groups for t in self.titles_of(ids, 6))
                  + f"\n\nSTEPS LEFT: {self.left()}")
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2400) or {}
        cap = self.read_words()
        run = {q[: cap or None] for q in self.queries_run}
        lines = []
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            queries = []
            for query in _items(entry.get("queries")):
                words = self.query_words(str(query))[: cap or None]
                if words and tuple(words) not in run and " ".join(words) not in queries:
                    queries.append(" ".join(words))
            if not queries and entry.get("browse") is not True:
                continue
            text = f"item {number} ({self.items[number - 1].name}): " + (("run " + ", then ".join(repr(q) for q in queries[:3])) if queries else "")
            if entry.get("browse") is True:
                text += ("; " if queries else "") + "browse its category"
            note = str(entry.get("note") or "").strip()
            lines.append(text + (f" ({note[:200]})" if note else ""))
        self.trace.append({"kind": "search_plan", "plan": lines})
        return ("\n[Kyo's search plan for this store] " + " | ".join(lines)) if lines else ""

    def keyword_call(self, words: str) -> tuple[str, dict[str, Any]] | None:
        """A search by free words: the tool, its parameter and its other arguments as the brief names them; else the
        search tool when free text is all it requires."""
        spec = self.brief.get("keyword_search") if isinstance(self.brief.get("keyword_search"), dict) else {}
        tool = spec.get("tool") if spec.get("tool") in self.schemas else self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        param = spec.get("words") if spec.get("words") in texts else None
        if not param:
            param = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not param:
            return None
        fixed = spec.get("fixed") if isinstance(spec.get("fixed"), dict) else {}
        cap = self.read_words()
        if cap and len(self.query_words(words)) > cap:
            words = " ".join(self.query_words(words)[:cap])
        args = {**{k: v for k, v in fixed.items() if k in props}, param: words}
        if any(k not in args for k in schema.get("required") or []):
            return None
        faults: list[str] = []
        shaped = _shape(args, schema, "args", faults)
        return None if faults else (tool, shaped)

    def chain_budget(self) -> int:
        """Steps the follower may spend: what the stated limit leaves after the questions still to ask, the cart and the
        checkout, and a few searches and reads for each requested item."""
        questions = 0 if self.asked else max(0, self.msgs_free())
        keep = self.tail() + 3 + questions + 4 * len(self.items)
        return max(0, min(FOLLOW_STEPS, self.max_steps // 3, self.left() - keep))

    def open_batch(self, rows: list[str]) -> int:
        """Open listings' pages: side by side in one step within the store's real cap when that view shows the pages'
        words, else the first two one by one with the page view. Returns the steps used."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        page = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(page)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif page and target:
            plan = [(page, {target: i}) for i in rows[:2]]
        else:
            return 0
        used = 0
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults or self.finished:
                break
            reply = self.act(name, args)
            if reply is None:
                break
            used += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.note_view(name, seen)
            if name == joiner and isinstance(args.get(many), list):
                shown = _dump(seen)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                else:
                    self.compare_ok = max(self.compare_ok, len([i for i in args[many] if str(i) in shown]))
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in self.records_in(seen))
        return used

    def follow(self) -> str:
        """Kyo follows the pages requirements point at itself: the light model reads the pages opened so far and names
        the next listing to find (or the chain's value); Kyo searches for it and opens the likely rows within the store's
        real side-by-side room (the rest of them, then the next result page, when that listing is asked for again), all
        within the step budget the stated limit leaves."""
        if self.finished or not self.unresolved() or not self.keyword_call("x"):
            return ""
        budget = self.chain_budget()
        start, rounds, misses = self.step, 0, 0
        tried: dict[frozenset[str], dict[str, Any]] = {}
        before = dict(self.ref_values)
        opened_before = set(self.opened)
        while self.unresolved() and rounds < FOLLOW_ROUNDS and self.step - start + 1 <= budget and self.cost < SPEND_SOFT and not self.finished:
            rounds += 1
            self.follow_reads += 1
            self.trail(early=True, settle=True)
            needs = self.unresolved()
            query = next((self.next_find[n] for n in needs if self.next_find.get(n)), None)
            if not needs:
                break
            if not query:
                # the read gave no answer to go on: once more, then it is left to the agent
                misses += 1
                if misses < 2:
                    continue
                break
            key = _stem(query)
            slot = tried.get(key)
            if slot:
                # another query may have read some of these rows meanwhile
                slot["pool"] = [i for i in slot["pool"] if i not in self.opened]
            if not slot or not slot["pool"]:
                call = self.keyword_call(query)
                if slot:
                    # the same listing asked for again and this result's likely rows all read: its next page
                    plan = self.paging(slot["tool"], slot["seen"])
                    if not plan or plan[3] is None:
                        break
                    call = (slot["tool"], {**slot["args"], plan[2]: plan[3]})
                if not call:
                    break
                reply = self.act(call[0], call[1])
                if reply is None:
                    break
                seen = reply.get("observation")
                self.observe(call[0], seen)
                terms = query.split()
                while not slot and not self.records_in(seen) and len(terms) > 2 and self.step - start + 2 <= budget and not self.finished:
                    # no listing holds every word: the first half of them (the shop's name and kind come first)
                    terms = terms[: max(2, len(terms) // 2)]
                    shorter = self.keyword_call(" ".join(terms))
                    reply = self.act(shorter[0], shorter[1]) if shorter else None
                    if reply is None:
                        break
                    call, seen = shorter, reply.get("observation")
                    self.observe(call[0], seen)
                words = _stem(query)
                found = [(i, len(words & _stem(" ".join(str(v) for k, v in r.items() if isinstance(v, str) and len(v) <= 200))))
                         for i, r in self.records_in(seen) if i not in self.opened and not self.sold_out(i)]
                top = max((n for _, n in found), default=0)
                # the rows that share most of the query's words, best first (its shop, its values)
                pool = [i for i, n in sorted(found, key=lambda x: -x[1]) if n and n >= top - 1][:FOLLOW_ROWS]
                slot = tried[key] = {"tool": call[0], "args": call[1], "seen": seen, "pool": pool}
            if slot["pool"] and self.step - start + 1 <= budget:
                room = self.compare_room() if not self.compare_bare() else 2
                chunk, slot["pool"] = slot["pool"][:room], slot["pool"][room:]
                self.open_batch(chunk)
                slot["pool"] = [i for i in chunk if i not in self.opened] + slot["pool"]
        if self.unresolved() and self.step > start:
            # the last pages opened may have finished a chain
            self.follow_reads += 1
            self.trail(early=True, settle=True)
        used = self.step - start
        self.followed |= self.opened - opened_before
        self.trace.append({"kind": "follow", "steps": used, "rounds": rounds, "values": dict(self.ref_values)})
        new = {k: v for k, v in self.ref_values.items() if before.get(k) != v}
        if not used and not new:
            return ""
        text = f"[Kyo followed the pages requirements point at itself ({used} steps)]"
        if new:
            text += (" Values read along them, by each requirement's relation: " + "; ".join(f"{k} -> {v}" for k, v in new.items())
                     + ". Every page of these chains is open, so the order's page check is met: do not search for them or open them again; look for the product with that value.")
        left = self.unresolved()
        if left:
            text += " Not finished: " + " | ".join(f"{n} (next to find: {self.next_find.get(n) or 'the listing its last page names'})" for n in left[:2]) + "."
        return text

    def look_up_pointers(self) -> None:
        """Before a checkout: a cart line whose own text names another listing to read about it has that listing looked
        up once with a keyword search (its result row shows it), as a store's page check may ask; one light-model
        reading per cart (other rows can carry the same code in their own pointer, so a code seen elsewhere still gets
        its search)."""
        mark = self.cart_mark()
        lines = [i for i in self.cart_ids() if self.full.get(i) or self.rows.get(i)]
        if mark in self.pointed or not lines or self.left() < len(lines) + 4 or not self.keyword_call("x") or self.cost >= SPEND_SOFT:
            return
        self.pointed.add(mark)
        records = "\n".join(f"{i}: " + _dump(_clip(self.full.get(i) or self.rows.get(i), 300))[:1500] for i in lines)
        got = self.ask_json(POINT + "\n\nCART LINES:\n" + records, [LIGHT], 800) or {}
        done, served = [], set()
        for entry in _items(got.get("find")):
            query = str((entry or {}).get("query") or "").strip() if isinstance(entry, dict) else ""
            line = str(entry.get("line") or "") if isinstance(entry, dict) else ""
            # one search for each cart line, its first named listing
            if not query or len(query) > 60 or line in served or len(served) >= len(lines) or self.left() < 4 or self.finished:
                continue
            served.add(line)
            call = self.keyword_call(query)
            reply = self.act(call[0], call[1]) if call else None
            if reply is None:
                break
            self.observe(call[0], reply.get("observation"))
            done.append(query)
        self.trace.append({"kind": "pointers", "find": _clip(got.get("find"), 80), "searched": done})

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            # compare the stored record, so a re-read that arrives wrapped together with a note still shows its change
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """A note or a refusal named the listing as gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """A cart line whose newest record says it is sold out: name replacements among listings already read, once per line."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        """A refusal said a chosen listing sold out without naming it: a lone cart line is that listing; several lines
        are re-read with the offer view so their records show which one. Returns what the re-read showed."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > SPEND_SOFT:
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1200)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def fits_now(self) -> str:
        """Right after a chain's value is read: which listings already opened meet every requirement with it (light model)."""
        if self.cost > SPEND_SOFT:
            return ""
        records = []
        for ident in reversed(self.seen_order):
            if ident in self.full and not self.sold_out(ident):
                records.append(_dump(_clip(self.full[ident], 200))[:1200])
            if len(records) >= 20:
                break
        if not records:
            return ""
        prompt = FITS + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nOPENED LISTING RECORDS (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1200)
        self.trace.append({"kind": "fits", "got": got})
        named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "Opened listings that appear to meet every requirement with that value: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + ". Cart the best of them by the pick rule unless its record breaks a requirement.\n"

    def audit(self) -> str:
        if self.audited or self.cost > SPEND_SOFT:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            hidden = [i for i in self.hidden_leads() if i not in rows][:2]
            cheap = [i for i in self.cheap_unread() if i not in rows and i not in hidden][:2]
            rows = rows[:2] + cheap + hidden + rows[2:]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def cheap_unread(self) -> list[str]:
        """Per requested item's category, its cheapest buyable row within the budget whose page was never read."""
        limit = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        out: list[str] = []
        for ids in self.item_groups():
            pool = [i for i in ids if i not in self.full and i not in self.opened and self.buyable(i) and not self.sold_out(i)
                    and self.price_of(i) is not None and (limit is None or self.price_of(i) <= limit)]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            out += pool[:1]
        return out

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            mark = " (the store matched it to words searched that its row does not show: its page may carry them)" if ident in self.hidden_rows and not record else ""
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500] + mark))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 3000)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            # a decline is wrong whenever a fitting row was left unread: four when steps are plentiful
            plan = [(tool, {target: i}) for i in rows[: 4 if self.left() >= 15 else 2]]
        else:
            return False
        if self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not self.cart_sets(name):
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def closest_mode(self) -> bool:
        return self.brief.get("if_nothing_fits") == "closest" and bool(self.roles.get("cart") and self.roles.get("buy"))

    def fallback_mode(self) -> bool:
        """Unless the request asks for an ending without a purchase, the nearest listings are read once more before it: an
        order is paid only when it meets everything, so they are ordered when that reading finds they miss nothing (the
        agent misjudged a record), and the ending stands otherwise (it is paid when only the budget stands in the way of
        a fitting order). A request that asks for the nearest listings always gets them (closest_mode)."""
        return self.brief.get("if_nothing_fits") != "decline" and bool(self.roles.get("cart") and self.roles.get("buy"))

    def lines_missing(self) -> int:
        """Requested items the cart has no line for yet."""
        return max(0, len(self.items) - len([i for i, n in self.cart_lines.items() if n > 0 and not self.sold_out(i)]))

    def closest_due(self) -> bool:
        """Steps are about to run out with items still without a line: time to put the closest listings forward."""
        missing = self.lines_missing()
        return self.fallback_mode() and not self.finished and self.closest_tries == 0 and missing > 0 and self.left() <= missing + self.tail() + 1

    def closest_lines(self) -> tuple[dict[str, int], bool]:
        """The closest listing of each requested item, by a model's reading of the records seen, and whether that order
        falls short: a line misses a requirement, an item has no line, or the budget alone stops a fitting order."""
        records = []
        order = sorted(self.seen_order, key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            record = self.full.get(ident)
            records.append(("OPENED " + _dump(_clip(record, 200))[:900]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:400]))
            if len(records) >= 25:
                break
        if not records:
            return {}, False
        prompt = (CLOSEST + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                  + "\n\nSOLD OUT: " + (", ".join(i for i in self.seen_order if self.sold_out(i)) or "none") + "\n\nLISTING RECORDS:\n" + "\n".join(records))
        got = self.ask_json(prompt, [LIGHT, HEAVY], 3500) or {}
        self.trace.append({"kind": "closest", "got": _clip(got, 150)})
        plan: dict[str, int] = {}
        worst = 0
        for line in _items(got.get("lines")):
            if isinstance(line, dict) and str(line.get("id") or "") in self.rows:
                number = min(max(1, _count(line.get("item")) or 1), len(self.items))
                plan[str(line["id"])] = self.items[number - 1].qty
                worst = max(worst, _count(line.get("misses")) or 0)
        return plan, worst > 0 or len(plan) < len(self.items) or got.get("budget_blocks") is True

    def order_closest(self) -> str:
        """The request wants the closest listing put forward when nothing fits: Kyo sets the cart to each item's closest
        listing and orders. Returns what happened, or nothing when it could not act."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not self.fallback_mode() or self.finished or self.closest_tries >= 2 or self.left() < 1:
            return ""
        self.closest_tries += 1
        plan, short = self.closest_lines()
        asked = self.closest_mode()
        if asked and (short or not plan):
            # an order that misses a requirement is never paid: the nearest listings are put forward by naming them in
            # the ending without a purchase (race 13 paid only such an ending)
            self.nothing_fits = True
            self.near_ids = list(plan)
            self.trace.append({"kind": "closest_named", "ids": self.near_ids})
            return ""
        if not asked:
            # no request for the nearest listings: the ending without a purchase stands unless the nearest
            # order meets everything within the budget
            budget = self.budget if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" else None
            prices = [self.price_of(i) for i in plan]
            total = sum(p * n for p, n in zip(prices, plan.values())) if plan and None not in prices else None
            if short or not plan or (budget is not None and total is not None and total > budget):
                self.nothing_fits = True
                self.trace.append({"kind": "closest_declined", "short": short, "total": total, "budget": budget})
                return ""
        # no answer from the models: the lines already carted are the order
        plan = plan or {i: n for i, n in self.cart_lines.items() if n > 0}
        if not plan:
            return ""
        now = [i for i, n in self.cart_lines.items() if n > 0]
        calls = self.cart_calls(plan, [i for i in now if i not in plan]) or []
        said = []
        for args in calls[: max(0, self.left() - 1)]:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                continue
            reply = self.act(cart, args)
            if reply is None or self.finished:
                break
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            said.append(_dump(args))
        if self.finished or self.left() < 1:
            return "Kyo set the cart to the closest listings."
        go, gated = self.gate("order")
        if not go:
            return "Kyo set the cart to the closest listings (" + "; ".join(said) + "). " + gated
        if self.finished or self.left() < 1:
            return "Kyo set the cart to the closest listings."
        schema = self.schemas.get(buy) or {}
        texts = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        order = dict(self.brief.get("buy_args") or {})
        if texts and asked:
            # the request asks to be told: the ending tool's free text says so
            order[texts[0]] = "Nothing met every requirement, so this order holds the closest listings."
        faults = []
        order = _shape(order, schema, "args", faults)
        if faults:
            return "Kyo set the cart to the closest listings (" + "; ".join(said) + "); check out now."
        reply = self.act(buy, order)
        self.trace.append({"kind": "order_closest", "lines": plan, "reply": _clip(reply, 200)})
        if reply is None:
            return "Kyo set the cart to the closest listings (" + "; ".join(said) + "); the checkout failed, try again."
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return "Not run (Kyo acted instead): the order must hold the closest listings, so Kyo set the cart (" + "; ".join(said) + ") and checked out. Store reply: " + self.show(reply.get("observation"))

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), SPEND_HARD)
                if (not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- tool use ----------
    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get("ask")
        keep: set[str] | None = None
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("offer")}
        elif left <= len(self.items) + 7 and self.buys == 0 and self.cart_lines and self.lines_missing() == 0:
            # every item has a line: the first checkout goes now, so a change the store announces at the checkout (a
            # sold-out line, a new price or budget) still leaves steps to act on
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        tools = [self.cart_explained(t) for t in self.tools if t["function"]["name"] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if self.closest_mode() and self.near_ids:
                # the nearest listings are named in the ending's free text, whether the call gave one or not
                out[key] = ("Nothing in the store meets every requirement. Closest listings: " + ", ".join(self.near_ids))[: spec.get("maxLength", 200)]
                continue
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one)
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return value.strip(), ""
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any(k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)"

        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), "")
            if problem:
                return args, problem
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ""

    def cart_count(self, name: str | None) -> str | None:
        """The cart tool's integer parameter: how many units, whatever the store calls it."""
        props = (self.schemas.get(name or "") or {}).get("properties") or {}
        return next((k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v)), None)

    def cart_explained(self, tool: dict[str, Any]) -> dict[str, Any]:
        """The tool as the model sees it: a cart tool's count says plainly what it does (the store's name for it can mislead)."""
        name = tool["function"]["name"]
        count = self.cart_count(name) if name == self.roles.get("cart") else None
        if not count:
            return tool
        said = ("the quantity this listing's line must hold, not a position: it sets the line (3 gives 3 units whatever it held before, 0 takes the line off)"
                if self.cart_sets(name) else "how many units of this listing to add or take off")
        params = dict(tool["function"].get("parameters") or {})
        props = dict(params.get("properties") or {})
        props[count] = {**props[count], "description": said}
        params["properties"] = props
        return {**tool, "function": {**tool["function"], "parameters": params}}

    def item_of(self, ident: str) -> Item | None:
        """The requested item a listing serves: by its category label; None when that cannot be told."""
        if len(self.items) == 1:
            return self.items[0]
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        if not label:
            return None
        exact = [it for it in self.items if _words(label) in (_words(it.shelf or ""), _words(it.name))]
        if len(exact) == 1:
            return exact[0]
        long = {w for w in _stem(label) if len(w) >= 4}
        near = [it for it in self.items if long & {w for w in _stem(it.shelf or "") | _stem(it.name) if len(w) >= 4}]
        return near[0] if len(near) == 1 else None

    def cart_guard(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        """A cart call on a store whose count sets the line: the requested quantity goes in, whatever the model sent; a
        count of 0 on a listing not in the cart does nothing. Returns the args, a refusal (no step) and a remark."""
        count = self.cart_count(name)
        if name != self.roles.get("cart") or not count or not self.cart_sets(name) or not isinstance(args.get(count), int):
            return args, "", ""
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        if not ident:
            return args, "", ""
        if args[count] <= 0:
            if ident not in self.cart_lines:
                want = self.item_of(ident)
                return args, f"Not run (no step used): {ident} is not in the cart, and {count} is the line's quantity, so 0 would do nothing. Send the quantity you want" + (f" ({want.qty})." if want else "."), ""
            return args, "", ""
        want = self.item_of(ident)
        if want and args[count] != want.qty:
            return {**args, count: want.qty}, "", f"\n[Kyo] Sent {count}={want.qty}: it is this line's quantity, and the request asks for {want.qty} of {want.name}."
        return args, "", ""

    def fix_quantities(self) -> list[str]:
        """Just before a checkout: lines whose quantity is not the one the request asks for get it (a step each)."""
        name = self.roles.get("cart")
        count = self.cart_count(name)
        target = self.single_id_arg(name)
        if not name or not count or not target or not self.cart_sets(name):
            return []
        done = []
        for ident, have in list(self.cart_lines.items()):
            want = self.item_of(ident)
            if not want or have == want.qty or self.left() <= self.tail() + 1 or self.finished:
                continue
            faults: list[str] = []
            args = _shape({target: ident, count: want.qty}, self.schemas.get(name), "args", faults)
            if faults:
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            self.note_cart(name, args)
            self.observe(name, reply.get("observation"))
            done.append(f"{ident}: {have} -> {want.qty}")
        return done

    def single_id_arg(self, name: str | None) -> str | None:
        """The cart tool's parameter taking the listing id."""
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        return next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any, chars: int = SHOW_CHARS) -> str:
        text = _dump(_clip(observation, chars))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """How many listings one side-by-side call may carry: the schema's limit, else what the store has shown to work."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        limit = spec["maxItems"] if isinstance(spec.get("maxItems"), int) and spec["maxItems"] > 0 else None
        # what the store has shown wins over its schema: a refused count halves the room, a taken one holds it up
        if self.compare_bad:
            room = max(2, self.compare_ok, (self.compare_bad + 1) // 2)
        else:
            room = max(self.compare_ok, max(2, limit // 2) if limit else COMPARE_FIRST)
        return min(room, limit) if limit else room

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
        # only into a side-by-side view that shows what the opens would (attributes)
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_obj(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.gone:
                picks.append((index, ident))
        picks = picks[: self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.trace.append({"kind": "joined", "ids": [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        plain_roles = {tool for job, tool in self.roles.items() if job != "search"}
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: Kyo opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Not run (no step used): too few steps are left for that; use: " + ", ".join(sorted(allowed)) + "."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or "").strip() else {}
                if args is None:
                    out.append((cid, "Not run (no step used): the arguments were not one JSON object."))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], "args", faults)
            if faults:
                out.append((cid, "Not run (no step used): " + "; ".join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong + "."))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room():
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only (one side-by-side call takes {room} here). Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            if name == self.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_text()))
                    continue
            args, refusal, remark = self.cart_guard(name, args)
            if refusal:
                out.append((cid, refusal))
                continue
            trimmed += remark
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, "Not run (no step used): send the finishing call on its own, after reading your other results."))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the step the request asks for at the order, when one is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get("cart") and mark in self.recent:
                out.append((cid, "Not run (no step used): you just made this same call and nothing changed; its result is above."))
                continue
            readers = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and cached[0] == self.version:
                out.append((cid, f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]))
                continue
            if target and name in readers and name != self.roles.get("compare") and str(args.get(target)) in self.followed and str(args.get(target)) not in self.gone:
                out.append((cid, f"Not run (no step used): Kyo opened {args[target]} while following a requirement's chain, and it counts as opened. Its record: " + self.record_of(str(args[target]))))
                continue
            blind = self.blind_call(name, args) if name not in plain_roles else ""
            if blind:
                out.append((cid, blind))
                continue
            if target and name in readers and name != self.roles.get("compare") and self.shown_already(name, str(args.get(target))):
                out.append((cid, f"Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: " + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, "Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n"
                                + "\n".join(f"{i}: {self.record_of(str(i))}" for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    fixed = self.fix_quantities()
                    if fixed:
                        trimmed += "\n[Kyo] Set the requested quantities: " + "; ".join(fixed)
                    self.look_up_pointers()
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3:
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get("observation")
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 3:
                        # a requirement whose value is on another listing's page: read the opened chain first, since the
                        # value it gives (often not the last page's own) can make listings fit that seemed not to
                        if self.pending_refs() and self.early_trails <= EARLY_TRAILS:
                            self.early_trails += 1
                            chain, _ = self.trail(early=True)
                            if chain and self.left() > 4:
                                out.append((cid, "Not run yet (no step used). " + chain + self.fits_now() + "Search again with this value (and the other requirements) and buy a listing that fits; end without a purchase only when none does."))
                                continue
                        review = "" if self.closest_mode() else self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.left() > 3 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 3 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
                # too late for the review: a decline here usually throws away a built cart because time ran out
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart() or self.order_closest()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            pages_before = len(self.opened)
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                sent = list(dict.fromkeys(str(i) for i in args[many]))
                got = [i for i in sent if i in shown]
                if not got:
                    self.compare_bad = min(self.compare_bad or 999, len(sent))
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(sent)} listings: send at most {self.compare_room()} per call."
                elif len(got) < len(sent):
                    missing = [i for i in sent if i not in got]
                    if missing == sent[len(got):]:
                        # the first ones answered, the rest dropped: what it answered is all it takes per call
                        self.compare_ok = max(self.compare_ok, len(got))
                        self.compare_bad = min(self.compare_bad or 999, len(got) + 1)
                    trimmed += f"\n[Kyo] The side-by-side view answered {len(got)} of the {len(sent)} listings sent. Not shown: {', '.join(missing)}."
                else:
                    self.compare_ok = max(self.compare_ok, len(sent))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            learned = self.learn_search(name, args, observation) if name not in plain and not ending else ""
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if name not in plain and not ending:
                learned += self.probe_search(name, args)
                if self.over_reach and not self.reach_told:
                    # more listings than paging can show: the rest only come up under other, narrower searches
                    self.reach_told = True
                    learned += (f"\n[Kyo] A search here reported {self.over_reach} listings, but paging reaches only the first {self.reach}:"
                                f" listings past them never show up on later pages of the same search. Find them with narrower keyword"
                                f" searches whose matches fit within {self.reach} rows, and read every page of each.") + self.plan_search()
                elif learned and self.read_words() and self.plans == 0:
                    learned += self.plan_search()
                paged += learned
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get("open") and not self.facts_from[name]:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if not self.facts_from[name] and name != self.roles.get("open") and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                trimmed += "\n[Kyo] Cart by Kyo's record now (listing: units): " + (_dump(self.cart_lines) if self.cart_lines else "empty")
                self.recent = []
                # lines taken off by a zero or negative count (unless the reply still lists them), and the fullest
                # cart built so far, which is offered back when a late decline would throw it away
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and value in self.rows:
                        if counts[0] <= 0 and not _in_rows(observation, value):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            text = self.show(observation, ROW_CHARS if name not in plain and not ending else SHOW_CHARS) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
                text += self.hidden_note([observation] + self.paged_obs)
                texts = {k: v for k, v in args.items() if isinstance(v, str) and v.strip()}
                if self.records_in(observation):
                    self.filters_ok.update((name, k, v.lower()) for k, v in texts.items())
                else:
                    text += self.label_hint(args)
                    # the one text filter an empty result can be laid on (no number filter, every other value found rows
                    # before); read again once listings show its value
                    suspects = [k for k, v in texts.items() if (name, k, v.lower()) not in self.filters_ok]
                    if len(suspects) == 1 and len(texts) == len(args) and (name, suspects[0], texts[suspects[0]]) not in self.empty_filters:
                        self.empty_filters.append((name, suspects[0], texts[suspects[0]]))
            self.learn_blind()
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out." + self.after_change()
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in self.page_tools() and len(self.opened) > pages_before and self.early_trails < EARLY_TRAILS and self.pending_refs() and self.cost < SPEND_SOFT and not self.finished:
                self.early_trails += 1
                chain, _ = self.trail(early=True)
                if chain:
                    text += "\n" + chain + self.fits_now()
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if self.fallback_mode() and self.lines_missing() > 0 and self.closest_tries == 0:
            done = self.order_closest()
            if self.finished or self.left() <= 0:
                return ""
            if done and again:
                return done
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # a cart short of a line is never paid; when the nearest listings fall short too, the ending without a purchase can be
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)) and not (self.nothing_fits and self.lines_missing() > 0):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                self.look_up_pointers()
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        followed = self.follow()
        if followed:
            intro += "\n\n" + followed
        facts = self.facts_tool()
        intro += ("\n\nStart by searching for each item in its category and opening several listings" + (f" with {facts}" if facts else " in full")
                  + ", so the interviewer can see which properties these listings carry.\n\n" + self.status())
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap:
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            if self.closest_due():
                done = self.order_closest()
                if self.finished:
                    break
                if done:
                    self.chat.append({"role": "user", "content": done + "\n" + self.status()})
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered(),
                                 plain=(self.plain_loop and self.loop_calls > LATE_PLAIN) or self.calls >= PLAIN_AFTER)
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 3:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = _dump(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
