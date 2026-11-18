"""Pharos: buys for one shopper inside an ORO store world, built to find the best option when a request asks for one.

Built on the released ORO agents KYO v6 (the whole walk: a light model uses the world's own tools, code keeps the ledger
and stops wasted steps, a heavy model checks the cart) and, for the questions, Know Thyself (one adaptive question at a
time, items with no need found first, key table and question ranking). Added in Pharos:
- version 2: one message per product names several of its properties, then the list moved on by one, then one property with example values, falling back to one property a message when the shopper asks for that;
- a page sweep for cheapest, most and least requests: code reads as many pages of one search as the step limit allows,
  one step a page and no model turn, with the steps kept for messages, openings and the checkout; a tool that can cap
  the price lists the kind under a ceiling once the rows show the cap holds; rows that arrive price-sorted are read on
  only as far as needed;
- compact results: every row goes into the code's own record and the walker gets a short report, not the pages;
- verification in price order (or by the measure when the rows show it): a light reading sets aside rows whose own words
  break a requirement, the rest are opened from the best end, side by side when the store can, and the first that meets
  every requirement is the answer; the last check is told what was opened, rejected and left unread, and how much of the
  store's match count was read; a changed requirement or budget, a sold-out line or a moved price resumes the walk;
- search words made of the stated and revealed values with the kind of product, then the kind alone; a search read to its
  end below the store's result cap is recorded as complete;
- a last try before ending without a purchase: when the review finds nothing, or while the steps or the money can still
  pay for it with an item not carted, one light reading over the opened listings for a listing per item that no record
  contradicts, bought unless the request itself forbids it;
- a spend rule that holds a 90-task race run under the run key's five dollars: before every model call its worst case is
  worked out; a heavy call that does not fit becomes a light one, a light one that does not fit is not made and the code
  finishes without a model; only the question plan, the first full-cart check and the first review use the heavy model
  first;
- a finish made by code always ends the episode: a store with one finishing tool is read as such, an order turned down
  on an unchanged cart is not sent again, and a step stays for the ending that follows it;
- one listing with no attributes no longer stops a tool being used to read attributes, and a result with a note beside
  its rows is paged like any other.
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

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_LIMIT = 0.053         # dollars a task: 90 tasks stay under the run key's five dollars even when every task reaches it
HEAVY_MAX_TOKENS = 1800
# dollars per million tokens of input (at the cache-write rate) and output: the most a call can cost
RATES = {LIGHT: (0.125, 0.5), SPARE: (0.125, 0.5), HEAVY: (2.5, 10.0)}
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
SWEEP_PICKS = ("cheapest", "most", "least")
SWEEP_PAGES = 24            # pages of searches read for one item at most
SCREEN_ROWS = 30            # rows one light reading sets aside at most
WALK_READS = 8              # light readings of opened listings in one walk at most
LAST_TRY_MONEY = 0.004      # dollars kept for the last try's reading: with an item not carted, it runs before less than this is left
GUARD_KEYS = 3
GUARD_GAP = 0.34
WIDEN_EACH = 6
# listing details that are no property to ask about
NOT_PROPERTY = r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition)\b"

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Result rows are often in no price order, so once the shopper has been asked Kyo itself reads the pages of a search for each item, opens the rows of the right kind in ascending price and reports which one fits and how much of the store it read: build on that report instead of paging by hand. When rows remain unread or nothing fitted, search with other words (the requirement's value, other names for the product) and check candidates in ascending price, taking the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"price_cap": {"tool": the exact tool that can limit its results to listings priced at most a given amount, "arg": that tool's argument taking the amount}, or null when no tool can.
"""


WIDE_KEYS = 8

PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don't mind", and one the shopper does not think about at all gets a puzzled reply. One message names several properties of one item, so rank enough of them.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item's three most carried keys.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; once five or more listings were read, avoid keys seen on only one or two of them.
Skip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}, each item's picks in rank order"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

FITS = """You check listings for a shopping agent. Below are one requested item's requirements and the records of some listings of that kind. Reading rules: """ + READING + """
A listing fits only when it meets every requirement by those rules; a requirement no record addresses is not met.
Return ONE JSON object and nothing else: {"fits": [ids of the listings that meet every requirement], "fails": {"listing id": "the requirement it fails, in a few words"} for every other listing}"""

SCREEN = """A shopping agent wants the best listing of one kind of product and can open only a few. Below are the item's requirements and search rows (short records, not the listing's full attributes). Name the rows whose own words clearly break a requirement: another value for that property, or another kind of product. A row that only fails to mention a requirement is not broken.
Return ONE JSON object and nothing else: {"drop": [{"id": listing id, "why": "the requirement it breaks, in a few words"}]}"""

CEILING = """A store's search tool can limit its results by price. From the tool's description and parameters, give the arguments of one call that lists the listings of the category below priced at most the ceiling. Some stores apply a price limit only in one of the tool's modes, or only without search words: choose the arguments under which it holds.
Return ONE JSON object and nothing else: {"args": {the tool's arguments}}"""

LAST_TRY = """A shopping agent is about to end without buying, and a review found no listing that clearly meets everything. A wrong ending scores nothing, so look once more at the listings it opened, with these reading rules: """ + READING + """
A listing is a candidate for an item when it is of that item's kind, not shown out of stock, inside the budget, and no record of it contradicts a requirement; a requirement its records do not address does not stop it. The first listing named for each item must together, at their quantities, stay inside a total budget.
Return ONE JSON object and nothing else: {"candidates": {"item number": [listing ids, best first, at most 3]}, "request_forbids": true when the request itself says to end without buying rather than buy something unconfirmed, or says what to do when its conditions are not all confirmed; else false, "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "page_arg": when the result names no following page, the tool parameter that asks for a later page or a later row, or null, "page_by": "page" when that parameter counts pages, "row" when it counts rows, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


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
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
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


def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search(r"(?:ch|sh|ss|x|z)es$", w) else w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        if len(w) >= 3 and w not in ("and", "for", "the", "with"):
            out.add(w)
    return out


def _bundled(values: list[str]) -> bool:
    """Values that each hold several things at once (a list of three or more, or a phrase of six words or more)."""
    many = sum(1 for v in values if len(re.split(r"[,;|]", v)) >= 3 or len(v.split()) >= 6)
    return bool(values) and many * 2 >= len(values)


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


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
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
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
        self.ask_width = WIDE_KEYS
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
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
        self.trails = 0
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
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.searched: list[dict[str, Any]] = []
        self.spent_out = False
        self.swept = False
        self.walks: dict[int, dict[str, Any]] = {}
        self.coverage: dict[int, list[dict[str, Any]]] = {}
        self.match_counts: dict[int, set[str]] = {}
        self.page_notes: dict[str, tuple[int, int, str, Any, bool]] = {}
        self.main_labels: dict[int, str] = {}
        self.cap_bad = False
        self.rewalks = 0
        self.heavy_checked = False
        self.audit_calls = 0
        self.last_tried = False
        self.trusted = False
        self.turned_down: tuple[int, Any, bool] | None = None
        self.bare_once: set[str] = set()

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

    def worst_case(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> float:
        """The most one call can cost: every three characters of the prompt a token, and all of max_tokens written."""
        rate_in, rate_out = RATES.get(model, RATES[HEAVY])
        chars = len(_dump(messages)) + (len(_dump(tools)) if tools else 0)
        return (chars / 3 * rate_in + max_tokens * rate_out) / 1e6

    def affordable(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[str, int] | None:
        """The model and token limit a call may use under the task's spending limit: a heavy call that does not fit becomes
        a light one; None when a light one does not fit either."""
        limit = min(max_tokens, HEAVY_MAX_TOKENS) if model == HEAVY else max_tokens
        if self.cost + self.worst_case(model, messages, limit, tools) <= SPEND_LIMIT:
            return model, limit
        if model == HEAVY and self.cost + self.worst_case(LIGHT, messages, max_tokens, tools) <= SPEND_LIMIT:
            return LIGHT, max_tokens
        return None

    def token_bill(self, model: str, messages: list[dict[str, Any]], usage: dict[str, Any], message: Any) -> float:
        """A reply with no cost on it is billed from its token counts, at the highest rates."""
        rate_in, rate_out = RATES.get(model, RATES[HEAVY])
        sent, wrote = usage.get("prompt_tokens"), usage.get("completion_tokens")
        sent = sent if isinstance(sent, int) else len(_dump(messages)) / 3
        wrote = wrote if isinstance(wrote, int) else len(_dump(message)) / 3
        return (sent * rate_in + wrote * rate_out) / 1e6

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for wanted in models:
            if self.calls >= ALL_CALLS:
                return None
            fit = self.affordable(wanted, messages, max_tokens, tools)
            self.spent_out = fit is None
            if fit is None:
                self.trace.append({"kind": "spend_stop", "cost": round(self.cost, 5), "calls": self.calls})
                return None
            model, limit = fit
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, limit, tools)
                if message is not None:
                    return message
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
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
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
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            message = data["choices"][0].get("message") or {}
            if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
                self.cost += float(usage["cost"])
            else:
                self.cost += self.token_bill(model, messages, usage, message)
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
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
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
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
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
        props = (self.schemas.get(buy or "") or {}).get("properties") or {}
        modes = {k: v["enum"] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2}
        if buy and not refuse and modes:
            # a checkout tool with a mode and no tool named for the other ending: the one tool does both
            self.roles["refuse"] = refuse = buy
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
            elif len(modes) == 1:
                # no reading of the mode to be had: one of its two values known gives the other ending
                key, options = next(iter(modes.items()))
                for known, other in (("buy_args", "refuse_args"), ("refuse_args", "buy_args")):
                    if len(options) == 2 and brief[known].get(key) in options and brief[other].get(key) in (None, brief[known].get(key)):
                        brief[other] = {key: options[1 - options.index(brief[known][key])]}
                        break
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
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
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
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
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
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
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
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
        for rule in _items(brief.get("rules")):
            parts.append(f"rule from the request: {rule}")
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        left = self.left()
        text = f"[Kyo] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        best = [f"item {n}: {w['fit']}" + (f" at {w['price']:g}" if w.get("price") is not None else "") for n, w in sorted(self.walks.items()) if w.get("fit")]
        if best:
            text += "\n[best option found] " + "; ".join(best)
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

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values), most carried first per category."""
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
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted) or self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: (-kv[1]["n"], len(kv[0]))):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                recur = sum(len(set(re.findall(r"[a-z]{3,}", v)) & set(re.findall(r"[a-z]{3,}", " ".join(o for o in values if o is not v)))) > 0 for v in values)
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and recur * 2 < len(values):
                    continue
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot["per"].items():
                        twin["per"].setdefault(ident, text)
                        if text not in twin["values"]:
                            twin["values"].append(text)
                    twin["n"] = len(twin["per"])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]["n"])
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted) or self.owner_of(label)]
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
    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "page_arg": named(got.get("page_arg"), props), "page_by": got.get("page_by"),
                 "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
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
        """A search that matched more listings than it showed, under a cheapest, most or least pick: before the sweep it is
        left to the sweep; after it, Kyo reads a short rest itself (a step a page, no model turn), and a long rest is named
        so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in SWEEP_PICKS else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if not later:
            self.page_notes[_dump(args)] = (count, shown, param, value, pick == "cheapest" and self.price_sorted(observation))
            self.note_count(count, _dump(args))
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if self.sweep_due():
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}. Kyo reads the pages of a search itself once the shopper has been asked,"
                    " so do not page on now: open a few listings of the kind.")
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
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
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            # under a cheapest pick the walker gets the rows' cheapest list below, not the pages
            texts.append(f"{param}={_dump(value)}: {shown_text}" if pick != "cheapest" else f"{param}={_dump(value)}: {len(self.records_in(seen))} rows")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        kinds = {i for n in range(1, len(self.items) + 1) for i in self.kind_ids(n)}
        found = (found & kinds) or found
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
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
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
                if _fact_block(seen) or tool in self.bare_once:
                    self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
                self.bare_once.add(tool)
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
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and not any(i in w["dropped"] for w in self.walks.values())
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
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
        cap = props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
            self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                # the side-by-side view shows no attributes here: read them singly instead
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and self.left() > 12:
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        # turns between the item groups; with a cheapest pick the cheapest buyable rows first, those naming the wanted values before others
        wanted = {w for _, need in self.needs_now() for w in re.findall(r"[a-z0-9]{3,}", need.split(":", 1)[-1].lower())}
        sweeping = self.brief.get("pick") in SWEEP_PICKS
        each = 4 if sweeping or len(self.items) > 2 else WIDEN_EACH
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            pool.sort(key=lambda i: -len(wanted & set(re.findall(r"[a-z0-9]{3,}", _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[: max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        # while steps are plentiful: 22 stay for the rest of the episode (30 for the later reads)
        read = self.fill_facts(first[:cap][:8], 22)
        if not sweeping:
            self.fill_facts((first[cap:] + more)[: max(0, 10 - len(read))], 30)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

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
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        lanes = self.plan_lanes(WIDE_KEYS)
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, str, bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # one message per item, naming its top properties: an item with no need found yet goes first; an answer that wants
        # a single question turns the rest to one property a message, narrower and then with two example values
        while self.msgs_free() > 0 and not self.finished and self.left() > 6:
            if follow:
                number, label, shown = follow.pop(0)
                labels = [label]
            else:
                number, labels = self.next_ask(lanes, found, spent, used)
                shown = False
                if not labels:
                    break
                labels = self.fitting(number, labels)
                label = labels[0]
                shown = self.ask_width > 1 and len(labels) == 1
            if len(labels) == 1:
                used[number].add(_words(label))
            text = self.wide_question(number, labels) if len(labels) > 1 else self.question(number, label, self.samples(number, label) if shown else None)
            batch = self.send([(number, label, text)])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_count(x) for x in _items(heard.get("crowded"))]:
                continue
            if len(labels) > 1:
                self.ask_width = 1
                continue
            head = label.split()[-1]
            if len(label.split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, head, False))
            elif not shown and len(self.samples(number, label)) >= 2:
                follow.insert(0, (number, label, True))
        if told:
            self.version += 1
        found_items = [n for n in found if found[n] > 0]
        if self.brief.get("pick") in SWEEP_PICKS:
            searched = self.sweep()
        else:
            searched = self.value_search(found_items) if found_items else ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if searched:
            summary += "\n" + searched
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples and len(samples) >= 2 else "?")

    def wide_question(self, number: int, labels: list[str]) -> str:
        return f"For the {self.items[number - 1].name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

    def fitting(self, number: int, labels: list[str]) -> list[str]:
        """As many of the properties as one message of the store's allowed length can hold."""
        for size in range(len(labels), 1, -1):
            text = self.wide_question(number, labels[:size])
            if len(text) <= 300 and self.message_args(text):
                return labels[:size]
        return labels[:1]

    def samples(self, number: int, label: str) -> list[str]:
        """Two values the listings show for this property: they say which property a question means."""
        want = _words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

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

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (LIGHT, HEAVY)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 2500) or {}

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
            got = self.ask_json(prompt, [LIGHT, HEAVY], 3500)
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
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
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
            item.dropped.extend(hit or [target])
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
        if self.brief.get("pick") == "cheapest" and (drops or gone or self.note_sold or any(c.startswith("budget now") for c in changed)):
            self.wider_held = self.swept
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

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
        order = self.newest_first()
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

    def check_cart(self, ids: list[str] | None = None) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in SWEEP_PICKS:
            facts = [self.walk_facts(number) for number in sorted(self.walks)]
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 6 if facts else 15)
            if facts:
                prompt += "\n\nHOW THE STORE WAS READ FOR THE PICK (worked out by code; rows not named here were not judged):\n" + "\n".join(facts)
        # the strong model checks the first cart that has a line for every item; later carts the light one
        if len(ids) >= len(self.items) and not self.heavy_checked:
            self.heavy_checked = True
            got = self.ask_json(prompt, [HEAVY], 1500) or self.ask_json(prompt, [LIGHT], 3000)
        else:
            got = self.ask_json(prompt, [LIGHT, HEAVY], 3000)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
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

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < SPEND_LIMIT:
            self.trails += 1
            records = []
            for ident in self.newest_first():
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 4500) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
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
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

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
        picks = self.walk_swaps(lost) or self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest" and not self.swept:
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
        records = []
        for ident in self.newest_first():
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 3500)
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

    def audit(self) -> str:
        if self.audited:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in self.newest_first():
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 160))[:800]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 100))[:400]))
            if len(records) >= 20:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        # the first review by the strong model, with a limit that leaves it room under the spending limit
        self.audit_calls += 1
        got = (self.ask_json(prompt, [HEAVY], 1200) if self.audit_calls == 1 else None) or self.ask_json(prompt, [LIGHT], 4000)
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
            plan = [(tool, {target: i}) for i in rows[:2]]
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
            if pick and index and 1 <= index <= len(self.items) and (self.trusted or kind(pick) not in {kind(p) for p in plan}):
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
        # a cart the checks have not passed yet gets one check, unless the last try chose it for want of anything better
        self.fill_facts(list(lines), len(calls) + 3)
        if not self.trusted:
            self.check_cart(list(lines))
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
        args = _shape(self.spell_out(buy, dict(self.brief.get("buy_args") or {})), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                if not self.trusted:
                    self.check_cart(list(lines))
                if ((not self.trusted and not self.verdicts.get(self.check_key(list(lines)))) or self.left() < 1 or self.finished
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

    # ---------- which item a category label belongs to, and the questions (Know Thyself's released code) ----------
    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        best, top = 0, 0.0
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ""):
                score = 3.0
            elif words == _words(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or "") | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = number, score
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and self.owner_of(label) == self.items.index(item) + 1

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum(1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or "")) and _fact_block(record))

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ""
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(":")
            if re.search(r"\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\b", value, re.I):
                continue
            words = set(re.findall(r"[a-z0-9]{3,}", value.lower()))
            shown = set(re.findall(r"[a-z0-9]{3,}", " ".join(values).lower()))
            if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and not words <= shown:
                return f" [not the stated {prop.strip()}: another property]"
        return ""

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = set(name.split()), set(other.split())
        if not a or not b or not a & b:
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = re.findall(r"[a-z0-9]+", x.lower()), re.findall(r"[a-z0-9]+", y.lower())
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and len(long) <= 2 * len(short) + 2)

        both = [i for i in spec["per"] if i in slot["per"] and spec["per"][i].strip() and slot["per"][i].strip()]
        same = sum(1 for i in both if agree(spec["per"][i], slot["per"][i]))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec["values"]) & set(slot["values"])) - {""})

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool, the arguments to keep and the argument that takes the words, from the searches that returned rows."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        named = [k for k in texts if re.search(r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword", k, re.I)]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search(r"[a-z]", args[k], re.I)]
            if not named:
                labels = {str(r.get(self.category_key()) or "").strip().lower() for r in self.rows.values()} - {""} if self.category_key() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any(len(str(a.get(k)).split()) >= 2 for a in self.searched)
                          and not any(str(a.get(k) or "").strip().lower() in labels for a in self.searched)]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get("enum"), list) or k in (schema.get("required") or []))}
                cat_key = self.category_key()
                shelves = {_words(item.shelf or "") for item in self.items} | {_words(item.name) for item in self.items}
                shelves |= {_words(str(row.get(cat_key) or "")) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get("required") or []) or not isinstance(v, str)
                        or (_words(v) not in shelves and not any({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search(r"[a-z]", a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get("enum"), list):
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next(a[k] for a in worded if k in a and _dump(a[k]) == best)
                return tool, keep, field
        return None

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ":" not in need:
                continue
            value = need.split(":", 1)[1].strip()
            if not value or re.search(r"\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\b", value, re.I):
                continue
            out.append(value)
        return out

    def value_search(self, numbers: list[int]) -> str:
        """After an answer reveals a value: search each such item's values together and read the first rows of its kind."""
        plan = self.search_template() or self.bare_search()
        if not plan or self.left() < 22:
            return ""
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any(":" in t and t.split(":", 1)[1].strip() == v for t in item.told)]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: " ".join(words)}, self.schemas.get(tool), "args", faults)
                if faults or self.finished:
                    break
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get("observation")
                self.observe(tool, observation)
                cat_key = self.category_key()
                found = [i for i, _ in self.records_in(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all(p is not None and p > 10 * cap for p in prices):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ""))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                seen.append(f"{_dump(args.get(field))}: " + (", ".join(rows[:6]) or "no row of that kind"))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][: min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append("opened: " + " | ".join(opened)[:2400])
                    break
        if not seen:
            return ""
        self.trace.append({"kind": "value_search", "seen": [s[:200] for s in seen]})
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + "\n".join(seen)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            told = [_words(need.split(":", 1)[0]) for need in self.items[number - 1].told]
            open_keys = [label for label in lanes[number] if not any(_words(label) in t or t in _words(label) for t in told)]
            left = [label for label in open_keys if _words(label) not in used[number]]
            if self.ask_width > 1 and spent.get(number, 0) < 2:
                wide = open_keys[spent.get(number, 0):spent.get(number, 0) + self.ask_width]
                if len(wide) >= 2:
                    return number, wide
            if left:
                return number, left[:1]
        return 0, []

    def bundled_keys(self, table: list[tuple[str, str, int, list[str]]]) -> set[str]:
        """Keys whose values hold several things at once, by those values themselves: they name no one property, so they are asked last."""
        return {name for _, name, _, values in table if _bundled(values)}

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(":", 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum(1 for v in values if re.match(r"^\s*[\d.,/x -]+\s*[a-z]{0,6}\s*$", v)) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and not rows:
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any(_names(prop, _stem(key)) for prop in settled) and not re.search(NOT_PROPERTY, key)]
        bundled = self.bundled_keys(table)
        return [key for key in keys if key not in bundled] + [key for key in keys if key in bundled]

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        """One ranked list of properties per item: the model's picks, then a code guard that puts a far more widely carried key in place of a weak unhinted pick."""
        table = self.key_table()
        bundled = self.bundled_keys(table)
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values) + self.not_stated(label, name, values) + (" [several features in one: ask last]" if name in bundled else ""))
        prompt = PLAN.replace("COUNT", str(max(1, count))) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY], 1800) or self.ask_json(prompt, [LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for other in lanes[number]):
                lanes[number].append(label)
            if str(pick.get("hint") or "").strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_keys(number, table)
            read = max(1, self.records_of_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub(r"our\b", "or", re.sub(r"(?<=[a-z]{3})s\b", "", _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _words(label) in own:
                    return own[_words(label)]
                near = [s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))]
                return max(near) if near else 0.0

            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane: list[str] = []
            for pick in lanes[number]:
                if all(_words(spelled(pick)) != _words(other) for other in lane):
                    lane.append(spelled(pick))
            for key in [k for k in keys if k not in bundled][:GUARD_KEYS]:
                if any(_words(key) == _words(other) for other in lane[:3]):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _words(other) != _words(key)]
                if weak is not None and len(self.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any(_words(x) in hinted[number] and all(_words(x) != _words(y) for y in placed[:3]) for x in lane[:3]):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all(_words(key) != _words(other) for other in lanes[number]):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if _words(x) not in bundled] + [x for x in lanes[number] if _words(x) in bundled]
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "lanes": {n: v[:8] for n, v in lanes.items()}})
        return lanes

    # ---------- the best-option sweep (cheapest, most, least) ----------
    def sweep_due(self) -> bool:
        return self.brief.get("pick") in SWEEP_PICKS and not self.swept and not self.finished and bool(self.roles.get("search"))

    def checkout_steps(self) -> int:
        """Steps the finish takes: each line carted and re-read, the cart view, the checkout, and one refusal dealt with
        (a re-read, a swap of the line, another view and another checkout)."""
        return 2 * len(self.items) + 7

    def unsent(self) -> int:
        """Messages the interviewer has still to send, each a step."""
        return max(0, self.msgs_free()) if not self.asked and self.roles.get("ask") else 0

    def can_compare(self) -> bool:
        """The side-by-side view may serve the walk: it has not been seen to show no attributes (one step tells)."""
        return bool(self.compare_target() and self.roles.get("compare") and self.compare_facts() is not False)

    def note_count(self, count: int, search: str) -> None:
        self.match_counts.setdefault(count, set()).add(search)

    def is_cap(self, count: int | None) -> bool:
        """The same match count for different searches is the store's cap on results, not how many listings match."""
        return count is not None and count >= 50 and len(self.match_counts.get(count, ())) >= 2

    def kind_ids(self, number: int) -> list[str]:
        """Rows of this item's kind by the store's category label: the labels that match the item best; with no such
        label, the label most rows of the item's own first search carried (for a single item, of all rows seen until
        then); with no labels and one item, every row."""
        cat_key = self.category_key()
        if not cat_key:
            return list(self.seen_order) if len(self.items) == 1 else []
        item = self.items[number - 1]
        mine = _roots(item.shelf or "") | _roots(item.name)
        scores: dict[str, float] = {}
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "")
            if label and label not in scores:
                if not self.owns(item, label):
                    scores[label] = 0.0
                elif _words(label) == _words(item.shelf or ""):
                    scores[label] = 3.0
                elif _words(label) == _words(item.name):
                    scores[label] = 2.0
                else:
                    long = _roots(label)
                    scores[label] = len(long & mine) / max(1, len(long | mine))
        top = max(scores.values(), default=0.0)
        if top > 0:
            labels = {label for label, score in scores.items() if score >= top}
        else:
            main = self.main_labels.get(number, "")
            if not main and len(self.items) == 1:
                found = [str((self.rows.get(i) or {}).get(cat_key) or "") for i in self.seen_order]
                main = max(set(found), key=found.count, default="")
            labels = {main} - {""}
        return [i for i in self.seen_order if str((self.rows.get(i) or {}).get(cat_key) or "") in labels]

    def newest_first(self) -> list[str]:
        """Listings seen, newest first; after a sweep the opened ones come before the many rows it read."""
        order = list(reversed(self.seen_order))
        return [i for i in order if i in self.full] + [i for i in order if i not in self.full] if self.swept else order

    def needs_mark(self, number: int) -> str:
        return _dump([need for n, need in self.needs_now() if n == number]) + f"|{self.budget}"

    def rows_level(self, name: str, observation: Any) -> Any:
        """A result that carries a note beside its rows holds them one level down: the part of the result with the rows."""
        if not isinstance(observation, dict):
            return observation
        key = (self.shapes.get(name) or {}).get("rows")

        def holds(part: dict[str, Any]) -> bool:
            if key:
                return isinstance(part.get(key), list)
            return any(isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v) for v in part.values())

        if holds(observation):
            return observation
        return next((v for v in observation.values() if isinstance(v, dict) and holds(v)), observation)

    def page_step(self, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], int | None, str | None, Any] | None:
        """(row ids, matches reported, the parameter that asks for the next page, its value or None on the last page), by
        the tool's named result shape; a store that names no next page but takes a page number or an offset is counted on."""
        observation = self.rows_level(name, observation)
        shape = self.read_shape(name, observation)
        if not shape or not isinstance(observation, dict) or not isinstance(observation.get(shape["rows"]), list):
            return None
        ids = [i for i, _ in self.records_in(observation)]
        count = observation.get(shape["count"]) if shape["count"] else None
        count = count if isinstance(count, int) and not isinstance(count, bool) else None
        param, value = shape["next_arg"], observation.get(shape["next"]) if shape["next"] else None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            value = None
        if not shape["next"]:
            param = param or shape["page_arg"]
            if param and ids:
                given = args.get(param)
                given = given if isinstance(given, int) and not isinstance(given, bool) else None
                value = (given or 1) + 1 if shape["page_by"] == "page" else (given or 0) + len(ids)
        return ids, count, param, value

    def read_pages(self, tool: str, args: dict[str, Any], room: int, number: int, label: str, seen: Any = None) -> dict[str, Any]:
        """Read a search page by page, at most `room` steps, with no model turn: every row goes into the code's own record
        and nothing into the chat. `seen` is a first page already called. Stops at the last page, at two pages in a row with
        none of the item's kind, or after the first page of rows that arrive in price order (those are read on as needed)."""
        record: dict[str, Any] = {"tool": tool, "words": label, "count": None, "rows": 0, "pages": 0, "all": False, "sorted": False, "next": None, "args": args, "stopped": ""}
        self.coverage.setdefault(number, []).append(record)
        call = args
        dry = 0
        first = True
        note = self.page_notes.get(_dump(args)) if seen is None else None
        if note:
            # the agent's own search already called this first page: its rows are in the record, go on from its next page
            count, shown, param, value, ordered = note
            record.update(count=count, rows=shown, next=(param, value), sorted=ordered)
            self.note_count(count, _dump(args))
            first = False
            if value is None or shown >= count:
                record.update(all=True, next=None)
                return record
            if ordered:
                return record
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(tool), "args", faults)
            if faults:
                return record
        while not self.finished:
            if seen is None:
                if room <= 0 or self.left() <= self.tail():
                    break
                reply = self.act(tool, call)
                if reply is None:
                    break
                room -= 1
                record["pages"] += 1
                page_seen = reply.get("observation")
                self.observe(tool, page_seen)
            else:
                page_seen, seen = seen, None
            page = self.page_step(tool, call, page_seen)
            if page is None:
                break
            ids, count, param, value = page
            record["rows"] += len(ids)
            if count is not None:
                record["count"] = count
                if first:
                    self.note_count(count, _dump(args))
            cat_key = self.category_key()
            if cat_key and ids and number not in self.main_labels:
                labels = [str((self.rows.get(i) or {}).get(cat_key) or "") for i in ids]
                self.main_labels[number] = max(set(labels), key=labels.count)
            kinds = self.kind_ids(number)
            dry = 0 if not cat_key or any(i in kinds for i in ids) else dry + 1
            record["next"] = (param, value) if param and value is not None else None
            if not ids or not record["next"] or (count is not None and record["rows"] >= count):
                record["all"] = True
                record["next"] = None
                break
            if first and self.brief.get("pick") == "cheapest" and self.price_sorted(page_seen):
                record["sorted"] = True
                break
            first = False
            if dry >= 2:
                record["stopped"] = "two pages in a row held none of this kind"
                break
            faults = []
            call = _shape({**args, param: value}, self.schemas.get(tool), "args", faults)
            if faults:
                break
        self.trace.append({"kind": "sweep_pages", "item": number, "words": label[:80], "pages": record["pages"], "rows": record["rows"], "count": record["count"], "all": record["all"], "sorted": record["sorted"]})
        return record

    def sweep_phrases(self, number: int) -> list[str]:
        """The stated and revealed values with the kind of product first (a store's results stop at a capped count), then the kind alone."""
        kind = self.items[number - 1].name
        values = self.value_words(number)[-3:]
        phrases = [" ".join(values + [kind])] if values else []
        return phrases + [kind]

    def bare_search(self) -> tuple[str, dict[str, Any], str] | None:
        """No call of the search tool to copy (the agent listed a category another way): the tool's one free-text argument."""
        tool = self.roles.get("search") or ""
        schema = self.schemas.get(tool) or {}
        texts = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        field = next((k for k in texts if k in (schema.get("required") or [])), None) or (texts[0] if len(texts) == 1 else None)
        return (tool, {}, field) if field else None

    def sweep(self) -> str:
        """Under a cheapest, most or least pick: per item, read one search's pages as far as the steps allow and verify the
        rows of its kind in price (or measure) order. The steps are split between the items; the agent gets a short report."""
        if not self.sweep_due():
            return ""
        self.swept = True
        plan = self.search_template() or self.bare_search()
        if not plan and not self.cap_tool():
            self.trace.append({"kind": "sweep_off", "why": "no search returned rows yet"})
            return ""
        self.wider_held = True
        texts = []
        total = len(self.items)
        for number in range(1, total + 1):
            if self.finished:
                break
            start = self.step
            share = (self.left() - self.checkout_steps() - self.unsent()) // (total - number + 1)
            # the steps left after the checkout go to the pages and to opening rows: opened one by one, rows take two steps in five
            room = min(SWEEP_PAGES, max(0, share - 2 if self.can_compare() else share * 3 // 5))
            used = self.capped_listing(number, room)
            whole = bool(used) and (self.coverage.get(number) or [{}])[-1].get("all")
            if plan and room - used > 0 and not whole:
                tool, base, field = plan
                pages = room - used
                for words in self.sweep_phrases(number):
                    faults: list[str] = []
                    args = _shape({**base, field: words}, self.schemas.get(tool), "args", faults)
                    if faults or pages <= 0 or self.finished:
                        break
                    pages -= self.read_pages(tool, args, pages, number, words)["pages"]
            self.price_walk(number, max(0, share - (self.step - start)))
            last = (self.coverage.get(number) or [{}])[-1]
            if not self.walks[number].get("fit") and last.get("ceiling") and last.get("all"):
                # nothing fits under the ceiling and all of it was read: the wanted values' own search, then the category again
                # up to the price of what that found (only cheaper rows can beat it), or higher up when it found nothing
                worded = self.sweep_phrases(number)[:-1] if plan else []
                left = share - (self.step - start)
                if worded and left >= 3:
                    tool, base, field = plan
                    faults = []
                    args = _shape({**base, field: worded[0]}, self.schemas.get(tool), "args", faults)
                    if not faults:
                        self.read_pages(tool, args, 2, number, worded[0])
                        self.price_walk(number, max(0, share - (self.step - start)))
                left = share - (self.step - start)
                found = self.walks[number].get("price")
                higher = found if found is not None else last["ceiling"] * 3
                if higher > last["ceiling"] and left >= 4 and self.capped_listing(number, max(2, left * 3 // 5), higher):
                    self.price_walk(number, max(0, share - (self.step - start)))
            texts.append(self.walk_text(number))
        return "\n".join(texts)

    # a store with a tool that can cap the price
    def cap_tool(self) -> tuple[str, str] | None:
        """The tool and the argument that cap a row's price, as the brief read them off the tools; only the rows can show it holds."""
        named = self.brief.get("price_cap")
        if not isinstance(named, dict):
            return None
        tool, key = str(named.get("tool") or ""), str(named.get("arg") or "")
        spec = ((self.schemas.get(tool) or {}).get("properties") or {}).get(key)
        return (tool, key) if isinstance(spec, dict) and {"number", "integer"} & set(_kinds(spec)) else None

    def cap_call(self, tool: str, key: str, ceiling: float, label: str, number: int) -> tuple[dict[str, Any] | None, Any]:
        """Call the tool under a price ceiling, with arguments the light model reads off the tool. The cap only counts when
        the rows show it holds: stores can ignore it in one of their modes. Returns the arguments and the result, or None."""
        fn = next((t["function"] for t in self.tools if t["function"]["name"] == tool), {})
        note = ""
        for _ in range(2):
            prompt = (CEILING + "\n\nTOOL:\n" + _dump({"name": tool, "description": fn.get("description", ""), "parameters": self.schemas.get(tool)})
                      + f"\n\nCATEGORY: {label}\nCEILING: {ceiling:g}" + note)
            got = self.ask_json(prompt, [LIGHT, SPARE], 1500) or {}
            if not isinstance(got.get("args"), dict) or self.left() <= self.checkout_steps():
                return None, None
            faults: list[str] = []
            args = _shape({**got["args"], key: ceiling}, self.schemas.get(tool), "args", faults)
            if faults:
                return None, None
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return None, None
            seen = reply.get("observation")
            self.observe(tool, seen)
            rows = [i for i, _ in self.records_in(seen)]
            over = [i for i in rows if (self.price_of(i) or 0) > ceiling]
            cat_key = self.category_key()
            kinds = self.kind_ids(number)
            if rows and not over and (not cat_key or any(i in kinds for i in rows)):
                return args, seen
            note = f"\n\nThe call {_dump(args)} returned " + (f"{len(over)} rows above the ceiling" if over else "no rows of that category") + ": use another mode or other arguments."
        return None, None

    def capped_listing(self, number: int, room: int, ceiling: float | None = None) -> int:
        """Under a cheapest pick, with a tool that caps the price: list the item's kind under a ceiling, lower it while the
        rows would not fit in `room` steps, and read the pages. Returns the steps used."""
        capper = self.cap_tool()
        if self.brief.get("pick") != "cheapest" or not capper or not self.price_key or self.cap_bad or room < 2:
            return 0
        tool, key = capper
        start = self.step
        item = self.items[number - 1]
        kinds = self.kind_ids(number)
        prices = sorted(p for p in (self.price_of(i) for i in kinds) if p is not None)
        limit = self.budget / max(1, item.qty) if isinstance(self.budget, (int, float)) else None
        first = ceiling is None
        if ceiling is None:
            # low in the prices seen: rows above the first fit are of no use, and a walk with no fit raises the ceiling
            ceiling = prices[len(prices) // 4] if len(prices) >= 8 else prices[len(prices) // 2] if len(prices) >= 3 else prices[-1] if prices else limit
        if limit is not None and ceiling is not None:
            ceiling = min(ceiling, limit)
        if not ceiling or ceiling <= 0:
            return 0
        cat_key = self.category_key()
        labels = [str((self.rows.get(i) or {}).get(cat_key) or "") for i in kinds] if cat_key else []
        label = max(set(labels), key=labels.count) if labels else item.shelf or item.name
        integral = "number" not in _kinds(self.schemas[tool]["properties"][key])
        ceiling = int(ceiling) if integral else ceiling
        args, seen = self.cap_call(tool, key, ceiling, label, number)
        if args is None:
            self.cap_bad = True
            return self.step - start
        # lowered only on the first listing: a raised ceiling is read as far as the steps go
        for _ in range(2 if first else 0):
            page = self.page_step(tool, args, seen)
            if not page or page[1] is None or not page[0]:
                break
            priced = sorted(p for p in (self.price_of(i) for i in page[0]) if p is not None)
            if page[1] <= len(page[0]) * max(1, room - (self.step - start)) or len(priced) < 3 or priced[len(priced) // 2] >= ceiling:
                break
            ceiling = int(priced[len(priced) // 2]) if integral else priced[len(priced) // 2]
            faults: list[str] = []
            lower = _shape({**args, key: ceiling}, self.schemas.get(tool), "args", faults)
            if faults or self.left() <= self.checkout_steps():
                break
            reply = self.act(tool, lower)
            if reply is None or self.finished:
                break
            args, seen = lower, reply.get("observation")
            self.observe(tool, seen)
        record = self.read_pages(tool, args, room - (self.step - start), number, f"the category under {ceiling:g}", seen)
        record["ceiling"] = ceiling
        return self.step - start

    # verifying in price or measure order
    def measure_of(self, ident: str) -> float | None:
        """The pick's measure as this listing's own record states it: the first number under a key naming the measure."""
        words = {w for w in _stem(str(self.brief.get("pick_measure") or "")) if len(w) >= 3}
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = {**record, **(_fact_block(record) or {})}
        if not words:
            return None
        for key, value in block.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and words & _stem(key):
                found = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
                if found:
                    return float(found.group(0).replace(",", ""))
        return None

    def walk_pool(self, number: int) -> tuple[list[str], bool]:
        """The rows of this item's kind that can still be bought within the budget, best first; and whether that order is
        final (the first fit wins): always by price; by the measure when at least half the rows show it."""
        item = self.items[number - 1]
        ids = [i for i in self.kind_ids(number) if self.buyable(i) and i not in self.gone]
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)):
            limit = self.budget if self.brief.get("budget_scope") == "each" else self.budget / max(1, item.qty)
            known = [p for p in prices.values() if p is not None]
            # prices far above the limit everywhere are in another unit than the request's: the limit cannot sift them
            if known and not all(p > 10 * limit for p in known):
                ids = [i for i in ids if prices[i] is None or prices[i] <= limit]
        pick = self.brief.get("pick")
        if pick == "cheapest":
            ids.sort(key=lambda i: (prices[i] is None, prices[i] or 0.0))
            return ids, True
        measures = {i: self.measure_of(i) for i in ids}
        sign = -1 if pick == "most" else 1
        ids.sort(key=lambda i: (measures[i] is None, sign * (measures[i] or 0.0)))
        return ids, sum(1 for m in measures.values() if m is not None) * 2 >= len(ids) > 0

    def needs_text(self, number: int) -> str:
        return "\n".join("- " + need + (f" (the page it points at gives: {self.ref_values[need]})" if need in self.ref_values else "") for n, need in self.needs_now() if n == number) or "- none beyond the kind of product"

    def screen_rows(self, number: int, walk: dict[str, Any], pool: list[str]) -> None:
        """One light reading over the next unopened rows: a row whose own words break a requirement costs no step to open."""
        rows = [i for i in pool if i not in self.full and i not in walk["screened"] and i not in walk["failed"] and i not in walk["dropped"] and i not in walk["fits"]][:SCREEN_ROWS]
        walk["screened"].update(rows)
        if len(rows) < 4 or walk["screens"] >= 3 or not any(n == number for n, _ in self.needs_now()):
            return
        walk["screens"] += 1
        prompt = (SCREEN + f"\n\nITEM: {self.items[number - 1].name}\nREQUIREMENTS:\n" + self.needs_text(number)
                  + "\n\nROWS:\n" + "\n".join(_dump(_clip(self.rows.get(i), 120))[:500] for i in rows))
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        for entry in _items(got.get("drop")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            if ident in rows:
                walk["dropped"][ident] = str(entry.get("why") or "")[:80] if isinstance(entry, dict) else ""

    def names_values(self, number: int, ident: str) -> bool:
        """The row's own words carry every wanted value of the item: the likeliest fits among rows not opened yet."""
        words = {w for value in self.value_words(number) for w in re.findall(r"[a-z0-9]{3,}", value.lower())}
        text = " ".join(str(v).lower() for v in (self.rows.get(ident) or {}).values() if isinstance(v, str))
        return bool(words) and all(w in text for w in words)

    def walk_batch(self, number: int, walk: dict[str, Any], pool: list[str], final: bool, size: int) -> tuple[list[str], str | None]:
        """The next listings to judge, and the best fit so far when the order is final. Steps are few: the rows naming the
        wanted values come first (cheapest first); once one fits, every row that would beat it, those again first."""
        live = [i for i in pool if i not in walk["failed"] and i not in walk["dropped"]]
        fits = [i for i in live if i in walk["fits"]]
        rest = live[: live.index(fits[0])] if final and fits else [i for i in live if i not in walk["fits"]]
        named = [i for i in rest if self.names_values(number, i)] if final else []
        return (named or rest)[:size], fits[0] if final and fits else None

    def fit_read(self, number: int, ids: list[str]) -> tuple[list[str], dict[str, str]] | None:
        """Which of these listings meet every requirement of the item, by the light model; None when no model can be had."""
        if not any(n == number for n, _ in self.needs_now()):
            return list(ids), {}
        prompt = (FITS + f"\n\nREQUEST:\n{self.request}\n\nITEM: {self.items[number - 1].name}\nREQUIREMENTS:\n" + self.needs_text(number)
                  + "\n\nRECORDS:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in ids))
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500)
        if got is None:
            return None
        fits = [str(i) for i in _items(got.get("fits")) if str(i) in ids]
        fails = got.get("fails") if isinstance(got.get("fails"), dict) else {}
        return fits, {str(k): str(v)[:80] for k, v in fails.items()}

    def open_walk(self, ids: list[str], opens: int) -> None:
        """Open these listings: side by side in one step when that view shows attributes, else one by one, within `opens` steps."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        start = self.step
        if many and joiner and self.can_compare() and (len(ids) >= 2 or not self.facts_tool()):
            faults: list[str] = []
            args = _shape({many: ids}, self.schemas.get(joiner), "args", faults)
            reply = self.act(joiner, args) if not faults else None
            if reply is not None:
                seen = reply.get("observation")
                self.observe(joiner, seen)
                self.note_view(joiner, seen)
                rows = self.records_in(seen)
                if rows:
                    self.facts_from[joiner] = self.facts_from.get(joiner, False) or any(_fact_block(r) for _, r in rows)
                    self.compare_ok = max(self.compare_ok, len(ids))
                elif len(ids) <= 2:
                    # it showed nothing even for two listings: this view is no way to read them
                    self.facts_from[joiner] = False
                else:
                    self.compare_bad = min(self.compare_bad or 999, len(ids))
                ids = [i for i in ids if i not in self.full]
        self.fill_facts(ids[: max(0, opens - (self.step - start))], self.tail())

    def price_walk(self, number: int, opens: int) -> dict[str, Any]:
        """Open the rows of this item's kind from the best end of the order until the best listing that meets every
        requirement has nothing left that could beat it; `opens` is the steps it may use. What it opened, rejected and left out stays on record for the last
        check. A change of the requirements or the budget starts the judging again; what is open stays open."""
        walk = self.walks.setdefault(number, {"mark": None})
        mark = self.needs_mark(number)
        if walk["mark"] != mark:
            walk.update(mark=mark, fit=None, price=None, fits=[], failed={}, dropped={}, screened=set(), screens=0, reads=0, ended="")
        if walk["fit"] and (self.sold_out(walk["fit"]) or walk["fit"] in self.gone):
            walk["fits"] = [i for i in walk["fits"] if i != walk["fit"]]
        walk["fit"], walk["ended"], walk["reads"] = None, "", 0
        while True:
            pool, final = self.walk_pool(number)
            # the side-by-side view is tried with two listings until it has shown attributes: stores can show nothing for more
            size = (min(4, self.compare_room()) if self.compare_facts() is True else 2) if self.can_compare() else (2 if opens > 3 else 1)
            batch, best = self.walk_batch(number, walk, pool, final, size)
            if any(i not in self.full and i not in walk["screened"] for i in batch) and walk["screens"] < 3:
                self.screen_rows(number, walk, batch + [i for i in pool if i not in batch])
                continue
            if best and not batch:
                walk["fit"], walk["ended"] = best, "fit"
                break
            if not batch:
                more = next((r for r in self.coverage.get(number, []) if r["sorted"] and r["next"] and not r.get("spent")), None)
                if more and opens > 0 and self.left() > self.tail():
                    # rows in price order are read on only as far as the walk needs them
                    param, value = more["next"]
                    faults: list[str] = []
                    tool = more["tool"]
                    call = _shape({**more["args"], param: value}, self.schemas.get(tool), "args", faults)
                    reply = self.act(tool, call) if not faults else None
                    opens -= 1
                    if reply is None:
                        more["spent"] = True
                        continue
                    seen = reply.get("observation")
                    self.observe(tool, seen)
                    page = self.page_step(tool, call, seen)
                    more["pages"] += 1
                    if page:
                        more["rows"] += len(page[0])
                        more["next"] = (page[2], page[3]) if page[2] and page[3] is not None and page[0] else None
                        more["all"] = not more["next"]
                    else:
                        more["spent"] = True
                    continue
                walk["ended"] = "no more rows"
                break
            if walk["reads"] >= WALK_READS:
                walk["ended"] = "reads"
                break
            closed = [i for i in batch if i not in self.full]
            if closed:
                need = 1 if self.can_compare() and len(closed) >= 2 else len(closed)
                if opens <= 0 or self.left() <= self.tail():
                    batch = [i for i in batch if i in self.full]
                    closed = []
                    if not batch:
                        walk["ended"] = "steps"
                        break
                elif need > opens:
                    keep = closed[:opens]
                    batch = [i for i in batch if i in self.full or i in keep]
                    closed = keep
                before = self.step
                if closed:
                    self.open_walk(closed, opens)
                opens -= self.step - before
            verdict = self.fit_read(number, batch)
            if verdict is None:
                walk["ended"] = "no model"
                break
            walk["reads"] += 1
            fits, why = verdict
            for ident in batch:
                if ident in fits:
                    if ident not in walk["fits"]:
                        walk["fits"].append(ident)
                else:
                    walk["failed"][ident] = why.get(ident, "")
            self.trace.append({"kind": "walk", "item": number, "read": batch, "fits": fits, "why": why})
        if not walk["fit"] and walk["fits"]:
            # no final order (a measure few rows show) or the steps ran out: the best fit opened
            live = [i for i in walk["fits"] if not self.sold_out(i) and i not in self.gone]
            order, _ = self.walk_pool(number)
            ranked = [i for i in order if i in live] or live
            walk["fit"] = ranked[0] if ranked else None
        walk["price"] = self.price_of(walk["fit"]) if walk["fit"] else None
        self.trace.append({"kind": "walk_end", "item": number, "fit": walk["fit"], "ended": walk["ended"], "failed": len(walk["failed"]), "dropped": len(walk["dropped"])})
        return walk

    def coverage_text(self, number: int) -> str:
        parts = []
        for record in self.coverage.get(number, []):
            count = record["count"]
            if record["all"] and self.is_cap(count):
                how = f"all {record['rows']} rows it shows, and the store seems to stop at {count} results"
            elif record["all"]:
                how = f"all {record['rows']} rows, so this search is complete"
            elif record["sorted"]:
                how = f"{record['rows']} of {count if count is not None else 'an unknown number of'} rows, which arrive cheapest first"
            else:
                how = f"{record['rows']} of {count if count is not None else 'an unknown number of'} rows" + (f" (the store seems to stop at {count} results)" if self.is_cap(count) else "")
            parts.append(f"{record['words']!r}: {how}" + (f" ({record['stopped']})" if record["stopped"] else ""))
        return "; ".join(parts)

    def walk_facts(self, number: int) -> str:
        """What the sweep read and judged for this item, and what it left out: for the agent and for the last check."""
        walk = self.walks.get(number)
        if not walk:
            return ""
        item = self.items[number - 1]
        kinds = self.kind_ids(number)
        pick = self.brief.get("pick")
        tag = lambda i: f"{i} ({self.price_of(i):g})" if self.price_of(i) is not None else i
        text = f"item {number} ({item.name}): read {self.coverage_text(number) or 'no further pages'}; {len(kinds)} rows of this kind known"
        order, _ = self.walk_pool(number)
        failed = [i for i in order if i in walk["failed"]] + [i for i in walk["failed"] if i not in order]
        if failed:
            text += ". Opened and rejected: " + "; ".join(f"{tag(i)}: {walk['failed'][i] or 'a requirement not met'}" for i in failed[:8])
        if walk["dropped"]:
            text += f". Left unopened because the row's own words break a requirement: {len(walk['dropped'])} rows (" + ", ".join(tag(i) for i in list(walk["dropped"])[:4]) + ")"
        if walk["fit"]:
            ahead = order[: order.index(walk["fit"])] if walk["fit"] in order else []
            unread = [i for i in ahead if i not in walk["failed"] and i not in walk["dropped"] and i not in walk["fits"]]
            text += f". First that meets every requirement: {tag(walk['fit'])}"
            if pick == "cheapest":
                text += ", with no unopened row of this kind cheaper" if not unread else ", but rows priced lower were not opened: " + ", ".join(tag(i) for i in unread[:6])
        else:
            rest = [i for i in order if i not in walk["failed"] and i not in walk["dropped"]]
            text += ". No opened listing meets every requirement" + (f"; next rows in order, not opened: " + ", ".join(tag(i) for i in rest[:6]) if rest else "")
        return text

    def walk_text(self, number: int) -> str:
        walk = self.walks.get(number)
        if not walk:
            return ""
        text = "[Kyo read the store for this pick itself, one step a page] " + self.walk_facts(number)
        if walk["fit"]:
            return text + f". Cart {walk['fit']} at quantity {self.items[number - 1].qty} unless its record shows otherwise; there is no need to page again."
        return text + ". Search with other words or open the rows named, in that order."

    def walk_stale(self) -> list[int]:
        """Items whose walk the shopper or the store has overtaken: requirements or budget changed, the fit sold out or moved in price."""
        if not self.swept or self.rewalks >= 2:
            return []
        stale = []
        for number, walk in self.walks.items():
            fit = walk.get("fit")
            if walk["mark"] != self.needs_mark(number) or (fit and (self.sold_out(fit) or fit in self.gone or self.price_of(fit) != walk["price"])):
                stale.append(number)
        return stale

    def walk_swaps(self, lost: list[str]) -> list[str]:
        """A cart line sold out under a pick the sweep walked: the walk goes on from where it stopped."""
        if not self.swept:
            return []
        cat_key = self.category_key()
        picks: list[str] = []
        for ident in lost:
            number = 1 if len(self.items) == 1 or not cat_key else self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or ""))
            if number in self.walks:
                fit = self.price_walk(number, max(0, self.left() - 2 * len(self.items) - 3)).get("fit")
                if fit and fit not in picks and fit not in self.cart_ids():
                    picks.append(fit)
        return picks

    # ---------- the last try before ending without a purchase ----------
    def running_out(self) -> bool:
        """Some item has no cart line, and the steps or the money to cart one are about to run out: the moment for the last
        try, while it can still act."""
        have = len(self.cart_ids())
        if self.last_tried or self.finished or have >= len(self.items):
            return False
        props = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        whole = any(isinstance(v, dict) and isinstance(v.get("items"), dict) and isinstance(v["items"].get("properties"), dict) for v in props.values())
        return self.left() <= (1 if whole else len(self.items) - have) + 2 or self.cost >= SPEND_LIMIT - LAST_TRY_MONEY

    def last_try(self) -> bool:
        """The review found nothing: one light reading over what has been opened, for a listing per item that no record
        contradicts. Running short of proof is not a reason to end without a purchase, unless the request itself says so.
        Sets the lines for complete_cart, which buys them without a further check; False when an item has no candidate."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.last_tried or self.confirmed or self.completed or self.finished or not cart or not buy or self.left() < 2:
            return False
        self.last_tried = True
        total = len(self.items)
        for number in range(1, total + 1):
            kinds = self.kind_ids(number)
            if kinds and not any(i in self.full for i in kinds):
                self.fill_facts([i for i in kinds if self.buyable(i)][:2], 2 * total + 3)
        opened = [i for i in reversed(self.seen_order) if i in self.full and not self.sold_out(i)]
        # each item's newest opened listings, taken in turn so that a short prompt still holds every item; an item with
        # none opened has its rows stand in
        lists = []
        for number in range(1, total + 1):
            kinds = set(self.kind_ids(number))
            lists.append([i for i in opened if i in kinds] or [i for i in reversed(self.seen_order) if i in kinds and self.buyable(i)][:6])
        chosen: list[str] = []
        for rank in range(24):
            chosen += [ids[rank] for ids in lists if rank < len(ids) and ids[rank] not in chosen]
        chosen = (chosen + [i for i in opened if i not in chosen])[:24]
        if not chosen:
            return False
        # one call is always left for this reading: only the spending limit can stop it, and it gets shorter before that does
        self.calls = min(self.calls, ALL_CALLS - 1)
        for size in (24, 12, 6):
            prompt = (LAST_TRY + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nLISTINGS OPENED:\n"
                      + "\n".join(f"{i}: " + _dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1200] for i in chosen[:size]))
            if self.affordable(LIGHT, [{"role": "user", "content": prompt}], 2500, None):
                chosen = chosen[:size]
                break
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500)
        self.trace.append({"kind": "last_try", "got": _clip(got, 150)})
        if not got or got.get("request_forbids") is True or not isinstance(got.get("candidates"), dict):
            return False
        named = {str(k): [str(i) for i in _items(v) if str(i) in chosen and not self.sold_out(str(i))][:3] for k, v in got["candidates"].items()}
        named = {k: v for k, v in named.items() if v}
        if any(str(number) not in named for number in range(1, total + 1)):
            return False
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each":
            def bill() -> float:
                return sum((self.price_of(v[0]) or 0.0) * self.items[int(k) - 1].qty for k, v in named.items() if k.isdigit() and 1 <= int(k) <= total)
            known = [p for v in named.values() for p in (self.price_of(i) for i in v) if p is not None]
            # prices far above the budget everywhere are in another unit than the request's: the budget cannot sift them
            if known and not all(p > 10 * self.budget for p in known) and bill() > self.budget:
                named = {k: sorted(v, key=lambda i: self.price_of(i) or 0.0) for k, v in named.items()}
                if bill() > self.budget:
                    self.trace.append({"kind": "last_try_over", "bill": bill()})
                    return False
        self.lead_items = named
        self.confirmed = named["1"][0]
        self.trusted = True
        return True

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
        tools = [t for t in self.tools if t["function"]["name"] != ask]
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

    def show(self, observation: Any) -> str:
        text = _dump(_clip(observation, SHOW_CHARS))
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
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

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
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, "Not run (no step used): send the finishing call on its own, after reading your other results."))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
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
                    reread = self.reread_offers()
                    # the re-read can show a moved price; a change of requirements or budget can have overtaken the walk too
                    stale = self.walk_stale() if self.left() > self.tail() + 1 else []
                    if stale:
                        self.rewalks += 1
                        for number in stale:
                            # after a refusal the finish is short: each line swapped, the cart view, the checkout
                            self.price_walk(number, max(0, self.left() - 2 * len(self.items) - 3) // len(stale))
                        moved = [number for number in stale if self.walks[number]["fit"] not in self.cart_ids()]
                        if moved:
                            out.append((cid, "Not run yet (no step used). What the shopper or the store changed sent Kyo back over its reading of the store:\n" + "\n".join(self.walk_text(number) for number in moved) + "\nCart now: " + self.cart_text()))
                            continue
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
                    if reread and self.spent_out:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5:
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first])
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
                self.last_try()
                done = self.complete_cart()
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
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
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
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
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
                    if not self.facts_from[name] and name == self.roles.get("open") and name not in self.bare_once:
                        # still the attributes view, with no other to turn to
                        del self.facts_from[name]
                    self.bare_once.add(name)
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("search") and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if name == self.roles.get("cart"):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
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
                    self.turned_down = (self.cart_at, self.budget, bool(self.loose_lines(observation)))
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
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
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
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
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and not self.order_barred(holding) and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(self.spell_out(buy, dict(self.brief.get("buy_args") or {})), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ""
                told = reply is not None and bool(self.loose_lines(reply.get("observation")))
                if reply is not None:
                    self.turned_down = (self.cart_at, self.budget, told)
                # with no line on record and no word from the shopper the cart is empty: nothing for the agent to mend
                if reply is not None and again and (holding or told):
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
                if told and holding:
                    # the last round: only a note came back and nobody is left to act on it, so the order goes in again
                    reply = self.act(buy, args)
                    self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                    if self.finished or self.left() <= 0:
                        return ""
        if refuse and not holding and self.last_try():
            self.complete_cart()
            if self.finished or self.left() <= 0:
                return ""
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
                    if not again and self.left() > 0:
                        # the last round: nobody is left to act on what came back, so the ending goes in again
                        reply = self.act(refuse, args)
                        self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                        if reply is None or self.finished:
                            return ""
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def order_barred(self, holding: bool) -> bool:
        """The store turned down this same cart: sending that order again would only spend the step the ending needs,
        unless what came back was a note from the shopper about a cart on record (not the cart's fault)."""
        last = self.turned_down
        return bool(last) and last[:2] == (self.cart_at, self.budget) and not (last[2] and holding)

    def spell_out(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A free-text argument the finishing tool requires, on an order made by code."""
        schema = self.schemas.get(name) or {}
        out = dict(args)
        for key in schema.get("required") or []:
            spec = (schema.get("properties") or {}).get(key)
            if key not in out and isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec:
                out[key] = "Ordering the cart as it stands."[: spec.get("maxLength", 200)]
        return out

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
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_LIMIT)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_LIMIT)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap:
            if self.running_out() and self.last_try():
                done = self.complete_cart()
                if done and not self.finished:
                    self.chat.append({"role": "user", "content": done + "\n" + self.status()})
                continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.spent_out or self.calls >= ALL_CALLS or self.misses > 2:
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
            if self.sweep_due() and any(c["function"]["name"] in committing for c in calls):
                # no interview to run it after: the sweep runs before the first commit
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the pick counts every fitting listing, so Kyo reads the store first."})
                summary = self.sweep()
                self.chat.append({"role": "user", "content": (summary or "Kyo could not read further pages of this store's search.") + "\n" + self.status()})
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
