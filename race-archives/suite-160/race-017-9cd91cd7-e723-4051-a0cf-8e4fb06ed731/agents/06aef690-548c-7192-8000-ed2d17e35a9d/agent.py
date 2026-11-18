"""Umi: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.

n48 over n47: one property per message, each answer read before the next question (no lists); the most widely carried
properties of each product among its first questions; listing records sent as text lines read like any other record.

n49 over n48: before a checkout, code sets each line to its item's requested quantity (a cart tool may add to a line
rather than set it); a listing belongs to the item whose store label it carries exactly, and a checkout with a requested
item missing from the cart is held once with that item named; a line that states a requirement nowhere is swapped for a
read listing stating it under its own key.
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
SPEND_SOFT = 0.05
SPEND_HARD = 0.07
CHAIN_EXTRA = 0.02            # an episode whose requirement points at another listing may spend this much more: walking its pages costs model calls
LOOP_SPEND = 0.05             # the agent's own loop stops here: a race run shares one inference budget over all its tasks, so long tasks must not drain it
LOOP_SPEND_MULTI = 0.035      # the same for a request of several products: these long tasks are the ones that drain a run
EPISODE_MAX = 0.07            # no model call starts past this: the episode then ends with what it holds, which needs no model
MULTI_MAX = 0.05              # the ceiling for a request of several products without a pointer: these long, rarely passed tasks
                              # are what drained race 12's $5 run key, while single-product tasks pass often and keep their budget
SECOND_MULTI = 0.01           # extra loop budget after a refused checkout for several products (one product keeps 0.02)
KEY_ROWS = 24                # keys of one category shown to the question plan
# a race run may spend $5 over 90 tasks: the code's searches run only while an episode is still cheap
HUNT_SPEND = 0.035
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
# wording of a limit, an exclusion or a closed list: its words are not what a listing shows, so no search is built on them
BOUNDED = re.compile(r"[<>\u2264\u2265]|\b(?:no|not|non|never|without|unlike|except|avoid|anything|other than|different|one of|at least|at most|up to|under|over|below|above|more than|less than|fewer than|between|within|minimum|maximum)\b", re.I)
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
SHORT_RESULT = 40            # a search matching at most this many listings is read to its end early, whatever the pick rule
DECLINE_PAGES = 5            # before an ending without a purchase, at most this many unread search pages are read
AUTO_PAGES = 8              # further search pages Umi reads itself for a cheapest, most or least pick
DEEP_PAGES = 3              # under such a pick, a search with many more pages has this many of them read, as short rows
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Umi reads before a cheapest checkout
TASK_SECONDS = 270          # Umi's own limit inside the network's 300 s: a call under way when it passes still ends in time
LATE_SECONDS = 50           # below this nothing optional runs and the agent is told to check out
FORCE_SECONDS = 25          # below this the loop stops and the finish goes without its model checks
SEARCH_WORDS = 6            # a keyword search reads this many words of its query
HUNT_ROUNDS = 3             # reveals Umi searches for itself in one episode
HUNT_STEPS = 10             # steps one such search may take, across its items
HUNT_TOTAL = 18             # and all of them together
HUNT_TRIES = 3              # queries per item, narrowest first, until one finds rows
HUNT_PAGES = 5              # a result needing more further pages than this is too wide to read to its end
AUDIT_RECORDS = 36           # listings a decline audit reads
AUDIT_ROWS = 8               # n47: and, on the first audit only, at most this many earlier unopened rows of a requested kind, in short form
CHAIN_ROUNDS = 5             # rounds of the early chain walk: read the trail, open or search for the next link
CHAIN_STEPS = 12             # steps it may take in all
CHAIN_KEEP = 8               # steps beyond the checkout's that stay for choosing and carting
CHAIN_SWEEP = 6              # of a shop's listings, how many one walk reads looking for the details a pointer gives
ORDER_WALKS = 2              # walks a checkout may start while a chain is unfinished: an order before its end cannot pass
NOTE_FIELD = re.compile(r"note|annotat|fine.?print|remark|notice|good.?to.?know|small.?print|caveat", re.I)
NOTE_DESC = re.compile(r"\bnotes?\b|remarks?|fine print|\bpage\b", re.I)
# a need whose value is read from another listing, in the shopper's own words: "... on the page of/for ...", "... like the one sold by X with ..."
LIST_QUESTIONS = False       # n48: lists off; a shopper may refuse any message naming more than one property
SINGLE_QUESTIONS = True      # n48: one property per message, each answer read before the next question is chosen
CARRIED_GAP = 0.34           # n48: a pick from the plan gives its early place to a key carried this much more widely
CARRIED_TOP = 3              # n48: the early places, and the most carried keys considered for them
ORDER_KEEP = 4               # n49: steps kept for the checkout when code fixes the cart before it
SHELF_BLOCKS = 2             # n49: checkouts held at most this often for a requested item with no cart line
QTY_KEY = re.compile(r"^(?:qty|quantity|count|units|number|held|amount|amount held|how many|line count|item count|in cart|qty in cart|quantity in cart|number held)$")   # n49: a cart row's count key, as words
NOT_QTY = re.compile(r"price|cost|total|due|sum|discount|stock|rating|review|min|max|limit|\bid\b|code|sku")   # n49: never a count


def _count_key(key: str) -> bool:
    """n49: a cart row key that names a count and nothing else ("quantity", "amount_held", "qty"), never a discount, a stock
    count, a rating count or a price."""
    words = " ".join(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " ").lower()))
    return bool(QTY_KEY.search(words)) and not NOT_QTY.search(words)
LIST_SIZE = 8                # properties one message names
LIST_CHARS = 700             # a message longer than this drops its last properties: a store tool may cap its text
ONE_EACH = "A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property."
SEVERAL = "A question reveals a preference only when it names that exact property. One message can name several properties of one product and the shopper answers about one they care about, so your list decides what the messages can reach: put the likeliest properties first."
# a reply that wants fewer questions says so in words like these; a line that only mentions being busy does not
ONE_AT_A_TIME = re.compile(r"(?<!that )(?<!no )\bone\b(?! ear)|\b(?:single|separate\w*|split|too many|too much|bit much|in turn|at once|first|pile of|batch\w*)\b", re.I)
COLOUR = re.compile(r"\bcolou?rs?\b|^\s*shades?\s*$", re.I)
NOT_A_QUESTION = re.compile(r"\b(?:sku|ids?|codes?|upc|ean|asin|barcodes?|models?|warrant\w*|guarantees?|urls?|links?|images?|photos?|videos?|shipping|delivery|variants?|options?)\b", re.I)
LAST_RESORT = ("material", "size", "type")   # only when the listings read name too few properties
POINTER = re.compile(r"\b(?:on|from|in) the page (?:of|for)\b|\bpage (?:of|for) the\b|\b(?:same|unlike|like|near|within|either way)\b[^.]{0,80}\b(?:listing|page|one)\b[^.]{0,60}\bsold by\b", re.I)
HUNT_MIN_STEPS = 45         # with a shorter step limit the agent keeps its steps for reading and checkout
HUNT_OPENS = 4              # listings opened per item

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Umi, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Umi knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Umi's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Umi carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Umi's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by meaning: case, singular or plural, plain synonyms and a slip in spelling (a letter missing, doubled or swapped) are fine, and so are words that only repeat the property's name or give a unit ("14 size" is size 14, "3 mm" is 3mm). A value with added words counts when it names a kind, shade or version of the required value ("navy blue" is blue, "acme os 11" is acme os, "intel x5 200" is the x5 200), but not when the added word makes it a different thing ("rose gold" is not gold, which alone means yellow gold; "faux leather" is not leather). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). When the listings show both a value that is exactly the required one and one with added words, the exact one is the fit. Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides over the title and description: if it says something else, the title does not rescue the listing. When several attributes name that property (a plain one and a more specific one, such as shape and frame shape, or material and fabric material), the requirement is met when any of them states the value. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Umi adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the request names the shopper's own model or machine code and a listing's note says which models it fits ("Fits: every ABC 12xx model", "Fits part series XY2, XY5"), the listing fits the shopper only when that note covers the code (every ABC 12xx model covers ABC 1234 unless it is excepted; a part series list must include the series of the shopper's machine); an item the request ties to that code fails when its note does not cover it. Umi marks this beside results.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. That other listing supplies only the value of the property the requirement names: its other properties, and what its page says about itself, are not requirements on what you buy. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Umi reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it. With several items, never hold one back to finish comparing: cart each item's best fitting listing found so far before turning to the next item, and replace it when a better one turns up, so the cart holds every item whenever the task has to end.
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
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A requirement that one requested item must fit, work with, suit or be compatible with another requested item (the one the agent picks for this order) is never a lookup: it goes under "fits".
"fits": a list with one entry {"item": the number of the requested item that must fit, "with": the number of the requested item it must fit} (items numbered from 1 in the order of "items") for each requested item the request says must fit, work with, suit or be compatible with another requested item of this order, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"search_args": {"text": the "search" tool's parameter that takes free words, or null, "fixed": the fixed arguments that put that tool in its free-words mode, else {}}.
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property. One message can name several properties of one product and the shopper answers about one they care about, so your list decides what the messages can reach: put the likeliest properties first.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by a key listed for that item when one fits the hint: a hint about who will wear it points at the gender or audience key, a hint about its looks at the looks key most of its listings carry, which is often not colour.
Colour is a weak guess: choose it early only when the story itself talks about colour; otherwise it comes after the item's other listed keys.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product compares first (its kind or style, its form, its size or capacity, its material), then keys carried by most of the item's listings whose values differ between listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value.
- A line that turns away everything apart from one value requires that value ("every size is refused unless it is large", "no wood besides oak will do" make "size: large", "wood: oak"). A line that welcomes everything apart from one value excludes it ("any size but large is fine" makes "size: not large")."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, and so do words that only repeat the property's name or give a unit ("14 size" for size 14) and added words that name a kind, shade or version of the required value ("navy blue" for blue, "acme os 11" for acme os), but not added words that make it a different thing ("rose gold" for gold, "faux leather" for leather), units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides over the title and description; when several attributes name that property (a plain one and a more specific one, such as shape and frame shape, or material and fabric material), the requirement is met when any of them states the value; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value; when the request ties an item to the shopper's model or machine code, that item's line fails if its fitting note ("Fits: every ABC 12xx model", "Fits part series ...") does not cover the code. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong. When a cart line meets a requirement only through added words and another listing shown states exactly the required value and fits every requirement, the cart is wrong too: name that listing in "better".
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, and so do words that only repeat the property's name or give a unit ("14 size" for size 14) and added words that name a kind, shade or version of the required value ("navy blue" for blue), unless the added word plainly makes it a different thing ("rose gold" for gold), limits include the edge); when several attributes name that property, one stating the value is enough; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.
A shop can have several listings of one kind: the link is the one whose details match how it is described (the description's properties shown in its title or record), so pick that one, and when several could match, name the best match. An OPENED page of the right shop that neither sends the reader to a further listing nor states the property is not the link: name another listing of that shop that matches the description better (or "" when none is listed), and never mark a chain done from such a page."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


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


def _shared_word(values: list[str]) -> bool:
    """True when two of the values share a plain word of four or more letters."""
    seen: set[str] = set()
    for value in values:
        mine = {w for w in re.findall(r"[a-z]+", value.lower()) if len(w) >= 4}
        if mine & seen:
            return True
        seen |= mine
    return False


# a requirement text with these words is a limit, an exclusion or a pointer: code cannot judge it by matching words
_NOT_PLAIN = re.compile(r"\b(?:at most|at least|up to|under|over|less|more|than|between|not|no|without|except|same|like|unlike|near|page|listing|minimum|maximum|max|min|or)\b")


def _phrase_in(want: str, got: str) -> bool:
    """The required value's words together and in order inside the listing's value (the store's own reading of a value),
    with or without a space between a number and its unit."""
    def norm(text: str) -> list[str]:
        plain = " ".join(re.findall(r"[a-z0-9.]+", str(text).lower()))
        return [plain, re.sub(r"(\d) (?=[a-z])", r"\1", plain)]
    for w in norm(want):
        for g in norm(got):
            if w and re.search(r"(^| )" + re.escape(w) + r"( |$)", g):
                return True
    return False


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


def _peel(observation: Any) -> Any:
    """The result inside a wrapper that sets shopper or store notes beside it (one dict plus a list of lines)."""
    if isinstance(observation, dict) and len(observation) == 2:
        inner = [v for v in observation.values() if isinstance(v, dict)]
        lines = [v for v in observation.values() if isinstance(v, (list, str))]
        if len(inner) == 1 and len(lines) == 1 and all(isinstance(x, str) for x in _items(lines[0])):
            return inner[0]
    return observation


# ---------- n48: listing records sent as one text line ----------
# Some stores send a listing as one line of "key: value" parts joined by " | ", with its attributes as "name=value"
# pairs joined by "; " inside one part and a free-text description as the last part. The code below turns such a line
# into the same dict a store sending records as objects gives, so every reader of records works on either. It reads
# structure only (separators, key shapes, an id-like first value); no key name is assumed.
_TEXT_PART = re.compile(r" \| (?=[A-Za-z][\w ]{0,40}: )")
_TEXT_KEY = re.compile(r"^([A-Za-z][\w ]{0,40}): ?(.*)$", re.S)
_PAIR_SPLIT = re.compile(r"; (?=[^=;|]{1,60}=)")
_PAIR = re.compile(r"^([^=;|]{1,60})=(.*)$", re.S)


def _text_value(text: str) -> Any:
    """One part's value: None, a flag, a whole number, a block of name=value pairs, else the text itself."""
    value = text.strip()
    if value == "None":
        return None
    if value in ("True", "False"):
        return value == "True"
    if re.fullmatch(r"-?\d{1,12}", value):
        return int(value)
    if re.fullmatch(r"-?\d{1,12}\.\d{1,6}", value):
        return float(value)
    pieces = _PAIR_SPLIT.split(value)
    # one pair is a block too when its name is a short key (a description holding "=" has spaces around it or is long)
    single = len(pieces) == 1 and re.fullmatch(r"[A-Za-z][\w ]{0,30}=[^=]{0,200}", value) is not None
    if "=" in value and (len(pieces) >= 2 or single) and all(_PAIR.match(p) for p in pieces):
        return {m.group(1).strip(): m.group(2).strip() for m in (_PAIR.match(p) for p in pieces)}
    return value


def _text_record(text: Any) -> dict[str, Any] | None:
    """A text line holding one listing record as a dict, or None when the line is not such a record: at least four
    "key: value" parts, the first value id-like (no spaces, a digit), keys distinct."""
    if not isinstance(text, str) or text.count(" | ") < 3 or len(text) > 20000:
        return None
    parts = _TEXT_PART.split(text.strip())
    if len(parts) < 4:
        return None
    record: dict[str, Any] = {}
    for part in parts:
        found = _TEXT_KEY.match(part)
        if not found or found.group(1) in record:
            return None
        record[found.group(1)] = _text_value(found.group(2))
    first = next(iter(record.values()))
    return record if _id_like(first) else None


def _untext(observation: Any, depth: int = 0) -> Any:
    """The observation with every text-line listing record turned into a dict: a whole result that is one record (or
    wraps one under a single key), and records inside lists. Anything else is left as it is."""
    if depth > 5:
        return observation
    if isinstance(observation, str):
        record = _text_record(observation)
        return record if record is not None else observation
    if isinstance(observation, list):
        return [_untext(v, depth + 1) for v in observation]
    if isinstance(observation, dict):
        if len(observation) == 1:
            only = next(iter(observation.values()))
            record = _text_record(only)
            if record is not None:
                return record
        return {k: _untext(v, depth + 1) for k, v in observation.items()}
    return observation


def _tokens(value: Any) -> frozenset[str]:
    text = value if isinstance(value, str) else _dump(value)
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text.lower()))


def _time_given(problem: dict[str, Any]) -> float | None:
    """Seconds this episode still has when the runtime says so: a duration or a wall-clock moment under a telling key."""
    environment = problem.get("environment")
    view = environment.get("policy_view") if isinstance(environment, dict) else None
    for block in (view, environment, problem):
        for key, value in (block if isinstance(block, dict) else {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not re.search(r"deadline|time.?(limit|left|budget|out)|seconds", str(key), re.I):
                continue
            moment = value / 1000 if value > 1e12 else value
            left = moment - time.time() if moment > 1e9 else moment
            if 0 < left <= 3600:
                return float(left)
    return None


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


# ---------- n47: fit pairs between requested items, read by structure ----------
# n47: one requested item that must fit another item of the same order (the one the agent picks) is a pair to choose
# together, not a pointer to walk. The patterns below are structural (code shapes, "every ABC 12xx" wildcards, part
# numbers) or plain English fit words used as detectors; nothing here names a product, shop or code from any store.
# a model code: 2-5 capitals, an optional hyphen or space, 3-6 digits; case, spaces and hyphens do not matter in it
_CODE = r"[A-Z]{2,5}[- ]?\d{3,6}"
_WILD = r"([A-Z]{2,5})[- ]?(\d{0,5})([xX]{1,4})"
_SERIES = r"[A-Z]{1,4}\d{1,3}"
_SERIES_PART = re.compile(r"\b(" + _SERIES + r")-\d{3,}\b")
_ALIAS = r"(?:" + _CODE + r"|[A-Z][a-z]{2,}[- ]\d{1,4}[A-Za-z]?)"
_FIT_VERB = (r"(?:fits?|fitting|fitted|work(?:s|ing)?\s+(?:with|on|in|alongside)|go(?:es)?\s+(?:with|together\s+with)"
             r"|(?:be\s+)?compatible\s+with|suits?|pairs?\s+(?:up\s+)?with|mates?\s+(?:up\s+)?with|match(?:es)?"
             r"|made\s+for|designed\s+for|meant\s+for|built\s+for|plugs?\s+into|connects?\s+(?:to|with)|attach(?:es)?\s+to"
             r"|be\s+used\s+with|usable\s+with)")
# the thing to fit is an item this order picks: "the tablet you choose", "whichever one we buy", "in the same order"
_CHOSEN = (r"(?:the|that|this|my|our|your|whichever|whatever)\s+(?:[\w&'-]+\s+){0,5}?"
           r"(?:(?:you|we|i)(?:'ll|'re|'ve| will| are| am| have|'m)?\s+(?:\w+\s+){0,3}?(?:pick\w*|choos\w*|chose\w*|select\w*"
           r"|buy|buys|buying|order|orders|ordering|get|gets|getting|go\s+for|going\s+for|settle\w*\s+on|end\s+up\s+with|add\w*|put\s+in|cart\w*)\b"
           r"|(?:being\s+)?(?:picked|chosen|selected)\b|in\s+(?:the|this|my|our)\s+(?:same\s+)?(?:order|cart|basket))")
_FIT_CLAUSE = re.compile(r"\b" + _FIT_VERB + r"\s+(?:" + _CHOSEN + r"|(?:each\s+other|one\s+another|together))"
                         r"|\b(?:work(?:s|ing)?\s+(?:with|alongside)|go(?:es)?\s+with|compatible\s+with|pairs?\s+with|plugs?\s+into|connects?\s+to"
                         r"|fits?\s+(?:with|onto|into))\s+(?:it|them|both)\b(?!\s+(?:way|is|was|all))", re.I)
# a fit verb that names its target by a noun ("compatible with the tablet"); "match" and a bare "for" are left out:
# they also start pointers to other listings ("must match the one that shop sells")
_FIT_NAMED = re.compile(r"\b(?:fits?|fitting|work(?:s|ing)?\s+(?:with|on|alongside)|go(?:es)?\s+with|compatib\w*(?:\s+(?:with|for))?|suits?"
                        r"|pairs?\s+with|made\s+for|designed\s+for|meant\s+for|plugs?\s+into|connects?\s+to|usable\s+with|be\s+used\s+with)\b", re.I)
# the value is read from some other listing: a page, a listing, a seller or shop, a shop's possessive
_ELSEWHERE = re.compile(r"\b(?:page|listing|listed|sold\s+by|sells?|seller|shop|store'?s?|link|brand)\b", re.I)
_OWNER = re.compile(r"\b[A-Z][\w-]+(?:'s|’s)\s")
# n47 (review): something the shopper already owns is no item of this order ("the one I already have at home")
# ("we have picked", "you already chose" speak of this order's pick: a pick verb right after lets them through)
_PICK_NEXT = r"(?!\s+(?:\w+\s+)?(?:pick\w*|choos\w*|chose\w*|select\w*))"
_OWNED = re.compile(r"\b(?:already" + _PICK_NEXT + r"|at\s+home|last\s+(?:year|month|week)|years?\s+ago|existing|current(?:ly)?|my\s+own)\b"
                    r"|\b(?:i|we)\s+(?:\w+\s+)?(?:have|own|use|got|bought)\b" + _PICK_NEXT, re.I)
# n47 (review): a fit verb that is negated ("don't match each other") states no fit
_NEGATED = re.compile(r"\b(?:not|never|no)\s*$|n't\s*$", re.I)


def _fit_words(text: str) -> bool:
    """n47: True when the text says one requested item must fit / work with / suit another item of this order (the one the
    agent picks). A sentence whose fit target is another listing (a page, a seller's listing) is a pointer: False."""
    for sentence in re.split(r"(?<=[.!?;])\s+", text or ""):
        for found in _FIT_CLAUSE.finditer(sentence):
            rest = sentence[found.start():]
            if _NEGATED.search(sentence[:found.start()]) or _ELSEWHERE.search(rest) or _OWNER.search(rest) or _OWNED.search(rest):
                continue
            return True
    return False


_MODEL_LEAD = re.compile(r"\bmodel(?:\s*(?:no|nr|number|num|ref|code|id|#))?\s*[.:#]*\s*(" + _CODE + r")\b", re.I)
_ALIAS_LEAD = re.compile(r"\b(?i:also\s+(?:sold|known|marketed|listed|available|offered|branded|badged|called|named)\s+as"
                         r"|(?:sold|marketed|known)\s+elsewhere\s+as|a\.?k\.?a\.?|otherwise\s+known\s+as)\s*:?\s*"
                         r"(" + _ALIAS + r"(?:\s*(?:,|/|\bor\b|\band\b)\s*" + _ALIAS + r")*)")
_SERIES_LEAD = re.compile(r"\b(?:fits?|takes?|accepts?|uses?|supports?|works\s+with)\s+(?:the\s+)?parts?\s+(?:from\s+)?series\s*:?\s*"
                          r"(" + _SERIES + r"(?:\s*(?:,|/|\bor\b|\band\b)\s*" + _SERIES + r")*)", re.I)
# a list that goes on elsewhere ("see part ... for the rest"): what it leaves out is unknown, never a misfit
_MORE_LEAD = re.compile(r"\bsee\s+(?:also\s+)?(?:part|item|listing|product|page)\b[^.]{0,60}\b(?:rest|others?|more|remaining|full\s+list|remainder)\b"
                        r"|\b(?:more|rest|others)\s+(?:is\s+|are\s+)?(?:on|at|in)\s+(?:part|item|listing|product|page)\b"
                        r"|\b(?:and\s+(?:many\s+)?more|among\s+others|full\s+list\s+(?:on|at|in))\b", re.I)
_ACC_LEAD = re.compile(r"\b(?:compatib(?:le|ility)(?:\s+(?:with|for))?|(?:supported|compatible|suitable)\s+(?:models?|devices?|machines?)"
                       r"|works?\s+(?:with|on)|fits?(?:\s+(?:for|on|with))?|goes\s+with|go\s+with|suits?|made\s+for|designed\s+for"
                       r"|for\s+use\s+(?:with|on)|pairs?\s+with|matches|suitable\s+for)\b\s*:?\s*", re.I)
_EXCEPT = re.compile(r"\b(?:except(?:\s+for)?|excluding|but\s+not|other\s+than|apart\s+from|save\s+for|not\s+(?:for|the))\b", re.I)
# a sentence about some other part ("Part QZ-123 also fits: ..."): its codes are not this listing's
_OTHER_PART = re.compile(r"^\s*(?:the\s+)?(?:part|item|listing|product|sku)\s+[A-Z0-9][\w-]*\s+(?:also\s+)?$", re.I)
_EXTRA_SERIES = re.compile(r"\bmodel\s+(" + _CODE + r")\s+(?:also\s+)?(?:takes|accepts|fits|uses|supports)\s+(?:the\s+)?parts?\s+series\s+"
                           r"(" + _SERIES + r"(?:\s*(?:,|\bor\b|\band\b)\s*" + _SERIES + r")*)", re.I)


def _norm_code(text: str) -> str:
    return re.sub(r"[\s\-_./]", "", text).upper()


def _flat_lines(node: Any, key: str = "", depth: int = 0) -> list[str]:
    """n47: a record as plain lines ("model: ABC-1234"; a sentence stays on its own), so the code patterns read its words
    rather than JSON quotes."""
    if depth > 5 or node is None or isinstance(node, bool):
        return []
    if isinstance(node, dict):
        return [line for k, v in node.items() for line in _flat_lines(v, _words(k), depth + 1)]
    if isinstance(node, list):
        if node and all(not isinstance(v, (dict, list)) for v in node):
            return [(key + ": " if key else "") + ", ".join(str(v) for v in node)]
        return [line for v in node for line in _flat_lines(v, key, depth + 1)]
    text = str(node).strip()
    if not text:
        return []
    return [text] if not key or len(text.split()) >= 6 else [f"{key}: {text}"]


def _flat(record: Any) -> str:
    return "\n".join(_flat_lines(record))


def _device_codes(text: str) -> set[str]:
    """n47: the model codes a device listing carries (its model number and any second name it is sold as), normalised;
    the part series it takes as "SERIES:XY3"; "MORE" when its series list goes on elsewhere; "NAMED" when it has a
    second product name."""
    out: set[str] = set()
    for found in _MODEL_LEAD.finditer(text or ""):
        out.add(_norm_code(found.group(1)))
    for found in _ALIAS_LEAD.finditer(text or ""):
        for name in re.findall(_ALIAS, found.group(1)):
            out.add(_norm_code(name))
            out.add("NAMED")
    for found in _SERIES_LEAD.finditer(text or ""):
        out.update("SERIES:" + s.upper() for s in re.findall(_SERIES, found.group(1)))
    if any(c.startswith("SERIES:") for c in out) and _MORE_LEAD.search(text or ""):
        out.add("MORE")
    return out


def _wild_covers(letters: str, start: str, xs: str, code: str) -> bool:
    found = re.fullmatch(r"([A-Z]+)(\d+)", code)
    return bool(found) and found.group(1) == letters.upper() and found.group(2).startswith(start) and len(found.group(2)) == len(start) + len(xs)


def _accessory_fits(text: str, device: set[str]) -> bool | None:
    """n47: whether an accessory's own compatibility words cover a device's codes: explicit code lists, "every ABC 12xx
    model" wildcards with "except" exclusions, a range or family, part series. True when they cover it; False when they
    name codes of the device's own family (or its series, or second names against a named device) and none covers it;
    None when they say nothing about such codes or the list goes on elsewhere (unknown is never a misfit)."""
    codes = {c for c in device if not c.startswith("SERIES:") and c not in ("MORE", "NAMED")}
    series = {c[7:] for c in device if c.startswith("SERIES:")}
    if not text or not (codes or series):
        return None
    stated = False
    covered = False
    open_ended = bool(_MORE_LEAD.search(text))
    own = {m.group(1).upper() for m in _SERIES_PART.finditer(text)}
    if own and series:
        stated = True
        covered = bool(own & series)
    completed = False
    for found in _EXTRA_SERIES.finditer(text):
        if _norm_code(found.group(1)) in codes:
            completed = True
            if own & {s.upper() for s in re.findall(_SERIES, found.group(2))}:
                covered = True
    if own and series and not covered and "MORE" in device and not completed:
        open_ended = True
    families = {re.match(r"[A-Z]+", c).group(0) for c in codes if re.match(r"[A-Z]+", c)}
    for sentence in [s for s in re.split(r"\.(?=\s|$)|[\n;!?]", text) if s.strip()]:
        for lead in _ACC_LEAD.finditer(sentence):
            if _OTHER_PART.match(sentence[:lead.start()]):
                continue
            tail = sentence[lead.end():]
            gap = _EXCEPT.search(tail)
            body, barred_text = (tail[:gap.start()], tail[gap.end():]) if gap else (tail, "")
            barred = {_norm_code(c) for c in re.findall(_CODE, barred_text)}
            wild = [(w.group(1), w.group(2), w.group(3)) for w in re.finditer(r"\b" + _WILD + r"\b", body)]
            wild += [(w.group(1), w.group(2), "") for w in re.finditer(r"\b([A-Z]{2,5})[- ]?(\d{1,4})\s+(?:range|family|line|series)\b", body)]
            listed = {_norm_code(c) for c in re.findall(r"\b" + _CODE + r"\b", body)}
            named = {_norm_code(c) for c in re.findall(r"\b[A-Z][a-z]{2,}[- ]\d{1,4}[A-Za-z]?\b", body)}
            if re.search(r"\bseries\b", body, re.I):
                in_series = {x.upper() for x in re.findall(r"\b" + _SERIES + r"\b", body)}
                covered = covered or bool(in_series & series)
                stated = stated or bool(in_series and series)
            if (listed | named) & codes:
                covered = True
            for letters, start, xs in wild:
                for code in codes - barred:
                    if (xs and _wild_covers(letters, start, xs, code)) or (not xs and re.fullmatch(letters.upper() + start + r"\d*", code)):
                        covered = True
            if {re.match(r"[A-Z]+", c).group(0) for c in listed} & families or {w[0].upper() for w in wild} & families:
                stated = True
            if named and "NAMED" in device:
                stated = True
            if listed or wild or named:
                break
    if covered:
        return True
    if not stated or open_ended:
        return None
    return False


# ---------- n47: requirement values read by structure ----------
# an extracted value that excludes one value: "not X", "other than X", "anything but X", "except X", at its start
_EXCLUDE = re.compile(r"^\s*(?:not|other\s+than|anything\s+(?:but|other\s+than|except)|except(?:\s+for)?)\s+(?P<x>\S.*?)\s*\.?\s*$", re.I)
# a comparison after "not" is a limit, not an exclusion ("not more than 5 kg")
_COMPARE = re.compile(r"^(?:more|less|fewer|over|under|above|below|exceed\w*|bigger|smaller|larger|heavier|lighter|longer|shorter|higher|lower|wider|narrower|than|too|much|many|quite|very|so)\b", re.I)


_NO_VALUE = re.compile(r"^(?:specified|stated|sure|applicable|none|any|anything|known|available|required|needed|necessary|important|fussed|bothered"
                       r"|picky|given|listed|set|mentioned|decided|that|this|it|those|these|one|really|much|many)\b", re.I)


def _excluded(value: str) -> str | None:
    """n47: the value an extracted requirement excludes ("not wicker" -> "wicker"), None for any other value."""
    found = _EXCLUDE.match(value or "")
    if not found or len(value.split()) > 7:
        return None
    core = re.sub(r"^(?:a|an|the)\s+", "", found.group("x").strip(" '\""), flags=re.I)
    # n47 (review): "not specified", "not sure", "except none" are no values; several values joined are left to the model
    if not core or _COMPARE.match(core) or _NO_VALUE.match(core) or re.search(r",|\b(?:or|and|nor)\b", core, re.I):
        return None
    return core


def _measure(value: str) -> bool:
    """n47: a value holding a figure ("16 GB", "14-inch", "2.5"), not a name that has digits in it ("x5", "a7-12")."""
    return any(re.fullmatch(r"\d+(?:[.,]\d+)?(?:-?[a-z]{1,5}|%|\")?", t) for t in str(value).lower().split())


def _same_value(a: str, b: str) -> bool:
    """n47: two values are the same words, case, plural and punctuation aside ("Wicker" and "wicker")."""
    def norm(text: str) -> list[str]:
        return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())]
    return bool(norm(a)) and norm(a) == norm(b)


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Umi:
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
        self.reaudit = False         # a shopper note changed the task since the last decline review
        self.noted: set[str] = set()  # listings viewed with the tool that shows the shop's notes
        self.walk_resumed = False
        self.order_walks = 0
        self.walk_tried: set[str] = set()
        self.swept: set[str] = set()
        self.mark_hits: set[str] = set()
        self.detail_words: dict[str, frozenset[str]] = {}
        self.swap_searched = False
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
        self.qty_rounds = 0          # n49: rounds of code's own quantity fixes
        self.shelf_blocks = 0        # n49: checkouts held for a requested item with no cart line
        self.stated_swapped = False  # n49: one swap round for lines stating a requirement nowhere
        self.pushed = False
        self.decline_paged = False
        self.fits: dict[str, tuple[bool, str]] = {}
        self.deep_reads = 0
        self.chain_links: set[str] = set()
        self.source_held = False
        self.unpaged: dict[str, list[Any]] = {}
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
        self.empty_blocks = 0
        self.swap_picks: dict[str, list[str]] = {}
        self.finish_swapped = False
        self.finish_filled = False
        self.pointer_needs: set[str] = set()
        self.pointer_blocks = 0
        self.pointer_filled = False
        self.pointer_forced = False
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
        self.extra_held = False
        self.drop_open = False
        self.paged_obs: list[Any] = []
        self.began = time.monotonic()
        given = _time_given(problem)
        self.limit = min(TASK_SECONDS, max(60.0, given - 30)) if given else TASK_SECONDS
        self.hunt_queue: list[tuple[int, str]] = []
        self.hunts = 0
        self.hunt_steps = 0
        self.trail_lost: list[str] = []
        self.hunted: set[str] = set()
        self.typed: list[dict[str, Any]] = []
        self.last_found: Any = None
        self.budget_first: Any = None
        self.basket_marks: set[tuple[Any, str]] = set()
        # n47: pairs (item that must fit, item it must fit) and the requirement texts that state such a fit
        self.fit_pairs: list[tuple[int, int]] = []
        self.fit_needs: set[str] = set()
        self.code_held: set[str] = set()      # n47: carts the code check has held once
        self.pair_held = False                # n47: a decline held once for an untried pair switch
        self.decline_filled = False           # n47: a decline turned once into a cart of code-confirmed listings
        self.leads_filled = False             # n47: a forced finish's empty cart filled once from the review's named fits
        self.rows_audited = False             # n47: the earlier rows go into the first decline review only (cost)

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        return self.limit - (time.monotonic() - self.began)

    def late(self) -> bool:
        return self.time_left() < LATE_SECONDS

    def rushed(self) -> bool:
        return self.time_left() < FORCE_SECONDS

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
        if self.facts_from.get(tool or "") is not False and self.single_target(tool):
            return tool
        # the brief named no working view: any other one-listing tool, once the jobs that also take one text are known
        jobs = {self.roles.get(r) for r in ("search", "cart", "cart_view", "buy", "refuse", "ask")}
        if not all(self.roles.get(r) for r in ("search", "cart", "buy", "ask")):
            return None
        return next((t for t in self.schemas if t not in jobs and self.single_target(t) and self.facts_from.get(t) is not False), None)

    def one_read(self, tool: str | None, ident: str) -> tuple[str, dict[str, Any]] | None:
        """The call showing one listing: this single view, or the side-by-side view when no single view exists."""
        target, many = self.single_target(tool), self.compare_target()
        name, args = (tool, {target: ident}) if tool and target else (self.roles.get("compare"), {many: [ident]}) if many else (None, {})
        faults: list[str] = []
        args = _shape(args, self.schemas.get(name or ""), "args", faults)
        return (name, args) if name and not faults else None

    def compare_facts(self) -> bool | None:
        """Whether the side-by-side view shows attributes: True or False once seen, None before."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or, not tried yet, the store has a separate
        attributes view (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        if seen is None and not self.facts_tool():
            return False
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            # out of credit: no call starts, so the episode ends at once with what it holds
            if self.calls >= ALL_CALLS or self.rushed() or self.no_credit or self.cost >= self.ceiling():
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # rate limits, timeouts and server errors pass: wait and send the same call again
                # no credit is final: waiting would only keep the episode from finishing
                passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL or self.time_left() - WAITS[tries] < FORCE_SECONDS:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            # a spent run key answers 402 (no credit) or 403 (key limit): final for the whole run
            if status in (402, 403):
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
        # a call never outlives the episode: the client's 50 s shrink as the clock runs down
        _llm.timeout = max(8, min(50, int(self.time_left())))
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
        _env.timeout = max(5, min(95, int(self.time_left()) + 20))
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                # n48: a store sending listing records as text lines gets them read as dicts, like any other store
                if "observation" in data:
                    try:
                        data["observation"] = _untext(data["observation"])
                    except Exception:
                        pass
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                if name == self.notes_tool() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose page with the shop's notes the agent has seen: only these count as read links of a chain
                    for value in args.values():
                        self.noted.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
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
        if self.no_credit:
            # no model could read the tools: the finish still needs a way to end, so the ending tools are known by their names
            for job, words in (("buy", r"checkout|check_out|purchase|buy|place_order|finali[sz]e"), ("refuse", r"refuse|decline|abstain|abandon|give_?up|no_purchase")):
                if not self.roles.get(job):
                    hit = [t for t in self.schemas if re.search(words, t, re.I) and not (self.schemas.get(t) or {}).get("required")]
                    if len(hit) == 1:
                        self.roles[job] = hit[0]
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
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        # n47: one requested item that must fit another one of this order is a pair to choose together, never a pointer:
        # the brief's own pairs, then every must worded as such a fit
        # a pair the brief names counts only when the request itself has fit wording (a misread pair would add a requirement)
        worded = _fit_words(self.request)
        for entry in _items(brief.get("fits")) if worded else []:
            if isinstance(entry, dict):
                self.add_pair(_count(entry.get("item")), _count(entry.get("with")))
        for number, item in enumerate(self.items, 1):
            for must in item.musts:
                target = self.guarded(self.fit_target, None, must, number)
                if target is not None:
                    self.fit_needs.add(must)
                    self.add_pair(number, target)
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            given = text
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            # n47: safety net: a "lookup" that is a fit between requested items is moved out of the pointer chain
            target = next((t for t in (self.guarded(self.fit_target, None, given, number), self.guarded(self.fit_target, None, text, number)) if t is not None), None)
            if text and target is not None:
                self.fit_needs.update({given, text})
                self.add_pair(number, target)
                self.trace.append({"kind": "fit_not_lookup", "need": text[:160], "with": target})
                continue
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        self.budget_first = self.budget
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
            # the messages a stated step right before the order takes; a note before an ending without a purchase gets none held
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
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            if item.dropped:
                text += "; given up by the shopper, so no longer required even where the request still states it: " + "; ".join(item.dropped)
            parts.append(text)
        for a, b in self.fit_pairs:
            # n47: the pair is a requirement on the two picks together, judged from the listings' own words
            parts.append(f"fit: item {a} ({self.items[a - 1].name}) must fit the item {b} ({self.items[b - 1].name}) listing you pick; choose the two together"
                         " (for an accessory and a device: by the accessory's own compatibility words and the device's model codes, which Umi marks beside results)")
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
            when = {"choose": "before choosing", "order": "at the order (Umi does it when you check out)", "end": "at an ending without a purchase (Umi does it then)"}[entry["when"]]
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
        text = f"[Umi] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and len(self.items) > 1 and self.cost >= 0.7 * self.spend_cap() and self.uncarted():
            text += f" Most of this task's budget is used and {self.uncarted()} item(s) have no cart line: cart each item's best fitting listing found so far now, then check out."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.late() and self.buys == 0:
            text += " The episode's time is nearly out: set the cart to the chosen lines and check out now."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
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
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                # all different and long reads as identifiers (models, part numbers), unless two listings share a word:
                # "mediatek helio g85" and "mediatek dimensity 6300" are values of one property, not identifiers
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and not _shared_word(values):
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:KEY_ROWS]:
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
        observation = _peel(observation)
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
        observation = _peel(observation)
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        # no model reading (spend, calls or time ran out): the parts by their look
        if got.get("rows") not in lists:
            got = self.eyed_shape(name, observation, lists)
        rows = got.get("rows")
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def eyed_shape(self, name: str, observation: dict[str, Any], lists: list[str]) -> dict[str, Any]:
        """A row list's parts without a model: the longest list, the largest number as the match count, the other value
        as the next page, the schema's paging parameter, the row's one number and one flag."""
        rows = max(lists, key=lambda k: len(observation[k]))
        figures = [k for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)]
        count = max(figures, key=lambda k: observation[k], default=None)
        follow = next((k for k, v in observation.items() if k not in (rows, count) and (v is None or isinstance(v, (int, str)))), None)
        props = (self.schemas.get(name) or {}).get("properties") or {}
        turn = next((k for k, v in props.items() if isinstance(v, dict) and "enum" not in v
                     and re.search(r"page|offset|cursor|token|start|begin|after|next|skip", f"{k} {v.get('description') or ''}", re.I)), None)
        record = observation[rows][0]
        prices = [k for k, v in record.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        flags = [k for k, v in record.items() if isinstance(v, bool)]
        return {"rows": rows, "count": count, "next": follow, "next_arg": turn, "price": prices[0] if len(prices) == 1 else None, "stock": flags[0] if len(flags) == 1 else None}

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        observation = _peel(observation)
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
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Umi reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        ranked = pick in ("cheapest", "most", "least")
        plan = self.paging(name, observation)
        if not plan:
            return ""
        count, shown, param, value = plan
        # any pick: a short result is read to its end while steps are plentiful, since the fitting listing is often on a later page
        if not ranked and not (shown < count <= SHORT_RESULT and self.left() - self.tail() - (-(-(count - shown) // shown)) >= 16):
            return ""
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if not ranked and (self.pages_read + pages > AUTO_PAGES):
            return ""
        deep = pick == "cheapest" and pages > 2
        if deep:
            # a short task affords one deep read: its steps are needed for the shopper's changes at the checkout
            if self.max_steps <= 40 and self.deep_reads >= 1:
                return (f"\n[Umi] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                        f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
            pages = DEEP_PAGES
        if ranked and ((pages > 2 and not deep) or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12):
            return (f"\n[Umi] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            self.note_pages(name, call, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: " + (self.short_rows(seen) if deep else shown_text))
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts), "deep": deep})
        if deep and texts:
            self.deep_reads += 1
        if deep and texts:
            rest = f" More pages remain ({param}={_dump(value)})." if value is not None else ""
            return (f"\n[Umi read the next {len(texts)} pages of this search itself (one step a page; rows shortened: id | price | stock | title). "
                    f"It matched {count} listings, in no price order, so the cheapest fitting one may be further on: a search with the requirement's value words narrows it.{rest}]\n" + "\n".join(texts))
        return ("\n[Umi read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def short_rows(self, observation: Any) -> str:
        """A result page as short rows: id, price, stock and the start of the title, with the compatibility mark."""
        out = []
        for ident, record in self.records_in(observation):
            keys = [k for k, v in record.items() if isinstance(v, str) and re.search(r"name|title", k, re.I) and not re.search(r"shop|brand|seller|maker|store|vendor", k, re.I)]
            title = str(record[keys[0]]) if keys else ""
            price = self.price_of(ident)
            fit = self.fit_mark(ident)
            out.append(f"{ident} | {price if price is not None else '?'} | {'in stock' if self.buyable(ident) else 'sold out'} | {title[:90]}" + (f" | {fit}" if fit else ""))
        return "\n".join(out) or self.show(observation)

    def note_pages(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Where each search stopped: its arguments without the page parameter, the next page's value, matches and rows a page."""
        plan = self.paging(name, observation)
        if not plan:
            return
        count, shown, param, value = plan
        base = {k: v for k, v in args.items() if k != param}
        key = name + _dump(base)
        if value is None or count <= shown:
            self.unpaged.pop(key, None)
        else:
            self.unpaged[key] = [name, base, param, value, count, shown]

    def rest_pages(self, entry: list[Any]) -> int:
        _, _, _, value, count, shown = entry
        start = value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else shown
        return max(1, -(-(count - start) // max(1, shown)))

    def page_rest(self, room: int) -> str:
        """Before an ending without a purchase: the unread pages of searches already run, shortest rest first, a step a
        page and no model turn. A fitting listing is often on a page nobody read; on a task with none, the pages show
        none and the ending stands. A rest longer than the room is left alone."""
        budget = min(room, DECLINE_PAGES)
        texts: list[str] = []
        for key, entry in sorted(self.unpaged.items(), key=lambda kv: self.rest_pages(kv[1])):
            if budget <= 0 or self.finished:
                break
            if self.rest_pages(entry) > budget:
                continue
            name, base, param, value, _, _ = entry
            while value is not None and budget > 0 and not self.finished:
                faults: list[str] = []
                call = _shape({**base, param: value}, self.schemas.get(name), "args", faults)
                if faults:
                    break
                reply = self.act(name, call)
                if reply is None:
                    break
                budget -= 1
                self.pages_read += 1
                seen = reply.get("observation")
                self.observe(name, seen)
                shown_text = self.show(seen)
                self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
                texts.append(f"{_dump(base)} {param}={_dump(value)}: {shown_text}" + self.fit_marks(seen))
                after = self.paging(name, seen)
                value = after[3] if after else None
            self.unpaged.pop(key, None)
        self.trace.append({"kind": "decline_pages", "read": len(texts), "left": self.left()})
        return "\n".join(texts)

    def own_codes(self) -> dict[str, set[str]]:
        """Model or machine codes the request names (letters then digits, such as ABC 1234 or ABC-1234), by their letters."""
        found: dict[str, set[str]] = {}
        for letters, digits in re.findall(r"\b([A-Z]{2,5})[- ]?(\d{3,5})\b", self.request or ""):
            found.setdefault(letters, set()).add(digits)
        for series in re.findall(r"\bpart series ([A-Z]{1,4}\d{1,3})\b", self.request or "", re.I):
            found.setdefault("series", set()).add(series.upper())
        return found

    def fit_of(self, text: str) -> tuple[bool, str] | None:
        """Whether a listing's fitting note covers a code the request names: "Fits: every ABC 12xx model" covers ABC 1234
        (the x's stand for the remaining digits) unless an "except" names it; "Fits part series XY2, XY5" covers a series the
        request or the named machine has. None when the note names nothing the request has a code for."""
        codes = self.own_codes()
        if not codes:
            return None
        note = re.search(r"(?:[Ff]its|[Ww]orks with|[Cc]ompatible with)[\"':\s]+(?:every|all|any)\s+([A-Z]{2,5})[- ]?(\d+)([xX]+)\s+models?([^.;)\n\"]{0,100})", text)
        if note and note.group(1) in codes:
            letters, start, wild, tail = note.group(1), note.group(2), note.group(3), note.group(4) or ""
            gap = re.search(r"\bexcept\b(.*)", tail)
            barred = set(re.findall(r"\d{3,5}", gap.group(1))) if gap else set()
            mine = sorted(codes[letters])
            ok = [d for d in mine if len(d) == len(start) + len(wild) and d.startswith(start) and d not in barred]
            shown = f"{letters} {(ok or mine)[0]}"
            told = f"its note covers every {letters} {start}{'x' * len(wild)} model" + (f", except{gap.group(1).rstrip()}" if gap else "")
            return bool(ok), f"{shown}: {told}"
        series = re.search(r"part series\s*([^.;)\n\"]{1,80})", text)
        if not series or re.search(r"[Mm]odel[\"':\s]+[A-Z]{2,5}[- ]?\d{3,5}", text):
            return None
        wanted = set(codes.get("series", set())) | self.machine_series()
        if wanted:
            have = {x.upper() for x in re.findall(r"\b[A-Z]{1,4}\d{1,3}\b", series.group(1))}
            if have:
                return bool(have & wanted), f"part series {', '.join(sorted(wanted))}: its note lists part series {', '.join(sorted(have))}"
        return None

    def machine_series(self) -> set[str]:
        """Part series of a machine whose model code the request names, from that machine's listing ("Model: ABC-1234. Fits part series XY3, XY9")."""
        codes = self.own_codes()
        out: set[str] = set()
        for record in self.rows.values():
            text = _dump(record)
            model = re.search(r"[Mm]odel[\"':\s]+([A-Z]{2,5})[- ]?(\d{3,5})", text)
            if model and model.group(2) in codes.get(model.group(1), set()):
                series = re.search(r"part series\s*([^.;)\n\"]{1,80})", text)
                if series:
                    out |= {x.upper() for x in re.findall(r"\b[A-Z]{1,4}\d{1,3}\b", series.group(1))}
        return out

    def fit_mark(self, ident: str) -> str:
        record = self.rows.get(ident)
        got = self.fit_of(_dump(record)) if record else None
        if got is None:
            return ""
        self.fits[ident] = got
        named, _, why = got[1].partition(": ")
        return ("fits " if got[0] else "does NOT fit ") + f"{named} ({why})"

    def fit_marks(self, observation: Any) -> str:
        """The compatibility mark of each listing a result shows, when the request names a code its fitting notes speak of."""
        marks = [f"{ident}: {mark}" for ident, _ in self.records_in(observation) for mark in [self.fit_mark(ident) or self.guarded(self.pair_mark, "", ident)] if mark]
        return ("\n[Umi compatibility, from each listing's fitting note] " + " | ".join(marks[:12])) if marks else ""

    # ---------- n47: fit pairs between requested items ----------
    def add_pair(self, item: int | None, other: int | None) -> None:
        if item and other and item != other and 1 <= item <= len(self.items) and 1 <= other <= len(self.items) and (item, other) not in self.fit_pairs:
            self.fit_pairs.append((item, other))

    def fit_target(self, text: str, number: int) -> int | None:
        """n47: whether a requirement of item `number` says it must fit another requested item of this order (the one the
        agent picks): that item's number, or None when it is no such fit or names no requested item clearly (n45's reading
        then stands). A requirement that takes its value from another listing (a page, a shop's or seller's listing, a
        capitalised name), or that speaks of something the shopper already owns, gives None."""
        if len(self.items) < 2 or not text or POINTER.search(text):
            return None

        def long(words: str) -> set[str]:
            return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", words.lower()) if len(w) >= 4}
        others = {n: long(it.name + " " + (it.shelf or "")) for n, it in enumerate(self.items, 1) if n != number}

        def named(words: str) -> int | None:
            # the other item sharing the most words with the fit's target; a tie names none
            scores = sorted(((len(have & long(words)), n) for n, have in others.items()), reverse=True)
            if not scores or not scores[0][0] or (len(scores) > 1 and scores[0][0] == scores[1][0]):
                return None
            return scores[0][1]
        if _fit_words(text):
            for clause in _FIT_CLAUSE.finditer(text):
                rest = text[clause.start():]
                if _NEGATED.search(text[:clause.start()]) or _ELSEWHERE.search(rest) or _OWNER.search(rest) or _OWNED.search(rest):
                    continue
                return named(rest)
            return None
        # a fit verb in the value (not the property's name, "fit: slim"), then "the / that / whichever ..." and another
        # requested item's name, with no other listing, capitalised name or owned thing after it
        head, colon, tail = text.partition(":")
        value = tail if colon and len(head.split()) <= 4 else text

        def plain_target(rest: str) -> bool:
            words = re.findall(r"[\w'-]+", rest)[:6]
            return (bool(re.match(r"\s*(?:in\s+)?(?:with\s+)?(?:the|that|this|whichever|whatever|each|both)\b", rest, re.I))
                    and not _ELSEWHERE.search(rest) and not _OWNER.search(rest) and not _OWNED.search(rest)
                    and not any(w[:1].isupper() and w != "I" for w in words))
        for verb in _FIT_NAMED.finditer(value):
            rest = value[verb.end():]
            if _NEGATED.search(value[:verb.start()]) or not plain_target(rest):
                continue
            target = named(" ".join(re.findall(r"[\w'-]+", rest)[:6]))
            if target:
                return target
        # the brief already paired this item: its fit wording with a plain target is that pair
        declared = [b for a, b in self.fit_pairs if a == number]
        for verb in _FIT_NAMED.finditer(value) if declared else []:
            if not _NEGATED.search(value[:verb.start()]) and plain_target(value[verb.end():]) and not _ELSEWHERE.search(text) and not _OWNER.search(text):
                return declared[0]
        return None

    def item_of(self, ident: str) -> int | None:
        """n47: the requested item a listing is for, for n47's own checks: its category label equal to an item's label or
        name, else the one item whose words hold the label's head noun (its last word: "projector mounts" is a mount),
        else the one item sharing the most long words with it; None when that is not clear. n45's item_number_of keeps
        its first-match reading for n45's own uses."""
        if len(self.items) == 1:
            return 1
        cat_key = self.category_key()
        record = self.full.get(ident) or self.rows.get(ident) or {}
        label = str(record.get(cat_key) or "") if cat_key else ""
        if not label.strip():
            return None
        for number, item in enumerate(self.items, 1):
            if _words(label) in (_words(item.shelf or ""), _words(item.name)):
                return number

        def long(text: str) -> set[str]:
            return {w for w in _stem(text) if len(w) >= 4}
        words = {n: long(it.shelf or "") | long(it.name) for n, it in enumerate(self.items, 1)}
        head = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", label.lower())][-1:]
        heads = [n for n, have in words.items() if head and head[0] in have]
        if len(heads) == 1:
            return heads[0]
        scores = sorted(((len(long(label) & have), n) for n, have in words.items()), reverse=True)
        if scores[0][0] and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            return scores[0][1]
        return None

    def pair_of(self, accessory: str, device: str) -> bool | None:
        """n47: whether the accessory listing's own compatibility words cover the device listing's model codes (its model
        number and second names, read from its own text); None when either says nothing code-shaped about it."""
        acc = self.full.get(accessory) or self.rows.get(accessory)
        dev = self.full.get(device) or self.rows.get(device)
        if not acc or not dev or accessory == device:
            return None
        codes = _device_codes(_flat(dev))
        text = _flat(acc)
        # a listing that is itself a device of those codes is not judged as their accessory
        if not codes or {_norm_code(m.group(1)) for m in _MODEL_LEAD.finditer(text)} & codes:
            return None
        return _accessory_fits(text, codes)

    def pair_verdict(self, ident: str, number: int, lines: list[str] | None = None) -> tuple[bool | None, str]:
        """n47: the pair check of a listing of item `number` against the lines of its partner item (the cart's unless
        `lines` is given): False when any partner is not covered, True when one is, else None; and that partner."""
        if not self.fit_pairs:
            return None, ""
        lines = self.cart_ids() if lines is None else lines
        result: tuple[bool | None, str] = (None, "")
        for a, b in self.fit_pairs:
            if number not in (a, b):
                continue
            for other in lines:
                if other == ident or self.item_of(other) != (b if number == a else a):
                    continue
                got = self.pair_of(ident, other) if number == a else self.pair_of(other, ident)
                if got is False:
                    return False, other
                if got is True:
                    result = (True, other)
        return result

    def pair_mark(self, ident: str) -> str:
        """n47: beside a result, whether a listing of a paired item fits the partner listing in the cart."""
        number = self.item_of(ident) if self.fit_pairs else None
        if not number or ident in self.cart_ids():
            return ""
        got, other = self.pair_verdict(ident, number)
        if got is True:
            return f"fits the cart's {other} (its compatibility words cover that model)"
        if got is False:
            return f"does NOT fit the cart's {other} (its compatibility words name other models)"
        # a list that goes on on another part: unknown, so that part is worth opening (never a misfit)
        devices = [i for a, b in self.fit_pairs if a == number for i in self.cart_ids() if self.item_of(i) == b]
        if devices and _MORE_LEAD.search(_flat(self.full.get(ident) or self.rows.get(ident) or {})):
            return f"its compatibility list goes on on another part: open that part to see whether it covers the cart's {devices[0]}"
        return ""

    def pair_options(self, accessory_item: int, device_item: int, device: str | None) -> list[str]:
        """n47: what a pair switch can take: accessories seen whose own words cover the device, else pairs of another device
        and an accessory seen that covers it (buyable listings of the two items only)."""
        def ok(ident: str, number: int) -> bool:
            return self.buyable(ident) and ident not in self.gone and self.item_of(ident) == number
        accessories = [i for i in reversed(self.seen_order) if ok(i, accessory_item)][:40]
        if device:
            covering = [i for i in accessories if self.pair_of(i, device) is True]
            if covering:
                return [f"{i} covers {device}" + (" (read)" if i in self.full else " (row only: open it)") for i in covering[:4]]
        combos = []
        for dev in [i for i in reversed(self.seen_order) if ok(i, device_item) and i != device][:15]:
            hit = next((a for a in accessories if self.pair_of(a, dev) is True), None)
            if hit:
                combos.append(f"{hit} covers {dev}")
            if len(combos) >= 3:
                break
        return combos

    def adds_gone(self, name: str, args: dict[str, Any]) -> list[str]:
        """n47: listings a cart call would put back although they sold out from the cart and are still shown as unavailable
        (a rebuy of a sold-out line has never passed). Removals are never blocked."""
        if name != self.roles.get("cart") or not self.gone:
            return []
        spec = (self.schemas.get(name) or {}).get("properties") if isinstance(self.schemas.get(name), dict) else None
        spec = spec if isinstance(spec, dict) else {}
        options = {k: v.get("enum") for k, v in spec.items() if isinstance(v, dict) and isinstance(v.get("enum"), list)}
        removal = any(len(options.get(k) or []) >= 2 and v in options[k][1:] for k, v in args.items())
        counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
        if removal or (counts and counts[0] <= 0):
            return []
        out: list[str] = []
        for value in args.values():
            for line in (value if isinstance(value, list) else [value]):
                if isinstance(line, dict):
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.gone), None)
                    count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
                    if ident and ident in self.unavailable and count and count > 0:
                        out.append(ident)
                elif isinstance(line, str) and line in self.gone and line in self.unavailable:
                    out.append(line)
        return out

    def cart_note(self, args: dict[str, Any]) -> str:
        """n47: after a cart call, what code reads on the listings it put in: a hard fault or a requirement not stated."""
        put = [v for value in args.values() for v in (value if isinstance(value, list) else [value])
               if isinstance(v, str) and v in self.rows] + [v for value in args.values() if isinstance(value, list)
                                                             for line in value if isinstance(line, dict) for v in line.values() if isinstance(v, str) and v in self.rows]
        notes = []
        for ident in dict.fromkeys(put):
            number = self.item_of(ident)
            if not number or ident not in self.cart_ids():
                continue
            fault = self.hard_fault(ident, number)
            gaps = self.gap_notes(ident, number, self.cart_ids())
            if fault:
                notes.append(f"{ident}: {fault}; replace it")
            elif gaps:
                notes.append(f"{ident}: not stated under its own name: " + "; ".join(gaps[:4]) + " (a hint: if this line meets it by meaning, keep it)")
        return ("\n[Umi code reading] " + " | ".join(notes)) if notes else ""

    # ---------- n47: requirements checked by structure on a pick ----------
    def plain_needs(self, number: int) -> list[tuple[str, str, str]]:
        """n47: (need, property, value) of item `number` that code may read: not a pointer, not a fit between items."""
        pointed = {need for _, need in self.refs}
        out = []
        for n, need in self.needs_now():
            prop, _, value = need.partition(":")
            if n == number and prop.strip() and value.strip() and need not in pointed and need not in self.fit_needs:
                out.append((need, prop.strip(), value.strip()))
        return out

    def own_keys(self, ident: str, prop: str, exact: bool = False) -> list[tuple[str, Any]]:
        """n47: the listing's attributes whose key names the property: any spelling (case, camel case, underscores, a plural,
        a prefix such as "frame material" for "material"); list values count element by element."""
        record = self.full.get(ident) or {}
        blocks = [v for v in record.values() if isinstance(v, dict)]
        out = []
        for block in blocks:
            for key, value in block.items():
                if isinstance(value, (dict, bool)) or value is None:
                    continue
                same = _stem(prop) == _stem(key) or _words(prop).replace(" ", "") == _words(key).replace(" ", "")
                # n47 (review): with `exact`, only a key that is the property itself ("strap color" is not "color")
                if same or (not exact and _names(_stem(prop), _stem(key))):
                    out.append((str(key), value))
        return out

    def excluded_hit(self, ident: str, number: int) -> str:
        """n47: hard rule. An extracted requirement "not X" / "other than X" / "anything but X" / "except X": the listing's own
        attribute for that property states exactly X. Code never reads the shopper's sentences for this, only the need."""
        if ident not in self.full:
            return ""
        for need, prop, value in self.plain_needs(number):
            barred = _excluded(value)
            if not barred:
                continue
            for key, got in self.own_keys(ident, prop, exact=True):
                parts = [str(x) for x in got] if isinstance(got, list) else re.split(r"\s*[,/;|]\s*", str(got))
                if any(_same_value(barred, part) for part in parts):
                    return f"its {key} is {str(got)[:40]!r}, which the requirement '{need}' excludes"
        return ""

    def hard_fault(self, ident: str, number: int | None, lines: list[str] | None = None) -> str:
        """n47: why code alone rules a listing out for item `number`: an excluded value under its own key, or a pair check
        that its partner line is not covered. Empty when code sees nothing against it (unknown never rules out)."""
        if not number:
            return ""
        hit = self.excluded_hit(ident, number)
        if hit:
            return hit
        got, other = self.pair_verdict(ident, number, lines)
        if got is False:
            return f"its compatibility words do not cover {other}, the item it must fit" if any(a == number for a, _ in self.fit_pairs) \
                else f"the accessory {other} it must pair with does not list its model"
        return ""

    def unconfirmed(self, ident: str, number: int) -> list[str]:
        return [prop for prop, _ in self.gap_needs(ident, number)]

    def stated_by(self, number: int, prop: str, value: str, skip: list[str]) -> list[str]:
        """n47 (review): opened, buyable listings of the item that state the value under the property's own key."""
        out = []
        for ident in self.full:
            if ident in skip or not self.buyable(ident) or ident in self.gone or self.item_of(ident) != number:
                continue
            if any(_phrase_in(value, ", ".join(map(str, v)) if isinstance(v, list) else str(v)) for _, v in self.own_keys(ident, prop)):
                out.append(ident)
        return out

    def gap_notes(self, ident: str, number: int, skip: list[str]) -> list[str]:
        """n47 (review): an unstated requirement is worth a note only when an opened alternative states it under its own key."""
        notes = []
        for prop, value in self.gap_needs(ident, number):
            alts = self.stated_by(number, prop, value, skip + [ident])
            if alts:
                notes.append(f"{prop} (stated under its own key by {', '.join(alts[:3])})")
        return notes

    def gap_needs(self, ident: str, number: int) -> list[tuple[str, str]]:
        """n47: requirements the listing does not state at all as code reads it: no attribute names the property with the
        value, and for a word value the title, description and other attributes do not hold its words either (a number
        counts only under a key naming its property). Limits, exclusions and closed lists are not read here. These are
        flags for the model's check, never a rejection: a synonym or another key spelling can still meet them."""
        record = self.full.get(ident)
        if not record:
            return []
        out = []
        for need, prop, value in self.plain_needs(number):
            if BOUNDED.search(value) or _NOT_PLAIN.search(value.lower()) or len(value.split()) > 6:
                continue
            keys = self.own_keys(ident, prop)
            if any(_phrase_in(value, ", ".join(map(str, v)) if isinstance(v, list) else str(v)) for _, v in keys):
                continue
            if keys:
                continue      # another value under the property's own key: the model judges synonyms and units
            if not _measure(value) and _phrase_in(value, " ".join(_flat_lines(record))):
                continue
            out.append((prop, value))
        return out

    def code_note(self, ids: list[str]) -> str:
        """n47: what code reads on the cart's lines, for the model's check: hard faults and unconfirmed requirements."""
        lines = []
        for ident in ids:
            number = self.item_of(ident)
            if not number:
                continue
            fault = self.hard_fault(ident, number, ids)
            gaps = self.gap_notes(ident, number, ids)
            if fault:
                lines.append(f"{ident}: [code: {fault}]")
            if gaps:
                lines.append(f"{ident}: [hint: not stated under its own name here: " + "; ".join(gaps[:4]) + "]")
        if not lines:
            return ""
        self.trace.append({"kind": "code_note", "lines": lines[:6]})
        return ("\n\nCODE READING (word matching only, a hint: your reading rules decide; when a line meets the requirement by meaning,"
                " such as a synonym, another key spelling or a unit, a hint about it decides nothing):\n" + "\n".join(lines[:8]))

    def code_hold(self) -> str:
        """n47: before the model's check, once per cart: a line whose own attribute states a value the shopper excludes, or
        an accessory whose compatibility words do not cover the device carted with it, holds the checkout (no step used)."""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.code_held or len(self.code_held) >= 3 or self.left() <= self.tail() + 2:
            return ""
        lines = self.cart_ids()
        faults = []
        for ident in lines:
            number = self.item_of(ident)
            why = self.hard_fault(ident, number, lines) if number else ""
            if why:
                faults.append((ident, number, why))
        if not faults:
            return ""
        self.code_held.add(mark)
        self.trace.append({"kind": "code_hold", "faults": [(i, w[:120]) for i, _, w in faults]})
        said = []
        for ident, number, why in faults:
            text = f"{ident} (item {number}): {why}."
            pair = next(((a, b) for a, b in self.fit_pairs if number in (a, b)), None)
            if pair and ("compatib" in why or "pair with" in why):
                a, b = pair
                device = next((i for i in lines if self.item_of(i) == b), None)
                options = self.pair_options(a, b, device)
                text += (" Pairs seen that fit: " + "; ".join(options) + "." if options else
                         f" No accessory seen covers it: search item {a}'s category for one whose compatibility words list the device's model code, or switch item {b} to a device that such an accessory lists.")
            else:
                others = [i for i in self.full if i not in lines and self.item_of(i) == number and self.buyable(i) and not self.hard_fault(i, number, lines)]
                others.sort(key=lambda i: (not self.code_fits(i, number), self.price_of(i) if self.price_of(i) is not None else float("inf")))
                if others:
                    text += " Read listings of that item without it: " + ", ".join(others[:4]) + "."
            said.append(text)
        return ("Not run yet (no step used). Umi's own reading of the cart rules a line out: " + " ".join(said)
                + " Replace that line with a listing that meets every requirement, then check out; never end without a purchase while such a switch is untried.")

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Umi read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Umi] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def extra_hold(self) -> str:
        """Once, before checkout: a cart with more lines than items keeps a line from an earlier choice, and such an order fails."""
        lines = self.cart_ids()
        if self.extra_held or len(lines) <= len(self.items):
            return ""
        self.extra_held = True
        self.trace.append({"kind": "extra_hold", "lines": len(lines), "items": len(self.items)})
        return (f"Not run yet (no step used). The cart holds {len(lines)} lines for {len(self.items)} requested items: {', '.join(lines)}."
                " Remove every line that is not the chosen listing of a requested item, then check out.")

    def drop_hold(self) -> str:
        """A dropped requirement widens the set of fits, so on a cheapest pick the cart's line may no longer be the cheapest.
        Held once per drop, with a smaller step margin than `wider_hold`: this is the one look that can change the answer."""
        wanted = self.drop_open and self.brief.get("pick") == "cheapest"
        self.drop_open = False
        if not wanted or self.late() or self.left() < 6 + self.tail():
            return ""
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        self.trace.append({"kind": "drop_hold", "cart": lines})
        return ("Not run yet (no step used). A requirement was dropped, so more listings fit now and the pick is the cheapest. Before checking out,"
                " list each item's category below the cart line's price, with the search's maximum price when it takes one"
                + (f" ({'; '.join(lines)})" if lines else "") + ". Read the attributes of the cheapest such rows, judged only on the requirements still in force, swap in any that fits, then check out.")

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.late() or self.brief.get("pick") != "cheapest" or self.left() < max(8, 14 - self.hunt_steps) + 2 * len(self.items):
            return ""
        self.wider_held = True
        self.trace.append({"kind": "wider_hold", "left": self.left()})
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            call = self.one_read(tool, ident)
            if not call or self.left() <= keep or self.finished or self.facts_from.get(call[0]) is False:
                break
            reply = self.act(*call)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(call[0], seen)
            self.note_view(call[0], seen)
            shown = self.records_in(seen)
            if shown:
                self.facts_from[call[0]] = self.facts_from.get(call[0], False) or any(_fact_block(r) for _, r in shown)
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
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Umi's own call and read any note in it; returns what the note changed."""
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
        # turns between the item groups, as many in all as a side-by-side call shows, two each at least; with a
        # cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        groups = self.item_groups()
        depth = 2
        for ids in groups:
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool[: max(0, min(depth, 8 - opened))])
        queue = [lane[k] for k in range(depth) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode, half of a short one
        self.fill_facts(queue[:5], min(22, max(12, self.max_steps // 2)))

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0 or self.late():
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
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        # one pick more than the messages allow: the last message goes to it when no answer needs a second try
        many = len(self.items)
        share = max(-(-count // many) + 1, 4) if LIST_QUESTIONS else -(-count // many) + 1
        rules = PLAN if LIST_QUESTIONS else PLAN.replace(SEVERAL, ONE_EACH)
        prompt = rules.replace("COUNT", str((max(count + 1, 6) if LIST_QUESTIONS else count + 1) if many == 1 else share * many)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        if many > 1:
            prompt += f"\n\nEach item gets its own messages: return up to {share} properties for EVERY item number, each item's own best first."
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 2500 if LIST_QUESTIONS else 1800) or {}
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        ordered = self.last_resort(self.colour_later(self.fill_plan(ordered, count), plan), count)
        ordered = self.spread(ordered)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "order": ordered})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        batch: list[tuple[int, str, str]] = []
        heard: dict[str, Any] = {}
        head: list[tuple[int, str]] = []
        if SINGLE_QUESTIONS:
            self.ask_singles(ordered, count, plan, sent, told)
            if not sent:
                return ""
        elif LIST_QUESTIONS:
            self.ask_lists(ordered, count, plan, sent, told)
            if not sent:
                return ""
        else:
            # all messages but the last first; the last re-asks a question whose answer wanted a single question
            # (named with example values this time), else it goes to the next pick
            head = ordered[: count - 1] if count >= 2 else ordered[:count]
            batch = self.send([(number, label, self.question(number, label)) for number, label in head])
            if not batch:
                return ""
            sent += batch
            # the light model reads the answers: it keeps a refusal of all but one value the right way round more often
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
        if not SINGLE_QUESTIONS and not LIST_QUESTIONS and count >= 2 and self.msgs_free() > 0:
            crowded =[batch[i - 1] for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(batch)]
            again = None
            # a re-ask does not take the message of an item that has had none
            starved = many > 1 and len(ordered) > len(head) and ordered[len(head)][0] not in {n for n, _, _ in batch}
            for number, text, _ in [] if starved else crowded:
                label = next((lab for n, lab in head if n == number and self.question(n, lab) == text), None)
                near = self.neighbour(number, label, {lab for n, lab in head if n == number}) if label else ""
                if near:
                    again = (number, near, self.question(number, near))
                    self.trace.append({"kind": "neighbour", "refused": label, "asked": near})
                    break
            for number, text, _ in [] if starved or again else crowded:
                label = next((lab for n, lab in head if n == number and self.question(n, lab) == text), None)
                samples = self.samples(number, label) if label else []
                if label and len(samples) >= 2:
                    again = (number, label, self.question(number, label, samples))
                    break
            if again is None and len(ordered) > len(head):
                number, label = ordered[len(head)]
                again = (number, label, self.question(number, label))
            if again:
                batch = self.send([again])
                if batch:
                    sent += batch
                    # one answer: the light model reads it
                    self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
        if told:
            self.version += 1
        walked = self.chain_walk()
        found = "\n".join(x for x in (walked, self.retrieve()) if x)
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Umi's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if found:
            summary += "\n" + found
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def list_lanes(self, ordered: list[tuple[int, str]], plan: dict[str, Any], turns: list[int]) -> dict[int, list[str]]:
        """The properties to ask about for each item, best first: the plan's picks, the keys of its category, then common
        properties of any product, colour after the rest. A category goes to the item its label matches best; one matching no
        item goes to an item with no category of its own."""
        mark = self.list_mark

        def overlap(label: str, number: int) -> int:
            item = self.items[number - 1]
            if _words(label) in (_words(item.shelf or ""), _words(item.name)):
                return 100
            return len({w for w in _stem(label) if len(w) >= 4} & {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4})
        table = self.key_table()
        owner: dict[str, int] = {}
        for label in dict.fromkeys(label for label, _, _, _ in table):
            best = max(turns, key=lambda n: overlap(label, n))
            if overlap(label, best) or len(self.items) == 1:
                owner[label] = best
        # a category matching no item is only placed when there is no doubt: one such category and one item without any
        loose = [label for label in dict.fromkeys(label for label, _, _, _ in table) if label not in owner]
        bare = [n for n in turns if n not in owner.values()]
        if len(loose) == 1 and len(bare) == 1:
            owner[loose[0]] = bare[0]
        lanes: dict[int, list[str]] = {}
        for number in turns:
            item = self.items[number - 1]
            settled = {mark(m.split(":", 1)[0]) for m in item.musts + item.told}
            keys = [label for n, label in ordered if n == number]
            keys += [name for label, name, _, _ in table if owner.get(label) == number]
            keys += list(LAST_RESORT)
            seen: set[str] = set()
            lane: list[tuple[int, str]] = []
            for key in keys:
                sign = mark(key)
                if sign not in seen and sign not in settled and not NOT_A_QUESTION.search(key):
                    seen.add(sign)
                    lane.append((number, key))
            lanes[number] = [label for _, label in self.colour_later(lane, plan)]
        return lanes

    def list_chars(self) -> int:
        """The longest list message: LIST_CHARS, or less when the message tool's text field is shorter (a longer text would be
        cut mid-word, losing its last properties and the question's end)."""
        schema = self.schemas.get(self.roles.get("ask") or "") or {}
        caps = [v.get("maxLength") for v in (schema.get("properties") or {}).values() if isinstance(v, dict) and isinstance(v.get("maxLength"), int)]
        return min([LIST_CHARS] + [c - 10 for c in caps if c > 40])

    def list_mark(self, key: str) -> str:
        return " ".join(sorted(_stem(key.lower().replace("colour", "color")))) or _words(key)

    def list_question(self, number: int, labels: list[str]) -> str:
        if len(labels) < 2:
            return self.question(number, labels[0])
        return f"For the {self.items[number - 1].name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

    def ask_lists(self, ordered: list[tuple[int, str]], count: int, plan: dict[str, Any], sent: list[tuple[int, str, str]], told: list[str]) -> None:
        """Each message names up to LIST_SIZE properties of one item. Every item gets a turn in the plan's order; a further
        message goes to the item with the most properties not asked yet. A later message takes those first: an answer of no
        preference clears nearly all a list named, while one that reveals a preference leaves the rest of its list open, so
        those come next. Properties asked only in lists are then asked alone; no text is sent twice. An answer wanting a single
        question ends the lists: each message left names one property, a refused one's neighbour or examples first."""
        turns = list(dict.fromkeys(n for n, _ in ordered)) or [1]
        turns += [n for n in range(1, len(self.items) + 1) if n not in turns]
        lanes = self.list_lanes(ordered, plan, turns)
        asked: dict[int, list[str]] = {n: [] for n in turns}
        open_again: dict[int, list[str]] = {n: [] for n in turns}
        alone: dict[int, set[str]] = {n: set() for n in turns}
        turned: set[int] = set()
        texts: set[str] = set()
        retry: tuple[int, str, str] | None = None
        flagged_before = False
        size = LIST_SIZE

        def fresh(n: int) -> list[str]:
            return [k for k in lanes[n] if k not in asked[n]]
        for _ in range(count):
            if self.msgs_free() <= 0 or self.finished or self.late():
                break
            live = [n for n in turns if fresh(n) or any(k not in alone[n] for k in lanes[n])]
            if retry:
                number, first, text = retry
                labels = [first]
                retry = None
            elif not live:
                break
            else:
                waiting = [n for n in live if n not in turned]
                number = waiting[0] if waiting else max(live, key=lambda n: (len(fresh(n)), len([k for k in open_again[n] if k in lanes[n]])))
                turned.add(number)
                again = [k for k in open_again[number] if k in lanes[number] and k not in fresh(number)]
                labels = (fresh(number) + again)[:size]
                if not labels:
                    # all asked in lists: one alone, since a list's answer may have been about another of its properties
                    labels = [next(k for k in lanes[number] if k not in alone[number])]
                text = self.list_question(number, labels)
                while len(text) > self.list_chars() and len(labels) > 1:
                    labels = labels[:-1]
                    text = self.list_question(number, labels)
                if text in texts:
                    labels = [next((k for k in lanes[number] if k not in alone[number]), labels[0])]
                    text = self.question(number, labels[0])
                    if text in texts:
                        alone[number].add(labels[0])
                        continue
            batch = self.send([(number, labels[0], text)])
            if not batch:
                break
            texts.add(text)
            sent += batch
            if len(labels) == 1:
                alone[number].add(labels[0])
            # one answer: the light model reads it
            heard = self.hear(batch, (LIGHT, HEAVY))
            flagged = any(c == 1 for c in (_count(x) for x in _items(heard.get("crowded")) if not isinstance(x, bool)))
            # the reader can take a busy shopper's line for this: its flag counts with such words, or twice running
            wanted_one = flagged and (bool(ONE_AT_A_TIME.search(batch[0][2])) or flagged_before)
            flagged_before = flagged
            # a reply of no preference to a list does not say which of its properties it means
            if len(labels) > 1:
                heard = {k: v for k, v in heard.items() if k != "free"}
            before = len(told)
            self.take_heard(heard, told)
            gone: dict[int, list[str]] = {}
            for entry in _items(heard.get("needs")):
                if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                    owner = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                    stated = self.list_mark(str(entry["need"]).split(":", 1)[0])
                    if owner in lanes:
                        gone.setdefault(owner, []).extend(k for k in lanes[owner] if self.list_mark(k) == stated)
            found = len(told) > before
            if found:
                # under another name: the listed key the reply itself names, unless it shares a word with the stated property
                if not gone.get(number):
                    said = _stem(batch[0][2])
                    stems = [_stem(str(e.get("need") or "").split(":", 1)[0]) for e in _items(heard.get("needs")) if isinstance(e, dict)]
                    gone.setdefault(number, []).extend(k for k in labels if _stem(k) and _stem(k) <= said and not any(_stem(k) & x for x in stems))
                for n, keys in gone.items():
                    lanes[n] = [k for k in lanes[n] if k not in keys]
            if wanted_one:
                self.trace.append({"kind": "list_refused", "q": text[:200], "size": len(labels)})
                if len(labels) > 1:
                    size = 1
                    alone[number].discard(labels[0])
                    continue
                waiting = [n for n in turns if n not in turned and (fresh(n) or any(k not in alone[n] for k in lanes[n]))]
                near = self.neighbour(number, labels[0], set(asked[number]) | {labels[0]})
                if near and NOT_A_QUESTION.search(near):
                    near = ""
                samples = [] if near or labels[0] not in lanes[number] else self.samples(number, labels[0])
                again_text = self.question(number, near) if near else self.question(number, labels[0], samples) if len(samples) >= 2 else ""
                # a product still waiting for its turn comes before a second try
                if again_text and again_text not in texts and not waiting:
                    retry = (number, near or labels[0], again_text)
            asked[number] += [k for k in labels if k not in asked[number]]
            if found:
                open_again[number] += [k for k in labels if k in lanes[number] and k not in open_again[number]]
            else:
                open_again[number] = [k for k in open_again[number] if k not in labels]

    def category_sizes(self) -> dict[str, int]:
        """How many opened listings each category label has (the count key_table's carried figures are out of)."""
        cat_key = self.category_key()
        sizes: dict[str, int] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if record:
                label = str(record.get(cat_key) or "listings") if cat_key else "listings"
                sizes[label] = sizes.get(label, 0) + 1
        return sizes

    def carried_first(self, lanes: dict[int, list[str]], plan: dict[str, Any]) -> dict[int, list[str]]:
        """n48: each item's first CARRIED_TOP questions hold the properties its listings carry most widely. A pick of the
        plan that the story did not hint at gives its early place to one of the item's CARRIED_TOP most carried keys when
        that key is on at least CARRIED_GAP more of the item's opened listings; the pick moves just after the early places.
        A hinted pick, or one the request asks to check, keeps its place. With no opened listing nothing moves."""
        hinted: dict[int, set[str]] = {}
        for pick in _items(plan.get("ask")):
            if isinstance(pick, dict) and str(pick.get("hint") or "").strip():
                number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
                label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
                hinted.setdefault(number, set()).add(self.list_mark(label))
        first = " ".join(str(x) for x in _items(self.brief.get("ask_first"))).lower()
        sizes = self.category_sizes()
        table = self.key_table()
        out: dict[int, list[str]] = {}
        for number, lane in lanes.items():
            item = self.items[number - 1]
            near = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            share: dict[str, float] = {}
            for label, name, carried, _ in table:
                if len(self.items) == 1 or _stem(label) & near or _words(label) in (_words(item.shelf or ""), _words(item.name)):
                    size = sizes.get(label, 0)
                    if size:
                        share[self.list_mark(name)] = max(share.get(self.list_mark(name), 0.0), carried / size)
            lane = list(lane)
            keep = hinted.get(number, set())
            marks = {self.list_mark(k): k for k in lane}
            # never promoted: a colour key unless the story or the request names colour (colour_later keeps it last), a
            # notes or remarks field, a key that is no property (codes, models, warranty)
            loud = bool(COLOUR.search(first + " " + " ".join(str(p.get("hint") or "") for p in _items(plan.get("ask")) if isinstance(p, dict))))
            fit_key = lambda k: not NOTE_FIELD.search(k) and not NOT_A_QUESTION.search(k) and (loud or not COLOUR.search(k))
            top = sorted((m for m in share if m in marks and fit_key(marks[m])), key=lambda m: -share[m])[:CARRIED_TOP]
            for mark in top:
                head = lane[:CARRIED_TOP]
                if any(self.list_mark(k) == mark for k in head):
                    continue
                weak = [i for i, k in enumerate(head) if self.list_mark(k) not in keep and not (first and _words(k) in first)
                        and share.get(self.list_mark(k), 0.0) + CARRIED_GAP <= share[mark]]
                if not weak:
                    continue
                slot = min(weak, key=lambda i: share.get(self.list_mark(lane[i]), 0.0))
                moved = lane[slot]
                key = marks[mark]
                lane.remove(key)
                lane[slot] = key
                lane.insert(min(CARRIED_TOP, len(lane)), moved)
            out[number] = lane
        return out

    def ask_singles(self, ordered: list[tuple[int, str]], count: int, plan: dict[str, Any], sent: list[tuple[int, str, str]], told: list[str]) -> None:
        """n48: one property per message, each answer read before the next question is chosen. The next question goes to
        the item with no revealed preference yet that has had the fewest messages (the plan's order breaks ties), and asks
        that item's next property: the plan's picks with the most carried keys early, then its category's keys, then
        common properties. A property an answer stated is not asked again under another of its names. An answer wanting a
        single question moves on to the next pick (in race 16, asking a word of the refused property again was refused 62%
        of the time and revealed 7%; another property or item revealed 13-17%); no message names two properties."""
        turns = list(dict.fromkeys(n for n, _ in ordered)) or [1]
        turns += [n for n in range(1, len(self.items) + 1) if n not in turns]
        lanes = self.carried_first(self.list_lanes(ordered, plan, turns), plan)
        self.trace.append({"kind": "lanes", "lanes": {n: v[:6] for n, v in lanes.items()}})
        found = {n: 0 for n in turns}
        spent = {n: 0 for n in turns}
        asked: dict[int, set[str]] = {n: set() for n in turns}
        texts: set[str] = set()
        for _ in range(count + 2):
            if self.msgs_free() <= 0 or self.finished or self.late():
                break
            live = [n for n in turns if any(self.list_mark(k) not in asked[n] for k in lanes[n])]
            if not live:
                break
            number = min(live, key=lambda n: (found[n] > 0, spent[n], turns.index(n)))
            label = next(k for k in lanes[number] if self.list_mark(k) not in asked[number])
            asked[number].add(self.list_mark(label))
            text = self.question(number, label)
            if text in texts:
                continue
            had = [len(item.told) for item in self.items]
            batch = self.send([(number, label, text)])
            if not batch:
                break
            texts.add(text)
            sent += batch
            spent[number] += 1
            # one answer: the light model reads it
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            for n, item in enumerate(self.items, 1):
                if n in found and len(item.told) > had[n - 1]:
                    found[n] += 1
            # a stated property counts as asked under each of its names
            for entry in _items(heard.get("needs")):
                if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                    owner = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                    if owner in asked:
                        asked[owner].add(self.list_mark(str(entry["need"]).split(":", 1)[0]))
            if any(_count(x) == 1 for x in _items(heard.get("crowded")) if not isinstance(x, bool)):
                self.trace.append({"kind": "single_refused", "q": text[:200]})

    def fill_plan(self, ordered: list[tuple[int, str]], count: int) -> list[tuple[int, str]]:
        """A plan shorter than the messages allow must not leave one unsent: it takes more keys from the listings read,
        most carried first, in turns over the items."""
        want = count + 1
        if len(ordered) >= want:
            return ordered
        table = self.key_table()
        lanes: list[tuple[int, list[str]]] = []
        for number, item in enumerate(self.items, 1):
            settled = " ".join(item.musts + item.told).lower()
            near = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            own = [name for label, name, _, _ in table if _stem(label) & near or len(self.items) == 1]
            lanes.append((number, [k for k in dict.fromkeys(own) if k not in settled and not NOT_A_QUESTION.search(k)]))
        out = list(ordered)
        used = {(n, _words(label)) for n, label in out}
        while len(out) < want and any(keys for _, keys in lanes):
            for number, keys in lanes:
                while keys and len(out) < want:
                    key = keys.pop(0)
                    if (number, _words(key)) not in used:
                        used.add((number, _words(key)))
                        out.append((number, key))
                        break
        return out

    def last_resort(self, ordered: list[tuple[int, str]], count: int) -> list[tuple[int, str]]:
        """Listings read so far name too few properties for the messages left: common properties of any product, in turns."""
        out = list(ordered)
        used = {(n, _words(label)) for n, label in out}
        for label in LAST_RESORT:
            for number, item in enumerate(self.items, 1):
                settled = " ".join(item.musts + item.told).lower()
                if len(out) <= count and (number, _words(label)) not in used and label not in settled:
                    used.add((number, _words(label)))
                    out.append((number, label))
        return out

    def colour_later(self, ordered: list[tuple[int, str]], plan: dict[str, Any]) -> list[tuple[int, str]]:
        """A colour question goes after the item's other picks, unless the story talks about colour or asks for it."""
        named = " ".join(str(x) for x in _items(self.brief.get("ask_first")))
        for pick in _items(plan.get("ask")):
            if isinstance(pick, dict):
                named += " " + str(pick.get("hint") or "")
        if COLOUR.search(named):
            return ordered
        later = [(n, label) for n, label in ordered if COLOUR.search(label)]
        return [(n, label) for n, label in ordered if not COLOUR.search(label)] + later

    def spread(self, ordered: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Rounds over the items: each item gets its next pick in the plan's order before any item gets a further one.
        An item the plan left out gets the keys its own listings carry most. One item: the plan as it came."""
        if len(self.items) < 2:
            return ordered
        lanes: dict[int, list[str]] = {}
        for number, label in ordered:
            lanes.setdefault(number, []).append(label)
        table = self.key_table()
        for number, item in enumerate(self.items, 1):
            if number not in lanes:
                settled = " ".join(item.musts + item.told).lower()
                near = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
                keys = [name for label, name, _, _ in table if _stem(label) & near and name not in settled]
                lanes[number] = keys[:2]
        queue: list[tuple[int, str]] = []
        while any(lanes.values()):
            for number, labels in lanes.items():
                if labels:
                    queue.append((number, labels.pop(0)))
        return queue

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def neighbour(self, number: int, label: str, asked: set[str]) -> str:
        """The key an answer wanting a single question points at. Such an answer to a question about one property means the
        words matched two properties the shopper keeps in mind, so a listed key near the refused name is very likely the
        other one: a key holding it or held by it ("lid colour" and "colour") before one sharing a longer word, then the
        most carried. A key already asked about or settled, or one that only extends or shortens such a key's name
        ("system memory new" once system memory is answered), does not count: that property is already known."""
        want = set(_words(label).split())
        item = self.items[number - 1]
        settled = " ".join(item.musts + item.told).lower()
        taken = {_words(x) for x in asked}
        known = [set(_words(x).split()) for x in asked if set(_words(x).split()) != want] + [set(_words(m.split(":", 1)[0]).split()) for m in item.musts + item.told]
        near = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        best: tuple[int, int] = (0, 0)
        found = ""
        for shelf, name, carried, _ in self.key_table():
            words = set(name.split())
            if len(self.items) > 1 and not (_stem(shelf) & near):
                continue
            if words == want or name in taken or name in settled:
                continue
            if any(k and (k <= words or words <= k) for k in known):
                continue
            rank = 2 if words < want or want < words else 1 if any(len(w) >= 5 for w in words & want) else 0
            if rank and (rank, carried) > best:
                best, found = (rank, carried), name
        return found

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
        # the shopper's replies are made on the run's own key: a message once that key is spent ends every episode of
        # the run (the store refuses all calls after it), so none is sent
        if self.no_credit:
            self.trace.append({"kind": "no_message", "why": "run key spent"})
            return sent
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
                self.hunt_queue.append((number, need))
            # n47: an answer saying this item must fit another requested item is a pair, not a pointer to walk
            fit = self.guarded(self.fit_target, None, need, number)
            if fit is not None:
                self.fit_needs.add(need)
                self.add_pair(number, fit)
                continue
            # the reader can miss a pointer: its wording marks it too
            if (entry.get("points") is True or POINTER.search(need)) and (number, need) not in self.refs:
                self.refs.append((number, need))
            if entry.get("points") is True or POINTER.search(need):
                self.pointer_needs.add(need)
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
                self.drop_open = self.drop_open or bool(hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
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
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
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
                return False, ("Before going on, Umi did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
                self.done_log.append(f"(Umi: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
        return True, ("Before going on, Umi did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
            self.drop_open = True
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
                self.hunt_queue.append((self.items.index(item) + 1, need))
            added.add(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed:
            # a new budget, need or sold-out line can make an earlier "nothing fits" wrong: the next decline is reviewed again
            self.audited = False
            self.reaudit = True
            self.refuse_blocks = 0
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
        return "lines by Umi's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def key_facts(self, ident: str) -> str:
        """The listing's attributes whose keys name a requirement in force, put in front of its record so that cutting a
        long record never hides what decides the requirements."""
        block = _fact_block(self.full.get(ident) or {}) or _fact_block(self.rows.get(ident) or {}) or {}
        if not block:
            return ""
        props = {need.split(":", 1)[0].strip() for _, need in self.needs_now()} | {m.split(":", 1)[0].strip() for it in self.items for m in it.musts}
        hits = [f"{k}={str(v)[:60]}" for k, v in block.items() if any(p and _names(_stem(p), _stem(k)) for p in props)]
        return ("[keys for the requirements: " + "; ".join(hits[:10]) + "] ") if hits else ""

    def item_number_of(self, ident: str) -> int | None:
        """The requested item a listing is for, by its category label; the only item when there is one. n49: n47's reading
        first (the item whose label or name is the store label exactly, else the one holding the label's head noun, else the
        one sharing the most long words), so "Tripods" is the tripods, not "tripods and camera supports"; the first item
        sharing a long word only when that reading is unclear."""
        if len(self.items) == 1:
            return 1
        clear = self.item_of(ident)
        if clear is not None:
            return clear
        cat_key = self.category_key()
        record = self.full.get(ident) or self.rows.get(ident) or {}
        label = str(record.get(cat_key) or "") if cat_key else ""
        if not label:
            return None
        for number, item in enumerate(self.items, 1):
            long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            if _words(label) in (_words(item.shelf or ""), _words(item.name)) or {w for w in _stem(label) if len(w) >= 4} & long:
                return number
        return None

    def own_gaps(self, ident: str, number: int) -> list[str] | None:
        """The item's required properties that no attribute key of this listing names (None when its attributes were not
        read). Such a line leans on its title or description, which the store may not count."""
        # a listing whose attribute block came back empty or with one entry is kept only as a row
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = _fact_block(record)
        if block is None:
            # an attribute block with fewer than two entries, even an empty one, still says what the listing states
            small = [v for v in record.values() if isinstance(v, dict) and all(not isinstance(x, (dict, list)) for x in v.values())]
            if len(small) != 1:
                return None
            block = small[0]
        keys = [_stem(k) for k in block]
        joined = {_words(k).replace(" ", "") for k in block}
        gaps = []
        for n, need in self.needs_now():
            prop = need.split(":", 1)[0].strip()
            named = any(_names(_stem(prop), k) for k in keys) or _words(prop).replace(" ", "") in joined
            if n == number and prop and not named and prop not in gaps:
                gaps.append(prop)
        return gaps

    def own_attribute_note(self, ids: list[str]) -> str:
        """For cart lines that lack an attribute of their own for a required property: the read, in-stock listings for the
        same item whose own attributes name every required property. Empty when there is no such line or no such listing."""
        marks, offers = [], []
        for ident in ids:
            number = self.item_number_of(ident)
            gaps = self.own_gaps(ident, number) if number else None
            if not gaps:
                continue
            others = [j for j in self.full if j not in ids and not self.sold_out(j) and self.item_number_of(j) == number and self.own_gaps(j, number) == []]
            if not others:
                continue
            others.sort(key=lambda j: self.price_of(j) if self.price_of(j) is not None else float("inf"))
            marks.append(f"{ident} (item {number}) has no attribute of its own for: " + ", ".join(gaps))
            offers.extend(f"item {number}: " + j + ": " + self.record_of(j)[:900] for j in others[:4])
        if not marks:
            return ""
        self.trace.append({"kind": "own_attributes", "lines": marks[:4]})
        return ("\n\nOWN ATTRIBUTES: " + "; ".join(marks) + ". The store may not count a value that only its title or description gives."
                " Each listing below states every required property of that item under its own attributes. If one of them meets every"
                " requirement by value, the stock rule and the budget, the cart is wrong: return ok=false and name it in \"better\" (best"
                " first). If none of them fits, this note decides nothing.\n" + "\n".join(offers[:8]))

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
        if key in self.checked or self.cost > limit or not ids or self.rushed():
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
        prompt += self.own_attribute_note(ids)
        prompt += self.guarded(self.code_note, "", ids)     # n47: code's structural reading of the lines, as flags for this check
        got = self.ask_json(prompt, [HEAVY, LIGHT], 3000)
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

    def chain_query(self, lost: str) -> str:
        """Search words for a chain's next link: the shop the pointer names with the kind of product it names ("the lamp
        from Tidal's listing", "the Desk Lamp product by Tidal", "to the Tidal Desk Lamp listing"), the requested product's
        name when no kind is named; else the description's own words."""
        need, _, about = lost.partition(" -> find ")
        item = next((self.items[n - 1].name for n, r in self.refs if r == need.strip() and 1 <= n <= len(self.items)), self.items[0].name if self.items else "")
        vague = {"one", "listing", "product", "page", "item", "same", "thing"}

        def plain(kind: str) -> bool:
            return 0 < len(kind.split()) <= 3 and not set(kind.split()) & vague
        skip = {"the", "this", "that", "its", "a", "an", "for", "use", "take", "look"}
        both = (r"\bthe ([a-z][\w &,-]*?) (?:from|by) ([A-Z][\w-]+)", r"\bthe ([A-Z][\w &-]*?) (?:product|listing|item) (?:by|from) ([A-Z][\w-]+)")
        for text in (about, need):
            found = re.search(r"\b([A-Z][\w-]+)(?:'s|\u2019s) (?:listing|page) of the ([a-z][\w &-]*)", text)
            if found and found.group(1).lower() not in skip:
                kind = re.split(r"\s-\s| (?:with|in|from|by|that|which|for)\b", found.group(2).strip())[0].strip()
                named = 0 < len(kind.split()) <= SEARCH_WORDS - 1 and not set(kind.split()) & vague
                # a plural's last s goes, unless the word needs its "es"
                return f"{found.group(1)} {re.sub(r'(?<!s)(?<!sse)(?<!xe)(?<!she)(?<!che)s$', '', kind) if named else item}".strip()
            for pattern in both:
                for found in re.finditer(pattern, text):
                    kind, shop = found.group(1).strip(" ,").lower().split(" of the ")[-1], found.group(2)
                    kind = re.sub(r"\s+(?:sold|made|offered|listed|sell)$", "", kind)
                    if shop.lower() not in skip:
                        return f"{shop} {kind if plain(kind) else item}".strip()
            found = re.search(r"\bto the ([A-Z][\w-]+) ([A-Za-z][\w &-]*?) (?:listing|product|item)\b", text)
            if found and found.group(1).lower() not in skip:
                return f"{found.group(1)} {found.group(2).lower() if plain(found.group(2).lower()) else item}".strip()
            found = re.search(r"\b(?:to|at|from) the ([A-Z][\w-]+) ([a-z][\w&-]*(?: [a-z][\w&-]*){0,2})", text)
            if found and found.group(1).lower() not in skip:
                kind = re.split(r" (?:with|in|from|by|and|that|which|for)\b", found.group(2))[0].strip()
                return f"{found.group(1)} {kind if plain(kind) else item}".strip()
            for pattern in (r"\b(?:by|from)\s+([A-Z][\w-]+)", r"\b([A-Z][\w-]+)(?:'s|\u2019s)\b"):
                for found in re.finditer(pattern, text):
                    if found.group(1).lower() not in skip and found.group(1).lower() not in item.lower():
                        return f"{found.group(1)} {item}".strip()
        return self.chain_words(about)

    def chain_detailed(self, lost: str) -> str:
        """The shop-and-kind query with the details a pointer gives to tell that shop's listings apart: the values in brackets
        after the shop ("by Tidal (brass, dimmable)") or listed after "with" ("Tidal's listing, with shade colour teal and
        base brass"), each with its property's name dropped. Empty when no details are given."""
        base = self.chain_query(lost)
        if not base or len(base.split()) < 2:
            return ""
        shop = base.split()[0]
        need, _, about = lost.partition(" -> find ")
        names = sorted({name for _, name, _, _ in self.key_table()} | {_words(m.split(":")[0]) for _, m in self.needs_now()}, key=len, reverse=True)
        values: list[str] = []
        for text in (about, need):
            if values:
                break
            found = re.search(re.escape(shop) + r"[^()]{0,60}\(([^)]{3,120})\)", text)
            parts = re.split(r",|\band\b", found.group(1)) if found else []
            if not parts:
                found = re.search(re.escape(shop) + r"(?:'s|\u2019s)?[^.;()]{0,70}?\b(?:with|that has|which has|having) ([^.;()]{3,160})", text)
                # what follows a dash is no longer about that listing
                parts = re.split(r",|\band\b", re.split(r"\s[\u2013\u2014-]\s", found.group(1))[0]) if found else []
            for part in parts:
                words = re.sub(r"^(?:a|an|the)\s+|\s+(?:instead|only|too)$", "", _words(part)).strip()
                for name in names:
                    if name and words.startswith(name + " "):
                        words = words[len(name):].strip()
                        break
                # a property's name no key has shown stays in front of its value: the value is at the end
                words = " ".join(words.split()[-2:]) if len(words.split()) > 3 else words
                if words and words not in values:
                    values.append(words)
        if not values:
            return ""
        # a search reads only its first words: the details pick the listing out, so the kind's words give way first
        self.detail_words[lost] = _tokens(" ".join(values)) - _tokens(base)
        detail = " ".join(values[:3]).split()
        kind = base.split()[1:]
        room = max(0, SEARCH_WORDS - 1 - len(detail))
        return " ".join(([shop] + (kind[-room:] if room else []) + detail)[:SEARCH_WORDS])

    def chain_words(self, about: str) -> str:
        """Search words for a link: the shop and the details the page gives, without the small words."""
        small = {"the", "a", "an", "of", "from", "with", "and", "by", "that", "this", "is", "are", "for", "to", "listing", "product", "item", "sold", "find", "page",
                 "requirement", "requirements", "points", "point", "pointed", "refers", "shop", "store", "seller", "one", "look", "its", "on", "at", "in",
                 "asks", "ask", "describes", "described", "description", "previous", "record", "records", "row", "rows", "seen", "only", "opened", "sends", "reader", "notes", "note", "says", "it"}
        words = [re.sub(r"[\u2019']s$", "", w) for w in re.findall(r"[^\W_][^\s,;:()]*", about)]
        words = [w for w in words if w and w.lower() not in small]
        return " ".join(words[:SEARCH_WORDS])

    def chain_walk(self, keep: int = CHAIN_KEEP) -> str:
        """Right after a requirement points at another listing: walk its pages now, while steps remain, rather than at the
        checkout. Each round reads the trail once, opens what it can name (the page view, which holds the notes) and
        searches for the next link it cannot. It stops on steps, spend, time or a round that finds nothing."""
        live = {need for _, need in self.needs_now()}
        if not [need for _, need in self.refs if need in live and need not in self.ref_values] or not self.roles.get("search"):
            return ""
        start = self.step
        said: list[str] = []
        tried = self.walk_tried      # kept between walks: a search that found nothing new is not run again
        unread: set[str] = set()
        facts_read = 0
        stop = "rounds"
        for _ in range(CHAIN_ROUNDS):
            room = CHAIN_STEPS - (self.step - start)
            cap = SPEND_SOFT if keep >= CHAIN_KEEP else self.spend_cap() - 0.01
            if room < 2 or self.left() < keep + self.tail() or self.cost >= cap or self.late() or self.finished:
                stop = "room" if room < 2 else "steps" if self.left() < keep + self.tail() else "spend" if self.cost >= cap else "time"
                break
            before = self.step
            self.trails = 3            # one reading of the trail each round
            text, _ = self.trail(True)
            if text:
                said.append(text.strip())
            if not any(need in live and need not in self.ref_values for _, need in self.refs):
                break
            # per lost link: the shop, kind and the details that set its listing apart; then shop and kind; then the description's words
            options = [w for x in self.trail_lost for w in (self.chain_detailed(x), self.chain_query(x), self.chain_words(x.split(" -> find ", 1)[-1])) if len(w.split()) >= 2]
            words = next((w for w in options if w and w not in tried), "")
            detailed = {self.chain_detailed(x) for x in self.trail_lost} - {""}
            # the details a pointer adds to its shop and kind, for its detailed search and for the plain one that follows it
            # when a store's search needs every word and no row shows them
            marks: frozenset[str] = frozenset()
            for x in self.trail_lost:
                aimed, plain = self.chain_detailed(x), self.chain_query(x)
                if aimed and words in (aimed, plain):
                    marks = self.detail_words.get(x) or (_tokens(aimed) - _tokens(plain))
                    break
            if words:
                pages = min(3, room - 1, self.left() - keep - self.tail())
                if pages < 1:
                    stop = "steps"
                    break
                tried.add(words)
                # every page of a narrow result: a shop's listing named by its details is often past the first page
                found, _, taken, _ = self.hunt_search(words, pages)
                if not taken or self.finished:
                    break
                # a search narrowed by the pointer's details to one or two of the named shop's listings: open their pages now
                shop = words.split()[0].lower()
                mine = [i for i in found if any(isinstance(v, str) and v.strip().lower() == shop for v in (self.rows.get(i) or {}).values())]
                viewer = self.notes_tool()
                aimed_at = words in detailed or bool(marks)
                if aimed_at and 1 <= len(mine) <= 2 and viewer:
                    for ident in mine:
                        call = self.one_read(viewer, ident)
                        if ident in self.noted or not call or self.left() < keep + self.tail() + 1:
                            continue
                        reply = self.act(*call)
                        if reply is None or self.finished:
                            break
                        self.observe(call[0], reply.get("observation"))
                        said.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
                        said += self.end_facts(ident, reply.get("observation"))
                        self.trace.append({"kind": "chain_open", "id": ident, "query": words})
                elif aimed_at and len(mine) >= 3:
                    # several of the shop's listings: their attributes carry the details that pick out the link, and a row often
                    # shows none of them, so they are read (those whose row shows most of the details first, else in the
                    # search's order) until one states the details; the next reading of the trail then names it
                    details = _tokens(" ".join(words.split()[1:])) | marks
                    # the words can match by chance: each such listing stops one reading, and the trail then judges it
                    named_now = [i for i in mine if marks and i in self.full and i not in self.mark_hits and marks <= self.stated(self.full[i])]
                    self.mark_hits.update(named_now)
                    known = bool(named_now)
                    ranked = [] if known else sorted((i for i in mine if i not in self.full and i not in self.swept), key=lambda i: -len(details & _tokens(self.rows.get(i) or {})))
                    tool = self.facts_tool()
                    for ident in ranked[: max(0, (CHAIN_SWEEP if marks else 3) - facts_read)]:
                        call = self.one_read(tool, ident) if tool else None
                        # two steps stay beyond the walk's own: the page of the listing found is still to be opened
                        if not call or self.left() < keep + self.tail() + 2 or self.step - start >= CHAIN_STEPS - 1 or self.late():
                            break
                        reply = self.act(*call)
                        if reply is None or self.finished:
                            break
                        self.observe(call[0], reply.get("observation"))
                        self.swept.add(ident)
                        said.append(f"{ident}: " + self.show(reply.get("observation"))[:500])
                        facts_read += 1
                        self.trace.append({"kind": "chain_facts", "id": ident, "query": words})
                        if marks and marks <= self.stated(reply.get("observation")):
                            self.mark_hits.add(ident)
                            break
                    if any(i not in self.full and i not in self.swept for i in mine):
                        unread.add(words)
            if self.step == before:
                break
        self.trails = 0
        tried -= unread              # a later walk may search again for the shop's rows still unread
        self.trace.append({"kind": "chain_walk", "steps": self.step - start, "values": len(self.ref_values), "stop": stop, "keep": keep})
        return ("[Umi walked the pages a requirement points at, one step a call]\n" + "\n".join(said)) if said else ""

    def stated(self, observation: Any) -> frozenset[str]:
        """The words a listing's record states about itself: its attributes and its shop's notes, not its price or figures."""
        record = _peel(observation)
        if not isinstance(record, dict):
            return frozenset()
        notes = " ".join(str(v) for k, v in record.items() if NOTE_FIELD.search(str(k)) and isinstance(v, str))
        blocks = [v for v in record.values() if isinstance(v, dict)]
        return _tokens(blocks) | _tokens(notes)

    def end_facts(self, ident: str, observation: Any) -> list[str]:
        """A chain page that shows no shop notes is the chain's end: the property's value is on that listing itself, so its
        attributes are read too (one step) when the page view does not show them and they have not been read yet."""
        def noted(node: Any, depth: int = 0) -> bool:
            if depth > 5:
                return False
            if isinstance(node, dict):
                return any((NOTE_FIELD.search(str(k)) and v) or noted(v, depth + 1) for k, v in node.items())
            if isinstance(node, list):
                return any(noted(v, depth + 1) for v in node)
            return False
        tool = self.facts_tool()
        if noted(observation) or ident in self.full or not tool or tool == self.notes_tool() or self.left() < self.tail() + 3 or self.finished:
            return []
        call = self.one_read(tool, ident)
        reply = self.act(*call) if call else None
        if reply is None:
            return []
        self.observe(call[0], reply.get("observation"))
        self.trace.append({"kind": "chain_end_facts", "id": ident})
        return [f"{ident} (its attributes, the end of the chain): " + self.show(reply.get("observation"))[:1500]]

    def notes_tool(self) -> str | None:
        """The one-listing view that shows a shop's notes, where a page names the next listing of a chain: one whose results
        have held a notes field, else one whose description promises notes or a page, else the page view by role."""
        jobs = {self.roles.get(r) for r in ("search", "cart", "cart_view", "buy", "refuse", "ask", "compare")} - {None}
        singles = [t["function"]["name"] for t in self.tools if t["function"]["name"] not in jobs and self.single_target(t["function"]["name"])]
        shown = [t for t in singles if any(NOTE_FIELD.search(str(k)) for k in self.tool_keys.get(t, ()))]
        if shown:
            return next((t for t in (self.roles.get("offer"), self.roles.get("open")) if t in shown), shown[0])
        facts = self.facts_tool()
        said = [t["function"]["name"] for t in self.tools if t["function"]["name"] in singles
                and (NOTE_DESC.search(str(t["function"].get("description") or "")) or re.search(r"page|note|remark", t["function"]["name"], re.I))]
        # a chain's link counts only when its page is opened: the attributes view is not that page while another view exists
        pages = [t for t in said if t != facts] or said
        if pages:
            return next((t for t in (self.roles.get("offer"), self.roles.get("open")) if t in pages), pages[0])
        others = [t for t in singles if t != facts]
        if len(others) == 1:
            return others[0]
        return self.roles.get("offer") or self.roles.get("open")

    def trail(self, walking: bool = False) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Umi opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the view that shows the shop's notes
        tool = self.notes_tool()
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < self.spend_cap() and not self.late():
            self.trails += 1
            records = []
            # a shop a pointer names: its rows are shown more fully, so the pointer's details can tell its listings apart
            named = {w.lower() for need in pending for w in re.findall(r"\b[A-Z][\w-]{2,}", need)}
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened or ident in self.noted else None
                if record and ident not in self.noted:
                    records.append("ROW (its attributes were read, not its page with the shop's notes) " + _dump(_clip(record, 120))[:600])
                elif record:
                    records.append("OPENED " + self.key_facts(ident) + _dump(_clip(record, 300))[:1400])
                else:
                    row = self.rows.get(ident) or {}
                    shop = bool(named & {str(v).strip().lower() for v in row.values() if isinstance(v, str) and len(v) < 40})
                    records.append("ROW " + (_dump(_clip(row, 200))[:900] if shop else _dump(_clip(row, 80))[:300]))
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
                self.chain_links.update(i for i in ids if i in self.rows)
                if chain.get("done") is True and links and all(i in self.noted for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    self.hunt_queue += [(n, need) for n, r in self.refs if r == need]
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.noted]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not self.one_read(tool, reach[0]) or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                call = self.one_read(tool, ident)
                if not call:
                    break
                reply = self.act(*call)
                if reply is None or self.finished:
                    break
                self.observe(call[0], reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
                seen += self.end_facts(ident, reply.get("observation"))
        text = ""
        if seen:
            text += "[Umi opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        self.trail_lost = lost
        hold = bool(lost) and not walking and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + f". Search for it by its shop, brand or words, open its page with {self.notes_tool()} (the view with the shop's notes), apply the value its page gives, then check out.")
        return text, hold

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            call = self.one_read(tool, ident)
            if not call:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(*call)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.observe(call[0], observation)
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
        text = f"[Umi] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        # with steps to spare, a replacement is one whose attributes were read: rows seen only in a search are opened first,
        # and a thin set of read candidates gets one fresh search before the swap
        read = [i for i in picks if i in self.full]
        unread = [i for i in picks if i not in self.full]
        if self.left() > 10 and len(read) < 2 and not self.swap_searched:
            self.swap_searched = True
            self.trace.append({"kind": "swap_verify", "read": read, "unread": unread, "left": self.left()})
            todo = []
            if unread:
                todo.append(f"open the attributes of {', '.join(unread[:2])} (seen only as search rows, so not yet known to fit)")
            todo.append("run one fresh search for the item with its required values and open the attributes of the candidates it shows")
            known = f" Read so far and appearing to fit: {', '.join(read)}." if read else ""
            return text + known + " Before replacing the line: " + "; ".join(todo) + ". Add a listing only after its attributes show it meets every requirement in force, remove the sold-out line, then check out."
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
            records.append(self.key_facts(ident) + _dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 2500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        for ident in lost:
            self.swap_picks[ident] = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        return ranked[:4]

    def audit(self) -> str:
        # after a change the review may run up to the hard cap (the episode ceiling still stops it): it turns a refused checkout into an order
        if self.audited or self.cost > (max(SPEND_HARD, self.spend_cap()) if self.reaudit else SPEND_SOFT):
            return ""
        self.audited = True
        if self.reaudit:
            self.trace.append({"kind": "reaudit", "left": self.left(), "budget": self.budget})
            self.reaudit = False
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

    def audit_order(self) -> list[str]:
        """The listings an audit reads, up to AUDIT_RECORDS. Newest first alone would lose an order's earlier items: each item's
        listings get an even share, opened pages first, and what is left goes to the rest, newest first."""
        live = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        groups = [[i for i in live if i in set(ids)] for ids in self.item_groups()]
        share = AUDIT_RECORDS // max(1, len(groups))
        picked: list[str] = []
        for ids in groups:
            picked += sorted(ids, key=lambda i: i not in self.full)[:share]
        # the pages of a chain a requirement points at are often of another kind of product: they come first, whatever their kind
        chain = [i for i in live if i in self.noted][:8] if self.refs else []
        return list(dict.fromkeys(chain + picked + live))[:AUDIT_RECORDS]

    def audit_rows(self, picked: list[str]) -> list[str]:
        """n47: rows of a requested kind left out of the newest records: seen in a search, never opened, buyable and inside
        the budget, oldest first, items with no fit named yet first, at most AUDIT_ROWS. A fitting listing seen early must
        not be lost to the decline review's window."""
        named = set(self.lead_numbers(self.lead_items))
        groups: list[tuple[bool, list[str]]] = []
        for number, item in enumerate(self.items, 1):
            ids = []
            for ident in self.seen_order:
                if ident in picked or ident in self.full or ident in self.noted or not self.buyable(ident) or ident in self.gone:
                    continue
                if (self.item_of(ident) != number) if len(self.items) > 1 else not self.of_kind(ident, item):
                    continue
                price = self.price_of(ident)
                if isinstance(self.budget, (int, float)) and (price is None or price * item.qty > self.budget):
                    continue
                ids.append(ident)
            groups.append((number in named, ids))
        groups.sort(key=lambda g: g[0])
        share = max(1, AUDIT_ROWS // max(1, len(groups)))
        out = [i for _, ids in groups for i in ids[:share]]
        return list(dict.fromkeys(out))[:AUDIT_ROWS]

    def row_line(self, ident: str) -> str:
        """n47: a row in short form: id | title (cut) | price | stock."""
        record = self.rows.get(ident) or {}
        keys = [k for k, v in record.items() if isinstance(v, str) and re.search(r"name|title", str(k), re.I) and not re.search(r"shop|brand|seller|maker|store|vendor", str(k), re.I)]
        title = str(record[keys[0]]) if keys else ""
        price = self.price_of(ident)
        return f"ROW {ident} | {title[:60]} | {f'{price:g}' if price is not None else '?'} | {'in stock' if self.buyable(ident) else 'sold out'}"

    def lead_numbers(self, fits: Any) -> dict[int, list[str]]:
        """n47: the review's fits by item number. A key may be the number ("1", 1, "item 1") or the item's name, read on whole
        words (case, hyphens and underscores, a plural "s" aside): the same name (also with its spaces closed up), else the
        longest item name the key holds (two names held side by side name no item), else the one item whose name holds the
        key, else the one item sharing the most words; a tie or no match names no item. Leads not of that item's kind drop."""
        out: dict[int, list[str]] = {}
        if not isinstance(fits, dict):
            return out

        def norm(text: str) -> list[str]:
            return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower().replace("-", " ").replace("_", " "))]

        def spans(key: list[str], name: list[str]) -> list[tuple[int, int]]:
            return [(k, k + len(name)) for k in range(len(key) - len(name) + 1) if name and key[k:k + len(name)] == name]

        def one(hits: list[int]) -> int | None:
            hits = list(dict.fromkeys(hits))
            return hits[0] if len(hits) == 1 else None
        forms = {n: [f for f in (norm(it.name), norm(it.shelf or "")) if f] for n, it in enumerate(self.items, 1)}
        for key, ids in fits.items():
            number = None
            text = str(key).strip().lower()
            found = re.fullmatch(r"(?:item\s*)?#?\s*(\d+)", text)
            words = norm(text)
            if found:
                number = int(found.group(1))
            elif words:
                number = one([n for n, fs in forms.items() if any(f == words or "".join(f) == "".join(words) for f in fs)])
                if number is None:
                    held = [(b - a, a, b, n) for n, fs in forms.items() for f in fs for a, b in spans(words, f)]
                    if held:
                        size, a, b, best = max(held)
                        apart = [n for _, x, y, n in held if n != best and (y <= a or x >= b)]
                        longest = {n for s_, _, _, n in held if s_ == size}
                        number = best if not apart and len(longest) == 1 else None
                    else:
                        number = one([n for n, fs in forms.items() if any(spans(f, words) for f in fs)])
                        if number is None:
                            scores = sorted(((len(set(words) & {w for f in fs for w in f if len(w) >= 3}), n) for n, fs in forms.items()), reverse=True)
                            if scores and scores[0][0] and (len(scores) == 1 or scores[0][0] > scores[1][0]):
                                number = scores[0][1]
            if number and 1 <= number <= len(self.items):
                have = out.setdefault(number, [])
                have += [str(i) for i in _items(ids) if str(i) not in have and self.of_kind(str(i), self.items[number - 1])]
        return {n: v for n, v in out.items() if v}

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        picked = self.audit_order()
        for ident in picked:
            # a page read with the notes view counts as opened even when it carries no fact sheet
            record = self.full.get(ident) if ident in self.full else (self.rows.get(ident) if ident in self.noted else None)
            records.append(("OPENED " + self.key_facts(ident) + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
        if not records:
            return None
        # n47: earlier rows of a requested kind, beyond the newest records, in short form, on the first audit only (cost)
        earlier = [] if self.rows_audited else [self.row_line(i) for i in self.guarded(self.audit_rows, [], picked)]
        self.rows_audited = True
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if self.notes:
            # the request keeps its first figures: say which side wins, or the strictest one gets used
            prompt += ("\nThe shopper has changed the task since writing the request (" + "; ".join(self.notes[-3:])
                       + "). Where the requirements in force differ from the request, such as a new budget, judge by the requirements in force, not by the request.")
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        if earlier:
            prompt += "\n\nEARLIER ROWS OF A REQUESTED KIND, IN STOCK AND INSIDE THE BUDGET, NOT OPENED YET (oldest first):\n" + "\n".join(earlier)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 3000)
        self.trace.append({"kind": "audit", "got": got, "earlier": len(earlier)})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else one by one (four while
        steps are plentiful, else two)."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[: 4 if self.left() >= 12 else 2]]
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
        """The agent ends without buying although the independent check confirmed a reviewed listing: Umi carts the reviewed
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
        return ("Not run (Umi acted instead): the independent check confirmed " + self.confirmed + ", so Umi set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- Umi's own search after a reveal ----------
    def wanted(self, need: str) -> str:
        """The words a listing would show for a need; empty for a limit, an exclusion or a value still on another page."""
        stated = need.split(":", 1)[-1].strip()
        if BOUNDED.search(stated) or (any(n == need for _, n in self.refs) and need not in self.ref_values):
            return ""
        if need in self.fit_needs:
            return ""      # n47: a fit between requested items has no value words a search could use
        return re.split(r"\s+or\s+|[,;/(]", self.ref_values.get(need, stated))[0].strip()

    def ladder(self, item: Item, fresh: list[str]) -> list[str]:
        """Keyword queries, narrowest first: the new values, the stated ones and the noun inside the words a search reads;
        the newest value and the noun; that value alone. A query matches on all its words, so the first may find nothing."""
        noun = [w for w in re.findall(r"[^\W_]+", item.name.lower()) if w not in ("and", "of", "the", "for")][-2:]
        room = SEARCH_WORDS - len(noun)
        stated = [self.wanted(m) for m in item.told + item.musts if m not in item.dropped]
        words: list[str] = []
        for value in fresh + [v for v in stated if v and v not in fresh]:
            part = re.findall(r"[^\W_]+", value.lower())
            if words and len(words) + len(part) > room:
                break
            words += [w for w in part[:room] if w not in words]
        first = re.findall(r"[^\W_]+", fresh[0].lower())
        queries = [" ".join(q) for q in (words + noun, first[:room] + noun, first[:SEARCH_WORDS])]
        return [q for q in dict.fromkeys(queries) if q and q not in self.hunted]

    def word_call(self, words: str) -> dict[str, Any] | None:
        """A keyword search on this store: the parameter and mode the brief named, else what the schema and the agent's
        own searches show. None when that cannot be said."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        spec = self.brief.get("search_args") if isinstance(self.brief.get("search_args"), dict) else {}
        key = spec.get("text") if spec.get("text") in texts else None
        key = (key or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
               or next((k for k in texts if re.search(r"quer|keyword|words|search|free.?text", f"{k} {props[k].get('description') or ''}", re.I)), None)
               or next((k for c in reversed(self.typed) for k in c if k in texts and isinstance(c[k], str) and c[k].strip()), None))
        if not key:
            return None
        fixed = {k: v for k, v in spec.get("fixed").items() if k in props and k != key and v in (props[k].get("enum") or [v])} if isinstance(spec.get("fixed"), dict) else {}
        used = next((c for c in reversed(self.typed) if key in c), {})
        for name, entry in props.items():
            if name in fixed or name == key:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("enum"), list) and entry["enum"]:
                worded = next((v for v in entry["enum"] if re.search(r"text|keyword|word|phrase|free", str(v), re.I)), None)
                if worded is not None:
                    fixed[name] = worded
                elif name in used:
                    fixed[name] = used[name]
                elif name in (schema.get("required") or []):
                    fixed[name] = entry["enum"][0]
            elif name in (schema.get("required") or []):
                return None
        faults: list[str] = []
        args = _shape({**fixed, key: words}, schema, "args", faults)
        return None if faults else args

    def of_kind(self, ident: str, item: Item) -> bool:
        key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(key) or "") if key else ""
        near = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        return not label or not near or bool({w for w in _stem(label) if len(w) >= 4} & near)

    def hunt_search(self, words: str, room: int) -> tuple[list[str], int, int, bool]:
        """One keyword search, then the rest of its pages when the total says it is narrow enough, leaving `room` as the
        limit on steps. Returns the listing ids, the total, the steps taken and whether every page was read."""
        tool = self.roles.get("search") or ""
        call = self.word_call(words)
        self.hunted.add(words)
        self.trace.append({"kind": "word_call", "words": words, "built": bool(call), "args": sorted(call) if call else []})
        reply = self.act(tool, call) if call else None
        if reply is None or self.finished:
            return [], 0, 1 if call else 0, False
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.read_cache[tool + _dump(call)] = (self.version, self.step, self.show(seen))
        found = [i for i, _ in self.records_in(seen)]
        if found:
            self.last_found = seen
        plan = self.paging(tool, seen) if found else None
        if not plan:
            return found, len(found), 1, len(found) < 2
        count, shown, param, value = plan
        taken = 1
        pages = -(-(count - shown) // shown)
        while value is not None and pages <= HUNT_PAGES and taken < room and not self.finished:
            faults: list[str] = []
            args = _shape({**call, param: value}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            taken += 1
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))
            found += [i for i, _ in self.records_in(seen) if i not in found]
            after = self.paging(tool, seen)
            value = after[3] if after else None
        return found, count, taken, value is None

    def hunt_pool(self, number: int, found: list[str]) -> list[str]:
        """Listings to open for an item: rows showing more of its values first, among those just found and those seen
        before and never opened; buyable, inside the budget, of the item's kind when any such row exists."""
        item = self.items[number - 1]
        values = [_tokens(v) for m, n in self.needs_now() if m == number for v in [self.wanted(n)] if v]
        hits = {i: sum(1 for v in values if v <= _tokens(self.rows.get(i) or {})) for i in dict.fromkeys(found + self.seen_order)}
        pool = [i for i, n in hits.items() if i not in self.full and self.buyable(i) and (n or i in found)]
        if self.budget is not None:
            pool = [i for i in pool if (self.price_of(i) or 0.0) * item.qty <= self.budget]
        kept = [i for i in pool if self.of_kind(i, item)] or pool
        cheap = self.brief.get("pick") == "cheapest"
        return sorted(kept, key=lambda i: (-hits[i], (self.price_of(i) or 0.0) if cheap else 0.0, i not in found))

    def read_rows(self, ids: list[str], keep: int) -> None:
        """Read these listings' attributes: side by side when that view shows them, the rest one at a time."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        if many and joiner and not self.compare_bare():
            room = max(1, self.compare_room())
            for start in range(0, len(ids), room):
                faults: list[str] = []
                args = _shape({many: ids[start:start + room]}, self.schemas.get(joiner), "args", faults)
                if faults or self.left() <= keep or self.finished:
                    break
                reply = self.act(joiner, args)
                if reply is None:
                    break
                seen = reply.get("observation")
                self.observe(joiner, seen)
                self.note_view(joiner, seen)
                shown = self.records_in(seen)
                if shown:
                    self.facts_from[joiner] = self.facts_from.get(joiner, False) or any(_fact_block(r) for _, r in shown)
                if not shown or self.facts_from.get(joiner) is False:
                    break
        self.fill_facts([i for i in ids if i not in self.full], keep)

    def hunt_item(self, number: int, fresh: list[str], room: int) -> list[str]:
        """Search, page and open for one item within `room` steps; lines saying what came of it."""
        item = self.items[number - 1]
        found: list[str] = []
        said: list[str] = []
        tries = self.ladder(item, fresh)
        if room < 3 and len(tries) > 1:
            self.trace.append({"kind": "short_room", "room": room, "skipped": tries[0]})
        for words in (tries[1:] if room < 3 and len(tries) > 1 else tries)[:HUNT_TRIES]:
            # room for at least one listing to be opened afterwards
            if room < 2 or self.finished:
                break
            rows, total, taken, whole = self.hunt_search(words, room - 1)
            room -= taken
            if taken:
                said.append(f"{words!r}: {total} listings" + ("" if whole else ", only part of them read"))
            found += [i for i in rows if i not in found]
            if any(self.of_kind(i, item) for i in rows):
                break
        before = set(self.full)
        picks = self.hunt_pool(number, found)[: min(HUNT_OPENS, max(0, room))]
        self.read_rows(picks, self.hunt_keep())
        values = [self.wanted(n) for m, n in self.needs_now() if m == number]
        values = [v for v in values if v]
        cards = []
        for ident in [i for i in picks if i in self.full and i not in before]:
            have = _tokens(self.full[ident])
            shown = [v for v in values if _tokens(v) <= have]
            record = self.full[ident]
            first = sorted(record, key=lambda k: (not isinstance(record[k], dict), len(str(record[k]))))
            cards.append((-len(shown), f"{ident} (states {len(shown)} of {len(values)} values): " + _dump(_clip({k: record[k] for k in first}, 160))[:800]))
        lines = [f"item {number} ({item.name}), searched " + "; ".join(said)] if said else []
        lines += [card for _, card in sorted(cards, key=lambda c: c[0])]
        rest = [i for i in self.hunt_pool(number, found) if i not in picks][:6]
        if rest and lines:
            priced = [f"{i} ({self.price_of(i):g})" if self.price_of(i) is not None else i for i in rest]
            lines.append("rows left unopened: " + ", ".join(priced))
        return lines

    def hunt_keep(self) -> int:
        """Steps that stay for carting every item and finishing."""
        return self.tail() + 2 * len(self.items) + 4

    def retrieve(self) -> str:
        """After a reveal (or a chain's value): Umi searches for the new values itself, reads a narrow result to its end
        and opens the rows that show them, so the agent chooses from pages instead of going by a reminder to search.
        It costs steps and no model turn, and does less, or nothing, when steps, spend or time are short."""
        fresh, self.hunt_queue = self.hunt_queue, []
        live = self.needs_now()
        fresh = [(n, need) for n, need in fresh if (n, need) in live and self.wanted(need)]
        room = min(HUNT_STEPS, HUNT_TOTAL - self.hunt_steps, self.left() - self.hunt_keep())
        if self.max_steps < HUNT_MIN_STEPS:
            if fresh:
                self.trace.append({"kind": "hunt_skipped", "limit": self.max_steps, "needs": len(fresh)})
            return ""
        if not fresh or room < 2 or not self.roles.get("search") or self.hunts >= HUNT_ROUNDS or self.cost >= HUNT_SPEND or self.late():
            return ""
        self.hunts += 1
        start = self.step
        numbers = list(dict.fromkeys(n for n, _ in fresh))
        lines: list[str] = []
        for done, number in enumerate(numbers):
            spent = self.step - start
            share = -(-(room - spent) // (len(numbers) - done))
            # the newest value leads its query
            lines += self.hunt_item(number, [self.wanted(need) for n, need in reversed(fresh) if n == number], share)
            if self.finished or self.late():
                break
        self.hunt_steps += self.step - start
        self.trace.append({"kind": "hunt", "steps": self.step - start, "items": numbers, "opened": len(lines)})
        if not lines:
            return ""
        return "[Umi searched for the new values itself, one step a call; \"states\" counts the requirement values found in a record, the fit is yours to judge]\n" + "\n".join(lines)

    def basket_gap(self) -> str:
        """After a budget change on an order of several items: whether the cart still fits, and which cheaper rows of a
        line's kind would bring it back inside (their attributes read, so the agent can judge them). No model reading."""
        if len(self.items) < 2 or self.budget is None or self.budget == self.budget_first or self.brief.get("budget_scope") == "each":
            return ""
        lines = [i for i in self.cart_ids() if not self.sold_out(i)]
        mark = (self.budget, self.cart_mark())
        if not lines or mark in self.basket_marks:
            return ""
        self.basket_marks.add(mark)
        if not self.price_key and self.last_found is not None:
            self.read_shape(self.roles.get("search") or "", self.last_found)
        prices = {i: self.price_of(i) for i in lines}
        if None in prices.values():
            return ""
        units = {i: self.cart_lines.get(i, 1) for i in lines}
        total = sum(prices[i] * units[i] for i in lines)
        self.trace.append({"kind": "basket_gap", "budget": self.budget, "total": total, "lines": len(lines)})
        if total <= self.budget:
            if len(lines) >= len(self.items):
                return ""
            return f"[Umi] The budget is now {self.budget} and the cart holds {total:.10g}: {self.budget - total:.10g} is left for the items not carted yet."
        cat_key = self.category_key()
        cart = self.cart_ids()
        swaps: list[tuple[str, str]] = []
        for line in sorted(lines, key=lambda i: -prices[i] * units[i]):
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
            pool = [i for i in self.seen_order if kind and i not in cart and self.buyable(i) and self.price_of(i) is not None
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and total - (prices[line] - self.price_of(i)) * units[line] <= self.budget]
            pool.sort(key=lambda i: (i not in self.full, self.price_of(i)))
            swaps += [(line, i) for i in pool[:2]]
        if not self.late():
            self.fill_facts([i for _, i in swaps][:CHEAPER_OPENS], self.tail() + 4)
        text = f"[Umi] The budget is now {self.budget} and the cart totals {total:.10g}, over by {total - self.budget:.10g}."
        if swaps:
            return text + " Swaps that bring it inside, if the listing meets that line's requirements: " + "; ".join(f"{b} ({self.price_of(b):.10g}) for {a} ({prices[a]:.10g})" for a, b in swaps[:4]) + "."
        return text + " No single swap among the listings seen does it: search each line's kind under a lower maximum price."

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
        elif (left <= len(self.items) + 5 or self.late()) and self.buys == 0:
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
                out.append((cid, "Shown in the side-by-side result above: Umi opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Umi's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Umi] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            back = self.guarded(self.adds_gone, [], name, args)
            if back:
                out.append((cid, f"Not run (no step used): {back[0]} sold out at step {self.gone[back[0]]} and will not come back. Cart another listing that meets every requirement."))
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
                    trimmed += f"\n[Umi] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2 and not self.rushed():
                if self.is_buy(name, args):
                    # a checkout of a cart Umi knows is empty only fails and costs a step (at most twice, in case the record is wrong)
                    if not self.cart_ids() and self.empty_blocks < 2:
                        self.empty_blocks += 1
                        self.trace.append({"kind": "empty_cart_held"})
                        out.append((cid, "Not run (no step used): the cart is empty, so a checkout would only fail. Add each item's fitting listing first, then check out."
                                    + "\nCart now: " + self.cart_text()))
                        continue
                    wider = self.extra_hold() or self.drop_hold() or self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
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
                    # the listing a requirement takes its value from is the reference, not the product to buy
                    source = [i for i in self.cart_ids() if i in self.chain_links]
                    if source and not self.source_held and self.left() > 4:
                        self.source_held = True
                        self.trace.append({"kind": "source_held", "ids": source})
                        out.append((cid, "Not run yet (no step used). " + ", ".join(source) + " is the listing a requirement takes its value from, not the product to buy. "
                                    "Put in its place a listing that meets every requirement of the request, with the value that page gives, then check out."
                                    + "\nCart now: " + self.cart_text()))
                        continue
                    # an order before a chain's end cannot pass: while steps and spend remain, walk on instead of only saying so
                    open_chain = {need for _, need in self.needs_now()} & {need for _, need in self.refs if need not in self.ref_values}
                    if open_chain and self.order_walks < ORDER_WALKS and self.left() >= CHAIN_KEEP + self.tail() and self.cost < self.spend_cap() - 0.01:
                        self.order_walks += 1
                        state = (len(self.ref_values), len(self.noted), len(self.full))
                        walked = self.chain_walk(3)
                        if self.finished:
                            out.append((cid, "The episode ended during the walk. [episode finished]"))
                            continue
                        if state != (len(self.ref_values), len(self.noted), len(self.full)):
                            out.append((cid, "Not run yet (no step used). Before the checkout, Umi walked further along the pages a requirement points at.\n" + walked
                                        + "\nApply the value the last page gives, replace any cart line that no longer fits, then check out.\nCart now: " + self.cart_text()))
                            continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # n47: a value the shopper excludes under a line's own key, or an accessory not covering its device
                    ruled = self.guarded(self.code_hold, "")
                    if ruled:
                        out.append((cid, ruled + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Umi read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Umi re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    # an unfinished chain is no reason to end: walk on once before the review
                    live = {need for _, need in self.needs_now()}
                    if not self.walk_resumed and self.left() >= 6 and any(need in live and need not in self.ref_values for _, need in self.refs):
                        self.walk_resumed = True
                        walked = self.chain_walk(3)
                        if walked:
                            out.append((cid, "Not run yet. Before ending, Umi walked further along the pages a requirement points at:\n" + walked
                                        + "\nApply the value the last page gives and buy what fits; end without a purchase only if nothing does."))
                            continue
                    # unread pages of searches already run come before giving up, while steps remain to buy from them
                    if not self.decline_paged and self.left() >= 7 and self.unpaged and self.cost < self.spend_cap():
                        self.decline_paged = True
                        rest = self.page_rest(self.left() - 5)
                        if rest:
                            out.append((cid, "Not run yet. Before ending, Umi read the unread pages of your earlier searches (one step a page):\n" + rest
                                        + "\nOpen any listing there that may meet every requirement in force and buy what fits; end without a purchase only if none does."))
                            continue
                    # after a change the review needs no step of its own: three left still buy a listing it names
                    if self.left() > 5 or (self.reaudit and self.left() >= 3):
                        review = self.audit()
                        # n47: a review that keyed its fits by item name: by number, as every consumer reads them
                        keyed = self.guarded(self.lead_numbers, {}, self.lead_items)
                        if keyed and set(map(str, keyed)) != set(self.lead_items):
                            self.trace.append({"kind": "lead_rekeyed", "from": list(self.lead_items)[:4]})
                            self.lead_items = {str(n): ids for n, ids in keyed.items()}
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
            if self.is_refuse(name, args) and self.pointer_live():
                if self.pointer_blocks < 2 and self.left() > self.tail() + 2:
                    self.pointer_blocks += 1
                    self.trace.append({"kind": "pointer_held", "round": self.pointer_blocks})
                    out.append((cid, "Not run yet (no step used). The shopper's answer takes a value from another listing, so this shopper expects a purchase: do not end without one. "
                                "For each requested item put in the cart the listing read so far that meets the most requirements, as near the value that page gives as you can tell, then check out.\nCart now: " + self.cart_text()))
                    continue
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                if not self.pointer_filled and self.pointer_fill():
                    out.append((cid, "Not run yet. Umi put the closest read listings in the cart. Check out now.\nCart now: " + self.cart_text()))
                    continue
                if not self.pointer_forced and self.pointer_filled and not self.pointer_faults():
                    self.pointer_forced = True
                    turned = self.finish(again=True)
                    out.append((cid, "Umi checked out the closest read listings. " + turned + (" [episode finished]" if self.finished else "")))
                    continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                # n47: no ending while a pair switch is untried, nor while opened listings state every requirement of each item
                switch = self.guarded(self.pair_decline, "")
                if switch:
                    out.append((cid, "Not run yet (no step used). " + switch))
                    continue
                filled = self.guarded(self.decline_fill, "") if not self.rushed() else ""
                if filled:
                    out.append((cid, "Not run yet. " + filled + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            # n49: the cart code checks before a checkout the agent asked for (swaps, quantities, a missing item)
            if self.is_buy(name, args) and self.roles.get("cart") and not self.rushed():
                go, fixed = self.order_fixes()
                if not go:
                    out.append((cid, "Not run yet (no step used). " + fixed + "\nCart now: " + self.cart_text()))
                    continue
                if fixed:
                    trimmed = "\n[Umi] " + fixed + "\nCart now: " + self.cart_text() + trimmed
            if ending and self.left() >= 2 and not self.rushed():
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Umi] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Umi] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if name == self.roles.get("search"):
                self.last_found = observation
                self.typed = (self.typed + [args])[-6:]
                self.note_pages(name, args, observation)
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
                inner = _peel(observation)
                listing = self.ident(inner) if isinstance(inner, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(inner))
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
                trimmed += self.guarded(self.cart_note, "", args)
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            if name in readers:
                for seen in [observation] + self.paged_obs:
                    text += self.fit_marks(seen)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Umi] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Umi re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            gap = self.basket_gap()
            if gap:
                text += "\n" + gap
            if self.queued:
                text += "\n[notes seen on Umi's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def code_fits(self, ident: str, number: int) -> bool:
        """A read, buyable listing whose own attributes state every requirement of the item by plain word matching. False
        whenever a requirement is a limit, an exclusion or a pointer, which code does not judge."""
        block = _fact_block(self.full.get(ident) or {})
        if not block or ident in self.gone or self.sold_out(ident) or not self.buyable(ident):
            return False
        pointed = {need for _, need in self.refs}
        for n, need in self.needs_now():
            if n != number or need in self.fit_needs:
                continue      # n47: a fit between requested items is judged by the pair check below
            prop, _, value = need.partition(":")
            prop, value = prop.strip(), value.strip()
            if not prop or not value or need in pointed or _NOT_PLAIN.search(value.lower()) or len(value.split()) > 5 or _excluded(value):
                return False
            keys = [k for k in block if _names(_stem(prop), _stem(k)) or _words(prop).replace(" ", "") == _words(k).replace(" ", "")]
            if not any(_phrase_in(value, str(block[k])) for k in keys):
                return False
        # n47: an accessory whose own words do not cover the partner line carted for it (or the reverse) does not fit
        return self.guarded(self.pair_verdict, (None, ""), ident, number)[0] is not False

    def finish_fill(self) -> None:
        """The episode is ending with an item that has no cart line: an order without it fails and so does an ending without a
        purchase, so Umi carts, for each such item, the cheapest listing already read that meets every requirement by
        plain word matching (no model call), once, when steps and the budget allow; else nothing changes."""
        if self.finish_filled or not self.items:
            return
        self.finish_filled = True
        now = self.cart_ids()
        held = {self.item_number_of(i) for i in now}
        missing = [n for n in range(1, len(self.items) + 1) if n not in held]
        if not missing:
            return
        plan: dict[str, int] = {}
        for number in missing:
            picks = [i for i in self.full if i not in now and i not in plan and self.item_number_of(i) == number and self.code_fits(i, number)]
            if not picks:
                return
            picks.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            plan[picks[0]] = self.items[number - 1].qty
        lines = {**{i: self.cart_lines.get(i, 1) for i in now}, **plan}
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each":
            prices = [self.price_of(i) for i in lines]
            if any(p is None for p in prices) or sum(p * lines[i] for p, i in zip(prices, lines)) > self.budget:
                return
        calls = self.cart_calls(lines, [])
        cart = self.roles.get("cart")
        if not cart or not calls or len(calls) + 1 > self.left():
            return
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return
            self.note_cart(cart, args)
        self.trace.append({"kind": "finish_fill", "items": missing, "put": list(plan)})

    def pointer_live(self) -> bool:
        """The shopper pointed at another listing and that requirement is still in force."""
        return any(need in self.pointer_needs for _, need in self.needs_now())

    def pointer_bad(self, ident: str) -> bool:
        """A line an order must not hold: sold out, or only the reference a pointer takes its value from."""
        return ident in self.gone or self.sold_out(ident) or ident in self.chain_links

    def pointer_faults(self) -> list[str]:
        """Why the cart is no sound order: an unsound line, an uncarted item, or a total or line over the budget."""
        now = self.cart_ids()
        faults = [i for i in now if self.pointer_bad(i)]
        if not now or self.uncarted():
            faults.append("an item has no line")
        if isinstance(self.budget, (int, float)):
            prices = {i: self.price_of(i) for i in now}
            if self.budget and any(p is None for p in prices.values()):
                faults.append("a price is unknown")
            elif self.brief.get("budget_scope") == "each":
                faults += [i for i, p in prices.items() if p is not None and p > self.budget]
            elif sum((p or 0) * self.cart_lines.get(i, 1) for i, p in prices.items()) > self.budget:
                faults.append("over the budget")
        return faults

    def pointer_fill(self) -> bool:
        """The shopper pointed at another listing and the episode would end with an item uncarted or an unsound line: Umi carts,
        once, for each such item, the read listing of its kind stating the most of its plain requirements, cheapest first on a
        tie, inside the budget (no model call, unless the price field of the store's rows is still unknown). Nothing changes
        unless every item gets a line. Returns whether it changed the cart into one with a line for every item."""
        if not self.pointer_live() or self.pointer_filled or not self.items:
            return False
        self.pointer_filled = True
        if not self.pointer_faults():
            return False
        capped = isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each"
        if isinstance(self.budget, (int, float)) and not self.price_key and self.last_found is not None:
            self.read_shape(self.roles.get("search") or "", self.last_found)
        now = self.cart_ids()
        keep = [i for i in now if not self.pointer_bad(i)]
        room = self.budget if isinstance(self.budget, (int, float)) and not capped else None
        if room is not None:
            keep = [i for i in keep if (self.price_of(i) if self.price_of(i) is not None else float("inf")) <= room]
        spent = 0.0
        if capped:
            within = []
            for i in sorted(keep, key=lambda x: self.price_of(x) if self.price_of(x) is not None else float("inf")):
                price = self.price_of(i)
                if price is not None and spent + price * self.cart_lines.get(i, 1) <= self.budget:
                    within.append(i)
                    spent += price * self.cart_lines.get(i, 1)
            keep = within
        held = {self.item_number_of(i) for i in keep}
        missing = [n for n in range(1, len(self.items) + 1) if n not in held]
        pointed = {need for _, need in self.refs}
        plan: dict[str, int] = {}
        for number in missing:
            item = self.items[number - 1]
            scored = []
            for ident in self.full:
                if ident in keep or ident in plan or self.item_number_of(ident) != number or self.pointer_bad(ident) or not self.buyable(ident) or not self.of_kind(ident, item):
                    continue
                if self.guarded(self.excluded_hit, "", ident, number):
                    continue      # n47: its own attribute states a value the shopper excludes
                price = self.price_of(ident)
                if (capped and (price is None or spent + price * item.qty > self.budget)) or (room is not None and (price is None or price > room)):
                    continue
                block = _fact_block(self.full.get(ident) or {}) or {}
                hits = 0
                for n, need in self.needs_now():
                    prop, _, value = need.partition(":")
                    if n != number or need in pointed or need in self.fit_needs or not value.strip():
                        continue
                    keys = [k for k in block if _names(_stem(prop.strip()), _stem(k))]
                    hits += any(_phrase_in(value.strip(), str(block[k])) for k in keys)
                scored.append((-hits, price if price is not None else float("inf"), ident))
            if not scored:
                return False
            pick = min(scored)
            plan[pick[2]] = item.qty
            spent += (pick[1] if pick[1] != float("inf") else 0) * item.qty
        lines = {**{i: self.cart_lines.get(i, 1) for i in keep}, **plan}
        calls = self.cart_calls(lines, [i for i in now if i not in lines])
        cart = self.roles.get("cart")
        if not cart or not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
        self.trace.append({"kind": "pointer_fill", "items": missing, "put": list(plan)})
        return True

    def uncarted(self) -> int:
        """How many requested items have no cart line."""
        held = {self.item_number_of(i) for i in self.cart_ids()} - {None}
        return max(0, len(self.items) - len(held))

    def finish_swap(self) -> None:
        """The episode ends with a cart line the store said is sold out: an order holding it fails, so Umi swaps it for the
        best replacement the swap review named among listings already read (no model call), once, when steps allow."""
        if self.finish_swapped:
            return
        now = self.cart_ids()
        lost = [i for i in now if i in self.gone or self.sold_out(i)]
        if not lost:
            return
        self.finish_swapped = True
        plan: dict[str, int] = {}
        for ident in lost:
            pool = [p for p in self.swap_picks.get(ident, []) if p not in now and p not in plan and p not in self.gone and not self.sold_out(p)]
            pick = self.swap_choice(ident, pool, [i for i in now if i not in lost] + list(plan), len(lost))
            if pick:
                plan[pick] = self.cart_lines.get(ident, 1)
        if len(plan) < len(lost):
            return
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in lost}
        calls = self.cart_calls({**keep, **plan}, lost)
        cart = self.roles.get("cart")
        if not cart or not calls or len(calls) + 1 > self.left():
            return
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return
            self.note_cart(cart, args)
        self.trace.append({"kind": "finish_swap", "lost": lost, "put": list(plan)})

    def swap_choice(self, ident: str, pool: list[str], lines: list[str], lost: int) -> str | None:
        """n47: the replacement for a sold-out line at a forced finish. The first one or two unread picks are opened when steps
        and time allow; a pick code rules out is dropped. Under a cheapest pick the swap review's price order stands; else a
        read pick meeting the requirements comes first. With nothing better, n45's choice: the review's first pick (a sold-out
        line left in the cart fails for certain). Any error falls back to that choice too."""
        try:
            if not self.rushed():
                unread = [p for p in pool[:2] if not self.page_read(p)]
                if unread:
                    self.read_pages(unread, 2 * lost + 2)
            number = self.item_of(ident) or (self.item_of(pool[0]) if pool else None)
            # n47 (review): what the reads just showed counts: sold out, no longer buyable or over the budget drops a pick
            units = {line: self.cart_lines.get(line, 1) for line in lines}
            clean = [p for p in pool if self.buyable(p) and not self.sold_out(p) and p not in self.gone
                     and not (number and self.hard_fault(p, number, lines)) and self.budget_ok({**units, p: self.cart_lines.get(ident, 1)})]
            if not clean:
                return pool[0] if pool else None      # n45's choice when nothing passes
            if self.brief.get("pick") == "cheapest":
                return clean[0]
            return self.best_read(clean, number, lines) or clean[0]
        except Exception as exc:
            self.trace.append({"kind": "n47_guard", "where": "swap_choice", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return pool[0] if pool else None

    def page_read(self, ident: str) -> bool:
        """n47: the listing's own page or attributes were read (a search row alone is not that)."""
        return ident in self.full or ident in self.opened or ident in self.noted

    def read_pages(self, ids: list[str], keep: int) -> None:
        """n47: open these listings with code's own calls, keeping `keep` steps: the attributes view, else the page view."""
        for ident in ids:
            if self.page_read(ident) or ident not in self.rows:
                continue
            call = self.one_read(self.facts_tool(), ident) or self.one_read(self.offer_tool(), ident)
            if not call or self.left() <= keep or self.finished:
                break
            reply = self.act(*call)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(call[0], seen)
            self.note_view(call[0], seen)
            self.trace.append({"kind": "read_page", "id": ident})

    def best_read(self, pool: list[str], number: int | None, lines: list[str]) -> str | None:
        """n47: of candidates in their given order (cheapest first under a cheapest pick), the first whose attributes were read,
        buyable, that code does not rule out; among those, one meeting every requirement by plain word matching first, then
        one stating every requirement somewhere, then the rest (a cheapest pick keeps price order but for unstated values)."""
        ok = [p for p in pool if self.page_read(p) and self.buyable(p) and p not in self.gone and not (number and self.hard_fault(p, number, lines))]
        if not ok:
            return None

        def tier(p: str) -> int:
            if not number:
                return 1
            return 0 if self.code_fits(p, number) else 1 if not self.unconfirmed(p, number) else 2
        if self.brief.get("pick") == "cheapest":
            return min(ok, key=lambda p: (tier(p) == 2, ok.index(p)))
        return min(ok, key=lambda p: (tier(p), ok.index(p)))

    def budget_ok(self, lines: dict[str, int]) -> bool:
        """n47: the lines' total (or each line, under a per-item limit) inside the budget; unknown prices fail a set budget."""
        if not isinstance(self.budget, (int, float)):
            return True
        prices = {i: self.price_of(i) for i in lines}
        if any(p is None for p in prices.values()):
            return False
        if self.brief.get("budget_scope") == "each":
            return all(p <= self.budget for p in prices.values())
        return sum(p * lines[i] for i, p in prices.items()) <= self.budget

    def put_lines(self, lines: dict[str, int], drop: list[str]) -> bool:
        """n47: set the cart to these lines with code's own calls (no model); whether every call went through."""
        calls = self.cart_calls(lines, drop)
        cart = self.roles.get("cart")
        if not cart or not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        return True

    # ---------- n49: the cart code checks right before a checkout ----------
    def whole_cart_tool(self) -> bool | None:
        """n49: the cart tool takes the whole cart in one call (a list of lines): True with a count field in its lines, None
        without one (units cannot be said), False for a tool taking one line per call."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        for spec in (schema.get("properties") or {}).values():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                return True if any(isinstance(s, dict) and "integer" in _kinds(s) for s in shape["properties"].values()) else None
        return False

    def held_counts(self) -> dict[str, int]:
        """n49: units of each cart line: from a cart view newer than the last cart change (a row holding the listing's id and
        a key that names a count and nothing else), else from code's own count of the cart calls."""
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        if fresh:
            found: dict[str, int] = {}

            def walk(node: Any, depth: int) -> None:
                if depth > 5:
                    return
                if isinstance(node, dict):
                    ident = next((v for v in node.values() if isinstance(v, str) and v in self.rows), None)
                    count = next((v for k, v in node.items() if isinstance(v, int) and not isinstance(v, bool) and _count_key(str(k))), None)
                    if ident and count is not None:
                        found[ident] = found.get(ident, 0) + count
                        return
                    for value in node.values():
                        walk(value, depth + 1)
                elif isinstance(node, list):
                    for value in node:
                        walk(value, depth + 1)
            walk(self.cart_view, 0)
            kept = {i: c for i, c in found.items() if c > 0}
            if kept:
                return kept
        return {i: c for i, c in self.cart_lines.items() if c and c > 0}

    def look_at_cart(self) -> None:
        """n49: one cart view call (no model), when the store has one and steps allow."""
        view = self.roles.get("cart_view")
        if not view or self.left() <= ORDER_KEEP or self.finished:
            return
        faults: list[str] = []
        args = _shape({}, self.schemas.get(view), "args", faults)
        if faults:
            return
        reply = self.act(view, args)
        if reply is None or self.finished:
            return
        self.cart_view = reply.get("observation")
        self.cart_seen_at = self.step
        self.observe(view, self.cart_view)

    def unit_call(self, ident: str, units: int) -> dict[str, Any] | None:
        """n49: one call of a tool that takes one line per call, changing a line by `units` (more when positive, fewer when
        negative), or setting it to `units` when the tool sets counts; None when the tool cannot say that."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target or not count:
            return None
        if self.cart_sets(name):
            # a setting tool: `units` is the new count, 0 takes the line off
            return {target: ident, count: units} if units >= 0 else None
        if units == 0:
            return None
        if units > 0:
            return {target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), count: units}
        if mode:
            return {target: ident, mode: props[mode]["enum"][1], count: -units}
        low = props[count].get("minimum")
        return {target: ident, count: units} if isinstance(low, int) and low <= units else None

    def send_cart(self, args: dict[str, Any]) -> bool:
        """n49: one cart call by code (no model); whether it went through."""
        cart = self.roles.get("cart")
        faults: list[str] = []
        args = _shape(args, self.schemas.get(cart), "args", faults)
        if not cart or faults or self.finished:
            return False
        reply = self.act(cart, args)
        if reply is None or self.finished:
            return False
        self.note_cart(cart, args)
        self.observe(cart, reply.get("observation"))
        return not (isinstance(reply.get("observation"), dict) and reply["observation"].get("error"))

    def set_count(self, ident: str, want: int, held: dict[str, int]) -> bool:
        """n49: one line to `want` units with one call, keeping every other line: the whole new cart for a tool taking the
        whole cart (only with a count field), the new count for a tool setting counts, the difference for a tool adding or
        taking off units. Nothing is sent when the tool cannot say it."""
        have = held.get(ident, 0)
        if have == want:
            return True
        whole = self.whole_cart_tool()
        if whole is None:
            return False
        if whole:
            lines = dict(held)
            lines[ident] = want
            calls = self.cart_calls({i: c for i, c in lines.items() if c > 0}, [])
            return bool(calls) and len(calls) == 1 and self.send_cart(calls[0])
        call = self.unit_call(ident, want if self.cart_sets(self.roles.get("cart") or "") else want - have)
        return bool(call) and self.send_cart(call)

    def asked_quantity(self, number: int) -> int | None:
        """n49: the item's quantity when code can trust it: the brief read a product (not its stand-in) and the request
        states that number (as digits, or one as a word)."""
        item = self.items[number - 1]
        want = item.qty
        if not isinstance(want, int) or want <= 0 or item.name == "the requested product":
            return None
        words = {1: r"one|a|an|single", 2: r"two|pair of", 3: r"three", 4: r"four", 5: r"five"}.get(want, "")
        return want if re.search(r"(?<![\d.])" + str(want) + r"(?![\d.])" + (f"|\\b(?:{words})\\b" if words else ""), self.request, re.I) else None

    def quantity_faults(self) -> list[tuple[str, int, int]]:
        """n49: (line, units held, units requested) for lines whose item is clear, has that one line in the cart, and has a
        quantity code can trust."""
        held = self.held_counts()
        owners: dict[int, list[str]] = {}
        for ident in held:
            number = self.item_of(ident)
            if number:
                owners.setdefault(number, []).append(ident)
        out = []
        for ident, have in held.items():
            number = self.item_of(ident)
            if not number or len(owners.get(number, [])) != 1:
                continue
            want = self.asked_quantity(number)
            if want and have != want:
                out.append((ident, have, want))
        return out

    def confirmed_order(self) -> bool:
        """n49: the request asks for a message about the order before it and no message is left: code then leaves the cart
        as the shopper last saw it."""
        return any(p["when"] == "order" and p["messages"] > 0 for p in self.process) and self.msgs_free() <= 0

    def fix_quantities(self) -> str:
        """n49: lines holding another count than their item asks for are set to it by code, one call each, while the steps
        kept for the checkout stay free (two rounds at most, the cart looked at between them when the store shows it): a cart
        tool may add to a line where the agent meant to set it. Only lines that changed are reported."""
        if self.confirmed_order():
            return ""
        notes = []
        while self.qty_rounds < 2 and not self.finished:
            faults = self.quantity_faults()
            if not faults:
                break
            self.qty_rounds += 1
            held = self.held_counts()
            for ident, have, want in faults:
                if self.left() <= ORDER_KEEP + 1:
                    break
                if self.set_count(ident, want, held):
                    notes.append(f"{ident} {have} -> {want}")
                    held = self.held_counts()
            self.trace.append({"kind": "qty_fix", "round": self.qty_rounds, "lines": faults, "done": list(notes)})
            self.look_at_cart()
            if not self.roles.get("cart_view") or self.cart_seen_at < self.cart_at:
                break
        return ("Umi set the cart lines to the requested quantities (" + "; ".join(notes) + ")") if notes else ""

    def missing_items(self) -> list[int]:
        """n49: requested items with no cart line, by the store label each line carries; none when a line's item is unclear
        or the cart already holds as many lines as items are requested (two items may share one store label)."""
        ids = self.cart_ids()
        held = [self.item_of(i) for i in ids]
        if not held or None in held or len(ids) >= len(self.items):
            return []
        return [n for n in range(1, len(self.items) + 1) if n not in held]

    def shelf_note(self, missing: list[int]) -> str:
        """n49: the requested items the cart lacks, with read listings of each (the store label they carry) and the label
        of each line in the cart, so a line on another item's label is seen as that item's."""
        cat_key = self.category_key()

        def label(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "?") if cat_key else "?"
        parts = []
        for number in missing:
            item = self.items[number - 1]
            pool = [i for i in self.full if self.item_of(i) == number and self.buyable(i) and i not in self.gone and not self.hard_fault(i, number)]
            pick = self.best_read(pool, number, self.cart_ids())
            seen = ([pick] if pick else []) + [i for i in pool if i != pick][:2]
            parts.append(f"item {number} ({item.name}, quantity {item.qty}) has no cart line"
                         + (f"; read listings on its own store label: {', '.join(f'{i} ({label(i)})' for i in seen)}" if seen else "; search its own store label"))
        lines = ", ".join(f"{i} = {label(i)}, item {self.item_of(i)}" for i in self.cart_ids())
        return ("Each requested item needs its own cart line, on the store label that names it: " + "; ".join(parts)
                + f". Cart lines now: {lines}. Add the missing item(s) at the requested quantity, then check out.")

    def stated_swap(self) -> str:
        """n49: a cart line that states one of its item's word requirements nowhere (no attribute names the property with the
        value, and neither title, description nor another attribute holds its words: n47's gap reading) is swapped, once, for
        a read, buyable listing of the same item that states every such value under the property's own key. Only when: no
        model check passed the cart as it is; every requirement of that item is a plain word value code reads (no limits,
        closed lists, numbers or measures, which code cannot compare); the new listing is not ruled out by code and keeps the
        budget; the steps for both calls and the checkout are free; and no order message left the cart to the shopper."""
        if self.stated_swapped or self.rushed() or self.confirmed_order() or self.whole_cart_tool() is None:
            return ""
        now = self.cart_ids()
        if self.verdicts.get(self.check_key(now)) is True:
            return ""
        notes = []
        for ident in list(now):
            number = self.item_of(ident)
            if not number or sum(1 for j in now if self.item_of(j) == number) != 1:
                continue
            needs = self.plain_needs(number)
            if not needs or any(BOUNDED.search(v) or _NOT_PLAIN.search(v.lower()) or len(v.split()) > 6 or re.search(r"\d", v) for _, _, v in needs):
                continue
            gaps = self.gap_needs(ident, number)
            if not gaps:
                continue
            others = [k for k in now if k != ident]
            pool = [j for j in self.full if j not in now and self.item_of(j) == number and self.buyable(j) and j not in self.gone]
            good = [j for j in pool if not self.gap_needs(j, number) and all(j in self.stated_by(number, prop, value, []) for prop, value in gaps)
                    and not self.hard_fault(j, number, others) and not self.pointer_bad(j)]
            if self.brief.get("pick") == "cheapest":
                good.sort(key=lambda j: self.price_of(j) if self.price_of(j) is not None else float("inf"))
            pick = self.best_read(good, number, others)
            if not pick:
                continue
            held = self.held_counts()
            qty = held.get(ident) or self.items[number - 1].qty or 1
            lines = {k: c for k, c in held.items() if k != ident}
            lines[pick] = qty
            if not self.budget_ok(lines):
                continue
            whole = self.whole_cart_tool()
            sets = self.cart_sets(self.roles.get("cart") or "")
            calls = self.cart_calls(lines, [ident]) if whole else [self.unit_call(pick, qty), self.unit_call(ident, 0 if sets else -held.get(ident, qty))]
            if not calls or any(c is None for c in calls) or self.left() <= ORDER_KEEP + len(calls):
                continue
            self.stated_swapped = True
            # the new line goes in first: a failed removal then leaves two lines for the model to see, never none
            done = all(self.send_cart(c) for c in calls)
            self.trace.append({"kind": "stated_swap", "out": ident, "in": pick, "gaps": gaps, "ok": done})
            if done:
                notes.append(f"{ident} -> {pick} ({pick} states " + "; ".join(f"{p} {v}" for p, v in gaps) + f" under its own key; {ident} states it nowhere)")
            break
        return ("Umi swapped a cart line for one that states the requirement: " + "; ".join(notes)) if notes else ""

    def order_fixes(self) -> tuple[bool, str]:
        """n49: right before a checkout the agent asked for: swap a line stating a requirement nowhere, set quantities, then
        hold the checkout (at most SHELF_BLOCKS times, with steps to act) while a requested item has no cart line."""
        said = [x for x in (self.guarded(self.stated_swap, ""), self.guarded(self.fix_quantities, "")) if x]
        missing = self.guarded(self.missing_items, []) if len(self.items) > 1 else []
        if missing and self.shelf_blocks < SHELF_BLOCKS and self.left() > ORDER_KEEP + 3 and not self.late():
            note = self.guarded(self.shelf_note, "", missing)
            if note:
                self.shelf_blocks += 1
                self.trace.append({"kind": "shelf_held", "missing": missing, "cart": self.cart_ids()})
                return False, " ".join(said + [note])
        return True, " ".join(said)

    def finish_leads(self) -> None:
        """n47: a forced finish with an empty cart: the decline review named listings that appear to fit, so Umi carts, once,
        one read listing per item from them (opening the first one or two unread ones when steps allow), when every item
        gets one inside the budget; else nothing changes and the later fills or the ending go as before."""
        if self.leads_filled or not self.lead_items or not self.items or self.cart_ids():
            return
        self.leads_filled = True
        plan: dict[str, int] = {}
        for number, ids in self.lead_numbers(self.lead_items).items():
            ids = [i for i in ids if i in self.rows]
            pool = [i for i in ids if i not in plan and not self.sold_out(i) and i not in self.gone and self.of_kind(i, self.items[number - 1])]
            unread = [i for i in pool[:2] if not self.page_read(i)]
            if unread and not self.rushed():
                self.read_pages(unread, len(self.items) + 2)
            pick = self.best_read(pool, number, list(plan))
            if pick and number not in plan.values():
                plan[pick] = number
        if {n for n in plan.values()} != set(range(1, len(self.items) + 1)):
            return
        lines = {p: self.items[n - 1].qty for p, n in plan.items()}
        if not self.budget_ok(lines):
            return
        # n47 (review): a "may fit" lead is bought only when code confirms it or a model check passed it; else n45's ending
        sure = all(self.code_fits(p, n) or self.confirmed == p or self.verdicts.get(self.check_key([p])) is True for p, n in plan.items())
        if not sure and not self.late():
            # n47 (review): a model call inside the finish only with 50 s or more left; else n45's ending
            self.check_cart(list(lines), SPEND_HARD)
        if not sure and self.verdicts.get(self.check_key(list(lines))) is not True:
            return
        if not self.put_lines(lines, []):
            return
        self.trace.append({"kind": "finish_leads", "put": list(lines)})

    def decline_fill(self) -> str:
        """n47: the agent ends without buying while listings it opened state every requirement of each item under their own
        keys (plain word matching, at least one requirement per item, the item's category label, inside the budget): Umi
        carts them, once, instead (a model check that can still run must pass that cart first) and the agent checks out."""
        if self.decline_filled or not self.items or not self.roles.get("cart") or not self.category_key() or self.left() < len(self.items) + 4:
            return ""
        self.decline_filled = True
        cat_key = self.category_key()
        now = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.gone]
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            if not self.plain_needs(number):
                return ""          # nothing code can confirm: the agent's own reading stands
            held = [i for i in now if self.item_of(i) == number and self.code_fits(i, number)]
            if held:
                plan[held[0]] = self.cart_lines.get(held[0], item.qty)
                continue
            picks = [i for i in self.full if i not in plan and self.item_of(i) == number and str((self.full[i] or {}).get(cat_key) or "").strip()
                     and self.of_kind(i, item) and self.code_fits(i, number) and not self.hard_fault(i, number, list(plan))]
            if not picks:
                return ""
            picks.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            plan[picks[0]] = item.qty
        if not self.budget_ok(plan):
            return ""
        self.check_cart(list(plan), SPEND_HARD)
        if self.verdicts.get(self.check_key(list(plan))) is not True:
            return ""      # n47 (review): word matching alone never overrides a decline: the model's check must pass the cart
        if not self.put_lines(plan, [i for i in self.cart_ids() if i not in plan]):
            return ""
        self.trace.append({"kind": "decline_fill", "put": list(plan)})
        return ("Umi's own reading found listings you opened whose attributes state every requirement in force: " + ", ".join(plan)
                + ". Umi put them in the cart. Check out now; end without a purchase only if one of them clearly breaks a requirement, and say which.\nCart now: " + self.cart_text())

    def pair_decline(self) -> str:
        """n47: an ending without a purchase while a pair switch is untried: held once. A device may fit no accessory seen
        while another device does, or an accessory seen covers the device; with many steps left the search is not done."""
        if self.pair_held or not self.fit_pairs or self.left() <= 6:
            return ""
        options: list[str] = []
        for a, b in self.fit_pairs:
            device = next((i for i in self.cart_ids() if self.item_of(i) == b), None)
            options += self.pair_options(a, b, device)
        if not options and self.left() < 10:
            return ""
        self.pair_held = True
        self.trace.append({"kind": "pair_decline_held", "options": options[:4]})
        a, b = self.fit_pairs[0]
        text = (f"The request pairs item {a} ({self.items[a - 1].name}) with item {b} ({self.items[b - 1].name}): when no accessory fits the device you picked,"
                " switch the device to one that a fitting accessory lists (or the accessory to one whose compatibility words list the device), and never end without a purchase while that switch is untried.")
        if options:
            text += " Umi's pair check found among listings seen: " + "; ".join(options[:4]) + ". Open them, check their other requirements, cart the pair and check out."
        else:
            text += f" Search item {a}'s category for listings whose compatibility words list a device code, open them, and pick the pair together."
        return text

    def guarded(self, job: Any, fallback: Any, *args: Any) -> Any:
        """n47: run one of n47's own steps; an error is traced and gives `fallback`, so n45's path goes on unchanged."""
        try:
            return job(*args)
        except Exception as exc:
            self.trace.append({"kind": "n47_guard", "where": getattr(job, "__name__", "?"), "error": f"{type(exc).__name__}: {exc}"[:200]})
            return fallback

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and holding:
            self.finish_swap()
        if buy and self.roles.get("cart"):
            self.guarded(self.finish_leads, None)     # n47: an empty cart takes the decline review's read, named fits first
            self.finish_fill()
            self.pointer_fill()
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # n49: a forced checkout gets the same code fixes (no hold: nothing is left to act on one)
                if self.roles.get("cart"):
                    self.guarded(self.stated_swap, "")
                    self.guarded(self.fix_quantities, "")
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2 and not self.rushed():
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
                if self.left() >= 2 and not self.rushed():
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
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, 0.0)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Umi] Umi tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SECOND_MULTI if len(self.items) > 1 and not (self.refs or self.fit_pairs) else 0.02)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def ceiling(self) -> float:
        """No model call starts past this: several products without a pointer stop lower than one product or a chain."""
        # n47: a fit pair keeps the envelope n45 gave these tasks, which it read nearly always as lookups
        return MULTI_MAX if len(self.items) > 1 and not (self.refs or self.fit_pairs) else EPISODE_MAX

    def spend_cap(self) -> float:
        """The episode's spend limit for the agent's own loop: higher once a requirement points at another listing."""
        return (LOOP_SPEND_MULTI if len(self.items) > 1 else LOOP_SPEND) + (CHAIN_EXTRA if self.refs or self.fit_pairs else 0.0)

    def work(self, loop_cap: int, extra: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < self.spend_cap() + extra and not self.rushed():
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            if self.hunt_queue:
                found = self.retrieve()
                if found:
                    self.chat.append({"role": "user", "content": found + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
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
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8 and not self.late():
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
        episode = Umi(problem_data)
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
