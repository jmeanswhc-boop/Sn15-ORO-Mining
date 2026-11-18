"""agent_v4: buys for one shopper inside an online store, using the store's own tools.

Built on KYO v6's public code (released 2026-11-01). A light model walks the store with the world's own tools; code
keeps the ledger (items, needs, budget, listings seen), asks the shopper, and owns the checkout and the ending. On top
of that base this agent:
- paces the episode by steps and by the clock, and checks out early enough to answer a notice the checkout brings;
- ranks questions by how far the item's own listings differ on each property;
- opens the page of every listing it orders and follows page notes that send a property to another listing;
- repairs a refused checkout itself (sold-out line, new budget) when few moves are left, instead of giving up.
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

LOOP_CALLS = 42             # model turns for a task of up to LOOP_STEPS steps; a longer task gets one more per step
LOOP_STEPS = 46
LOOP_MAX = 110
ALL_CALLS = 60              # model calls of every role together, raised with the turns
SPEND_HARD = 0.078          # dollars a task of up to LOOP_STEPS steps may spend; longer tasks a little more
SPEND_STEP = 0.0003
SPEND_LONG = 0.084
# kept back from the tool loop, so the cart check, the decline audit and sold-out swaps still run on a long task
JUDGE_RESERVE = 0.012
# each task has its own 300 s on the network; the last seconds stay free for a slow reply
TASK_SECONDS = 300
CLOCK_MARGIN = 28
# waits before resending a failed model call, at most 45 s in all
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
PAGE_CHARS = 2400           # a listing the store wrote as plain sentences is shown this far (record fields stay at SHOW_CHARS)
OBS_LIMIT = 15000
HISTORY_LIMIT = 120000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages the desk reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes the desk reads before a cheapest checkout
LIST_PROPS = 2              # properties one question may name: a pair covers two properties per message and is rarely
                            # refused, where shoppers who take one question at a time refuse longer lists

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s; near the end of
# the clock a call gets the short client, so a slow reply cannot carry the episode past its limit
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short = ProxyClient(timeout=15, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once the desk knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by the interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. The desk carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through the interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. The desk adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- A listing's page (its seller's notes) can send one of its properties to another listing ("for the strap colour, go by <shop>'s <kind> with ..., not this item"). Then that other listing's value is this listing's real value for that property: find and open that page before you buy, and follow it on when it sends you further. A rule on the last page that ties one value to another ("<value A> goes with <value B>") gives the value the listing takes.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and the desk reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it. The desk opens the page of a listing you cart when steps allow, because an order counts only for listings whose page was opened.
- Check out as soon as the cart holds a fitting line for every item, and well before the last steps: the first checkout often brings a store notice (a sold-out line, a new budget, a changed price) that takes several steps to answer. The status line says when that point has come.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind, and neither does a refused checkout: a sold-out line takes another listing of that item (the same product from another shop first), and a new budget is met by checking the cart's total against it. Follow the request's wording for that ending. When the ending call takes a remark, it names, for each item with no fitting listing, the closest listing you read (its id and price) and the one requirement it misses, unless the request says not to suggest or name alternatives; then it names none.
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
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"kinds": [one word per question, in order: "need" when its answer states a firm preference, "points" when it gives the value by pointing at another listing, "free" when the shopper does not mind, "crowded" when it complains about too many things at once, "unclear" when it only says the question was not understood, "other" for anything else], "needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value; a record's "desk note" giving a property's value from the listing its seller sends that property to replaces the record's own value for that property, and a value written "(to be read from another listing)" is unknown. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; a record's "desk note" giving a property's value from the listing its seller sends that property to replaces the record's own value for that property, and a value written "(to be read from another listing)" is unknown, not a mismatch; out-of-stock listings and listings that break the budget do not fit."""

REVIEW = """A shopping agent wants to end without buying because it thinks nothing fits. For EVERY listing record below, say whether it breaks a requirement in force, with these reading rules: """ + READING + """
A requirement that only describes another listing (the one a pointing requirement names: its shop, its kind, the properties that pick it out) is not a requirement on the product being bought. The shopper's answers are in the transcript.
Each record shows the listing's attributes that bear on the requirements ("no attribute for ..." when it has none for a property).
Return ONE JSON object and nothing else: {"breaks": [{"id": the listing id, "req": the number of the requirement it breaks (R1, R2, ...), "quote": the record's own words that show the break, copied exactly, at most eight words (a value, or "key: value"), or "absent" when the record gives no value at all for that property}], "open": [ids of UNREAD ROWS of a requested kind whose attributes could show that they fit, most promising first, at most 4], "reason": "one short sentence"}
List only the listings that break a requirement, one entry each. A listing breaks a requirement only when its record shows it; a value that is unclear, or still to be read from another listing, is no break."""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown. A page can give the value through a rule that ties it to something else ("<value A> goes with <value B>", "listings with A take B"): read the rule for the listing the chain is about, and that page then ends the chain with the value the rule gives."""

REFER = """Each source below is a store listing's page (its sentences and its attribute keys) or a shopper's requirement. A source sometimes says that a property's value is to be read from another listing ("for the strap colour, go by the watch Pellmor sells with a steel case, not this item"; "the same lens mount as the camera Ostrel sells with 24 MP, or wherever that page leads"), or states a store rule pairing values ("our rule for the lens mount: K goes with R2, R2 with K"); one sentence about the listing itself can state such a rule too ("its walnut finish is paired with brass handles": for the handles, walnut goes with brass).
Return ONE JSON object and nothing else: {"refers": [{"source": the source's label as given, "property": the property whose value is read elsewhere or ruled, in the source's words, "key": the page's own attribute key for that property when one names it, else "", "shop": the other listing's shop or seller as written, or "", "kind": the other listing's kind of product as written, or "", "values": [each value the source uses to pick out the other listing, copied exactly, one per entry], "not": [each shop, kind or value the source says the other listing does NOT have, copied exactly], "relation": "same" when the property takes that listing's value, "pair" when a pairing rule turns that listing's value into the wanted one, "unlike" when it must differ from it, "at_most" when it may not go above that listing's value, "at_least" when it may not go below it, "within" when it may differ from it by no more than a stated amount, "other" for anything else, "tolerance": for "within", that allowed difference as written with its unit (such as "0.5 cm"), else "", "pairs": [[a value, the value it goes with], ...] as the source states them, else []}]}
Leave out sources that send nothing elsewhere and state no rule. A rule alone, naming no other listing, has empty shop, kind and values."""

PICK = """A shopping agent looks for the one listing a description picks out. Decide which listing record below it is. Every part of the description must hold, read from the record's own attributes, name and fields; numbers written in another unit count when they convert (convert with your own knowledge, compare at the precision the description writes, and show the arithmetic in "work"); a part the description says the listing does NOT have rules a record out. A seller's note or other free text counts only where a sentence positively describes this listing itself: a sentence saying what the listing is not, or sending the reader to another listing, says nothing about this one.
Return ONE JSON object and nothing else: {"work": "short arithmetic or reasoning", "id": the one listing id that matches every part, or "" when none or more than one does}"""

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


def _view(value: Any, depth: int = 0) -> Any:
    """A result clipped for the model: record fields and rows at SHOW_CHARS; a result that is one text, a text wrapped
    under one or two keys, or a listing page written as sentences, at PAGE_CHARS (such a text is the whole page)."""
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return _clip(value, PAGE_CHARS if depth == 0 else SHOW_CHARS)
    if isinstance(value, dict):
        return {k: _clip(v, PAGE_CHARS) if isinstance(v, str) and ((depth == 0 and len(value) <= 2) or (k == "page_text" and depth <= 2))
                else _view(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, SHOW_CHARS, depth + 1) for v in value]
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
            if isinstance(got, dict):
                return got
        except ValueError:
            pass
    # a reply that writes its reasoning before the object (a brace may sit in that text): the largest object that
    # decodes from any opening brace
    decoder = json.JSONDecoder()
    best: tuple[int, dict[str, Any]] | None = None
    for at in [i for i, ch in enumerate(text) if ch == "{"][:400]:
        try:
            got, end = decoder.raw_decode(text, at)
        except ValueError:
            continue
        if isinstance(got, dict) and (best is None or end - at > best[0]):
            best = (end - at, got)
    return best[1] if best else None


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
        # a code with letters before a bare figure (a price a text listing wrote out is unique on a short page too)
        score = ("id" in str(key).lower(), all(re.search(r"[A-Za-z]", v) for v in values), sum(len(v) for v in values) / len(values))
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


def _block_key(node: Any) -> str | None:
    """The key of the attribute block _fact_block reads."""
    if not isinstance(node, dict):
        return None
    best = None
    for key, value in node.items():
        if isinstance(value, dict) and len(value) >= 2:
            count = sum(1 for v in value.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
            if count >= 2 and (best is None or count > best[0]):
                best = (count, key)
    return best[1] if best else None


def _facts(node: Any) -> dict[str, Any]:
    """The attribute block with its yes/no flags too (_fact_block leaves true/false out)."""
    key = _block_key(node)
    if key is None:
        return {}
    return {k: v for k, v in node[key].items() if isinstance(v, (str, int, float, bool))}


def _flag(value: Any) -> Any:
    """A true/false attribute read the way a requirement writes it."""
    return ("yes" if value else "no") if isinstance(value, bool) else value


_FIELD = re.compile(r"\s*([A-Za-z][^:|=\n]{0,40}?)\s*:(?:\s+(.*)|\s*)", re.S)
_PAIRS = re.compile(r"\s*[;,]\s*(?=[^=;,|\n]{1,50}=)")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+")
_SEPS = (" | ", " || ", "\t", " ; ", "; ", ", ")


def _scalar(text: str) -> Any:
    """A text field's flag or empty value as a record holds it; figures stay text (an id may be all digits)."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    return text.strip()


def _pairs(text: str, least: int = 2) -> dict[str, str] | None:
    """A field written as "shade=linen; height=40 cm" -> its pairs; None for any other text."""
    parts = [p for p in _PAIRS.split(text.strip().rstrip(";")) if p.strip()]
    if len(parts) < least or not all(re.match(r"[^=;,|\n]{1,50}=", p.strip()) for p in parts):
        return None
    return {p.split("=", 1)[0].strip(): p.split("=", 1)[1].strip() for p in parts}


def _value(text: Any) -> Any:
    return text if isinstance(text, dict) else (_pairs(text) or _scalar(text))


def _coded(record: dict[str, Any]) -> bool:
    """A record carries a listing code: a one-word value with letters and digits, or a figure under a key naming an id."""
    for key, value in record.items():
        if not isinstance(value, str) or not 2 <= len(value.strip()) <= 64 or " " in value.strip():
            continue
        if re.search(r"\d", value) and (re.search(r"[A-Za-z]", value) or re.search(r"(id|ref|code|sku|num|number|no)$", str(key), re.I)):
            return True
    return False


def _keep(fields: dict[str, Any], head: list[str]) -> dict[str, Any] | None:
    if len(fields) < 3:
        return None
    record = {**({"heading": " ".join(head)} if head else {}), **{k: _value(v) for k, v in fields.items()}}
    return record if _coded(record) else None


def _line(text: str, seps: tuple[str, ...] = _SEPS) -> dict[str, Any] | None:
    """One listing written on one line: "key: value" parts with any separator the line repeats, or "key=value" pairs.
    A part that is no "key: value" goes on with the field before it; words before the first field are its heading."""
    for sep in seps:
        parts = text.split(sep)
        if len(parts) < 3:
            continue
        fields: dict[str, Any] = {}
        head: list[str] = []
        last = None
        for part in parts:
            found = _FIELD.fullmatch(part)
            if found and found.group(1).strip() not in fields and (found.group(2) or "").strip():
                last = found.group(1).strip()
                fields[last] = found.group(2)
            elif last is None:
                head.append(part.strip())
            else:
                fields[last] += sep + part
        if len(head) <= 1:
            record = _keep(fields, head)
            if record:
                return record
    pairs = _pairs(text, 3)
    return _keep(pairs, []) if pairs else None


def _blocks(lines: list[str]) -> list[dict[str, Any]]:
    """Listings written a field a line ("price: 22"), each perhaps under a heading line, a field with no value opening
    a section of indented or bulleted "key: value" lines; a field name met again starts the next listing."""
    out: list[dict[str, Any]] = []
    fields: dict[str, Any] = {}
    head: list[str] = []
    last = section = None

    def close() -> None:
        record = _keep({k: v for k, v in fields.items() if v != {}}, head)
        if record:
            out.append(record)

    for raw in lines:
        inner = bool(_BULLET.match(raw)) or raw[:1] in (" ", "\t")
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue
        found = _FIELD.fullmatch(line)
        pair = found or re.fullmatch(r"([^=]{1,50})=(.*)", line)
        if section is not None and inner and pair:
            fields[section][pair.group(1).strip()] = (pair.group(2) or "").strip()
            continue
        if found and found.group(1).strip() in fields:
            close()
            fields, head, last, section = {}, [], None, None
        if found:
            last = found.group(1).strip()
            value = (found.group(2) or "").strip()
            fields[last] = value if value else {}
            section = None if value else last
        elif last is None:
            head.append(line)
        elif isinstance(fields[last], str):
            fields[last] += " " + line
        elif not fields:
            head.append(line)
    close()
    return out


def _table(lines: list[str]) -> list[dict[str, Any]] | None:
    """Listings as a table of "|"-bounded rows under a header row."""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if l.strip().startswith("|")]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    if len(rows) < 2 or len(rows[0]) < 3:
        return None
    head = rows[0]
    out = [_keep(dict(zip(head, r)), []) for r in rows[1:] if len(r) == len(head)]
    out = [r for r in out if r]
    return out or None


def _text_records(text: str) -> Any:
    """Listings a store wrote as text, read into records: JSON in a string, one listing on a line, several (a line each,
    blocks apart, a table); None for any other text."""
    body = text.strip()
    if body[:1] in "{[":
        try:
            got = json.loads(body)
        except ValueError:
            got = None
        if isinstance(got, (dict, list)):
            return got
    if len(body) < 20:
        return None
    lines = [l for l in body.splitlines() if l.strip()]
    if len(lines) == 1:
        return _line(body)
    found = _table(lines)
    rest: list[str] = []
    if not found:
        each = [_line(_BULLET.sub("", l).strip()) for l in lines]
        found = [r for r in each if r]
        rest = [l.strip() for l, r in zip(lines, each) if not r]
        if len(found) < 2 or len(found) < 0.5 * len(lines):
            # one listing whose fields hold line breaks (cut by a separator its first line repeats), else a field a line
            whole = _line(body, tuple(s for s in _SEPS if lines[0].count(s) >= 2))
            found = [whole] if whole else _blocks(body.splitlines())
            rest = []
    if not found:
        return None
    if len(found) == 1 and not rest:
        return found[0]
    return {"note": " ".join(rest), "rows": found} if rest else found


def _prose_rows(texts: list[str]) -> list[dict[str, Any]] | None:
    """Rows written as plain sentences: the code word every row carries once, with the same lead (a "q-" or "SKU"
    prefix), unique per row, is taken for the listing's id."""
    seen: dict[str, list[set[str]]] = {}
    for n, text in enumerate(texts):
        for token in re.findall(r"(?<![\w-])([A-Za-z][\w-]{2,40})(?![\w-])", text):
            if re.search(r"\d", token):
                lead = (re.match(r"[A-Za-z]+[-_]", token) or re.match(r"[A-Za-z]+", token)).group(0).lower()
                seen.setdefault(lead, [set() for _ in texts])[n].add(token)
    for lead, per in sorted(seen.items(), key=lambda kv: -sum(1 for s in kv[1] if len(s) == 1)):
        ids = [next(iter(s)) if len(s) == 1 else None for s in per]
        if all(ids) and len(set(ids)) == len(ids):
            return [_prose_record("ref", i, t) for i, t in zip(ids, texts)]
    return None


def _prose_record(key: str, ident: str, text: str) -> dict[str, Any]:
    """A listing written as plain sentences: its id, its name (the words before the id, else the first sentence), the
    text itself, and any "Key: value" pairs in it as attributes."""
    name = text.split(ident, 1)[0].strip(" (#[:-") or re.split(r"(?<=[.!?])\s", text.strip(), 1)[0]
    facts = _prose_facts(text)
    return {key: ident, "name": name[:120], "page_text": text, **({"attributes": facts} if facts else {})}


def _prose_facts(text: str) -> dict[str, str] | None:
    """"Colour: teal. Height: 30 cm." inside plain sentences -> those pairs (two or more)."""
    found = re.findall(r"(?:^|[.;\n]\s*)([A-Za-z][\w /-]{1,30}?):\s*([^.;\n]{1,80})", text)
    return {k.strip(): v.strip() for k, v in found} if len(found) >= 2 else None


def _unpack(node: Any, depth: int = 0) -> Any:
    """A tool result with every listing written as text turned into records, so the code reads them as it reads records
    and the model is shown each field in full rather than one long line cut short."""
    if depth > 6:
        return node
    if isinstance(node, str):
        got = _text_records(node)
        if got is None:
            return node
        # JSON inside a string may itself hold text listings
        return _unpack(got, depth + 1) if node.strip()[:1] in "{[" else got
    if isinstance(node, dict):
        # a listing the store sent as a record already: its texts (descriptions, seller notes) stay as written
        if len(node) >= 3 and _coded(node):
            return node
        return {k: _unpack(v, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        out = [_unpack(v, depth + 1) for v in node]
        texts = [v for v in out if isinstance(v, str)]
        if len(out) >= 2 and len(texts) == len(out):
            return _prose_rows(texts) or out
        return out
    return node


def _holds_list(node: Any, depth: int = 0) -> bool:
    """Whether a result carries a non-empty list anywhere near its top (rows of some shape)."""
    if depth > 3:
        return False
    if isinstance(node, list):
        return bool(node)
    return isinstance(node, dict) and any(_holds_list(v, depth + 1) for v in node.values())


def _yn(word: str) -> str:
    return {"true": "yes", "false": "no"}.get(word, word)


def _edit1(a: str, b: str) -> bool:
    """One edit apart: a letter added, dropped or changed, or two neighbouring letters swapped."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def _close(a: str, b: str) -> bool:
    """Two words naming the same thing as far as spelling shows: equal, or both of four or more letters (never
    digits) with one the start of the other ("wood", "wooden") or one edit apart ("grey", "gray"; "color", "colour")."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or not a.isalpha() or not b.isalpha():
        return False
    return a.startswith(b) or b.startswith(a) or _edit1(a, b)


def _key_match(key: str, prop: str) -> bool:
    """An attribute key naming a property: the same words, words that are close spellings of each other, or one
    set of words inside the other."""
    a, b = _stem(key), _stem(prop)
    if not a or not b:
        return False
    if a == b or _names(b, a):
        return True
    return len(a) == len(b) and all(any(_close(x, y) for y in b) for x in a) and all(any(_close(y, x) for x in a) for y in b)


def _tokens(text: Any) -> list[str]:
    """Words and numbers of a value for matching a description against a listing (decimals kept, plural s cut)."""
    return [w[:-1] if w.isalpha() and len(w) > 3 and w.endswith("s") else w for w in re.findall(r"\d+(?:\.\d+)?|[a-z]+", str(text).lower())]


def _num_eq(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def _num_in(number: str, unit: str, words: list[str]) -> bool:
    """Whether `words` show this number with this unit word after it (or with no word of a unit after it, the unit
    then being the attribute's own: "rpm: 7200")."""
    for k, have in enumerate(words):
        if have[0].isdigit() and _num_eq(number, have):
            after = words[k + 1] if k + 1 < len(words) else ""
            if not unit or not after.isalpha() or _close(after, unit):
                return True
    return False


def _value_state(value: str, have: list[str], attrs: list[tuple[list[str], list[str]]] = ()) -> int:
    """How one value of a description stands against a listing's own words (its name, short fields and attributes;
    never its free text, where a note can name values the listing does not have): 2 every word is there and every
    number stands with its unit in one attribute (in the one the value names, when it names one), 1 a number is not
    shown so (it may be written in another unit, which only a reader can convert), 0 a word is missing (it may be
    worded otherwise: "autumn-sown" for a sowing season of autumn), -1 the attribute the value names gives another
    number in the same unit word ("ports: 4" against "6 ports")."""
    want = _tokens(value)
    # the attribute the value names: the one whose key words the value holds most of
    named, cover = None, 0
    for key, shown in attrs:
        words = [w for w in key if w not in _LINK_WORDS]
        if words and len(words) > cover and all(any(_close(w, x) for x in want) for w in words):
            named, cover = shown, len(words)
    state = 2
    for place, word in enumerate(want):
        if word in _LINK_WORDS or (place > 0 and want[place - 1][0].isdigit() and word.isalpha()):
            # a word right after a number is its unit: judged with the number
            continue
        if word[0].isdigit():
            unit = want[place + 1] if place + 1 < len(want) and want[place + 1].isalpha() else ""
            pools = [named] if named is not None else [shown for _, shown in attrs] + [have]
            if any(_num_in(word, unit, pool) for pool in pools):
                continue
            if named is not None and unit and any(named[k][0].isdigit() and _close(named[k + 1], unit) for k in range(len(named) - 1)):
                return -1
            state = 1
        elif not any(_close(word, h) for h in have):
            return 0
    return state


def _sentences(record: Any, least: int = 8) -> list[str]:
    """The free text of a record: strings of eight or more words, at any depth (a seller's notes live there); with a
    smaller `least`, also a shorter string written as a sentence (a one-line rule: "its oak top is paired with steel")."""
    found: list[str] = []

    def walk(node: Any, depth: int) -> None:
        words = len(node.split()) if isinstance(node, str) else 0
        if words >= 8 or (words >= least and node.rstrip().endswith((".", "!", "?"))):
            found.append(node)
        elif isinstance(node, dict) and depth < 4:
            for key, value in node.items():
                if key != "desk note":
                    walk(value, depth + 1)
        elif isinstance(node, list) and depth < 4:
            for value in node:
                walk(value, depth + 1)

    walk(record, 0)
    return found


# how a pointing requirement uses the other listing's value (REFER's reading)
RELATIONS = ("same", "pair", "unlike", "at_most", "at_least", "within", "other")


# the value a listing's property carries while the listing its seller sends that property to is still unread
PENDING = "(to be read from another listing)"


# words that carry no value of their own inside a requirement ("at least", "the", "for") and words that turn a
# requirement into an exclusion or a comparison, which plain word matching cannot judge
_LINK_WORDS = frozenset("a an the of for with and or in on to be is are at least most more less than any some".split())
_NOT_WORDS = frozenset("not no non unlike except without other different anything but bar avoid never excluding same".split())


def _bag(text: Any) -> frozenset[str]:
    """The words of a value, lower case, plural s cut, punctuation dropped."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower()))


def _key_stats(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per attribute key in plain words: how many listings carry it, its distinct values in read order, and the value
    each listing (by its place in `blocks`) shows."""
    stats: dict[str, dict[str, Any]] = {}
    for place, block in enumerate(blocks):
        done: set[str] = set()
        for key, value in (block or {}).items():
            name = _words(key)
            if not name or name in done or value == PENDING:
                continue
            done.add(name)
            slot = stats.setdefault(name, {"n": 0, "values": [], "per": {}})
            slot["n"] += 1
            text = str(value).strip().lower()
            slot["per"][place] = text
            if text not in slot["values"]:
                slot["values"].append(text)
    return stats


def _same_property(name: str, slot: dict[str, Any], other: str, spec: dict[str, Any]) -> bool:
    """Two keys that are one property written two ways: their values agree on every listing carrying both, and the
    names are the same words spelled differently ("graphic card series", "graphics card series"), or one name sits
    inside the other on listings that carry both ("system memory", "system memory new")."""
    shared = [p for p in slot["per"] if p in spec["per"]]
    if any(slot["per"][p] != spec["per"][p] for p in shared):
        return False
    a, b = name.split(), other.split()
    if len(a) == len(b) and all(any(_close(x, y) for y in b) for x in a) and all(any(_close(y, x) for x in a) for y in b):
        return True
    return bool(shared) and (set(a) < set(b) or set(b) < set(a))


def _merged(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """_key_stats with each property written two ways kept once, under the name more listings carry (its own counts:
    a second spelling only leaves, so no other key loses its place): duplicate spellings would otherwise take two of
    the planner's few slots."""
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(stats, key=lambda n: -stats[n]["n"]):
        if not any(_same_property(kept, out[kept], name, stats[name]) for kept in out):
            out[name] = stats[name]
    return {name: out[name] for name in stats if name in out}


def _askable(slot: dict[str, Any]) -> bool:
    """A key worth a question: not free text (long values), not a list of purchase variants, not an identifier
    (every listing a different long value)."""
    values = list(slot["per"].values())
    if not values:
        return False
    mean = sum(len(v) for v in values) / len(values)
    if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
        return False
    return not (len(values) >= 4 and len(set(values)) == len(values) and mean > 12)


def rank_keys(blocks: list[dict[str, Any]], settled: list[str]) -> list[tuple[str, float]]:
    """The attribute keys of one item's listings, ranked for asking the shopper (score = share of listings carrying
    the key). Order: keys on more of the listings first; among keys on as many, keys whose values are words before
    numeric ones, then the key whose values split the listings more evenly (1 - sum of squared value shares), then
    the first seen. Measured on four races' traces, this put a held-back need's key in the first three more often
    than coverage alone or any split-first order. A key naming a property the request already settles, and a second
    key with the same values as a shorter one, are left out; one property written two ways counts once (_merged)."""
    stats = _merged(_key_stats(blocks))
    total = max(1, len(blocks))
    taken = [_stem(m.split(":", 1)[0]) for m in settled if ":" in m]
    kept: list[tuple[str, dict[str, Any]]] = []
    for name, slot in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        words = _stem(name)
        if any(words and words == p for p in taken) or not _askable(slot):
            continue
        if any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept):
            continue
        kept.append((name, slot))
    order = {name: place for place, name in enumerate(stats)}

    def standing(entry: tuple[str, dict[str, Any]]) -> tuple[int, float, float, int]:
        name, slot = entry
        values = list(slot["per"].values())
        numeric = sum(1 for v in values if re.search(r"\d", v)) / len(values)
        even = 1 - sum((values.count(v) / len(values)) ** 2 for v in set(values))
        return -slot["n"], numeric, -even, order[name]

    return [(name, round(slot["n"] / total, 3)) for name, slot in sorted(kept, key=standing)]


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Buyer:
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
        self.t0 = time.monotonic()
        self.deadline = self.t0 + TASK_SECONDS - CLOCK_MARGIN
        self.first_at: float | None = None
        # per-task caps, set from the step limit once the request is read
        self.loop_cap = LOOP_CALLS
        self.call_cap = ALL_CALLS
        self.spend_cap = SPEND_HARD
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
        self.code_held = False
        self.offer_seen: dict[str, int] = {}
        self.reread_state: dict[str, tuple[str, int]] = {}
        self.facts_read: dict[str, int] = {}
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
        self.decline_refused_at = -1
        self.searched_at = -1
        self.after_note_held = False
        self.early_holds = 0
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
        # pages read for notes that send a property elsewhere, and what they sent: (listing, property, other listing)
        self.page_read: set[str] = set()
        self.redirects: dict[str, tuple[str, str]] = {}
        self.repairs = 0
        self.carted_at: dict[str, int] = {}
        self.need_at: dict[str, int] = {}
        self.ending = False
        self.drop_hold = False
        self.price_floor: float | None = None
        # notes that send a property to another listing, followed by the desk: chains by "source|property" (the source
        # a listing id or a requirement), the values they give a listing's properties, pairing rules by property
        self.chains: dict[str, dict[str, Any]] = {}
        self.effective: dict[str, dict[str, tuple[str, list[str], str]]] = {}
        self.pair_rules: dict[str, list[tuple[str, str]]] = {}
        self.ref_relation: dict[str, str] = {}
        self.ref_tol: dict[str, str] = {}
        self.title_seen: tuple[Any, str | None] = (None, None)
        self.note_seen: dict[str, set[str]] = {}
        self.need_scanned: set[str] = set()
        self.chain_said: set[str] = set()
        self.chain_named: set[str] = set()
        self.chain_blocks = 0
        self.search_seen: dict[str, dict[str, Any]] = {}
        self.item_search: dict[int, tuple[str, dict[str, Any]]] = {}   # per item: the model's search that found its kind
        self.search_vals: dict[str, set[str]] = {}   # per tool and free-text argument: the values the model gave it
        self.list_mode = True             # questions name several properties until a reply wants one at a time
        self.search_words: dict[str, list[int]] = {}
        self.desk_queries: set[str] = set()
        self.next_page: tuple[str, dict[str, Any]] | None = None
        # the review of an ending without a purchase: verified breaks by listing, at which version, holds so far;
        # items a search naming them found no listing of
        self.review_breaks: dict[str, tuple[str, str]] = {}
        self.review_failed = False
        self.review_at = -1
        self.review_blocks = 0
        self.review_rows: list[str] = []
        self.review_mark = ""
        self.review_calls = 0
        self.near_checked = False
        self.empty_search: set[int] = set()
        # the desk's quick store reads (no model turn between them), kept apart from the pace of model moves
        self.quick = 0
        self.quick_steps = 0
        self.quick_secs = 0.0

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def pace(self) -> float:
        """Seconds one store step has taken so far, model turns included (6 s until three steps are done). The desk's
        own quick reads (no model turn) are left out of the mean: the moves still to come are mostly model moves."""
        steps = self.step - self.quick_steps
        if self.first_at is None or steps < 3:
            return 6.0
        return max(1.5, (time.monotonic() - self.first_at - self.quick_secs) / steps)

    def room(self) -> int:
        """Moves left: the steps left, or fewer when the clock allows fewer at the pace so far. Every reserve and
        every "few steps left" decision reads this, so a slow episode finishes in time instead of being cut off."""
        return max(0, min(self.left(), int((self.deadline - time.monotonic()) / self.pace())))

    def cutoff(self) -> bool:
        """Past the point where only the episode's final call may still go out (ten seconds before the limit)."""
        return time.monotonic() > self.t0 + TASK_SECONDS - 10

    def recovery(self) -> int:
        """Moves the first checkout should leave for answering what it brings (a sold-out line, a new budget, a
        changed price): a few more for each basket line, at most three tenths of the task's steps."""
        return min(max(7, 4 + 3 * len(self.items)), max(7, self.max_steps * 3 // 10))

    def commit_point(self) -> int:
        """Moves left at which the first checkout is due: the recovery reserve, plus a cart write and a page for each
        item, plus the finishing step(s)."""
        return self.recovery() + 2 * len(self.items) + self.tail()

    def checkout_room(self) -> int:
        """Moves a checkout needs from an empty cart: a cart call per line, the order, and three more to answer the
        notice the first checkout usually brings (read the change, swap a line, order again)."""
        return len(self.items) + 4

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
              effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
            # no call starts after the deadline: the margin is the forced finish's, and a slow reply cannot use it
            if self.calls >= self.call_cap or time.monotonic() > self.deadline:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, effort)
                if message is not None:
                    self.misses = 0
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most; no wait runs past the deadline
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL
                        or time.monotonic() + WAITS[tries] > self.deadline):
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   effort: str | None = None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content).
        `effort` asks the heavy model for less hidden reasoning (its replies' main cost) on calls whose reading the
        code checks anyway."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if effort and model == HEAVY:
            body["reasoning"] = {"effort": effort}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        client = _llm_short if self.deadline - time.monotonic() < 60 else _llm
        result = client.post_verbose("/inference/chat/completions", body)
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

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, None, effort)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= self.call_cap:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if self.first_at is None:
            self.first_at = time.monotonic()
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        began = time.monotonic()
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data and name != self.roles.get("ask"):
                    # the shopper's replies stay as said
                    data["observation"] = self.pin_page(name, args, _unpack(data["observation"]))
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                if self.quick:
                    self.quick_steps += 1
                    self.quick_secs += time.monotonic() - began
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that); once the id key is
                    # known only the ids the reply shows as records count, so a refused side-by-side call opens none
                    shown = self.listed_ids(seen) if self.id_key else None
                    asked = [v.strip() for value in args.values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
                    if shown is not None and not shown and len(asked) == 1:
                        # one page whose reply carries no record of the usual shape (and no error) still showed it
                        shown = None
                    now = {i for i in asked if shown is None or i in shown} - self.opened
                    self.opened.update(now)
                    # a listing read for notes before its page was open had only its attributes and row text then:
                    # the page's notes are read when it arrives
                    self.page_read.difference_update(now)
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
        # room for the light model's thinking: a reply cut at its limit sends the brief to the heavy model for nothing
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
            self.reserve = self.held_back()

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
            # the desk's own choices among read fits go by price (by_price); the model is not sent hunting for cheaper
            # listings (measured: it re-searched, ran out of steps and bought listings that did not fit)
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
            when = {"choose": "before choosing", "order": "at the order (the desk does it when you check out)", "end": "at an ending without a purchase (the desk does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        cart = self.cart_ids() if self.chains else []
        for chain in self.chains.values():
            if chain["source"] in cart or (chain["source"] in self.leads and chain["state"] == "done"):
                parts.append("listing " + self.chain_line(chain))
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        left = self.room()
        text = f"[desk] step {self.step} of {self.max_steps} used, {self.left()} left"
        text += f" (the clock allows about {left} more moves)." if left < self.left() else "."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= self.commit_point():
            # the checkout's own notices come at the first attempt: it has to leave steps to answer them
            text += (" Time to check out: the first checkout often brings a store notice (a sold-out line, a new budget) that takes"
                     f" steps to answer, so put your best fitting listing for each item in the cart and check out now, while {left} moves are left.")
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
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
                    if ident in self.effective:
                        # a fresh copy of the record brings back its own value: the chain's value goes on it again
                        self.apply_effective(ident)
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

    def kind_fits(self, label: Any, item: Item) -> bool:
        """Whether a category label names a requested item: its store label or name, or a long word shared with them."""
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        return _words(str(label)) in (_words(item.shelf or ""), _words(item.name)) or bool({w for w in _stem(str(label)) if len(w) >= 4} & long)

    def key_table(self) -> list[tuple[str, str, int, int, int, list[str], int]]:
        """(category, key name, listings carrying it, listings read, distinct values, sample values, item number) for
        the categories that match a requested item, each category's keys ranked by rank_keys over its read listings."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            if ident in self.full:
                label = str(self.full[ident].get(cat_key) or "listings") if cat_key else "listings"
                groups.setdefault(label, []).append(ident)
        owners = {label: next((n for n, item in enumerate(self.items, 1) if self.kind_fits(label, item)), 0) for label in groups}
        if any(owners.values()):
            groups = {label: ids for label, ids in groups.items() if owners[label]}
        table = []
        for label, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            number = owners.get(label) or 0
            musts = self.items[number - 1].musts if number else [m for item in self.items for m in item.musts]
            blocks = [_fact_block(self.full[i]) or {} for i in ids]
            stats = _merged(_key_stats(blocks))
            for name, _ in rank_keys(blocks, musts)[:14]:
                slot = stats[name]
                table.append((label, name, slot["n"], len(blocks), len(slot["values"]), [v[:40] for v in slot["values"][:4]], number))
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any(self.kind_fits(label, item) for item in self.items)]
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
        if isinstance(value, str) and _count(value) is not None:
            # a count a text listing wrote out
            value = _count(value)
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
                    if self.id_key and _id_like(value.get(self.id_key), True):
                        # one listing wrapped under a key of the result
                        found.append((value[self.id_key].strip(), value))
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def pin_page(self, name: str, args: dict[str, Any], observation: Any) -> Any:
        """A single-listing view the store wrote as plain sentences (no record in it): the text that names the listing
        the call asked for becomes that listing's record, with any "Key: value" pairs in it as attributes, so the page
        counts as read."""
        target = self.single_target(name)
        asked = str(args.get(target) or "").strip() if target else ""
        jobs = ("cart", "buy", "refuse", "ask", "search")
        if not asked or not _id_like(asked, True) or name in [self.roles.get(j) for j in jobs] or self.records_in(observation):
            return observation

        def record(text: str) -> dict[str, Any]:
            return _prose_record(self.id_key or "ref", asked, text)
        def page(text: Any) -> bool:
            # a page, not a short notice that names the listing ("q-2 has just sold out")
            return isinstance(text, str) and asked in text and len(text.split()) >= 12 and (len(text) >= 160 or bool(_prose_facts(text)))
        if page(observation):
            return record(observation)
        if isinstance(observation, dict) and not any("error" in str(k).lower() for k in observation):
            hits = [k for k, v in observation.items() if page(v)]
            if len(hits) == 1:
                return {**observation, hits[0]: record(observation[hits[0]])}
        return observation

    def single_record(self, observation: Any) -> tuple[str, dict[str, Any]] | None:
        """The one listing a single-listing view showed: the result itself, or one record wrapped under a key (a row
        list holding one listing is a search result, not a view); None otherwise."""
        if not isinstance(observation, dict):
            return None
        ident = self.ident(observation)
        if ident:
            return ident, observation
        wrapped = [(v[self.id_key].strip(), v) for v in observation.values()
                   if isinstance(v, dict) and self.id_key and _id_like(v.get(self.id_key), True)]
        return wrapped[0] if len(wrapped) == 1 else None

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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
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
        """A search that matched more listings than it showed, under a cheapest, most or least pick: the desk reads a short
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
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.room() - self.tail() - pages < 12:
            return (f"\n[desk] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[desk read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages the desk read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[desk] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.room() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.room()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        self.quick += 1
        try:
            return self.read_facts(ids, keep)
        finally:
            self.quick -= 1

    def read_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        # a listing read once while the requirements stood as they are is not read again for attributes it did not show
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i) and self.facts_read.get(i) != self.version]:
            if self.room() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.facts_read[ident] = self.version
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            record = self.single_record(seen)
            if record:
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(record[1]))
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
        """Absorb a result of the desk's own call and read any note in it; returns what the note changed."""
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
        if self.room() <= 12:
            return
        # the size the store has shown to work (a schema limit, else learned from failed side-by-side calls)
        cap = self.compare_room()
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
            if len(self.full) == before and len(picks) > 2 and self.room() > 12:
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        seen = reply.get("observation")
                        self.observe(tool, seen)
                        self.note_view(tool, seen)
                        rows = self.records_in(seen)
                        if rows:
                            self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
                        if self.facts_from.get(tool) is False or len(self.full) == before:
                            # the smaller side-by-side call showed no attributes either: read them singly instead
                            self.widen_singly()

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        if not tool or not self.single_target(tool):
            return
        # each item group up to four read listings (the planner's key counts are reliable from about there), the
        # groups taking turns; with a cheapest pick the cheapest buyable rows first. A one-item task (and a long task)
        # has the moves to read more of the shelf: a held-back need's key ranks truer on more read listings
        # (measured: top-4 by coverage 86% on a whole shelf, 67% on ten listings)
        wide = len(self.items) == 1 or self.max_steps >= 60
        goal, more, cap = (8, 7, 10) if wide else (4, 3, 6)
        lanes: list[list[str]] = []
        short: list[str] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            short += pool[: max(0, 2 - opened)]
            lanes.append(pool[: max(0, min(more, goal - opened))])
        if len(self.items) > 1 and short:
            # a basket line with fewer than two read listings gets them even below the spare-move floor: a held-back
            # need's key is in the planner's table on 0% of lines with no own read, 88% with one to three
            self.fill_facts(short, self.commit_point() + self.msgs_free())
            lanes = [[i for i in lane if i not in self.full] for lane in lanes]
        queue = [lane[k] for k in range(more) for lane in lanes if k < len(lane)]
        # only from spare moves: the questions, the commit point and a few moves to cart stay untouched
        self.fill_facts(queue[:cap], max(12, self.commit_point() + self.msgs_free() + 4))

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.room() <= 8:
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

    def name_musts(self) -> None:
        """A word (or two) of an item's name that is the value of one attribute on the item's read listings is a
        requirement the request states, even when the reading of the request left it inside the name ("brass" in
        "brass desk lamp" while the listings' finish is brass, chrome or matte black). It is added as a
        requirement, so the checks hold it and no question asks about that property."""
        for number, item in enumerate(self.items, 1):
            ids = [i for i in self.seen_order if i in self.full and self.of_item(i, number)]
            if len(ids) < 3:
                continue
            cat_key, name_key = self.category_key(), self.title_key()
            label = set(_stem(item.shelf or "")) | {w for i in ids for w in _stem(str(self.full[i].get(cat_key) or ""))}
            titles = [_stem(str(self.full[i].get(name_key) or "")) for i in ids]
            words = [w for w in _words(item.name).split() if len(w) >= 3 and not w.isdigit()]
            # the kind itself (its label, the noun most titles carry) is no requirement
            words = [w for w in words if not ({w} & label or _stem(w) & label) and sum(1 for s in titles if _stem(w) <= s) < 0.5 * len(titles)]
            phrases = [frozenset(_stem(w)) for w in words] + [frozenset(_stem(a + " " + b)) for a, b in zip(words, words[1:])]
            live = [m for m in item.musts + item.told if m not in item.dropped]
            for phrase in sorted(set(phrases), key=len, reverse=True):
                if not phrase or any(phrase <= _stem(m.split(":", 1)[-1]) for m in live):
                    continue
                hits: dict[str, int] = {}
                for key in {k for i in ids for k in _facts(self.full[i])}:
                    seen = [str(_flag(_facts(self.full[i]).get(key))) for i in ids if key in _facts(self.full[i])]
                    parts = [_stem(x) for v in seen for x in re.split(r"[,/;]", v) if x.strip()]
                    # a property whose value varies among the item's listings, one of them the name's word
                    if len({_stem(v) for v in seen}) >= 2 and phrase in parts:
                        hits[key] = sum(1 for x in parts if x == phrase)
                ranked = sorted(hits.items(), key=lambda kv: -kv[1])
                if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
                    continue
                key = ranked[0][0]
                if any(_key_match(key, m.split(":", 1)[0]) or _key_match(m.split(":", 1)[0], key) for m in live):
                    continue
                need = f"{key}: {' '.join(w for w in words if _stem(w) <= phrase)}"
                item.musts.append(need)
                live.append(need)
                self.version += 1
                self.trace.append({"kind": "name_must", "item": number, "need": need})

    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        began = self.step
        self.widen()
        self.name_musts()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, read, distinct, values, _) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r} ({read} listings that fit the request so far):")
                last = label
            listing.append(f"  {index}. {name} (on {carried}, {distinct} different value{'s' if distinct != 1 else ''}): " + " | ".join(values))
        # one pick more than the messages allow: the last message goes to it when no answer needs a second try
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 2600, "low") or {}
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
        if not ordered:
            # no plan came back (a failed call): the table's own ranking, the items in turn, rather than no question
            ordered = self.table_picks(table)
        else:
            ordered = self.on_table(ordered, table, count + 1)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "order": ordered})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # one message at a time, each chosen after the last reply. Every item gets a message before any gets a second,
        # then the items that have given no need. A message leads with the item's next pick in the plan's order (what a
        # one-property question would ask) and names one more of the item's properties not named yet: a shopper who
        # answers only the first property still gets today's question, and one who takes one question at a time refuses
        # long lists far more often than a pair. A reply that wants one question at a time ends the pairs.
        lanes = self.lanes(ordered, table)
        led: dict[int, int] = {}
        named: dict[int, list[str]] = {}
        while self.msgs_free() > 0 and len(sent) < count and not self.finished:
            number = self.next_lane(lanes, led)
            if number is None:
                break
            item = self.items[number - 1]
            lane = lanes[number]
            lead = lane[led.get(number, 0)]
            led[number] = led.get(number, 0) + 1
            if self.said_about(number, lead):
                continue
            seen = named.setdefault(number, [])
            fill = [x for x in lane if x != lead and x not in seen and not self.said_about(number, x)] if self.list_mode else []
            text, fill = self.list_question(number, lead, fill[: LIST_PROPS - 1])
            batch = self.send([(number, lead, text)])
            if not batch:
                break
            seen.extend([lead] + fill)
            sent += batch
            heard = self.hear(batch)
            if fill:
                # a no-preference reply to a list says nothing about the properties it does not name ("size class" in a
                # reply names that property, not "size")
                reply = _words(batch[0][2])
                free = [e for e in _items(heard.get("free")) if isinstance(e, dict)]
                longest = sorted({_words(x) for x in [lead] + fill} | {_words(str(e.get("property") or "")) for e in free}, key=len, reverse=True)

                def names(prop: str) -> bool:
                    text = " " + reply + " "
                    for other in longest:
                        if len(other) > len(prop) and prop in other:
                            text = re.sub(rf"(?<![a-z0-9]){re.escape(other)}(?![a-z0-9])", " ", text)
                    return bool(prop) and re.search(rf"(?<![a-z0-9]){re.escape(prop)}(?![a-z0-9])", text) is not None
                heard["free"] = [e for e in free if names(_words(str(e.get("property") or "")))]
            self.take_heard(heard, told)
            if _items(heard.get("crowded")):
                self.list_mode = False
                if self.msgs_free() > 0 and len(sent) < count:
                    # the reply wants a single question: the lead property once more, plainly worded
                    again = self.send([(number, lead, f"Which {lead} do you want for the {item.name}?")])
                    if again:
                        sent += again
                        self.take_heard(self.hear(again), told)
        if not sent:
            return ""
        if told:
            self.version += 1
        searched = self.value_search([(n, need) for n, need in self.needs_now() if self.need_at.get(need, -1) >= began and (n, need) not in self.refs])
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "The interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
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

    def value_search(self, new: list[tuple[int, str]]) -> str:
        """Right after the interview, each item that gained a stated value gets one look of the desk's own: rows seen
        earlier whose text shows the value, and (unless two such rows are known) one search, the value's words with
        the item's name, shaped like the model's own searches. Up to three unread rows of that item's kind that can
        be bought, those whose own text shows the value first (cheaper first on a tie under a price ranking), get their
        attributes read, so the next pick weighs listings that carry the told value, not only
        the ones found before it was told. Store calls only, one search per item; skipped when moves are short or when
        a read, buyable listing of the item already shows the value."""
        keep = self.checkout_room() + 8
        lines: list[str] = []
        done: set[int] = set()
        for number, need in new:
            if number in done or self.finished or self.cutoff() or self.room() <= keep + 4:
                continue
            value = need.split(":", 1)[-1]
            # a value stated as an exclusion is no search term
            if re.search(r"\b(?:not|no|never|without|except|other than|anything but)\b", value.lower()):
                continue
            words = [w for w in re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", value) if w.lower() not in _LINK_WORDS]
            if not words or len(words) > 6:
                continue
            # a read listing of the item that can be bought already shows the value: no search is needed for it
            if any(i in self.full and self.buyable(i) and self.of_item(i, number) and self.verdict(self.full[i], need) is True for i in self.seen_order):
                continue
            item = self.items[number - 1]
            wanted = [w for w in _tokens(value) if w not in _LINK_WORDS]

            def shows(ident: str) -> tuple[int, bool, float]:
                have = self.text_of(ident)
                price = self.price_of(ident) if self.by_price() else None
                return -sum(1 for w in wanted if any(_close(w, h) for h in have)), price is None and self.by_price(), price or 0.0
            # rows already seen in earlier searches whose own text shows the value come first: they cost no search,
            # and a listing that showed up but was never opened is the commonest miss
            known = [i for i in self.seen_order if i not in self.full and self.buyable(i) and self.of_item(i, number) and shows(i)[0] < 0]
            rows = list(known)
            if len(known) < 2:
                # inside the item's own search with its category filter kept, the value's words alone; otherwise with the
                # item's name. Under a cheapest pick the cheaper rows showing the value may sit on later pages: up to two
                # more are read while moves allow
                template = self.item_search.get(number)
                props = (self.schemas.get(template[0]) or {}).get("properties") or {} if template else {}
                # a filter: besides the free words, another free-text argument the model set (a category, a shelf)
                filtered = sum(1 for k, v in (template[1] if template else {}).items()
                               if isinstance(v, str) and v.strip() and isinstance(props.get(k), dict) and "enum" not in props[k]) >= 2
                seen = self.desk_search(" ".join(words) + ("" if filtered else " " + item.name), template)
                pages = 0
                while seen is not None:
                    rows += [i for i, _ in self.records_in(seen) if i not in self.full and self.buyable(i) and self.of_item(i, number)]
                    if self.brief.get("pick") != "cheapest" or pages >= 2 or not self.next_page or self.room() <= keep + 8 or self.finished:
                        break
                    tool, args = self.next_page
                    reply = self.act(tool, args)
                    seen = reply.get("observation") if isinstance(reply, dict) else None
                    if seen is None:
                        break
                    self.observe(tool, seen)
                    plan = self.paging(tool, seen)
                    self.next_page = (tool, {**args, plan[2]: plan[3]}) if plan and plan[3] is not None else None
                    pages += 1
            if not rows:
                continue
            done.add(number)
            picks = sorted(dict.fromkeys(rows), key=shows)[:3]
            read = self.fill_facts(picks, keep) if picks else []
            self.trace.append({"kind": "value_search", "item": number, "q": " ".join(words), "rows": len(rows), "read": [t.split(":", 1)[0] for t in read]})
            lines.append(f"[the desk looked for {' '.join(words)!r} for item {number}; " + (f"attributes read (one step each):\n" + "\n".join(t[:700] for t in read) if read else "no unread listing of that item came up]"))
        return "\n".join(lines)

    def on_table(self, ordered: list[tuple[int, str]], table: list[tuple[str, str, int, int, int, list[str], int]], count: int) -> list[tuple[int, str]]:
        """The plan kept to the listings' own properties: a pick naming a key of its item's table is asked in that key's
        words; one that names no key is replaced by that item's next table key, unless the request itself names it or
        the table has fewer keys than the questions (measured: questions off the table drew 0 of 146 held-back needs)."""
        said = _stem(self.request)
        out: list[tuple[int, str]] = []
        for number, label in ordered:
            keys = [name for _, name, _, _, _, _, n in table if n in (number, 0)]
            hits = [k for k in keys if _key_match(k, label)]
            if hits:
                # asked in the store's own words for that property: the key spelled the same, else the one key it names
                same = [k for k in hits if _stem(k) == _stem(label)]
                named = same[0] if same else hits[0] if len(hits) == 1 else label
                if (number, named) not in out:
                    out.append((number, named))
                continue
            if len(keys) < count or (_stem(label) and _stem(label) <= said):
                out.append((number, label))
                continue
            spare = next((k for k in keys if not any(_key_match(k, lab) for n, lab in ordered + out if n == number)), None)
            if spare:
                out.append((number, spare))
        return out

    def lanes(self, ordered: list[tuple[int, str]], table: list[tuple[str, str, int, int, int, list[str], int]]) -> dict[int, list[str]]:
        """Each item's properties to ask about, best first: the plan's picks for it, then the keys of its table (the
        store's own spellings, in the table's ranking). A property the request settles, or one already in the lane under
        another spelling, is left out."""
        out: dict[int, list[str]] = {}
        for number, label in list(ordered) + self.table_picks(table):
            if not 1 <= number <= len(self.items):
                continue
            item = self.items[number - 1]
            lane = out.setdefault(number, [])
            gone = lane + [m.split(":", 1)[0] for m in item.musts if m not in item.dropped]
            if not any(_words(label) == _words(x) or _key_match(label, x) or _key_match(x, label) for x in gone) and not self.settled(number, label):
                lane.append(label)
        return {n: lane for n, lane in out.items() if lane}

    def settled(self, number: int, label: str) -> bool:
        """The request settles this property under another name: a stated requirement's value is one of the values the
        item's read listings give the property ("shade material: linen" settles a store's "lamp shade")."""
        item = self.items[number - 1]
        wanted = [_stem(m.split(":", 1)[1]) for m in item.musts if ":" in m and m not in item.dropped]
        if not any(wanted):
            return False
        have = {_stem(x) for i in self.seen_order if i in self.full and self.of_item(i, number)
                for k, v in _facts(self.full[i]).items() if _key_match(k, label) or _key_match(label, k)
                for x in re.split(r"[,/;]", str(_flag(v))) if x.strip()}
        return any(w and w in have for w in wanted)

    def next_lane(self, lanes: dict[int, list[str]], led: dict[int, int]) -> int | None:
        """The item the next message is about, while it has a pick left to lead with: one that has given no need before
        one that has, then the one with the fewest messages (so every item is asked once before any is asked twice)."""
        open_ = [n for n, lane in lanes.items() if led.get(n, 0) < len(lane)]
        return min(open_, key=lambda n: (bool(self.items[n - 1].told), led.get(n, 0), n)) if open_ else None

    def said_about(self, number: int, label: str) -> bool:
        """The shopper has already answered about this property of the item: a need told or no preference named."""
        item = self.items[number - 1]
        return any(_key_match(label, m.split(":", 1)[0]) or _key_match(m.split(":", 1)[0], label) for m in item.told + item.free)

    def list_question(self, number: int, lead: str, fill: list[str]) -> tuple[str, list[str]]:
        """The question naming `lead` first and then `fill`, cut to what the message tool takes; the properties it names
        after the lead are returned with it."""
        spec = ((self.schemas.get(self.roles.get("ask") or "") or {}).get("properties") or {})
        limit = min((v.get("maxLength") for v in spec.values() if isinstance(v, dict) and isinstance(v.get("maxLength"), int)), default=0)
        while True:
            names = [lead] + fill
            text = (self.question(number, lead) if len(names) == 1 else
                    f"What {', '.join(names[:-1])} or {names[-1]} would you like for the {self.items[number - 1].name}?")
            if not limit or len(text) <= limit or not fill:
                return text, fill
            fill = fill[:-1]

    def question(self, number: int, label: str) -> str:
        # plain: example values in a question cut the needs it draws (measured on normal shoppers)
        return f"What {label} would you like for the {self.items[number - 1].name}?"

    @staticmethod
    def table_picks(table: list[tuple[str, str, int, int, int, list[str], int]]) -> list[tuple[int, str]]:
        """Questions straight from the ranked key table: each item's best keys, the items taking turns."""
        lanes: dict[int, list[str]] = {}
        for _, name, _, _, _, _, number in table:
            lane = lanes.setdefault(number or 1, [])
            if name not in lane:
                lane.append(name)
        picks = []
        for depth in range(max((len(v) for v in lanes.values()), default=0)):
            picks += [(number, lane[depth]) for number, lane in sorted(lanes.items()) if depth < len(lane)]
        return picks

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.room() <= floor or self.msg_used >= self.msg_limit:
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
            self.qa.append((text, words[:300]))
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
                self.need_at[need] = self.step
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
        """Read the answers: the light model first; the heavy one only when the light reading leaves an answer unplaced
        (no kind, or "other") or names a need it did not write out. Most answers are a plain no-preference, so the
        heavy model's price is paid only where an answer is hard to read."""
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = self.ask_json(prompt, [models[0]], 1500) or {}
        kinds = [str(k).strip().lower() for k in _items(got.get("kinds"))]
        named = sum(1 for k in kinds if k in ("need", "points"))
        placed = len(kinds) == len(sent) and all(k in ("need", "points", "free", "crowded", "unclear") for k in kinds)
        if len(models) > 1 and not self.no_credit and (not placed or named > len(_items(got.get("needs")))):
            got = self.ask_json(prompt, list(models[1:]), 1500) or got
        self.trace.append({"kind": "hear", "kinds": kinds, "placed": placed})
        return got

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
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.room()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            near = self.closest_note() if stage == "end" and self.brief.get("quiet_decline") is not True else ""
            if near:
                # a message at the ending tells the shopper what came closest, rather than asking leave to say it
                prompt += ("\n\nCLOSEST LISTINGS (a message a step needs at this ending states these, with id, price and"
                           " what each misses, and does not ask leave to offer them): " + near)
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
                return False, ("Before going on, the desk did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a
            # message is about the final cart, so its line must come after the cart last changed
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.room() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(desk: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.room()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
                break
            if self.room() < 2 or (name == self.roles.get("ask") and self.msg_used >= self.msg_limit):
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
        return True, ("Before going on, the desk did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
        hint = ""
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            old, self.budget = self.budget, budget
            changed.append(f"budget now {budget}")
            # a new budget can let in a listing set aside for its price: a later ending is reviewed and held afresh
            self.review_blocks = 0
            hint = self.budget_hint(old)
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
            # the cheapest fit can change with a requirement gone: the next checkout looks at cheaper listings first
            self.drop_hold = self.brief.get("pick") == "cheapest"
            self.review_blocks = 0
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
                self.need_at[need] = self.step
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
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "") + hint).strip()

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
        return "lines by the desk's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        lead = ids is not None
        self.name_musts()
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        # the check runs up to the task's cap (the tool loop stopped short of it to leave this money)
        limit = self.spend_cap - 0.002 if limit is None else limit
        if key in self.checked or self.cost > limit or not ids:
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
        got = self.ask_json(prompt, [HEAVY, LIGHT], 2600, "low")
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
        The desk opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        # requirements whose value is on another listing, and cart lines whose own page sends a property elsewhere
        cart = set(self.cart_ids())
        live = {need for _, need in self.needs_now()} | {self.redirect_need(i) for i in self.redirects if i in cart}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < self.spend_cap:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
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
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
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
                    self.relate(need, next((c for c in self.chains.values() if c["source"] == need), None))
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or not target or self.room() < len(reach) + 4 or self.finished:
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
            text += "[desk opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.room() > 5
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
        # once per cart and requirements as they are, unless a refusal came after the last read
        state = (self.cart_mark(), self.version)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone
                 and (self.reread_state.get(i) != state or self.offer_seen.get(i, -99) < self.refused_at)]
        if not stale or self.room() < len(stale) + 3:
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
            self.reread_state[ident] = state
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
        text = f"[desk] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.room() <= 6:
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
        if self.cost > self.spend_cap - 0.004:
            return []
        # only listings of a sold-out line's own item can replace it, shown by the attributes its requirements name
        numbers = {self.item_for(i, True) for i in lost} - {None}
        records = []
        refs = self.references()
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident) or ident in refs:
                continue
            number = next((n for n in numbers if self.of_item(ident, n)), None) if numbers else self.item_for(ident, True)
            if number is None:
                continue
            records.append(self.compact(ident, number) if ident in self.full else "ROW " + self.compact(ident, number)[:200])
            if len(records) >= 15:
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
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.by_price() else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    # ---------- the review of an ending without a purchase ----------
    def of_item(self, ident: str, number: int) -> bool:
        """Whether a listing is of a requested item for the review and the leads: its category label names the item;
        a label naming no item counts only while no listing seen carries a label that names it (labels worded unlike
        the request). A listing read on the way (a chain's, a search's other kinds) is not a candidate."""
        strict = self.item_for(ident)
        if strict is not None:
            return strict == number
        cat_key = self.category_key()
        if cat_key and any(self.kind_fits(row.get(cat_key), self.items[number - 1]) for row in self.rows.values() if row.get(cat_key) is not None):
            return False
        return self.item_for(ident, True) == number

    def references(self) -> set[str]:
        """Listings a pointing requirement's chain passed through that are not to be bought: the requirement only
        describes them, unless one is itself the best buy (see ref_buyable)."""
        return {i for c in self.chains.values() if c["source"] not in self.rows for i in c["path"] if not self.ref_buyable(i)}

    def ref_buyable(self, ident: str) -> bool:
        """A listing a pointing requirement leads to may be bought when it is itself the best: of a requested item,
        every requirement of that item shown met by its own record, and no read listing of the item that could still
        fit (none shown failing, at most one unknown) costs less or has no readable price."""
        number = next((n for n in range(1, len(self.items) + 1) if self.of_item(ident, n)), None)
        price = self.price_of(ident)
        if number is None or ident not in self.full or not self.buyable(ident) or price is None:
            return False
        needs = [need for n, need in self.needs_now() if n == number]
        if not needs or any(self.verdict(self.full[ident], need) is not True for need in needs):
            return False
        for other in self.full:
            if other == ident or not self.buyable(other) or not self.of_item(other, number):
                continue
            marks = [self.verdict(self.full[other], need) for need in needs]
            cost = self.price_of(other)
            if False not in marks and marks.count(None) <= 1 and (cost is None or cost < price):
                return False
        return True

    def review_pool(self, number: int) -> list[str]:
        """The read listings of one requested item that could still be bought: of its kind, buyable, within a budget
        their prices can be read against (a listing a chain passed through counts only of its own kind, and one a
        pointing requirement leads to not at all)."""
        item = self.items[number - 1]
        readable = self.budget is not None and self.units_agree() and self.brief.get("budget_scope") != "each"
        passed = {i for c in self.chains.values() for i in c["path"]}
        refs = self.references()
        pool = []
        for ident in self.seen_order:
            if ident not in self.full or not self.buyable(ident) or not self.of_item(ident, number) or ident in refs:
                continue
            if ident in passed and not self.item_for(ident):
                continue
            price = self.price_of(ident)
            if readable and price is not None and price * item.qty > self.budget:
                continue
            pool.append(ident)
        return pool

    def compact(self, ident: str, number: int | None = None) -> str:
        """A listing for the review in about 300 characters: id, name, price, the attributes that name a property its
        item's requirements are about (the rest decide nothing), and any desk note."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        name = self.title_key()
        facts = _facts(record)
        needs = [need for n, need in self.needs_now() if number is None or n == number]
        props = [need.split(":", 1)[0] for need in needs]
        shown = [f"{k}: {_flag(v)}" for k, v in facts.items() if any(_key_match(k, p) for p in props)]
        missing = [p for p in props if facts and not any(_key_match(k, p) for k in facts)]
        for need in needs:
            prop, _, value = need.partition(":")
            want = {w for w in _bag(value) if w not in _LINK_WORDS and w not in _bag(prop)}
            other = self.value_key(record, prop, lambda v: bool(want) and all(any(_close(w, h) for h in _bag(_flag(v))) for w in want)) if prop in missing else None
            if other:
                # the key that holds this property under another name (its values include the wanted one)
                shown += [f"{k} (holds {prop.strip()}): {_flag(v)}" for k, v in facts.items() if _stem(k) == other]
                missing.remove(prop)
        parts = [ident, str(record.get(name) or "")[:70] if name else "", f"price {self.price_of(ident)}" if self.price_of(ident) is not None else "",
                 "; ".join(shown), "; ".join(f"no attribute for {p}" for p in missing[:3]),
                 ("desk note: " + str(record.get("desk note"))[:160]) if record.get("desk note") else ""]
        return " | ".join(p for p in parts if p)[:420]

    def quote_holds(self, ident: str, need: str, quote: str) -> bool:
        """A claimed break stands only on the listing's own words: every word and number of the quote is in its
        record ("absent": the listing's attributes were read and none names that property)."""
        quote = quote.strip().strip('"').strip()
        if quote.lower() == "absent":
            prop = need.split(":", 1)[0]
            return ident in self.full and _block_key(self.full[ident]) is not None and not any(_key_match(k, prop) for k in _facts(self.full[ident]))
        want = [w for w in _tokens(quote) if w not in _LINK_WORDS]
        record = self.full.get(ident) or {}
        own = [f"{k} {v}" for k, v in _facts(record).items() if _key_match(k, need.split(":", 1)[0]) and v != PENDING]
        if own:
            # the listing's own attribute for that property decides: one that meets the requirement is no break, and
            # a quote from its name, text or notes ("modern" in a title) cannot outweigh it
            if self.verdict(record, need) is True:
                return False
            have = [h for text in own for h in _tokens(text)]
        else:
            have = self.text_of(ident)
        return bool(want) and all(any(_close(w, h) or (w[0].isdigit() and h[0].isdigit() and _num_eq(w, h)) for h in have) for w in want)

    def code_break(self, ident: str, need: str) -> tuple[str, str] | None:
        """A break the code can show by itself, quoting the listing's own attribute: a number against the same unit
        word, or a word value when the store's own vocabulary for that attribute is confirmed (another read listing's
        attribute of the same name holds the wanted value, so a different word is a different value, not a synonym)."""
        record = self.full.get(ident) or {}
        if self.verdict(record, need) is not False:
            return None
        prop = need.split(":", 1)[0]
        key, value = next(((k, v) for k, v in _facts(record).items() if _stem(k) == _stem(prop)), (None, None))
        if key is None or value == PENDING:
            return None
        quote = f"{key}: {_flag(value)}"
        if re.search(r"\d", need.split(":", 1)[-1]):
            return (need, quote)
        heard = any(self.verdict(self.full[o], need) is True for o in self.full if o != ident and any(_stem(k) == _stem(prop) for k in _facts(self.full[o])))
        return (need, quote) if heard else None

    def review_call(self) -> dict[str, tuple[str, str]]:
        """One heavy reading of every read listing the review weighs (36 at most, the items taking turns): the
        verified breaks by listing id."""
        self.name_musts()
        live = {need for _, need in self.needs_now()}
        # a break found before stands while its requirement is in force and the listing's own words still show it;
        # breaks the code can show by itself need no model at all
        kept = {i: (need, quote) for i, (need, quote) in self.review_breaks.items() if need in live and self.quote_holds(i, need, quote)}
        for number in range(1, len(self.items) + 1):
            for ident in self.review_pool(number):
                if ident not in kept:
                    found = next((b for need in [x for n, x in self.needs_now() if n == number] for b in [self.code_break(ident, need)] if b), None)
                    if found:
                        kept[ident] = found
        lanes = [[i for i in self.review_pool(n)[::-1] if i not in kept] for n in range(1, len(self.items) + 1)]
        ids = [lane[k] for k in range(24) for lane in lanes if k < len(lane)][:24]
        # rows of a requested kind never read, within a readable budget: the review may name some worth reading
        readable = self.budget is not None and self.units_agree() and self.brief.get("budget_scope") != "each"
        rows = [i for i in reversed(self.seen_order) if i not in self.full and self.buyable(i)
                and any(self.of_item(i, n) for n in range(1, len(self.items) + 1))
                and not (readable and self.price_of(i) is not None and self.price_of(i) * self.items[(self.item_for(i, True) or 1) - 1].qty > self.budget)][:10]
        # the same listings against the same requirements and budget were weighed already: their breaks stand;
        # two weighings a task at most (the heavy model's tokens are the review's whole cost)
        mark = _dump([ids, rows, sorted(live), self.budget, sorted(self.effective)])
        if mark == self.review_mark or self.review_calls >= 2:
            return kept
        self.review_rows = []
        if not ids:
            # every listing read is ruled out by its own attributes: nothing for a model to weigh. Unread rows worth a
            # step are those whose own words show the most of their item's wanted values (cheapest first on a tie
            # under a cheapest pick); a row showing none is not worth one
            def promise(ident: str) -> tuple[int, float]:
                number = self.item_for(ident, True) or 1
                have = self.text_of(ident)
                wanted = [w for n, need in self.needs_now() if n == number for w in _tokens(need.split(":", 1)[-1]) if w not in _LINK_WORDS and not w[0].isdigit()]
                price = self.price_of(ident) if self.brief.get("pick") == "cheapest" else None
                return sum(1 for w in set(wanted) if any(_close(w, h) for h in have)), -(price or 0.0)
            ranked = sorted(rows, key=promise, reverse=True)
            # rows showing a wanted value first; with none, two unread rows of the item still get a look while moves
            # allow (a row's few words rarely show an attribute value)
            self.review_rows = [i for i in ranked if promise(i)[0] > 0][:4] or (ranked[:2] if self.room() > self.checkout_room() + 2 else [])
            self.review_failed = False
            self.trace.append({"kind": "review", "weighed": 0, "breaks": len(kept), "rows": len(self.review_rows)})
            return kept
        if self.cost > self.spend_cap + 0.01 or self.no_credit:
            self.review_failed = not kept
            return kept
        self.review_mark = mark
        self.review_calls += 1
        transcript = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.qa[-8:]) or "(no questions asked)"
        numbered = list(dict.fromkeys(n for _, n in self.needs_now()))
        reqs = "\n".join(f"R{k} (item {next(n for n, x in self.needs_now() if x == need)}): {need}" + (f" (the page it points at gives: {self.ref_values[need]})" if need in self.ref_values else "")
                         for k, need in enumerate(numbered, 1))
        prompt = (REVIEW + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + (reqs or "(none beyond the kind of product)") + "\n\nTHE TASK:\n" + self.ledger()
                  + "\n\nTRANSCRIPT:\n" + transcript + "\n\nLISTING RECORDS:\n" + ("\n".join(self.compact(i, self.item_for(i, True)) for i in ids) or "(none read)")
                  + "\n\nUNREAD ROWS:\n" + ("\n".join(self.compact(i)[:120] for i in rows) or "(none)"))
        got = self.ask_json(prompt, [HEAVY, LIGHT], 3000, "low") or {}
        if not isinstance(got.get("breaks"), list):
            # no reading came back (a failed call, the money cap): that says nothing about the listings
            self.review_failed = not kept
            self.trace.append({"kind": "review", "weighed": len(ids), "failed": True})
            return kept
        self.review_failed = False
        # the review's own picks, then rows the code finds showing the most of their item's wanted values (a fit seen in
        # a search but never opened is the commonest wrong ending)
        self.review_rows = list(dict.fromkeys([str(i) for i in _items(got.get("open")) if str(i) in rows] + self.promising(rows)))[:4]
        breaks: dict[str, tuple[str, str]] = dict(kept)
        for entry in _items(got.get("breaks")):
            if not isinstance(entry, dict):
                continue
            ident, said, quote = str(entry.get("id") or "").strip(), str(entry.get("req") or entry.get("breaks") or "").strip(), str(entry.get("quote") or "")
            # the requirement by its number (or, read loosely, its words as the ledger writes them)
            index = _count(said.upper().lstrip("R"))
            need = (numbered[index - 1] if index and 1 <= index <= len(numbered) else
                    next((n for n in live if said == n or _words(said) == _words(n) or (said.startswith(n) and len(n) >= 6)), None))
            if ident in ids and need and self.quote_holds(ident, need, quote):
                breaks[ident] = (need, quote.strip())
        self.trace.append({"kind": "review", "weighed": len(ids), "breaks": len(breaks), "claims": len(_items(got.get("breaks")))})
        return breaks

    def promising(self, rows: list[str]) -> list[str]:
        """Unread rows of an item worth a step before an ending without a purchase: those whose own words show any of
        their item's wanted values (stated or told), the most first, cheaper first on a tie under a price ranking; three
        at most."""
        def score(ident: str) -> tuple[int, float]:
            number = self.item_for(ident, True) or 1
            have = self.text_of(ident)
            wanted = {w for n, need in self.needs_now() if n == number for w in _tokens(need.split(":", 1)[-1]) if w not in _LINK_WORDS and not w[0].isdigit()}
            price = self.price_of(ident) if self.by_price() else None
            return sum(1 for w in wanted if any(_close(w, h) for h in have)), -(price or 0.0)
        ranked = sorted((i for i in rows if i not in self.full and self.buyable(i)), key=score, reverse=True)
        return [i for i in ranked if score(i)[0] > 0][:3]

    def mark_empty(self, args: dict[str, Any], observation: Any) -> None:
        """The items a search named whose kind it showed no listing of. Rows the desk cannot read as listings are no
        proof the search found none: only read rows, or a result with no rows at all, mark an item."""
        said = " ".join(v for v in args.values() if isinstance(v, str)).lower()
        found = self.records_in(observation)
        if not found and _holds_list(observation):
            return
        kinds = {self.item_for(i, True) for i, _ in found}
        for number, item in enumerate(self.items, 1):
            if number not in kinds and any(len(w) >= 4 and w in _stem(said) for w in _stem(item.name)):
                self.empty_search.add(number)

    def decline_proof(self) -> bool:
        """Some requested item is shown to have no fitting listing: four or more of its buyable listings read in full,
        and each breaks a requirement by its own record (a code verdict or a quote-checked break, never an unknown) or
        breaks a budget its price can be read against."""
        readable = self.budget is not None and self.units_agree() and self.brief.get("budget_scope") != "each"
        for number, item in enumerate(self.items, 1):
            needs = [need for n, need in self.needs_now() if n == number]
            read = [i for i in self.seen_order if i in self.full and self.of_item(i, number) and self.buyable(i)]
            if len(read) < 4:
                continue

            def broken(ident: str) -> bool:
                price = self.price_of(ident)
                if readable and price is not None and price * item.qty > self.budget:
                    return True
                return ident in self.review_breaks or any(self.verdict(self.full[ident], need) is False for need in needs)
            if all(broken(i) for i in read):
                return True
        return False

    def explored(self, number: int) -> bool:
        """An item has been looked for when two or more of its listings were read in full, or a search naming it
        found no listing of its kind."""
        read = sum(1 for i in self.full if self.item_for(i, True) == number and (i in self.opened or self.facts_tool()))
        return read >= 2 or number in self.empty_search

    def measure_of(self, ident: str, measure: str) -> float | None:
        value = next((v for k, v in _facts(self.full.get(ident) or {}).items() if _key_match(k, measure)), None)
        found = re.search(r"\d+(?:\.\d+)?", str(value)) if value is not None else None
        return float(found.group()) if found else None

    def by_price(self) -> bool:
        """Whether listings that fit are ranked by price: under a cheapest pick, and when the request names no other
        priority (a store weighing several fits takes the cheapest); a most or least pick ranks by its measure. The
        cheapest pick's own extra searching (paging, cheaper rows) stays with an asked-for cheapest."""
        return self.brief.get("pick") not in ("most", "least")

    def best_plan(self, fits: dict[int, list[str]]) -> dict[str, int]:
        """The order the request's pick rule makes of read listings that break nothing: the cheapest, the most or the
        least of the measure, else (no priority named) the cheapest with no chain open; under a total budget the
        cheapest of each when the rule's choice breaks it. {} when an item has no such listing."""
        pick, measure = self.brief.get("pick"), str(self.brief.get("pick_measure") or "")

        def rank(ident: str) -> tuple:
            price = self.price_of(ident)
            if pick == "cheapest":
                return (price is None, price or 0.0)
            if pick in ("most", "least") and measure:
                value = self.measure_of(ident, measure)
                return (value is None, -(value or 0.0) if pick == "most" else (value or 0.0))
            # no priority named: the lowest-priced fit (a store weighing several fits takes the cheapest), a listing
            # with a chain still open last
            return (self.chain_open(ident), price is None, price or 0.0, ident not in self.opened)

        if len([n for n in fits if fits[n]]) < len(self.items):
            return {}
        plan = {sorted(ids, key=rank)[0]: self.items[n - 1].qty for n, ids in fits.items()}
        total = self.cart_total(plan)
        if self.budget is not None and (total is None or total > self.budget):
            plan = {min(ids, key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0)): self.items[n - 1].qty for n, ids in fits.items()}
            total = self.cart_total(plan)
            if total is None or total > self.budget:
                return {}
        return plan

    def decline_review(self) -> str:
        """Before an ending without a purchase: every read listing of each item is weighed and each claimed break
        checked against the listing's own words. The ending stands when an item that was looked for has no listing
        left, or the cheapest such basket breaks the budget. Otherwise: an item barely looked at gets two listings
        read; listings that break nothing get their pages read and a second weighing; then the ending is held once
        with those listings named (the endgame's other holds follow); in the last moves the desk orders the best of
        them under the pick rule instead."""
        if self.room() < 2 or self.cutoff():
            return ""
        if self.review_at != self.version:
            self.review_breaks = self.review_call()
            self.review_at = self.version
        if self.review_failed:
            return ""
        fits = {n: [i for i in self.review_pool(n) if i not in self.review_breaks] for n in range(1, len(self.items) + 1)}
        empty = [n for n, ids in fits.items() if not ids]
        # extra reads only from moves a checkout does not need: carting each line, the order, and the answer to the
        # notice the first checkout usually brings
        keep = self.checkout_room()
        if empty and self.review_rows and self.review_blocks == 0 and self.room() > keep:
            # rows the review thinks could fit are read once (a step each) and the listings weighed again
            self.review_blocks = 1
            self.fill_facts(self.review_rows, keep)
            self.review_breaks = self.review_call()
            self.review_at = self.version
            if self.review_failed:
                return ""
            fits = {n: [i for i in self.review_pool(n) if i not in self.review_breaks] for n in range(1, len(self.items) + 1)}
            empty = [n for n, ids in fits.items() if not ids]
            if not empty:
                self.review_blocks = 0
        if any(self.explored(n) for n in empty) and not self.near_checked and self.room() > keep + 1:
            # before an ending stands: the pages of a few listings ruled out (never opened, where the page is a view of
            # its own) are read for a note that sends the failing property elsewhere, the trap a plain reading falls in
            self.near_checked = True
            page = self.roles.get("offer") or self.roles.get("open")
            near = [i for n in empty for i in self.review_pool(n) if i in self.review_breaks and i not in self.opened][:3]
            if near and page and page != self.facts_tool():
                found = len(self.chains)
                self.prove_pages(near, keep)
                if len(self.chains) > found:
                    self.advance_chains(keep, set(near))
                    self.review_breaks = self.review_call()
                    self.review_at = self.version
                    if self.review_failed:
                        return ""
                    fits = {n: [i for i in self.review_pool(n) if i not in self.review_breaks] for n in range(1, len(self.items) + 1)}
                    empty = [n for n, ids in fits.items() if not ids]
        if any(self.explored(n) for n in empty):
            return ""
        if not empty:
            cheapest = {min(ids, key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0.0)): self.items[n - 1].qty for n, ids in fits.items()}
            total = self.cart_total(cheapest)
            if self.budget is not None and self.units_agree() and total is not None and total > self.budget:
                return ""
        # too few moves for the agent to cart, check out and answer a notice itself: the desk orders the best fits
        # under the pick rule and answers a refusal itself
        if self.room() <= self.checkout_room():
            plan = self.best_plan(fits) if not empty else {}
            if not plan or self.repairs >= 2:
                return ""
            self.repairs += 1
            fixed = self.place(plan, [i for i in self.cart_ids() if i not in plan], "The review found read listings that break no requirement for every item; the best under the pick rule is ordered", self.room() < 3)
            if fixed and not self.finished and self.refused_at == self.step:
                self.find_sold()
                fixed += ("\n" + self.autopilot()) if not self.finished else ""
            return ("Not run (the desk acted instead). " + fixed) if fixed else ""
        if self.review_blocks >= 1:
            return ""
        self.review_blocks = 1
        # an item barely looked at: two of its listings read; listings that break nothing: their pages and attributes
        rows = [i for n in empty for i in [r for r in self.seen_order if r in self.rows and r not in self.full and self.item_for(r, True) == n and self.buyable(r)][:2]]
        unread = [i for ids in fits.values() for i in ids if i not in self.opened][:3]
        if (rows or unread) and self.room() > keep:
            self.fill_facts(rows, keep)
            self.prove_pages(unread, keep)
            self.review_breaks = self.review_call()
            self.review_at = self.version
            if self.review_failed:
                return ""
            fits = {n: [i for i in self.review_pool(n) if i not in self.review_breaks] for n in range(1, len(self.items) + 1)}
            empty = [n for n, ids in fits.items() if not ids]
            if any(self.explored(n) for n in empty):
                return ""
        named = {n: ids for n, ids in fits.items() if ids}
        if not named:
            return ("Not run yet (no step used). " + "; ".join(f"item {n} ({self.items[n - 1].name}) has had too few of its listings read to say nothing fits" for n in empty)
                    + ". Search for it, read the attributes of a few of its listings, then decide.")
        order = self.best_plan(named) if not empty else {}
        self.lead_items = {str(n): ids[:4] for n, ids in named.items()}
        self.leads = list(order) + [i for ids in named.values() for i in ids if i not in order]
        self.trace.append({"kind": "decline_held", "fits": named, "empty": empty})
        text = ("Not run yet (no step used). A review of every listing read, each claimed break checked against the listing's own words, found listings that break no requirement in force: "
                + "; ".join(f"item {n}: {', '.join(ids[:4])}" for n, ids in named.items()) + ".")
        if order:
            text += " Under the pick rule the best are " + ", ".join(order) + "."
        if empty:
            text += " " + "; ".join(f"item {n} ({self.items[n - 1].name}) needs more of its listings read" for n in empty) + "."
        return text + (" Put the best one for each item in the cart and check out; end without a purchase only if each of them breaks a requirement:"
                       " name it and quote the listing's words, then send the ending call again.")

    def code_leads(self, strict: bool = False) -> dict[int, list[str]]:
        """Read listings that, by their own records, break no requirement in force: of the item's kind, buyable, within
        a budget the prices can be read against, at least one requirement shown met and at most one the record cannot
        settle. A free second look before an ending without a purchase (no model call)."""
        found: dict[int, list[str]] = {}
        readable = self.budget is not None and self.units_agree() and self.brief.get("budget_scope") != "each"
        passed = {i for c in self.chains.values() for i in c["path"]}
        refs = self.references()
        for number, item in enumerate(self.items, 1):
            needs = [need for n, need in self.needs_now() if n == number]
            picks = []
            for ident in reversed(self.seen_order):
                if ident not in self.full or not self.buyable(ident) or not self.of_item(ident, number):
                    continue
                if (ident in passed and not self.item_for(ident)) or ident in refs:
                    # a listing a chain passed through (the one a note points at) meets that note's value by itself
                    continue
                price = self.price_of(ident)
                # with a budget, a lead needs a price that can be read against it: an unknown price is no lead
                if self.budget is not None and (not readable or price is None or price * item.qty > self.budget):
                    continue
                marks = [self.verdict(self.full[ident], need) for need in needs]
                # strict (to overrule an ending the model chose): every requirement shown met by the record
                if strict and (not needs or None in marks or False in marks):
                    continue
                if False not in marks and (True in marks or not needs) and marks.count(None) <= 1:
                    # a lead whose chains are all followed first, then the cheapest (ranked by price unless a most or
                    # least pick ranks by its measure: then one whose page is already opened, which costs no page step)
                    picks.append((self.chain_open(ident), ident not in self.opened if not self.by_price() else False, price if self.by_price() and price is not None else 0.0, ident))
            if picks:
                found[number] = [i for *_, i in sorted(picks)[:3]]
        return found

    def lead_plan(self, leads: dict[int, list[str]]) -> dict[str, int]:
        """The cart a set of strict leads makes, one line per item at its quantity, or {} when it does not cover every
        item or its total breaks a budget (each lead passing alone is not enough for a basket)."""
        if len(leads) < len(self.items):
            return {}
        plan = {ids[0]: self.items[n - 1].qty for n, ids in leads.items()}
        total = self.cart_total(plan)
        if self.budget is not None and (total is None or total > self.budget):
            return {}
        return plan

    def built_plan(self) -> dict[str, int]:
        """The fullest cart built so far, one line per item at its quantity, when every line can still be bought, no
        line's record breaks a requirement in force and the total fits the budget in force; {} otherwise. A cart taken
        apart for a budget that a note has since raised is worth putting back."""
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            ident = next((i for i in self.built if self.of_item(i, number) and i not in plan), None)
            if ident is None or self.sold_out(ident) or not self.buyable(ident):
                return {}
            if ident in self.full and any(self.verdict(self.full[ident], need) is False for n, need in self.needs_now() if n == number):
                return {}
            plan[ident] = item.qty
        total = self.cart_total(plan)
        if self.budget is not None and (total is None or total > self.budget):
            return {}
        return plan

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
        """The agent ends without buying although the independent check confirmed a reviewed listing: the desk carts the reviewed
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
        if not calls or len(calls) + 1 > self.room():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed just past the cap
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), self.spend_cap + 0.006)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        text = self.place(lines, drop, f"The independent check confirmed {self.confirmed}")
        if not text:
            return ""
        # a refused checkout (a price, budget or stock change) gets the repair the endgame uses
        if not self.finished and self.refused_at == self.step:
            self.find_sold()
            text += ("\n" + self.autopilot()) if not self.finished else ""
        return "Not run (the desk acted instead). " + text

    # ---------- code's own reading of listings ----------
    def verdict(self, record: dict[str, Any], need: str) -> bool | None:
        """Whether a listing's record meets one requirement as far as plain words show. True: the attribute naming the
        requirement's property (or, with no such attribute, the listing's own text) holds every word of one wanted
        value ("red or black" is either; the property's own words and filler words do not count; "wooden" matches
        "wood"). False: an attribute named exactly for that property shares no word with any wanted value. None: the
        record cannot say (numbers and limits, exclusions, a value still to be read from another listing, a related
        key only, or a partly matching value). Spellings one letter apart ("grey", "gray") and a true/false attribute
        read as yes/no count as the same word; a property whose value is still to be read elsewhere is unknown."""
        prop, _, value = need.partition(":")
        relation = "same"
        if need in self.ref_values:
            value = self.ref_values[need]
            relation = self.ref_relation.get(need, "same")
            if relation == "other":
                return None
            if relation in ("at_most", "at_least", "within"):
                # a limit or a tolerance around another listing's number: judged on numbers in one unit word only
                return self.number_verdict(record, prop, value, relation, self.ref_tol.get(need, "")) if re.search(r"\d", value) else None
        elif any(need == n for _, n in self.refs):
            return None
        if _bag(value) & _NOT_WORDS:
            return None
        if re.search(r"\d", value):
            return self.number_verdict(record, prop, value) if relation == "same" else None
        own = _bag(prop)
        options = [frozenset(_yn(w) for w in _bag(part) if w not in _LINK_WORDS and w not in own) for part in re.split(r"\s+or\s+|[,/;]", value)]
        options = [o for o in options if o]
        if not options:
            return None

        def words(texts: list[Any]) -> frozenset[str]:
            return frozenset().union(*(frozenset(_yn(w) for w in _bag(_flag(t))) for t in texts), frozenset())

        def holds(have: frozenset[str]) -> bool:
            return any(all(any(_close(w, h) for h in have) for w in option) for option in options)

        block = _facts(record)
        exact_values = [v for k, v in block.items() if _stem(k) == _stem(prop)]
        loose_values = [v for k, v in block.items() if _key_match(k, prop)]
        if PENDING in exact_values + loose_values:
            return None
        exact = words(exact_values)
        loose = words(loose_values)
        if relation == "unlike":
            # the requirement: differ from the value the other listing shows
            if not exact:
                return None
            if holds(exact):
                return False
            return True if not any(_close(w, h) for option in options for w in option for h in exact) else None
        if exact:
            if holds(exact):
                return True
            return None if any(_close(w, h) for option in options for w in option for h in exact) else False
        if loose:
            # a key for a related property (a part, an aspect) decides nothing against the listing
            return True if holds(loose) else None
        other = self.value_key(record, prop, lambda v: holds(words([v])))
        shown = [v for k, v in block.items() if other and _stem(k) == other]
        if shown and PENDING not in shown:
            # no key names the property, but one key of this kind of listing holds its values ("linen" is a
            # "lamp shade" value): that key decides (every spelling of it), as one named for the property would
            have = words(shown)
            if holds(have):
                return True
            return None if any(_close(w, h) for option in options for w in option for h in have) else False
        return True if holds(words([v for v in record.values() if isinstance(v, str)])) else None

    def value_key(self, record: dict[str, Any], prop: str, wanted: Any) -> frozenset[str] | None:
        """The attribute holding a requirement's property under another name, as its key's words (spellings of one key,
        "lamp shade" and "lamp_shade", count as one): the one key whose values, on read listings of the record's kind,
        include a wanted value (`wanted` tests one value) while no key names the property. None when no key or more
        than one does, or fewer than two such listings carry the key."""
        cat_key = self.category_key()
        kind = record.get(cat_key) if cat_key else None
        hits: set[frozenset[str]] = set()
        carried: dict[frozenset[str], int] = {}
        for other in self.full.values():
            if cat_key and other.get(cat_key) != kind:
                continue
            for key in {_stem(k) for k in _facts(other)}:
                carried[key] = carried.get(key, 0) + 1
            for key, value in _facts(other).items():
                if value != PENDING and wanted(value):
                    hits.add(_stem(key))
        keys = [k for k in hits if carried.get(k, 0) >= 2]
        return keys[0] if len(keys) == 1 else None

    def number_verdict(self, record: dict[str, Any], prop: str, value: str, relation: str = "same", tol: str = "") -> bool | None:
        """A requirement with one number, judged against the attribute named exactly for its property when that
        attribute gives a number in the same unit word ("25 cm" against "15 cm"; "at least 3 shelves" against "4
        shelves"). Another unit, a range, several numbers or no such attribute: None (no unit tables). `relation`
        (a value read off another listing): at most / at least that number, or within `tol` of it, the tolerance in
        the same unit word."""
        found = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)?", value.lower())
        words = _bag(value)
        low = words & {"least", "minimum", "min", "over", "above", "more"}
        high = words & {"most", "maximum", "max", "under", "below", "less", "within", "up"}
        if relation != "same":
            low, high = (relation == "at_least"), (relation == "at_most")
        if len(found) != 1 or (low and high):
            return None
        want, unit = float(found[0][0]), found[0][1] or ""
        block = _fact_block(record) or {}
        shown = [float(n) for k, v in block.items() if _stem(k) == _stem(prop)
                 for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)?", str(v).lower()) if (u or "") == unit]
        if not shown:
            return None
        if relation == "within":
            room = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)?", tol.lower())
            if len(room) != 1 or (room[0][1] or unit) != unit:
                return None
            # at the precision written: a hair of float error is no difference
            return any(abs(n - want) <= float(room[0][0]) + 1e-9 for n in shown)
        if low:
            return any(n >= want for n in shown)
        if high:
            return any(n <= want for n in shown)
        return any(n == want for n in shown)

    def item_for(self, ident: str, loose: bool = False) -> int | None:
        """The requested item a listing is of, by its category label; with no label to go by (or, `loose`, a label
        worded unlike the request), the only item when there is one. Otherwise a label naming no item gives None."""
        cat_key = self.category_key()
        label = (self.rows.get(ident) or {}).get(cat_key) if cat_key else None
        if label is not None:
            fits = [number for number, item in enumerate(self.items, 1) if self.kind_fits(label, item)]
            if len(fits) == 1:
                return fits[0]
            # two requested items of one kind (a blue and a red mug): the label alone cannot say which line is which
            if fits or not loose:
                return None
        return 1 if len(self.items) == 1 else None

    def title_key(self) -> str | None:
        """The row key that names a listing: a few words on most rows, the most varied such key, the shorter on a tie
        (a description runs longer). Worked out once a step: the matcher asks for it for every listing it weighs."""
        stamp = (len(self.rows), self.step)
        if self.title_seen[0] == stamp:
            return self.title_seen[1]
        texts: dict[str, list[str]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if key != self.id_key and isinstance(value, str) and 2 <= len(value.split()) <= 20:
                    texts.setdefault(key, []).append(value.strip().lower())
        best = None
        for key, values in texts.items():
            if len(values) < 0.8 * max(1, len(self.rows)):
                continue
            score = (len(set(values)) / len(values), -sum(len(v) for v in values) / len(values))
            if best is None or score > best[0]:
                best = (score, key)
        self.title_seen = (stamp, best[1] if best else None)
        return self.title_seen[1]

    def twin(self, one: str, other: str) -> bool:
        """Two listings of the same product (another shop selling it): the same attributes, or the same name in the
        same category. A sold-out line's twin is its safest replacement."""
        first, second = _fact_block(self.full.get(one) or {}), _fact_block(self.full.get(other) or {})
        if first and second:
            return {_words(k): str(v).strip().lower() for k, v in first.items()} == {_words(k): str(v).strip().lower() for k, v in second.items()}
        name, cat_key = self.title_key(), self.category_key()
        a, b = self.rows.get(one) or {}, self.rows.get(other) or {}
        if not name or not isinstance(a.get(name), str) or not isinstance(b.get(name), str):
            return False
        same_kind = not cat_key or str(a.get(cat_key) or "").lower() == str(b.get(cat_key) or "").lower()
        return same_kind and a[name].strip().lower() == b[name].strip().lower()

    def listed_ids(self, observation: Any) -> set[str]:
        """Ids of every listing record a result holds, at any depth (a page wrapped together with a notice too)."""
        found: set[str] = set()

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                ident = self.ident(node, True, depth)
                if ident:
                    found.add(ident)
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        return found

    def cart_total(self, lines: dict[str, int]) -> float | None:
        prices = [self.price_of(i) for i in lines]
        if not lines or any(p is None for p in prices):
            return None
        return sum((p or 0.0) * n for p, n in zip(prices, lines.values()))

    def units_agree(self) -> bool:
        """Whether the budget and the listings' prices read in one unit: the budget within a wide band of the typical
        price (a budget in cents against prices in whole units sits a hundredfold apart)."""
        prices = sorted(p for i in self.seen_order for p in [self.price_of(i)] if p)
        if not prices or not isinstance(self.budget, (int, float)):
            return False
        return 0.05 <= self.budget / prices[len(prices) // 2] <= 30

    def qty_for(self, ident: str) -> int:
        number = self.item_for(ident, True)
        return self.cart_lines.get(ident) or (self.items[number - 1].qty if number else 1)

    # ---------- pages: proof and the notes that send a property elsewhere ----------
    def prove_pages(self, ids: list[str], keep: int, hurry: bool = False) -> list[str]:
        """Open the page of each listing not opened yet, keeping `keep` moves: an order counts only for listings
        whose page was opened, and the page carries the seller's notes. Returns what each page showed, with what
        its notes send elsewhere after "[desk] ". `hurry` (the forced finish) counts steps only and reads no notes:
        the clock's margin is kept for these last store calls."""
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        if not tool or not target:
            return []
        shown: list[tuple[str, str]] = []
        for ident in [i for i in dict.fromkeys(ids) if i not in self.opened and i not in self.gone and i in self.rows]:
            if (self.left() if hurry else self.room()) <= keep or self.finished or self.cutoff():
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if tool == self.offer_tool():
                self.offer_seen[ident] = self.step
            shown.append((ident, self.show(seen)[:1500]))
        if not shown:
            return []
        if not hurry:
            # the pages just opened are read for notes together: one light call for the batch
            self.scan_pages([i for i, _ in shown], force=True)
        texts = []
        for ident, text in shown:
            sent = " ".join(self.chain_line(c) for c in self.chains_of(ident))
            texts.append(f"{ident}: " + text + (f"\n[desk] {sent}" if sent else ""))
        self.trace.append({"kind": "proof", "n": len(texts)})
        return texts

    def redirect_need(self, ident: str) -> str:
        prop, to = self.redirects[ident]
        return f"{prop} of listing {ident}: as stated on the listing its page sends it to ({to})"

    def unread_redirects(self, ids: list[str]) -> list[str]:
        """Pages already opened (by the model) of listings now about to be bought, read for notes; what still sends
        a property elsewhere unread is named."""
        self.scan_pages([i for i in ids if i in self.opened], force=True)
        fresh = [c for i in ids for c in self.chains_of(i) if c["state"] != "done" and c["key"] not in self.chain_named]
        self.chain_named.update(c["key"] for c in fresh)
        return [self.chain_line(c) for c in fresh]

    # ---------- chains: a property whose value another listing gives ----------
    def scan_pages(self, ids: list[str] | None = None, force: bool = False) -> list[str]:
        """Read newly opened pages (and requirements that point at another listing) for notes that send a property
        to another listing or state a pairing rule: one light call for a batch, their sentences and attribute keys
        only. Unforced, it waits for three new pages (or the last stretch). Returns the keys of new chains."""
        pool = ids if ids is not None else self.seen_order
        fresh = [i for i in dict.fromkeys(pool) if i not in self.page_read and (i in self.opened or i in self.full)]
        needs = [need for _, need in self.refs if need not in self.need_scanned and need in {n for _, n in self.needs_now()}]
        if not fresh and not needs:
            return []
        if not force and not needs and len(fresh) < 3 and self.room() > self.commit_point() + 4:
            return []
        if self.cost > self.spend_cap - 0.003 or self.cutoff() or self.no_credit:
            return []
        entries: list[str] = []
        labels: dict[str, str] = {}
        for ident in fresh[:10]:
            self.page_read.add(ident)
            # only text not read before: a page opened after its row and attributes were read adds its notes alone
            seen = self.note_seen.setdefault(ident, set())
            texts = [t for t in _sentences(self.rows.get(ident), 4) if t not in seen]
            seen.update(texts)
            if texts:
                keys = ", ".join(list(_facts(self.rows.get(ident) or {}))[:20])
                entries.append(f"SOURCE {ident} (a listing page; its attribute keys: {keys or 'none'}): " + " | ".join(t[:400] for t in texts)[:1000])
        for number, need in enumerate(needs[:4], 1):
            self.need_scanned.add(need)
            labels[f"need-{number}"] = need
            entries.append(f"SOURCE need-{number} (a shopper's requirement): {need}")
        if not entries:
            return []
        got = self.ask_json(REFER + "\n\nSOURCES:\n" + "\n".join(entries), [LIGHT, SPARE], 1500) or {}
        found = self.take_refers(got, labels)
        self.trace.append({"kind": "refer", "sources": len(entries), "chains": found})
        return found

    def take_refers(self, got: dict[str, Any], labels: dict[str, str]) -> list[str]:
        """Chains (and pairing rules) from a REFER reading. A listing whose property goes elsewhere has that property
        marked unknown at once: its own value does not count, whatever it says."""
        found = []
        for entry in _items(got.get("refers")):
            if not isinstance(entry, dict):
                continue
            label, prop = str(entry.get("source") or "").strip(), str(entry.get("property") or "").strip()
            if not prop:
                continue
            pairs = [(str(p[0]).strip(), str(p[1]).strip()) for p in _items(entry.get("pairs")) if isinstance(p, list) and len(p) == 2 and str(p[0]).strip() and str(p[1]).strip()]
            if pairs:
                self.pair_rules.setdefault(_words(prop), []).extend(p for p in pairs if p not in self.pair_rules.get(_words(prop), []))
            relation = str(entry.get("relation") or "same").strip().lower().replace(" ", "_")
            ref = {"shop": str(entry.get("shop") or "").strip(), "kind": str(entry.get("kind") or "").strip(),
                   "values": [str(v).strip() for v in _items(entry.get("values")) if str(v).strip()][:6],
                   "not": [str(v).strip() for v in _items(entry.get("not")) if str(v).strip()][:4],
                   "relation": relation if relation in RELATIONS else "other", "pairs": pairs,
                   "tolerance": str(entry.get("tolerance") or "").strip() if relation == "within" else ""}
            if not (ref["shop"] or ref["kind"] or ref["values"]):
                continue
            if label in labels:
                source = labels[label]
                number = next((n for n, need in self.needs_now() if need == source), None)
            elif label in self.rows:
                source, number = label, self.item_for(label, True)
            else:
                continue
            key = f"{source}|{_words(prop)}"
            if key in self.chains:
                continue
            self.chains[key] = {"key": key, "source": source, "item": number, "prop": prop, "own_key": str(entry.get("key") or "").strip(),
                                "ref": ref, "state": "open", "value": "", "path": [], "tries": 0}
            if source in self.rows:
                self.effective.setdefault(source, {})[_words(prop)] = (PENDING, [], self.chains[key]["own_key"])
                self.redirects.setdefault(source, (prop, self.describe(ref)))
                self.apply_effective(source)
            self.version += 1
            found.append(key)
        return found

    @staticmethod
    def describe(ref: dict[str, Any]) -> str:
        parts = [p for p in (ref["kind"], f"from {ref['shop']}" if ref["shop"] else "") if p]
        if ref["values"]:
            parts.append("with " + ", ".join(ref["values"]))
        if ref["not"]:
            parts.append("not " + ", ".join(ref["not"]))
        return " ".join(parts) or "another listing"

    def chains_of(self, ident: str) -> list[dict[str, Any]]:
        return [c for c in self.chains.values() if c["source"] == ident]

    def chain_open(self, ident: str) -> bool:
        return any(c["state"] != "done" for c in self.chains_of(ident))

    def apply_effective(self, ident: str) -> None:
        """Put the chains' values on a listing's record (rows and full): each property sent elsewhere carries the
        value the last listing of its chain gives, or PENDING while unread, and a desk note says so."""
        marks = self.effective.get(ident)
        record = self.rows.get(ident)
        if not marks or not isinstance(record, dict):
            return
        where = _block_key(record)
        block = dict(record[where]) if where else {}
        notes = []
        for prop, (value, path, own_key) in marks.items():
            key = (own_key if own_key in block else None) or next((k for k in block if _key_match(k, prop)), None) or prop
            own = block.get(key)
            block[key] = value
            if value == PENDING:
                notes.append(f"its seller sends its {prop} to another listing, not read yet: its own value ({own}) does not count")
            else:
                notes.append(f"its {prop} is {value}, as the listing its seller sends that property to states ({' -> '.join(path)}); its own value ({own}) does not count")
        if where:
            record[where] = block
        else:
            record.update(block)
        record["desk note"] = "; ".join(notes)
        if ident in self.full:
            self.full[ident] = dict(record)

    def constrained(self, chain: dict[str, Any]) -> bool:
        """A chain worth steps now: a requirement's own, or a buyable listing's whose item has a requirement on that
        property (or which is in the cart or a lead), priced within a readable budget."""
        source = chain["source"]
        if source not in self.rows:
            return True
        if not self.buyable(source):
            return False
        price = self.price_of(source)
        if self.budget is not None and isinstance(self.budget, (int, float)) and price is not None and self.units_agree() and price > self.budget:
            return False
        if source in self.cart_ids() or source in self.leads:
            return True
        number = chain["item"]
        return any(_key_match(need.split(":", 1)[0], chain["prop"]) for n, need in self.needs_now() if number is None or n == number)

    def fact_for(self, ident: str, prop: str, key: str = "") -> str | None:
        block = _facts(self.full.get(ident) or self.rows.get(ident) or {})
        for want in [k for k in (key, prop) if k]:
            hit = next((v for k, v in block.items() if _key_match(k, want)), None)
            if hit is not None and hit != PENDING:
                return str(_flag(hit))
        return None

    def pair_value(self, value: str, pairs: list[tuple[str, str]]) -> str | None:
        """The value a pairing rule gives for `value`: the one pair whose first side has the same words."""
        have = _tokens(value)

        def same(text: str) -> bool:
            want = _tokens(text)
            return bool(want) and all(any(_close(w, h) or _num_eq(w, h) for h in have) for w in want) and all(any(_close(h, w) or _num_eq(h, w) for w in want) for h in have)

        hits = {b for a, b in pairs if same(a)}
        return hits.pop() if len(hits) == 1 else None

    def text_of(self, ident: str, notes: bool = True) -> list[str]:
        """A listing's words for matching a description: its text and its attributes' values (not its id, and no
        bare number outside the attributes: a price or a count of stock says nothing about what the listing is).
        Without `notes`, its own words only: free text (a description, a seller's note) is left out, since a note can
        name values the listing does not have or that belong to another listing; its name stays."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        where = _block_key(record)
        name = None if notes else self.title_key()
        words: list[str] = []

        def walk(node: Any, depth: int, facts: bool, key: Any = None) -> None:
            if depth > 4:
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    if key not in ("desk note", self.id_key):
                        if facts:
                            # an attribute's name is part of how a description names it ("a metal body")
                            words.extend(_tokens(_words(key)))
                        walk(value, depth + 1, facts or (depth == 0 and key == where), key)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1, facts, key)
            elif isinstance(node, str) and node != PENDING and node.strip() != ident:
                if not notes and not facts and key != name and len(node.split()) >= 8:
                    return
                words.extend(_tokens(node))
            elif facts and isinstance(node, (int, float)) and not isinstance(node, bool):
                words.extend(_tokens(node))

        walk(record, 0, False)
        return words

    def describes(self, ident: str, ref: dict[str, Any]) -> int | None:
        """How well a listing fits a chain's description: None when a part rules it out (a NOT part present, a kind
        word missing, or, once its attributes are read, a missing word of its shop or values), else the sum of its
        parts' states (higher is surer). A search row does not show everything: what it lacks is still open (1)."""
        return (self.description_states(ident, ref) or (None,))[0]

    def description_states(self, ident: str, ref: dict[str, Any]) -> tuple[int, bool] | None:
        """(score, whether the listing's own words show the description's shop, or every value when it names no
        shop), or None when a part rules the listing out (see describes)."""
        have = self.text_of(ident, notes=False)
        record = self.full.get(ident) or self.rows.get(ident) or {}
        attrs = [(_tokens(_words(k)), _tokens(_flag(v))) for k, v in _facts(record).items() if v != PENDING]
        read = ident in self.full
        for phrase in ref["not"]:
            words = [w for w in _tokens(phrase) if w not in _LINK_WORDS and not w[0].isdigit()]
            if words and all(any(_close(w, h) for h in have) for w in words):
                return None
        kind = [w for w in _tokens(ref["kind"]) if len(w) >= 4 and w.isalpha() and w not in _LINK_WORDS]
        if kind and not any(_close(w, h) for w in kind for h in have):
            return None
        shop = _value_state(ref["shop"], have) if ref["shop"] else 2
        states = [_value_state(v, have, attrs) for v in ref["values"]]
        if -1 in states or (read and shop <= 0):
            return None
        shown = shop == 2 if ref["shop"] else bool(states) and all(x == 2 for x in states)
        # an unread row lacks what only its attributes show: its missing words are still open
        return (shop if read else max(shop, 1)) + sum(x if read else max(x, 1) for x in states), shown

    def note_search(self, name: str, args: dict[str, Any], observation: Any = None) -> None:
        """The model's own searches (any tool whose result held listing rows) teach the desk which tool and argument
        take free words: the pair given the most words on average. A search whose rows are mostly of one item's kind is
        kept for that item (without its page argument), so the desk's own searches for it keep the model's filters."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            spec = props.get(key)
            if isinstance(spec, dict) and "enum" not in spec and isinstance(value, str) and value.strip():
                self.search_words.setdefault(f"{name}\n{key}", []).append(len(value.split()))
                self.search_vals.setdefault(f"{name}\n{key}", set()).add(value.strip().lower())
        self.search_seen[name] = dict(args)
        free = self.free_key(name)
        worded = bool(free) and isinstance(args.get(free), str) and bool(str(args.get(free)).strip())
        rows = [i for i, _ in self.records_in(observation)]
        page = (self.shapes.get(name) or {}).get("next_arg")
        for number in range(1, len(self.items) + 1):
            # a search that took free words (a later browse of the shelf has none to replace)
            if worded and rows and 5 * sum(1 for i in rows if self.of_item(i, number)) >= 3 * len(rows):
                self.item_search[number] = (name, {k: v for k, v in args.items() if k != page})

    def free_key(self, name: str) -> str | None:
        """The argument of a search tool that takes free words: of its free-text arguments, leaving out one whose every
        value is a category label the store's rows carry (a shelf filter), the one whose values change from search to
        search, then the one given the most words."""
        cat = self.category_key()
        labels = {str(r.get(cat)).strip().lower() for r in self.rows.values() if cat and r.get(cat) is not None}
        keys = {k.split("\n", 1)[1]: (len(self.search_vals.get(k, ())), sum(v) / len(v))
                for k, v in self.search_words.items() if v and k.split("\n", 1)[0] == name and not set() < self.search_vals.get(k, set()) <= labels}
        return max(keys, key=keys.get) if keys else None

    def desk_search(self, query: str, template: tuple[str, dict[str, Any]] | None = None) -> Any:
        """A search the desk runs itself (one step), shaped like the model's own searches with the free words
        replaced; None when the store's search cannot be read that way. With `template` (the model's search for one
        item), every other argument of that search is kept: its category or shelf filter and its mode."""
        # the free-words argument: the one whose values change from search to search (a category or shelf filter
        # repeats), then the one given the most words
        cat = self.category_key()
        labels = {str(r.get(cat)).strip().lower() for r in self.rows.values() if cat and r.get(cat) is not None}
        heard = {k: (len(self.search_vals.get(k, ())), sum(v) / len(v)) for k, v in self.search_words.items()
                 if v and not set() < self.search_vals.get(k, set()) <= labels}
        if template:
            mine = {k: v for k, v in heard.items() if k.split("\n", 1)[0] == template[0] and k.split("\n", 1)[1] in template[1]}
            heard = mine or heard
            template = template if mine else None
        tool, free = max(heard, key=heard.get).split("\n", 1) if heard else ("", None)
        if not heard:
            # no search of the model's to copy: the search role, else a tool taking one required free string that is
            # no listing id (not a page, cart, message or finishing tool)
            busy = {self.roles.get(r) for r in ("open", "offer", "compare", "cart", "cart_view", "buy", "refuse", "ask")}
            for name in [self.roles.get("search") or ""] + sorted(self.schemas):
                if not name or (name in busy and name != self.roles.get("search")):
                    continue
                spec = self.schemas.get(name) or {}
                texts = [k for k, v in (spec.get("properties") or {}).items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
                need = [k for k in spec.get("required") or [] if k in texts and (name, k) not in self.id_args]
                if need or (name == self.roles.get("search") and len(texts) == 1):
                    tool, free = name, (need or texts)[0]
                    break
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        query = " ".join(query.split())
        if not tool or not free or not query or query.lower() in self.desk_queries or self.finished:
            return None
        self.desk_queries.add(query.lower())
        base = self.search_seen.get(tool) or {}
        args: dict[str, Any] = {k: v for k, v in base.items() if k in props and isinstance(props[k], dict) and "enum" in props[k]}
        if template:
            args = {k: v for k, v in template[1].items() if k in props}
        for key in schema.get("required") or []:
            if key not in args and key != free:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        args[free] = query
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return None
        reply = self.act(tool, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))
        self.trace.append({"kind": "desk_search", "q": query, "rows": len(self.records_in(seen))})
        plan = self.paging(tool, seen)
        if plan and plan[3] is not None:
            self.next_page = (tool, {**args, plan[2]: plan[3]})
        else:
            self.next_page = None
        return seen

    def open_full(self, ident: str, keep: int) -> None:
        """A chain's listing read in full: its page (the notes live there) and its attributes when the page shows none."""
        if ident not in self.opened:
            self.prove_pages([ident], keep)
        if not _fact_block(self.full.get(ident) or {}) and self.room() > keep:
            self.fill_facts([ident], keep)

    def find_target(self, ref: dict[str, Any], skip: set[str], keep: int) -> str | None:
        """The one listing a description picks out: the listings seen first, then the desk's searches (its shop and
        kind, then its shop and values, a second page when the first shows none), the closest few read in full, a
        light pick when units or several close ones leave it open. None unless exactly one fits."""
        # every part shown scores 2, the shop's part included (a description naming no shop gets it free)
        top = 2 * (len(ref["values"]) + 1)

        def pool() -> list[tuple[int, str]]:
            return sorted(((s, i) for i in self.seen_order if i not in skip for s in [self.describes(i, ref)] if s is not None), key=lambda e: -e[0])

        def shown() -> bool:
            # some listing seen so far shows the description's shop (or all its values) in its own words and could
            # still be the one: a row not read yet, or a read listing showing every part
            return any((self.description_states(i, ref) or (0, False))[1] and (i not in self.full or score == top) for score, i in found)

        queries = [" ".join(p for p in (ref["shop"], ref["kind"]) if p), " ".join([ref["shop"]] + ref["values"][:2]).strip(), " ".join([ref["kind"]] + ref["values"][:2]).strip()]
        found: list[tuple[int, str]] = []
        sure: list[str] = []
        for _ in range(2):
            found = pool()
            for query in queries:
                if shown() or self.room() <= keep + 2 or self.finished:
                    break
                self.next_page = None
                if self.desk_search(query) is None:
                    continue
                found = pool()
                if not shown() and self.next_page and self.room() > keep + 2:
                    tool, args = self.next_page
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))
                        found = pool()
            # the closest ones' attributes read one by one (those showing the shop first), stopping at a sure match
            found.sort(key=lambda e: (not (self.description_states(e[1], ref) or (0, False))[1], -e[0]))
            for _, ident in found[:3]:
                sure = [i for s, i in pool() if s == top and i in self.full]
                if sure or self.room() <= keep + 1 or self.finished:
                    break
                if not _fact_block(self.full.get(ident) or {}):
                    self.fill_facts([ident], keep) if self.facts_tool() else self.open_full(ident, keep)
            found = pool()
            sure = [i for s, i in found if s == top and i in self.full]
            if sure:
                break
        if not found:
            return None
        if len(sure) == 1:
            return sure[0]
        # one candidate left that does not show every part (a look-alike of the same shop missing one) goes to the pick
        close = sure[:3] if sure else [i for _, i in found[:3]]
        self.trace.append({"kind": "find", "about": self.describe(ref)[:120], "top": top,
                           "pool": [(i, s, (self.description_states(i, ref) or (0, False))[1], i in self.full) for s, i in found[:5]]})
        return self.pick_one(ref, close) if close else None

    def pick_one(self, ref: dict[str, Any], ids: list[str]) -> str | None:
        if self.cost > self.spend_cap - 0.003 or self.no_credit:
            return None
        records = "\n".join(f"{i}: " + _dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1200] for i in ids)
        got = self.ask_json(PICK + "\n\nDESCRIPTION: " + self.describe(ref) + "\n\nLISTING RECORDS:\n" + records, [LIGHT, SPARE], 800) or {}
        pick = str(got.get("id") or "").strip()
        self.trace.append({"kind": "pick", "ids": ids, "got": pick})
        return pick if pick in ids else None

    def follow(self, chain: dict[str, Any], keep: int) -> None:
        """Walk one chain: the listing its description picks out, read in full, then on while that page sends the
        property further (three listings at most, none twice); the last page's value, turned by a pairing rule when
        one applies, is the chain's value. A chain not found twice is lost."""
        chain["tries"] += 1
        ref = chain["ref"]
        # each walk starts at the chain's own description: a walk cut short left no half-path to skip past
        path: list[str] = []
        skip = {chain["source"]} if chain["source"] in self.rows else set()
        value = None
        while len(path) < 3:
            target = self.find_target(ref, skip, keep)
            if not target:
                if chain["tries"] >= 2:
                    chain["state"] = "lost"
                return
            if target not in self.opened:
                self.prove_pages([target], min(keep, self.tail()))
            if target not in self.opened:
                # a link counts only once its page is open (its notes may send the property further): left open
                return
            path.append(target)
            chain["path"] = list(path)
            skip.add(target)
            self.scan_pages([target], force=True)
            onward = next((c for c in self.chains_of(target) if _key_match(c["prop"], chain["prop"])), None)
            if onward is None:
                value = self.fact_for(target, chain["prop"], chain["own_key"])
                break
            if onward["state"] == "done":
                value = onward["value"]
                break
            ref = onward["ref"]
        if value is None:
            chain["state"] = "lost"
            return
        pairs = ref["pairs"] or chain["ref"]["pairs"] or next((p for k, p in self.pair_rules.items() if _key_match(k, chain["prop"])), [])
        if ref["relation"] == "pair" or chain["ref"]["relation"] == "pair" or pairs:
            turned = self.pair_value(value, pairs)
            if turned is None:
                chain["state"] = "lost"
                return
            value = turned
        chain["state"], chain["value"] = "done", value
        self.version += 1
        source = chain["source"]
        if source in self.rows:
            self.effective.setdefault(source, {})[_words(chain["prop"])] = (value, list(chain["path"]), chain["own_key"])
            self.apply_effective(source)
            if source in self.redirects:
                self.ref_values[self.redirect_need(source)] = value
        else:
            self.ref_values[source] = value
            self.relate(source, chain)
        self.trace.append({"kind": "chain", "key": chain["key"][:120], "path": chain["path"], "value": value})

    def relate(self, need: str, chain: dict[str, Any] | None) -> None:
        """How a pointing requirement uses the value it reads, as the reading of the requirement gave it: the same
        value (a pairing rule already turned it), unlike it, a limit or a tolerance around it; with no reading of the
        requirement, unknown (the code then judges nothing on it)."""
        if chain is None:
            self.ref_relation[need] = "other"
            return
        relation = chain["ref"]["relation"]
        self.ref_relation[need] = "same" if relation == "pair" else relation
        if chain["ref"].get("tolerance"):
            self.ref_tol[need] = chain["ref"]["tolerance"]

    def chain_line(self, chain: dict[str, Any]) -> str:
        source, prop = chain["source"], chain["prop"]
        head = f"{source}'s page sends its {prop} to another listing ({self.describe(chain['ref'])})" if source in self.rows else f"the requirement '{source}' points at another listing"
        if chain["state"] == "done":
            return head + f": the desk followed it ({' -> '.join(chain['path'])}), which gives {chain['value']}" + (f", so {source}'s {prop} counts as {chain['value']} (its own value does not count)." if source in self.rows else ".")
        if chain["state"] == "lost":
            return head + ": the desk could not find that listing; find it by its shop and kind and open it before judging " + (f"{source}." if source in self.rows else "this requirement.")
        return head + f"; until that listing is read, {source if source in self.rows else 'the requirement'} can be neither passed nor failed on {prop}."

    def advance_chains(self, keep: int, only: set[str] | None = None, needs: bool = True) -> list[str]:
        """Follow the open chains that matter now (constrained ones; with `only`, those of these listings and of
        requirements; without `needs`, listings' chains only), cheapest source first under a cheapest pick, while
        moves above `keep` allow. Returns what each newly settled chain gave, once."""
        def order(chain: dict[str, Any]) -> tuple[int, float]:
            price = self.price_of(chain["source"]) if chain["source"] in self.rows else None
            return (chain["source"] in self.rows, price if self.by_price() and price is not None else 0.0)

        said = []
        for chain in sorted(self.chains.values(), key=order):
            if chain["state"] != "open" or chain["tries"] >= 2:
                continue
            if only is not None and chain["source"] in self.rows and chain["source"] not in only:
                continue
            if not needs and chain["source"] not in self.rows:
                continue
            if not self.constrained(chain):
                continue
            if self.room() <= keep + 3 or self.finished or self.cutoff() or self.cost > self.spend_cap - 0.004:
                break
            self.quick += 1
            try:
                self.follow(chain, keep)
            finally:
                self.quick -= 1
            if chain["state"] != "open" and chain["key"] not in self.chain_said:
                self.chain_said.add(chain["key"])
                said.append(self.chain_line(chain))
        return said

    def chain_turn(self) -> str:
        """After each turn: new pages read for notes (in batches), and the chains that matter followed while spare
        moves allow (the questions, the commit point and the cart's moves stay untouched). A requirement that points
        at another listing is followed as soon as it is read, keeping only the cart's and the questions' moves: its
        value steers the pick, and the same walk at the checkout costs as much but comes after the cart is chosen."""
        if self.finished:
            return ""
        self.scan_pages()
        said = self.advance_chains(self.checkout_room() + self.msgs_free(), set())
        said += self.advance_chains(max(self.commit_point() + self.msgs_free(), self.tail() + 2), needs=False)
        return ("\n[desk] " + " ".join(said)) if said else ""

    def cart_rejected(self) -> bool:
        """The cart as it is now was turned down: the last cart check judged it failing, or a line's own attributes break
        a requirement in force. A forced end does not order such a cart (an order of a cart known to fail earns nothing,
        an ending without a purchase may be the right one)."""
        lines = self.cart_ids()
        if not lines:
            return False
        if self.verdicts.get(self.check_key(lines)) is False:
            return True
        return any(i in self.full and self.verdict(self.full[i], need) is False for i in lines for n, need in self.needs_now() if self.of_item(i, n))

    def chain_hold(self) -> str:
        """An ending without a purchase while a chain matters: the desk follows what it can and shows the values (the
        listings get judged again); a chain it cannot find holds the ending twice at most while moves allow; with
        few moves left it orders the best lead instead, since a decline here is almost never right."""
        self.scan_pages(force=True)
        if not self.chains:
            return ""
        said = self.advance_chains(self.tail() + 1)
        if said:
            return ("Not run yet (no step used). The desk followed the notes that send a property to another listing: " + " ".join(said)
                    + " Judge those listings with these values (their own values for that property do not count) and buy what fits; end without a purchase only if each one fails another requirement.")
        pending = [c for c in self.chains.values() if c["state"] != "done" and self.constrained(c)]
        bearing = [c for c in self.chains.values() if self.constrained(c) and (c["source"] not in self.rows or any(self.of_item(c["source"], n) for n in range(1, len(self.items) + 1)))]
        if not bearing:
            return ""
        if pending and self.chain_blocks < 2 and self.room() > self.recovery():
            self.chain_blocks += 1
            return ("Not run yet (no step used). " + " ".join(self.chain_line(c) for c in pending[:3])
                    + " Ending without a purchase is wrong while such a listing may fit: find the listing each note names (search its shop and kind), open it, use its value, then decide.")
        if self.room() <= self.recovery() and self.repairs < 2:
            plan = self.chain_order()
            if plan:
                self.repairs += 1
                fixed = self.place(plan, [i for i in self.cart_ids() if i not in plan], "Few moves are left and a property of these listings is read from another listing: the best lead for each item is ordered rather than nothing", self.room() < 3)
                if fixed:
                    return "Not run (the desk acted instead). " + fixed
        return ""

    def chain_order(self) -> dict[str, int]:
        """The order at the end of a task with chains: listings whose records (with the chains' values) show every
        requirement met first; else ones whose only unknown is a property still to be read from another listing, the
        best per item under the pick rule and within the basket's budget."""
        plan = self.lead_plan(self.code_leads(strict=True))
        if plan:
            return plan
        fits: dict[int, list[str]] = {}
        for number in range(1, len(self.items) + 1):
            needs = [need for n, need in self.needs_now() if n == number]
            for ident in self.review_pool(number):
                marks = [(need, self.verdict(self.full[ident], need)) for need in needs]
                open_props = [c["prop"] for c in self.chains_of(ident) if c["state"] != "done"]
                if marks and all(mark is True or (mark is None and any(_key_match(need.split(":", 1)[0], p) for p in open_props)) for need, mark in marks):
                    fits.setdefault(number, []).append(ident)
        return self.best_plan(fits)

    # ---------- the endgame: repairs after a refused checkout ----------
    def best_swap(self, ident: str, taken: set[str], cap: float | None = None) -> str | None:
        """A listing to take a cart line's place: of the line's kind, buyable, priced under `cap` when given, breaking
        no requirement in force by its record. The same product from another shop first; otherwise a listing whose
        attributes were read, with at most one requirement its record cannot settle; under a cheapest pick the cheaper."""
        cat_key = self.category_key()
        kind = str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
        number = self.item_for(ident, True)
        needs = [need for n, need in self.needs_now() if n == number] if number else []
        pool = []
        for other in self.seen_order:
            if other == ident or other in taken or not self.buyable(other):
                continue
            row = self.rows.get(other) or {}
            if kind and str(row.get(cat_key) or "").strip().lower() != kind:
                continue
            price = self.price_of(other)
            if cap is not None and (price is None or price > cap):
                continue
            marks = [self.verdict(self.full.get(other) or row, need) for need in needs]
            twin = self.twin(ident, other)
            # a line whose item cannot be told (two items of its kind) takes only the same product
            if not twin and number is None and len(self.items) > 1:
                continue
            if False in marks or (not twin and (other not in self.full or marks.count(None) > 1)):
                continue
            cheap = price if self.by_price() and price is not None else 0.0
            pool.append(((not twin, marks.count(None), cheap, other not in self.opened), other))
        return min(pool)[1] if pool else None

    def repair(self) -> tuple[dict[str, int], list[str], list[str]] | None:
        """The cart a refused checkout calls for, made by code: a sold-out line, or one its record now shows breaking a
        requirement, takes another listing of its kind; a total over a budget the prices can be read against takes a
        cheaper listing for its dearest line. None when some line has no such replacement."""
        lines = {}
        for ident in self.cart_ids():
            number = self.item_for(ident, True)
            # the quantity the item asks for (the check judges that one), else what the cart holds
            lines[ident] = self.items[number - 1].qty if number else self.cart_lines.get(ident, 1)
        drop: list[str] = []
        notes: list[str] = []
        for ident in list(lines):
            number = self.item_for(ident, True)
            # only a requirement that arrived after the line was carted can break it: the earlier ones were weighed
            # when it was chosen (and by the cart check), which reads values better than plain word matching
            since = self.carted_at.get(ident, -1)
            broken = number and any(self.need_at.get(need, 0) > since and self.verdict(self.full.get(ident) or self.rows.get(ident) or {}, need) is False
                                    for n, need in self.needs_now() if n == number)
            if not self.sold_out(ident) and not broken:
                continue
            swap = self.best_swap(ident, set(lines))
            if not swap:
                return None
            lines[swap] = lines.pop(ident)
            drop.append(ident)
            notes.append(f"{ident} {'sold out' if self.sold_out(ident) else 'no longer fits'} -> {swap}" + (" (the same product from another shop)" if self.twin(ident, swap) else ""))
        total = self.cart_total(lines)
        if self.budget is not None and total is not None and total > self.budget and self.units_agree() and self.brief.get("budget_scope") != "each":
            for ident in sorted(lines, key=lambda i: -(self.price_of(i) or 0.0) * lines[i]):
                cap = (self.price_of(ident) or 0.0) - (total - self.budget) / max(1, lines[ident])
                swap = self.best_swap(ident, set(lines), cap)
                if swap:
                    lines[swap] = lines.pop(ident)
                    drop.append(ident)
                    notes.append(f"total {total:g} over the budget {self.budget:g}: {ident} -> {swap}")
                    break
            else:
                return None
        if len(lines) < len(self.items):
            return None
        return lines, [i for i in drop if i in self.cart_ids()], notes

    def place(self, lines: dict[str, int], drop: list[str], why: str, hurry: bool = False) -> str:
        """The desk's own checkout: set the cart to `lines` (taking `drop` off), open every line's page not opened yet
        while a step stays for the order, run the request's order steps, check out. Returns what happened, or ""
        when nothing could be sent. `hurry` (the forced finish) counts steps only and makes no model call."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        calls = self.cart_calls(lines, drop) if cart and buy else None
        if calls is None or len(calls) + 1 > (self.left() if hurry else self.room()):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults or self.cutoff():
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        pages = self.prove_pages(list(lines), self.tail(), hurry)
        sent = [p.split("\n[desk] ", 1)[1] for p in pages if "\n[desk] " in p]
        if sent and not hurry and not self.ending:
            # a page just opened sends a property elsewhere: that chain decides whether the line fits, so the desk
            # follows it while moves allow and otherwise leaves the move (and the steps) to the agent
            said = self.advance_chains(self.tail() + 1, set(lines))
            blocks = self.trail_blocks
            chain, _ = self.trail()
            self.trail_blocks = blocks
            chain = " ".join(said) + (" " + chain if chain.strip() else "")
            if any(self.chain_open(i) for i in lines) or said:
                return (f"[desk] {why}: the desk set the cart ({'; '.join(done) or 'as it was'}) and opened its pages, but not checked out yet. "
                        + " ".join(sent) + (" " + chain.strip() if chain.strip() else "") + " Then check out.")
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults or self.left() < 1:
            return ""
        go, said = self.gate("order") if self.room() >= 2 else (True, "")
        if not go or self.left() < 1:
            return said
        why += (" (" + said + ")") if said else ""
        reply = self.act(buy, args)
        self.trace.append({"kind": "placed", "why": why[:200], "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
        if reply is None:
            return ""
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        text = f"[desk] {why}: the desk set the cart ({'; '.join(done) or 'as it was'})"
        text += (f", opened {len(pages)} page(s)" if pages else "") + " and checked out. Store reply: " + self.show(reply.get("observation"))
        return text + (" [episode finished]" if self.finished else "")

    def autopilot(self) -> str:
        """After a refused checkout with few moves left (or a decline that would follow one): the desk repairs the cart
        and checks out itself, so the last steps are not spent on a model that may give up. A Sonnet check passes the
        repaired cart first while money allows; at most twice an episode."""
        if self.repairs >= 2 or self.finished or self.left() < 2 or not self.roles.get("buy") or not self.cart_ids():
            return ""
        if self.note_sold and not any(self.sold_out(i) for i in self.cart_ids()):
            # the refusal says a line sold out without naming it: read the lines' offers to find it (a step each);
            # an unchanged cart sent again would only be refused again
            tool = self.offer_tool()
            target = self.single_target(tool)
            for ident in self.cart_ids():
                if not tool or not target or self.room() < 3 or self.finished:
                    break
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                reply = None if faults else self.act(tool, args)
                if reply is not None:
                    self.offer_seen[ident] = self.step
                    self.observe(tool, reply.get("observation"))
                if any(self.sold_out(i) or not self.buyable(i) for i in self.cart_ids()):
                    break
            for ident in self.cart_ids():
                if not self.buyable(ident):
                    self.unavailable.add(ident)
            if not any(self.sold_out(i) for i in self.cart_ids()):
                return ""
        fix = self.repair()
        if fix is None:
            return ""
        lines, drop, notes = fix
        self.repairs += 1
        # with the clock out (steps still left) the repair does what the forced finish would: store calls only
        hurry = self.room() < 2
        if not hurry and self.cost < self.spend_cap and self.room() >= 3:
            # only a check that found a problem stops the repair (one that could not answer does not); a cart found
            # wrong before stays wrong
            key = self.check_key(list(lines))
            fresh = key not in self.checked
            verdict = self.check_cart(list(lines), self.spend_cap + 0.006)
            if verdict or (not fresh and self.verdicts.get(key) is False):
                return verdict or "A careful review already found this cart wrong: fix the lines it named, then check out."
        why = "The checkout was refused" + (" (" + "; ".join(notes) + ")" if notes else " and the cart still meets every requirement in force")
        return self.place(lines, drop, why, hurry)

    def cheaper_after_drop(self) -> str:
        """After a dropped requirement under a cheapest pick: cheaper listings of a cart line's kind whose records break
        nothing still in force may fit now; they are named so they are judged before the checkout."""
        cat_key = self.category_key()
        cart = self.cart_ids()
        found: list[tuple[float, str]] = []
        for line in cart:
            price, number = self.price_of(line), self.item_for(line, True)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
            if price is None or not kind:
                continue
            needs = [need for n, need in self.needs_now() if n == number]
            pool = [(p, i) for i in self.seen_order for p in [self.price_of(i)]
                    if i not in cart and p is not None and p < price and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and not any(self.verdict(self.full.get(i) or self.rows.get(i) or {}, need) is False for need in needs)]
            found += [f for f in sorted(pool)[:4] if f not in found]
        if not found:
            return ""
        return ("A requirement was dropped, so a listing that failed only on it may fit now, and the pick is the cheapest. Cheaper listings of the cart's kinds: "
                + ", ".join(f"{i} ({p:g})" for p, i in found[:6]) + ". Read the attributes of any not read yet, put the cheapest that meets every requirement in force in the cart, then check out.")

    def budget_hint(self, old: Any) -> str:
        """After the budget rose: listings of a requested kind priced above the old budget and within the new one, which
        an earlier pass set aside for their price alone. After it fell with the cart over it: read fits inside it."""
        if not isinstance(old, (int, float)) or not isinstance(self.budget, (int, float)) or not self.units_agree():
            return ""
        if self.budget < old:
            # the budget fell (often at the first checkout, steps to spare): name read fits inside the new one
            total = self.cart_total({i: self.qty_for(i) for i in self.cart_ids()})
            if total is not None and total <= self.budget:
                return ""
            leads = self.code_leads()
            if not leads:
                return ""
            return (" The budget fell: read listings whose records break no requirement and fit inside it: "
                    + "; ".join(f"item {n}: " + ", ".join(f"{i} ({self.price_of(i):g})" for i in ids) for n, ids in leads.items()) + ".")
        if self.budget <= old:
            return ""
        rows = sorted((p, i) for i in self.seen_order for p in [self.price_of(i)]
                      if p is not None and old < p <= self.budget and self.buyable(i) and self.item_for(i))
        if not rows:
            return ""
        return " The budget rose: listings read before that were over the old budget and are within the new one: " + ", ".join(f"{i} ({p:g})" for p, i in rows[:6]) + "."

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
        left = self.room()
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

    def held_back(self) -> int:
        """Messages the interviewer leaves for the steps the request asks for at the order and at the ending, one at
        least for each of the two that needs any. A step at the ending holds none back when the ending call takes a
        remark of its own, which carries it (the closest listings, by id), so that message asks the shopper a property
        instead."""
        remark = bool(self.remark_keys(self.roles.get("refuse")))
        talk = [p["messages"] for p in self.process if p["messages"] and (p["when"] == "order" or (p["when"] == "end" and not remark))]
        return min(max(talk), self.msg_limit) if talk else 0

    def remark_keys(self, name: str | None) -> list[str]:
        """The free-text fields of a tool (strings with no fixed choices): where an ending call takes its remark."""
        props = (self.schemas.get(name or "") or {}).get("properties") or {}
        return [k for k, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec]

    def closest(self) -> list[tuple[int, str, str]]:
        """For each item with no fitting listing, the read listing that comes nearest and what it misses: of the
        item's weighed listings, the one whose own text shows the most of the item's wanted values, the cheaper on a
        tie. An item whose listings all fit (the basket fails on its total) gives its cheapest fit."""
        out = []
        for number in range(1, len(self.items) + 1):
            # the weighed listings; when every read one is over the budget, those (the budget is then what they miss)
            pool = [i for i in self.review_pool(number) if not self.sold_out(i)] or [
                i for i in self.seen_order if i in self.full and self.of_item(i, number) and not self.sold_out(i)]
            if not pool:
                continue
            wanted = {w for n, need in self.needs_now() if n == number for w in _tokens(need.split(":", 1)[-1]) if w not in _LINK_WORDS}

            def near(ident: str) -> tuple[int, bool, float]:
                have = self.text_of(ident)
                price = self.price_of(ident)
                return -sum(1 for w in wanted if any(_close(w, h) for h in have)), price is None, price or 0.0
            fits = [i for i in pool if i not in self.review_breaks]
            best = min(fits or pool, key=near)
            price = self.price_of(best)
            over = self.budget is not None and price is not None and self.units_agree() and price * self.items[number - 1].qty > self.budget
            out.append((number, best, self.review_breaks[best][0] if best in self.review_breaks else ("the budget" if over else "")))
        return out

    def closest_note(self) -> str:
        """The closest listings in words, for an ending without a purchase ("" when no listing was weighed)."""
        parts = []
        for _, ident, miss in self.closest():
            price = self.price_of(ident)
            parts.append(ident + (f" at {price:g}" if price is not None else "") + (f", which misses {miss}" if miss else ""))
        return ("No listing meets every requirement. Closest: " + "; ".join(parts) + ". Nothing was ordered.") if parts else ""

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The ending's remark: when the request forbids naming alternatives, a plain apology that names nothing;
        otherwise the agent's own words, led by the closest listings (id, price, what each misses) when its words name
        no listing the store showed. A decline that names what came closest lets the shopper see the near-misses."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        sorry = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."
        forbid = self.brief.get("quiet_decline") is True
        note = "" if forbid else self.closest_note()
        for key in self.remark_keys(name):
            spec = props[key]
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = " ".join(str(value or "").split())
            if forbid:
                text = sorry
            elif not any(ident in text for ident in self.rows):
                text = (note + " " + text).strip() if note else (text or sorry)
            limit = spec.get("maxLength")
            out[key] = text[:limit] if isinstance(limit, int) and limit > 0 else text
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
        before = set(self.cart_lines)
        self.note_lines(name, args)
        for ident in set(self.cart_lines) - before:
            self.carted_at[ident] = self.step
        for ident in before - set(self.cart_lines):
            self.carted_at.pop(ident, None)

    def note_lines(self, name: str, args: dict[str, Any]) -> None:
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
        text = _dump(_view(observation))
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
                out.append((cid, "Shown in the side-by-side result above: the desk opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): the interviewer handles every message to the shopper."))
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
                trimmed = f"\n[desk] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
            if not ending and self.room() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
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
                    trimmed += f"\n[desk] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if self.is_refuse(name, args) and self.refused_at >= 0 and self.room() <= self.recovery() and self.cart_ids():
                # a decline soon after a refused checkout: the refusal brought a change to answer, not proof that nothing
                # fits; the desk repairs the cart and checks out instead when it can
                fixed = self.autopilot()
                if fixed:
                    out.append((cid, "Not run (the desk acted instead). " + fixed))
                    continue
            if (self.is_refuse(name, args) and self.refused_at >= 0 and self.room() <= self.recovery() and not self.cart_ids()
                    and self.repairs < 2 and self.room() >= 2):
                # an empty cart and a refused ending that brought a change (often a new budget) with few moves left:
                # read listings whose records show every requirement in force met are ordered instead
                plan = self.lead_plan(self.code_leads(strict=True))
                if plan:
                    self.repairs += 1
                    fixed = self.place(plan, [],
                                       "The ending was refused with a change, and these read listings meet every requirement in force", self.room() < 3)
                    if fixed:
                        out.append((cid, "Not run (the desk acted instead). " + fixed))
                        continue
            if ending and self.is_buy(name, args) and self.room() == 2:
                # two moves left: the page of a line never opened is worth more than the guards (an order counts only
                # for opened pages)
                self.prove_pages(self.cart_ids(), self.tail())
            if ending and self.room() > 2:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    if self.drop_hold and self.room() > self.tail() + 4:
                        self.drop_hold = False
                        cheaper = self.cheaper_after_drop()
                        if cheaper:
                            out.append((cid, "Not run yet (no step used). " + cheaper + "\nCart now: " + self.cart_text()))
                            continue
                    # every line's attributes read before the order (an order counts only for read listings); a line they
                    # show failing a requirement holds the checkout once
                    unread = [i for i in self.cart_ids() if i not in self.full]
                    facts = self.read_facts(unread, self.tail() + 2) if unread else []
                    failing = [i for i in unread if i in self.full and any(self.verdict(self.full[i], need) is False
                               for n, need in self.needs_now() if self.of_item(i, n))]
                    if facts and failing and self.room() > self.tail() + 3:
                        out.append((cid, "Not run yet. The desk read the attributes of cart lines never read (one step each):\n" + "\n".join(facts)
                                    + f"\n{', '.join(failing)} shows a requirement unmet: replace that line with a listing that meets it, then check out.\nCart now: " + self.cart_text()))
                        continue
                    # every line's own page before the order (an order counts only for opened pages, and the page holds the
                    # seller's notes); a note there that sends a property to another listing holds the checkout once
                    pages = self.prove_pages(self.cart_ids(), self.tail() + 1)
                    sent = self.unread_redirects(self.cart_ids()) + [p.split("\n[desk] ", 1)[1] for p in pages if "\n[desk] " in p]
                    held = self.references()
                    refs = [i for i in self.cart_ids() if i in held]
                    if refs and self.room() > self.tail() + 2:
                        out.append((cid, f"Not run yet (no step used). {', '.join(refs)} is the listing a requirement points at (the shopper wants another listing with its value), not the product to buy: replace that line with a listing that has the value, then check out.\nCart now: " + self.cart_text()))
                        continue
                    # the cart's chains are followed by the desk while moves allow; their values decide the lines
                    followed = self.advance_chains(self.tail() + 2, set(self.cart_ids()))
                    if (sent or followed) and self.room() > self.tail() + 3:
                        shown = ("The desk opened the cart's pages (one step each):\n" + "\n".join(pages) + "\n") if pages else ""
                        out.append((cid, "Not run yet. " + shown + " ".join(followed or sent)
                                    + " Check that each cart line still fits with these values, then check out.\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.room() > 3:
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
                        seen = ("The desk read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.spend_cap - 0.004:
                        out.append((cid, "Not run yet (no step used). The desk re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    after_note = self.decline_refused_at > self.searched_at and not self.after_note_held
                    plan = (self.lead_plan(self.code_leads(strict=True)) or self.built_plan()) if after_note else {}
                    missing = [i for i in plan if self.cart_lines.get(i, 0) < plan[i]]
                    # few moves left are enough when a fitting cart is known: putting its lines back and the order
                    if after_note and (self.room() > self.checkout_room() + 1 or (plan and len(missing) + 1 <= self.room())):
                        # a second ending right after one the store sent back with a change, with no search since
                        self.after_note_held = True
                        self.trace.append({"kind": "decline_after_note_held", "plan": plan})
                        out.append((cid, "Not run yet (no step used). The store answered the last ending with a change (see its note), and nothing has been searched since. "
                                    + (("These listings fit the request under the new limits: " + ", ".join(f"{i} x{q}" for i, q in plan.items())
                                        + ". Put them in the cart at those quantities and check out.") if plan else
                                       "Search again under the new limits (each requested item, and the listings set aside before for the old limits), then decide.")))
                        continue
                    # a property sent to another listing decides fits a plain reading misses: those chains first
                    held = self.chain_hold()
                    if held:
                        out.append((cid, held + (" [episode finished]" if self.finished else "")))
                        continue
                    review = self.decline_review()
                    if review:
                        out.append((cid, review + (" [episode finished]" if self.finished else "")))
                        continue
                    if self.room() > 5:
                        if self.leads and self.room() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.room() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], self.spend_cap + 0.006)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and not self.leads and not self.code_held and self.room() > self.tail() + 2:
                # no review named a fit (or none ran): what the records themselves show gets one look, when they show a
                # candidate for every item (a basket with an item nobody can fit may rightly end without a purchase)
                leads = self.code_leads()
                if len(leads) >= len(self.items):
                    self.code_held = True
                    self.trace.append({"kind": "refuse_blocked", "code_leads": leads})
                    out.append((cid, "Not run yet (no step used). By their own records these listings break no requirement in force: "
                                + "; ".join(f"item {k}: {', '.join(v)}" for k, v in leads.items())
                                + ". Put the best one for each item in the cart and check out; end without a purchase only if each of them clearly fails a requirement (say which), then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and self.room() <= 5 and not self.late_held:
                # too late for the review: a decline here usually throws away a built cart because time ran out
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                # a cart taken apart because its total broke the budget in force is no cart to put back
                units = {i: max(self.cart_lines.get(i, 0), next((it.qty for n, it in enumerate(self.items, 1) if self.of_item(i, n)), 1)) for i in lines}
                total = self.cart_total(units) if lines else None
                over = self.budget is not None and total is not None and total > self.budget
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.room() and not over:
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if (self.is_refuse(name, args) and self.room() > self.max_steps // 3 and (self.pushed or self.leads)
                    and self.early_holds < 2 and not self.decline_proof()):
                # more than a third of the steps left and no requested item shown to lack a fit: right declines come
                # later than that (measured: races 14-15, 20 of 20 right declines with at most a third of the steps left)
                self.early_holds += 1
                self.trace.append({"kind": "early_decline_held", "room": self.room()})
                out.append((cid, f"Not run yet (no step used). {self.room()} of {self.max_steps} steps are left, and no requested item has been shown to have no fitting listing "
                            "(for that, enough of its listings must be read and each must break a requirement by its own record). Keep looking: open unread listings of each "
                            "item, page on through its category, search with the property's other names, then decide."))
                continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.room() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.room()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.room() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[desk] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                # a side-by-side reply with no listing record in it is a refusal of its size, even when its error text
                # repeats the ids it could not find
                if not self.listed_ids(observation) & {str(i) for i in args[many]}:
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[desk] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if len(self.records_in(observation)) >= 2 and name not in self.page_tools():
                self.note_search(name, args, observation)
            if name in self.search_seen or name == self.roles.get("search"):
                self.searched_at = self.step
                self.mark_empty(args, observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            if name == self.roles.get("search") and not self.price_key:
                # the rows' price and stock keys, learned once per search tool for every pick (KYO read them only
                # for a cheapest, most or least pick): budget checks in code need them
                self.read_shape(name, observation)
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                record = self.single_record(observation)
                listing = record[0] if record else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(record[1]))
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
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            elif self.is_refuse(name, args) and not reply.get("finished"):
                # a store that turns an ending down is handing over a change (often a new budget) to answer
                self.refused_at = self.step
                self.decline_refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[desk] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.decline_refused_at == self.step and self.is_refuse(name, args):
                # the ending came back with a change: listings already read may fit under it
                plan = self.lead_plan(self.code_leads(strict=True))
                if plan:
                    self.leads = list(plan) + [i for i in self.leads if i not in plan]
                    text += ("\n[desk] Under the store's new limits these read listings meet every requirement in force by their own records: "
                             + ", ".join(f"{i} x{q}" for i, q in plan.items()) + ". Put them in the cart at those quantities and check out.")
            if name == self.roles.get("cart") and not self.finished:
                # a listing just carted gets its page opened now, while moves are plenty: the order needs it, and its
                # seller's notes (a property sent to another listing) are better known before the checkout
                fresh = [i for i in self.cart_ids() if i not in self.opened and i not in self.gone]
                if fresh and self.room() > self.tail() + len(fresh):
                    pages = self.prove_pages(fresh, self.tail() + 2)
                    if pages:
                        text += "\n[desk opened the page of each listing just carted, one step each]\n" + "\n".join(pages)
                sent = self.unread_redirects(self.cart_ids())
                if sent:
                    text += "\n[desk] " + " ".join(sent)
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[desk re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on the desk's own calls] " + " | ".join(self.queued)
                self.queued = []
            if self.is_buy(name, args) and not reply.get("finished") and not self.finished and self.room() <= self.recovery():
                # a refused checkout with few moves left: the desk answers the change itself
                fixed = self.autopilot()
                if fixed:
                    text += "\n" + fixed
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        if out and not self.finished:
            # pages read this turn are read for notes in batches; the chains that matter are followed on spare moves
            extra = self.chain_turn()
            if extra:
                out[-1] = (out[-1][0], out[-1][1] + extra)
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        # the finish runs on the step count: it is what the clock's margin was kept for; past the hard cutoff only the
        # one final call goes out (no cart work, pages, gate or repair)
        if self.finished or self.broke or self.left() <= 0:
            return ""
        self.ending = True
        late = time.monotonic() > self.t0 + TASK_SECONDS - 7
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # the built cart, one line per item, without lines taken out on purpose or sold out since
        built: list[str] = []
        for ident in self.built:
            number = self.item_for(ident, True)
            if self.sold_out(ident) or ident in self.dropped or (number is not None and number in {self.item_for(i, True) for i in built}):
                continue
            built.append(ident)
        refused = ""
        if buy and not holding and self.left() >= 2 and not late:
            # an empty cart at a forced end (steps, turns, money or the clock ran out): an ending without a purchase
            # almost never pays in these stores, so the lines built for every item are ordered, else the read
            # listings whose records break no requirement, one per item
            # strict: an ending nobody chose is still an ending that may be right; only listings whose records show
            # every requirement met are ordered in its place
            plan = ({i: self.qty_for(i) for i in built} if len(built) >= len(self.items)
                    else self.chain_order() if self.chains else self.lead_plan(self.code_leads(strict=True)))
            if plan:
                said = self.place(plan, [], "Ending: the cart was empty, but these lines meet the requirements by their records", hurry=True)
                if self.finished or self.left() <= 0:
                    return ""
                if self.refused_at == self.step:
                    # refused: the same cart is not sent again at once; the repair below answers the refusal
                    refused = said or "the checkout was refused"
                holding = bool(self.cart_lines)
        if buy and holding and self.left() >= 2 and not late and not refused and self.cart_rejected():
            # a cart the check turned down is not ordered; read listings whose records show every requirement met
            # take its place when there are such for every item
            plan = self.lead_plan(self.code_leads(strict=True))
            if plan and set(plan) != set(self.cart_ids()):
                said = self.place(plan, [i for i in self.cart_ids() if i not in plan], "Ending: the cart was turned down, and these lines meet every requirement by their records", hurry=True)
                if self.finished or self.left() <= 0:
                    return ""
                if self.refused_at == self.step:
                    refused = said or "the checkout was refused"
        if buy and not refused and (holding or (self.cart_touched and not fresh and self.left() >= 2)) and not self.cart_rejected():
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # pages first while a step stays for the order; a reply that adds a requirement does not stop a forced
                # finish: no steps are left to act on it
                if not late:
                    # a line's attributes, then its page, while a step stays for the order (an order counts only for
                    # listings read both ways)
                    self.read_facts([i for i in self.cart_ids() if i not in self.full], 2)
                    self.prove_pages(self.cart_ids(), 1, hurry=True)
                if self.room() >= 2 and not late:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    refused = self.show(reply.get("observation"))
        if refused and not late and self.room() <= self.recovery():
            self.find_sold()
            self.autopilot()
            if self.finished or self.left() <= 0:
                return ""
        if refused and again:
            self.ending = False
            return "the checkout was refused: " + refused
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.room() >= 2 and not late:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    self.ending = False
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
        opening = _unpack(self.view.get("opening"))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        # caps grow with the task's step limit; the loop stops a reserve short of the money cap, so the checks of
        # the cart and the ending still run on a long task
        extra = max(0, self.max_steps - LOOP_STEPS)
        self.loop_cap = min(LOOP_MAX, LOOP_CALLS + extra)
        self.call_cap = ALL_CALLS + self.loop_cap - LOOP_CALLS
        self.spend_cap = min(SPEND_LONG, SPEND_HARD + SPEND_STEP * extra)
        self.work(self.loop_cap, self.spend_cap - JUDGE_RESERVE)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.room() > 1:
                self.chat.append({"role": "user", "content": "[desk] The desk tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.spend_cap)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.room() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap:
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
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2:
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
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.room() > 8:
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
        episode = Buyer(problem_data)
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
