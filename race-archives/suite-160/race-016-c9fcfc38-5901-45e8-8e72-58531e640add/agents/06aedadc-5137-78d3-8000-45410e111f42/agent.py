"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
# Spend. A run's tasks share one key capped at $5 (race, 90 tasks) or $2 (qualifying, 30 tasks); once it is spent every
# unfinished task of the run scores 0, and a task cannot see the others' spend. So every task holds itself to TASK_CAP
# (90 x 5.0 cents = $4.50 a race run) under the conditional bound below. SPEND_HARD ends the agent's
# own loop with room left for the checks before an order; past SPEND_SOFT reviews use only the light model.
TASK_CAP = 0.050
SPEND_HARD = 0.042
SPEND_SOFT = 0.032
# Conditional cost bound: one token per serialized byte, plus framing allowance. Prices must remain at or
# below these measured rates, and provider-added framing must fit the allowance. Unknown models are not called.
PRICE = {"anthropic/claude-haiku-5.5": (0.125e-6, 0.50e-6), "anthropic/claude-sonnet-5.5": (2.0e-6, 10.0e-6)}
BYTES_PER_TOKEN = 1.0
FRAME_BYTES = 128
FRAME_TOKENS = 512
OUT_FLOOR = 400
HEAVY_FLOOR = 1200
UNREAD_LIMIT = 6            # search rows the code opens before it gives up on an item with the models out
# the strong model is 70-90% of a task's cost (local runs, 2026-11-03): at most this many of its calls per task, after
# which a job that would take it falls to the next model in its list
HEAVY_LIMIT = 5
SWAP_TOKENS = 2400
LIGHT_REVIEW_TOKENS = 4000      # Haiku 5.5 reasons first: smaller review allowances came back empty (SPEC-006 r14d)
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
# F4 (SPEC-009): an episode has 300 s in all. No model call starts after MODEL_DEADLINE seconds: one call can take
# 50 s and the code still needs a few store calls to check out what it holds
MODEL_DEADLINE = 215
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

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
- The kind of product is decided by the store's category label (its aisle): only a label that names the requested item as a whole counts, never a broader or neighbouring one (a seasonal or general clothing aisle is not the aisle for a top). When the label does name it, a title word or a type attribute naming a style within it does not change that (a blouse filed under sweaters and cardigans counts as a sweater or cardigan).
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. Two keys that differ only in spaces, underscores or case ("dietary needs" and "dietary_needs") are one attribute: when they disagree, a value under either one counts. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words: the accessory's list of what it works with must name the device's model number, a name the device is also sold as, or a series that covers it ("every XY-45xx model", less any exception). "Part P also fits: M" is about the listing whose own code is P, not the page it is on; "See part P for the rest" or "More on part P" means the rest of that listing's list is on part P's page. When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
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
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right for the REQUIREMENTS IN FORCE.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails (two keys that differ only in spaces, underscores or case, such as "dietary needs" and "dietary_needs", are one attribute: when they disagree, one of them stating the value is enough); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole (a broader or neighbouring label, such as a seasonal or general clothing aisle for a top, does not), and when it does, a title word or a type attribute naming a style within it does not change that (a blouse filed under sweaters and cardigans is a sweater or cardigan); a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Show your reading in "evidence": one entry per requested item, {"item": its number (1, 2, ...), "id": the listing's id, "identity": {"path": [keys leading to its category label in the record (its title or type only when the record has no category label)], "actual": that value, "matches": true or false}, "requirements": [one per requirement in force on that item: {"need": the requirement as listed, "path": [keys leading to the value you read in that listing's record], "actual": that value as the record shows it, "op": "ge", "gt", "le", "lt" or "range" for a numeric limit, "eq" or "one_of" for a stated value or closed list, else "semantic", "value": the limit as a number (two numbers for range) or the stated value(s), taken from the requirement, never from the listing (omit for semantic), "unit": the requirement's unit as it is written, or "", "ok": true or false by the reading rules}]}.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first], "evidence": [...]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses. Be concise."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, and two keys that differ only in spaces, underscores or case are one attribute where either value counts; one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item as a whole (a broader or neighbouring label does not), and a title word or a type attribute naming a style within a matching label does not change that; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

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

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


CHECK += """
Also return cart_items: [{"item": one-based integer item number, "id": listing ID}], one distinct listing for
each requested item, even when two items share a category. Do not merge items or quantities. For an observed
cart, optionally return cart_quantity_path: a list of keys locating units within one actual cart-line record
when its quantity field differs from the cart tool's parameter name. Copy actual keys; do not invent a path.
The code verifies counts independently. For a planned cart, omit the observed quantity path."""

AUDIT += """
Distinguish an observed contradiction from missing evidence. A need whose property no opened record addresses
does not prove that nothing fits. Never remove that need. Return unknown: [{"item": one-based item number,
"need": exact active requirement}] for unresolved properties, and open: [promising existing row IDs] when
opening them could resolve it. Fits are suggestions for verification, not permission to buy unverified items."""

# SPEC-009 F2: which requested item must fit which, read with the request (no extra call)
BRIEF += """"fits": a list of {"item": item number, "with": item number}, one for each requested item that the request says has to fit, work with or be compatible with another requested item (an accessory and the device bought with it); an item only named after a kind of device ("a laptop sleeve") is not one; [] when the request says no such thing.
"""


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


def _record_scalar(text: str, preserve_numeric_text: bool = False) -> Any:
    """Only explicit scalar syntax is typed; units and malformed numbers stay text."""
    text = text.strip()
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    if text in ("None", "null"):
        return None
    if len(text) <= 64 and re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", text):
        if preserve_numeric_text:
            return text
        return float(text) if "." in text else int(text)
    if text.startswith(('"', '[')):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (str, list)):
                return parsed
        except (ValueError, RecursionError):
            pass
    return text


def _record_parts(text: str, separator: str) -> list[tuple[int, int]]:
    """Split outside JSON quotes/brackets, retaining source offsets."""
    spans = []
    start = pos = 0
    quoted = escaped = False
    nesting = 0
    while pos < len(text):
        c = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"' and (pos == 0 or text[pos - 1] in "=:[ "):
            # Inch marks in ordinary titles are not the start of a quoted value.
            quoted = True
        elif c in "[{":
            nesting += 1
        elif c in "]}":
            nesting = max(0, nesting - 1)
        elif not nesting and text.startswith(separator, pos):
            spans.append((start, pos))
            pos += len(separator)
            start = pos
            continue
        pos += 1
    spans.append((start, len(text)))
    return spans


def _text_record(text: str, id_key: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Conservative labelled-record grammar, not a reader of arbitrary product prose.

    Original bytes and spans are kept outside product records. Conflicting duplicate fields
    become unknown. A description/blurb is terminal free text; apparent fields inside it
    cannot create attributes. Unsupported layouts remain raw, with an explicit diagnostic.
    """
    info: dict[str, Any] = {"status": "not_record"}
    if " | " not in text:
        return None, info
    header = re.match(r"^([\w][\w .()/+\-]{0,79}):\s*(\S+) \| ", text)
    if header and _id_like(header.group(2), True) and (
            header.group(1) == id_key or set(_words(header.group(1)).split()) &
            {"id", "ref", "reference", "code", "sku"}):
        info = {"status": "unsupported", "raw": text, "reason": "record grammar incomplete"}
    if len(text) > 24000:
        return None, {"status": "unsupported", "raw": text, "reason": "record length limit"}
    pieces = _record_parts(text, " | ")
    fields: list[tuple[str, str, int, int]] = []
    for start, end in pieces:
        part = text[start:end]
        match = re.fullmatch(r"([\w][\w .()/+\-]{0,79}):\s?(.*)", part, re.S)
        if not match:
            if not fields:
                return None, info
            key, value, begin, _ = fields[-1]
            fields[-1] = (key, value + " | " + part, begin, end)
            continue
        key, value = match.group(1).strip(), match.group(2)
        if fields and set(_words(fields[-1][0]).split()) & {"comment", "comments", "note", "notes"}:
            # Narrative values cannot invent arbitrary new product attributes via a pipe.
            # Resume only at a labelled structural field; keep other fragments in the prose.
            structural_key = (set(_words(key).split()) & {"stock", "hand", "available", "availability"}
                              and isinstance(_record_scalar(value), bool))
            if not structural_key:
                prior, previous, begin, _ = fields[-1]
                fields[-1] = (prior, previous + " | " + part, begin, end)
                continue
        fields.append((key, value, start + match.start(2), end))
    if len(fields) < 3:
        return None, info
    # At least one grounded identifier plus a typed field or explicit fact block.
    raw = {k: v.strip() for k, v, _, _ in fields}
    identifiers = [k for k, v in raw.items() if _id_like(v, True) and
                   (k == id_key or any(w in _words(k).split() for w in ("id", "ref", "reference", "code", "sku")))]
    if not identifiers:
        return None, info
    chosen = id_key if id_key in identifiers else identifiers[0] if len(identifiers) == 1 else None
    info = {"status": "unsupported", "raw": text, "fields": {}, "conflicts": []}
    if chosen is None:
        return None, info
    row: dict[str, Any] = {}
    structural = False
    for index, (key, value, start, end) in enumerate(fields):
        words = set(_words(key).split())
        terminal = bool(words & {"description", "blurb"})
        if terminal and value.strip().startswith('"'):
            try:
                terminal = not isinstance(json.loads(value.strip()), str)
            except (ValueError, RecursionError):
                pass
        if terminal:
            # Unquoted prose can contain delimiters and fake key/value pairs. Retain it whole.
            value = text[start:]
            end = len(text)
        parsed = value.strip() if key == chosen else _record_scalar(value)
        if words & {"facts", "attributes", "specifications", "properties", "specs"}:
            block: dict[str, Any] = {}
            spans = {}
            valid = True
            for left, right in _record_parts(value, "; "):
                segment = value[left:right].strip()
                if not segment:
                    continue
                match = re.fullmatch(r"([\w][\w .()/+\-]{0,79})=(.*)", segment, re.S)
                if not match:
                    valid = False
                    break
                # Numeric-looking facts may be version/model labels. Preserve their
                # spelling and precision; numeric consumers can interpret them later.
                prop, val = match.group(1).strip(), _record_scalar(match.group(2), preserve_numeric_text=True)
                if prop in block:
                    block[prop] = None
                    info["conflicts"].append([key, prop])
                else:
                    block[prop] = val
                spans[prop] = [start + left, start + right]
            if valid:
                parsed = block
                info.setdefault("fact_spans", {})[key] = spans
                structural = True
            else:
                parsed = None
                info["conflicts"].append([key, "unparsed block"])
        structural |= isinstance(parsed, (int, float, bool))
        if key in row:
            row[key] = None
            info["conflicts"].append([key])
        else:
            row[key] = parsed
        info["fields"].setdefault(key, []).append([start, end])
        if terminal:
            if index + 1 < len(fields):
                info["terminal_prose"] = {"start": start, "apparent_fields_kept_as_text": len(fields) - index - 1}
            break
    if not structural or not _id_like(row.get(chosen), True) or len(row) < 3:
        return None, info
    info.update(status="parsed", id_key=chosen, ident=row[chosen])
    return row, info


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


def _key_name(text: Any) -> str:
    """SPEC-007 W3: one property name for every spelling of a key: case, `_`/`-`/spaces, a trailing plural and the
    colour/color spelling aside ("dietary_needs" = "dietary needs", "sizes" = "size"). Nothing broader: "available
    sizes" stays another property."""
    out = []
    for word in _words(str(text)).split():
        if len(word) > 3 and word.endswith("es") and word[:-2].endswith(("s", "x", "ch", "sh")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append("color" if word == "colour" else word)
    return " ".join(out)


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
    # Explicit fact containers retain single, boolean and list-valued attributes too.
    for key, value in node.items():
        if isinstance(value, dict) and set(_words(key).split()) & {"facts", "attributes", "specifications", "properties", "specs"}:
            return value or None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


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
        self.started = time.monotonic()     # F4: the episode's clock starts with the agent
        self.time_up_seen = False
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
        self.price_scale: dict[str, float] = {}
        self.heavy_used = 0
        self.byte_rate = BYTES_PER_TOKEN
        self.spent_out = False
        self.loop_spent = False     # the agent's own loop stopped on its spend limit (no model refused anything)
        self.guarded = 0
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
        self.cart_assignment: dict[int, str] = {}
        self.cart_assignment_key = ""
        self.cart_quantity_path: list[str] = []
        self.cart_guard_error = ""
        self.listing_evidence: dict[tuple[int, str], dict] = {}
        self.need_contracts: dict[tuple[int, str, str], str] = {}
        self.choose_receipt = ""
        self.choosing = False
        self.decline_unknown: list[tuple[int, str]] = []
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
        self.contradiction_holds: set[tuple] = set()
        self.literal_held = False
        self.last_cut = False
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
        if stage == "choose":
            return any(p["when"] == stage for p in self.process) and self.choose_receipt != self.cart_need_key()
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def price(self, model: str) -> tuple[float, float]:
        inp, out = PRICE.get(model, (float("inf"), float("inf")))
        scale = self.price_scale.get(model, 1.0)
        return inp * scale, out * scale

    def wire_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Account for the same cache wrappers sent to the provider, including tool-call arguments and metadata."""
        sent = [dict(m) for m in messages]
        if len(sent) > 1:
            for index in (0, len(sent) - 1):
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return sent

    def prompt_tokens(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
        """Conservative text request allowance; provider framing and measured prices remain explicit assumptions."""
        payload = {"messages": self.wire_messages(messages), "tools": tools or []}
        size = len(json.dumps(payload, ensure_ascii=True).encode("utf-8")) + FRAME_BYTES * len(messages)
        return int(size / self.byte_rate) + 1 + FRAME_TOKENS

    def room(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> int:
        """The output tokens this call may use so that, at its worst, the task stays within TASK_CAP; 0 when even a
        useful answer would not fit."""
        if model not in PRICE:
            return 0
        inp, out = self.price(model)
        left = TASK_CAP - self.cost - self.prompt_tokens(messages, tools) * inp
        allowed = min(max_tokens, int(left / out)) if left > 0 else 0
        floor = HEAVY_FLOOR if model == HEAVY else OUT_FLOOR
        return allowed if allowed >= min(max_tokens, floor) else 0

    def worst(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> float:
        inp, out = self.price(model)
        return self.prompt_tokens(messages, tools) * inp + max_tokens * out

    def time_up(self) -> bool:
        """F4 (SPEC-009): past MODEL_DEADLINE seconds no model call starts. The loop then counts as stopped on its
        limit (loop_spent), so the ending checks out what it holds in code (code_order) instead of declining."""
        if time.monotonic() - self.started < MODEL_DEADLINE:
            return False
        if not self.time_up_seen:
            self.time_up_seen = True
            self.loop_spent = True
            self.trace.append({"kind": "deadline", "seconds": round(time.monotonic() - self.started, 1), "step": self.step,
                               "cost": round(self.cost, 5)})
        return True

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if self.no_credit or getattr(self, "code_finishing", False) or self.time_up():
            return None
        for model in models:
            if self.calls >= ALL_CALLS or self.time_up():
                return None
            if model == HEAVY and self.heavy_used >= HEAVY_LIMIT:
                continue
            allowed = self.room(model, messages, tools, max_tokens)
            if not allowed:
                # the next model in the list may be cheaper; none affordable: the task's models are out
                self.guarded += 1
                if self.guarded == 1:
                    self.trace.append({"kind": "spend_guard", "model": model, "cost": round(self.cost, 5),
                                       "worst_prompt": round(self.prompt_tokens(messages, tools) * self.price(model)[0], 5)})
                continue
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, allowed, tools)
                if message is not None:
                    return message
                if status in (402, 403):
                    # the key is spent (its run limit or the account's credit): no call can succeed any more, so no
                    # retry and no wait
                    self.no_credit = True
                    return None
                # rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL
                        or self.time_up()):
                    break
                # spread the waits without randomness: the strategy stays deterministic (F29)
                pause = WAITS[tries] * (0.9 + 0.1 * (self.calls % 3))
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
                # a timed-out call was booked at its worst: the resend must fit again
                allowed = self.room(model, messages, tools, max_tokens)
                if not allowed:
                    break
        if not any(self.room(m, messages, tools, max_tokens) for m in models):
            self.spent_out = True
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = self.wire_messages(messages) if model.startswith("anthropic/") else messages
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        worst = self.worst(model, messages, tools, max_tokens)
        bound = self.prompt_tokens(messages, tools)
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            # a reply without its charge is booked at the call's worst case
            paid = float(usage["cost"]) if isinstance(usage.get("cost"), (int, float)) else worst
            self.cost += paid
            self.heavy_used += model == HEAVY
            if paid > worst * 1.000001:
                # the price table is wrong for this model: this call passed its bound, the next ones are priced at
                # the rate it was charged
                self.price_scale[model] = self.price_scale.get(model, 1.0) * paid / worst
                self.trace.append({"kind": "over_bound", "model": model, "paid": round(paid, 5), "worst": round(worst, 5)})
            used = usage.get("prompt_tokens")
            if isinstance(used, int) and used > bound - FRAME_TOKENS:
                # denser prompts than measured: count fewer bytes per token from now on
                self.byte_rate = min(self.byte_rate, 0.9 * (bound - 1 - FRAME_TOKENS) * self.byte_rate / used)
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            # no answer: when the model spent the whole allowance (on reasoning), the reply ends with "length"
            self.last_cut = data["choices"][0].get("finish_reason") == "length"
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0, "cut": self.last_cut})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        if status in (None, 408):
            # no reply in time: whether it was charged is unknown, so it is booked at its worst case
            self.cost += worst
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
        if name == self.roles.get("cart") and self.gate_due("choose"):
            if self.choosing:
                self.cart_guard_error = "Cart held: a required prerequisite has not been performed."
                return None
            # the gate runs once per change of the requirements (its receipt) and again only after a new store step;
            # a gate that cannot run (no model left) does not hold the cart (C2r4)
            self.choosing = True
            try:
                ready, reason = self.gate("choose")
            finally:
                self.choosing = False
            self.trace.append({"kind": "choose_gate", "ok": ready, "reason": reason})
            if not ready:
                self.cart_guard_error = "Cart held: " + reason
                return None
        if self.is_buy(name, args):
            if not self.reconcile_quantities():
                return None
            carted = self.cart_at
            for stage in ("choose", "order"):
                # the checkout's own boundary: a due step gets its gate (no new model call when it already failed for
                # this cart and nothing happened since; none at all when no model is left, which lets it pass)
                if self.gate_due(stage) and not self.choosing and not self.gate(stage)[0]:
                    self.cart_guard_error = "Checkout held: required process steps have no current execution evidence."
                    self.trace.append({"kind": "process_hold", "stage": stage, "reason": self.cart_guard_error})
                    return None
            if self.cart_at != carted:
                self.cart_guard_error = "Checkout held: the cart changed while the required steps were done; check out again."
                return None
            if self.gate_due("choose") or self.gate_due("order"):
                self.cart_guard_error = "Checkout held: required process steps have no current execution evidence."
                self.trace.append({"kind": "process_hold", "reason": self.cart_guard_error})
                return None
            if not self.checkout_evidence_ready():
                return None
        if self.is_refuse(name, args) and self.decline_unknown:
            self.trace.append({"kind": "unverified_end", "unknown": self.decline_unknown,
                               "reason": "unable to verify a purchase; missing evidence is not proof of infeasibility"})
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
                if name != self.roles.get("ask"):
                    seen = self.adapt_observation(seen)
                    data = {**data, "observation": seen}
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                if name != self.roles.get("ask"):
                    # the process record holds every store step the episode took, whoever chose it (the loop, the
                    # code or the gate); messages are entered with their reply where they are sent
                    change = " (this call changes the cart)" if name == self.roles.get("cart") else ""
                    self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:160]}{change}")
                    self.done_steps.append(self.step)
                self.cap_note(name, args, seen)   # F3 (SPEC-009)
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
        self.cap_tools = self.cap_schema()   # F3 (SPEC-009): a search's maximum-price parameter
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
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            choose = sum(p["messages"] for p in self.process if p["when"] == "choose")
            ending = max(sum(p["messages"] for p in self.process if p["when"] == stage) for stage in ("order", "end"))
            self.reserve = min(choose + ending, self.msg_limit)

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
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    # ---------- listings ----------
    def adapt_observation(self, observation: Any) -> Any:
        """One bounded adapter used at the tool boundary and by replayed helper inputs."""
        if not hasattr(self, "record_cache"):
            self.record_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
            self.record_sources: dict[str, list[dict[str, Any]]] = {}
            self.record_issues: list[dict[str, Any]] = []

        def walk(node: Any, path: list[Any], depth: int) -> Any:
            if depth > 6:
                return node
            if isinstance(node, str):
                if node in self.record_cache:
                    record, info = self.record_cache[node]
                else:
                    record, info = _text_record(node, self.id_key)
                    if info["status"] != "not_record":
                        if len(self.record_cache) >= 128:
                            self.record_cache.pop(next(iter(self.record_cache)))
                        self.record_cache[node] = (record, info)
                        self.trace.append({"kind": "record_decode", "status": info["status"],
                                           "conflicts": info.get("conflicts", []), "step": self.step})
                if record is not None:
                    self.id_key = self.id_key or info["id_key"]
                    ident = info["ident"]
                    sources = self.record_sources.setdefault(ident, [])
                    if not any(s["raw"] == node for s in sources):
                        sources.append({**info, "path": path, "step": self.step})
                        del sources[:-4]
                    return record
                if info["status"] != "not_record" and not any(x["raw"] == node for x in self.record_issues):
                    self.record_issues.append({"raw": node, "path": path, "step": self.step, "status": info["status"]})
                    del self.record_issues[:-16]
                return node
            if isinstance(node, dict):
                # A structured listing is already authoritative: do not reinterpret its prose fields.
                if self.ident(node, True, depth):
                    return node
                return {k: walk(v, path + [k], depth + 1) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v, path + [i], depth + 1) for i, v in enumerate(node)]
            return node

        return walk(observation, [], 0)

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

    def absorb(self, observation: Any, cart_record: bool = False) -> None:
        observation = self.adapt_observation(observation)
        # keys each listing has shown outside the cart (search rows, pages): its product fields. A key seen only on
        # cart lines (the line's units, its line total) is cart bookkeeping and stays out of the product record.
        product_keys: dict[str, set[str]] = getattr(self, "product_keys", None) or {}
        self.product_keys = product_keys

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
                    # Partial views cannot erase previously observed facts. New explicit conflicts
                    # (None) remain unknown; only absent fields inherit earlier evidence.
                    merged = dict(node)
                    if not cart_record:
                        for key, value in node.items():
                            if isinstance(value, dict) and isinstance(old.get(key), dict):
                                merged[key] = {**old[key], **value}
                    self.rows[ident].update(merged)
                    keys = product_keys.setdefault(ident, set())
                    if not cart_record:
                        keys.update(node)
                    # the product record merges every search row and page seen of the listing (C2r4), newest value
                    # per field; a cart line still updates the fields the listing already showed (price, stock)
                    if (_fact_block(node) or ident in self.full) and keys:
                        self.full[ident] = {k: v for k, v in self.rows[ident].items() if k in keys}
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
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
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
        """Listing records across bounded result envelopes, using the shared adapter."""
        observation = self.adapt_observation(observation)
        found: list[tuple[str, dict[str, Any]]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, True, depth)
                if ident:
                    found.append((ident, node))
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        return found

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
        observation = self.adapt_observation(observation)
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
        observation = self.adapt_observation(observation)
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
            texts.append(f"{param}={_dump(value)}: {shown_text}")
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

    # ---------- SPEC-009 F3: price-cap sweep for cheapest picks ----------
    # Rows come in no price order and many worlds cap paging (about 21-80 rows reachable), so a search with the store's
    # maximum price set just below the pick lists only cheaper listings, including ones paging cannot reach. Lowered
    # below each new pick until nothing cheaper fits, it finds the cheapest fit. Code only: store calls and no model
    # call (the checkout's own CHECK reviews a swapped cart).
    CAP_SWEEPS = 3          # capped searches per cart line
    CAP_OPENS = 4           # listings per cart line whose attributes the sweep reads
    CAP_KEEP = 6            # steps kept past the finish: a swap (2), the new line's re-read, the cart view, 2 spare
    _CAP_STRONG = frozenset({"cap", "ceiling", "upper", "under", "below"})
    _CAP_MAX = frozenset({"max", "maximum", "most", "to", "top", "limit", "highest", "upto"})
    _CAP_PRICE = frozenset({"price", "prices", "cost", "costs", "amount", "spend", "budget", "total"})
    _CAP_FILLER = frozenset({"bound", "value", "filter", "level", "at", "of", "the", "in", "cents", "usd", "dollars", "eur"})
    _CAP_LABEL = ("shelf", "category", "categories", "section", "department", "dept", "aisle", "group", "range",
                  "class", "family", "collection", "label")
    _CAP_BRAND = frozenset({"brand", "make", "maker", "vendor", "seller", "dealer", "manufacturer", "marque", "shop",
                            "store", "merchant", "outlet"})
    _CAP_PAGING = frozenset({"page", "token", "next", "after", "cursor", "offset", "start", "begin", "skip", "continue",
                             "resume", "bookmark", "marker", "first", "from", "row"})
    _CAP_TEXT = frozenset({"q", "query", "keyword", "keywords", "phrase", "text", "terms", "term", "search", "words",
                           "string", "needle", "wanted", "freetext", "fulltext", "free"})

    def cap_schema(self) -> list[tuple[str, str, str]]:
        """Run where the tool roles are learned: the tools taking a maximum price, read from each schema by the
        parameter's name (max_price, price_max, max_cost, upper, price_to, ceiling, under, cap and their spellings)
        and numeric type: (tool, parameter, "integer" or "number"), the search tool first. Single-listing, cart,
        message and finishing tools are left out. Also starts the sweep's bookkeeping."""
        self.cap_calls: list[tuple[str, dict[str, Any], Any]] = []
        self.cap_done: set[tuple[str, str]] = set()
        self.cap_bad: set[str] = set()
        busy = {tool for job, tool in self.roles.items() if job != "search"}
        allowed = self._CAP_STRONG | self._CAP_MAX | self._CAP_PRICE | self._CAP_FILLER
        found: list[tuple[str, str, str]] = []
        for name, schema in self.schemas.items():
            if name in busy:
                continue
            for key, spec in ((schema or {}).get("properties") or {}).items():
                kinds = _kinds(spec) if isinstance(spec, dict) else []
                words = set(_words(key).split())
                if not words or not words <= allowed or "enum" in spec or not {"integer", "number"} & set(kinds):
                    continue
                if words & self._CAP_STRONG or (words & self._CAP_MAX and words & self._CAP_PRICE):
                    found.append((name, key, "number" if "number" in kinds else "integer"))
                    break
        found.sort(key=lambda entry: entry[0] != self.roles.get("search"))
        if found:
            self.trace.append({"kind": "price_cap_schema", "tools": found})
        return found

    def cap_note(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Every call of a tool that takes a maximum price is kept (the newest 40), so the search that showed a cart
        line can be run again with a cap."""
        tools = getattr(self, "cap_tools", None)
        if not tools or name not in {tool for tool, _, _ in tools} or not isinstance(args, dict):
            return
        if isinstance(observation, dict) and observation.get("error"):
            return
        self.cap_calls = (self.cap_calls + [(name, dict(args), observation)])[-40:]

    def cap_param(self, key: str) -> str | None:
        """What a search parameter is, by its name: "brand", "paging", "text" or "label" (the aisle), else None."""
        words = set(_words(key).split())
        if words & self._CAP_BRAND:
            return "brand"
        if words & self._CAP_PAGING:
            return "paging"
        if words & self._CAP_TEXT:
            return "text"
        if words & set(self._CAP_LABEL):
            return "label"
        return None

    def cap_label(self, ident: str) -> str | None:
        """The listing's aisle label as the store writes it: a short text field whose key names an aisle, category,
        department, section, shelf or group (r3/r10 department, r5 item_shelf, r8/r13 category); None when no such
        field was seen (the data-driven category_key() can settle on titles when rows carry no label)."""
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            for key, value in record.items():
                if (isinstance(value, str) and 0 < len(value.strip()) <= 60 and not set(_words(key).split()) & self._CAP_BRAND
                        and set(_words(key).split()) & {"aisle", "category", "department", "dept", "section", "shelf", "group"}):
                    return value.strip()
        return None

    def cap_aisle(self, ident: str) -> str | None:
        label = self.aisle_of(ident) or self.cap_label(ident)
        return " ".join(label.casefold().split()) if label else None

    def cap_value(self, price: float, kind: str) -> int | float | None:
        """A cap just below the price: one unit when the store's prices are whole numbers (always for an integer
        parameter), else one hundredth; None when nothing positive is left."""
        whole = all(float(p).is_integer() for p in (self.code_price(i) for i in self.rows) if p is not None)
        if kind == "integer":
            cap: int | float = int(-(-price // 1)) - 1
        elif whole and float(price).is_integer():
            cap = int(price) - 1
        else:
            cap = round(price - 0.01, 2)
        return cap if cap > 0 else None

    def cap_forms(self, line: str) -> list[tuple[str, dict[str, Any], str, str, str]]:
        """Ways to list the line's kind under a cap, best first: (tool, arguments without the cap, cap parameter, its
        kind, a key naming the form). First the aisle browse (the tool's aisle parameter set to the line's own label,
        in a mode other than free text when the tool has modes: free-text modes ignored the cap in the logged worlds);
        then the line's own search run again (its paging and any earlier cap left out)."""
        forms: list[tuple[str, dict[str, Any], str, str, str]] = []
        label = self.cap_label(line)
        for tool, key, kind in self.cap_tools:
            schema = self.schemas.get(tool) or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
            shelves = sorted((k for k in texts if self.cap_param(k) == "label"),
                             key=lambda k: min(self._CAP_LABEL.index(w) for w in _words(k).split() if w in self._CAP_LABEL))
            # a search-mode switch: an enum with a free-text value and another one (r3 text/narrow, r10 text/criteria)
            modes = [k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list)
                     and all(isinstance(o, str) for o in s["enum"])
                     and any(set(_words(o).split()) & self._CAP_TEXT for o in s["enum"])
                     and any(not set(_words(o).split()) & self._CAP_TEXT for o in s["enum"])]
            if label and shelves:
                browse: dict[str, Any] = {shelves[0]: label}
                for mode in modes:
                    plain = [o for o in props[mode]["enum"] if not set(_words(o).split()) & (self._CAP_TEXT | self._CAP_BRAND)]
                    plain.sort(key=lambda o: not set(_words(o).split()) & (set(self._CAP_LABEL) | {"criteria", "narrow", "browse", "filter"}))
                    if plain:
                        browse[mode] = plain[0]
                forms.append((tool, browse, key, kind, tool + _dump(sorted((k, v if k in modes else "*") for k, v in browse.items()))))
            shape = self.shapes.get(tool) or {}
            for name, args, seen in reversed(self.cap_calls):
                if name != tool or not _in_rows(seen, line):
                    continue
                base = {k: v for k, v in args.items() if k != key and k != shape.get("next_arg")
                        and (self.cap_param(k) != "paging" or k in required)}
                forms.append((tool, base, key, kind, tool + _dump(sorted((k, v if k in modes else "*") for k, v in base.items()))))
                break
        out, seen_forms = [], set()
        for form in forms:
            mark = form[0] + _dump(form[1])
            if form[4] not in self.cap_bad and mark not in seen_forms:
                seen_forms.add(mark)
                out.append(form)
        return out

    def cap_reads(self, number: int, ident: str) -> bool:
        """The code reads every requirement in force on item `number` as met by the opened listing's attributes: as
        code_fits reads it (the value's words under a key naming the property), or as W3's literal reading (closed
        lists, exclusions), or, for a numeric limit the need's words state, a value inside it in the same unit."""
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        if not block:
            return False
        if self.code_fits(number, ident):
            return True
        fields = self.literal_fields(record)
        for n, need in self.needs_now():
            if n != number:
                continue
            prop, _, value = need.partition(":")
            keys = [k for k in block if _names(_stem(prop), _stem(k))]
            words = set(_words(value).split())
            if words and any(words <= set(_words(str(block[k])).split()) for k in keys):
                continue
            if self.literal_states(need, fields):
                continue
            limit = self.need_limit(need, {})
            if limit and any(self.numeric_fit(block[k], limit[0], limit[1], limit[2], k) == "in" for k in keys):
                continue
            return False
        return True

    def cap_fits(self, number: int, line: str, ident: str) -> bool:
        """The fit evidence the code already uses: the opened listing is in the line's aisle, buyable, contradicts no
        requirement and no current review rejected it; the code reads every requirement as met (cap_reads); and its
        own attributes state at least as many of the item's requirements as the line's (literal count). The
        checkout's CHECK then reviews the swapped cart."""
        if self.cap_aisle(ident) != self.cap_aisle(line) or not self.buyable(ident) or self.contradicted([ident]):
            return False
        saved = self.listing_evidence.get((number, ident), {})
        if saved.get("ok") is False and saved.get("state") == self.evidence_state(ident):
            return False
        return self.cap_reads(number, ident) and self.literal_count(number, ident)[0] >= self.literal_count(number, line)[0]

    def cap_walk(self, number: int, line: str) -> tuple[str, list[str], int, int]:
        """For one cart line: search under a cap just below the line's price, read the cheapest new rows under it
        (cheapest first) and keep the first that fits; then search again below that one, until nothing cheaper fits,
        CAP_SWEEPS searches or CAP_OPENS readings are spent, or the steps left reach the floor. A form whose rows break
        its cap (the store ignored it) is dropped for the episode. Returns (the pick, what was done, searches, reads)."""
        price = self.code_price(line)
        aisle = self.cap_aisle(line)
        if price is None or not aisle:
            return line, [], 0, 0
        if not self.cap_reads(number, line):
            # requirements the code cannot read on the line itself (a pointer to another listing, a semantic or
            # unit-converted value) it cannot read on a cheaper listing either: no steps spent
            return line, ["the code cannot read the line's own requirements"], 0, 0
        qty = self.items[number - 1].qty
        rest = {i: self.cart_lines.get(i) or 1 for i in self.cart_ids() if i != line}
        if not self.cart_calls({**rest, line + "#new": qty}, [line]):
            # the cart tool cannot take this line out in code (an add-only list): a swap could not follow
            return line, ["the cart tool cannot swap this line"], 0, 0
        forms = self.cap_forms(line)
        best, searches, opens, notes = line, 0, 0, []
        while (forms and searches < self.CAP_SWEEPS and opens < self.CAP_OPENS and not self.finished
               and self.left() > self.tail() + self.CAP_KEEP + 1):
            tool, base, key, kind, form = forms[0]
            cap = self.cap_value(price, kind)
            if cap is None:
                break
            faults: list[str] = []
            args = _shape({**base, key: cap}, self.schemas.get(tool), "args", faults)
            if faults:
                forms.pop(0)
                continue
            reply = self.act(tool, args)
            searches += 1
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            prices = {i: self.code_price(i) for i, _ in self.records_in(seen)}
            if any(p is not None and p > cap for p in prices.values()):
                self.cap_bad.add(form)
                forms.pop(0)
                notes.append(f"{tool} {_dump(args)}: rows above the cap (the store ignored it)")
                continue
            pool = sorted((p, i) for i, p in prices.items() if p is not None and p < price and i != line
                          and i not in self.cart_ids() and self.buyable(i) and not self.contradicted([i])
                          and self.cap_aisle(i) in (None, aisle) and self.code_budget_ok({**rest, i: qty}))
            notes.append(f"{tool} {_dump(args)}: {len(prices)} rows, {len(pool)} cheaper")
            fit = None
            for _, ident in pool:
                if ident not in self.full:
                    if opens >= self.CAP_OPENS or self.left() <= self.tail() + self.CAP_KEEP:
                        break
                    self.fill_facts([ident], self.tail() + self.CAP_KEEP)
                    opens += 1
                now = self.code_price(ident)
                if ident in self.full and now is not None and now < price and self.cap_fits(number, line, ident):
                    fit = ident
                    break
            if not fit:
                break
            best, price = fit, self.code_price(fit) or price
            notes.append(f"fits: {fit} at {price:g}")
        return best, notes, searches, opens

    def cap_lines(self, ids: list[str]) -> dict[int, str]:
        """Which cart line holds which item: the current assignment, the lone line of a lone item, else lines whose
        aisle matches one item's shelf or name words (one line per item)."""
        if len(self.items) == 1 and len(ids) == 1:
            return {1: ids[0]}
        mapping = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        if mapping and set(mapping.values()) <= set(ids):
            return dict(mapping)
        out: dict[int, str] = {}
        for number, item in enumerate(self.items, 1):
            long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            match = [i for i in ids if i not in out.values() and self.cap_aisle(i)
                     and (self.cap_aisle(i) == _words(item.shelf or "") or {w for w in _stem(self.cap_aisle(i) or "") if len(w) >= 4} & long)]
            if len(match) == 1:
                out[number] = match[0]
        return out

    def cap_swap(self, number: int, line: str, best: str) -> bool:
        """Put `best` in the cart instead of `line` (the new line first, so a failed call never leaves the item
        without one), within the budget and the steps kept for the finish."""
        cart = self.roles.get("cart")
        if not cart or self.finished or self.sold_out(best):
            return False
        plan = {i: self.cart_lines.get(i) or 1 for i in self.cart_ids() if i != line}
        plan[best] = self.items[number - 1].qty
        if not self.code_budget_ok(plan):
            return False
        calls = self.cart_calls(plan, [line])
        if not calls or len(calls) + self.tail() + 2 > self.left():
            return False
        calls.sort(key=lambda call: line in [v for v in call.values() if isinstance(v, str)])
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.cart_said = _dump(reply.get("observation"))
            self.observe(cart, reply.get("observation"))
        self.dropped.add(line)
        self.dropped.discard(best)
        self.built = [best if i == line else i for i in self.built]
        if self.cart_assignment_key == self.cart_need_key() and self.cart_assignment.get(number) == line:
            self.cart_assignment = {**self.cart_assignment, number: best}
        return True

    def cap_sweep(self) -> list[str]:
        """Before a cheapest checkout, once per cart line and state of the requirements: the price-cap walk for each
        line, and a cheaper listing that fits replaces the line. Nothing without a cap parameter, for other picks, when
        the code finishes alone or no model is left, past the loop's own spend limit (the sweep calls no model; the
        swapped cart's CHECK uses the light model past SPEND_SOFT) or with few steps left. Returns what changed, for
        the checkout's reply."""
        if (not getattr(self, "cap_tools", None) or self.brief.get("pick") != "cheapest" or not self.roles.get("cart")
                or getattr(self, "code_finishing", False) or self.no_credit or self.spent_out or self.broke or self.finished
                or self.cost >= SPEND_HARD or self.left() <= self.tail() + self.CAP_KEEP + 1):
            return []
        ids = self.cart_ids()
        needs = self.cart_need_key()
        texts = []
        for number, line in sorted(self.cap_lines(ids).items()):
            # once per line and state of the requirements: a walk already ended where nothing cheaper fits
            if line not in ids or self.sold_out(line) or self.finished or (line, needs) in self.cap_done:
                continue
            self.cap_done.add((line, needs))
            was = self.code_price(line)
            best, notes, searches, opens = self.cap_walk(number, line)
            swapped = best != line and self.cap_swap(number, line, best)
            if swapped:
                self.cap_done.add((best, needs))
            self.trace.append({"kind": "price_cap", "item": number, "line": line, "to": best if swapped else None,
                               "searches": searches, "opens": opens, "notes": notes[:8]})
            if swapped:
                texts.append(f"[Kyo searched below the cart's price with the store's maximum-price filter] {best}"
                             f" ({self.code_price(best):g}) states every requirement of item {number} and costs less than {line}"
                             f" ({was:g}): Kyo put {best} in the cart instead.")
        return texts

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        observation = self.adapt_observation(observation)
        self.absorb(observation, name in (self.roles.get("cart"), self.roles.get("cart_view")))
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
        # turns between the item groups, up to two each; with a cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[:5], 22)

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
        """SPEC-009 F1: questions go out one at a time and each reply is read (light model) before the next is chosen.
        Basket lines with no need heard yet take turns, so every line is asked once before any line is asked twice. A
        reply that states no need (garbled, or the shopper does not mind) moves that line on to its next property; a
        property is never asked twice for a line, and a line whose need was heard is not asked again. Asking stops
        when every line has its need, or the messages, the step floor, the spend or the reader run out."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        queues, turn = self.question_queues(self.msgs_free())
        tried: dict[int, set[str]] = {number: set() for number in queues}
        needs_in: set[int] = set()
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        pending = self.question_obligations(queues, turn)
        while (self.msgs_free() > 0 and self.left() > 6 and not self.finished and not self.broke
               and not self.no_credit and self.cost < SPEND_HARD):
            choice = None
            while pending and choice is None:
                number, label = pending.pop(0)
                if _key_name(label) not in tried[number]:
                    choice = (number, label)
            if choice is None:
                choice = self.next_question(queues, turn, tried, needs_in)
            if choice is None:
                break
            number, label = choice
            tried[number].add(_key_name(label))
            before = [list(item.told) for item in self.items]
            batch = self.send([(number, label, self.question(number, label))])
            if not batch:
                break
            sent += batch
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            fresh = self.fresh_needs(before)
            needs_in.update(fresh)
            self.trace.append({"kind": "adaptive_question", "item": number, "property": label, "heard": fresh,
                               "read": bool(heard), "messages_left": self.msgs_free(), "steps_left": self.left()})
            if not heard and (self.no_credit or self.spent_out or self.calls >= ALL_CALLS):
                break       # the reply went unread for want of a model: a further one would too
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def question_obligations(self, queues: dict[int, list[str]], turn: list[int]) -> list[tuple[int, str]]:
        """(item, property) for the planner's own property labels that the request explicitly says to check with the
        shopper (the brief's ask_first, free text such as "favourite protein flavour" or "colour preferences before
        buying"): a label counts when its words appear in one such text. Each is asked once, before the per-line
        turns, whether or not that line's need was already heard: a heard need does not discharge an explicit
        instruction to ask. Text no label matches ("the shopper's preferences") adds no question."""
        texts = []
        for text in _items(self.brief.get("ask_first")):
            words = {_key_name(w) for w in _words(str(text)).split()}
            if words:
                texts.append(words)
        if not texts:
            return []
        out: list[tuple[int, str]] = []
        for number in turn:
            for label in queues.get(number, []):
                words = {_key_name(w) for w in _words(label).split()}
                if words and any(words <= text for text in texts) and (number, label) not in out:
                    out.append((number, label))
        return out

    def question_queues(self, count: int) -> tuple[dict[int, list[str]], list[int]]:
        """One PLAN call read into a ranked list of properties per item, and the item order (the order in which items
        first appear in the plan, then the rest). Each list ends with the item's own observed keys, so a line whose
        planned properties drew no need still has a next property to ask."""
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
        # one pick per message and one more per item: a line whose first property draws no need moves to its next
        prompt = PLAN.replace("COUNT", str(count + len(self.items))) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        queues: dict[int, list[str]] = {number: [] for number in range(1, len(self.items) + 1)}
        turn: list[int] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_key_name(label) != _key_name(other) for other in queues[number]):
                queues[number].append(label)
                if number not in turn:
                    turn.append(number)
        turn += [number for number in queues if number not in turn]
        for number, key in self.spare_properties(table):
            if all(_key_name(key) != _key_name(other) for other in queues[number]):
                queues[number].append(key)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queues": {n: q[:6] for n, q in queues.items()}})
        return queues, turn

    def spare_properties(self, table: list[tuple[str, str, int, list[str]]]) -> list[tuple[int, str]]:
        """(item, key) for the keys seen on opened listings of each item's own category, most carried first, leaving
        out identifiers and logistics and the properties the request already settles. With several items a category
        must match the item's label or name; keys of an unmatched category belong to no item."""
        logistics = r"\b(?:id|sku|code|model|serial|warranty|shipping|delivery|stock|availability|price|cost|rating|reviews?|seller|shop|store|url|image|title)\b"
        many = len(self.items) > 1
        out: list[tuple[int, str]] = []
        for label, key, _, _ in sorted(table, key=lambda row: -row[2]):
            if re.search(logistics, key):
                continue
            for number, item in enumerate(self.items, 1):
                if many and not self.shelf_matches(item, label):
                    continue
                if any(_names(_stem(must.split(":", 1)[0]), _stem(key)) for must in item.musts):
                    continue
                out.append((number, key))
        return out

    def shelf_matches(self, item: Item, label: str) -> bool:
        """The category label names this item (key_table's own matching rule, for one item)."""
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        return _words(label) in (_words(item.shelf or ""), _words(item.name)) or bool({w for w in _stem(label) if len(w) >= 4} & long)

    def next_question(self, queues: dict[int, list[str]], turn: list[int], tried: dict[int, set[str]],
                      needs_in: set[int]) -> tuple[int, str] | None:
        """The next (item, property): among the lines whose need is not heard yet, the one asked least so far (ties in
        plan order), with its best property not asked yet and not settled since (a need heard on it, or the shopper
        said they do not mind)."""
        best: tuple[tuple[int, int], int, str] | None = None
        for place, number in enumerate(turn):
            if number in needs_in:
                continue
            item = self.items[number - 1]
            closed = set(tried[number])
            closed.update(_key_name(need.split(":", 1)[0]) for need in item.told if need not in item.dropped)
            closed.update(_key_name(str(prop).split(":", 1)[0]) for prop in item.free)
            label = next((name for name in queues[number] if _key_name(name) not in closed), None)
            if label is None:
                continue
            rank = (len(tried[number]), place)
            if best is None or rank < best[0]:
                best = (rank, number, label)
        return (best[1], best[2]) if best else None

    def fresh_needs(self, before: list[list[str]]) -> list[int]:
        """Items whose ledger gained a need from the last reply; a need that only repeats the request does not count."""
        return [number for number, item in enumerate(self.items, 1)
                if any(need not in before[number - 1] and need not in item.musts for need in item.told)]

    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.items[number - 1].name}?"

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
        """Right before choosing ("choose"), a checkout ("order") or an ending without a purchase ("end"): the steps the
        request asks for at that point, done by the model's choice of store calls (a message to the shopper, a page, any
        store tool) until it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the
        agent. A step counts as done only on a numbered RECORD line of a real store step; a gate that cannot run (no
        model left, no usable reply, no message or step left for the missing step) lets the agent go on (C2r4)."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return True, ""
        said: list[str] = []
        made: set[str] = set()
        nudged = False

        def passed(why: str = "") -> tuple[bool, str]:
            self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
            if stage == "choose":
                self.choose_receipt = self.cart_need_key()
            if why:
                self.trace.append({"kind": "gate_waived", "stage": stage, "reason": why})
            return True, "\n".join(said)

        def missing(required: list[dict[str, Any]]) -> bool:
            # the missing steps cannot be done with what is left: a message step without a message, or no step
            return self.left() < 2 or all(p["messages"] > 0 and self.msg_used >= self.msg_limit for p in required)

        if (self.no_credit or self.broke or getattr(self, "code_finishing", False) or self.calls >= ALL_CALLS
                or self.cost >= TASK_CAP):
            # no model can run the gate (the credit, the spend cap, the code's own finish): C2r4 goes on
            return passed("no model can run the gate")
        if self.left() < 2:
            # no step is left to do a missing step: the gate would let the agent go on whatever it found
            return passed("no step left for the steps")
        key = (stage, self.cart_need_key() if stage == "choose" else self.cart_mark() + f"#{self.version}")
        failed: dict[tuple[str, str], tuple[int, str]] = getattr(self, "gate_failed", None) or {}
        self.gate_failed = failed
        if key in failed and failed[key][0] == self.step:
            # the gate found these steps missing for this cart (these requirements) and no store step happened since:
            # the same question would get the same answer, so no new model call
            return False, failed[key][1]
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        if stage == "choose" and self.roles.get("cart"):
            ending.add(self.roles["cart"])
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        unproven: list[dict[str, Any]] = list(steps)
        for _ in range(GATE_ROUNDS):
            prompt = (GATE.replace("STAGE", "choose products" if stage == "choose" else "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW (numbered):\n" + "\n".join(f"{n}. {p['step']}" for n, p in enumerate(steps, 1))
                      + "\nIn \"steps\" give one entry for each numbered step, in that order, with \"number\": its number."
                      + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                      + f"\nTHE CART LAST CHANGED AT STEP: {self.cart_at if self.cart_touched else 'never'}"
                      + "\n\nRECORD (every store step and message so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending' if stage == 'end' else 'cart and the order'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                # no model can answer (spend cap, credit, the code's own finish, no usable reply): C2r4 goes on
                return passed("no model reply")
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
            # message is about the final cart, so its line must come after the cart last changed. Receipts are matched
            # to the steps by their number (or exact text, or position when one entry is given per step): the model's
            # wording of a step is not checked
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if i <= len(self.done_steps) and self.done_steps[i - 1] > 0
                     and (not talk or self.done_steps[i - 1] > self.cart_at)}
            # a step that takes a message is done only by a message line (send() writes those), not by another store step
            spoken = {i for i in proof if ": message to the shopper " in self.done_log[i - 1]}
            claimed: dict[int, list[int]] = {}
            for number, required in enumerate(steps, 1):
                claimed[number] = [k for k, e in enumerate(shown) if number in (_count(e.get("number")), _count(e.get("step")))]
                if not claimed[number]:
                    claimed[number] = [k for k, e in enumerate(shown) if isinstance(e.get("step"), str) and e["step"].strip() == required["step"]]
            taken = {k for ks in claimed.values() for k in ks}
            unproven = []
            for number, required in enumerate(steps, 1):
                picks = claimed[number]
                if not picks and len(shown) == len(steps) and number - 1 not in taken:
                    picks = [number - 1]
                evidence = [shown[k] for k in picks]
                valid = spoken if required["messages"] > 0 else proof
                if (len(evidence) != 1 or evidence[0].get("done") is not True
                        or not any(type(n) is int and n in valid for n in _items(evidence[0].get("lines")))):
                    unproven.append(required)
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Kyo: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not unproven:
                return passed()
            if not call or name not in self.schemas or name in ending:
                if not call and missing(unproven):
                    return passed("no message or step left for the missing step")
                break
            if self.left() < 2 or (name == self.roles.get("ask") and self.msg_used >= self.msg_limit):
                return passed("no message or step left for the gate's call")
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name], "args", faults)
            if faults or name + _dump(args) in made:
                # a malformed call, or the same call again: no step is spent on it
                break
            made.add(name + _dump(args))
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
            # act() entered the call in the process record
            self.absorb(seen, name in (self.roles.get("cart"), self.roles.get("cart_view")))
            if name == self.roles.get("cart"):
                self.note_cart(name, args)
                self.cart_said = _dump(seen)
            if name == self.roles.get("cart_view"):
                self.cart_view = seen
                self.cart_seen_at = self.step
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        reason = "Required steps are not yet evidenced: " + "; ".join(p["step"] for p in (unproven or steps))
        failed[(stage, self.cart_need_key() if stage == "choose" else self.cart_mark() + f"#{self.version}")] = (self.step, reason)
        return False, reason

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
        observation = self.adapt_observation(observation)
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
        return [line for line in lines if line not in self.heard and _text_record(line, self.id_key)[1]["status"] == "not_record"]

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

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        observation = self.adapt_observation(observation)
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
    def evidence_state(self, ident: str) -> str:
        return _dump([self.cart_need_key(), self.full.get(ident), self.code_price(ident), sorted(self.ref_values.items())])

    def evidence_value(self, record: Any, path: Any) -> tuple[bool, Any]:
        """The value a proof's path points at in a listing record. Lenient on form (digit strings index lists, a lone key
        may sit one level down when exactly one nested part holds it), never on content."""
        if isinstance(path, str) and path.strip():
            path = [path]
        if not isinstance(path, list) or not 1 <= len(path) <= 8:
            return False, None
        node = record
        for key in path:
            if isinstance(node, dict) and isinstance(key, str) and key in node:
                node = node[key]
            elif isinstance(node, list) and _count(key) is not None and not isinstance(key, bool) and 0 <= _count(key) < len(node):
                node = node[_count(key)]
            else:
                node = None
                break
        else:
            return True, node
        if len(path) == 1 and isinstance(path[0], str) and isinstance(record, dict):
            inner = [part[path[0]] for part in record.values() if isinstance(part, dict) and path[0] in part]
            if len(inner) == 1:
                return True, inner[0]
        return False, None

    def exact_value(self, actual: Any, expected: Any) -> bool:
        if isinstance(actual, str) and isinstance(expected, str):
            return ' '.join(actual.split()).casefold() == ' '.join(expected.split()).casefold()
        return type(actual) is type(expected) and actual == expected

    def same_need(self, said: Any, need: str) -> bool:
        """A proof's need names this need: the same words (case, spacing and end punctuation aside), also when one side
        carries a bracketed remark the other lacks, such as the ledger's note of the value a reference page gives."""
        if not isinstance(said, str):
            return False

        def norm(text: str) -> str:
            return ' '.join(text.casefold().split()).strip(' .;,')

        a, b = norm(said), norm(need)
        if not a or not b:
            return False
        if a == b:
            return True
        short, long = sorted((a, b), key=len)
        rest = long[len(short):].strip()
        return long.startswith(short) and rest.startswith('(') and rest.endswith(')') and rest.count('(') == 1

    def value_words(self, value: Any) -> list[str]:
        # letters and numbers apart, so "3440x1440", "12v" and "22inch" read like "3440 x 1440", "12 v", "22 inch"
        return re.findall(r"[^\W\d_]+|\d+(?:\.\d+)?", str(value).casefold())

    def same_word(self, a: str, b: str) -> bool:
        """Equal words, singular or plural (s, es, y/ies), or equal numbers. Generic spelling only, no word lists."""
        if a == b:
            return True
        if a[:1].isdigit() or b[:1].isdigit():
            try:
                return float(a) == float(b)
            except ValueError:
                return False
        short, long = sorted((a, b), key=len)
        return len(short) >= 2 and (long in (short + 's', short + 'es') or (short.endswith('y') and long == short[:-1] + 'ies'))

    def same_unit(self, a: Any, b: Any) -> bool:
        """Units equal after trivial normalisation: case, spaces, dots, dashes, outer brackets, a trailing plural."""
        def norm(unit: Any) -> str:
            return re.sub(r"[\s._\-]+", "", str(unit).casefold()).strip("()[]{},;:")

        a, b = norm(a), norm(b)
        if a == b:
            return True
        short, long = sorted((a, b), key=len)
        return len(short) >= 2 and short.isalpha() and long in (short + 's', short + 'es')

    # limit words in a need's own text, read only to place a bound on its side; a strict word is read inclusively,
    # so an observed value on the edge is never called a contradiction ("limits include the edge")
    _AT_LEAST = r"at least|no less than|not less than|no fewer than|not fewer than|minimum(?: of)?|min\.?|more than|greater than|over|above|>=|≥|>"
    _AT_MOST = r"at most|no more than|not more than|up to|maximum(?: of)?|max\.?|less than|fewer than|under|below|<=|≤|<"
    _OR_MORE = r"or more|or above|or over|or greater|or higher|minimum|min\.?"
    _OR_LESS = r"or less|or fewer|or below|or under|or lower|maximum|max\.?"
    _NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    # words that turn a stated value into an exclusion or a relation: such a need is never an exact categorical value
    _NEGATED = r"\b(?:not|no|non|never|without|except|excluding|unlike|other than|anything but|nor)\b|n't"

    def need_limit(self, need: str, proof: dict[str, Any]) -> tuple[float | None, float | None, str, bool] | None:
        """The numeric limit a need states: (low, high, unit, from_words). From the need's own words when they are just a
        limit word and one number (two for a range) with a short unit; a resolved reference need may give the number
        from the page it points at ("at least the height shown on ..."). Otherwise from the proof's numeric operator, its
        bounds and unit grounded in the need's words. None for fractions, thousands, dimensions or anything else."""
        text = ' '.join((need.partition(':')[2].strip() if ':' in need else need).casefold().split())
        ref = ' '.join(str(self.ref_values.get(need, '')).casefold().split())
        both = (text + ' ' + ref).strip()
        if not both or re.search(r"\d\s*[,/]\s*\d|\d\s*[x×*]\s*\d", both):
            return None
        number = self._NUMBER

        def short(unit: str) -> str | None:
            unit = unit.strip()
            return unit if len(unit.split()) <= 2 and not re.search(r"\d", unit) else None

        form = re.fullmatch(rf"(?P<cmp>{self._AT_LEAST}|{self._AT_MOST})\s*(?P<n>{number})\s*(?P<u>[^\d]*)", text)
        if form and short(form['u']) is not None:
            low = re.fullmatch(self._AT_LEAST, form['cmp']) is not None
            value = float(form['n'])
            return (value, None, short(form['u']), True) if low else (None, value, short(form['u']), True)
        form = re.fullmatch(rf"(?P<n>{number})\s*(?P<u>[^\d]*?)\s*(?P<cmp>{self._OR_MORE}|{self._OR_LESS})", text)
        if form and short(form['u']) is not None:
            value = float(form['n'])
            if re.fullmatch(self._OR_MORE, form['cmp']):
                return value, None, short(form['u']), True
            return None, value, short(form['u']), True
        form = (re.fullmatch(rf"between\s+(?P<a>{number})\s*(?P<ua>[^\d]*?)\s*and\s*(?P<b>{number})\s*(?P<ub>[^\d]*)", text)
                or re.fullmatch(rf"(?:from\s+)?(?P<a>{number})\s*(?P<ua>[^\d]*?)\s*(?:-|–|—|to)\s*(?P<b>{number})\s*(?P<ub>[^\d]*)", text))
        if form and short(form['ub']) is not None and short(form['ua']) is not None:
            unit = short(form['ub']) or short(form['ua'])
            if (not form['ua'].strip() or self.same_unit(form['ua'], unit)) and float(form['a']) <= float(form['b']):
                return float(form['a']), float(form['b']), unit, True
        if ref and not re.search(r"\d", text):
            head = re.match(rf"(?P<cmp>{self._AT_LEAST}|{self._AT_MOST})\s+(?:as|the)\s", text + ' ')
            value = self.numeric(ref)
            if head and value and short(value[1]) is not None:
                if re.fullmatch(self._AT_LEAST, head['cmp']):
                    return value[0], None, value[1], True
                return None, value[0], value[1], True
        op, expected = proof.get('op'), proof.get('value')
        if op in ('ge', 'gt', 'le', 'lt'):
            bounds = [expected]
        elif op == 'range' and isinstance(expected, list) and len(expected) == 2:
            bounds = list(expected)
        else:
            return None
        values = []
        for bound in bounds:
            parsed = self.numeric(bound)
            if parsed is None:
                return None
            values.append(parsed[0])
        found = list(re.finditer(rf"(?<![\w.]){number}", both))
        if len(found) != len(values) or any(float(m.group()) != v for m, v in zip(found, values)):
            return None
        unit = short(both[found[-1].end():])
        if unit is None:
            return None
        if op in ('ge', 'gt'):
            return values[0], None, unit, False
        if op in ('le', 'lt'):
            return None, values[0], unit, False
        return (values[0], values[1], unit, False) if values[0] <= values[1] else None

    def numeric_fit(self, actual: Any, low: float | None, high: float | None, unit: str, key: str) -> str | None:
        """Where an observed value lies against a limit in the need's own unit: "in" (wholly inside), "out" (wholly
        outside), "partial", or None when it is not a number in that same unit. A list is never "out"."""
        if isinstance(actual, list):
            return 'in' if any(self.numeric_fit(v, low, high, unit, key) == 'in' for v in actual if not isinstance(v, list)) else None
        if isinstance(actual, bool):
            return None
        if type(actual) in (int, float):
            scalar = self.numeric(actual)
            if scalar is None:
                return None
            # a bare number takes the unit its key states ("width_cm", "Height (inches)")
            named = unit and any(self.same_unit(word, unit) for word in _words(key).split())
            seen: tuple[float, float, str] | None = (scalar[0], scalar[0], unit if named else "")
        else:
            seen = self.numeric_interval(actual)
        if seen is None or not self.same_unit(seen[2], unit):
            return None
        lo, hi = seen[0], seen[1]
        if (low is None or lo >= low) and (high is None or hi <= high):
            return 'in'
        if (low is not None and hi < low) or (high is not None and lo > high):
            return 'out'
        return 'partial'

    def stated_values(self, need: str, proof: dict[str, Any]) -> list[list[str]] | None:
        """The words of the value(s) an exact categorical proof names (eq, or one_of alternatives), each grounded in the
        need's own words; None when the proof is not exact, a value is not the need's, or the need is an exclusion."""
        op, expected = proof.get('op'), proof.get('value')
        if op not in ('eq', 'one_of') or (op == 'eq' and isinstance(expected, list)):
            return None
        options = expected if isinstance(expected, list) else [expected]
        text = (need.partition(':')[2].strip() if ':' in need else need) + ' ' + str(self.ref_values.get(need, ''))
        if not options or re.search(self._NEGATED, text.casefold()):
            return None
        grounds = self.value_words(text)
        out = []
        for option in options:
            if type(option) not in (str, int, float) or not str(option).strip() or re.search(self._NEGATED, str(option).casefold()):
                return None
            words = self.value_words(option)
            if not words or not all(any(self.same_word(w, g) for g in grounds) for w in words):
                return None
            out.append(words)
        return out

    def has_words(self, actual: Any, options: list[list[str]]) -> bool | None:
        """Whether an observed value contains every word of one stated value (singular or plural); a list holding it
        counts, a list without it is uncertain, as is anything that is not text or a number."""
        if isinstance(actual, list):
            parts = [v for v in actual if type(v) in (str, int, float)]
            return True if any(self.has_words(v, options) for v in parts) else None
        if type(actual) not in (str, int, float):
            return None
        seen = self.value_words(actual)
        if not seen:
            return None
        return any(all(any(self.same_word(w, s) for s in seen) for w in words) for words in options)

    def compare_evidence(self, actual: Any, proof: dict[str, Any], need: str) -> bool | None:
        """The code's reading of one advisory proof: the observed value at the proof's path against the need in force.
        False only for a demonstrated contradiction: an observed number in the need's own unit (after case, punctuation
        and plural normalisation) outside a limit the need's words state, whatever operator the proof names, semantic
        included, when the proof's key names the need's property or the proof itself reports the need unmet (with a
        limit only the proof's operator gives, or a range partly outside, the proof must report it unmet); or an exact
        stated value whose words the observed value lacks while the proof also reports it unmet. True when the code
        sees the need met, or the proof says met and nothing contradicts it. None when uncertain: other or unparsable
        units, list values, malformed or ungrounded proofs, or a code reading and a proof that disagree."""
        proof = proof if isinstance(proof, dict) else {}
        said = proof.get('ok') if isinstance(proof.get('ok'), bool) else None
        path = proof.get('path') if isinstance(proof.get('path'), list) else [proof.get('path')]
        key = next((k for k in reversed(path) if isinstance(k, str)), '')
        prop = need.partition(':')[0].strip() if ':' in need else ''
        named = bool(key and prop) and _names(_stem(prop), _stem(key))
        limit = self.need_limit(need, proof)
        if limit is not None:
            low, high, unit, from_words = limit
            where = self.numeric_fit(actual, low, high, unit, key)
            if where == 'out' and (said is False or (from_words and named)):
                return False
            if where == 'partial' and said is False:
                return False
            if where == 'in':
                return True if said is not False else None
            return True if said is True and where is None else None
        options = self.stated_values(need, proof)
        if options:
            has = self.has_words(actual, options)
            if has is False and said is False:
                return False
            if has is True:
                return True if said is not False else None
        return True if said is True else None

    def remember_evidence(self, got: dict[str, Any], ids: list[str]) -> bool:
        """Keep per listing and item what a usable review shows. Proofs are advisory: a listing is verified when the
        review approves it and nothing the code reads contradicts it; a missing, malformed or ungrounded proof, an
        unresolved reference or a split basket is uncertain, never a mismatch. A demonstrated contradiction is kept
        while its need is in force and its cited observation unchanged; a later grounded proof of the same need replaces
        it (no first-interpretation lock). An unusable reply changes nothing. Returns whether this review approved
        every listing in `ids` with no contradiction and no explicit negative."""
        if not isinstance(got, dict) or not isinstance(got.get('ok'), bool):
            return False
        approved = got['ok'] is True
        entries = [e for e in got.get('evidence') if isinstance(e, dict)] if isinstance(got.get('evidence'), list) else []
        live = self.needs_now()
        covered: set[tuple[int, str]] = set()
        clean = approved
        for entry in entries:
            number, ident = _count(entry.get('item')), entry.get('id')
            ident = ident.strip() if isinstance(ident, str) else None
            if number is None or not 1 <= number <= len(self.items) or ident not in ids or (number, ident) in covered:
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            if not record:
                continue
            covered.add((number, ident))
            old = self.listing_evidence.get((number, ident)) or {}
            against = dict(old.get('against') or {})
            negative, unsure = [], []
            identity = entry.get('identity')
            if isinstance(identity, dict) and identity.get('matches') is False:
                negative.append('review judged the product family different')
            elif not isinstance(identity, dict) or not self.evidence_value(record, identity.get('path'))[0]:
                unsure.append('identity proof missing or ungrounded')
            proofs = [p for p in entry.get('requirements') or [] if isinstance(p, dict)] if isinstance(entry.get('requirements'), list) else []
            for need in dict.fromkeys(need for n, need in live if n == number):
                results = []
                for proof in proofs:
                    if not self.same_need(proof.get('need'), need):
                        continue
                    exists, actual = self.evidence_value(record, proof.get('path'))
                    if not exists:
                        continue
                    results.append((self.compare_evidence(actual, proof, need), proof, actual))
                if not results:
                    unsure.append('no grounded proof: ' + need)
                    continue
                if all(result is False for result, _, _ in results):
                    _, proof, actual = results[0]
                    against[need] = {'path': proof.get('path'), 'actual': _dump(actual), 'ref': self.ref_values.get(need, '')}
                    continue
                against.pop(need, None)
                if any(result is None for result, _, _ in results) and not any(result is True for result, _, _ in results):
                    unsure.append('uncertain: ' + need)
                if any(p.get('ok') is False for _, p, _ in results):
                    negative.append('review reports unmet: ' + need)
            if any(n == number and need not in self.ref_values for n, need in self.refs if (n, need) in live):
                unsure.append('referenced value not resolved')
            standing = [need for need in against if self.contradiction_stands(number, ident, need, against[need])]
            ok = approved and not negative and not standing
            clean = clean and ok
            self.listing_evidence[(number, ident)] = {'state': self.evidence_state(ident), 'ok': ok, 'errors': negative,
                                                      'unsure': unsure, 'against': against}
            self.trace.append({'kind': 'verified_listing', 'item': number, 'id': ident, 'ok': ok, 'errors': negative,
                               'unsure': unsure, 'against': standing})
        # The cart assignment: the reply's cart_items unless they contradict its own evidence entries, then the
        # entries themselves, which settle it whenever they name one listing per item.
        pairs = {(_count(e.get('item')), str(e.get('id')).strip()) for e in entries}
        listed = got.get('cart_items') if isinstance(got.get('cart_items'), list) else []
        if all(pair in {(_count(c.get('item')), str(c.get('id')).strip()) for c in listed if isinstance(c, dict)} for pair in pairs):
            self.remember_cart_assignment(got, ids)
        self.remember_cart_assignment({**got, 'cart_items': [{'item': e.get('item'), 'id': e.get('id')} for e in entries]}, ids)
        # a listing this reply gives no entry for: an approval stands for it (one item, or a known current assignment);
        # a disapproval withdraws any earlier approval of it, keeping its recorded contradictions
        mapping = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        if len(self.items) == 1 and len(ids) == 1:
            mapping = {1: ids[0]}
        for ident in ids:
            numbers = [n for n, i in covered if i == ident] or [n for n, i in mapping.items() if i == ident]
            for number in numbers:
                if (number, ident) in covered:
                    continue
                if not (self.full.get(ident) or self.rows.get(ident)):
                    continue
                old = self.listing_evidence.get((number, ident)) or {}
                against = dict(old.get('against') or {})
                standing = [need for need in against if self.contradiction_stands(number, ident, need, against[need])]
                ok = approved and not standing
                clean = clean and ok
                self.listing_evidence[(number, ident)] = {'state': self.evidence_state(ident), 'ok': ok, 'errors': [],
                                                          'unsure': ['no proof entry'], 'against': against}
            for key, saved in self.listing_evidence.items():
                if key[1] == ident and key not in covered and not approved:
                    saved['ok'] = False
        return clean and not self.contradicted(ids)

    def contradiction_stands(self, number: int, ident: str, need: str, seen: dict[str, Any]) -> bool:
        """A recorded contradiction still holds: its need is in force for that item, the page value it rests on is
        unchanged, and the listing's record still shows the same value at the same place."""
        if (number, need) not in self.needs_now() or str(self.ref_values.get(need, '')) != seen.get('ref', ''):
            return False
        exists, actual = self.evidence_value(self.full.get(ident) or self.rows.get(ident), seen.get('path'))
        return exists and _dump(actual) == seen.get('actual')

    def contradicted(self, ids: list[str]) -> list[str]:
        """The demonstrated contradictions that stand for these listings, under the item each is assigned to (any item
        while no current assignment names it). Only these block a purchase."""
        mapping = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        found = []
        for (number, ident), saved in self.listing_evidence.items():
            if ident not in ids or (ident in mapping.values() and mapping.get(number) != ident):
                continue
            for need, seen in (saved.get('against') or {}).items():
                if self.contradiction_stands(number, ident, need, seen):
                    found.append(f"{ident} (item {number}) against {need}")
        return found

    def verified_listing(self, number: int, ident: str) -> bool:
        """A current listing a usable review approved for this item, with nothing contradicting it (proofs advisory)."""
        saved = self.listing_evidence.get((number, ident), {})
        return (saved.get('ok') is True and saved.get('state') == self.evidence_state(ident)
                and bool(self.full.get(ident) or self.rows.get(ident)) and self.buyable(ident)
                and not self.contradicted([ident]))

    def selection_verified(self, ids: list[str]) -> bool:
        mapping = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        if len(self.items) == 1 and len(ids) == 1:
            mapping = {1: ids[0]}
        return (set(mapping) == set(range(1, len(self.items) + 1)) and len(set(mapping.values())) == len(self.items)
                and set(mapping.values()) == set(ids)
                and all(self.verified_listing(n, ident) for n, ident in mapping.items()))

    def checkout_evidence_ready(self) -> bool:
        """Purchase gate, shared by every checkout path. It blocks only a demonstrated contradiction that still stands
        for a cart line. A cart no review has approved gets one review here (C2r4: review once, then proceed); when that
        review cannot run (spend cap, unaffordable, timeout, unusable after its one retry, code-only completion) the
        evidence is uncertain and the purchase goes ahead."""
        ids = self.cart_ids()
        if ids and not getattr(self, 'code_finishing', False) and not self.selection_verified(ids) and not self.contradicted(ids):
            self.check_cart(ids)
        against = self.contradicted(ids)
        if against:
            # SPEC-006 r14c: our own verification redirects but never ends a task. A contradiction holds this cart once,
            # with the reason; the same cart presented again proceeds (C2r4 hold-once), since a decline scores 0.
            key = (tuple(sorted(ids)), tuple(sorted(against)))
            if key in self.contradiction_holds:
                self.trace.append({'kind': 'checkout_evidence', 'ok': True, 'ids': ids, 'against': against, 'held_before': True})
                return True
            self.contradiction_holds.add(key)
            self.cart_guard_error = ('Checkout held: the opened records contradict a requirement in force: ' + '; '.join(against[:3])
                                     + '. Replace that line with a listing that meets it (or re-read the record if it changed);'
                                     + ' if no other listing fits better, check out this cart again rather than ending without a purchase.')
            self.trace.append({'kind': 'checkout_evidence', 'ok': False, 'ids': ids, 'against': against})
            return False
        if not self.literal_redirect(ids):
            return False
        return self.compat_redirect(ids)

    # ---------- SPEC-007 W3: literal evidence (ranking and one redirect, never a block) ----------
    def literal_fields(self, record: Any) -> dict[str, list[str]]:
        """The record's attribute values under one name per property (_key_name): its own scalar fields and those of
        the attribute blocks one level down; several spellings of a key merge, a list counts element by element."""
        fields: dict[str, list[str]] = {}

        def put(key: Any, value: Any) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, list):
                values = [str(v) for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            elif isinstance(value, (str, int, float)):
                values = [str(value)]
            else:
                return
            name = _key_name(key)
            if name and values:
                fields.setdefault(name, []).extend(values)

        if isinstance(record, dict):
            for key, value in record.items():
                if isinstance(value, dict):
                    for inner, item in value.items():
                        put(inner, item)
                else:
                    put(key, value)
        return fields

    def literal_needs(self, number: int) -> list[str]:
        """Categorical needs in force on item `number` that name a property: not numeric limits (need_limit reads
        them from their own words), not values taken from another listing."""
        pointed = {need for _, need in self.refs}
        out = []
        for n, need in self.needs_now():
            prop, colon, value = need.partition(":")
            if n != number or not colon or not _key_name(prop) or not self.value_words(value) or need in pointed:
                continue
            if self.need_limit(need, {}) is not None:
                continue
            out.append(need)
        return out

    def literal_states(self, need: str, fields: dict[str, list[str]]) -> bool:
        """The record states this need under the need's own property name: every word of the value (one alternative of
        a closed list is enough) in one value there; for an exclusion, the property is stated without the excluded
        words. A record without that property states nothing (unknown, never a contradiction)."""
        prop, _, value = need.partition(":")
        values = fields.get(_key_name(prop))
        if not values:
            return False
        held = [self.value_words(v) for v in values]

        def inside(words: list[str], have: list[str]) -> bool:
            return bool(words) and all(any(self.same_word(w, h) for h in have) for w in words)

        text = ' '.join(value.casefold().split())
        if re.search(self._NEGATED, text):
            banned = [w for w in self.value_words(re.sub(self._NEGATED, ' ', text)) if w not in ('a', 'an', 'the', 'or', 'and', 'any')]
            return bool(banned) and not any(inside(banned, have) for have in held)
        options = [self.value_words(part) for part in re.split(r"\s*(?:,|/|\bor\b)\s*", text) if part.strip()]
        return any(inside(words, have) for words in options for have in held)

    def literal_count(self, number: int, ident: str, record: Any = None) -> tuple[int, list[str]]:
        """How many categorical needs on item `number` the listing's own attributes state, and which."""
        record = record if record is not None else (self.full.get(ident) or self.rows.get(ident))
        fields = self.literal_fields(record)
        stated = [need for need in self.literal_needs(number) if self.literal_states(need, fields)]
        return len(stated), stated

    def aisle_of(self, ident: str) -> str | None:
        """The store's aisle/category label on the listing's record (a scalar field whose key names one), else None."""
        for record in (self.full.get(ident) or {}, self.rows.get(ident) or {}):
            for key, value in record.items():
                if isinstance(value, str) and value.strip() and set(_words(key).split()) & {"aisle", "category", "department", "section"}:
                    return ' '.join(value.casefold().split())
        return None

    def literal_redirect(self, ids: list[str]) -> bool:
        """Before an order, once per task: when an opened listing in the same aisle states more of a cart line's
        requirements in its own attributes than the carted listing (or, for a cheapest request, states all of them
        like the carted one and costs less), hold the checkout with that listing named. The same cart presented again
        proceeds. No model call; skipped when steps are short, the code finishes alone or no model can act on it."""
        # a hold reaches a model only when the loop model itself is checking out; a held forced finish declined
        if (self.literal_held or not ids or getattr(self, 'code_finishing', False) or self.left() <= 5
                or self.no_credit or self.spent_out or self.broke or not self.loop_checkout()):
            return True
        mapping = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        if len(self.items) == 1 and len(ids) == 1:
            mapping = {1: ids[0]}
        cheapest = self.brief.get("pick") == "cheapest"
        for number, ident in sorted(mapping.items()):
            if ident not in ids:
                continue
            aisle = self.aisle_of(ident)
            needs = self.literal_needs(number)
            if not aisle or not needs:
                continue
            have, mine = self.literal_count(number, ident)
            price = self.code_price(ident)
            qty = self.items[number - 1].qty
            plan_rest = {m: self.items[k - 1].qty for k, m in mapping.items() if m != ident}
            best = None
            for other in self.full:
                if other in ids or other == ident or not self.buyable(other) or self.contradicted([other]):
                    continue
                if self.aisle_of(other) != aisle:
                    continue
                saved = self.listing_evidence.get((number, other), {})
                if saved.get('ok') is False and saved.get('state') == self.evidence_state(other):
                    continue  # a current review rejected it: not offered again
                plan = dict(plan_rest)
                plan[other] = qty
                if not self.code_budget_ok(plan):
                    continue
                count, stated = self.literal_count(number, other)
                cost = self.code_price(other)
                if count > have:
                    rule = "more"
                elif cheapest and count == have == len(needs) and cost is not None and price is not None and cost < price:
                    rule = "cheaper"
                else:
                    continue
                rank = (-count, cost if cost is not None else float("inf"))
                if best is None or rank < best[0]:
                    best = (rank, other, rule, stated)
            if not best:
                continue
            _, other, rule, stated = best
            self.literal_held = True
            if rule == "more":
                extra = [need for need in stated if need not in mine]
                reason = (f"listing {other} states {'; '.join(extra[:3])} in its own attributes, under the requirement's own"
                          f" name, where the carted {ident} does not")
            else:
                reason = (f"listing {other} ({self.code_price(other):g}) states every requirement in its own attributes, as the"
                          f" carted {ident} ({price:g}) does, and the request asks for the cheapest")
            self.cart_guard_error = (f"Checkout held once: {reason}. Open {other} if needed, put it in the cart instead of {ident},"
                                     f" and check out; if {ident} is still the right listing, check out this cart again.")
            self.trace.append({'kind': 'literal_redirect', 'item': number, 'line': ident, 'to': other, 'rule': rule,
                               'counts': [have, len(stated), len(needs)]})
            return False
        return True

    # ---------- SPEC-009 F2: compatibility matching (a ranking key and one checkout hold, never a block; no model call) ----------
    # Forms the store worlds write: a device page "Model no. LUR6230. Also sold as Lutor 623." ("Model: VIT-8030.", "Model
    # number: PET2330."); an accessory page "Works with LUR1510, Lutor 623." / "Compatible with: VIR3110." / "Compatible
    # with: every VIT-45xx model, except VIT-4530."; "Part RO17407 also fits: VIR5010." (about the listing whose own code is
    # RO17407, written on the page that listing points to); "More on part LU46877." / "See part RO28520 for the rest." (the
    # rest of this listing's list is on that part's page). A part's own code is in its title: "[LU92396]", "(part VO9996)".
    _COMPAT_MODEL = r"[A-Z]{2,5}[- ]?\d{3,6}"
    _COMPAT_ALIAS = r"[A-Z][a-z]{1,11} \d{2,5}"
    _COMPAT_ONE = r"(?:" + _COMPAT_MODEL + r"|" + _COMPAT_ALIAS + r")"
    _COMPAT_PART = r"[A-Z]{1,5}\d{3,7}"
    _COMPAT_WORDS = r"\b(?:fit|fits|fitting|fitted|compatib\w*|works? with|goes with|go with|pairs? with)\b"
    # the label after "Model" varies by world ("no.", "number", "ref." in race 14, "code", "id", "#")
    _COMPAT_DEVICE = (r"\b(?i:model)(?:\s*(?i:no\.?|nr\.?|number|num\.?|ref\.?|reference|code|id|type|#))?\s*[:.#]?\s*("
                      + _COMPAT_MODEL + r")\b")
    _COMPAT_SOLD = r"\b(?i:also sold as|also known as|also sold under)\s+([^.;]+)"
    # the verb varies by world ("Works with", "Compatible with", race 14 "Goes with"); a list still counts only when every
    # element is a model code or an also-sold-as name, so prose ("goes with any outfit") reads as nothing
    _COMPAT_LIST = (r"(?<![A-Za-z])(?i:(?:works?|goes|go|pairs?|is compatible|compatible|for use|designed|made|built|meant|suited)"
                    r"\s+(?:with|for)|fits|suits|matches)\s*:?\s*([^.;]+)")
    _COMPAT_REVERSE = (r"\b(?i:part)\s+(" + _COMPAT_PART + r")\s+(?i:also fits|also works with|also goes with|also pairs with|also suits"
                       r"|is also compatible with)\s*:?\s*([^.;]+)")
    _COMPAT_POINTER = r"\b(?i:more on|see|details on)\s+(?i:part)\s+(" + _COMPAT_PART + r")\b"
    _COMPAT_TITLE_CODE = (r"\[\s*(?:(?i:part)\s+)?(" + _COMPAT_PART + r")\s*\]|\(\s*(?i:part)\s+(" + _COMPAT_PART + r")\s*\)"
                          r"|[-–]\s*(" + _COMPAT_PART + r")\s*$")
    _COMPAT_TITLE = frozenset({"title", "name", "heading", "headline", "caption"})
    _COMPAT_STOP = frozenset({"and", "for", "the", "with", "set", "pack", "item", "product", "kit", "pair"})

    def compat_code(self, code: Any) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()

    def compat_list(self, text: str) -> tuple[set[str], list[tuple[str, int, frozenset[str]]], list[str]] | None:
        """A world-written list of what a part fits: (codes, series rules, the elements as written). Every element must be
        a model code or an also-sold-as name ("LUR1510, Lutor 623"), or the whole list one series ("every VIT-45xx model,
        except VIT-4530"); anything else (scraped prose such as "compatible with most cameras") gives None."""
        text = text.strip()
        series = re.fullmatch(r"(?i:every|all|any)\s+(?:(?i:the)\s+)?([A-Z]{2,5}[- ]?\d{0,5})([xX]{1,4})(?:\s+(?i:models?|series|units?))?"
                              r"(?:\s*,?\s*(?i:except|but not|excluding|other than)\s+(.+))?", text)
        if series:
            prefix = self.compat_code(series.group(1))
            excepts = self.compat_list(series.group(3)) if series.group(3) else (set(), [], [])
            if len(prefix) < 3 or excepts is None or excepts[1]:
                return None
            return set(), [(prefix, len(series.group(2)), frozenset(excepts[0]))], [text]
        parts = [re.sub(r"\s+(?i:models?|series|units?)$", "", p.strip())
                 for p in re.split(r"\s*(?:,|;|/|&|\band\b|\bor\b)\s*", text) if p and p.strip()]
        if not parts or not all(re.fullmatch(self._COMPAT_ONE, p) for p in parts):
            return None
        return {self.compat_code(p) for p in parts}, [], parts

    def compat_record(self, record: Any) -> dict[str, Any]:
        """What one listing's record says about fitting: its identifiers as a device (model numbers, names it is also sold
        as), its own part code, the codes and series its lists name, what it says other parts also fit, and the parts it
        points to for the rest of its list. Each string field is read on its own; attribute blocks as "key: value"."""
        rec: dict[str, Any] = {"models": set(), "aliases": set(), "self_code": "", "listed": set(), "series": [], "shown": [],
                               "others_fit": {}, "others_series": {}, "pointers": [], "said": False, "written": {}}
        if not isinstance(record, dict):
            return rec
        texts: list[str] = []
        titles: list[str] = []
        for key, value in record.items():
            if isinstance(value, str):
                texts.append(value)
                if set(_words(key).split()) & self._COMPAT_TITLE:
                    titles.append(value)
            elif isinstance(value, dict):
                for inner, item in value.items():
                    if not isinstance(item, (str, int, float)) or isinstance(item, bool):
                        continue
                    texts.append(f"{_words(inner)}: {item}")
                    if set(_words(inner).split()) & {"compatible", "compatibility", "fit", "fits"}:
                        got = self.compat_list(str(item))
                        if got:
                            rec["listed"] |= got[0]
                            rec["series"] += got[1]
                            rec["shown"] += got[2]
            elif isinstance(value, list):
                texts += [v for v in value if isinstance(v, str)]
        for text in texts:
            for hit in re.finditer(self._COMPAT_DEVICE, text):
                rec["models"].add(self.compat_code(hit.group(1)))
                rec["written"][self.compat_code(hit.group(1))] = hit.group(1)
            for hit in re.finditer(self._COMPAT_SOLD, text):
                got = self.compat_list(hit.group(1))
                if got and not got[1]:
                    rec["aliases"] |= got[0]
                    rec["written"].update((self.compat_code(p), p) for p in got[2])
            for hit in re.finditer(self._COMPAT_REVERSE, text):
                got = self.compat_list(hit.group(2))
                if got:
                    part = hit.group(1).upper()
                    rec["others_fit"].setdefault(part, set()).update(got[0])
                    rec["others_series"].setdefault(part, []).extend(got[1])
            for hit in re.finditer(self._COMPAT_LIST, text):
                before = text[max(0, hit.start() - 14):hit.start()].casefold()
                if re.search(r"(?:\bnot|\bnever|n't|\bno longer|\balso)\s*$", before):
                    continue  # a negation, or "Part P also fits" (about another listing)
                got = self.compat_list(hit.group(1))
                if got:
                    rec["listed"] |= got[0]
                    rec["series"] += got[1]
                    rec["shown"] += got[2]
            for hit in re.finditer(self._COMPAT_POINTER, text):
                if hit.group(1).upper() not in rec["pointers"]:
                    rec["pointers"].append(hit.group(1).upper())
        for title in titles:
            hit = re.search(self._COMPAT_TITLE_CODE, title)
            if hit:
                rec["self_code"] = next(g for g in hit.groups() if g).upper()
                break
        rec["said"] = bool(rec["models"] or rec["listed"] or rec["series"] or rec["others_fit"] or rec["pointers"])
        return rec

    def compat_index(self) -> dict[str, Any]:
        """Every seen listing's reading, plus what pages say other parts also fit (by part code), which codes name the same
        device (a model number and the names it is also sold as) and which listings carry each part code."""
        records: dict[str, dict[str, Any]] = {}
        also: dict[str, set[str]] = {}
        also_series: dict[str, list[tuple[str, int, frozenset[str]]]] = {}
        linked: dict[str, set[str]] = {}
        owners: dict[str, list[str]] = {}
        written: dict[str, str] = {}
        for ident in self.seen_order:
            got = self.compat_record(self.rows.get(ident))
            records[ident] = got
            written.update(got["written"])
            for part, codes in got["others_fit"].items():
                also.setdefault(part, set()).update(codes)
            for part, rules in got["others_series"].items():
                also_series.setdefault(part, []).extend(rules)
            names = got["models"] | got["aliases"]
            if got["models"] and got["aliases"]:
                for name in names:
                    linked.setdefault(name, set()).update(names)
            if got["self_code"]:
                owners.setdefault(got["self_code"], []).append(ident)
        return {"records": records, "others_fit": also, "others_series": also_series, "linked": linked, "owners": owners,
                "written": written}

    def device_names(self, ident: str | None, index: dict[str, Any]) -> set[str]:
        """A device listing's identifiers: the model numbers and also-sold-as names its own record states (and the other
        names any page gives for those)."""
        got = index["records"].get(ident or "") or {}
        ids = set(got.get("models") or set()) | set(got.get("aliases") or set())
        for name in list(ids):
            ids |= index["linked"].get(name, set())
        return ids

    def need_names(self, named: frozenset[str], index: dict[str, Any]) -> set[str]:
        """Identifiers a requirement names: model codes as written, and also-sold-as names some page gives a device."""
        ids = {code for strict, code in named if strict or code in index["linked"]}
        for name in list(ids):
            ids |= index["linked"].get(name, set())
        return ids

    def pointer_read(self, part: str, index: dict[str, Any]) -> bool:
        """The page of the listing whose own code is `part` has been read with its fitting sentences."""
        return any(index["records"].get(i, {}).get("said") for i in index["owners"].get(part, []))

    def fits_device(self, ident: str, ids: set[str], index: dict[str, Any]) -> tuple[bool | None, list[str]]:
        """Whether listing `ident` fits a device with these identifiers. True: its own list, a series it names (less the
        exceptions) or another page's "Part <its code> also fits" names one. False: it carries a list that names none, and
        every page it points to for the rest has been read. None: it says nothing about fitting, or the rest of its list
        is on a page not read yet (those part codes come second)."""
        got = index["records"].get(ident)
        if not got or not ids:
            return None, []
        codes = set(got["listed"])
        rules = list(got["series"])
        if got["self_code"]:
            codes |= index["others_fit"].get(got["self_code"], set())
            rules += index["others_series"].get(got["self_code"], [])
        if ids & codes:
            return True, []
        for prefix, width, excepts in rules:
            if any(re.fullmatch(re.escape(prefix) + r"\d{%d}" % width, i) and i not in excepts for i in ids):
                return True, []
        if got["pointers"]:
            if not got["self_code"]:
                return None, []       # the pointed page names parts by code: without its own code it cannot be matched
            unread = [p for p in got["pointers"] if not self.pointer_read(p, index)]
            if unread:
                return None, unread
        return (False if codes or rules else None), []

    def compat_pairs(self, index: dict[str, Any] | None = None) -> list[tuple[int, int | None, frozenset, str]]:
        """Which requested item must fit which: (accessory item, device item or None, identifiers a requirement names as
        (strict, code) pairs, that requirement). From the request's reading ("fits": item n must fit item m) and from
        requirements in force that ask for a fit and name a model code or another requested item."""
        count = len(self.items)
        found: list[tuple[int, int | None, frozenset, str]] = []
        numbers: list[int] = []
        seen = 0
        for entry in _items(self.brief.get("items")):
            seen += 1 if isinstance(entry, dict) else 0
            numbers.append(seen if isinstance(entry, dict) else 0)
        for entry in _items(self.brief.get("fits")):
            if not isinstance(entry, dict):
                continue
            part, main = _count(entry.get("item")), _count(entry.get("with"))
            part = numbers[part - 1] if part and 1 <= part <= len(numbers) else 0
            main = numbers[main - 1] if main and 1 <= main <= len(numbers) else 0
            if part and main and part != main and part <= count and main <= count:
                found.append((part, main, frozenset(), ""))
        for number, need in self.needs_now():
            if not re.search(self._COMPAT_WORDS, need, re.I):
                continue
            value = need.partition(":")[2] if ":" in need else need
            named = {(True, self.compat_code(c)) for c in re.findall(r"\b" + self._COMPAT_MODEL + r"\b", value)}
            named |= {(False, self.compat_code(c)) for c in re.findall(r"\b" + self._COMPAT_ALIAS + r"\b", value)}
            if any(strict for strict, _ in named) or (named and (index or self.compat_index())["linked"].keys() & {c for _, c in named}):
                found.append((number, None, frozenset(named), need))
                continue
            words = _stem(value)
            best: list[tuple[int, int]] = []
            for other, item in enumerate(self.items, 1):
                if other == number:
                    continue
                name = {w for w in _stem(item.name + " " + item.shelf) if len(w) >= 3 and w not in self._COMPAT_STOP}
                best.append((len(name & words), other))
            best.sort(reverse=True)
            if best and best[0][0] > 0 and (len(best) == 1 or best[1][0] < best[0][0]):
                found.append((number, best[0][1], frozenset(), need))
        return list(dict.fromkeys(found))

    def item_line(self, number: int, pool: list[str]) -> str | None:
        """The listing in `pool` that stands for item `number`: its current cart assignment, else the one listing whose
        aisle (then title) shares the most words with the item's name and store label."""
        mapping = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        if mapping.get(number) in pool:
            return mapping[number]
        if not 1 <= number <= len(self.items):
            return None
        if len(self.items) == 1 and len(pool) == 1:
            return pool[0]
        item = self.items[number - 1]
        want = {w for w in _stem(item.name + " " + item.shelf) if len(w) >= 3 and w not in self._COMPAT_STOP}
        scored = []
        for ident in pool:
            if mapping and ident in mapping.values():
                continue
            record = self.rows.get(ident) or {}
            title = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and set(_words(k).split()) & self._COMPAT_TITLE)
            scored.append(((len(want & _stem(self.aisle_of(ident) or "")), len(want & _stem(title))), ident))
        scored.sort(reverse=True)
        if not scored or scored[0][0] == (0, 0) or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        return scored[0][1]

    def compat_targets(self, number: int, pool: list[str], index: dict[str, Any],
                    pairs: list[tuple[int, int | None, frozenset, str]]) -> list[tuple[int | None, str | None, set[str], str]]:
        """For item `number` when it must fit something: (device item, its listing in `pool`, the identifiers to match,
        the requirement that names them or "")."""
        out = []
        for part, main, named, need in pairs:
            if part != number:
                continue
            if main is None:
                ids = self.need_names(named, index)
                host = None
            else:
                host = self.item_line(main, pool)
                ids = self.device_names(host, index) if host else set()
            if ids:
                out.append((main, host, ids, need))
        return out

    def compat_rank(self, number: int, order: list[str], taken: set[str]) -> dict[str, int]:
        """best_candidate's key for item `number` when it must fit a device (the device line already planned or carted, or
        the code a requirement names): 0 when a listing's lists name the device, 1 when unknown, 2 when they name none.
        All 0 for an item that need not fit anything."""
        rank = {i: 0 for i in order}
        pairs = [p for p in self.compat_pairs() if p[0] == number] if order else []
        if not pairs:
            return rank
        index = self.compat_index()
        pool = [i for i in dict.fromkeys(list(taken) + self.cart_ids()) if i not in order]
        for _, _, ids, _ in self.compat_targets(number, pool, index, pairs):
            for ident in order:
                state = self.fits_device(ident, ids, index)[0]
                rank[ident] = max(rank[ident], 0 if state is True else 1 if state is None else 2)
        return rank

    def loop_checkout(self) -> bool:
        """The checkout under way is the loop model's own call (its message is the newest in the chat), so a hold reaches a
        model that can act on it. A hold of a forced finish or of the code's own completion would end without a purchase."""
        last = self.chat[-1] if self.chat else {}
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return False
        for call in last.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict):
                raw = fn.get("arguments")
                args = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                if self.is_buy(str(fn.get("name") or ""), args if isinstance(args, dict) else {}):
                    return True
        return False

    def compat_alternative(self, number: int, line: str, ids: set[str], cart: list[str], index: dict[str, Any]) -> str | None:
        """An opened listing in the carted line's aisle that fits these identifiers: buyable, not contradicted or rejected
        by a current review, inside the budget in place of the line, stating at least as many of the item's needs in its
        own attributes as the line does; most needs stated first, then (for a cheapest request) the lowest price, then
        the newest."""
        aisle = self.aisle_of(line)
        if not aisle:
            return None
        floor = self.literal_count(number, line)[0]
        qty = self.items[number - 1].qty
        plan = {i: (self.cart_lines.get(i) or 1) for i in cart if i != line}
        cheapest = self.brief.get("pick") == "cheapest"
        order = {ident: pos for pos, ident in enumerate(self.seen_order)}
        best = None
        for other in self.seen_order:
            if other not in self.full and other not in self.opened:
                continue
            if other in cart or not self.buyable(other) or self.aisle_of(other) != aisle or self.contradicted([other]):
                continue
            saved = self.listing_evidence.get((number, other), {})
            if saved.get("ok") is False and saved.get("state") == self.evidence_state(other):
                continue
            if self.fits_device(other, ids, index)[0] is not True:
                continue
            trial = dict(plan)
            trial[other] = qty
            if not self.code_budget_ok(trial):
                continue
            stated = self.literal_count(number, other)[0]
            if stated < floor:
                continue
            price = self.code_price(other)
            rank = (-stated, (price if price is not None else float("inf")) if cheapest else 0.0, -order.get(other, -1))
            if best is None or rank < best[0]:
                best = (rank, other)
        return best[1] if best else None

    def compat_redirect(self, ids: list[str]) -> bool:
        """SPEC-009 F2, before an order, once per task: an accessory line that must fit a device (the request's reading, or
        a requirement in force naming the device or its code) carries a list of what it fits that names none of the
        device's identifiers. When an opened listing of its aisle does name one, or when the rest of its list is on a part
        page not read yet, the checkout is held once with that listing or part named. The same cart presented again
        proceeds. Only the loop model's own checkout is held; no model call."""
        if (getattr(self, "compat_held", False) or not ids or getattr(self, "code_finishing", False) or self.left() <= 5
                or self.no_credit or self.spent_out or self.broke or not self.loop_checkout()):
            return True
        pairs = self.compat_pairs()
        if not pairs:
            return True
        index = self.compat_index()
        for number in sorted({p[0] for p in pairs}):
            line = self.item_line(number, ids)
            if not line:
                continue
            for main, host, want, need in self.compat_targets(number, [i for i in ids if i != line], index, pairs):
                state, unread = self.fits_device(line, want, index)
                if state is not False and not unread:
                    continue
                other = self.compat_alternative(number, line, want, ids, index)
                kind = self.items[number - 1].name
                listed = index["records"][line]["shown"][:4]
                shown = f"lists what it fits as {', '.join(listed)}," if listed else "names nothing it fits on its own page,"
                names = ", ".join(index["written"].get(code, code) for code in sorted(want)[:4])
                target = (f"the names of the {self.items[main - 1].name} in the cart, {host} ({names})" if main is not None
                          else f'{names}, which the requirement "{need}" names')
                if state is False and other:
                    text = (f"Checkout held once: {line} ({kind}) {shown} which names none of {target};"
                            f" {other} names one. Put {other} in the cart instead of {line} if it meets the {kind}'s other"
                            f" requirements, then check out; if the {kind} need not fit it, check out this cart again.")
                    rule, part = "alternative", None
                elif state is False:
                    # no opened listing names the device: the loop searches by its names (a part's list counts only
                    # when it names one of them), once
                    first = index["written"].get(sorted(want)[0], sorted(want)[0]) if want else ""
                    text = (f"Checkout held once: {line} ({kind}) {shown} which names none of {target}, and no {kind}"
                            f" opened so far names one. Search for a {kind} that names one of them (for example"
                            f" \"{first} {kind}\"), open the most promising results, and put one whose page names one in the"
                            f" cart instead of {line} if it meets the {kind}'s other requirements; if the {kind} need not fit"
                            f" it, check out this cart again.")
                    rule, part = "search", None
                else:
                    part = unread[0]
                    owners = index["owners"].get(part, [])
                    how = (f"open {owners[0]} (part {part})" if owners else
                           f"find part {part}: search for {part} (when a search by the code finds nothing, look for {part}"
                           f" in the titles of results for that kind of item) and open it")
                    text = (f"Checkout held once: {line} ({kind}) {shown} which names none of {target},"
                            f" and its page says the rest of its list is on part {part}'s page. Before checking out,"
                            f" {how}: when its sentence \"Part {index['records'][line]['self_code']} also fits: ...\" names none of"
                            f" {names} either, replace {line} with a {kind} that names one"
                            + (f" ({other} does)" if other else "") + "; otherwise check out this cart again.")
                    rule = "pointer"
                self.compat_held = True
                self.cart_guard_error = text
                self.trace.append({"kind": "compat_hold", "item": number, "line": line, "device": host,
                                   "ids": sorted(want), "to": other, "pointer": part, "rule": rule})
                return False
        return True

    def numeric(self, value: Any) -> tuple[float, str] | None:
        if type(value) in (int, float):
            try:
                number = float(value)
            except (ValueError, OverflowError):
                return None
            return (number, "") if -float("inf") < number < float("inf") else None
        if not isinstance(value, str):
            return None
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\d]*?)\s*", value)
        if not match:
            return None
        number = float(match[1])
        return (number, match[2].strip()) if -float("inf") < number < float("inf") else None

    def numeric_interval(self, value: Any) -> tuple[float, float, str] | None:
        scalar = self.numeric(value)
        if scalar:
            return scalar[0], scalar[0], scalar[1]
        if not isinstance(value, str):
            return None
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        match = re.fullmatch(r"\s*(" + number + r")\s*([^\d]*?)\s*(?:\.\.|[-–—]|\bto\b)\s*(" + number + r")\s*([^\d]*?)\s*", value)
        if not match:
            return None
        low, high = self.numeric(match[1]), self.numeric(match[3])
        first_unit, second_unit = match[2].strip(), match[4].strip()
        if not low or not high or low[0] > high[0] or (first_unit and second_unit and first_unit != second_unit):
            return None
        return low[0], high[0], first_unit or second_unit

    def cart_need_key(self) -> str:
        return _dump({"items": [(it.name, it.shelf, it.qty, [m for m in it.musts + it.told if m not in it.dropped])
                                for it in self.items], "budget": self.budget, "scope": self.brief.get("budget_scope")})

    def remember_cart_assignment(self, got: dict[str, Any], ids: list[str]) -> None:
        mapping: dict[int, str] = {}
        entries = got.get("cart_items")
        if isinstance(entries, list) and len(entries) == len(self.items):
            for entry in entries:
                if not isinstance(entry, dict):
                    break
                number, ident = entry.get("item"), entry.get("id")
                if type(number) is not int or not 1 <= number <= len(self.items) or number in mapping:
                    break
                if not isinstance(ident, str) or ident not in ids or ident in mapping.values():
                    break
                mapping[number] = ident
            if len(mapping) == len(self.items) and set(mapping.values()) == set(ids):
                self.cart_assignment = mapping
                self.cart_assignment_key = self.cart_need_key()
        path = got.get("cart_quantity_path")
        if isinstance(path, list) and 0 < len(path) <= 5 and all(isinstance(k, str) and k for k in path):
            self.cart_quantity_path = list(path)

    def observed_cart_counts(self, view: Any = None) -> dict[str, int | None] | None:
        """Lines of the one unambiguous line collection in an observed cart (the current cart view unless `view` is
        given), as {listing id: units}; None when there is no such collection. The units field is learned for this task
        from the observations themselves, never from the cart tool's parameter names: the line field holding the units
        Kyo's own cart calls asked for, else a line's single integer field besides the id and the price. Units stay None
        while that field cannot be told. A line under an id Kyo never saw keeps that id, so it reads as an extra line."""
        if view is None:
            if self.cart_view is None or self.cart_seen_at < self.cart_at:
                return None
            view = self.cart_view
        collections: list[list[Any]] = []

        def walk(node: Any, depth: int = 0) -> None:
            if depth > 6:
                return
            if isinstance(node, list):
                if any(isinstance(row, dict) and any(isinstance(v, str) and v in self.rows for v in row.values()) for row in node):
                    collections.append(node)
                else:
                    for child in node:
                        walk(child, depth + 1)
            elif isinstance(node, dict):
                for child in node.values():
                    walk(child, depth + 1)

        walk(view)
        if len(collections) != 1 or not all(isinstance(row, dict) for row in collections[0]):
            return None
        keys = {k for row in collections[0] for k, v in row.items() if isinstance(v, str) and v in self.rows}
        if len(keys) != 1:
            return None
        key = next(iter(keys))
        lines: list[tuple[str, dict[tuple[str, ...], int]]] = []
        for row in collections[0]:
            if sum(isinstance(v, str) and v in self.rows for v in row.values()) > 1:
                return None
            ident = row.get(key)
            if not isinstance(ident, str) or not ident.strip():
                continue  # no listing under the line key (a total, a discount): not a cart line
            numbers: dict[tuple[str, ...], int] = {}
            for name, value in row.items():
                for sub, number in (value.items() if isinstance(value, dict) else [(None, value)]):
                    if name != key and (type(number) is int or (type(number) is float and number.is_integer())):
                        numbers[(name,) if sub is None else (name, str(sub))] = int(number)
            lines.append((ident.strip(), numbers))
        if not lines:
            return None
        common = set(lines[0][1]).intersection(*(numbers for _, numbers in lines[1:]))
        hint = tuple(self.cart_quantity_path) if self.cart_quantity_path else None

        def pick(options: list[tuple[str, ...]]) -> tuple[str, ...] | None:
            # the review's optional path only breaks a tie between fields the observation itself cannot tell apart
            return options[0] if len(options) == 1 else (hint if hint in options else None)

        path = getattr(self, "cart_count_path", None)
        if path not in common:
            score = {p: sum(self.cart_lines.get(ident) == numbers[p] for ident, numbers in lines) for p in common}
            best = max(score.values(), default=0)
            path = pick(sorted(p for p, n in score.items() if n == best)) if best else None
            if path is not None:
                self.cart_count_path = path  # matched Kyo's own request: kept for this task's later observations
            else:
                prices = [self.code_price(ident) for ident, _ in lines]
                plain = [p for p in common if p != (self.price_key,) and all(
                    1 <= numbers[p] and (price is None or numbers[p] < price) for (_, numbers), price in zip(lines, prices))]
                path = pick(sorted(plain))
        counts: dict[str, int | None] = {}
        for ident, numbers in lines:
            units = numbers.get(path) if path is not None else None
            units = units if units is None or units >= 0 else None
            if ident in counts:
                units = None if units is None or counts[ident] is None else counts[ident] + units
            counts[ident] = units
        return {ident: units for ident, units in counts.items() if units != 0} or None

    def reconcile_quantities(self) -> bool:
        """Common purchase boundary. The cart's units come from the freshest observation: a cart view, taken again only
        while the steps leave room for it, the checkout and one spare step; else the cart tool's own reply to the latest
        change. The purchase is held only on a demonstrated contradiction (a requested item without its own line, an
        extra line, or a known count unlike the request); known wrong counts are repaired with validated calls and
        confirmed. Units that cannot be read do not hold the checkout."""
        self.cart_guard_error = ""

        def blocked(reason: str) -> bool:
            self.cart_guard_error = "Checkout held: " + reason
            self.trace.append({"kind": "cart_quantity", "status": "blocked", "reason": reason})
            return False

        def passed(status: str, counts: Any) -> bool:
            self.trace.append({"kind": "cart_quantity", "status": status, "counts": counts})
            return True

        def said() -> Any:
            # The cart tool's reply to the latest change, when its caller recorded it (note_cart clears older replies).
            try:
                return json.loads(self.cart_said) if self.cart_said else None
            except ValueError:
                return None

        def refresh(spare: int) -> bool:
            tool = self.roles.get("cart_view")
            if not tool or self.left() < self.tail() + 1 + spare:
                return False
            faults: list[str] = []
            args = _shape({}, self.schemas.get(tool), "args", faults)
            if faults:
                return False
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return False
            self.cart_view = reply.get("observation")
            self.cart_seen_at = self.step
            self.observe(tool, self.cart_view)
            return not self.finished

        before = self.cart_need_key()
        if self.cart_view is None or self.cart_seen_at < self.cart_at:
            refresh(1)
        if before != self.cart_need_key():
            return blocked("the cart review changed the request; review the new requirements")
        actual = self.observed_cart_counts()
        whole = actual is not None
        if not whole:
            answer = said()
            actual = self.observed_cart_counts(answer) if answer is not None else None
        if actual is None:
            return passed("unknown", None)
        lines = set(actual) if whole else set(self.cart_lines)
        if len(self.items) == 1 and len(lines) == 1:
            assignment = {1: next(iter(lines))}
        else:
            assignment = self.cart_assignment if self.cart_assignment_key == before else {}
        complete = set(assignment) == set(range(1, len(self.items) + 1)) and len(set(assignment.values())) == len(self.items)
        if not complete and len(self.items) > 1:
            # No usable review named the lines: assign each item the one cart line its current requirements fit by
            # the inherited code reading, only when that is unique and distinct for every item (else unknown).
            guess = {}
            for number in range(1, len(self.items) + 1):
                fits = [ident for ident in lines if self.code_fits(number, ident)]
                if len(fits) == 1:
                    guess[number] = fits[0]
            if set(guess) == set(range(1, len(self.items) + 1)) and len(set(guess.values())) == len(self.items):
                assignment, complete = guess, True
        # An unknown item-to-line assignment is uncertainty, not a fault (SPEC-006 repair): only a complete assignment
        # that the fresh cart contradicts (a missing or extra line) holds the purchase.
        if whole and complete and set(assignment.values()) != set(actual):
            return blocked("the cart's lines differ from the requested items; keep all quantities")
        if not complete:
            return passed("unknown", actual)
        wanted = {ident: self.items[number - 1].qty for number, ident in assignment.items()}
        known = {ident: units for ident, units in actual.items() if ident in wanted and units is not None}
        if not whole:
            # A tool's reply may show only the units this call added: a count that merely echoes the call's own units,
            # unlike Kyo's running record of the line, says nothing about the line.
            echo = getattr(self, "cart_call_units", {})
            known = {ident: units for ident, units in known.items()
                     if units == self.cart_lines.get(ident) or units != echo.get(ident)}
        if all(units == wanted[ident] for ident, units in known.items()):
            return passed(("pass" if whole else "reply") if len(known) == len(wanted) else "unknown", actual)
        cart = self.roles.get("cart") or ""
        props = (self.schemas.get(cart) or {}).get("properties") or {}
        one_count = sum(isinstance(s, dict) and "integer" in _kinds(s) for s in props.values()) == 1
        if not (self.cart_sets(cart) and one_count) and not self.cart_replaces(cart):
            return blocked("wrong quantities observed, but a quantity-setting tool is not established")
        if not self.code_budget_ok(wanted):
            return blocked("correct quantities would exceed the known budget or their prices are unknown")
        if whole:
            self.cart_lines = {ident: units if units is not None else self.cart_lines.get(ident, 0) for ident, units in actual.items()}
        calls = self.cart_calls(wanted, [])
        if calls is None or len(calls) + max(2, self.tail() + 1) > self.left():
            return blocked("not enough verified steps to correct quantities and re-read the cart")
        shaped = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return blocked("quantity correction does not match the tool schema")
            shaped.append(args)
        for args in shaped:
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return blocked("quantity correction did not complete")
            self.note_cart(cart, args)
            self.cart_said = _dump(reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            if before != self.cart_need_key():
                return blocked("the request changed during quantity correction")
        # Confirmation: a fresh view where the store has one (its steps were reserved above). Without one, the
        # correction's own reply can only agree with the validated call, so it never contradicts.
        seen = self.observed_cart_counts() if refresh(0) else None
        if before != self.cart_need_key():
            return blocked("the request changed during quantity correction")
        if seen is not None and (set(seen) != set(wanted) or any(units is not None and units != wanted[ident]
                                                                 for ident, units in seen.items())):
            return blocked("corrected quantities are not confirmed by the current cart")
        if any(not self.buyable(ident) for ident in wanted):
            return blocked("a corrected cart line is no longer available")
        # Quantity changes expire final-cart receipts; the checkout boundary re-runs any due order gate when a model
        # can run (and waives it when none can), so no separate hold here.
        self.trace.append({"kind": "cart_quantity", "status": "fixed", "before": actual, "counts": wanted})
        return True

    def review_ask(self, prompt: str, model: str, max_tokens: int) -> dict[str, Any] | None:
        """One review call. Haiku gets at least LIGHT_REVIEW_TOKENS; a Haiku reply emptied by its token limit is sent
        once more with twice the allowance. think()/room() admit (and shrink) every call under the task cap."""
        tokens = max(max_tokens, LIGHT_REVIEW_TOKENS) if model == LIGHT else max_tokens
        messages = [{"role": "user", "content": prompt}]
        self.last_cut = False
        message = self.think(messages, [model], tokens)
        if message is None and model == LIGHT and self.last_cut:
            self.last_cut = False
            self.trace.append({"kind": "review_cut_retry", "model": model, "tokens": tokens * 2})
            message = self.think(messages, [model], tokens * 2)
        return message

    def review_json(self, prompt: str, models: list[str], max_tokens: int, valid: Any = None) -> dict[str, Any] | None:
        """Preserve early routing; late reviews use only affordable light calls, including on fallback. With `valid`, a
        reply that comes back unparsable or fails `valid` gets one bounded retry, on the light model only and admitted
        by think()/room() like any call; after that the review is uncertain (None)."""
        if valid is None:
            for model in models:
                if self.cost >= TASK_CAP or self.broke or self.no_credit or self.calls >= ALL_CALLS:
                    return None
                if self.cost >= SPEND_SOFT and model != LIGHT:
                    continue
                got = _json_obj((self.review_ask(prompt, model, max_tokens) or {}).get("content"))
                if got is not None:
                    return got
                if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                    return None
            return None
        order, bad, index = list(models), False, 0
        while index < len(order):
            model = order[index]
            index += 1
            if self.cost >= TASK_CAP or self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
            if (self.cost >= SPEND_SOFT or bad) and model != LIGHT:
                continue
            message = self.review_ask(prompt, model, max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None and valid(got):
                return got
            if message is None:
                continue
            self.trace.append({"kind": "review_unusable", "model": model, "retry": not bad})
            if bad:
                return None
            bad = True
            if LIGHT not in order[index:]:
                order.append(LIGHT)
        return None

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
        # one review per set of listings, request state and listing records (C2r4: per listing set and version); a
        # quantity change alone does not repeat it, since the code reconciles quantities from the observed cart
        return _dump([sorted(ids), self.version, self.cart_need_key(),
                      [(ident, self.full.get(ident)) for ident in sorted(ids)], sorted(self.ref_values.items())])

    def check_cart(self, ids: list[str] | None = None, limit: float = TASK_CAP) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost >= limit or not ids:
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
        # a reply without a true/false verdict is unusable: one bounded light retry, then the review is uncertain
        got = self.review_json(prompt, [HEAVY, LIGHT], 3000, lambda reply: isinstance(reply.get("ok"), bool))
        # remember_evidence also keeps the reply's cart assignment (cart_items, unless its own entries contradict them)
        clean = self.remember_evidence(got if isinstance(got, dict) else {}, ids)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        # C2r4's verdict (a usable approving review), minus listings the code shows contradicting a need in force
        self.verdicts[key] = clean
        against = self.contradicted(ids)
        if against:
            self.leads = [i for i in self.leads if i not in ids]
            return ("The records contradict a requirement in force: " + "; ".join(against[:4])
                    + ". Replace that line with a listing that meets it (or re-read the record if it changed), then check out.")
        # proofs are advisory: an uncertain or unusable review holds nothing; a negative one holds once (C2r4), as does
        # an approval whose own entries call a listing's family different or a requirement unmet
        if not isinstance(got, dict) or (got.get("ok") is True and clean):
            return ""
        if got.get("ok") is not False:
            flagged = [f"{i} (item {n}): " + "; ".join(saved["errors"][:2]) for (n, i), saved in self.listing_evidence.items()
                       if i in ids and saved.get("errors") and saved.get("state") == self.evidence_state(i)]
            if not flagged:
                return ""
            return ("A careful review approved this cart but flagged: " + " | ".join(flagged[:3])
                    + ". Check those lines against the records; check out again when they fit.")
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
        while pending and self.trails < 4 and self.cost < TASK_CAP:
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
        if self.cost >= TASK_CAP:
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
        # matching records against the requirements is reading work: the light model first, the strong one only when
        # the light one gives no usable answer. The light model reasons before it answers (820-1,147 tokens in local
        # runs), so 1,200 left it with no answer in 3 of 8 swaps
        got = self.review_json(prompt, [LIGHT, HEAVY], SWAP_TOKENS)
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

    def compat_leads(self) -> dict[str, list[str]]:
        """SPEC-009 F2 at a decline (race 14: 64 of 85 compatibility episodes ended without a purchase; in 51 of 65 of
        those an opened accessory's own list named an opened device). For each accessory item that must fit a device
        item: an opened device listing whose page states a model code or also-sold-as name, and an opened accessory
        listing whose page says it fits that device (fits_device True), both buyable, neither shown to contradict a
        requirement, all leads together inside the budget. Item number -> listings. No model call."""
        pairs = [p for p in self.compat_pairs() if p[1] is not None]
        if not pairs:
            return {}
        index = self.compat_index()
        records = index["records"]
        opened = [i for i in reversed(self.seen_order) if (i in self.full or i in self.opened) and self.buyable(i)
                  and not self.contradicted([i])]
        devices = [i for i in opened if self.device_names(i, index) and not (records.get(i) or {}).get("listed")]
        for device in devices:
            names = self.device_names(device, index)
            leads: dict[str, list[str]] = {}
            for part, main, _, _ in pairs:
                fits = [a for a in opened if a != device and self.fits_device(a, names, index)[0] is True]
                if not fits:
                    break
                leads.setdefault(str(main), [device])
                leads[str(part)] = fits[:2]
            else:
                if not leads:
                    continue
                total = 0.0
                for number, ids in leads.items():
                    price = self.code_price(ids[0])
                    qty = self.items[int(number) - 1].qty if 0 < int(number) <= len(self.items) else 1
                    total = float("inf") if price is None else total + price * qty
                if self.budget is not None and total > float(self.budget):
                    continue
                return leads
        return {}

    def audit(self) -> str:
        if self.audited or self.cost >= TASK_CAP:
            return ""
        self.audited = True
        got = self.audit_call()
        if got is None:
            # SPEC-006 r14c: the review could not run (timeout, unaffordable, unusable). Opened listings whose records
            # meet every requirement by the inherited code reading, in stock and within budget, become leads, so the
            # decline is held once as with review leads (no new model call).
            named_code: dict[str, list[str]] = {}
            for number, item in enumerate(self.items, 1):
                ids = []
                for ident in reversed(self.seen_order):
                    price = self.code_price(ident)
                    over = self.budget is not None and (price is None or price * item.qty > float(self.budget))
                    if ident in self.full and self.code_fits(number, ident) and self.buyable(ident) and not over:
                        ids.append(ident)
                if ids:
                    named_code[str(number)] = ids[:3]
            if named_code:
                self.lead_items = named_code
                self.leads = [i for ids in named_code.values() for i in ids]
                self.trace.append({"kind": "audit_code_leads", "leads": self.leads})
                return ("The independent review could not run. By their opened records these listings meet every stated "
                        "requirement and fit the budget: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named_code.items())
                        + ". Buy what truly fits; end without a purchase only if each one fails.")
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            fitted = self.compat_leads()
            if fitted:
                self.lead_items = fitted
                self.leads = [i for ids in fitted.values() for i in ids]
                self.trace.append({"kind": "audit_compat_leads", "leads": self.leads})
                return ("By the store's own pages these opened listings fit together: "
                        + "; ".join(f"item {k}: {', '.join(v)}" for k, v in fitted.items())
                        + " (each accessory's own list names the device's model code or name). Check them against the other"
                        " requirements and buy what truly fits; end without a purchase only if each one fails.")
        if not named:
            active = set(self.needs_now())
            unknown = []
            for entry in _items((got or {}).get("unknown")):
                if isinstance(entry, dict) and type(entry.get("item")) is int and isinstance(entry.get("need"), str):
                    pair = (entry["item"], entry["need"])
                    if pair in active and pair not in unknown:
                        unknown.append(pair)
            if unknown:
                self.decline_unknown = unknown
                self.trace.append({"kind": "decline_unknown", "needs": unknown})
                return ("Missing listing evidence does not establish that nothing fits. Keep these requirements active: "
                        + "; ".join(f"item {n}: {need}" for n, need in unknown)
                        + ". Review promising existing records within the remaining resources. Buy only after verification; "
                        "if verification remains impossible, end without claiming the store has no fitting item.")
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.review_json(prompt, [HEAVY, LIGHT], 1500)
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
                if not self.cart_replaces(name):
                    # An array tool not documented as replacing the whole cart may add to it: only absent lines are
                    # sent; changing a present line's count or dropping a line cannot be said safely.
                    add: dict[str, int] = {}
                    for i, n in lines.items():
                        have = self.cart_lines.get(i, 0) if i in self.cart_ids() else 0
                        if have == n:
                            continue
                        if have:
                            return None
                        add[i] = n
                    if drop:
                        return None
                    return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in add.items()]}] if add else []
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
        plan: dict[str, int] = {}
        assignment: dict[int, str] = {}
        current = self.cart_assignment if self.cart_assignment_key == self.cart_need_key() else {}
        for number, item in enumerate(self.items, 1):
            choices = list(self.lead_items.get(str(number), []))
            if number in current:
                choices.append(current[number])
            pick = next((i for i in choices if i in self.rows and i not in plan and not self.sold_out(i)), None)
            if not pick:
                return ""
            plan[pick] = item.qty
            assignment[number] = pick
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        drop = [i for i in now if i not in plan]
        lines = plan
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), TASK_CAP)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        self.cart_assignment = assignment
        self.cart_assignment_key = self.cart_need_key()
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
            self.cart_said = _dump(reply.get("observation"))
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
                self.check_cart(list(lines), TASK_CAP)
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
        description = next((t["function"].get("description", "") for t in self.tools if t["function"]["name"] == name), "")
        documented = bool(re.search(r"\bset\b[^.]{0,80}\bquantit", description, re.I))
        return documented or any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def cart_replaces(self, name: str) -> bool:
        """Only an explicitly documented whole-cart replacement permits automatic array correction."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        arrays = [k for k, s in props.items() if isinstance(s, dict) and "array" in _kinds(s)
                  and isinstance(s.get("items"), dict) and isinstance(s["items"].get("properties"), dict)]
        description = next((t["function"].get("description", "") for t in self.tools if t["function"]["name"] == name), "")
        inner = props[arrays[0]]["items"]["properties"] if len(arrays) == 1 else {}
        identifiers = [k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        quantities = [k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)]
        return (len(identifiers) == len(quantities) == 1 and len(arrays) == 1
                and all(k == arrays[0] for k in schema.get("required", []))
                and bool(re.search(r"\breplace\b[^.]{0,50}\bcart\b|\bset\b[^.]{0,20}\b(?:entire|whole) cart\b", description, re.I)))

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        # The caller records the tool's reply to this change after this note (the main loop does); a reply to an
        # earlier change is not the cart after this one (reconcile_quantities).
        self.cart_said = ""
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
        self.cart_call_units = dict(lines)  # units this call names per listing, which a reply may only echo
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
        observation = self.adapt_observation(observation)

        def present(node: Any, depth: int = 0) -> Any:
            if depth > 7:
                return "[omitted: nesting limit; original observation retained]"
            if isinstance(node, dict):
                if self.ident(node, True, depth):
                    # Facts and references precede decorative titles. Keep complete scalar facts
                    # instead of shortening an entire serialized listing at SHOW_CHARS.
                    keys = sorted(node, key=lambda k: (0 if isinstance(node[k], dict) else
                        1 if k == self.id_key else 2 if set(_words(k).split()) &
                        {"description", "blurb", "comments", "comment"} else 3))
                    return {k: _clip(node[k], 2000 if isinstance(node[k], dict) or
                        set(_words(k).split()) & {"description", "blurb", "comments", "comment"} else SHOW_CHARS)
                        for k in keys}
                return {k: present(v, depth + 1) for k, v in node.items()}
            if isinstance(node, list):
                return [present(v, depth + 1) for v in node]
            if isinstance(node, str) and _text_record(node, self.id_key)[1]["status"] != "not_record":
                return "[unparsed record: evidence unknown, not proof of no fit] " + node[:3000]
            return _clip(node, SHOW_CHARS)

        text = _dump(present(observation))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT - 100] + "...[omitted: observation limit; original records retained, reopen/compare fewer IDs]"
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
                    swept = self.cap_sweep()   # F3 (SPEC-009): before the re-read, which then reads a swapped-in line
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
                    read = swept + self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
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
                            self.check_cart([first], TASK_CAP)
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
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.is_buy(name, args) and not self.reconcile_quantities():
                # quantities are settled before the order gate, so its receipt is for the cart actually bought
                out.append((cid, self.cart_guard_error or "Not run yet: the cart's quantities need correcting."))
                continue
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, self.cart_guard_error if (self.is_buy(name, args) or name == self.roles.get("cart")) and self.cart_guard_error
                            else "The call failed (network or server). You may try again."))
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
            # cart lines (and a checkout reply's lines) are bookkeeping: they update a listing's known fields only
            self.absorb(observation, name in (self.roles.get("cart"), self.roles.get("cart_view")) or self.is_buy(name, args))
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
        constrained = self.spent_out or self.no_credit or self.loop_spent
        if constrained and buy and self.code_order():
            return ""
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if not constrained and buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
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
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if not constrained and self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def code_fits(self, number: int, ident: str) -> bool:
        """The code's own reading (C2r4): the listing's opened record shows every requirement in force on item `number`,
        each under a key naming the requirement's property, with all the words of the required value. A fallback pick
        only, used when no model can review; a reviewed listing is preferred (best_candidate)."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        needs = [need for n, need in self.needs_now() if n == number]
        if not block or not needs:
            return False
        for need in needs:
            prop, _, value = need.partition(":")
            words = set(_words(value).split())
            keys = [k for k in block if _names(_stem(prop), _stem(k))]
            if not words or not any(words <= set(_words(str(block[k])).split()) for k in keys):
                return False
        return True

    def best_candidate(self, number: int, taken: set[str]) -> str | None:
        """For item `number`: the best checked fit (a current opened listing whose review passed, the reviewed current
        selection first, newest first); else, as C2r4, the listing the independent check confirmed, the held cart
        line, the review's fits, then opened listings the code reads as fitting (newest first). A cheapest request
        sorts all of them by known price (checked fits first on a tie). Never sold out, over budget, taken, or a
        listing a review showed to contradict a requirement."""
        need_key = self.cart_need_key()
        current = self.cart_assignment.get(number) if self.cart_assignment_key == need_key else None
        checked = [i for i in dict.fromkeys(([current] if current else []) + list(reversed(self.seen_order)))
                   if self.verified_listing(number, i)]
        held = self.cart_ids()
        line = current if current in held else (held[0] if len(self.items) == 1 and len(held) == 1 else None)
        loose: list[str] = []
        if self.confirmed and len(self.items) == 1:
            loose.append(self.confirmed)
        if line:
            loose.append(line)
        loose += self.lead_items.get(str(number), [])
        loose += [i for i in reversed(self.seen_order) if i in self.full and self.code_fits(number, i)]
        qty = self.items[number - 1].qty

        def usable(ident: str) -> bool:
            price = self.code_price(ident)
            over = self.budget is not None and (price is None or price * qty > float(self.budget))
            return ident in self.rows and ident not in taken and self.buyable(ident) and not over

        # SPEC-006 r14c: a contradicted listing is ranked last, not excluded; completion does not decline while an
        # affordable, buyable candidate exists.
        order = [i for i in dict.fromkeys(checked + loose) if usable(i)]
        against = {i for i in order if self.contradicted([i])}
        # SPEC-007 W3: listings whose own attributes state more of the item's requirements first (contradicted last)
        literal = {i: self.literal_count(number, i)[0] for i in order}
        # SPEC-009 F2: for an accessory line that must fit a device, listings whose lists name it first (all 0 otherwise)
        fit = self.compat_rank(number, order, taken)
        if self.brief.get("pick") == "cheapest":
            order = [i for i in order if self.code_price(i) is not None]
            order.sort(key=lambda i: (i in against, fit[i], -literal[i], self.code_price(i), i not in checked))
        else:
            order.sort(key=lambda i: (i in against, fit[i], -literal[i], i not in checked))
        return order[0] if order else None

    def code_price(self, ident: str) -> float | None:
        """The listing's price for the code's own checks: under the price key the model named, else the first number
        under a key whose name says price (search row first, then the opened record)."""
        known = self.price_of(ident)
        if known is not None:
            return known
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            for key, value in record.items():
                if "price" in _words(key).split() and isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
        return None

    def open_unread(self, number: int, taken: set[str], below: float | None = None) -> bool:
        """Read bounded unseen alternatives while preserving cart/checkout steps. Cheapest requests read cheaper
        known-price rows first; other requests rank overlap with the item requirements."""
        words = {w for n, need in self.needs_now() if n == number for w in _words(need.partition(":")[2]).split()}
        if not words:
            return False
        qty = self.items[number - 1].qty
        rows: list[tuple[float, int, str]] = []
        for index, ident in enumerate(self.seen_order):
            if ident in self.full or ident in taken or not self.buyable(ident):
                continue
            price = self.code_price(ident)
            if self.budget is not None and price is not None and price * qty > float(self.budget):
                continue
            if below is not None and (price is None or price >= below):
                continue
            rank = price if self.brief.get("pick") == "cheapest" and price is not None else -len(words & set(_words(_dump(self.rows.get(ident))).split()))
            rows.append((rank, index, ident))
        ids = [ident for _, _, ident in sorted(rows)[:UNREAD_LIMIT]]
        if not ids:
            return False
        before = len(self.full)
        self.fill_facts(ids, len(self.items) + len(self.cart_ids()) + 2)
        self.trace.append({"kind": "open_unread", "item": number, "rows": ids, "opened": len(self.full) - before})
        return len(self.full) > before

    def code_budget_ok(self, plan: dict[str, int]) -> bool:
        if self.budget is None:
            return True
        prices = {i: self.code_price(i) for i in plan}
        if any(p is None for p in prices.values()):
            return False
        totals = [prices[i] * n for i, n in plan.items()]
        amount = max(totals, default=0) if self.brief.get("budget_scope") == "each" else sum(totals)
        return amount <= float(self.budget)

    def code_order(self, attempts: int = 2) -> bool:
        """Completion after the loop stopped (its spend limit, the credit, no affordable model): order the best checked
        fit for each item, else C2r4's pick (best_candidate). When the intended cart has no current proof and a model
        may still be affordable, ONE review (CHECK, through review_json/think, so light past the soft limit and admitted
        by room()) runs on it first; after it no model is called. A review that cannot run or gives no usable proof
        does not stop the order (C2r4); a demonstrated contradiction still does. Replan once if a newly carted fit is
        already sold out; never buy known unavailable lines or exceed the total budget."""
        # the code finishes alone: no model call from here on but the one completion review below
        self.code_finishing = True
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not cart or not buy or self.finished or self.broke:
            return False

        def pick(explore: bool) -> tuple[dict[str, int], dict[int, str]] | None:
            plan: dict[str, int] = {}
            assignment: dict[int, str] = {}
            for number, item in enumerate(self.items, 1):
                choice = self.best_candidate(number, set(plan))
                cheapest = self.brief.get("pick") == "cheapest"
                if explore and (not choice or cheapest) and self.open_unread(number, set(plan), self.code_price(choice) if choice and cheapest else None):
                    choice = self.best_candidate(number, set(plan))
                if not choice:
                    self.trace.append({"kind": "code_order", "done": False, "missing_item": number})
                    return None
                plan[choice] = item.qty
                assignment[number] = choice
            return (plan, assignment) if plan else None

        picked = pick(True)
        if picked is None:
            return False
        plan, assignment = picked
        proved = all(self.verified_listing(n, i) for n, i in assignment.items())
        if (not proved and not getattr(self, "completion_checked", False) and not self.no_credit
                and self.calls < ALL_CALLS and self.cost < TASK_CAP):
            # the completion review: the intended cart, judged as planned at the requested quantities
            self.completion_checked = True
            self.cart_assignment = assignment
            self.cart_assignment_key = self.cart_need_key()
            before = self.calls
            self.code_finishing = False
            try:
                self.check_cart(list(plan))
            finally:
                self.code_finishing = True
            proved = all(self.verified_listing(n, i) for n, i in assignment.items())
            self.trace.append({"kind": "completion_check", "plan": plan, "calls": self.calls - before, "proved": proved,
                               "cost": round(self.cost, 5)})
            picked = pick(False)
            if picked is None:
                return False
            plan, assignment = picked
        if not self.code_budget_ok(plan):
            self.trace.append({"kind": "code_order", "done": False, "reason": "budget"})
            return False
        self.cart_assignment = assignment
        self.cart_assignment_key = self.cart_need_key()
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or len(calls) + self.tail() > self.left():
            self.trace.append({"kind": "code_order", "done": False, "plan": plan, "calls": bool(calls)})
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
            self.cart_said = _dump(reply.get("observation"))
            self.observe(cart, reply.get("observation"))
        if any(not self.buyable(i) for i in plan):
            return self.code_order(attempts - 1) if attempts > 1 else False
        if not self.code_budget_ok(plan):
            return False
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return False
        reply = self.act(buy, args)
        self.trace.append({"kind": "code_order", "done": reply is not None, "plan": plan, "reply": _clip(reply, 200),
                           "held": "" if reply is not None else self.cart_guard_error})
        if reply is None:
            # held at the checkout boundary: when a review showed a planned line to contradict a requirement, one
            # replan without it (no model call)
            if attempts > 1 and self.contradicted(list(assignment.values())):
                return self.code_order(attempts - 1)
            return False
        self.observe(buy, reply.get("observation"))
        return self.finished

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
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and not self.spent_out and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, min(SPEND_HARD + 0.01, TASK_CAP))
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        self.run_loop(loop_cap, spend_cap)
        if not self.finished and self.cost >= spend_cap:
            # the loop ran out of budget, not of work: the ending must not decline for that reason alone (R-02)
            self.loop_spent = True

    def run_loop(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and not self.time_up()):
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
