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

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
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

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
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
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
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
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # all messages but the last first; the last re-asks a question whose answer wanted a single question
        # (named with example values this time), else it goes to the next pick
        head = ordered[: count - 1] if count >= 2 else ordered[:count]
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        heard = self.hear(batch)
        self.take_heard(heard, told)
        if count >= 2 and self.msgs_free() > 0:
            crowded =[batch[i - 1] for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(batch)]
            again = None
            for number, text, _ in crowded:
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

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

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
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
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
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
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

    def audit(self) -> str:
        if self.audited or self.cost > SPEND_SOFT:
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
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
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
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
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


# ---------- Cristopher's own layer over the public Kyo: guards over store mechanics, each behind its own switch ----------
# Every mechanism waits for its own trigger in the store's replies; until then the episode runs as the parent's would.
REPEAT_GUARD = True       # a finishing call identical to one the store answered with an error is held once while nothing has changed
STATED_LIMIT_CODE = True  # a step limit the request states binds even when the model's reading left it out
CART_RECONCILE = True     # a list cart call whose reply still shows lines the list left out: the record follows the store's own view
LIST_LIMITS = True        # a list argument's cap is learned from the first refused call and written into the live schema
CLOSEST_ORDER = True      # a decline or a forced ending with nothing carted, past every hold, orders the closest read fit when its record supports it
AUDIT_ITEMS = 3           # 0 = off. A request of this many items or more skips the parent's costly review of a decline
CHASE_ROUNDS = 8          # 0 = off. Right after the questions, code follows each requirement that points at another listing, page by page, up to this many model rounds
CHASE_STEPS = 14          # store calls an episode's chase may use
CHASE_KEEP = 6            # steps the chase leaves for the rest of the episode, whatever the request's size
CHASE_KEEP_ITEM = 4       # and for each item of the request
CHASE_SPEND = 0.012       # model spend an episode's chase may use, so the rest of the episode keeps its own room
CHASE_TOKENS = 4000       # the light model can think through a small reply and return nothing
CHASE_OPEN = 2            # rows a round of the chase opens: a model round costs far less than the steps a wrong row takes
CHASE_ITEMS = 2           # the chase runs on requests of at most this many items; steps and spend go further there
ROW_CHARS = 100             # 0 = off. On requests of AUDIT_ITEMS or more items, each field of a search's rows is shown up to this long; the rows stay in full in the record
TRAIL_ROWS = True        # the parent's check before a checkout shows each record with its id first and a page's own fields next, so a long row cannot cut them off
FILL_ASK = True          # off = frozen ky2. On: a message that would otherwise go unsent asks the next open property, right after the questions when the request names no step of its own, else at the checkout after the request's own steps there
STACK_ASK = True         # off = kira-r15f. On: the first message asks one property the item's own listings carry, the planner's likeliest; each later message names up to STACK_SIZE of one item's listing keys, the planner's next pick first, then the keys most listings carry; an answer to a list that says it asked too much, or asks for it again, turns the rest to one property per message
STACK_SIZE = 8            # properties a later message names at most
STACK_SETTLE = True      # off = kira-r15sc. On: an answer to a list that reveals nothing closes only its first property (a shopper reads the first listed one most surely); the rest stay askable
POOL_BEFORE_DECLINE = True  # off = kira-r15sd. On: before the closest order at an ending without a purchase, an item with two or more heard requirements gets the unopened rows of the logged searches whose own words hold a heard value read, and, when still nothing read fits it, one search joining the heard values with its name
MULTI_NEED_HOLD = True   # off = kira-r15sd. On: at the checkout, a line of an item with two or more heard requirements that does not state them all (read as the closest order reads them), while an opened listing does, holds the checkout once
WORD_VALUE_SEARCH = True  # off = kira-r15sg. On: right after the questions, a heard requirement whose value holds no figure is searched as that value as heard with the item's name, after the figure searches and within the same BUY_SEARCHES
PLAN_SKIP = True         # off = kira-r15sg. On: the question planner also marks the listed keys that are not about the product itself, and a later message lists those keys last
ONE_VALUE_KEEP = True    # off = kira-r15sh. On: a key that every listing carrying it gives one same value stays in the item's lists, in its place by how many listings carry it (after the keys of that many listings that show several values)
LISTS_AFTER_UNCLEAR = True  # off = kira-r15si. On: an answer to a list that only says it did not follow the question counts as an answer revealing nothing (the list's first name closed, lists go on); only an answer saying the message asked too much turns the rest to single questions
ITEM_SWEEP = True        # off = kira-r15sj. On: after the questions, and again at an ending without a purchase, each item with a heard requirement and no read listing stating them all gets its value-holding rows read and one joined search with its top rows read; searches go out as the agent's own latest search with the words replaced when the tool takes more than its words
ORDER_OVER_DECLINE = True  # off = kira-r15sj. On: the closest order at an ending without a purchase takes a read listing that states every heard requirement and contradicts nothing, though the record is silent on a requirement of the request
HEAR_POLARITY = True     # off = kira-r15sj. On: the answer reader is told that an answer turning down everything other than a value wants that value
SEARCH_FIX = True        # off = kira-r15sj. On: a figure counts as a value to search only when it stands on its own (not inside a word), and a negated requirement is not searched
CHECK_SWAP = True        # off = kira-r15sj. On: at a checkout sent again with the cart the independent check found wrong, a listing that check named is put in the cart in the line's place when it is read, buyable within the budget, states every heard requirement and contradicts none; once per item
HOLD_GUARD = True        # off = kira-r15sj. On: a cart line the independent check passed under the requirements now in force is not held for a record that is silent on a requirement
CHECK_FILTER = True      # off = kira-r16a. On: when the independent check objects to a complete cart whose lines state every heard requirement, and no listing it names does, its objection is dropped
SWAP_AT_CHECK = True     # off = kira-r16a. On: CHECK_SWAP's replacement is made as soon as the check names the listing (when the cart's own line also states every heard requirement, only for a pick rule the records settle in its favour)
CART_OVER_DECLINE = True  # off = kira-r16a. On: an ending without a purchase sent while the cart is complete and its lines state every heard requirement becomes the checkout of that cart, once
LATE_LISTS = 5            # 0 = off = kira-r16b. N: with fewer than N messages for the questions, lists go only in the last half of them (rounded down) and each earlier message names one property; with N or more, every message after the first is a list
REFUSED_FIRST = True     # off = kira-r16b. On: after a list the shopper refused for its length, a single question takes the planner's next pick wherever it stands, then that list's own names in their order, before any key the list left out
REARM_ORDER = True       # off = kira-r16c. On: an order Kyo sent in place of an ending without a purchase (the complete cart's checkout, the closest order) that the store refused does not use up its one turn: it may act once more at a later ending, on the cart and the requirements as they stand then
FACT_TEXTS = True        # off = kira-r16c. On: what the keeper's own holds hand back for a call they did not run says what was found and that the call was not run, and leaves what to do with it to the agent
STRICT_OVERRIDE = True   # off = kira-r16c. On: an order sent in place of an ending without a purchase needs every requirement of the request, written or heard, stated by the listing's record; a record silent on one is unproven and does not qualify (a forced ending keeps its closest fit)
FIRST_LIST = 2            # 0 = off = kira-r16c. N: on a request of N or more items the first message is a list too, led by the property it would have asked alone (never when the request itself says what to ask first); once the shopper has answered a list, every later message is a list
ENDING_ROOM = 0           # 0 = off = kira-r16f. N: until the store has answered one finishing call of the episode, N more steps are kept back for the ending, so a first ending the store turns down with a change still leaves room to act on it and end again
FACT_PLAIN = True        # off = kira-r16f. On: a hold's text ends at what was found and that the call was not run (it says nothing about sending the call again), and the page-chain hold names only what is still to open
PICK_CHECKED = False      # off = kira-r16h. On: a listing the independent check failed on its own, under the requirements now in force, does not hold a checkout as the better pick
END_LADDER = 0            # 0 = off = kira-r16h. N: at an ending without a purchase with an item not carted, before the closest order, up to N searches by code in words no search has used: a value past a stated bound as the item's rows write it, then each phrase of the item's name alone; their rows are read, those holding a requirement's value first. A closest order under a most or least pick then takes the best read fit by that measure
LADDER_OPENS = 6          # listings the ladder reads in an episode at most
ROLE_FILL = False         # off = kira-r16h. On: when the reading of the request names no tool for ending without a purchase, one short question on the remaining tools names it
FLAT_RECORDS = True      # off = kira-r16h. On: a tool result that carries listings as flat text ("key: value | key: value", with "a=b; c=d" inside a field, or a one-string wrapper round such a text) is read into records where results enter the agent; results in any other shape stay as they are
SHAPE_KEYS = True        # off = kira-r16h. On: when pages were opened and the readers hold no full record, one short model question per item lists the property names those pages show, and the interview uses them
SHAPE_ASK = """A shopping agent opened these pages in an online store, in a form its reader could not split into fields. List the names of the properties they show that a shopper could hold a preference on for the item named below: the product's own attributes, not ids, prices, stock, sellers or free text. Use the pages' own words. Return ONE JSON object and nothing else: {"names": [up to 12 property names, those most pages carry first]}"""
SEARCH_MODE = False       # off = kira-r16h. On: a search sent by code takes each fixed-choice argument as most of the agent's own searches with words set it, not as its latest one did (that one may be a mode that ignores the words)
CHASE_SEARCH = False      # off = kira-r16h. On: the chase's search for a link also goes out on a search tool that takes more than its words (as the agent's own latest search, with only what that tool requires)
LAST_SWAP = True         # off = kira-r16h. On: a cart change split over several calls in one turn that cannot all run before
                          # the kept last steps runs not at all, so the cart stands whole instead of losing a line it never gets back.
FINISH_CLOCK = 250          # 0 = off = kira-r16h. T: past T seconds of an episode no model call runs, so reading and the judges stop and the
                          # ordinary finish closes the episode with the cart it has; each model call's timeout is held to the room left.
FINISH_END = 285          # seconds by which the last model call of an episode is over
FINISH_CALL = 50          # a model call's usual timeout
FINISH_FLOOR = 3          # the shortest timeout a call is given
ROLE_ASK = """A shopping agent has a store's tools. One tool checks out. Say which of the other tools ends the episode without buying anything. Return ONE JSON object and nothing else: {"refuse": the exact tool name, or null when none of them does}"""
SWEEP_OPENS = 12          # listings the sweeps read in an episode at most
SWEEP_ROWS = 4            # seen rows a sweep reads for one item
SWEEP_TOP = 6             # rows of a sweep's search it reads
POOL_OPENS = 8            # listings the pool reads in an episode at most, logged rows and the joined searches' hits together
BUY_SEARCH = True        # off = kira-r15s. On: right after the questions, a heard requirement holding a figure is searched by its value as written with the item's name (an item name holding figures, also without them); with a search tool that takes more than its words, the agent is told those searches
BUY_OPEN = True          # off = kira-r15s. On: before the checkout's check and before an ending without a purchase, the unopened hits of each search of 24 matches or fewer whose words held a heard requirement's value are read; at the checkout, an opened listing stating every requirement where the cart's line does not holds it once
BUY_PICK = True          # off = kira-r15s. On: under a cheapest, most or least pick, an opened listing stating every requirement that beats the cart's line on the records' own values holds the checkout once
BUY_KEEP = True          # off = kira-r15s. On: the closest order at an ending without a purchase also counts a requirement the records say nothing about as stated by a key of a related name or by the listing's title
BUY_SEARCHES = 3          # value searches an episode makes at most
BUY_OPENS = 8             # listings BUY_OPEN reads in an episode at most

CHASE = """A shopping agent has just finished asking the shopper its questions. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read). A page's words can concern a different property: follow only what concerns this requirement's own property.
One seller can have many listings of a kind, and a row does not show the details a page's own notes state. So a link is settled only by an OPENED page whose own words state the details the requirement, or the previous page, uses to describe that link. A page that does not state them is not the link, even when the seller and the kind match.
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "no_other_listing": true when the requirement only relates to a listing the agent is still choosing itself and names no seller or other listing, else false, "settled": [ids of the OPENED pages already settled as links, in order], "link_seller": the seller the next link names, exactly as written, or "", "read_next": [ids of at most 2 ROWs that could be the next link and are not OPENED, the likeliest first: rows of that seller whose own words fit the details stated for the link, or the first rows of a search for those details], "finished": true when the last settled page states this property's value itself and sends the reader nowhere further for it, else false, "value": the value the requirement asks for as that last page gives it (when the page pairs its own value with another, saying one goes with the other, the partner value), or "", "search": while no search has yet found this link: the seller's name, the kind and the most telling detail stated for the link, at most five plain words, fewer after a search that found nothing, not tried before; else ""}]}"""

# the parent's answer reader, also asked to mark an answer that did not follow its question or asks for it again
STACK_HEAR = HEAR.replace(', "budget": new total limit', ', "unclear": [numbers of the questions whose answer says it did not follow the question or asks for it again in simpler words], "budget": new total limit', 1)

# the parent's question plan, also asked to mark the listed keys that are not about the product itself
STACK_PLAN = PLAN.replace(', or an empty string"}]}', ', or an empty string"}], "skip": [the numbers of the listed attribute keys that describe a listing\'s paperwork, its identity codes or the seller\'s terms rather than the product a shopper would choose between]}', 1)

# one more line for the answer readers (HEAR_POLARITY)
POLARITY = "\n- An answer that turns down everything other than one value wants that value: state the need as that value, never as something to avoid."

STEP_FIGURE = r"(\d[\d,]*)\s+(?:(?:more|total|tool|store)\s+)?(?:steps?|moves?|actions?|turns?|calls?)\b"
NOT_WORDS = r"\b(?:not|no|without|except|other than)\b"
FLOOR_WORDS = r"\b(?:least|min\w*|more|over|above)\b"
CEILING_WORDS = r"\b(?:most|max\w*|under|below|up to|less)\b"


def _plain(text: Any) -> set[str]:
    """A text's lower-case words, with a plural's final s dropped (the spelling rule the reading rules allow)."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower())}


def _scalars(record: Any, depth: int = 0) -> list[str]:
    """Every string or number a record holds, nested blocks included."""
    if isinstance(record, dict):
        return [t for v in record.values() for t in _scalars(v, depth + 1)] if depth < 4 else []
    if isinstance(record, list):
        return [t for v in record for t in _scalars(v, depth + 1)] if depth < 4 else []
    if isinstance(record, bool) or record is None:
        return []
    return [str(record)]


def _figure_pairs(text: str) -> list[tuple[float, str]]:
    """(number, the letters written right after it) for each figure in a text."""
    return [(float(n), u) for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]*)", text.lower())]


def _flat_value(text: str) -> Any:
    """One field's text as the value it writes: a flag, an empty mark, a number, or inner "a=b; c=d" pairs as a record."""
    plain = text.strip()
    if plain in ("True", "False"):
        return plain == "True"
    if plain == "None":
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", plain):
        return float(plain) if "." in plain else int(plain)
    pairs: list[list[str]] = []
    for part in plain.split("; "):
        key, eq, value = part.partition("=")
        if eq and re.fullmatch(r"[^\s=;|][^=;|]{0,40}", key.strip()):
            pairs.append([key.strip(), value.strip()])
        elif pairs:
            pairs[-1][1] += "; " + part
        else:
            return plain
    return {key: value for key, value in pairs} if len(pairs) >= 2 else plain


def _flat_record(text: Any) -> dict[str, Any] | None:
    """A "key: value | key: value" text as a record; None when it is not one (three fields at least, a key is a short name, and a
    part without a key belongs to the field before it)."""
    if not isinstance(text, str) or " | " not in text:
        return None
    fields: list[list[str]] = []
    for part in text.split(" | "):
        key, sep, value = part.partition(":")
        if sep and re.fullmatch(r"[A-Za-z][\w \-]{0,40}", key.strip()) and key.strip() not in {name for name, _ in fields}:
            fields.append([key.strip(), value])
        elif fields:
            fields[-1][1] += " | " + part
        else:
            return None
    if len(fields) < 3:
        return None
    return {key: _flat_value(value) for key, value in fields}


def _flat_read(node: Any, depth: int = 0) -> Any:
    """The result with its flat listing texts read into records: the strings of a list, and a one-string wrapper. Nothing else changes."""
    if depth > 4:
        return node
    if isinstance(node, list):
        return [(_flat_record(v) or v) if isinstance(v, str) else _flat_read(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            only = next(iter(node.values()))
            record = _flat_record(only)
            if record:
                return record
        notes = len(node) == 2 and any(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in node.values())
        return {k: ((_flat_record(v) or v) if notes and isinstance(v, str) else _flat_read(v, depth + 1) if isinstance(v, (list, dict)) else v) for k, v in node.items()}
    return node


class Keeper(Kyo):
    """The parent's episode with own guards. Each override calls the parent first or last and changes nothing until its
    trigger is seen; what fired is written to the returned log."""

    def __init__(self, problem: dict[str, Any]):
        self.began = time.monotonic()
        self.clock_out = False
        super().__init__(problem)
        if FLAT_RECORDS and self.view.get("opening") is not None:
            self.view = {**self.view, "opening": _flat_read(self.view.get("opening"))}
        self.finish_fails: dict[str, int] = {}
        self.fail_state = ""
        self.cart_reply: Any = None
        self.cart_last: Any = None
        self.cart_note = ""
        self.list_floor: dict[tuple[str, str], int] = {}
        self.closest_sent = False
        self.passed_under: dict[str, str] = {}
        self.closest_why: dict[str, dict[str, Any]] = {}
        self.chase_calls = 0
        self.chase_spent = 0.0
        self.audit_skipped = False
        self.chase_words: set[str] = set()
        self.chase_read_all = False
        self.chase_hits: list[str] = []
        self.dry_searches = 0     # searches in a row that led to nothing to open
        self.no_reader: set[str] = set()
        self.fills = 0
        self.stack_done: dict[int, set[str]] = {}
        self.stack_carry: dict[int, list[str]] = {}
        self.stack_off = False
        self.stack_listed = False   # a list went out and was not refused for its length
        self.stack_skip: set[str] = set()
        self.buy_searched: set[str] = set()
        self.buy_log: list[dict[str, Any]] = []
        self.buy_opened = 0
        self.buy_held: set[str] = set()
        self.pool_opened = 0
        self.pool_searched: set[int] = set()
        self.multi_held: set[str] = set()
        self.multi_covered: set[str] = set()
        self.sweep_opened = 0
        self.ladder_opened = 0
        self.raw_pages: list[str] = []
        self.shape_named: dict[int, list[str]] = {}
        self.sweep_searched: set[tuple[int, str]] = set()
        self.swapped: set[int] = set()
        self.cart_ordered = False
        self.rearmed: set[str] = set()
        self.end_tried = False   # the store has answered a finishing call of this episode

    def slip(self, where: str, err: Exception) -> None:
        """A guard that failed stays out of the way and leaves a mark in the log."""
        self.trace.append({"kind": "keeper_error", "where": where, "error": f"{type(err).__name__}: {err}"[:300]})

    # ---------- every store call passes here ----------
    def tail(self) -> int:
        """ENDING_ROOM: the steps kept for the finish, with more kept back while the store has not yet answered a finishing call."""
        return super().tail() + (ENDING_ROOM if ENDING_ROOM and not self.end_tried else 0)

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if self.past_clock() and name not in {self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("cart")}:
            return None
        reply = super().act(name, args)
        if isinstance(reply, dict):
            if FLAT_RECORDS:
                reply["observation"] = _flat_read(reply.get("observation"))
            seen = reply.get("observation")
            if SHAPE_KEYS and name in self.page_tools():
                self.raw_pages = (self.raw_pages + [_dump(seen)[:1600]])[-6:]
            try:
                if name in (self.roles.get("buy"), self.roles.get("refuse")):
                    self.end_tried = True
                if name == self.roles.get("cart"):
                    self.cart_reply = self.cart_last = seen
                if (BUY_OPEN or POOL_BEFORE_DECLINE or ITEM_SWEEP) and name == self.roles.get("search"):
                    target = self.single_target(name)
                    words = args.get(target) if target else " ".join(str(v) for v in args.values() if isinstance(v, str))
                    self.buy_log = (self.buy_log + [{"tool": name, "words": str(words or "").lower(), "ids": [i for i, _ in self.records_in(seen)], "seen": seen, "args": dict(args)}])[-30:]
                self.book_finish(name, args, reply, seen)
                self.learn_list_limit(name, args, seen)
            except Exception as err:
                self.slip("act", err)
        return reply

    # ---------- a step limit in the request's own words ----------
    def read_request(self) -> None:
        super().read_request()
        if ROLE_FILL:
            try:
                self.fill_refuse()
            except Exception as err:
                self.slip("role_fill", err)
        if not STATED_LIMIT_CODE or _count(self.brief.get("steps")) is not None:
            return
        try:
            stated = self.stated_steps()
            if stated is not None and 10 <= stated < self.max_steps:
                self.max_steps = stated
                self.trace.append({"kind": "stated_limit_code", "steps": stated})
        except Exception as err:
            self.slip("stated_limit", err)

    def fill_refuse(self) -> None:
        """ROLE_FILL: the reading named no tool that ends without a purchase (or named the checkout tool with no argument telling the two
        apart): one question on the tools no job took. Without it a decline runs past every review and hold."""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if not buy or (refuse and (refuse != buy or self.brief.get("refuse_args"))):
            return
        taken = {v for k, v in self.roles.items() if k != "refuse"}
        rest = [t["function"] for t in self.tools if t["function"]["name"] not in taken]
        if not rest:
            return
        lines = "\n".join(f"- {fn['name']}: {fn.get('description', '')}" for fn in rest)
        got = self.ask_json(ROLE_ASK + f"\n\nCHECKOUT TOOL: {buy}\n\nOTHER TOOLS:\n" + lines, [LIGHT], CHASE_TOKENS) or {}
        name = got.get("refuse")
        if isinstance(name, str) and name in self.schemas and name not in taken:
            self.roles["refuse"] = name
            self.brief["refuse_args"] = {}
            self.trace.append({"kind": "role_fill", "refuse": name})

    def stated_steps(self) -> int | None:
        """The one figure the request writes right before a word for a store call, unless the reading took that figure
        for a requirement on a product (a part the product has, not a limit on the work)."""
        found = {int(f.replace(",", "")) for f in re.findall(STEP_FIGURE, self.request.lower())}
        if len(found) != 1:
            return None
        figure = found.pop()
        musts = " ".join(m for item in self.items for m in item.musts)
        return None if re.search(rf"\b{figure}\b", musts) else figure

    # ---------- an identical finishing call ----------
    def state_mark(self) -> str:
        """What a repeated finish could have been waiting for: the cart's lines, the notes read, the messages sent."""
        return f"{self.cart_mark()}#{self.version}#{self.msg_used}"

    @staticmethod
    def bare_error(seen: Any) -> bool:
        """A finish reply that is the store's error rather than the shopper's refusal: no list of sentences anywhere at
        its top level (the shopper's lines come as a list)."""
        if isinstance(seen, str):
            return True
        if not isinstance(seen, dict):
            return False
        return not any(isinstance(v, list) and v and all(isinstance(x, str) for x in v) for v in seen.values())

    def book_finish(self, name: str, args: dict[str, Any], reply: dict[str, Any], seen: Any) -> None:
        if not REPEAT_GUARD or reply.get("finished") or name not in {self.roles.get("buy"), self.roles.get("refuse")} - {None}:
            return
        if self.bare_error(seen):
            mark = name + _dump(args)
            self.finish_fails[mark] = self.finish_fails.get(mark, 0) + 1
            self.fail_state = self.state_mark()

    def past_clock(self) -> bool:
        return bool(FINISH_CLOCK) and time.monotonic() - self.began >= FINISH_CLOCK

    def ready_to_ask(self) -> bool:
        return not self.past_clock() and super().ready_to_ask()

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """FINISH_CLOCK: past the limit no model call goes out (every layer treats that as a failed call, the loop ends, the ordinary finish runs).
        Before it, a call's timeout stays inside the episode, and close to it a failed call is not waited on and sent again."""
        if not FINISH_CLOCK:
            return super().call_model(model, messages, max_tokens, tools)
        used = time.monotonic() - self.began
        if used >= FINISH_CLOCK:
            if not self.clock_out:
                self.clock_out = True
                self.trace.append({"kind": "clock_stop", "seconds": round(used, 1), "step": self.step, "calls": self.calls, "cart": self.cart_ids()})
            return None, 400
        _llm.timeout = max(FINISH_FLOOR, min(FINISH_CALL, FINISH_END - used))
        message, status = super().call_model(model, messages, max_tokens, tools)
        if message is None and status != 402 and time.monotonic() - self.began + max(WAITS) >= FINISH_CLOCK:
            return None, 400
        return message, status

    def swap_room_hold(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """One turn takes a listing out of the cart and puts another in, and the steps before the kept finish fit only part of it:
        none of it runs, so the guard on the last steps cannot leave the cart short of the line it was replacing."""
        tool = self.roles.get("cart")
        if not LAST_SWAP or not tool or self.finished or len(calls) < 2 or not (self.roles.get("buy") or self.roles.get("refuse")):
            return []
        room = self.left() - self.tail()
        if room < 1:
            return []
        cart = set(self.cart_ids())
        needed = out_of_cart = into_cart = 0
        for index, call in enumerate(calls, 1):
            fn = call.get("function") or {}
            if fn.get("name") != tool:
                continue
            raw = fn.get("arguments")
            args = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else None)
            named = [v for v in (args or {}).values() if isinstance(v, str) and (v in cart or v in self.rows)]
            needed = index
            out_of_cart += any(v in cart for v in named)
            into_cart += any(v not in cart for v in named)
        if not out_of_cart or not into_cart or needed <= room:
            return []
        self.trace.append({"kind": "swap_room_hold", "room": room, "needed": needed, "cart": sorted(cart)})
        text = (f"Not run (no step used): {room} step(s) are left before the finish and this change takes {needed}, so none of it ran. "
                "Cart now: " + self.cart_text() + " Finish with this cart.")
        return [(call["id"], text) for call in calls]

    def repeat_finish_hold(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """A lone finishing call the store already answered with an error, nothing changed since: held once with the lesson;
        a further attempt goes through, so a finish can still happen."""
        if not REPEAT_GUARD or not self.finish_fails or len(calls) != 1:
            return []
        fn = calls[0].get("function") or {}
        name = fn.get("name")
        if name not in {self.roles.get("buy"), self.roles.get("refuse")} - {None}:
            return []
        raw = fn.get("arguments")
        args = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
        if args is None:
            return []
        faults: list[str] = []
        args = _shape(args, self.schemas.get(name) or {}, "args", faults)
        if self.is_refuse(name, args):
            args = self.quiet(name, args)
        mark = name + _dump(args)
        if self.finish_fails.get(mark) != 1 or self.fail_state != self.state_mark():
            return []
        self.finish_fails[mark] = 2
        self.trace.append({"kind": "repeat_finish_hold", "tool": name, "args": _clip(args, 120)})
        other = next((t for t in (self.roles.get("buy"), self.roles.get("refuse")) if t and t != name), None)
        if FACT_TEXTS:
            return [(calls[0]["id"], "Not run (no step used). This exact finishing call already came back without ending the episode, and nothing has "
                     "changed since, so the store would answer it the same way. Its reply names what it lacks: an argument (a required choice or a "
                     "confirmation token the store handed back) or a change of the cart." + (f" The other finishing call is {other}." if other else ""))]
        return [(calls[0]["id"], "Not run yet (no step used). This exact finishing call already came back without ending the episode, and nothing has "
                 "changed since, so sending it again cannot do better. Read that reply: fill the argument it names (a required choice or a "
                 "confirmation token the store handed back), or change the cart" + (f", or use {other}" if other else "") + "; then finish.")]

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        try:
            held = self.repeat_finish_hold(calls)
        except Exception as err:
            held = []
            self.slip("repeat_hold", err)
        if held:
            return held
        try:
            held = self.swap_room_hold(calls)
        except Exception as err:
            held = []
            self.slip("swap_room_hold", err)
        if held:
            return held
        out = super().run_calls(calls)
        if self.cart_note and out:
            out[-1] = (out[-1][0], out[-1][1] + self.cart_note)
            self.cart_note = ""
        return out

    # ---------- the cart by the store's own reply ----------
    def store_lines(self, seen: Any) -> dict[str, int] | None:
        """The lines a cart reply shows ({listing: units}), read by shape: the first list of records that each carry a
        listing id; a list of records with no units reads as one unit each. None when the reply shows no such list."""
        found: list[dict[str, int] | None] = [None]

        def units(row: dict[str, Any]) -> int:
            counts = [n for k, v in row.items() if k != self.price_key and not isinstance(v, str)
                      for n in [_count(v)] if n is not None]
            return min(counts) if counts else 1

        def walk(node: Any, depth: int) -> None:
            if depth > 3 or found[0] is not None:
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list) and node and all(isinstance(r, dict) for r in node):
                ids = [next((v.strip() for v in r.values() if isinstance(v, str) and v.strip() in self.rows), None) for r in node]
                if all(ids):
                    found[0] = {i: units(r) for i, r in zip(ids, node) if units(r) > 0}

        walk(seen, 0)
        return found[0]

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        super().note_cart(name, args)
        try:
            self.reconcile_book(args)
        except Exception as err:
            self.slip("reconcile", err)

    def reconcile_book(self, args: dict[str, Any]) -> None:
        """A list call whose reply still shows lines the list left out added instead of replacing: the record takes the
        store's own view, and the actor is told, once per call."""
        seen, self.cart_reply = self.cart_reply, None
        listed = [v for v in args.values() if isinstance(v, list)]
        if not CART_RECONCILE or not listed or seen is None:
            return
        shown = self.store_lines(seen)
        if not shown:
            return
        named = {x for v in listed for line in v for x in (line.values() if isinstance(line, dict) else [line]) if isinstance(x, str)}
        extra = sorted(i for i in shown if i not in named)
        if not extra:
            return
        self.cart_lines = dict(shown)
        self.trace.append({"kind": "cart_reconcile", "extra": extra, "shown": dict(shown)})
        self.cart_note = ("\n[Kyo] This cart call added to the cart rather than replacing it: the store's reply still shows " + ", ".join(extra)
                          + f", which the call left out. Kyo's record now follows that reply: {_dump(shown)}. Before checking out, take out any line that should not be there.")

    def take_out(self, extra: list[str], lines: dict[str, int]) -> bool:
        """Lines a code cart left behind in a list cart that adds: asked out in the forms the tool's schema allows, the
        cheapest first: the extra lines at zero units; else an empty list followed by the wanted lines. Each reply is the truth."""
        cart = self.roles.get("cart") or ""
        schema = self.schemas.get(cart) or {}
        props = schema.get("properties") or {}
        key = next((k for k, s in props.items() if isinstance(s, dict) and "array" in _kinds(s) and isinstance((s.get("items") or {}).get("properties"), dict)), None)
        if not key:
            return False
        inner = props[key]["items"]["properties"]
        id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
        count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        if not id_key:
            return False
        tries = [[{key: []}, {key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]]
        if count_key:
            tries.insert(0, [{key: [{id_key: i, count_key: 0} for i in extra]}])
        for calls in tries:
            shaped: list[dict[str, Any]] = []
            for call in calls:
                faults: list[str] = []
                shaped.append(_shape(call, schema, "args", faults))
                if faults:
                    shaped = []
                    break
            if not shaped or self.left() < len(shaped) + 1:
                continue
            seen = None
            for index, args in enumerate(shaped):
                reply = self.act(cart, args)
                if reply is None or self.finished:
                    return False
                seen = reply.get("observation")
                self.note_cart(cart, args)
                self.observe(cart, seen)
                # (an empty list that cleared nothing: sending the lines again would only add them twice)
                if index < len(shaped) - 1 and self.store_lines(seen):
                    break
            view = self.store_lines(seen)
            self.trace.append({"kind": "cart_takeout", "extra": extra, "calls": len(shaped), "shown": dict(view or {})})
            if view is not None:
                self.cart_lines = dict(view)
                if not [i for i in view if i not in lines]:
                    self.cart_note = ""
                    return True
        return False

    # ---------- a list argument's cap ----------
    def learn_list_limit(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """A store may cap one list argument without saying so. Per tool and argument: the longest list that returned
        records is the floor. A list of known listings, longer than the floor, that returned no record at all sets a cap of
        half its length (never below the floor) in the tool's live schema, so a longer list is refused before it is sent.
        The side-by-side view's own record (compare_bad) already answers a reply that names none of the listings, so
        that case is left to it."""
        if not LIST_LIMITS or name in {self.roles.get(r) for r in ("cart", "buy", "refuse", "ask")}:
            return
        spec = (self.schemas.get(name) or {}).get("properties")
        if not isinstance(spec, dict):
            return
        lists = [(k, v) for k, v in args.items() if isinstance(v, list) and isinstance(spec.get(k), dict)]
        if len(lists) != 1:
            return
        key, ids = lists[0]
        floor = self.list_floor.get((name, key), 0)
        if self.records_in(seen):
            self.list_floor[(name, key)] = max(floor, len(ids))
            return
        if len(ids) < 2 or len(ids) <= floor or not all(isinstance(i, str) and i.strip() in self.rows for i in ids):
            return
        if name == self.roles.get("compare") and self.compare_target() == key and not any(str(i) in _dump(seen) for i in ids):
            return
        limit = max(floor, len(ids) // 2, 1)
        old = spec[key].get("maxItems")
        if isinstance(old, int) and old <= limit:
            return
        spec[key]["maxItems"] = limit
        self.trace.append({"kind": "list_limit", "tool": name, "arg": key, "refused": len(ids), "limit": limit})

    # ---------- what large requests carry and spend ----------
    def show(self, observation: Any) -> str:
        if ROW_CHARS and AUDIT_ITEMS and len(self.items) >= AUDIT_ITEMS and len(self.records_in(observation)) >= 2:
            text = _dump(_clip(observation, ROW_CHARS))
            return text[:OBS_LIMIT] + "...(cut)" if len(text) > OBS_LIMIT else text
        return super().show(observation)

    def audit(self) -> str:
        if AUDIT_ITEMS and len(self.items) >= AUDIT_ITEMS:
            if not self.audit_skipped:
                self.audit_skipped = True
                self.trace.append({"kind": "audit_skip", "items": len(self.items), "cost": round(self.cost, 4), "left": self.left()})
            return ""
        return super().audit()

    # ---------- a message that would otherwise go unsent: the next open property ----------
    def own_pages(self, number: int) -> list[dict[str, Any]]:
        """Opened records of this item's own store categories, matched the way the parent's key table matches them; on a
        one-item request every opened record."""
        records = [self.full[ident] for ident in self.seen_order if self.full.get(ident)]
        if len(self.items) == 1:
            return records
        item = self.items[number - 1]
        shelf, name = _words(item.shelf or ""), _words(item.name)
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        cat_key = self.category_key()
        mine = []
        for record in records:
            label = str(record.get(cat_key) or "") if cat_key else ""
            if label and (_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long):
                mine.append(record)
        return mine

    def asked_about(self, number: int) -> set[str]:
        """Properties the interviewer already asked about this item: an answer without a value means the property was the
        wrong one, so it is not asked again."""
        name = self.items[number - 1].name
        asked = set()
        for event in self.trace:
            if event.get("kind") == "ask":
                found = re.match(r"What (.+?) would you like for the (.+?)(?:, for example .*)?\?$", str(event.get("q") or ""))
                if found and found.group(2) == name:
                    if STACK_ASK:
                        asked.update(_words(part) for part in re.split(r",\s*|\s+or\s+", found.group(1)) if part.strip())
                    else:
                        asked.add(_words(found.group(1)))
        return asked

    def fill_question(self) -> tuple[int, str, str] | None:
        """The next property to ask: the parent's planned questions in its own order, on an item with no need heard yet,
        that no question has asked; else the property most of that item's own pages carry."""
        unheard = [n for n in range(1, len(self.items) + 1) if not self.items[n - 1].told]

        def closed(number: int) -> set[str]:
            item = self.items[number - 1]
            done = self.asked_about(number)
            for raw in item.musts + item.told + item.dropped + item.free:
                done.add(_words(str(raw).split(":")[0]))
            return {c for c in done if c}

        for event in self.trace:
            if event.get("kind") != "plan":
                continue
            for pick in _items(event.get("ask")):
                if not isinstance(pick, dict):
                    continue
                number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
                label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
                if label and len(label.split()) <= 5 and number in unheard and _words(label) not in closed(number):
                    return number, label, "plan"
        for number in unheard:
            counts: dict[str, int] = {}
            for record in self.own_pages(number):
                for key in (_fact_block(record) or {}):
                    name = _words(key)
                    if name and len(name.split()) <= 5:
                        counts[name] = counts.get(name, 0) + 1
            shut = closed(number)
            for name in sorted(counts, key=lambda k: (-counts[k], k)):
                if name not in shut:
                    return number, name, "pages"
        return None

    def fill_asks(self, spare, where: str, stop_on_need: bool) -> tuple[list[tuple[int, str, str]], list[str]]:
        """One question per message while `spare()` says a message would otherwise go unsent, some item has no need heard
        yet, and the steps to act on an answer remain."""
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while (not self.finished and spare() > 0 and self.left() - self.tail() > 6 + 2 * len(self.items)
               and any(not item.told for item in self.items)):
            pick = self.fill_question()
            if not pick:
                break
            number, label, why = pick
            mark = {"kind": "fill_ask", "at": where, "item": number, "label": label, "why": why}
            self.trace.append(mark)
            batch = self.send([(number, label, self.question(number, label))])
            if not batch:
                if self.trace and self.trace[-1] is mark:
                    self.trace.pop()
                break
            self.fills += 1
            before = len(told)
            self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
            sent += batch
            if stop_on_need and len(told) > before:
                break
        return sent, told

    def end_need(self) -> int:
        """Messages the request's own steps at an ending without a purchase may still take."""
        talk = [p["messages"] for p in self.process if p["when"] == "end" and p["messages"]]
        return min(max(talk), self.msg_limit) if talk else 0

    def fill_gate(self, stage: str) -> tuple[bool, str]:
        """At the checkout, after the request's own steps there: a message left over goes to the next open property. A new
        requirement holds the order so the cart can follow it."""
        go, said = super().gate(stage)
        if not FILL_ASK or not go or stage != "order" or self.finished or not self.roles.get("ask"):
            return go, said
        try:
            sent, told = self.fill_asks(lambda: self.msg_limit - self.msg_used - self.end_need(), "checkout", True)
        except Exception as err:
            self.slip("fill_checkout", err)
            return go, said
        if not sent:
            return go, said
        lines = "\n".join(f"Q: {q}\nA: {a}" for _, q, a in sent)
        head = (said + "\n") if said else ""
        if told:
            self.version += 1
            return False, head + "A message the request had left unsent went to one more question:\n" + lines + "\nNew firm requirements: " + "; ".join(told) + (". The cart was built before them, so the checkout was not run." if FACT_TEXTS else ". Fix the cart for them, then check out again.")
        return go, head + "A message the request had left unsent went to one more question:\n" + lines + "\nNo new requirement."

    # ---------- one property in the first message, then several properties of one item per message ----------
    def stacking(self) -> bool:
        """The switch's questions run when the request leaves at least two messages for them."""
        return STACK_ASK and bool(self.roles.get("ask")) and self.msgs_free() >= 2

    @staticmethod
    def same_property(a: str, b: str) -> bool:
        """Two names of one property: the same words, or one name's words all among the other's."""
        sa, sb = _stem(a), _stem(b)
        return _words(a) == _words(b) or bool(sa and sb and (sa <= sb or sb <= sa))

    def table_name(self, keys: list[str], label: str) -> str | None:
        """The listing key a planned label names: the same words, else a key whose words all lie in the label's or hold all of them."""
        for key in keys:
            if _words(key) == _words(label):
                return key
        return next((key for key in keys if self.same_property(key, label)), None)

    def stack_keys(self, number: int) -> list[str]:
        """This item's properties: its opened listings' keys in plain words, the keys most of those listings carry first, then those
        whose values differ most. Long texts, variant lists, a different code or long text on every listing, a key every listing
        carrying it gives one same value (kept with ONE_VALUE_KEEP: a shopper can hold a preference on it all the same), and a key
        whose value a settled or heard requirement already states are left out."""
        per: dict[str, dict[int, str]] = {}
        flags: set[str] = set()
        for index, record in enumerate(self.own_pages(number)):
            for key, value in (_fact_block(record) or {}).items():
                name, text = _words(key), str(value).strip().lower()
                if name and text and len(name.split()) <= 5 and index not in per.setdefault(name, {}):
                    per[name][index] = text
                    if isinstance(value, bool):
                        flags.add(name)
        item = self.items[number - 1]
        stated = [_stem(str(m).split(":", 1)[-1]) for m in item.musts + item.told]
        kept: list[str] = []
        for name, by in per.items():
            values = list(by.values())
            if not values:
                continue
            mean = sum(len(v) for v in values) / len(values)
            if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                continue
            if not ONE_VALUE_KEEP and len(values) >= 2 and len(set(values)) == 1:
                continue
            unique = len(values) >= 3 and len(set(values)) == len(values)
            if unique and ((len(values) >= 4 and mean > 20) or all(re.fullmatch(r"(?=\S*\d)(?=\S*[a-z])\S+", v) for v in values)):
                continue
            if name not in flags and any(len(v) >= 3 and any(s and _stem(v) <= s for s in stated) for v in set(values)):
                continue
            if any(name.startswith(other + " ") and by == per[other] for other in kept):
                continue
            kept.append(name)
        order = list(per)
        return sorted(kept, key=lambda k: (-len(per[k]), -len(set(per[k].values())), order.index(k))) or self.shape_names(number)

    def shape_names(self, number: int) -> list[str]:
        """SHAPE_KEYS: the readers hold no full record although pages were opened: the property names a light model reads off those pages."""
        if not SHAPE_KEYS or self.full or not self.raw_pages:
            return []
        if number not in self.shape_named:
            prompt = SHAPE_ASK + f"\n\nITEM: {self.items[number - 1].name}\n\nPAGES:\n" + "\n".join(self.raw_pages)
            got = self.ask_json(prompt, [LIGHT], CHASE_TOKENS) or {}
            names = [_words(str(n)) for n in _items(got.get("names")) if isinstance(n, str)]
            self.shape_named[number] = [n for n in dict.fromkeys(names) if n and len(n.split()) <= 5][:12]
            self.trace.append({"kind": "shape_names", "item": number, "names": self.shape_named[number]})
        return list(self.shape_named[number])

    def stack_open(self, number: int) -> list[str]:
        """The item's properties no question has closed and no requirement settles: a requirement on a property settles its keys
        of the same or wider words (one on "rim finish" settles "finish", one on "colour" leaves "lid colour" open)."""
        item = self.items[number - 1]
        done = self.stack_done.get(number, set())
        settled = [_stem(str(raw).split(":")[0]) for raw in item.musts + item.told + item.dropped]
        return self.skipped_last([k for k in self.stack_keys(number) if k not in done and not any(s and _stem(k) and _stem(k) <= s for s in settled)])

    def stack_first(self, ordered: list[tuple[int, str]]) -> tuple[int, str, str] | None:
        """The first message's one property: the planner's first when the request asks to be asked about something; else its
        likeliest that the item's listings carry, in the key's own words; else the first item's first key."""
        if ordered and [x for x in _items(self.brief.get("ask_first")) if str(x).strip()]:
            return ordered[0][0], ordered[0][1], "ask_first"
        for number, label in ordered:
            key = self.table_name(self.stack_open(number), label)
            if key:
                return number, key, "plan"
        for number in range(1, len(self.items) + 1):
            keys = self.stack_open(number)
            if keys:
                return number, keys[0], "keys"
        return (ordered[0][0], ordered[0][1], "plan_off_table") if ordered else None

    def stack_item(self, asked: dict[int, int]) -> int | None:
        """The item the next message is about: every item once before any item twice, then items with no need heard, then the
        fewest messages, in the request's order."""
        live = [n for n in range(1, len(self.items) + 1) if self.stack_open(n)]
        return min(live, key=lambda n: (asked.get(n, 0) > 0, bool(self.items[n - 1].told), asked.get(n, 0), n)) if live else None

    def stack_names(self, number: int, ordered: list[tuple[int, str]], leads: set[tuple[int, str]]) -> list[str]:
        """One item's next property list: the planner's next pick its listings carry first (else the item's first key), then the
        properties an earlier list named past the answered ones, then the item's keys in order, up to STACK_SIZE."""
        keys = self.stack_open(number)
        carry = [c for c in self.stack_carry.get(number, []) if c in keys]
        # (REFUSED_FIRST: a refused list answered nothing, so its names are as open as before it and stay first in line)
        refused = REFUSED_FIRST and self.stack_off
        lead = None
        for pick in ordered:
            if pick[0] == number and pick not in leads:
                found = self.table_name(keys, pick[1])
                if found and (refused or found not in carry):
                    leads.add(pick)
                    lead = found
                    break
        if refused:
            rest = [c for c in carry if c != lead] + [k for k in keys if k not in carry and k != lead]
            return ([lead] if lead else []) + self.skipped_last(rest)[: max(0, STACK_SIZE - 1)]
        fresh = [k for k in keys if k not in carry and k != lead]
        if lead is None and fresh:
            lead = fresh.pop(0)
        return ([lead] if lead else []) + self.skipped_last(carry + fresh)[: max(0, STACK_SIZE - 1)]

    def late_lists(self, count: int) -> bool:
        """LATE_LISTS: this message still names one property, because the lists wait for the last half of the `count` messages
        the questions started with."""
        if FIRST_LIST and self.stack_listed:
            return False   # (the shopper has answered a list: nothing is gained by holding the next ones back)
        return bool(LATE_LISTS) and count < LATE_LISTS and self.msgs_free() > count // 2

    def stack_question(self, number: int, names: list[str]) -> str:
        if len(names) == 1:
            return self.question(number, names[0])
        return f"What {', '.join(names[:-1])} or {names[-1]} would you like for the {self.items[number - 1].name}?"

    def stack_heard(self, number: int, names: list[str], before: int) -> None:
        """A list's answer: the properties up to the one whose need it revealed are closed (a shopper answers the first one it
        cares about), else the first two (the places a shopper answers most surely), with STACK_SETTLE the first one only; the rest
        go first into the item's next list."""
        new = [str(x).split(":")[0] for x in self.items[number - 1].told[before:]]
        hit = next((i for i, name in enumerate(names) if any(self.same_property(name, p) for p in new)), None)
        cut = hit + 1 if hit is not None else min(1 if STACK_SETTLE else 2, len(names))
        self.stack_done.setdefault(number, set()).update(names[:cut])
        self.stack_carry[number] = [n for n in names[cut:] if not any(self.same_property(n, p) for p in new)]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        if not HEAR_POLARITY:
            return super().hear(sent, models)
        prompt = HEAR + POLARITY + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 1500) or {}

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        super().take_heard(heard, told)
        self.trace.append({"kind": "heard", "needs": [str(x.get("need"))[:80] for x in _items(heard.get("needs")) if isinstance(x, dict)][:6]})

    def hear_one(self, sent: list[tuple[int, str, str]]) -> dict[str, Any]:
        prompt = STACK_HEAR + (POLARITY if HEAR_POLARITY else "") + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}

    def stack_plan(self, count: int) -> list[tuple[int, str]]:
        """The parent's question plan, asked with the parent's own prompt."""
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
        prompt = (STACK_PLAN if PLAN_SKIP else PLAN).replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        if PLAN_SKIP:
            # the keys the planner marked, by their numbers in the list it was shown; a key it also chose to ask is not one of them
            names = [name for _, name, _, _ in table]
            marked = {names[i - 1] for i in (_count(x) for x in _items(plan.get("skip"))) if i and 1 <= i <= len(names)}
            self.stack_skip = {name for name in marked if not any(self.same_property(name, label) for _, label in ordered)}
            if self.stack_skip:
                self.trace.append({"kind": "plan_skip", "keys": sorted(self.stack_skip)[:12]})
        return ordered

    def skipped_last(self, names: list[str]) -> list[str]:
        """The names in their order, those the planner marked as not about the product moved to the end."""
        if not PLAN_SKIP or not self.stack_skip:
            return names
        return [n for n in names if n not in self.stack_skip] + [n for n in names if n in self.stack_skip]

    def stack_interview(self) -> str:
        """The parent's interview with the switch's shape: the same plan, one property in the first message, then up to
        STACK_SIZE properties of one item per message, and one property per message again after an answer to a list that says
        it asked too much or asks for it again."""
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        ordered = self.stack_plan(count)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        asked: dict[int, int] = {}
        leads: set[tuple[int, str]] = set()
        retry: tuple[int, str] | None = None
        for turn in range(count):
            if self.finished or self.msgs_free() <= 0:
                break
            samples: list[str] = []
            try:
                if turn == 0:
                    first = self.stack_first(ordered)
                    if not first:
                        break
                    number, names, why = first[0], [first[1]], first[2]
                    if why == "plan":
                        leads.update(p for p in ordered if p[0] == number and self.table_name([first[1]], p[1]))
                    if FIRST_LIST and len(self.items) >= FIRST_LIST and why in ("plan", "keys"):
                        more = self.skipped_last([k for k in self.stack_open(number) if k != names[0]])[: max(0, STACK_SIZE - 1)]
                        if more:
                            names, why = names + more, "list_first"
                elif self.stack_off:
                    if retry:
                        (number, label), retry, why = retry, None, "single_again"
                        samples = self.samples(number, label)
                    else:
                        number = self.stack_item(asked)
                        label = self.stack_names(number, ordered, leads)[0] if number else ""
                        why = "single"
                    names = [label] if label else []
                else:
                    number = self.stack_item(asked)
                    names = self.stack_names(number, ordered, leads) if number else []
                    why = "list"
                    if names and self.late_lists(count):
                        names, why = names[:1], "single_early"
            except Exception as err:
                self.slip("stack_pick", err)
                break
            if not names:
                break
            text = self.question(number, names[0], samples if len(samples) >= 2 else None) if len(names) == 1 else self.stack_question(number, names)
            mark = {"kind": "stack_ask", "item": number, "names": names, "why": why}
            self.trace.append(mark)
            batch = self.send([(number, names[0], text)])
            if not batch:
                if self.trace and self.trace[-1] is mark:
                    self.trace.pop()
                break
            sent += batch
            asked[number] = asked.get(number, 0) + 1
            before = len(self.items[number - 1].told)
            heard = self.hear_one(batch)
            self.take_heard(heard, told)
            try:
                if len(names) == 1:
                    self.stack_done.setdefault(number, set()).add(names[0])
                elif any(_count(x) == 1 for x in _items(heard.get("crowded"))):
                    # a list refused for its length answered nothing: its first property goes again on its own
                    self.stack_off = True
                    retry = (number, names[0])
                    self.stack_carry[number] = names[1:]
                    self.trace.append({"kind": "stack_off", "item": number, "why": "crowded", "reply": batch[0][2][:120]})
                else:
                    self.stack_listed = True
                    self.stack_heard(number, names, before)
                    if not LISTS_AFTER_UNCLEAR and any(_count(x) == 1 for x in _items(heard.get("unclear"))):
                        # an answer asking for a list again: one property per message from here, nothing reworded
                        self.stack_off = True
                        self.trace.append({"kind": "stack_off", "item": number, "why": "unclear", "reply": batch[0][2][:120]})
            except Exception as err:
                self.slip("stack_heard", err)
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

    # ---------- buying after the questions: value searches, their hits opened, the pick on the records, no decline past a fit ----------
    @staticmethod
    def value_phrase(need: str) -> str:
        """A requirement's figure with the unit written right after it, as the requirement writes them (its words around them left out)."""
        text = need.partition(":")[2].lower()
        if SEARCH_FIX and re.search(NOT_WORDS, text):
            return ""
        found = re.search((r"(?<![a-z0-9.])" if SEARCH_FIX else "") + r"\d+(?:[.,]\d+)?(?:(?:\s*|-)(?:[a-z]+|[\"']))?", text)
        return found.group(0).strip() if found else ""

    def buy_searches(self) -> str:
        """Right after the questions: each heard requirement holding a figure is searched by its value as written with the item's
        name, and an item name holding figures is searched without them; with WORD_VALUE_SEARCH, a value in words follows the same
        way, after those and within the same BUY_SEARCHES. The rows join the record and the agent sees them; when the search tool
        takes more than its words, the agent is told those searches instead."""
        wanted: list[str] = []
        worded: list[str] = []
        pointed = {need for _, need in self.refs}
        for item in self.items:
            name = " ".join(_words(item.name).split()[:4])
            if BUY_SEARCH:
                wanted += [f"{self.value_phrase(need)} {name}" for need in item.told if self.value_phrase(need) and need not in item.dropped]
                bare = " ".join(w for w in _words(item.name).split() if not re.search(r"\d", w))
                if bare and bare != _words(item.name):
                    wanted.append(bare)
            if WORD_VALUE_SEARCH:
                # a value in words, as heard (its first choice when it gives several); not a figure, a negation or a value read off another listing
                for need in item.told:
                    value = re.split(r"\bor\b|[,/;]", need.partition(":")[2])[0].strip()
                    if need in item.dropped or need in pointed or not value or re.search(r"\d", value) or re.search(NOT_WORDS, value.lower()) or len(value.split()) > 4:
                        continue
                    worded.append(f"{value} {name}")
        wanted += worded
        wanted = [w for w in dict.fromkeys(wanted) if w.lower() not in self.buy_searched][: max(0, BUY_SEARCHES - len(self.buy_searched))]
        tool = self.roles.get("search")
        target = self.single_target(tool)
        if not wanted:
            return ""
        if not target and ITEM_SWEEP and self.search_args(wanted[0], None):
            lines = []   # (the tool takes more than its words: the agent's own latest search carries them)
            for words in wanted:
                found = self.code_search(words, None, 12 + 2 * len(self.items))
                if found is None:
                    break
                self.trace.append({"kind": "buy_search", "words": words, "hits": len(found)})
                lines.append(f"'{words}': " + (", ".join(found[:12]) if found else "no listings"))
            return ("Kyo searched the new requirements' values as written (one step each); the rows are in the record:\n" + "\n".join(lines)) if lines else ""
        if not target:
            self.buy_searched.update(w.lower() for w in wanted)
            self.trace.append({"kind": "buy_search", "told": wanted})
            return "Search next for the new requirements' values as written, with the item's name: " + "; ".join(f"'{w}'" for w in wanted) + ". Open the hits that could fit."
        lines: list[str] = []
        for words in wanted:
            if self.finished or self.left() - self.tail() < 12 + 2 * len(self.items):
                break
            self.buy_searched.add(words.lower())
            faults: list[str] = []
            args = _shape({target: words}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            hits = [i for i, _ in self.records_in(seen)]
            self.trace.append({"kind": "buy_search", "words": words, "hits": len(hits)})
            lines.append(f"'{words}': " + (self.show(seen)[:900] if hits else "no listings"))
        return ("Kyo searched the new requirements' values as written (one step each):\n" + "\n".join(lines)) if lines else ""

    def value_hits(self) -> list[str]:
        """Unopened, buyable hits of the searches that matched 24 listings or fewer and whose words held a heard requirement's value
        (its figure, or all of its words)."""
        values: list[tuple[str, frozenset[str]]] = []
        for item in self.items:
            for need in item.told:
                wanted = need.partition(":")[2].strip().lower()
                if need in item.dropped or not wanted or re.search(NOT_WORDS, wanted):
                    continue
                figure = re.search(r"\d+(?:[.,]\d+)?", wanted)
                values.append((figure.group(0) if figure else "", frozenset(_plain(wanted)) if not figure else frozenset()))
        hits: list[str] = []
        for entry in self.buy_log:
            words = entry["words"]
            if not any((figure and re.search(rf"(?<![\d.]){re.escape(figure)}(?![\d.])", words)) or (stems and stems <= _plain(words)) for figure, stems in values):
                continue
            plan = self.paging(entry["tool"], entry["seen"])
            if (plan[0] if plan else len(entry["ids"])) > 24:
                continue
            hits += [i for i in entry["ids"] if i not in hits and i in self.rows and i not in self.full and not self.sold_out(i)]
        return hits

    def open_value_hits(self, keep: int) -> list[str]:
        """BUY_OPEN: read the attributes of those hits, up to BUY_OPENS an episode, keeping `keep` steps."""
        ids = self.value_hits()[: max(0, BUY_OPENS - self.buy_opened)] if BUY_OPEN else []
        if not ids:
            return []
        texts = self.fill_facts(ids, keep)
        self.buy_opened += len(texts)
        self.trace.append({"kind": "buy_open", "ids": ids[: len(texts)]})
        return texts

    def loose_need(self, need: str, record: dict[str, Any], pointed: set[str]) -> int:
        """read_need, and where it reads nothing either way, the value counts as stated when a key of a related name (sharing a word's
        first four letters) or the listing's own short text fields (its title) state it as written. A contradiction stays one."""
        got = self.read_need(need, record, pointed)
        prop, _, wanted = need.partition(":")
        wanted = wanted.strip().lower()
        if got != 0 or need in pointed or not wanted or re.search(NOT_WORDS, wanted):
            return got
        heads = {w[:4] for w in _words(prop).split() if len(w) >= 4}
        texts = [str(v) for k, v in (_fact_block(record) or {}).items() if heads & {w[:4] for w in _words(str(k)).split() if len(w) >= 4}]
        texts.append(" ".join(str(v) for v in record.values() if isinstance(v, str) and len(v) <= 200))
        options = [o for o in (_plain(part) for part in re.split(r"\bor\b|[,/;]", wanted)) if o]
        for text in texts:
            if (self.read_figure(wanted, [], text) == 1) if re.search(r"\d", wanted) else any(o <= _plain(text) for o in options):
                return 1
        return got

    def pick_value(self, ident: str) -> float | None:
        """The listing's own value for the pick: its price for cheapest; for most or least, the figure under the key naming the measure."""
        if self.brief.get("pick") == "cheapest":
            return self.price_now(ident)
        measure = _stem(str(self.brief.get("pick_measure") or ""))
        record = self.full.get(ident) or {}
        for key, value in (_fact_block(record) or {}).items():
            if _names(measure, _stem(str(key))):
                pairs = _figure_pairs(str(value))
                if pairs:
                    return pairs[0][0]
        return None

    def buy_best(self, ids: list[str]) -> str | None:
        """Among listings, the best by the pick's own value (None when the pick is any or no value reads)."""
        pick = self.brief.get("pick")
        if pick not in ("cheapest", "most", "least"):
            return None
        valued = [(v, i) for i in ids for v in [self.pick_value(i)] if v is not None]
        if not valued:
            return None
        return (max(valued) if pick == "most" else min(valued))[1]

    def buy_hold(self) -> str:
        """At the checkout, per item: an opened listing that states every requirement where the cart's line does not (BUY_OPEN),
        or that beats a line stating them all on a cheapest, most or least pick by the records' own values (BUY_PICK), holds the
        checkout once for the cart as it is, twice an episode at most."""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.buy_held or len(self.buy_held) >= 2:
            return ""
        pointed = {need for _, need in self.refs}
        cart = self.cart_ids()
        whole = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        failed = self.ruled()[1] if PICK_CHECKED else set()
        notes: list[str] = []
        for number, item in enumerate(self.items, 1):
            needs = [need for n, need in self.needs_now() if n == number]
            line = next((i for i in cart if self.of_item(i, item)), None)
            if not needs or not line:
                continue
            fits = [i for i, record in self.full.items() if i not in cart and i not in failed and self.of_item(i, item) and self.in_stock_now(i)
                    and (whole is None or (self.price_now(i) is not None and self.price_now(i) * item.qty <= whole))
                    and all(self.loose_need(need, record, pointed) == 1 for need in needs)]
            if not fits:
                continue
            marks = [self.loose_need(need, self.full.get(line) or self.rows.get(line) or {}, pointed) for need in needs]
            if BUY_OPEN and any(m < 1 for m in marks) and not (HOLD_GUARD and line in self.ruled()[0]):
                best = self.buy_best(fits) or fits[0]
                notes.append(f"{best} states every requirement of item {number}; the cart's {line} does not state " + "; ".join(n for n, m in zip(needs, marks) if m < 1))
            elif BUY_PICK and all(m == 1 for m in marks):
                best = self.buy_best(fits + [line])
                if best and best != line and self.pick_value(best) != self.pick_value(line):
                    notes.append(f"{best} states every requirement of item {number} and its own {self.brief.get('pick_measure') or 'price'} beats the cart's {line} for a {self.brief.get('pick')} pick "
                                 f"({_dump(self.pick_value(best))} against {_dump(self.pick_value(line))})")
        if not notes:
            return ""
        self.buy_held.add(mark)
        self.trace.append({"kind": "buy_hold", "notes": notes})
        if FACT_TEXTS:
            return ("Kyo compared the opened listings with the cart: " + " | ".join(notes) + ". The checkout was not run (no step used)"
                    + ("." if FACT_PLAIN else "; this comparison holds it once, and the same checkout sent again runs."))
        return ("Kyo compared the opened listings with the cart: " + " | ".join(notes) + ". Put the better listing in the cart instead, at the requested quantity, "
                "or check out again if the cart's line is right.")

    def cheaper_rows(self) -> list[str]:
        texts = super().cheaper_rows()
        try:
            return texts + self.open_value_hits(self.tail() + 4)
        except Exception as err:
            self.slip("buy_open", err)
            return texts

    def supported(self, ident: str) -> int | None:
        """The number of the item this read listing fully answers: of its kind, buyable within the budget, stating every heard
        requirement and contradicting none of the request's. None when there is none or the item has no heard requirement."""
        record = self.full.get(ident)
        if not record or not self.in_stock_now(ident):
            return None
        pointed = {need for _, need in self.refs}
        read = self.loose_need if BUY_KEEP else self.read_need
        for number, item in enumerate(self.items, 1):
            heard = self.heard_of(item)
            if not heard or not self.of_item(ident, item):
                continue
            price, room = self.price_now(ident), self.pool_room(item)
            if price is None or (room is not None and price * item.qty > room):
                continue
            if all(read(n, record, pointed) == 1 for n in heard) and not any(self.read_need(m, record, pointed) == -1 for m in item.musts if m not in item.dropped):
                return number
        return None

    def states_all(self, ident: str, item: Item) -> bool:
        """STRICT_OVERRIDE: this listing's record states every requirement the item has, written or heard (read as the closest
        order reads them). A listing never read in full proves nothing; an item with no requirement needs no proof."""
        musts = [m for m in item.musts + item.told if m not in item.dropped]
        if not musts:
            return True
        record = self.full.get(ident)
        if not record:
            return False
        pointed = {need for _, need in self.refs}
        read = self.loose_need if BUY_KEEP else self.read_need
        return all(read(m, record, pointed) == 1 for m in musts)

    def cart_states_heard(self, strict: bool = False) -> bool:
        """The cart is complete (one line of each item's kind, none sold out, nothing else) and each line of an item with heard
        requirements states them all and contradicts none of the request's; at least one item has a heard requirement. With
        `strict`, every line also states every requirement of its item."""
        cart = self.cart_ids()
        if len(cart) != len(self.items):
            return False
        some = False
        for number, item in enumerate(self.items, 1):
            line = next((i for i in cart if self.of_item(i, item)), None)
            if not line or self.sold_out(line):
                return False
            if strict and not self.states_all(line, item):
                return False
            if self.heard_of(item):
                some = True
                if self.supported(line) != number:
                    return False
        return some

    def rearm(self, which: str) -> None:
        """REARM_ORDER: the store refused an order Kyo sent in place of an ending without a purchase (it had a change to
        announce), so that ending was not the last one. The rule keeps one more turn for a later ending."""
        if not REARM_ORDER or which in self.rearmed:
            return
        self.rearmed.add(which)
        if which == "cart":
            self.cart_ordered = False
        else:
            self.closest_sent = False
        self.trace.append({"kind": "order_rearmed", "which": which, "step": self.step})

    def order_cart(self) -> str:
        """CART_OVER_DECLINE: the checkout of a complete cart whose lines state every heard requirement, in place of an ending
        without a purchase; the request's own steps at the order run first. Returns what happened for the agent."""
        buy = self.roles.get("buy")
        if self.cart_ordered or not buy or self.left() < 1 or not self.cart_states_heard(STRICT_OVERRIDE):
            return ""
        faults: list[str] = []
        order = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        self.cart_ordered = True
        said: list[str] = []
        if self.left() >= 2:
            go, spoken = self.gate("order")
            if spoken:
                said.append(spoken)
            if not go:
                return "Not run (Kyo acted instead): the cart is complete and its lines state every requirement the shopper gave. " + " ".join(said)
        if self.finished or self.left() < 1:
            return ""
        reply = self.act(buy, order)
        self.trace.append({"kind": "cart_ordered", "cart": self.cart_ids(), "reply": _clip(reply, 200)})
        if reply is None:
            return "Not run (Kyo acted instead): the checkout call failed, so you may send it again."
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
            self.rearm("cart")
        self.observe(buy, reply.get("observation"))
        said.append(self.show(reply.get("observation")))
        return "Not run (Kyo acted instead): the cart is complete and its lines state every requirement the shopper gave, so Kyo checked out. Store reply: " + " | then: ".join(said)

    def check_swap(self) -> str:
        """At a checkout sent again with the cart the independent check found wrong: a listing that check named, of the same item,
        read, buyable within the budget, stating every heard requirement and contradicting none, goes into the cart in the line's
        place. Once per item; returns what was done for the agent."""
        cart_tool = self.roles.get("cart")
        cart = self.cart_ids()
        last = next((x for x in reversed(self.trace) if x.get("kind") == "check" and sorted(x.get("cart") or []) == sorted(cart)), None)
        got = last.get("got") if last else None
        if not cart_tool or not isinstance(got, dict) or got.get("ok") is not False:
            return ""
        pointed = {need for _, need in self.refs}
        read = self.loose_need if BUY_KEEP else self.read_need
        for better in [str(b) for b in _items(got.get("better"))]:
            record = self.full.get(better)
            if not record or better in cart or not self.in_stock_now(better):
                continue
            for number, item in enumerate(self.items, 1):
                line = next((i for i in cart if self.of_item(i, item)), None)
                heard = self.heard_of(item)
                if number in self.swapped or not line or not heard or not self.of_item(better, item):
                    continue
                price, room = self.price_now(better), self.pool_room(item)
                if price is None or (room is not None and price * item.qty > room):
                    continue
                if not all(read(need, record, pointed) == 1 for need in heard) or any(self.read_need(m, record, pointed) == -1 for m in item.musts if m not in item.dropped):
                    continue
                if SWAP_AT_CHECK and self.supported(line) == number and self.buy_best([better, line]) != better:
                    continue   # (both state everything: the line stays unless the pick rule's own value favours the named listing)
                lines = {i: self.cart_lines.get(i, 1) for i in cart if i != line}
                lines[better] = self.cart_lines.get(line, item.qty)
                calls = self.cart_calls(lines, [line])
                if not calls or len(calls) + 2 > self.left():
                    continue
                shaped: list[dict[str, Any]] = []
                for call in calls:
                    faults: list[str] = []
                    shaped.append(_shape(call, self.schemas.get(cart_tool), "args", faults))
                    if faults:
                        shaped = []
                        break
                if not shaped:
                    continue
                self.swapped.add(number)
                self.trace.append({"kind": "check_swap", "item": number, "out": line, "in": better})
                for args in shaped:
                    reply = self.act(cart_tool, args)
                    if reply is None or self.finished:
                        return ""
                    self.note_cart(cart_tool, args)
                    self.observe(cart_tool, reply.get("observation"))
                extra = [i for i in self.cart_lines if i not in lines]
                if extra:
                    self.take_out(extra, lines)
                return (f"Kyo replaced {line} with {better} in the cart: the review named it, and its record states every requirement the shopper gave. "
                        + ("The checkout was not run." if FACT_TEXTS else "Check the cart and check out."))
        return ""

    def gate(self, stage: str) -> tuple[bool, str]:
        go, said = self.fill_gate(stage)
        if go and stage == "order" and CHECK_SWAP and not self.finished:
            try:
                swapped = self.check_swap()
            except Exception as err:
                self.slip("check_swap", err)
                swapped = ""
            if swapped:
                return False, ((said + "\n") if said else "") + swapped
        if not go or stage != "order" or not (BUY_OPEN or BUY_PICK or MULTI_NEED_HOLD) or self.finished:
            return go, said
        hold = ""
        if BUY_OPEN or BUY_PICK:
            try:
                hold = self.buy_hold()
            except Exception as err:
                self.slip("buy_hold", err)
                hold = ""
        if MULTI_NEED_HOLD:
            if hold:
                # (this cart was held for a better listing already)
                self.multi_covered.add(self.cart_mark() + f"#{self.version}")
            else:
                try:
                    hold = self.multi_need_hold()
                except Exception as err:
                    self.slip("multi_hold", err)
                    hold = ""
        return (False, ((said + "\n") if said else "") + hold) if hold else (go, said)

    # ---------- two or more heard requirements on one item: pool the hits before a decline, hold a line that misses one ----------
    def heard_of(self, item: Item) -> list[str]:
        return [need for need in item.told if need not in item.dropped]

    @staticmethod
    def heard_value(need: str) -> tuple[str, frozenset[str]] | None:
        """A heard requirement's value as heard: its figure, else all of its words (None for an empty or negated value)."""
        wanted = need.partition(":")[2].strip().lower()
        if not wanted or re.search(NOT_WORDS, wanted):
            return None
        figure = re.search(r"\d+(?:[.,]\d+)?", wanted)
        return (figure.group(0), frozenset()) if figure else ("", frozenset(_plain(wanted)))

    @staticmethod
    def holds_value(text: str, value: tuple[str, frozenset[str]]) -> bool:
        figure, stems = value
        if figure:
            return bool(re.search(rf"(?<![\d.]){re.escape(figure)}(?![\d.])", text))
        return bool(stems) and stems <= _plain(text)

    @staticmethod
    def row_words(record: Any, ident: str) -> str:
        """A row's own words: its text fields only (never its id), so a figure matches what the row says, not its price or a count."""
        def walk(node: Any, depth: int) -> list[str]:
            if isinstance(node, dict):
                return [t for v in node.values() for t in walk(v, depth + 1)] if depth < 4 else []
            if isinstance(node, list):
                return [t for v in node for t in walk(v, depth + 1)] if depth < 4 else []
            return [node] if isinstance(node, str) and node != ident else []
        return " ".join(walk(record, 0)).lower()

    def pool_room(self, item: Item) -> float | None:
        """What this item's line may cost: the budget when it is for each item, else the total budget less the cart's other lines."""
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return None
        if self.brief.get("budget_scope") == "each":
            return self.budget
        return self.budget - sum((self.price_now(i) or 0.0) * self.cart_lines.get(i, 1) for i in self.cart_ids() if not self.of_item(i, item))

    def pool_before_decline(self) -> None:
        """Before the closest order at an ending without a purchase, per item with two or more heard requirements: the unopened rows
        of the logged searches whose own words hold any heard value are read. When still no read listing fits the item within the
        budget, one search joins the item's name with the heard values as heard (once per item; the name is kept whole, the values
        cut to fit) and its hits are read when it matched 24 listings or fewer. All of these reads share POOL_OPENS an episode, and
        each keeps the steps an order needs. The closest order then re-checks every read listing."""
        keep = 3 + 2 * len(self.items)
        for number, item in enumerate(self.items, 1):
            heard = self.heard_of(item)
            if len(heard) < 2 or self.finished:
                continue
            values = [v for v in (self.heard_value(need) for need in heard) if v]
            rows: list[str] = []
            for entry in self.buy_log:
                for ident in entry["ids"]:
                    if ident in rows or ident in self.full or ident not in self.rows or self.sold_out(ident) or not self.of_item(ident, item):
                        continue
                    words = self.row_words(self.rows[ident], ident)
                    if any(self.holds_value(words, v) for v in values):
                        rows.append(ident)
            ids = rows[: max(0, POOL_OPENS - self.pool_opened)]
            if ids:
                read = self.fill_facts(ids, keep)
                self.pool_opened += len(read)
                self.trace.append({"kind": "pool_open", "item": number, "ids": ids[: len(read)]})
            if (self.finished or number in self.pool_searched or self.pool_opened >= POOL_OPENS
                    or self.closest_rows(number, item, self.pool_room(item), set(), False)):
                continue
            self.pool_searched.add(number)
            tool = self.roles.get("search")
            target = self.single_target(tool)
            parts = [self.value_phrase(need) or need.partition(":")[2].strip() for need in heard]
            name = _words(item.name).split()[:4]
            words = " ".join(name + " ".join(p for p in parts if p).split()[: max(0, 6 - len(name))])
            if not target or not words or self.left() - keep < 2:
                continue
            faults: list[str] = []
            args = _shape({target: words}, self.schemas.get(tool), "args", faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None or self.finished:
                return
            seen = reply.get("observation")
            self.observe(tool, seen)
            hits = [i for i, _ in self.records_in(seen)]
            plan = self.paging(tool, seen)
            small = (plan[0] if plan else len(hits)) <= 24
            fresh = [i for i in hits if i in self.rows and i not in self.full and not self.sold_out(i)][: max(0, POOL_OPENS - self.pool_opened)] if small else []
            read = self.fill_facts(fresh, keep) if fresh else []
            self.pool_opened += len(read)
            self.trace.append({"kind": "pool_search", "item": number, "words": words, "hits": len(hits), "read": len(read)})

    def multi_need_hold(self) -> str:
        """At the checkout, per item with two or more heard requirements: when the cart's line does not state every one of them by the
        reading rules (a contradiction counts as unmet) while an opened listing states them all and contradicts none of the request's
        own, the checkout holds once for the cart as it is (twice an episode at most), naming the unmet requirements and that listing."""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.multi_held or mark in self.multi_covered or len(self.multi_held) >= 2:
            return ""
        pointed = {need for _, need in self.refs}
        cart = self.cart_ids()
        whole = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) else None
        read = self.loose_need if BUY_KEEP else self.read_need   # (the reading the closest order uses)
        notes: list[str] = []
        for number, item in enumerate(self.items, 1):
            heard = self.heard_of(item)
            line = next((i for i in cart if self.of_item(i, item)), None)
            if len(heard) < 2 or not line:
                continue
            record = self.full.get(line) or self.rows.get(line) or {}
            unmet = [need for need in heard if read(need, record, pointed) < 1]
            if not unmet:
                continue
            musts = [m for m in item.musts if m not in item.dropped]
            fits = [i for i, rec in self.full.items() if i not in cart and self.of_item(i, item) and self.in_stock_now(i)
                    and (whole is None or (self.price_now(i) is not None and self.price_now(i) * item.qty <= whole))
                    and all(read(need, rec, pointed) == 1 for need in heard) and not any(self.read_need(m, rec, pointed) == -1 for m in musts)]
            if not fits:
                continue
            best = self.buy_best(fits) or fits[0]
            notes.append(f"the cart's {line} for item {number} does not state " + "; ".join(unmet) + f", while the opened {best} states every requirement the shopper gave (" + "; ".join(heard) + ")")
        if not notes:
            return ""
        self.multi_held.add(mark)
        self.trace.append({"kind": "multi_hold", "notes": notes})
        if FACT_TEXTS:
            return ("Kyo checked the shopper's requirements against the cart: " + " | ".join(notes) + ". The checkout was not run (no step used)"
                    + ("." if FACT_PLAIN else "; this check holds it once, and the same checkout sent again runs."))
        return ("Kyo checked the shopper's requirements against the cart: " + " | ".join(notes) + ". Put the listing that states them in the cart instead, "
                "at the requested quantity, or check out again if the cart's line is right.")

    # ---------- conversion after the questions: a sweep per item, a search the code can send, an order where nothing is contradicted ----------
    def search_args(self, words: str, item: Item | None, bare: bool = False) -> dict[str, Any] | None:
        """Arguments of a search for these words: the search tool's one text argument when it needs nothing else; else the agent's own
        latest search that returned rows (of this item's kind when one did), with its words replaced and its paging argument left out."""
        tool = self.roles.get("search")
        target = self.single_target(tool)
        if target:
            return {target: words}
        schema = self.schemas.get(tool) or {}
        texts = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        required = set(schema.get("required") or [])
        paging = (self.shapes.get(tool) or {}).get("next_arg")
        cat = self.category_key()
        calls = [e for e in self.buy_log if e.get("args") and e["ids"]]
        mine = [e for e in calls if item is not None and sum(self.of_item(i, item) for i in e["ids"]) * 2 >= len(e["ids"])]
        for entry in reversed(mine or calls):
            args = entry["args"]
            labels = {str((self.rows.get(i) or {}).get(cat) or "").strip().lower() for i in entry["ids"]} if cat else set()
            # the words' argument: the one text argument of that call whose value is not a category label of its rows
            held = [k for k in texts if isinstance(args.get(k), str) and args[k].strip() and args[k].strip().lower() not in labels]
            if len(held) != 1:
                continue
            new = {k: v for k, v in args.items() if k != paging and (k in required or (isinstance(v, (str, bool)) and not bare))}
            new[held[0]] = words
            if SEARCH_MODE:
                for key, spec in (schema.get("properties") or {}).items():
                    if key not in new or not isinstance(spec, dict) or "enum" not in spec:
                        continue
                    counts: dict[Any, int] = {}
                    for other in calls:
                        rows = {str((self.rows.get(i) or {}).get(cat) or "").strip().lower() for i in other["ids"]} if cat else set()
                        if any(isinstance(other["args"].get(t), str) and other["args"][t].strip() and other["args"][t].strip().lower() not in rows for t in texts):
                            value = other["args"].get(key)
                            counts[value] = counts.get(value, 0) + 1
                    best = max((v for v in counts if v in spec["enum"]), key=lambda v: counts[v], default=None)
                    if best is not None and counts[best] > counts.get(new[key], 0):
                        new[key] = best
            return new
        return None

    def code_search(self, words: str, item: Item | None, keep: int, bare: bool = False) -> list[str] | None:
        """One search sent by the code (a step): the ids it returned, or None when no call could be shaped or run."""
        tool = self.roles.get("search")
        key = words.lower()
        if not tool or not words or key in self.buy_searched or self.finished or self.left() - self.tail() <= keep:
            return None
        args = self.search_args(words, item, bare)
        if not args:
            return None
        faults: list[str] = []
        shaped = _shape(args, self.schemas.get(tool), "args", faults)
        if faults:
            return None
        self.buy_searched.add(key)
        reply = self.act(tool, shaped)
        if reply is None or self.finished:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        return [i for i, _ in self.records_in(seen)]

    def sweep_fit(self, item: Item, heard: list[str]) -> str | None:
        """A read listing of this item's kind, buyable within the budget, whose record states every heard requirement."""
        pointed = {need for _, need in self.refs}
        read = self.loose_need if BUY_KEEP else self.read_need
        room = self.pool_room(item)
        for ident, record in self.full.items():
            if not self.of_item(ident, item) or not self.in_stock_now(ident):
                continue
            price = self.price_now(ident)
            if room is not None and (price is None or price * item.qty > room):
                continue
            if all(read(need, record, pointed) == 1 for need in heard):
                return ident
        return None

    def sweep_query(self, item: Item, heard: list[str], requirement: bool) -> str:
        """The item's name, the heard values as heard (a figure with its unit, else the value's first choice) and, the first time,
        one short requirement of the request in its own words; six words at most, the name kept whole."""
        name = _words(item.name).split()[:3]
        parts: list[str] = []
        for need in heard:
            text = need.partition(":")[2]
            value = self.value_phrase(need) or re.split(r"\bor\b|[,/;]", text)[0].strip()
            if value and not re.search(NOT_WORDS, text.lower()) and len(value.split()) <= 3:
                parts += value.split()
        if requirement:
            for must in item.musts:
                value = re.split(r"\bor\b|[,/;]", str(must).partition(":")[2])[0].strip()
                if (must not in item.dropped and value and len(value.split()) <= 2 and not re.search(NOT_WORDS, value.lower()) and not re.search(r"\d", value)
                        and not _plain(value) <= _plain(item.name)):
                    parts += value.split()
                    break
        return " ".join(name + parts[: max(0, 6 - len(name))])

    def sweep_read(self, ids: list[str], item: Item, heard: list[str], keep: int) -> list[str]:
        """Read these listings one at a time (a step each) until one states every heard requirement."""
        out: list[str] = []
        for ident in ids:
            if self.sweep_opened >= SWEEP_OPENS or self.finished or self.left() - self.tail() <= keep:
                break
            got = self.fill_facts([ident], keep)
            if not got:
                break
            self.sweep_opened += 1
            out += got
            if self.sweep_fit(item, heard):
                break
        return out

    def item_sweep(self, where: str) -> str:
        """For each item with a heard requirement and no read listing stating them all: its seen rows whose own words hold a heard
        value are read (those holding more values first), then one search joins its name with the heard values and its top unread
        rows are read. It stops at the first record stating every heard value, and keeps the steps the cart and the checkout need.
        At an ending without a purchase it runs again for the items the cart lacks, with a search that leaves the request's word out."""
        keep = 8 + 3 * len(self.items)
        pointed = {need for _, need in self.refs}
        lines: list[str] = []
        fits: list[str] = []
        for number, item in enumerate(self.items, 1):
            heard = [n for n in self.heard_of(item) if n not in pointed]
            if not heard or self.finished or self.sweep_opened >= SWEEP_OPENS:
                continue
            if where == "decline" and any(self.of_item(i, item) for i in self.cart_ids()):
                continue
            if self.sweep_fit(item, heard):
                continue
            values = [v for v in (self.heard_value(n) for n in heard) if v]
            ranked: list[tuple[int, int, str]] = []
            for ident, row in self.rows.items():
                if ident in self.full or self.sold_out(ident) or not self.of_item(ident, item):
                    continue
                hits = sum(self.holds_value(self.row_words(row, ident), v) for v in values)
                if hits:
                    ranked.append((-hits, len(ranked), ident))
            read = self.sweep_read([i for _, _, i in sorted(ranked)][:SWEEP_ROWS], item, heard, keep)
            searched = None
            if not self.sweep_fit(item, heard) and (number, where) not in self.sweep_searched and self.sweep_opened < SWEEP_OPENS:
                self.sweep_searched.add((number, where))
                words = self.sweep_query(item, heard, where != "decline")
                found = self.code_search(words, item, keep) if words else None
                if found is not None:
                    searched = words
                    top = [i for i in found if i in self.rows and i not in self.full and not self.sold_out(i) and self.of_item(i, item)][:SWEEP_TOP]
                    read += self.sweep_read(top, item, heard, keep)
            fit = self.sweep_fit(item, heard)
            self.trace.append({"kind": "sweep", "at": where, "item": number, "read": [r.split(":")[0] for r in read], "search": searched, "fit": fit})
            lines += [r[:600] for r in read]
            if fit:
                fits.append(f"{fit} states every requirement the shopper gave for item {number}")
        if not lines:
            return ""
        return ("Kyo read listings for the shopper's new requirements (one step each):\n" + "\n".join(lines[:8])
                + ("\n" + "; ".join(fits) + "." if fits else "\nNone of them states every new requirement yet."))

    # ---------- past the bound and down to the bare name, before an ending without a purchase ----------
    def past_bound(self, item: Item, needs: list[str]) -> list[str]:
        """Values past a stated bound as the rows of this item's kind write them, commonest first: for an at-least need, the larger
        figures in the same unit; for an at-most need, the smaller ones."""
        texts = [self.row_words(row, ident) for ident, row in self.rows.items() if self.of_item(ident, item)]
        counts: dict[tuple[float, str], int] = {}
        for need in needs:
            wanted = need.partition(":")[2].lower()
            pairs = [(v, u) for v, u in _figure_pairs(wanted) if len(u) >= 2]
            floor, ceiling = re.search(FLOOR_WORDS, wanted), re.search(CEILING_WORDS, wanted)
            if len(pairs) != 1 or bool(floor) == bool(ceiling):
                continue
            value, unit = pairs[0]
            for text in texts:
                for figure in {v for v, u in _figure_pairs(text) if u[:3] == unit[:3] and (v > value if floor else v < value)}:
                    counts[(figure, unit)] = counts.get((figure, unit), 0) + 1
        return [f"{figure:g} {unit}" for (figure, unit), _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def ladder_words(self, item: Item, needs: list[str]) -> list[str]:
        """The ladder's searches: a value past a bound with the item's shortest name phrase, then each phrase alone when the name has several."""
        parts = [p for p in re.split(r"\s*(?:,|/|&|\bor\b|\band\b)\s*", item.name.lower()) if p.strip()]
        phrases = sorted(parts, key=lambda p: len(p.split()))
        noun = phrases[0] if phrases else _words(item.name)
        words = [f"{value} {noun}" for value in self.past_bound(item, needs)[:2]] + (phrases if len(phrases) > 1 else [])
        used = {e["words"] for e in self.buy_log} | self.buy_searched
        return [w for w in dict.fromkeys(words) if w not in used][:END_LADDER]

    def end_ladder(self) -> None:
        """END_LADDER: for each item the cart lacks, the ladder's searches and their unread rows (those holding a requirement's value or a
        value past its bound first), keeping the steps the closest order needs. Under no cheapest, most or least pick it stops at a record
        stating every requirement."""
        keep = 2 + 2 * len(self.items)
        pointed = {need for _, need in self.refs}
        ranked_pick = self.brief.get("pick") in ("cheapest", "most", "least")
        for number, item in enumerate(self.items, 1):
            if self.finished or any(self.of_item(i, item) for i in self.cart_ids()):
                continue
            needs = [n for k, n in self.needs_now() if k == number and n not in pointed]
            values = [v for v in (self.heard_value(n) for n in needs) if v] + [(v.split()[0], frozenset()) for v in self.past_bound(item, needs)[:2]]
            room = self.pool_room(item)
            tried: list[str] = []
            read: list[str] = []
            done = False
            for words in self.ladder_words(item, needs):
                found = self.code_search(words, item, keep)
                if found is None:
                    continue
                tried.append(words)
                unread = [i for i in found if i in self.rows and i not in self.full and not self.sold_out(i) and self.of_item(i, item)
                          and (room is None or (self.price_now(i) is not None and self.price_now(i) * item.qty <= room))]
                held = {i: sum(self.holds_value(self.row_words(self.rows[i], i), v) for v in values) for i in unread}
                for ident in sorted(unread, key=lambda i: -held[i]):
                    if self.ladder_opened >= LADDER_OPENS or self.left() - self.tail() <= keep:
                        done = True
                        break
                    if not self.fill_facts([ident], keep):
                        done = True
                        break
                    self.ladder_opened += 1
                    read.append(ident)
                    if not ranked_pick and self.states_all(ident, item):
                        done = True
                        break
                if done:
                    break
            if tried:
                self.trace.append({"kind": "end_ladder", "item": number, "words": tried, "read": read})

    # ---------- the pages a requirement points at, right after the questions ----------
    def interview(self) -> str:
        if self.stacking():
            try:
                said = self.stack_interview()
            except Exception as err:
                self.slip("stack_interview", err)
                said = super().interview()
        else:
            said = super().interview()
        if FILL_ASK and not self.process and self.roles.get("ask"):
            # with no step of the request's own, no later call can take a message the questions left unsent
            try:
                sent, told = self.fill_asks(self.msgs_free, "questions", False)
            except Exception as err:
                self.slip("fill_questions", err)
                sent, told = [], []
            if sent:
                if told:
                    self.version += 1
                lines = "\n".join(f"Q: {q}\nA: {a}" for _, q, a in sent)
                said = (said + "\nKyo asked one more property with a message that would have gone unsent (each question used one step):\n" + lines
                        + "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none") + ". Apply them to every pick.").strip()
        if ITEM_SWEEP and self.roles.get("search"):
            try:
                swept = self.item_sweep("questions")
            except Exception as err:
                self.slip("sweep", err)
                swept = ""
            if swept:
                said = (said + "\n" + swept).strip()
        if (BUY_SEARCH or WORD_VALUE_SEARCH) and self.roles.get("search"):
            try:
                found = self.buy_searches()
            except Exception as err:
                self.slip("buy_search", err)
                found = ""
            if found:
                said = (said + "\n" + found).strip()
        if not CHASE_ROUNDS:
            return said
        try:
            more = self.chase()
        except Exception as err:
            self.slip("chase", err)
            more = ""
        return (said + "\n" + more).strip() if more else said

    def link_record(self, ident: str) -> str:
        """A record for a chain's reader: the id first, then what only the page view shows, so that a store's field order or a
        long description cannot cut the id, the seller or a page's own notes off the end."""
        record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
        if not record:
            return "ROW " + ident + " " + _dump(_clip(self.rows.get(ident), 80))[:500]
        page_only = self.tool_keys.get(self.roles.get("offer") or "", set()) - set(self.rows.get(ident) or {})
        ordered = {**{k: v for k, v in record.items() if k in page_only}, **{k: v for k, v in record.items() if k not in page_only}}
        return "OPENED " + ident + " " + _dump(_clip(ordered, 300))[:1800]

    def trail(self) -> tuple[str, bool]:
        """The parent's check before a checkout, with its records written so a row's id and seller stay readable."""
        if not TRAIL_ROWS:
            return super().trail()
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
            self.trails += 1
            records = [self.link_record(ident) for ident in list(reversed(self.seen_order))[:60]]
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
                     + " | ".join(lost[:3]) + ("; the value the order needs is on that page." if FACT_TEXTS and FACT_PLAIN else ". A search by its shop, brand or words finds it, and its page gives the value the order needs." if FACT_TEXTS
                                               else ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out."))
        return text, hold

    def single_readers(self) -> list[str]:
        """Every tool that takes one listing id and requires nothing else, the page view first: found by the tools' shape,
        because a store can split a listing across views and only one of them may show the page's own notes."""
        page, facts = self.roles.get("offer"), self.roles.get("open")
        other_jobs = set(self.roles.values()) - {page, facts, None}
        tools = [t for t in self.schemas if t not in other_jobs and t not in self.no_reader and self.single_target(t)]
        return sorted(tools, key=lambda t: t != page)

    def room_to_chase(self) -> bool:
        return (not self.finished and not self.broke and not self.no_credit and len(self.items) <= CHASE_ITEMS and self.chase_calls < CHASE_STEPS and self.chase_spent < CHASE_SPEND
                and self.left() - self.tail() >= CHASE_KEEP + CHASE_KEEP_ITEM * max(1, len(self.items)) and self.cost < SPEND_SOFT)

    def read_link(self, ident: str) -> list[str]:
        """A possible link of a chain. The first one is read with every view; after that, the view the model named as the page,
        else every view again. A view that answers with something other than this listing's record is not asked again."""
        texts: list[str] = []
        readers = self.single_readers()
        if self.chase_read_all and self.roles.get("offer") in readers:
            readers = [self.roles["offer"]]
        for tool in readers:
            if not self.room_to_chase():
                break
            if self.shown_already(tool, ident):
                continue
            faults: list[str] = []
            args = _shape({self.single_target(tool): ident}, self.schemas.get(tool), "args", faults)
            if faults:
                continue
            reply = self.act(tool, args)
            self.chase_calls += 1
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and ident in _dump(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
                self.opened.add(ident)
                texts.append(f"{ident}: " + self.show(seen)[:900])
            else:
                self.no_reader.add(tool)
        self.chase_read_all = True
        return texts

    def find_link(self, words: str) -> bool:
        """A search for a link no record shows yet; only a search tool that requires nothing but its words."""
        tool = self.roles.get("search")
        target = self.single_target(tool)
        key = words.lower()
        if not tool or not (target or CHASE_SEARCH) or not words or key in self.chase_words or not self.room_to_chase():
            return False
        self.chase_words.add(key)
        if not target:
            found = self.code_search(words, None, CHASE_KEEP, bare=True)
            if found is None:
                return False
            self.chase_calls += 1
            self.chase_hits = found
            return bool(found)
        faults: list[str] = []
        args = _shape({target: words}, self.schemas.get(tool), "args", faults)
        if faults:
            return False
        reply = self.act(tool, args)
        self.chase_calls += 1
        if reply is None or self.finished:
            return False
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.chase_hits = [i for i, _ in self.records_in(seen)]
        return bool(self.chase_hits)

    def chase(self) -> str:
        """Each requirement whose value is on another listing: a light model reads the records for the next link, code opens
        what could be it or searches for it, until the page that states the value. Bounded by rounds, store calls, steps left and spend;
        what is left goes to the parent's own check before a checkout."""
        seen: list[str] = []
        rounds = 0
        skipped: set[str] = set()
        while rounds < CHASE_ROUNDS and self.room_to_chase():
            live = {need for _, need in self.needs_now()}
            pending = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in skipped]
            if not pending:
                break
            rounds += 1
            # the rows of the chase's own last search come first: rows seen earlier keep their old place in the record
            newest = list(dict.fromkeys(self.chase_hits + list(reversed(self.seen_order))))[:60]
            records = [self.link_record(ident) for ident in newest]
            prompt = CHASE + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            if self.chase_words:
                prompt += "\n\nWORDS ALREADY SEARCHED FOR LINKS: " + "; ".join(sorted(self.chase_words))
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            before = self.cost
            got = self.ask_json(prompt, [LIGHT], CHASE_TOKENS) or {}
            self.chase_spent += self.cost - before
            moved = False
            opened: list[str] = []
            searched: list[str] = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                if chain.get("no_other_listing") is True:
                    # (left in place for the parent's own check; only this chase stops on it)
                    skipped.add(need)
                    moved = True
                    continue
                settled = [str(i).strip() for i in _items(chain.get("settled"))]
                if chain.get("finished") is True and settled and all(i in self.opened for i in settled) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    moved = True
                    continue
                # a search that names the link's details usually puts it first, for one step: it comes before reading rows,
                # and its first rows of the named seller are read at once, since a row can't show what only the page states
                named = str(chain.get("link_seller") or "").strip().lower()
                words = str(chain.get("search") or "").strip()
                closed = []
                if words and self.dry_searches < 3 and words.lower() not in self.chase_words:
                    self.chase_hits = []
                    self.find_link(words)
                    searched.append(words)
                    moved = True
                    self.dry_searches += 1
                    if named:
                        closed = [i for i in self.chase_hits if i in self.rows and i not in self.opened and named in _dump(self.rows[i]).lower()][:CHASE_OPEN]
                else:
                    closed = [i for i in (str(x).strip() for x in _items(chain.get("read_next"))) if i in self.rows and i not in self.opened and named in _dump(self.rows[i]).lower()][:CHASE_OPEN]
                if closed:
                    self.dry_searches = 0
                    for ident in closed:
                        texts = self.read_link(ident)
                        seen += texts
                        opened.append(ident)
                        moved = moved or bool(texts)
            self.trace.append({"kind": "chase", "round": rounds, "pending": pending, "got": _clip(got, 150), "opened": opened, "searched": searched, "calls": self.chase_calls, "left": self.left(), "spent": round(self.chase_spent, 4)})
            if not moved:
                break
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement may point at]\n" + "\n".join(seen[-4:]) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        return text.strip()

    # ---------- the closest read fit, past every hold ----------
    def read_figure(self, wanted: str, named_values: list[str], text: str) -> int:
        """A requirement holding one figure, against the attribute naming its property (else the whole record). Units are
        compared by their written letters only: different letters say nothing either way."""
        wanted = re.sub(r"\bno more than\b", "under", re.sub(r"\bno less than\b", "least", wanted))
        pairs = _figure_pairs(wanted)
        if len({n for n, _ in pairs}) != 1 or re.search(NOT_WORDS, wanted):
            return 0
        goal, unit = pairs[0]
        low, high = bool(re.search(FLOOR_WORDS, wanted)), bool(re.search(CEILING_WORDS, wanted))
        if named_values:
            same = [n for value in named_values for n, u in _figure_pairs(value) if not unit or not u or u == unit]
            if not same:
                return 0
            good = any(n >= goal if low else n <= goal if high else n == goal for n in same)
            return 1 if good else -1
        return 1 if any(n == goal and (not unit or u == unit) for n, u in _figure_pairs(text)) else 0

    def read_need(self, need: str, record: dict[str, Any], pointed: set[str]) -> int:
        """One requirement against one read record, by the record's own words: 1 stated, -1 contradicted by an attribute
        that names the property, 0 nothing either way. The reading rules are the ones the agent's own check is given."""
        prop, _, wanted = need.partition(":")
        wanted = wanted.strip().lower()
        # (a value read off another listing comes with a relation to it, the same, unlike, going with: only a model can weigh that)
        if need in pointed or not wanted:
            return 0
        block = _fact_block(record) or {}
        named_values = [str(v) for k, v in block.items() if _names(_stem(prop), _stem(k))]
        text = " ".join(_scalars(record))
        if re.search(r"\d", wanted):
            return self.read_figure(wanted, named_values, text)
        negated = re.search(NOT_WORDS, wanted)
        if negated:
            rest = _plain(wanted[negated.end():])
            return 0 if not rest else (-1 if rest <= _plain(" ".join(named_values)) else 1) if named_values else (0 if rest <= _plain(text) else 1)
        options = [o for o in (_plain(part) for part in re.split(r"\bor\b|[,/;]", wanted)) if o]
        where = _plain(" ".join(named_values) if named_values else text)
        if any(o <= where for o in options):
            return 1
        return 0 if not named_values or any(o & where for o in options) else -1

    def price_now(self, ident: str) -> float | None:
        """The listing's price: under the key the model named, else the one numeric field nearly every row carries."""
        if self.price_key:
            return self.price_of(ident)
        total = max(1, len(self.rows))
        tally: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    tally[key] = tally.get(key, 0) + 1
        keys = [k for k, n in tally.items() if n >= 0.8 * total]
        value = (self.rows.get(ident) or {}).get(keys[0]) if len(keys) == 1 else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def in_stock_now(self, ident: str) -> bool:
        """Buyable by the parent's test; with no stock key named, also not false on the one yes/no field nearly every row carries."""
        if not self.buyable(ident):
            return False
        if self.stock_key:
            return True
        total = max(1, len(self.rows))
        tally: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, bool):
                    tally[key] = tally.get(key, 0) + 1
        keys = [k for k, n in tally.items() if n >= 0.8 * total]
        return not (len(keys) == 1 and (self.rows.get(ident) or {}).get(keys[0]) is False)

    def of_item(self, ident: str, item: Item) -> bool:
        """The listing's category label names this item, by the same test the parent groups listings with; a listing with no
        label does not qualify."""
        cat = self.category_key()
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        if not cat:
            # (no store label to read: one item only, and the listing's own words must name it)
            return len(self.items) == 1 and bool(long & _plain(" ".join(_scalars(self.rows.get(ident) or {}))))
        label = str((self.rows.get(ident) or {}).get(cat) or "")
        if not label:
            return False
        return _words(label) in (_words(item.shelf or ""), _words(item.name)) or bool({w for w in _stem(label) if len(w) >= 4} & long)

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        said = super().check_cart(ids, limit)
        try:
            names = ids if ids is not None else self.cart_ids()
            if said and ids is None and CHECK_SWAP and SWAP_AT_CHECK:
                swapped = self.check_swap()
                if swapped:
                    return swapped
            if said and ids is None and CHECK_FILTER:
                last = next((x for x in reversed(self.trace) if x.get("kind") == "check" and sorted(x.get("cart") or []) == sorted(names)), None)
                better = [str(b) for b in _items(((last or {}).get("got") or {}).get("better"))]
                if better and not any(self.supported(b) for b in better if b not in names) and self.cart_states_heard():
                    # (the check prefers listings whose records do not state what the shopper said, over lines that do)
                    self.verdicts[self.check_key(names)] = True
                    self.trace.append({"kind": "check_filter", "dropped": better[:4]})
                    said = ""
            if self.verdicts.get(self.check_key(names)):
                under = _dump(self.needs_now())
                self.passed_under.update({i: under for i in names})
        except Exception as err:
            self.slip("check_cart", err)
        return said

    def ruled(self) -> tuple[set[str], set[str]]:
        """Listings an independent check passed under the requirements now in force, and lone listings it failed under them."""
        under = _dump(self.needs_now())
        passed = {i for i, needs in self.passed_under.items() if needs == under}
        failed: set[str] = set()
        for key, good in self.verdicts.items():
            ids, _, version = key.rpartition("#")
            names = ids.split("|")
            if not good and len(names) == 1 and version == str(self.version):
                failed.update(names)
        return passed - failed, failed

    def closest_rows(self, number: int, item: Item, room: float | None, taken: set[str], forced: bool) -> list[tuple[str, float]]:
        """The read listings of this item's kind that can be ordered within `room`, best first: one a check passed under the
        requirements now in force, then one whose own record states every requirement. A record that contradicts a requirement
        never qualifies, and a listing that merely fails to say otherwise is no reason to overrule an ending without a purchase.
        The review's named fits also count at a forced ending, where the actor never got to weigh them; at a decline the
        actor has already seen them and refused."""
        passed, failed = self.ruled()
        named = set(self.leads) if forced else set()
        pointed = {need for _, need in self.refs}
        needs = [need for n, need in self.needs_now() if n == number]
        rows: list[tuple[tuple[Any, ...], str, float]] = []
        for ident, record in self.full.items():
            if ident in taken or not self.in_stock_now(ident) or not self.of_item(ident, item):
                continue
            price = self.price_now(ident)
            if price is None or (room is not None and price * item.qty > room):
                continue
            marks = [(self.loose_need if BUY_KEEP else self.read_need)(need, record, pointed) for need in needs]
            if -1 in marks:
                continue
            tier = 0 if ident in passed else 1 if ident in named else 2
            heard = [m for need, m in zip(needs, marks) if need in item.told]
            # (ORDER_OVER_DECLINE: every heard requirement stated and nothing contradicted is enough; a silent record is not a failed one)
            # (STRICT_OVERRIDE: only at a forced ending; where the agent chose to end, a silent record is unproven)
            lenient = ORDER_OVER_DECLINE and (forced or not STRICT_OVERRIDE) and bool(heard) and all(m == 1 for m in heard) and ident not in failed
            if tier == 2 and not lenient and (not marks or 0 in marks or ident in failed):
                continue
            measure = self.pick_value(ident) if END_LADDER and self.brief.get("pick") in ("most", "least") else None
            rank = 0.0 if measure is None else (-measure if self.brief.get("pick") == "most" else measure)
            rows.append(((tier, marks.count(0), -marks.count(1), rank, price), ident, price))
            self.closest_why[ident] = {"tier": ("checked", "reviewed", "stated")[tier], "stated": marks.count(1), "silent": marks.count(0)}
        return [(ident, price) for _, ident, price in sorted(rows)]

    def closest_plan(self, forced: bool) -> tuple[dict[str, int], list[str], list[str]] | None:
        """(all lines the order would hold, the new picks, the lines to drop) when every item the cart lacks has a closest
        read fit within the budget; None otherwise. A one-item cart that holds anything is left alone."""
        now = [i for i in self.cart_ids() if not self.sold_out(i)]
        if not now and self.cart_touched:
            # (an empty record after cart calls: the store's last cart reply, when it shows lines, is the truth)
            now = [i for i in (self.store_lines(self.cart_last) or {}) if not self.sold_out(i)]
        if now and len(self.items) < 2:
            return None
        lines: dict[str, int] = {}
        missing: list[tuple[int, Item]] = []
        for number, item in enumerate(self.items, 1):
            held = [i for i in now if self.of_item(i, item) and i not in lines]
            if held:
                lines[held[0]] = self.cart_lines.get(held[0], item.qty)
            else:
                missing.append((number, item))
        if not missing:
            return None
        whole = self.budget if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" else None
        each = self.budget if isinstance(self.budget, (int, float)) and whole is None else None
        spent = 0.0
        for ident, units in lines.items():
            price = self.price_now(ident)
            if price is None and whole is not None:
                return None
            spent += (price or 0.0) * units
        picks: list[str] = []
        for number, item in missing:
            room = each if each is not None else (whole - spent if whole is not None else None)
            rows = self.closest_rows(number, item, room, set(lines), forced)
            if not rows:
                return None
            ident, price = rows[0]
            lines[ident] = item.qty
            picks.append(ident)
            spent += price * item.qty
        return lines, picks, [i for i in self.cart_ids() if i not in lines]

    def closest_order(self, forced: bool = False) -> str:
        """Past every hold, an ending without a purchase (or a forced ending) with an item uncarted becomes the order of the
        closest read fit per item, once. It asks the models nothing beyond what the order itself is due. Returns what happened
        for the actor, or nothing when the ending should go ahead."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not CLOSEST_ORDER or self.closest_sent or self.finished or self.broke or not cart or not buy:
            return ""
        plan = self.closest_plan(forced)
        if not plan:
            return ""
        lines, picks, drop = plan
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        shaped: list[dict[str, Any]] = []
        for call in calls:
            faults: list[str] = []
            shaped.append(_shape(call, self.schemas.get(cart), "args", faults))
            if faults:
                return ""
        faults = []
        order = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        self.closest_sent = True
        self.trace.append({"kind": "closest_order", "picks": picks, "lines": lines, "drop": drop, "left": self.left(), "forced": forced,
                           "why": {i: self.closest_why.get(i) for i in picks}})
        for args in shaped:
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        extra = [i for i in self.cart_lines if i not in lines]
        if extra:
            self.take_out(extra, lines)
        said: list[str] = []
        if self.left() >= 2:
            go, spoken = self.gate("order")
            if spoken:
                said.append(spoken)
            if not go:
                return "Not run (Kyo acted instead): Kyo carted " + ", ".join(picks) + ", whose records state what the request asks. " + " ".join(said)
        if self.finished or self.left() < 1:
            return ""
        reply = self.act(buy, order)
        self.trace.append({"kind": "closest_order_sent", "reply": _clip(reply, 200)})
        if reply is None:
            return "Not run (Kyo acted instead): Kyo carted " + ", ".join(picks) + "; the checkout call failed, so you may send it again."
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
            self.rearm("closest")
        self.observe(buy, reply.get("observation"))
        said.append(self.show(reply.get("observation")))
        return "Not run (Kyo acted instead): Kyo carted " + ", ".join(picks) + ", whose records state what the request asks, and checked out. Store reply: " + " | then: ".join(said)

    def complete_cart(self) -> str:
        done = super().complete_cart()
        if done or self.finished:
            return done
        if CART_OVER_DECLINE:
            try:
                done = self.order_cart()
            except Exception as err:
                self.slip("order_cart", err)
                done = ""
            if done or self.finished:
                return done
        if BUY_OPEN:
            try:
                self.open_value_hits(2 + 2 * len(self.items))
            except Exception as err:
                self.slip("buy_open_end", err)
        if ITEM_SWEEP:
            try:
                self.item_sweep("decline")
            except Exception as err:
                self.slip("sweep_end", err)
        elif POOL_BEFORE_DECLINE:
            try:
                self.pool_before_decline()
            except Exception as err:
                self.slip("pool", err)
        if END_LADDER:
            try:
                self.end_ladder()
            except Exception as err:
                self.slip("end_ladder", err)
        try:
            return self.closest_order()
        except Exception as err:
            self.slip("closest_order", err)
            return ""

    def finish(self, again: bool = False) -> str:
        if not (self.finished or self.broke or self.left() <= 0):
            if POOL_BEFORE_DECLINE:
                try:
                    self.pool_before_decline()
                except Exception as err:
                    self.slip("pool_forced", err)
            try:
                self.closest_order(forced=True)
            except Exception as err:
                self.slip("closest_forced", err)
        return super().finish(again)



def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Keeper(problem_data)
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
