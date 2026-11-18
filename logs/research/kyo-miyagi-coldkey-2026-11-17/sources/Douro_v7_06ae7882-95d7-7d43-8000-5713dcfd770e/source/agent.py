from __future__ import annotations

import difflib
import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST_LLM = "anthropic/claude-haiku-5.5"
STRONG_LLM = "anthropic/claude-sonnet-5.5"
BACKUP_LLM = "openai/gpt-6-luna"

TURN_LIMIT = 42
CALL_LIMIT = 60
SOFT_SPEND = 0.055
HARD_SPEND = 0.078

# pauses between resends of a failed LLM call, and their summed cap in seconds
RETRY_WAITS = (4, 10, 20, 35)
RETRY_BUDGET = 45

SHOW_LIMIT = 450
RESULT_LIMIT = 15000
CHAT_LIMIT = 150000

# world calls the step gate may spend ahead of one checkout or ending
GATE_CALLS = 3
# extra result pages read in code when the pick is a cheapest/most/least one
EXTRA_PAGES = 6
# cheaper same-kind rows opened before a cheapest checkout goes through
CHEAPER_LOOKS = 3
# seconds into the episode (ORO stops it at 300): urge checkout, skip the pre-buy LLM checks, stop starting LLM turns
CLOCK_ASK = 110
CLOCK_WARN = 190
CLOCK_LATE = 210
CLOCK_STOP = 240

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
LOOP_PROMPT = """Mission:
You are Kyo, a buyer for one shopper in an online store; you act only through its tools. Scoring is all or nothing: only an exactly correct finish earns 1, anything else 0. Correct means either an order holding every requested item at its requested quantity, each meeting every requirement in force at checkout, the whole order within budget; or, if the store truly offers nothing that meets everything, an ending through the no-purchase tool done the way the request says.

Materials:
* The store's tools, with names, parameters and result layouts unique to this store: study each description and schema and pass exact parameter names and allowed values.
* Kyo's ledger: the requirements in force, the steps requested along the way and, once Kyo knows it, the tool that shows attributes.
* Answers from Kyo's interviewer, who asks the shopper every question for you (never message the shopper yourself); replies show up here and enter the ledger.
* [evidence] lines that Kyo adds after opened listings, quoting the keys whose names match a requirement.

Checklist:
Working the store
* Each tool call uses one of a limited number of steps; thinking is free.
* Searches may be narrow: a category filter, a keyword mode reading only the first words of the query, short pages, a token or offset for the next page. When the answer needs every row, compare the reported total with the rows seen and keep paging with the token or offset exactly as returned. Lead short keyword queries with the value: the wanted value, then the product noun. An empty filtered search usually means a wrong filter value or parameter; retry with the label spelled exactly as rows show it, or another filter parameter.
* Search rows are summaries; stores split a listing's record differently: one view may carry its attributes (properties), another only its offer and description, and a side-by-side view may hold either. Judge property requirements on those attributes, so read them for any listing you might buy; one call covering several listings saves steps if it carries what you need. Never reopen a listing unchanged since you read it.
* The request may also demand steps along the way, such as an approval, a message, or something to check or read beforehand; the ledger lists them. Kyo does those due right before the order or the ending itself when you call the checkout or ending tool; never wait for them or end without a purchase because of them. Perform any other such step with the store's tools (product questions still go through Kyo's interviewer).
Deciding whether a listing fits
* The listing's category label settles what kind of product it is. Only a label matching the requested item as a whole qualifies; a neighbouring one never does.
* A requirement on a property is satisfied when the listing's attribute for that property states the value. Key spelling is irrelevant (spaces, underscores, camel case, prefixes), and one element of a comma or slash list suffices. Values compare word by word: case, singular versus plural and plain synonyms are acceptable, and a value including the required value's words qualifies ("deep teal" meets teal, "extra roomy fit" meets roomy). A value built from other words does not qualify: an abbreviation or code is not the word it abbreviates, and "teal green" is not "sea green". Convert units before comparing numbers; limits include their edge value, and a range has to fall entirely within a limit.
* When a key names the requirement's property, by that name or a plain synonym, that attribute is decisive: if it states otherwise, the listing fails whatever other keys or the title claim. A key about another property, even a related one (a handle colour is not the body's colour), settles nothing about the requirement.
* If no attribute names that property, the wanted value under some other attribute (for example a list of styles or features containing it) or in the title or description qualifies. A number measuring something else does not (a weight is not a capacity), and neither does a bare figure whose text never says what it measures.
* If the category label itself carries a requirement's value (a season or an audience inside the label), the category satisfies that requirement.
* A requirement may refer to another listing rather than give a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle it before anything else: locate that listing (search by its category, brand, shop or wording), open its page completely with the single-item or side-by-side tool (a search row does not suffice), and if that page sends you on to yet another listing for the property, find and open it likewise; the final page supplies the value. The order is valid only if every page in that chain was opened. When the wanted value is anything but that listing's value, the rule excludes only listings with that exact value and alone never justifies ending without a purchase.
* If a listing you are about to buy points you to a further page about itself, view that page before buying.
* If one requested item has to suit another (an accessory for a device), choose the pair together from the listings' own wording (a list of compatible products, a model or series code printed on both pages). If no accessory meeting its own requirements suits the device you chose, move to a different device that meets its requirements and that such an accessory suits; never end without a purchase before trying that move.
* Honour exclusions and closed lists. Wishes ("ideally", "would be nice") only break ties.
* Never buy something the store marks as sold out. The total (price times quantity, summed over all lines) must not exceed the budget; hitting it exactly is fine.
Picking among fits
* For a cheapest request (lowest total, smallest price), the order has to contain the lowest-priced listing out of ALL fitting listings. Page through the whole fitting set (rows often come in no price order; Kyo fetches further pages of a short result itself), examine candidates from the lowest price upward, and take the first that fits completely. If a search accepts a maximum price, use it to list the category below your current best pick's price, so only cheaper candidates remain to examine.
* For a most or least request on some measure, compare every fitting listing on that measure and choose the extreme one; a listing that does not state the measure is out of the running.
* Otherwise any listing meeting everything is correct; spend no steps looking for better.
* Cart an item the moment you have opened a listing satisfying all its requirements (for a cheapest, most or least pick, once the fitting set has been compared), then continue with the next one. Carted lines often sell out soon after; carting early keeps steps free for a replacement.
* Several different items call for one cart line each, every line at its requested quantity.
Handling changes
* The shopper or the store may change things: a new budget, a requirement added or removed, a price shift, a listing selling out. They appear in tool results, often when you try checkout (a refusal); a note counts no matter which tool's result displays it, and the ledger shows the requirements now in force. A refusal does not end the episode: apply every change, repair the cart, and check out again.
* Re-read a chosen listing's record shortly before checkout; price and stock can shift without notice.
Closing the episode
* Leave exactly the chosen lines in the cart (take everything else out), view the cart once, then check out in a separate call.
* End without a purchase only after a genuine search found no listing meeting everything. Running low on steps proves no such thing; in that case check out the best cart you have. Word that ending as the request says; any remark it calls for stays brief and names no listing.

Return:
Each reply must include at least one tool call. Keep any text brief: what you have established, then your next move."""


BRIEF_PROMPT = """Mission:
Turn a shopper's request into a task sheet for a shopping agent.

Materials:
The shopper's request and the store's tools.

Checklist (one bullet per answer key):
* "items": a list with an entry for each distinct product to be bought, each {"name": the type of product in the request's own wording, "shelf": its category label in the store, copied exactly, when some tool lists labels, otherwise "", "quantity": how many units of that single listing (an integer; 1 if unstated; a pair or a set sold as one listing counts as 1), "price_cap": when the request names several products and sets a price limit on this one alone, that limit for one unit as a number in the request's unit, otherwise null, "musts": a list of firm requirements on the product, each written "property: constraint" carrying the exact value, limit, exclusion or closed list. Call the property by the word the request itself uses ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"), never a neighbouring one. A must whose value comes from another listing is a SINGLE must containing that listing's entire description ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): whatever describes the other listing (its shop, its kind, a following "with <property> <value> and <property> <value>") stays inside that one must, never a must of this product}. Price, budget, stock and quantity never go into musts. The narrative around the request (who will use the product, where it will be placed, what failed before, the occasion behind it) is background, not a must, unless it plainly states a property's value or limit; a sentence that only says the shopper is particular about some property, or leaves it to you, without naming its value is never a must (that property gets asked about later); and words forming part of the product's own name (a season, an audience or a use stated together with the product) never become musts: they belong in "name". A product the request names in a single phrase linked by "and" (one "I want ..." with one quantity, frequently the name of one store department) is ONE item; make separate items only if the request gives each product its own quantity or its own requirements.
* "budget": the highest total the shopper will pay for the whole order, as a number in the request's unit, or null; with several products, a limit on one product alone is that item's "price_cap" and never the budget.
* "budget_unit": that unit spelled out in words (for example "cents"), or null.
* "budget_scope": "total" if the limit applies to the entire order, "each" if it applies per item, null without a limit.
* "in_stock_only": true, unless the request plainly accepts unavailable items.
* "pick": "cheapest" if the request wants the lowest price or the smallest total, "most" or "least" if it wants the extreme of some measure, otherwise "any". Remarks about tight money set no pick rule.
* "pick_measure": the measure being maximised or minimised for "most"/"least", otherwise "".
* "messages": the number of messages the shopper lets the agent send (integer), or null if unstated.
* "steps": the number of steps, moves or tool calls the request permits (integer), or null.
* "lookups": a list of those musts (copied exactly) whose value must first be read from another listing the request refers to (a listing outside this order, described by its shop, brand or wording), otherwise []. A must that one requested item suit or fit another item of this same order is never a lookup: it stays an ordinary must, judged on the pages of the items chosen.
* "rules": a list of every instruction in the request on working or finishing, each in 12 words or fewer.
* "wishes": a list of the shopper's mere nice-to-haves.
* "ask_first": a list of whatever the request explicitly says must be checked with the shopper before choosing (empty if nothing).
* "process": a list of the steps the request demands from the agent apart from finding and buying the right items (obtaining the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": the action required, in 15 words or fewer, "when": "choose" (before the products are chosen), "order" (just before checking out) or "end" (right before ending without a purchase), "messages": the number of messages to the shopper it needs (0 if none)}; [] if the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and the way to end when nothing fits are not steps.
* "quiet_decline": true if the request says alternatives must not be suggested or named when nothing fits, false if it asks for them, null if it is silent.
* "no_fit_rule": the request's own sentence (or sentences) on what to do when nothing meets every requirement, copied word for word with its conditions, or null if the request says nothing about that case.
* "roles": an object giving the exact tool for each job, or null for a job no tool performs: "search", "open" (displays one listing's attributes or properties; when there are two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (displays one listing's current offer, such as its price and stock, when that is a tool other than "open"), "compare" (several listings in full), "ask" (sends the shopper a message), "cart" (modifies the cart), "cart_view" (displays the cart), "buy" (checks out), "refuse" (ends without a purchase).
* "buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), otherwise {}.
* "refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, otherwise {}.

Return:
ONE compact JSON object with exactly these keys, no indentation, nothing around it.
"""


PLAN_PROMPT = """Mission:
Pick the questions a shopping agent will put to a shopper. Every shopper keeps a few firm preferences they never wrote down, and each concerns one property of one requested product. A question uncovers a preference only if it names that very property, and the shopper answers each question solely about the first property it names, so a question spends exactly one property.

Materials:
* The request, plus the items with what the request already fixes for each.
* A numbered list of attribute keys found on opened listings in each item's category, with how many of those listings carry each key and sample values. An unwritten preference may concern a property that none of the listed keys names.

Checklist (in priority order):
* Step 0 - anything the request itself tells you to check with the shopper (shown under ASK FIRST if present), each named by the property it concerns.
* Step 1 - hinted properties. The story in the request often signals an unwritten preference without revealing its value: a sentence about something the shopper cannot judge or hands over to you, a person whose taste is decisive, how the product looks or the surroundings it belongs to, who will use or wear it, or what it contains. For every such sentence, name the single property it signals and the item it concerns, following the story's order. Use the property's plain name: a hint about the wearer signals gender, a hint about looks signals colour or finish (whichever one the request leaves open).
* Step 2 - after those, the listed keys in ranked order: prefer the properties a buyer of that product type decides on first (frequently its size, capacity or dimensions, its type, its material or its colour, whichever that product type is chosen by) and keys present on most of the item's listings; stay away from keys found on just one or two.
* Leave out a property the request already fixes for that item, another name for a property already picked, a key listing purchase variants (a mixture of sizes, colours and pack counts), identifiers, warranty and other logistics. For several items, list all items' hinted properties first and then alternate among them.
* "label" holds the property's plain name: for a listed key, every word of the key in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never cut down to "colour" or "finish"); just the property's name, without example values, brackets or a second property.
* Pick at most COUNT, strongest first.

Return:
ONE JSON object and nothing more: {"ask": [{"item": item number, "label": "...", "hint": "no more than 8 words of the story it is drawn from, or an empty string"}]}"""


HEAR_PROMPT = """Mission:
A shopping agent questioned a shopper; read every answer.

Materials:
The questions and answers.

Checklist:
* "needs": each firm preference stated in an answer, carrying the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need after the property the answer itself mentions, even if another was asked about; several answers on a single property form a single need. Reproduce the value word for word as the answer gives it, with its nouns and units intact ("at least 3 shelves", not "at least 3").
* An answer may supply the value by referring to another listing (a value like, unlike or close to the one on some shop's product, identified by a few of its properties): record ONE need on the property the answer names, with "points": true. Its value is the answer's own wording copied exactly, running from the comparison words to the end of that description, including whatever it says about a listing that page might name; do not trim or rephrase it. The properties describing that other product are not needs of this item.
* "free": for each question the shopper is indifferent to, its property.
* Homely or indirect phrasing that names a value still states it.

Return:
ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "budget": new total limit as a number or null}"""


NOTE_PROMPT = """Mission:
Mid-task, a shopping agent was shown the note below (it may be from the shopper or the store).

Materials:
The note and current requirements.

Checklist:
* Record a replaced requirement as a drop of the old one and an add of the new one.
* Report all changes in the note.

Return:
ONE JSON object and nothing else:
{"kind": "change" if the note alters the task (budget, requirements, a price, availability, a refusal of the order), "error" if it only says a call was malformed or rejected, "other" otherwise, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true if it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true if it reports a price change, "summary": "one short sentence"}"""


CHECK_PROMPT = """Mission:
You are the final review before a shopping agent checks out: decide from the records shown whether the cart is exactly right.

Materials:
Request, requirements in force, cart and listing records; of several records for one listing, the newest counts.

Checklist:
* A requirement holds when the listing's attribute for that property states the value: key spelling is irrelevant, one list element suffices, case, singular/plural and plain synonyms pass, a value containing the required value's words qualifies (such as "deep teal" for teal), units converted, limits include their edge, and a range has to fall entirely inside.
* An attribute whose key names the requirement's property is decisive, and the listing fails if it states something else; a key about another property, even a related one (a handle colour is not the body's colour), settles nothing.
* Without an attribute for that property, the wanted value stated under another attribute or in the title or description qualifies (a number of a different kind does not: a weight is not a capacity; neither does a bare figure whose text does not say what it measures).
* A requirement stated by the category label itself is satisfied by that category; the label must match the requested item as a whole.
* A rule of differing from another listing's value is met unless the records show that same value.
* Every requested item appears as its own cart line at the requested quantity, the cart holds nothing else, nothing is shown out of stock, and price times quantity summed across all lines stays within the budget; a line whose item has its own price cap costs no more than that cap a unit.
* For a cheapest, most or least pick rule, also inspect the other listings shown: if one plainly meets every requirement and beats the cart's choice on that rule, the cart is wrong.
* Set ok=false only for a problem the records plainly show or a requirement no record covers.

Return:
ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}"""


READING_RULES = """an attribute of the listing is decisive for a property when its key names that very property (key spelling does not matter, one list element is enough, synonyms and unit conversions are accepted, a value holding the required value's words qualifies, such as "deep teal" for teal, limits include the edge); a key about another property, even a related one (a handle colour is not the body's colour), settles nothing; if no attribute names that property, the wanted value stated under another attribute or in the title or description qualifies (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label has to match the item; a rule of differing from another listing's value is met unless the records show that same value; listings out of stock, over the budget, or over their item's own price cap do not fit."""

AUDIT_PROMPT = """Mission:
A shopping agent believes nothing fits and wants to end without buying; check the records shown against the requirements.

Materials:
Request, requirements in force, listing records.

Checklist:
* Read records by these rules: """ + READING_RULES + """
* A record marked ROW was seen only as a search result row, so its page (the attributes) is unread: it can neither pass nor fail on a property the row does not show.

Return:
ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

CLOSEST_FIELD = ('"reason": "one sentence", "closest": {"item number": [listing ids the request\'s own rule on nothing'
                 ' fitting says to buy anyway, best first, at most 2]}}')

CLOSEST_CHECK = """
* The requirements quote the request's own rule for when nothing fits. When "fits" is empty for an item and that rule asks the agent to buy, order, put forward or go with some listing anyway (the nearest one, the one with the fewest shortfalls, a substitute), put the listings it points to under "closest", judged by that rule's own conditions. When the rule only lets the agent stop, forbids buying, or asks to check with the shopper first, leave "closest" empty."""

UNMIX_PROMPT = """Mission:
Firm product requirements were drawn from a shopper's request. Some borrow their value from a listing the request refers to, and the words identifying that listing (its kind, its shop, the properties named after it with "with ...") are sometimes misread as requirements on the product being bought.

Materials:
Request, product, requirement lists.

Checklist:
* Using the request's wording, find which OTHER REQUIREMENTS only describe that other listing.

Return:
ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty if all are real requirements on the product]}"""

TRAIL_PROMPT = """Mission:
A shopping agent will check out next. Some requirements get their value from another listing rather than stating it, and a listing's page may pass the reader on to a further listing for that property. The order is valid only if the agent has opened in full each listing along every such chain.

Materials:
The records below (OPENED: the agent opened that page; ROW: known only as a search result row, its page unread).

Checklist:
* Trace each chain from the requirement's own description, link after link, by the pages' own wording.
* Give each chain's links in order, and halt at the first link whose page is not OPENED or cannot be found, because its wording is still unknown.

Return:
ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}"""

GATE_PROMPT = """Mission:
A shopping agent is about to STAGE; the shopper's request demands the steps below along the way. From the numbered RECORD of what the agent did, decide for each step due before this STAGE whether it is complete for the current cart.

Materials:
The sections below.

Checklist:
* A step is complete only if a RECORD line shows it performed and, for a step concerning the cart or the order, that line follows the latest cart change and its reply grants it; questions about a product's properties never complete a step.
* For an open step, name the single store call that performs it next: a TOOLS tool plus its arguments. Shopper messages use the messaging tool: one brief sentence or question about that step only, in plain words, mentioning no product properties or values the step does not need.
* Carry the step out directly: a message asking whether the shopper wants it done does not perform it and only spends a message.
* Name no call when all due steps are complete or the remaining messages or steps cannot achieve the open one.

Return:
ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP_PROMPT = """Mission:
A shopping agent's cart listing just sold out; find that line's replacements among the other records shown.

Materials:
Request, requirements in force, records.

Checklist:
* Judge the records by these rules: """ + READING_RULES + """

Return:
ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number, as the record gives it} for each listing that seems to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE_PROMPT = """Mission:
Map the store tool result below.

Materials:
Tool, parameters, result.

Checklist:
* Identify each part by the exact key the result and tool parameters use.

Return:
ONE JSON object and nothing else: {"rows": the key of the listings list, "count": the key with the total number of matching listings (not the number shown), or null, "next": the key whose value requests the next page, or null, "next_arg": the tool parameter receiving that value, or null, "price": the in-row key for the listing's price, or null, "stock": the in-row key showing if the listing is buyable now, or null}"""


# ---------- json and text parsing ----------

def to_json(value: Any) -> str:
    compact = (",", ":")
    try:
        text = json.dumps(value, ensure_ascii=False, separators=compact)
    except (TypeError, ValueError):
        text = str(value)
    return text


def clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    deeper = depth + 1
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + "..."
        return value
    if isinstance(value, dict):
        return {key: clip(inner, limit, deeper) for key, inner in value.items()}
    if isinstance(value, list):
        return [clip(inner, limit, deeper) for inner in value]
    return value


def parse_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        parsed = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if not 0 <= start < end:
            return None
        try:
            parsed = json.loads(body[start:end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def key_words(key: str) -> str:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    joined = re.sub(r"[\s_.\-]+", " ", camel_split)
    return joined.strip().lower()


def word_stems(text: str) -> frozenset[str]:
    stems = []
    for word in key_words(text).split():
        stems.append(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return frozenset(stems)


def names_property(prop: frozenset[str], key: frozenset[str]) -> bool:
    if prop and key:
        if prop == key or prop <= key:
            return True
        return len(key) >= 2 and key <= prop
    return False


# ---------- schema coercion ----------

def schema_types(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    return list(filter(lambda entry: isinstance(entry, str), declared or []))


def matches_type(value: Any, kind: str) -> bool:
    plain_number = not isinstance(value, bool)
    checks = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and plain_number,
        "number": lambda: isinstance(value, (int, float)) and plain_number,
        "boolean": lambda: isinstance(value, bool),
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
        "null": lambda: value is None,
    }
    test = checks.get(kind)
    return True if test is None else test()


def nudge_toward_kind(value: Any, kind: str) -> tuple[bool, Any]:
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return True, int(value)
    numeric = kind in ("integer", "number")
    if numeric and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        number = float(value)
        if kind == "integer" or number.is_integer():
            return True, int(number)
        return True, number
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, str(value)
    if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return True, value.strip().lower() == "true"
    if kind == "array" and value is not None and not isinstance(value, list):
        return True, [value]
    return False, value


def enum_spelling(value: Any, options: list[Any]) -> Any:
    if not isinstance(value, str):
        return value
    wanted = value.strip().lower()
    for option in options:
        if isinstance(option, str) and option.lower() == wanted:
            return option
    return value


def coerce(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = schema_types(schema)
    if kinds and not any(matches_type(value, kind) for kind in kinds):
        for kind in kinds:
            changed, value = nudge_toward_kind(value, kind)
            if changed:
                break
        if not any(matches_type(value, kind) for kind in kinds):
            faults.append(f"{where} has to be {'/'.join(kinds)}")
            return value

    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        value = enum_spelling(value, options)
        if value not in options:
            faults.append(f"{where}: the valid choices: {options}")

    if isinstance(value, str):
        longest = schema.get("maxLength")
        if isinstance(longest, int) and len(value) > longest:
            value = value[:longest]
        shortest = schema.get("minLength")
        if isinstance(shortest, int) and len(value) < shortest:
            faults.append(f"{where} is too brief")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            faults.append(f"{where} is < min {low}")
        if isinstance(high, (int, float)) and value > high:
            faults.append(f"{where} is > max {high}")

    if isinstance(value, list):
        most, least = schema.get("maxItems"), schema.get("minItems")
        if isinstance(most, int) and len(value) > most:
            faults.append(f"{where} may have up to {most} members")
        if isinstance(least, int) and len(value) < least:
            faults.append(f"{where} takes at least {least} members")
        element = schema.get("items")
        if isinstance(element, dict):
            value = [coerce(entry, element, f"{where}[{pos}]", faults) for pos, entry in enumerate(value)]

    if isinstance(value, dict):
        declared = schema.get("properties")
        known = declared if isinstance(declared, dict) else {}
        if schema.get("additionalProperties") is False:
            for extra in [arg for arg in value if arg not in known]:
                faults.append(f"{where}.{extra} is not among the arguments")
        for needed in schema.get("required") or []:
            if needed not in value:
                faults.append(f"{where}.{needed} must be set")
        fixed = {}
        for arg, inner in value.items():
            fixed[arg] = coerce(inner, known.get(arg), f"{where}.{arg}", faults) if arg in known else inner
        value = fixed
    return value


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def as_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    digits = isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value)
    return int(float(value)) if digits else None


# ---------- listing and id helpers ----------

def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    bare = value.strip()
    if len(bare) < 3 or len(bare) > 64 or " " in bare:
        return False
    if named:
        return True
    return re.search(r"\d", value) is not None


def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    chosen_key = None
    chosen_score = None
    for key in rows[0]:
        id_named = "id" in str(key).lower()
        raw = [row.get(key) for row in rows]
        if not all(looks_like_id(entry, id_named) for entry in raw):
            continue
        cleaned = [entry.strip() for entry in raw]
        if len(rows) > 1 and len(set(cleaned)) < len(cleaned):
            continue
        score = (id_named, sum(len(entry) for entry in cleaned) / len(cleaned))
        if chosen_score is None or score > chosen_score:
            chosen_score, chosen_key = score, key
    return chosen_key


def row_carries_id(row: Any, ident: str) -> bool:
    if not isinstance(row, dict):
        return False
    return ident in [str(field).strip() for field in row.values() if isinstance(field, str)]


def rows_hold_id(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(rows_hold_id(inner, ident, depth + 1) for inner in node.values())
    if not isinstance(node, list):
        return False
    if any(row_carries_id(entry, ident) for entry in node):
        return True
    return any(rows_hold_id(entry, ident, depth + 1) for entry in node)


def scalar_fields(block: dict[str, Any]) -> dict[str, Any]:
    kept = {}
    for key, field in block.items():
        if isinstance(field, bool) or not isinstance(field, (str, int, float)):
            continue
        kept[key] = field
    return kept


def attribute_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    richest = None
    for inner in node.values():
        if not isinstance(inner, dict) or len(inner) < 2:
            continue
        plain = scalar_fields(inner)
        if len(plain) < 2:
            continue
        if richest is None or len(plain) > len(richest):
            richest = plain
    return richest


# ---------- product and state ----------

class Product:
    def __init__(self, name: str, category: str, quantity: int, needs: list[str]):
        self.name = name
        self.category = category
        self.quantity = quantity
        self.needs = needs
        self.price_cap: Any = None
        self.revealed, self.waived, self.open_needs = [], [], []


class State:
    def __init__(self, problem: dict[str, Any]):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.query = str(self.policy.get("query") or "")
        self.tool_specs = [
            spec for spec in self.policy.get("tools") or []
            if isinstance(spec, dict) and isinstance(spec.get("function"), dict)
            and isinstance(spec["function"].get("name"), str)
        ]
        self.arg_schemas = {}
        for spec in self.tool_specs:
            params = spec["function"].get("parameters")
            self.arg_schemas[spec["function"]["name"]] = (
                params if isinstance(params, dict) else {"type": "object", "properties": {}}
            )
        cap = as_count(self.policy.get("max_steps"))
        if cap is None and isinstance(self.policy.get("max_steps"), float):
            cap = int(self.policy["max_steps"])
        self.step_cap = cap if cap and cap > 0 else 30
        self.steps_used = 0
        self.done = bool(self.policy.get("finished"))
        self.out_of_funds = False
        self.llm_broke = False
        self.wait_spent = 0
        self.started = time.monotonic()
        self.empty_replies = 0
        self.spend = 0.0
        self.llm_calls = 0
        self.turns = 0
        self.log: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.role_of: dict[str, str] = {}
        self.wanted: list[Product] = []
        self.budget: Any = None
        self.msg_cap = 0
        self.msgs_sent = 0
        self.procedure: list[dict[str, Any]] = []
        self.msg_reserve = 0
        self.gate_sent = 0
        self.gate_passed: set[tuple[str, str]] = set()
        self.record: list[str] = []
        self.record_steps: list[int] = []
        self.interviewed = False
        self.answers: list[tuple[str, str]] = []
        self.listings: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.seen_ids: list[str] = []
        self.heard_lines: set[str] = set()
        self.store_notes: list[str] = []
        self.needs_version = 0
        self.cart_changed = False
        self.cart_changed_step = -1
        self.cart_viewed_step = -1
        self.cart_snapshot: Any = None
        self.basket: dict[str, int] = {}
        self.approved: set[str] = set()
        self.decline_checked = False
        self.offer_step: dict[str, int] = {}
        self.sold_step: dict[str, int] = {}
        self.blocked_ids: set[str] = set()
        self.sold_ids: set[str] = set()
        self.sold_noted = False
        self.holds = 0
        self.cap_holds = 0
        self.shape_holds = 0
        self.gap_holds = 0
        self.audit_got: dict[str, Any] | None = None
        self.closest_tried = False
        self.audit_ordered = False
        self.id_params: set[tuple[str, str]] = set()
        self.cart_reply = ""
        self.refuse_step = -1
        self.pointers: list[str] = []
        self.refuse_stops = 0
        self.nudged = False
        self.staged: list[str] = []
        self.removed_ids: set[str] = set()
        self.count_checked: set[tuple[str, int]] = set()
        self.late_hold = False
        self.pointer_items: dict[str, list[str]] = {}
        self.verdict_by_cart: dict[str, bool] = {}
        self.confirm_mark: str | None = None
        self.cart_completed = False
        self.row_id_key: str | None = None
        self.shows_facts: dict[str, bool] = {}
        self.queue: list[str] = []
        self.buy_tries = 0
        self.refuse_tries = 0
        self.recent_calls: list[str] = []
        self.messages: list[dict[str, Any]] = []
        self.opened_ids: set[str] = set()
        self.references: list[tuple[int, str]] = []
        self.reference_values: dict[str, str] = {}
        self.trail_runs = 0
        self.trail_stops = 0
        self.compare_hits = 0
        self.compare_misses = 0
        self.keys_by_tool: dict[str, set[str]] = {}
        self.view_seen_keys: dict[str, tuple[int, set[str]]] = {}
        self.page_cache: dict[str, tuple[int, int, str]] = {}
        self.extra_pages = 0
        self.result_shapes: dict[str, dict[str, Any] | None] = {}
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.cheaper_done: set[str] = set()
        self.wider_done = False
        self.page_results: list[Any] = []
        self.llm = None
        self.store = None


# ---------- small state functions ----------

def steps_left(st: State) -> int:
    return st.step_cap - st.steps_used


def free_messages(st: State) -> int:
    held_back = max(0, st.msg_reserve - st.gate_sent)
    return st.msg_cap - st.msgs_sent - held_back


def cart_signature(st: State) -> str:
    ordered_lines = sorted(st.basket.items())
    return to_json(ordered_lines)


def page_view_tools(st: State) -> set[str]:
    views = set()
    for tool in (st.role_of.get("offer") or st.role_of.get("open"), st.role_of.get("compare")):
        if tool:
            views.add(tool)
    return views


def attribute_tool(st: State) -> str | None:
    for tool, shows in st.shows_facts.items():
        if shows and single_id_param(st, tool):
            return tool
    fallback = st.role_of.get("open")
    if st.shows_facts.get(fallback or "") is False:
        return None
    return fallback if single_id_param(st, fallback) else None


def compare_shows_facts(st: State) -> bool | None:
    return st.shows_facts.get(st.role_of.get("compare") or "")


def compare_is_bare(st: State) -> bool:
    seen = compare_shows_facts(st)
    if seen is False:
        return True
    if seen is not None:
        return False
    if st.shows_facts.get(attribute_tool(st) or "") is True:
        return True
    return bool(st.role_of.get("offer"))


def gate_pending(st: State, stage: str) -> bool:
    if not any(step["when"] == stage for step in st.procedure):
        return False
    return (stage, cart_signature(st) + f"#{st.needs_version}") not in st.gate_passed


def finish_reserve(st: State) -> int:
    order_talk = any(step["when"] == "order" and step["messages"] > 0 for step in st.procedure)
    if order_talk and st.msgs_sent < st.msg_cap and gate_pending(st, "order"):
        return 2
    return 1


# ---------- LLM gateway ----------

def retryable_status(status: Any) -> bool:
    if status is None or status in (402, 408, 425, 429):
        return True
    return isinstance(status, int) and status >= 500


def retry_allowance(status: Any) -> int:
    # a timed-out call already used the client's whole timeout, so it gets one more go only
    return 1 if status in (None, 408) else len(RETRY_WAITS)


def mark_cached_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copies = [dict(turn) for turn in messages]
    for index in (0, len(copies) - 1):
        text = copies[index].get("content")
        if isinstance(text, str) and text:
            copies[index]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return copies


def inference_body(model: str, messages: list[dict[str, Any]], max_tokens: int,
                   tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    outgoing = messages
    if model.startswith("anthropic/") and len(messages) > 1:
        outgoing = mark_cached_turns(messages)
    request: dict[str, Any] = dict(model=model, messages=outgoing, max_tokens=max_tokens, temperature=0)
    if tools:
        request.update(tools=tools, tool_choice="auto")
    return request


class Brain:
    def __init__(self, st: State):
        self.st = st

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int,
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        st = self.st
        for model in models:
            if st.llm_calls >= CALL_LIMIT:
                return None
            st.llm_calls += 1
            attempt = 0
            while True:
                reply, status = self._call_once(model, messages, max_tokens, tools)
                if reply is not None:
                    return reply
                if not retryable_status(status) or attempt >= retry_allowance(status):
                    break
                wait = RETRY_WAITS[attempt]
                if st.wait_spent + wait > RETRY_BUDGET:
                    break
                pause = wait * (0.8 + 0.4 * random.random())
                st.wait_spent += wait
                attempt += 1
                time.sleep(pause)
            if status == 402:
                st.llm_broke = True
                return None
        return None

    def _call_once(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
                   tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        st = self.st
        result = _llm.post_verbose("/inference/chat/completions", inference_body(model, messages, max_tokens, tools))
        data = result.data
        if not (isinstance(data, dict) and data.get("choices")):
            failure = result.error or {}
            status = failure.get("status")
            inner = data.get("error") if isinstance(data, dict) else None
            if isinstance(inner, dict):
                # the provider reported its own error code inside an ordinary reply
                status = as_count(inner.get("code"))
                failure = {"body": inner.get("message")}
            st.log.append({"kind": "llm_fail", "model": model, "status": status,
                           "body": str(failure.get("body") or "")[:200]})
            return None, status
        usage = data.get("usage") or {}
        if isinstance(usage.get("cost"), (int, float)):
            st.spend += float(usage["cost"])
        reply = data["choices"][0].get("message") or {}
        if not (reply.get("content") or reply.get("tool_calls")):
            st.log.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        return reply, 200

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        st = self.st
        for model in models:
            single_turn = [{"role": "user", "content": prompt}]
            answer = (self.think(single_turn, [model], max_tokens) or {}).get("content")
            parsed = parse_json_object(answer)
            if parsed is not None:
                return parsed
            stalled = st.out_of_funds or st.llm_broke or st.llm_calls >= CALL_LIMIT
            if stalled:
                break
        return None


# ---------- world gateway ----------

def record_opened_ids(st: State, args: dict[str, Any]) -> None:
    for value in args.values():
        candidates = value if isinstance(value, list) else [value]
        st.opened_ids.update(entry.strip() for entry in candidates if isinstance(entry, str))


class Store:
    def __init__(self, st: State):
        self.st = st

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        st = self.st
        payload = dict(st.binding)
        payload["step"] = st.steps_used + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                return self._accept(name, args, data)
            failure = result.error or {}
            status = failure.get("status")
            body = str(failure.get("body") or "")
            st.log.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    st.out_of_funds = True
                return None
            if status != 409:
                continue
            lowered = body.lower()
            if "finish" in lowered or "quarantin" in lowered or "finaliz" in lowered:
                st.done = True
                return None
            expected = re.search(r"expected (\d+)", body)
            if expected:
                st.steps_used = int(expected.group(1)) - 1
            elif "reuse" in lowered or "different action" in lowered:
                st.steps_used += 1
            else:
                return None
            payload["step"] = st.steps_used + 1
        return None

    def _accept(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        st = self.st
        st.steps_used = data["step"] if isinstance(data.get("step"), int) else st.steps_used + 1
        if data.get("finished"):
            st.done = True
        shown = data.get("observation")
        failed = isinstance(shown, dict) and shown.get("error")
        if name in page_view_tools(st) and not failed:
            record_opened_ids(st, args)
        return data


# ---------- reading the request ----------


def tool_catalog(st) -> str:
    entries = []
    for tool in st.tool_specs:
        spec = tool["function"]
        params = to_json(spec.get("parameters") or {})
        entries.append(f"- tool {spec['name']}; purpose: {spec.get('description', '')}; parameters: {params}")
    return "\n".join(entries)


def request_materials(st) -> str:
    text = (BRIEF_PROMPT + "\nMaterials - store tools:\n" + tool_catalog(st)
            + "\n\nMaterials - shopper request:\n" + st.query)
    opening = st.policy.get("opening")
    if opening is not None:
        text += "\n\nMaterials - opening note from the store:\n" + to_json(clip(opening, 1500))
    return text


def plan_needs_retry(plan: dict) -> bool:
    if not as_list(plan.get("items")):
        return True
    roles = plan.get("roles")
    return not isinstance(roles, dict) or not roles.get("buy")


def settle_checkout_modes(st, plan: dict) -> None:
    buy = st.role_of.get("buy")
    if not buy or buy != st.role_of.get("refuse"):
        return
    buy_args, refuse_args = plan["buy_args"], plan["refuse_args"]
    if buy_args and refuse_args and buy_args != refuse_args:
        return
    question = (f"Store tool {buy} serves two purposes: depending on its arguments it either places the order "
                "or closes the episode with no purchase. Its schema is "
                + to_json(st.arg_schemas.get(buy))
                + "\nReply with ONE JSON object only: {\"buy_args\": the fixed arguments that place the order, "
                "\"refuse_args\": the fixed arguments that end with no purchase} (list just the arguments that "
                "select the mode).")
    answer = st.llm.ask_json(question, [FAST_LLM, STRONG_LLM], 600) or {}
    order_args, stop_args = answer.get("buy_args"), answer.get("refuse_args")
    if isinstance(order_args, dict) and isinstance(stop_args, dict) and order_args != stop_args:
        plan["buy_args"], plan["refuse_args"] = order_args, stop_args


def load_wanted_products(st, plan: dict) -> None:
    for entry in as_list(plan.get("items")):
        if not isinstance(entry, dict):
            continue
        amount = as_count(entry.get("quantity"))
        needs = [str(need) for need in as_list(entry.get("musts")) if need]
        product = Product(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""),
                          amount if (amount or 0) > 0 else 1, needs)
        cap = budget_amount(entry.get("price_cap"))
        if cap is not None and cap > 0:
            product.price_cap = cap
        st.wanted.append(product)
    if not st.wanted:
        st.wanted = [Product("the requested product", "", 1, [])]


def match_lookup_need(st, text: str) -> str:
    if any(text in product.needs for product in st.wanted):
        return text
    # loose copy of a need: fall back to the need on the same property
    prop = key_words(text.split(":")[0])
    for product in st.wanted:
        for need in product.needs:
            if key_words(need.split(":")[0]) == prop:
                return need
    return text


def need_owner(st, need: str) -> int:
    for pos, product in enumerate(st.wanted, 1):
        if need in product.needs:
            return pos
    return 1


def load_lookup_refs(st, plan: dict) -> None:
    for lookup in as_list(plan.get("lookups")):
        need = match_lookup_need(st, str(lookup).strip())
        pair = (need_owner(st, need), need)
        if need and pair not in st.references:
            st.references.append(pair)


def budget_amount(raw: Any) -> Any:
    if isinstance(raw, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", raw):
        return float(raw)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return raw


def load_limits(st, plan: dict) -> None:
    amount = budget_amount(plan.get("budget"))
    if amount is not None and amount > 0:
        st.budget = amount
    cap = as_count(plan.get("messages"))
    if cap is None or cap < 0:
        if st.role_of.get("ask"):
            st.msg_cap = 3
    else:
        st.msg_cap = cap
    steps = as_count(plan.get("steps"))
    if steps is not None and 10 <= steps < st.step_cap:
        st.step_cap = steps


def load_procedure(st, plan: dict) -> None:
    for entry in as_list(plan.get("process")):
        if not isinstance(entry, dict):
            continue
        step = str(entry.get("step") or "").strip()
        if not step:
            continue
        moment = str(entry.get("when") or "").strip().lower()
        st.procedure.append({"step": step,
                             "when": moment if moment in ("choose", "order", "end") else "order",
                             "messages": max(0, as_count(entry.get("messages")) or 0)})
    if not st.role_of.get("ask"):
        return
    spoken = [entry["messages"] for entry in st.procedure if entry["when"] in ("order", "end") and entry["messages"]]
    # a step due before choosing talks to the shopper too: the interview must leave its messages free
    chosen = sum(entry["messages"] for entry in st.procedure if entry["when"] == "choose")
    st.msg_reserve = min((max(spoken) if spoken else 0) + chosen, st.msg_cap)


def parse_request(st) -> None:
    prompt = request_materials(st)
    plan = st.llm.ask_json(prompt, [FAST_LLM], 4000) or {}
    if not st.llm_broke and plan_needs_retry(plan):
        plan = st.llm.ask_json(prompt, [STRONG_LLM, FAST_LLM], 2500) or plan
    st.plan = plan
    raw_roles = plan.get("roles")
    if not isinstance(raw_roles, dict):
        raw_roles = {}
    st.role_of = {role: tool for role, tool in raw_roles.items() if isinstance(tool, str) and tool in st.arg_schemas}
    for field in ("buy_args", "refuse_args"):
        if not isinstance(plan.get(field), dict):
            plan[field] = {}
    settle_checkout_modes(st, plan)
    label_unit_field(st)
    load_wanted_products(st, plan)
    load_lookup_refs(st, plan)
    drop_pointer_needs(st)
    load_limits(st, plan)
    load_procedure(st, plan)


def label_unit_field(st) -> None:
    # the cart tool's count field can carry any name (a "line number", a "slot"); Kyo's own cart calls already read
    # its first integer field as units, so the actor is told the same in the tool's description
    name = st.role_of.get("cart") or ""
    props = (st.arg_schemas.get(name) or {}).get("properties") or {}
    key = schema_first_key_of(props, "integer")
    if not key or any(isinstance(spec, dict) and "array" in schema_types(spec) for spec in props.values()):
        return
    if cart_sets_count(st, name):
        note = f" Its {key} is the number of units of that listing the cart will hold, not a line or position number ({key} = 0 takes the listing out)."
    elif any(isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2
             for spec in props.values()):
        note = f" Its {key} is the number of units of that listing the call puts in or takes out, not a line or position number."
    else:
        note = f" Its {key} is the number of units of that listing to add, not a line or position number."
    for pos, spec in enumerate(st.tool_specs):
        fn = spec["function"]
        if fn["name"] == name:
            st.tool_specs[pos] = {**spec, "function": {**fn, "description": str(fn.get("description") or "") + note}}


def drop_pointer_needs(st) -> None:
    for pos, product in enumerate(st.wanted, 1):
        pointing = [need for owner, need in st.references if owner == pos and need in product.needs]
        others = [need for need in product.needs if need not in pointing]
        if not (pointing and others):
            continue
        reply = st.llm.ask_json(pointer_question(st, product.name, pointing, others), [STRONG_LLM, FAST_LLM], 500) or {}
        flagged = {str(entry).strip() for entry in as_list(reply.get("describe_other"))}
        moved = [need for need in others if need in flagged]
        if moved:
            fold_into_pointer(st, pos, product, pointing[0], moved)


def pointer_question(st, product_name: str, pointing: list[str], others: list[str]) -> str:
    sections = ["Materials - shopper request:\n" + st.query,
                f"Materials - product being bought: {product_name}\n"
                + "Materials - requirements that point at another listing:\n" + "\n".join(pointing)
                + "\nMaterials - OTHER REQUIREMENTS (the ones to judge):\n" + "\n".join(others)]
    return UNMIX_PROMPT + "\n\n" + "\n\n".join(sections)


def fold_into_pointer(st, pos: int, product, anchor: str, moved: list[str]) -> None:
    merged = anchor + " (that listing: " + "; ".join(moved) + ")"
    kept = []
    for need in product.needs:
        if need in moved:
            continue
        kept.append(merged if need == anchor else need)
    product.needs = kept
    st.references = [(owner, merged if (owner, need) == (pos, anchor) else need) for owner, need in st.references]
    st.log.append({"kind": "unmix", "item": pos, "moved": moved})


# ---------- ledger and status ----------


def product_line(st, pos: int, product, several: bool) -> str:
    shown = []
    for need in product.needs + product.revealed:
        if need in product.waived:
            continue
        if need in st.reference_values:
            need += f" (the page it links to states: {st.reference_values[need]})"
        shown.append(need)
    line = f"item {pos}: {product.name}" if several else f"item: {product.name}"
    if product.category:
        line += f" [store shelf: {product.category}]"
    line += f" - wanted: {product.quantity}; must satisfy: "
    line += "; ".join(shown) if shown else "nothing beyond the product type"
    if product.price_cap is not None:
        unit = st.plan.get("budget_unit") or ""
        line += f"; price cap: at most {product.price_cap:g} {unit} a unit".replace("  ", " ")
    if product.open_needs:
        line += "; shopper is open about: " + ", ".join(product.open_needs)
    return line


def plan_lines(st) -> list[str]:
    plan = st.plan
    lines = []
    if st.budget is not None:
        scope = "on every item" if plan.get("budget_scope") == "each" else "combined"
        lines.append(f"cost cap: up to {st.budget} {plan.get('budget_unit') or ''} {scope}"
                     .replace("  ", " "))
    pick = plan.get("pick")
    if pick == "cheapest":
        lines.append("goal: the fitting listing with the lowest price tag")
    elif pick in ("most", "least"):
        lines.append(f"goal: the {pick} {plan.get('pick_measure') or ''} of any fitting listing"
                     .replace("  ", " "))
    else:
        lines.append("goal: any listing that meets all needs")
    if plan.get("in_stock_only", True) is not False:
        lines.append("only in-stock")
    lines += [f"wish (only breaks ties): {wish}" for wish in as_list(plan.get("wishes"))]
    lines += [f"rule the request sets: {rule}" for rule in as_list(plan.get("rules"))]
    if plan.get("quiet_decline") is True:
        lines.append("if nothing fits: stop without naming or proposing any listing")
    if no_fit_rule(st):
        lines.append(f'if nothing fits, the request says (its own words, every condition counts): "{no_fit_rule(st)}"')
    moments = {"choose": "before you pick",
               "order": "at checkout (Kyo carries it out when you buy)",
               "end": "on ending with no purchase (Kyo will carry it out)"}
    lines += [f"step named by the request, {moments[entry['when']]}: {entry['step']}" for entry in st.procedure]
    return lines


def attributes_line(st) -> str | None:
    facts = attribute_tool(st)
    if not facts:
        return None
    offer = st.role_of.get("offer")
    if st.shows_facts.get(facts) is not True and not offer:
        return None
    silent = sorted(name for name in page_view_tools(st) | {offer or ""} if name and st.shows_facts.get(name) is False)
    openers = sorted(name for name in page_view_tools(st) if name != facts)
    line = f"listing attributes appear in {facts}"
    if silent:
        line += f"; {', '.join(silent)} lack them"
    if openers:
        line += "; a listing's page (what a page chain or page check refers to) opens through " + " or ".join(openers)
    return line


def ledger_text(st) -> str:
    several = len(st.wanted) > 1
    lines = [product_line(st, pos, product, several) for pos, product in enumerate(st.wanted, 1)]
    lines += plan_lines(st)
    facts = attributes_line(st)
    if facts is not None:
        lines.append(facts)
    return "\n".join(f"- {line}" for line in lines)


def elapsed(st) -> float:
    return time.monotonic() - st.started


def route(st) -> str:
    # one product, two, or a basket: each shape gets its own handling where the two differ
    count = len(st.wanted)
    return "single" if count <= 1 else "pair" if count == 2 else "basket"


def item_share_note(st) -> str:
    # several items share one step budget: once most of it is gone, an item still without a line gets its best fit
    # so far, so the last steps refine a whole cart instead of hunting for the missing piece
    if route(st) == "single" or st.steps_used < 0.6 * st.step_cap:
        return ""
    missing = [pos for pos in uncovered_items(st) if pos < len(st.wanted)]
    if not missing or len(missing) == len(st.wanted):
        return ""
    named = ", ".join(f"item {pos + 1} ({st.wanted[pos].name})" for pos in missing)
    return (f" {named} still has no cart line: add its best fitting listing seen so far now, then use the steps left"
            " to swap in a better one if it turns up.")


def step_warning(st, left: int) -> str:
    no_buy = st.buy_tries == 0
    if no_buy and elapsed(st) >= CLOCK_WARN:
        return (" TIME IS NEARLY OUT: in this turn put a fitting line for every item in the cart and order;"
                " the episode ends soon whatever the step count.")
    if left <= finish_reserve(st) + 1:
        return (" LAST STEPS: place the order now if the cart has a line for every item; end with no purchase only"
                " if your searches proved some item has no fitting listing (simply being short of steps proves"
                " nothing).")
    if left <= 6 or (no_buy and left <= len(st.wanted) + 5):
        return " Few steps remain: set the cart to the chosen lines, then order now."
    if not no_buy:
        return ""
    if left <= 12:
        return " Steps are getting scarce: decide on your picks, fill the cart, and order within the next few steps."
    share = item_share_note(st)
    if share:
        return share
    if not st.cart_changed and st.steps_used >= st.step_cap // 2:
        return (" Half the steps are gone and the cart is empty: add every item's fitting listing now"
                " (the store may still sell one out).")
    if not st.cart_changed and elapsed(st) >= CLOCK_STOP / 2:
        return (" Half the time is gone and the cart is empty: add each item's best fitting listing found so far"
                " now, then keep checking and swap a line if a better fit turns up.")
    return ""


def status_extras(st) -> list[str]:
    extras = []
    live = {need for _, need in current_needs(st)}
    unresolved = [need for _, need in st.references if need in live and need not in st.reference_values]
    if st.buy_tries == 0 and unresolved:
        extras.append("[look up first: open the listing each of these refers to, plus each listing its page leads"
                      " to; the final page holds the value] " + " | ".join(unresolved[:3]))
    if st.pointers:
        extras.append("[candidates to vet] " + ", ".join(st.pointers))
    if st.store_notes:
        extras.append("[updates so far] " + " | ".join(st.store_notes[-4:]))
    extras.append("[ledger]\n" + ledger_text(st))
    return extras


def status_text(st) -> str:
    left = steps_left(st)
    header = f"[Kyo] turn {st.steps_used} of {st.step_cap} done; {left} more."
    return "\n".join([header + step_warning(st, left)] + status_extras(st))


# ---------- listings ----------


def find_id(st, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
    if not isinstance(node, dict) or len(node) < 3:
        return None
    known_key = st.row_id_key
    if known_key and looks_like_id(node.get(known_key), True):
        return node[known_key].strip()
    holds_dict = any(isinstance(value, dict) for value in node.values())
    if not in_list and depth != 0 and not holds_dict:
        return None
    guessed = guess_id_key([node])
    if not guessed:
        return None
    value = node[guessed].strip()
    return value if "id" in str(guessed).lower() or value in st.listings else None


def learn_listing_rows(st, node: list[Any]) -> bool:
    records = [entry for entry in node if isinstance(entry, dict) and len(entry) >= 3]
    if not records or 2 * len(records) < len(node):
        return False
    known_key = st.row_id_key
    if not known_key or not all(looks_like_id(record.get(known_key), True) for record in records):
        guessed = guess_id_key(records)
        if not guessed:
            return False
        st.row_id_key = guessed
    return True


def merge_listing(st, ident: str, node: dict) -> None:
    stored = st.listings.get(ident)
    if stored is None:
        st.seen_ids.append(ident)
        stored = st.listings[ident] = {}
    flips = [(stored.get(key), value) for key, value in node.items()]
    # a flag flipping true -> false means the listing can no longer be bought
    if any(before is True and after is False for before, after in flips):
        st.blocked_ids.add(ident)
    elif any(before is False and after is True for before, after in flips):
        st.blocked_ids.discard(ident)
    stored.update(node)
    full_page = st.pages.get(ident)
    if attribute_block(node):
        st.pages[ident] = dict(stored)
    elif full_page is not None:
        full_page.update(node)


def absorb_node(st, node: Any, depth: int, in_list: bool = False) -> None:
    if depth > 6:
        return
    if isinstance(node, list):
        row_list = learn_listing_rows(st, node)
        for child in node:
            absorb_node(st, child, depth + 1, row_list)
        return
    if not isinstance(node, dict):
        return
    ident = find_id(st, node, in_list, depth)
    if ident:
        merge_listing(st, ident, node)
        return
    for child in node.values():
        absorb_node(st, child, depth + 1)


def absorb_result(st, observation: Any) -> None:
    absorb_node(st, observation, 0)


def collect_enum_labels(schema: Any, depth: int, found: set[str]) -> None:
    if depth > 6 or not isinstance(schema, dict):
        return
    found.update(option.strip().lower() for option in schema.get("enum") or [] if isinstance(option, str))
    for child in schema.values():
        if isinstance(child, dict):
            collect_enum_labels(child, depth + 1, found)


def known_labels(st) -> set[str]:
    found: set[str] = set()
    for schema in st.arg_schemas.values():
        collect_enum_labels(schema, 0, found)
    return found


def category_field(st) -> str | None:
    distinct: dict[str, set[str]] = {}
    hits: dict[str, int] = {}
    for record in st.listings.values():
        for key, value in record.items():
            if not isinstance(value, str) or not 0 < len(value) <= 40 or re.search(r"\d{3}", value):
                continue
            distinct.setdefault(key, set()).add(value)
            hits[key] = hits.get(key, 0) + 1
    floor = 0.8 * max(1, len(st.listings))
    labels = known_labels(st)
    ranked = []
    for key, values in distinct.items():
        if hits[key] < floor or len(values) > 40:
            continue
        share = sum(1 for value in values if value.strip().lower() in labels) / len(values)
        average = sum(len(value) for value in values) / len(values)
        ranked.append(((-share, len(values), -hits[key], -average), key))
    if not ranked:
        return None
    return min(ranked, key=lambda pair: pair[0])[1]


def product_match_keys(st) -> list[tuple[str, str, set[str]]]:
    keys = []
    for product in st.wanted:
        stems = word_stems(product.category or "") | word_stems(product.name)
        keys.append((key_words(product.category or ""), key_words(product.name),
                     {stem for stem in stems if len(stem) >= 4}))
    return keys


def label_matches(label: str, match_keys: list[tuple[str, str, set[str]]]) -> bool:
    name = key_words(label)
    long_stems = {stem for stem in word_stems(label) if len(stem) >= 4}
    return any(name in (shelf, title) or long_stems & stems for shelf, title, stems in match_keys)


def tally_attribute_keys(members: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for ident, block in members:
        firsts: dict[str, str] = {}
        for key, value in block.items():
            name = key_words(key)
            if name and name not in firsts:
                firsts[name] = str(value).strip().lower()
        for name, text in firsts.items():
            entry = stats.setdefault(name, {"count": 0, "values": [], "per": {}})
            entry["count"] += 1
            entry["per"][ident] = text
            if text not in entry["values"]:
                entry["values"].append(text)
    return stats


def group_attribute_keys(st) -> dict[str, dict[str, Any]]:
    cat = category_field(st)
    members: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for ident in st.seen_ids:
        record = st.pages.get(ident)
        if record:
            label = str(record.get(cat) or "listings") if cat else "listings"
            members.setdefault(label, []).append((ident, attribute_block(record) or {}))
    return {label: {"count": len(rows), "keys": tally_attribute_keys(rows)} for label, rows in members.items()}


def noisy_values(texts: list[str]) -> bool:
    size = len(texts)
    average = sum(map(len, texts)) / size
    if average > 40:
        return True
    if sum(" / " in text for text in texts) * 2 >= size:
        return True
    return size >= 4 and average > 12 and len(set(texts)) == size


def useful_keys(keys: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    kept: list[tuple[str, dict[str, Any]]] = []
    ranked = sorted(keys.items(), key=lambda pair: -pair[1]["count"])
    for name, entry in ranked:
        if noisy_values(list(entry["per"].values())):
            continue
        twin = any(name.startswith(other + " ") and entry["per"] == spec["per"] for other, spec in kept)
        if not twin:
            kept.append((name, entry))
    return kept


def attribute_table(st) -> list[tuple[str, str, int, list[str]]]:
    groups = group_attribute_keys(st)
    match_keys = product_match_keys(st)
    relevant = {label: group for label, group in groups.items() if label_matches(label, match_keys)}
    if relevant:
        groups = relevant
    table = []
    for label, group in sorted(groups.items(), key=lambda pair: -pair[1]["count"]):
        table += [(label, name, entry["count"], [text[:40] for text in entry["values"][:4]])
                  for name, entry in useful_keys(group["keys"])[:14]]
    return table


def grouped_ids(st) -> list[list[str]]:
    cat = category_field(st)
    by_label: dict[str, list[str]] = {}
    for ident in st.seen_ids:
        label = str(st.listings[ident].get(cat) or "") if cat else ""
        by_label.setdefault(label, []).append(ident)
    match_keys = product_match_keys(st)
    pool = [ids for label, ids in by_label.items() if label_matches(label, match_keys)] or list(by_label.values())
    pool.sort(key=lambda ids: -len(ids))
    return pool[:max(1, len(st.wanted))]


def single_id_param(st, tool: str | None) -> str | None:
    if not tool:
        return None
    schema = st.arg_schemas.get(tool) or {}
    required = schema.get("required") or []
    free_text = [key for key, spec in (schema.get("properties") or {}).items()
                 if isinstance(spec, dict) and "string" in schema_types(spec) and "enum" not in spec]
    target = next((key for key in required if key in free_text), None)
    if not target and len(free_text) == 1:
        target = free_text[0]
    if not target or any(key != target for key in required):
        return None
    return target


def listing_price(st, ident: str) -> float | None:
    field = st.price_field
    raw = (st.listings.get(ident) or {}).get(field) if field else None
    if isinstance(raw, dict):
        numbers = [inner for inner in raw.values() if isinstance(inner, (int, float)) and not isinstance(inner, bool)]
        raw = numbers[0] if numbers else None
    if isinstance(raw, str):
        return float(raw) if re.fullmatch(r"\s*\d+(\.\d+)?\s*", raw) else None
    return None if isinstance(raw, bool) or not isinstance(raw, (int, float)) else float(raw)


def in_stock(st, ident: str) -> bool:
    if is_sold_out(st, ident):
        return False
    if not st.stock_field:
        return True
    flag = (st.listings.get(ident) or {}).get(st.stock_field)
    if flag is False:
        return False
    counted = isinstance(flag, (int, float)) and not isinstance(flag, bool)
    return not (counted and flag <= 0)


def rows_with_ids(st, rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    pairs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ident = find_id(st, row, True)
        if ident:
            pairs.append((ident, row))
    return pairs


def listing_records(st, observation: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(observation, list):
        return rows_with_ids(st, observation)
    if not isinstance(observation, dict):
        return []
    ident = find_id(st, observation)
    if ident:
        return [(ident, observation)]
    found: list[tuple[str, dict[str, Any]]] = []
    for value in observation.values():
        if isinstance(value, list):
            found += rows_with_ids(st, value)
            continue
        if isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, list):
                    found += rows_with_ids(st, inner)
    return found


def remember_view(st, name: str, observation: Any) -> None:
    version = st.needs_version
    for ident, record in listing_records(st, observation):
        fields = set(record)
        shown = st.keys_by_tool.get(name)
        if shown is None:
            shown = st.keys_by_tool[name] = set()
        shown |= fields
        earlier = st.view_seen_keys.get(ident)
        carried = earlier[1] if earlier is not None and earlier[0] == version else set()
        st.view_seen_keys[ident] = (version, carried | fields)


def already_shown(st, name: str, ident: str) -> bool:
    shows = st.keys_by_tool.get(name)
    if not shows:
        return False
    seen_version, seen_fields = st.view_seen_keys.get(ident, (-1, set()))
    return seen_version == st.needs_version and shows <= seen_fields and ident not in st.sold_step


# ---------- result shapes and paging ----------


def key_if_present(key: Any, where: dict[str, Any]) -> str | None:
    if isinstance(key, str) and key in where:
        return key
    return None


def row_list_keys(observation: Any) -> list[str]:
    if not isinstance(observation, dict):
        return []
    found = []
    for key, value in observation.items():
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(row, dict) for row in value):
            found.append(key)
    return found


def shape_question(name: str, schema: dict, observation: dict, row_lists: list[str]) -> str:
    preview = {}
    for key, value in observation.items():
        preview[key] = value[:3] if key in row_lists else value
    return (SHAPE_PROMPT + "\n\nMaterials - tool: " + name + "; its parameters: " + to_json(schema)
            + "\n\nMaterials - result (only its first rows):\n" + to_json(clip(preview, 120))[:4000])


def shape_from_reply(reply: dict, observation: dict, params: dict) -> dict[str, Any]:
    rows_key = reply["rows"]
    first = observation[rows_key][0]
    shape: dict[str, Any] = {"rows": rows_key}
    for part, holder in (("count", observation), ("next", observation), ("next_arg", params),
                         ("price", first), ("stock", first)):
        shape[part] = key_if_present(reply.get(part), holder)
    counter = shape["count"]
    if counter:
        total = observation[counter]
        if isinstance(total, bool) or not isinstance(total, int):
            shape["count"] = None
    return shape


def result_shape(st, name: str, observation: Any) -> dict[str, Any] | None:
    if name in st.result_shapes:
        return st.result_shapes[name]
    row_lists = row_list_keys(observation)
    if not row_lists:
        return None
    st.result_shapes[name] = None
    schema = st.arg_schemas.get(name) or {}
    question = shape_question(name, schema, observation, row_lists)
    reply = st.llm.ask_json(question, [FAST_LLM, STRONG_LLM], 1500) or {}
    picked = reply.get("rows")
    if not picked or picked not in row_lists:
        return None
    shape = st.result_shapes[name] = shape_from_reply(reply, observation, schema.get("properties") or {})
    if not st.price_field:
        st.price_field = shape["price"]
    if not st.stock_field:
        st.stock_field = shape["stock"]
    st.log.append({"kind": "shape", "tool": name, "shape": shape})
    return shape


def page_info(st, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
    shape = result_shape(st, name, observation)
    usable = bool(shape) and all(shape[part] for part in ("count", "next", "next_arg"))
    if not usable or not isinstance(observation, dict):
        return None
    rows, total, cursor = (observation.get(shape[part]) for part in ("rows", "count", "next"))
    rows_ok = isinstance(rows, list) and len(rows) > 0
    total_ok = isinstance(total, int) and not isinstance(total, bool)
    cursor_ok = cursor is None or (isinstance(cursor, (str, int)) and not isinstance(cursor, bool))
    if rows_ok and total_ok and cursor_ok:
        return total, len(rows), shape["next_arg"], cursor
    return None


def sorted_by_price(st, observation: Any) -> bool:
    prices = [listing_price(st, rid) for rid, _ in listing_records(st, observation)]
    if len(prices) < 3 or None in prices:
        return False
    return all(low <= high for low, high in zip(prices, prices[1:]))


def ceil_pages_left(total: int, shown: int) -> int:
    # pages still unread after the first one, rounded up
    return (total - 1) // shown


def given_page_is_later(given: Any) -> bool:
    if isinstance(given, int) and not isinstance(given, bool):
        return given not in (None, 0, 1)
    return bool(str(given or "").strip())


def fetch_following_pages(st, name: str, args: dict[str, Any], param: str, cursor: Any, budget: int) -> list[str]:
    chunks: list[str] = []
    while budget > 0:
        budget -= 1
        problems: list[str] = []
        page_args = coerce({**args, param: cursor}, st.arg_schemas.get(name), "args", problems)
        reply = None if problems else st.store.act(name, page_args)
        if reply is None or st.done:
            break
        st.extra_pages += 1
        result = reply.get("observation")
        observe_result(st, name, result)
        st.page_results.append(result)
        rendered = show_result(st, result)
        st.page_cache[name + to_json(page_args)] = (st.needs_version, st.steps_used, rendered)
        chunks.append(f"{param}={to_json(cursor)}: {rendered}")
        following = page_info(st, name, result)
        cursor = following[3] if following else None
        if cursor is None:
            break
    return chunks


def read_more_pages(st, name: str, args: dict[str, Any], observation: Any) -> str:
    st.page_results = []
    pick = st.plan.get("pick")
    if pick not in ("cheapest", "most", "least"):
        return ""
    paging = page_info(st, name, observation)
    if not paging:
        return ""
    total, shown, param, cursor = paging
    agent_pages = given_page_is_later(args.get(param))
    if cursor is None or agent_pages or total <= shown:
        return ""
    if pick == "cheapest" and sorted_by_price(st, observation):
        return ""
    remaining = ceil_pages_left(total, shown)
    if (remaining > 2 or st.extra_pages + remaining > EXTRA_PAGES
            or steps_left(st) - finish_reserve(st) - remaining < 12):
        # same segment lengths as the base note, so the cached-text cut keeps the same data
        return (f"\n[Kyo] Matches for search: {total}; listings in reply: {shown}, unsorted by price. "
                f"The pick weighs all fitting listings: use narrower words (the requirement's value) "
                f"or get the next page via {param}={to_json(cursor)}.")
    chunks = fetch_following_pages(st, name, args, param, cursor, remaining)
    st.log.append({"kind": "pages", "tool": name, "read": len(chunks)})
    if chunks:
        return "\n[Kyo] Rest of this search, fetched in code, a step a page:\n" + "\n".join(chunks)
    return ""


def cheapest_note(st, observations: list[Any]) -> str:
    if st.plan.get("pick") != "cheapest" or not st.price_field:
        return ""
    ids = {rid for result in observations for rid, _ in listing_records(st, result)}
    priced = []
    for rid in ids:
        if in_stock(st, rid) and listing_price(st, rid) is not None:
            priced.append((listing_price(st, rid), rid))
    if len(priced) < 2:
        return ""
    cheapest = sorted(priced)[:6]
    return "\n[Kyo] In-stock rows, lowest price first: " + ", ".join(
        f"{rid} ({price:g})" for price, rid in cheapest)


def hold_wider(st) -> str:
    if st.wider_done or st.plan.get("pick") != "cheapest":
        return ""
    if steps_left(st) < 14 + 2 * len(st.wanted):
        return ""
    st.wider_done = True
    cart_prices = []
    for rid in cart_listing_ids(st):
        price = listing_price(st, rid)
        if price is not None:
            cart_prices.append(f"{rid} at {price:g}")
    price_part = f" ({'; '.join(cart_prices)})" if cart_prices else ""
    return (f"Paused, no step spent. With {steps_left(st)} steps remaining and a cheapest pick: store search reads"
            " only titles, so a requirement's value (usually listed only among the attributes) turns up few of the"
            " fitting listings, and rows arrive unsorted by price. Before the checkout, search again for every item"
            " using other names for the product (its simple one-word name, synonyms, words from fitting titles"
            " already seen), watching only for in-stock rows cheaper than the cart's line" + price_part
            + "; read the attributes of the cheapest of those, swap in whichever fits, then check out.")


def attribute_page_text(st, tool: str, param: str, rid: str) -> str | None:
    problems: list[str] = []
    call = coerce({param: rid}, st.arg_schemas.get(tool), "args", problems)
    if problems:
        return None
    reply = st.store.act(tool, call)
    if reply is None:
        return None
    result = reply.get("observation")
    observe_result(st, tool, result)
    remember_view(st, tool, result)
    if isinstance(result, dict) and find_id(st, result):
        st.shows_facts[tool] = st.shows_facts.get(tool, False) or bool(attribute_block(result))
    return f"{rid}: " + show_result(st, result)[:1500]


def fetch_attributes(st, ids: list[str], keep: int) -> list[str]:
    tool = attribute_tool(st)
    param = single_id_param(st, tool)
    if not (tool and param):
        return []
    unread = [rid for rid in ids if rid in st.listings and rid not in st.pages and not is_sold_out(st, rid)]
    collected: list[str] = []
    for rid in unread:
        halt = steps_left(st) <= keep or st.done or st.shows_facts.get(tool) is False
        text = None if halt else attribute_page_text(st, tool, param, rid)
        if text is None:
            break
        collected.append(text)
    if collected:
        st.log.append({"kind": "fill_facts", "n": len(collected)})
    return collected


def listing_kind_text(st, rid: str, key: str) -> str:
    row = st.listings.get(rid) or {}
    value = row.get(key) or ""
    return str(value).strip().lower()


def cheaper_candidates(st) -> list[str]:
    if st.plan.get("pick") != "cheapest":
        return []
    signature = cart_signature(st) + f"#{st.needs_version}"
    kind_key = category_field(st)
    if not kind_key or signature in st.cheaper_done:
        return []
    st.cheaper_done.add(signature)
    in_cart = cart_listing_ids(st)
    chosen: list[str] = []
    for line_id in in_cart:
        ceiling = listing_price(st, line_id)
        kind = listing_kind_text(st, line_id, kind_key)
        if ceiling is None or not kind:
            continue
        rivals = []
        for rid in st.seen_ids:
            if rid in in_cart or rid in st.pages or not in_stock(st, rid):
                continue
            if listing_kind_text(st, rid, kind_key) != kind:
                continue
            price = listing_price(st, rid) if listing_price(st, rid) is not None else float("inf")
            if price < ceiling:
                rivals.append(rid)
        rivals.sort(key=lambda rid: listing_price(st, rid) or 0.0)
        fresh = [rid for rid in rivals[:2] if rid not in chosen]
        chosen.extend(fresh)
    return fetch_attributes(st, chosen[:CHEAPER_LOOKS], finish_reserve(st) + 4)


def observe_result(st, name: str, observation: Any) -> str:
    absorb_result(st, observation)
    lines = stray_lines(st, observation)
    if lines:
        change = apply_note(st, lines)
        if change:
            st.queue.append(f"(on call {name}) {change}")
        return change
    return ""


def best_array_param(props: dict[str, Any]) -> str | None:
    for key, spec in props.items():
        if isinstance(spec, dict) and "array" in schema_types(spec):
            return key
    return None


def compare_batch(st, limit: int) -> list[str]:
    groups = grouped_ids(st)
    per_group = max(1, limit // max(1, len(groups)))
    chosen: list[str] = []
    for group in groups:
        chosen.extend([rid for rid in group if rid not in st.pages][:per_group])
    return chosen[:limit]


def compare_view_records(st, tool: str, result: Any) -> list[Any]:
    observe_result(st, tool, result)
    remember_view(st, tool, result)
    records = listing_records(st, result)
    if records:
        shown_facts = st.shows_facts.get(tool, False) or any(attribute_block(rec) for _, rec in records)
        st.shows_facts[tool] = shown_facts
    return records


def retry_compare_pair(st, tool: str, schema: dict[str, Any], param: str, pair: list[str]) -> None:
    problems: list[str] = []
    call = coerce({param: pair}, schema, "args", problems)
    if problems:
        return
    reply = st.store.act(tool, call)
    if reply is not None:
        observe_result(st, tool, reply.get("observation"))


# ---------- widening the search ----------


def widen_search(st) -> None:
    tool = st.role_of.get("compare")
    schema = st.arg_schemas.get(tool or "") or {}
    props = schema.get("properties") or {}
    param = best_array_param(props)
    usable = bool(tool) and bool(param) and all(key == param for key in schema.get("required") or [])
    if not usable or compare_is_bare(st):
        widen_each(st)
        return
    if steps_left(st) <= 12:
        return
    chosen = compare_batch(st, compare_capacity(st))
    if len(chosen) < 2:
        return
    problems: list[str] = []
    call = coerce({param: chosen}, schema, "args", problems)
    if problems:
        return
    opened_before = len(st.pages)
    reply = st.store.act(tool, call)
    if reply is None:
        return
    records = compare_view_records(st, tool, reply.get("observation"))
    st.log.append({"kind": "widen", "ids": chosen, "opened": len(st.pages) - opened_before})
    if records and st.shows_facts.get(tool) is False:
        widen_each(st)
    elif len(st.pages) == opened_before and len(chosen) > 2 and steps_left(st) > 12:
        retry_compare_pair(st, tool, schema, param, chosen[:2])


def widen_each(st) -> None:
    tool = attribute_tool(st)
    param = single_id_param(st, tool)
    if not tool or not param:
        return
    lanes: list[list[str]] = []
    for group in grouped_ids(st):
        already = len([rid for rid in group if rid in st.pages])
        pool = [rid for rid in group if rid not in st.pages and in_stock(st, rid)]
        if st.plan.get("pick") == "cheapest":
            pool.sort(key=lambda rid: listing_price(st, rid) if listing_price(st, rid) is not None else float("inf"))
        lanes.append(pool[:max(0, min(2, 8 - already))])
    order: list[str] = []
    for slot in (0, 1):
        order.extend(lane[slot] for lane in lanes if slot < len(lane))
    fetch_attributes(st, order[:5], 22)


# ---------- asking the shopper ----------


def should_interview(st) -> bool:
    blocked = st.interviewed or not st.role_of.get("ask") or free_messages(st) <= 0 or steps_left(st) <= 8
    if blocked or elapsed(st) >= CLOCK_LATE:
        return False
    if len(st.pages) >= 3 * len(st.wanted) + 1 or elapsed(st) >= CLOCK_ASK:
        return True
    return st.steps_used >= max(8, st.step_cap // 4)


def message_arguments(st, text: str) -> dict[str, Any] | None:
    schema = st.arg_schemas.get(st.role_of.get("ask") or "") or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    text_keys = [key for key, spec in props.items()
                 if isinstance(spec, dict) and "string" in schema_types(spec) and "enum" not in spec]
    target = next((key for key in required if key in text_keys), None)
    if not target:
        target = text_keys[0] if text_keys else None
    if not target:
        return None
    call: dict[str, Any] = {target: text}
    for key in required:
        if key in call:
            continue
        choices = (props.get(key) or {}).get("enum")
        if not (isinstance(choices, list) and choices):
            return None
        call[key] = choices[0]
    problems: list[str] = []
    call = coerce(call, schema, "args", problems)
    if problems:
        return None
    return call


def interview_settled_lines(st) -> list[str]:
    return [f"item {index}: {product.name}; fixed already by the request: " + ("; ".join(product.needs) or "nothing")
            for index, product in enumerate(st.wanted, 1)]


def interview_key_lines(table: list[Any]) -> list[str]:
    out: list[str] = []
    groups_of: dict[str, set] = {}
    for group, key_name, _, _ in table:
        groups_of.setdefault(key_name, set()).add(group)
    for position, row in enumerate(table, 1):
        group, key_name, carried, values = row
        if position == 1 or group != table[position - 2][0]:
            out.append(f"category {group!r}:")
        shared = " [shared]" if len(groups_of[key_name]) > 1 else ""
        out.append(f"  {position}. {key_name}{shared} (on {carried}): " + " | ".join(values))
    return out


def first_property(raw: Any) -> str:
    # a label naming two properties keeps only the first one
    head = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(raw or ""))[0]
    return head.strip().rstrip("?.")


def spelling_fold(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(re.sub(r"ise$", "ize", re.sub(r"(?<=[a-z]{2})re$", "er", re.sub(r"(?<=[a-z])our$", "or", w)))
                    for w in words)


def store_spelling(label: str, keys: list[str]) -> str:
    # some shoppers match the asked property by its spelling: a label that differs from a listed key only in
    # British or American spelling ("colour" for color) takes the key's own words, and a lone word the keys spell
    # the other way takes their spelling
    folded = spelling_fold(label)
    if folded == label.lower():
        return label
    exact = next((key for key in keys if spelling_fold(key) == folded), None)
    if exact:
        return exact
    words = {word for key in keys for word in key.lower().split()}
    swapped = [new if old != new and new in words else old for old, new in zip(label.lower().split(), folded.split())]
    return " ".join(swapped) if len(swapped) == len(label.split()) else label


def interview_picks(st, plan: dict[str, Any]) -> list[tuple[int, str]]:
    taken: set[tuple[int, str]] = set()
    picks: list[tuple[int, str]] = []
    count = len(st.wanted)
    keys = [name for _, name, _, _ in attribute_table(st)]
    for entry in as_list(plan.get("ask")):
        if not isinstance(entry, dict):
            continue
        number = min(max(1, as_count(entry.get("item") or 1) or 1), count)
        label = store_spelling(first_property(entry.get("label")), keys)
        if label and len(label.split()) <= 5 and (number, key_words(label)) not in taken:
            taken.add((number, key_words(label)))
            picks.append((number, label))
    return picks


def interview_prompt(st, budget: int, asked: list[str] | None = None) -> str:
    table = attribute_table(st)
    sections = [
        PLAN_PROMPT.replace("COUNT", str(budget + len(st.wanted))),
        "Materials - the shopper's request:\n" + st.query,
        "Materials - items and what the request fixes for each:\n" + "\n".join(interview_settled_lines(st)),
    ]
    first = [str(x) for x in as_list(st.plan.get("ask_first")) if str(x).strip()]
    if first:
        sections.append("Materials - ASK FIRST (the request wants the shopper asked about these):\n" + "\n".join(first))
    keys = "\n".join(interview_key_lines(table)) or "(no listing has been opened in full so far)"
    sections.append("Materials - attribute keys seen on opened listings:\n" + keys)
    if asked:
        sections.append("Materials - already asked or answered (never pick these again, under any name):\n"
                        + "\n".join(asked))
    if len(st.wanted) > 1:
        sections.append(
            "Several items share the few messages, so each item gets one or two questions. For each item, rank its"
            " own listed keys first, and a key marked [shared] (carried by listings of more than one item) after the"
            " item's other keys; a story hint still comes first, and a property no key lists only when the story"
            " points to it.")
    return "\n\n".join(sections)


def interview_summary(st, sent: list[tuple[int, str, str]], told: list[str]) -> str:
    parts = ["The interviewer asked the shopper these questions (one step each):"]
    parts.extend(f"Q: {q}\nA: {a}" for _, q, a in sent)
    added = "; ".join(told) if told else "none"
    text = "\n".join(parts) + f"\nFirm requirements added by these answers: {added}."
    text += " Hold every pick to them and search again with the new values when needed."
    linked = [need for _, need in st.references if any(need in t for t in told)]
    if linked:
        text += (" These values are taken from another listing: " + " | ".join(linked) + ". Locate that listing,"
                 " open its full page, then open each listing that page points to; the final page holds the value,"
                 " and the order only counts once all of them are opened.")
    if st.queue:
        text += "\n[notes received since] " + " | ".join(st.queue)
        st.queue = []
    return text


def next_pick(st, pool: list[tuple[int, str]], asked: set[str], revealed: set[int],
              asked_per_item: dict[int, int]) -> tuple[int, str] | None:
    listed = {key_words(name) for _, name, _, _ in attribute_table(st)}
    ask_first = [key_words(str(x)) for x in as_list(st.plan.get("ask_first")) if str(x).strip()]
    # a label already spent (or a spelling of it) never revealed anything; items with no reveal yet go first
    def spent(label: str) -> bool:
        name = key_words(label)
        return any(name == old or difflib.SequenceMatcher(None, name, old).ratio() >= 0.9 for old in asked)
    fresh = [(rank, pick) for rank, pick in enumerate(pool) if not spent(pick[1])]
    if not fresh:
        return None
    # a property no opened listing names rarely turns out to be the hidden one; it waits behind the listed keys of
    # the same item unless the request itself asks for it
    def unlisted(label: str) -> bool:
        name = key_words(label)
        return bool(listed) and name not in listed and not any(name in first for first in ask_first)
    ranked = min(fresh, key=lambda pair: (pair[1][0] in revealed, asked_per_item.get(pair[1][0], 0),
                                          unlisted(pair[1][1]), pair[0]))
    return ranked[1]


def run_interview(st) -> str:
    st.interviewed = True
    budget = free_messages(st)
    if budget <= 0:
        return ""
    widen_search(st)
    # the planner ranks one pick per item beyond the message budget; questions then go one at a time and each
    # answer steers the next pick
    plan = st.llm.ask_json(interview_prompt(st, budget), [STRONG_LLM, FAST_LLM], 1800) or {}
    pool = interview_picks(st, plan)
    st.log.append({"kind": "plan", "ask": clip(plan.get("ask"), 120)})
    told: list[str] = []
    sent: list[tuple[int, str, str]] = []
    asked: set[str] = set()
    revealed: set[int] = set()
    asked_per_item: dict[int, int] = {}
    replanned = False
    while free_messages(st) > 0:
        pick = next_pick(st, pool, asked, revealed, asked_per_item)
        if pick is None and not replanned:
            # the ranked picks ran out with messages left: one more plan that knows what was asked
            replanned = True
            done = [f"item {n}: {question}" for n, question, _ in sent] + [f"(property) {name}" for name in sorted(asked)]
            more = st.llm.ask_json(interview_prompt(st, free_messages(st), done), [STRONG_LLM, FAST_LLM], 1800) or {}
            pool = interview_picks(st, more)
            st.log.append({"kind": "replan", "ask": clip(more.get("ask"), 120)})
            pick = next_pick(st, pool, asked, revealed, asked_per_item)
        if pick is None:
            break
        pool.remove(pick)
        number, label = pick
        delivered = send_questions(st, [(number, label, question_text(st, number, label))])
        if not delivered:
            break
        sent += delivered
        asked.add(key_words(label))
        asked_per_item[number] = asked_per_item.get(number, 0) + 1
        before = [len(product.revealed) for product in st.wanted]
        heard = read_answers(st, delivered, (FAST_LLM, STRONG_LLM))
        heard["needs"] = drop_misnamed_needs(st, as_list(heard.get("needs")), delivered)
        apply_answers(st, heard, told)
        for pos, product in enumerate(st.wanted, 1):
            for need in product.revealed[before[pos - 1]:]:
                revealed.add(pos)
                asked.add(key_words(need.split(":")[0]))
    if not sent:
        return ""
    if told:
        st.needs_version += 1
    return interview_summary(st, sent, told)


def question_text(st, number: int, label: str) -> str:
    return f"What {label} would you like for the {st.wanted[number - 1].name}?"


def log_question(st, text: str, words: str) -> None:
    st.record.append(f"step {st.steps_used}: asked the shopper {text!r}, who replied {words[:300]!r}"
                     f" (cart at that point: {cart_signature(st)})")
    st.record_steps.append(st.steps_used)
    st.log.append({"kind": "ask", "q": text, "a": words[:300]})


def send_questions(st, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
    tool = st.role_of["ask"]
    delivered: list[tuple[int, str, str]] = []
    for number, _, text in questions:
        room = not st.done and steps_left(st) > floor and st.msgs_sent < st.msg_cap
        call = message_arguments(st, text) if room else None
        reply = st.store.act(tool, call) if call else None
        if reply is None:
            break
        st.msgs_sent += 1
        answer = reply.get("observation")
        spoken = shopper_words(st, answer)
        delivered.append((number, text, spoken))
        log_question(st, text, spoken)
        failed = isinstance(answer, dict) and answer.get("error")
        if failed:
            break
    return delivered


def answer_item_number(st, raw: Any) -> int:
    return min(max(1, as_count(raw) or 1), len(st.wanted))


def absorb_stated_needs(st, entries: list[Any], told: list[str]) -> None:
    several = len(st.wanted) > 1
    for entry in entries:
        if not (isinstance(entry, dict) and str(entry.get("need") or "").strip()):
            continue
        number = answer_item_number(st, entry.get("item"))
        need = str(entry["need"]).strip()
        product = st.wanted[number - 1]
        # a later answer restating a property the shopper already settled adds nothing (a lone re-read can lose
        # its wording, such as an "anything other than")
        settled = {key_words(old.split(":")[0]) for old in product.revealed if old not in product.waived}
        if key_words(need.split(":")[0]) in settled:
            continue
        if need not in product.revealed:
            product.revealed.append(need)
            told.append(f"item {number}: {need}" if several else need)
        if entry.get("points") is True and (number, need) not in st.references:
            st.references.append((number, need))


def absorb_dropped_needs(st, entries: list[Any], told: list[str]) -> None:
    for entry in entries:
        if not (isinstance(entry, dict) and str(entry.get("property") or "").strip()):
            continue
        product = st.wanted[answer_item_number(st, entry.get("item")) - 1]
        given_up = key_words(str(entry["property"]).split(":")[0])
        matches = [need for need in product.needs + product.revealed
                   if need not in product.waived and key_words(need.split(":")[0]) == given_up]
        product.waived.extend(matches)
        told.extend(f"dropped {need}" for need in matches)


def absorb_free_properties(st, entries: list[Any]) -> None:
    for entry in entries:
        if isinstance(entry, dict) and entry.get("property"):
            st.wanted[answer_item_number(st, entry.get("item")) - 1].open_needs.append(str(entry["property"]))


def drop_misnamed_needs(st, entries: list[Any], delivered: list[tuple[int, str, str]]) -> list[Any]:
    # the reader sometimes files a repeated answer under the property that was asked instead of the one the answer
    # names; such a copy carries another need's value word for word under a property the answer never mentions
    def words_of(text: str) -> str:
        return spelling_fold(" ".join(re.findall(r"[a-z0-9]+", text.lower())))
    spoken = words_of(" ".join(answer for _, _, answer in delivered))
    known = [need for product in st.wanted for need in product.revealed]
    kept: list[Any] = []
    for entry in entries:
        need = str(entry.get("need") or "") if isinstance(entry, dict) else ""
        prop, value = need.partition(":")[0], words_of(need.partition(":")[2])[:60]
        twins = [old for old in known + [str(e.get("need") or "") for e in kept if isinstance(e, dict)]
                 if value and words_of(old.partition(":")[2])[:60] == value
                 and key_words(old.partition(":")[0]) != key_words(prop)]
        # the property counts as named only outside the copied value (a pointer can mention other properties)
        outside = spoken
        for old in twins:
            outside = outside.replace(words_of(old.partition(":")[2]), " ")
        mentioned = words_of(prop) in outside
        if need and not mentioned and twins:
            st.log.append({"kind": "misnamed_need", "need": need[:120], "twin": twins[0][:120]})
            continue
        kept.append(entry)
    return kept


def apply_answers(st, heard: dict[str, Any], told: list[str]) -> None:
    absorb_stated_needs(st, as_list(heard.get("needs")), told)
    absorb_dropped_needs(st, as_list(heard.get("drop")), told)
    absorb_free_properties(st, as_list(heard.get("free")))
    figure = heard.get("budget")
    numeric = isinstance(figure, (int, float)) and not isinstance(figure, bool)
    if numeric and figure > 0:
        st.budget = figure


def read_answers(st, sent: list[tuple[int, str, str]],
                 models: tuple[str, ...] = (STRONG_LLM, FAST_LLM)) -> dict[str, Any]:
    item_lines = [f"item {n}: {product.name}" for n, product in enumerate(st.wanted, 1)]
    qa_lines = [f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)]
    prompt = "\n\n".join([
        HEAR_PROMPT,
        "Materials - the requested items:\n" + "\n".join(item_lines),
        "Materials - each question with its answer:\n" + "\n".join(qa_lines),
    ])
    return st.llm.ask_json(prompt, list(models), 1500) or {}


# ---------- the step gate ----------


def gate_tool_menu(st, closing: set) -> list[str]:
    menu = []
    for spec in st.tool_specs:
        fn = spec["function"]
        if fn["name"] in closing:
            continue
        params = to_json(fn.get("parameters") or {})
        menu.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {params}")
    return menu


def gate_prompt_text(st, stage: str, due: list, menu: list[str]) -> str:
    ordering = stage in ("order", "choose")
    moment = {"order": "place the order", "choose": "put its chosen listings in the cart"}.get(stage, "end without a purchase")
    head = GATE_PROMPT.replace("STAGE", moment)
    numbered = "\n".join(f"{n}. {entry}" for n, entry in enumerate(st.record, 1))
    chunks = [
        head,
        "Materials - shopper request:\n" + st.query,
        "Materials - steps due at this point:\n" + "\n".join(f"- {p['step']}" for p in due),
        "Materials - ledger of items and requirements:\n" + ledger_text(st),
        "Materials - cart at present:\n" + cart_summary(st),
        "Materials - RECORD (numbered shopper messages and step calls):\n" + (numbered or "(empty so far)"),
    ]
    if st.store_notes:
        chunks.append("Materials - latest store notes:\n" + "\n".join(st.store_notes[-4:]))
    closing_word = "order" if ordering else "ending"
    chunks.append(
        "Materials - what remains:\n"
        f"messages remaining: {max(0, st.msg_cap - st.msgs_sent)}; steps remaining: {steps_left(st)}"
        f" (the {closing_word} itself uses one of them)"
    )
    chunks.append("Materials - TOOLS:\n" + "\n".join(menu))
    return "\n\n".join(chunks)


def gate_change_text(done_calls: list[str], told: list[str], hold: str) -> str:
    opening = ""
    if done_calls:
        opening = "First, Kyo carried out the steps the request wants here:\n" + "\n".join(done_calls) + "\n"
    changes = told + ([hold] if hold else [])
    return (opening + "The shopper's answer shifts the task: " + "; ".join(changes)
            + ". Update the cart to fit, then try checkout.")


def gate_step_proven(entry: dict, valid: set) -> bool:
    if entry.get("done") is not True:
        return False
    return any((as_count(ref) or 0) in valid for ref in as_list(entry.get("lines")))


def gate_unproven_steps(st, reply: dict, due: list, stage: str) -> list:
    claimed = [entry for entry in as_list(reply.get("steps")) if isinstance(entry, dict)]
    must_follow_cart = stage == "order" and any(p["messages"] > 0 for p in due)
    total = len(st.record)
    if must_follow_cart:
        valid = {n for n in range(1, total + 1)
                 if n <= len(st.record_steps) and st.record_steps[n - 1] > st.cart_changed_step}
    else:
        valid = set(range(1, total + 1))
    return [entry for entry in claimed if not gate_step_proven(entry, valid)]


def gate_execute(st, tool: str, args: dict, asker, done_calls: list[str]) -> bool:
    if tool == asker:
        text = next((value for value in args.values() if isinstance(value, str)), "")
        answered = send_questions(st, [(1, "", text)], 1)
        if not answered:
            return False
        st.gate_sent += 1
        done_calls.append(f"Q: {text}\nA: {answered[0][2]}")
        return True
    result = st.store.act(tool, args)
    if result is None:
        return False
    seen = result.get("observation")
    absorb_result(st, seen)
    shown = show_result(st, seen)
    st.record.append(f"step {st.steps_used}: {tool} {to_json(args)} -> {shown[:300]} "
                     f"(cart afterwards: {cart_signature(st)})")
    st.record_steps.append(st.steps_used)
    done_calls.append(f"{tool} {to_json(args)} -> {shown[:400]}")
    return True


def process_gate(st, stage: str) -> tuple[bool, str]:
    due = [p for p in st.procedure if p["when"] == stage]
    if not due or st.done or not gate_pending(st, stage):
        return True, ""
    closing = {st.role_of.get("buy"), st.role_of.get("refuse")} - {None}
    menu = gate_tool_menu(st, closing)
    asker = st.role_of.get("ask")
    done_calls: list[str] = []
    pushed = False
    rounds = 0
    reply = None
    while rounds < GATE_CALLS:
        rounds += 1
        models = [STRONG_LLM, FAST_LLM] if pushed else [FAST_LLM, STRONG_LLM]
        reply = st.llm.ask_json(gate_prompt_text(st, stage, due, menu), models, 2000)
        if reply is None:
            break
        told: list[str] = []
        hold = ""
        if done_calls:
            # needs are read only once the gate itself has sent something
            apply_answers(st, {"needs": reply.get("needs")}, told)
            hold = str(reply.get("hold") or "").strip()
        call = reply.get("call")
        if not isinstance(call, dict):
            call = None
        st.log.append({"kind": "gate", "stage": stage, "call": clip(call, 200), "steps": clip(reply.get("steps"), 120),
                       "hold": hold[:200], "told": told})
        if told or hold:
            st.needs_version += 1
            return False, gate_change_text(done_calls, told, hold)
        tool = str((call or {}).get("tool") or "")
        unproven = gate_unproven_steps(st, reply, due, stage)
        if (not call and unproven and not done_calls and not pushed and asker
                and st.msgs_sent < st.msg_cap and steps_left(st) >= 2):
            pushed = True
            st.log.append({"kind": "gate_unproven", "steps": clip(unproven, 120)})
            listed = "; ".join(str(entry.get("step") or "") for entry in unproven)[:200]
            st.record.append(
                f"(Kyo note: since the cart last changed at step {st.cart_changed_step}, no RECORD line proves these "
                f"done: {listed}. Remaining: {st.msg_cap - st.msgs_sent} message(s) and {steps_left(st)} steps; "
                "name the call that carries it out)"
            )
            st.record_steps.append(-1)
            continue
        if not call:
            st.gate_passed.add((stage, cart_signature(st) + f"#{st.needs_version}"))
            break
        blocked = tool not in st.arg_schemas or tool in closing
        if blocked or steps_left(st) < 2 or (tool == asker and st.msgs_sent >= st.msg_cap):
            break
        if tool == asker and not gate_message_room(st, stage, due, done_calls):
            st.log.append({"kind": "gate_quota", "stage": stage})
            break
        faults: list[str] = []
        given = call.get("args")
        args = coerce(given if isinstance(given, dict) else {}, st.arg_schemas[tool], "args", faults)
        if faults or not gate_execute(st, tool, args, asker, done_calls):
            break
    order_message_fallback(st, stage, due, reply, asker, done_calls)
    if done_calls:
        return True, "First, Kyo carried out the steps the request wants here:\n" + "\n".join(done_calls)
    return True, ""


def gate_message_room(st, stage: str, due: list, done_calls: list[str]) -> bool:
    # a step due before the order spends no more messages than its own steps state and leaves the messages that the
    # later steps (the order, the end) need
    if stage == "order":
        return True
    sent_here = sum(1 for call in done_calls if call.startswith("Q: "))
    own = max(1, sum(p["messages"] for p in due))
    later = max([p["messages"] for p in st.procedure if p["when"] in ("order", "end")] or [0])
    return sent_here < own and st.msg_cap - st.msgs_sent > later


def order_message_fallback(st, stage: str, due: list, reply: dict | None, asker: str | None,
                           done_calls: list[str]) -> None:
    # an order-time step that talks to the shopper is still unproven after the gate's rounds: Kyo sends the order
    # itself, its lines and its total, and asks to go ahead
    talking = [p for p in due if p["messages"] > 0]
    if stage != "order" or not talking or not asker or st.done:
        return
    if st.msgs_sent >= st.msg_cap or steps_left(st) < 2 or any(call.startswith("Q: ") for call in done_calls):
        return
    if reply is not None and not gate_unproven_steps(st, reply, due, stage):
        return
    lines, total = [], 0.0
    for ident in cart_listing_ids(st):
        units = st.basket.get(ident, 1)
        price = listing_price(st, ident)
        lines.append(f"{units} x {ident}" + (f" at {price:g} each" if price is not None else ""))
        total += (price or 0) * units
    if not lines:
        return
    unit = str(st.plan.get("budget_unit") or "").strip()
    text = ("Here is the order I am about to place: " + "; ".join(lines)
            + f". Total: {total:g}{' ' + unit if unit else ''}. Shall I go ahead?")
    if gate_execute(st, asker, {"text": text}, asker, done_calls):
        st.log.append({"kind": "order_message_fallback", "text": text})


# ---------- shopper words and store notes ----------


def iter_phrases(node: Any, depth: int = 0):
    if depth > 4:
        return
    if isinstance(node, str):
        if len(node.split()) >= 2:
            yield node
        return
    children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
    for child in children:
        yield from iter_phrases(child, depth + 1)


def shopper_words(st, observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    phrases = list(iter_phrases(observation))
    if not phrases:
        return to_json(observation)[:400]
    return " ".join(phrases)[:800]


def wordy_strings(values, min_words: int) -> list[str]:
    return [value for value in values if isinstance(value, str) and len(value.split()) >= min_words]


def stray_lines(st, observation: Any) -> list[str]:
    if not isinstance(observation, dict) or find_id(st, observation):
        return []
    found: list[str] = []
    for value in observation.values():
        if isinstance(value, str):
            found += wordy_strings([value], 5)
        elif isinstance(value, list):
            if value and all(isinstance(entry, str) for entry in value):
                found += wordy_strings(value, 5)
        elif isinstance(value, dict) and not find_id(st, value, False, 1):
            found += wordy_strings(value.values(), 6)
    return [line for line in found if line not in st.heard_lines]


def pick_product(st, raw: Any):
    position = min(max(1, as_count(raw) or 1), len(st.wanted))
    return st.wanted[position - 1]


def note_entries(reply: dict, key: str) -> list:
    return [e for e in as_list(reply.get(key)) if isinstance(e, dict) and str(e.get("need") or "").strip()]


def note_drop_need(st, entry: dict) -> str:
    product = pick_product(st, entry.get("item"))
    target = str(entry["need"]).strip()
    active = [need for need in product.needs + product.revealed if need not in product.waived]
    matched = [need for need in active if need == target]
    if not matched:
        label = key_words(target.split(":")[0])
        matched = [need for need in active if key_words(need.split(":")[0]) == label][:1]
    product.waived.extend(matched or [target])
    return f"removed {target}"


def note_add_need(st, entry: dict) -> str:
    product = pick_product(st, entry.get("item"))
    need = str(entry["need"]).strip()
    if need in product.waived:
        product.waived.remove(need)
    known = product.needs + product.revealed
    if need not in known:
        product.revealed.append(need)
    return f"needs {need}"


def note_sellout(st, reply: dict, lines: list[str]) -> list[str]:
    gone: list[str] = []
    for entry in as_list(reply.get("gone")):
        if isinstance(entry, (str, int)):
            gone.append(str(entry))
    st.blocked_ids.update(name for name in gone if name in st.listings)
    flagged = bool(gone) or reply.get("sold_out") is True
    st.sold_noted = flagged
    if flagged and len(lines) == 1:
        # remember a pure sell-out note by its exact text
        st.sold_ids.update(lines)
    return gone


def note_budget(st, reply: dict) -> list[str]:
    value = reply.get("budget")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return []
    if not value > 0 or value == st.budget:
        return []
    st.budget = value
    return [f"new budget {value}"]


def apply_note(st, lines: list[str]) -> str:
    st.heard_lines.update(lines)
    framing = ["Materials - current items and requirements:\n" + ledger_text(st),
               "Materials - the note itself:\n" + "\n".join(lines)]
    reply = st.llm.ask_json("\n\n".join([NOTE_PROMPT] + framing), [FAST_LLM, BACKUP_LLM], 1200) or {}
    gone = note_sellout(st, reply, lines)
    if reply.get("kind") is not None and reply.get("kind") != "change":
        return ""
    changes = note_budget(st, reply)
    removals, additions = note_entries(reply, "drop"), note_entries(reply, "add")
    for entry in removals:
        changes.append(note_drop_need(st, entry))
    for entry in additions:
        changes.append(note_add_need(st, entry))
    if gone:
        changes.append("not on sale: " + ", ".join(gone))
    if reply.get("price_moved") is True:
        changes.append("a price shifted")
    summary = str(reply.get("summary") or "").strip()
    if not changes and not summary:
        return summary
    st.needs_version += 1
    st.recent_calls = []
    st.store_notes.append(summary or "; ".join(changes))
    if not changes:
        return summary
    return f"{summary} ({'; '.join(changes)})".strip()


def current_needs(st) -> list[tuple[int, str]]:
    return [(number, need)
            for number, product in enumerate(st.wanted, 1)
            for need in product.needs + product.revealed
            if need not in product.waived]


# ---------- evidence, cart review and trails ----------


def collect_fact_ids(st, node: Any, depth: int, ids: list[str]) -> None:
    if depth > 6:
        return
    if isinstance(node, list):
        children = node
    elif isinstance(node, dict):
        listing_id = find_id(st, node, False, depth)
        if listing_id and attribute_block(node) and listing_id not in ids:
            ids.append(listing_id)
            return
        children = node.values()
    else:
        return
    for child in children:
        collect_fact_ids(st, child, depth + 1, ids)


def evidence_line(st, listing_id: str, needs: list, several: bool) -> str:
    attrs = attribute_block(st.pages.get(listing_id) or {}) or {}
    found = []
    for number, need in needs:
        prop = need.split(":", 1)[0].strip()
        prop_stems = word_stems(prop)
        hits = [f"{key}={str(value)[:50]}" for key, value in attrs.items()
                if names_property(prop_stems, word_stems(key))]
        if hits:
            prefix = f"(item {number}) " if several else ""
            found.append(prefix + f"{prop} -> " + ", ".join(hits[:3]))
    return f"{listing_id}: " + "; ".join(found) if found else ""


def evidence_text(st, observation: Any) -> str:
    ids: list[str] = []
    collect_fact_ids(st, observation, 0, ids)
    needs = current_needs(st)
    if not (ids and needs):
        return ""
    several = len(st.wanted) > 1
    rows = [line for line in (evidence_line(st, i, needs[:8], several) for i in ids[:12]) if line]
    if not rows:
        return ""
    return ("[evidence: attribute keys named with a requirement's property; a key naming another property (a different "
            "part or aspect) settles nothing, and a key under another name may still hold that property] "
            + " | ".join(rows))


def cart_view_fresh(st) -> bool:
    return st.cart_snapshot is not None and st.cart_viewed_step >= st.cart_changed_step


def cart_summary(st) -> str:
    if not cart_view_fresh(st):
        # an older view could show lines that were removed since
        return "cart lines as Kyo logged its cart calls (listing: units): " + to_json(st.basket)
    return to_json(clip(st.cart_snapshot, 300))[:3000]


def cart_listing_ids(st) -> list[str]:
    if not cart_view_fresh(st):
        return list(st.basket)
    snapshot = to_json(st.cart_snapshot)
    return [listing_id for listing_id in st.listings if listing_id in snapshot]


def listing_record(st, ident: str) -> str:
    record = st.pages.get(ident) or st.listings.get(ident)
    if not record:
        return "(none on record)"
    return to_json(clip(record, 400))[:3000]


def price_rank(st, listing_id: str) -> tuple:
    price = listing_price(st, listing_id)
    return listing_id not in st.pages, float("inf") if price is None else price


def other_records(st, skip: list[str], limit: int = 30) -> str:
    candidates = st.seen_ids[::-1]
    if st.plan.get("pick") == "cheapest":
        candidates.sort(key=lambda listing_id: price_rank(st, listing_id))
    picked: list[str] = []
    for listing_id in candidates:
        if listing_id in skip or is_sold_out(st, listing_id):
            continue
        record = st.pages.get(listing_id) or st.listings.get(listing_id)
        picked.append(to_json(clip(record, 160))[:900])
        if len(picked) >= limit:
            break
    return "\n".join(picked)


def check_signature(st, ids: list[str]) -> str:
    return f"{'|'.join(sorted(ids))}#{st.needs_version}"


def value_words(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return frozenset(word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words)


def key_conflicts(st, ids: list[str]) -> list[str]:
    # strict code side of the review: for each line, the listing's own attribute whose key names a required property,
    # where none of the required value's words appear in the stated value; limits with figures and values taken
    # from another listing are left to the review
    lookups = {need for _, need in st.references}
    found = []
    for ident in ids:
        block = attribute_block(st.pages.get(ident) or st.listings.get(ident) or {}) or {}
        pos = line_owner(st, ident)
        if pos is None or not block:
            continue
        product = st.wanted[pos]
        for need in product.needs + product.revealed:
            prop, _, wanted = need.partition(":")
            if need in product.waived or need in lookups or not wanted.strip() or re.search(r"\d", wanted):
                continue
            key = next((key for key in block if names_property(word_stems(prop), word_stems(key))), None)
            if key is None or not isinstance(block[key], (str, int, float)):
                continue
            if not value_words(block[key]) & value_words(wanted):
                found.append(f'{ident}: its own "{key}" says "{block[key]}"; the requirement says "{need.strip()}"')
    return found


def review_prompt(st, ids: list[str], planned: bool) -> str:
    parts = [CHECK_PROMPT,
             "Materials - shopper request:\n" + st.query,
             "Materials - requirements that apply:\n" + ledger_text(st)]
    if planned:
        per_item = "; ".join(f"item {n}: quantity {product.quantity}" for n, product in enumerate(st.wanted, 1))
        parts.append("Materials - planned cart (judge these listings as though carted at the requested quantities: "
                     + per_item + "):\n" + ", ".join(ids))
    else:
        parts.append("Materials - cart:\n" + cart_summary(st))
    sold_tag = "SOLD OUT (the store reports it can no longer be bought). "
    flagged = {i: sold_tag if is_sold_out(st, i) else "" for i in ids}
    parts.append("Materials - records of the listings in the cart:\n"
                 + "\n".join(f"{i}: {flagged[i]}{listing_record(st, i)}" for i in ids))
    if st.plan.get("pick") in ("cheapest", "most", "least"):
        parts.append("Materials - other listings seen, newest first:\n" + other_records(st, ids, 15))
    conflicts = key_conflicts(st, ids)
    if conflicts:
        parts.append("Materials - found in code: the listing's own attribute for a required property states a value"
                     " that shares no word with the requirement. Such a line fails, a related or broader word"
                     " included (a cartoon is not a character print), unless the requirement's own wording admits"
                     " that value (a negation, a range, a list of options, a value taken from another listing):\n"
                     + "\n".join(conflicts))
    return "\n\n".join(parts)


def review_complaint(st, reply: dict) -> str:
    issues = [str(p) for p in as_list(reply.get("problems"))][:5] or ["see the requirements"]
    swaps: list[str] = []
    for raw in as_list(reply.get("better")):
        candidate = str(raw)
        if candidate in st.listings and not is_sold_out(st, candidate):
            swaps.append(candidate)
    pieces = ["An independent review finds the cart still wrong: " + "; ".join(issues)]
    if swaps[:4]:
        pieces.append("Stronger-looking listings: " + ", ".join(swaps[:4]))
    pieces.append("Correct the cart (or re-read the records to prove the review flawed), then check out.")
    return ". ".join(pieces)


def review_cart(st, ids: list[str] | None = None, limit: float = SOFT_SPEND) -> str:
    planned = ids is not None
    chosen = ids if planned else cart_listing_ids(st)
    key = check_signature(st, chosen)
    if key in st.approved or st.spend > limit or not chosen:
        return ""
    st.approved.add(key)
    reply = st.llm.ask_json(review_prompt(st, chosen, planned), [STRONG_LLM, FAST_LLM], 1500)
    st.log.append({"kind": "check", "cart": chosen, "got": reply})
    passed = bool(reply) and reply.get("ok") is not False
    st.verdict_by_cart[key] = passed
    if passed or not reply:
        return ""
    st.pointers = [p for p in st.pointers if p not in chosen]
    return review_complaint(st, reply)


def trail_record_line(st, listing_id: str) -> str:
    if listing_id in st.opened_ids:
        page = st.pages.get(listing_id) or st.listings.get(listing_id)
        if page:
            return "OPENED " + to_json(clip(page, 300))[:1400]
    return "ROW " + to_json(clip(st.listings.get(listing_id), 80))[:300]


def trail_records(st) -> list[str]:
    return [trail_record_line(st, listing_id) for listing_id in st.seen_ids[::-1][:60]]


def chain_need(chain: dict, pending: list[str]) -> str | None:
    named = str(chain.get("need") or "").strip()
    match = next((p for p in pending if p == named), None)
    if match:
        return match
    return pending[0] if len(pending) == 1 else None


def read_chains(st, reply: dict, pending: list[str]) -> tuple[list[str], list[str]]:
    reach: list[str] = []
    lost: list[str] = []
    for chain in as_list(reply.get("chains")):
        need = chain_need(chain, pending) if isinstance(chain, dict) else None
        if not need:
            continue
        links = [link for link in as_list(chain.get("links")) if isinstance(link, dict)]
        link_ids = [str(link.get("id") or "").strip() for link in links]
        value = str(chain.get("value") or "").strip()
        if chain.get("done") is True and links and value and all(i in st.opened_ids for i in link_ids):
            st.reference_values[need] = value
            st.needs_version += 1
            continue
        unopened = [i for i in link_ids if i in st.listings and i not in st.opened_ids]
        if not unopened:
            unknown = (str(link.get("about") or "") for link, i in zip(links, link_ids) if i not in st.listings)
            about = next(unknown, "") or "the listing it refers to"
            lost.append(f"{need} -> find {about}")
            continue
        # judged against reach as it stood before this chain
        reach.extend([i for i in unopened if i not in reach])
    return reach, lost


def open_chain_links(st, tool: str, target: str, batch: list[str], opened: list[str]) -> None:
    for listing_id in batch:
        faults: list[str] = []
        args = coerce({target: listing_id}, st.arg_schemas.get(tool), "args", faults)
        result = None if faults else st.store.act(tool, args)
        if result is None or st.done:
            return
        page = result.get("observation")
        observe_result(st, tool, page)
        opened.append(f"{listing_id}: {show_result(st, page)[:1500]}")


def trail_report(st, opened: list[str], lost: list[str]) -> tuple[str, bool]:
    pieces = []
    if opened:
        pieces.append("[pages a requirement refers to, opened by Kyo]\n" + "\n".join(opened) + "\n")
    if st.reference_values:
        pairs = "; ".join(f"{need} -> {value}" for need, value in st.reference_values.items())
        pieces.append("[values taken from pages that a requirement names] " + pairs + "\n")
    hold = bool(lost) and st.trail_stops < 2 and steps_left(st) > 5
    if hold:
        st.trail_stops += 1
        pieces.append("An order only counts once each page a requirement refers to is opened in full (a search row "
                      "does not do), including every further listing such a page refers to in turn. Not yet found and "
                      "opened: "
                      + " | ".join(lost[:3])
                      + ". Look it up by its shop, brand or wording, open it, apply the value its page states, then "
                        "check out.")
    return "".join(pieces), hold


def follow_trails(st) -> tuple[str, bool]:
    live = {need for _, need in current_needs(st)}
    pending = [need for _, need in st.references if need in live and need not in st.reference_values]
    tool = offers_tool(st)
    target = single_id_param(st, tool)
    opened: list[str] = []
    lost: list[str] = []
    while pending and st.trail_runs < 4 and st.spend < HARD_SPEND:
        st.trail_runs += 1
        prompt = (TRAIL_PROMPT
                  + "\n\nMaterials - requirements whose value sits on another listing:\n" + "\n".join(pending)
                  + "\n\nMaterials - records, newest first:\n" + "\n".join(trail_records(st)))
        reply = st.llm.ask_json(prompt, [FAST_LLM, STRONG_LLM], 1800) or {}
        st.log.append({"kind": "trail", "pending": pending, "got": clip(reply, 150)})
        reach, lost = read_chains(st, reply, pending)
        pending = [p for p in pending if p not in st.reference_values]
        if not reach or not tool or not target or steps_left(st) < len(reach) + 4 or st.done:
            break
        open_chain_links(st, tool, target, reach[:max(0, min(3, 6 - len(opened)))], opened)
    return trail_report(st, opened, lost)


# ---------- offers and sold-out lines ----------


def offers_tool(st) -> str | None:
    roles = st.role_of
    return roles.get("offer") or roles.get("open")


def reread_listing(st, tool: str, target: str, listing_id: str) -> str | None:
    faults: list[str] = []
    args = coerce({target: listing_id}, st.arg_schemas.get(tool), "args", faults)
    if faults:
        return None
    snapshot = dict(st.listings.get(listing_id) or {})
    result = st.store.act(tool, args)
    if result is None or st.done:
        return None
    fresh = result.get("observation")
    st.offer_step[listing_id] = st.steps_used
    observe_result(st, tool, fresh)
    # compared on the stored record: a reply bundled with a note still reveals the change
    moved: list[str] = []
    for key, value in (st.listings.get(listing_id) or {}).items():
        if isinstance(value, (dict, list)) or key not in snapshot:
            continue
        if snapshot[key] != value:
            moved.append(key)
    if moved:
        return f"{listing_id} differs ({', '.join(moved[:4])}): {to_json(clip(fresh, 300))[:900]}"
    return ""


def refresh_offers(st) -> str:
    tool = offers_tool(st)
    target = single_id_param(st, tool)
    if not tool or not target:
        return ""
    floor = max(st.steps_used - 2, st.refuse_step)
    stale = [i for i in cart_listing_ids(st) if i not in st.sold_step and st.offer_step.get(i, -99) < floor]
    if not stale or steps_left(st) < len(stale) + 3:
        return ""
    changed: list[str] = []
    for listing_id in stale:
        outcome = reread_listing(st, tool, target, listing_id)
        if outcome is None:
            break
        changed += [outcome] if outcome else []
    st.log.append({"kind": "reread", "ids": stale, "changed": len(changed)})
    if changed:
        st.needs_version += 1
    return "\n".join(changed)


def is_sold_out(st, ident: str) -> bool:
    return any(ident in pool for pool in (st.blocked_ids, st.sold_step))


def swap_note(st) -> str:
    gone_now = [ident for ident in cart_listing_ids(st) if ident not in st.sold_step and is_sold_out(st, ident)]
    if not gone_now:
        return ""
    for ident in gone_now:
        st.sold_step[ident] = st.steps_used
    st.pointers = [ident for ident in st.pointers if ident not in gone_now]
    st.needs_version += 1
    options = find_replacements(st, gone_now)
    opening = (f"[Kyo] Out of stock: {', '.join(gone_now)} (the store says it can no longer be bought). It will not "
               "return, so skip reopening it.")
    if not options:
        return opening + (" Take that line out of the cart and put in the best other listing that fits; search for "
                          "one if no listing read fits.")
    if steps_left(st) <= 6:
        backups = ', '.join(options[1:]) or 'none'
        return opening + (f" Steps are nearly gone: right now add {options[0]} at the quantity asked for and drop "
                          f"the sold-out line (two calls), then check out. Other listings already read that "
                          f"look like a fit: {backups}.")
    joined = ', '.join(options)
    if st.plan.get("pick") == "cheapest":
        return opening + (f" Listings you have already read that look like a fit for it, cheapest first: "
                          f"{joined}. Since the rule is to pick the cheapest, unless every fitting listing is already "
                          f"seen, run one search below {options[0]}'s price (using the search's maximum price, where "
                          "it has one) before swapping the sold-out line for the lowest-cost fit.")
    return opening + (f" Listings you have read which look like a fit for that item: {joined}. Right away, take "
                      f"the sold-out line from the cart and put in {options[0]} at the quantity asked for, then "
                      "check out.")


def sold_note(st) -> str:
    candidates = [ident for ident in cart_listing_ids(st) if ident not in st.sold_step]
    if any(is_sold_out(st, ident) for ident in candidates):
        return ""
    if not candidates:
        return ""
    if len(candidates) > 1:
        return refresh_offers(st)
    st.blocked_ids.add(candidates[0])
    return ""


def newest_live_ids(st):
    for ident in reversed(st.seen_ids):
        if not is_sold_out(st, ident):
            yield ident


def swap_record_lines(st, lost: list[str]) -> list[str]:
    snippets: list[str] = []
    for ident in newest_live_ids(st):
        if len(snippets) == 25:
            break
        if ident not in lost and ident not in st.sold_step:
            best = st.pages.get(ident) or st.listings.get(ident)
            snippets.append(to_json(clip(best, 200))[:1200])
    return snippets


def replacement_price(raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return float("inf")
    return float(raw)


def find_replacements(st, lost: list[str]) -> list[str]:
    if st.spend > SOFT_SPEND:
        return []
    seen_records = swap_record_lines(st, lost)
    if not seen_records:
        return []
    sections = [
        SWAP_PROMPT,
        "Materials - the shopper's own request:\n" + st.query,
        "Materials - requirements that currently apply:\n" + ledger_text(st),
        "Materials - the cart line that sold out:\n" + "\n".join(
            f"{ident}: {listing_record(st, ident)}" for ident in lost),
        "Materials - the cart at this moment:\n" + cart_summary(st),
        "Materials - other listing records already seen, most recent first:\n" + "\n".join(seen_records),
    ]
    answer = st.llm.ask_json("\n\n".join(sections), [STRONG_LLM, FAST_LLM], 1200)
    found: list[tuple[float, str]] = []
    for entry in as_list((answer or {}).get("fits")):
        is_record = isinstance(entry, dict)
        ident = str(entry.get("id") if is_record else entry)
        if ident not in st.listings or ident in lost or ident in st.sold_step:
            continue
        if ident in cart_listing_ids(st) or ident in [known for _, known in found]:
            continue
        found.append((replacement_price(entry.get("price") if is_record else None), ident))
    if st.plan.get("pick") == "cheapest":
        # the LLM picks fits well but its price order is not dependable
        ordered = [ident for _, ident in sorted(found, key=lambda pair: pair[0])]
    else:
        ordered = [ident for _, ident in found]
    st.log.append({"kind": "swap", "lost": lost, "fits": found})
    return ordered[:4]


def listing_usable(st, ident: str) -> bool:
    return ident in st.listings and not is_sold_out(st, ident)


def no_fit_rule(st) -> str:
    return str(st.plan.get("no_fit_rule") or "").strip()


# ---------- decline audit ----------


def audit_decline(st) -> str:
    if st.decline_checked or st.spend > SOFT_SPEND:
        return ""
    st.decline_checked = True
    verdict = audit_request(st)
    chosen = audit_matches(st, verdict)
    if not chosen:
        proposed = [str(raw) for raw in as_list((verdict or {}).get("open"))]
        unread = [ident for ident in proposed if listing_usable(st, ident) and ident not in st.opened_ids]
        if unread and open_listings(st, unread):
            verdict = audit_request(st)
            chosen = audit_matches(st, verdict)
            if not chosen:
                return ""
        else:
            return ""
    st.pointer_items = chosen
    st.pointers = []
    for group in chosen.values():
        st.pointers.extend(group)
    summary = "; ".join("item " + number + ": " + ", ".join(ids) for number, ids in chosen.items())
    if verdict.get("reason"):
        summary += f" ({verdict.get('reason')})"
    return (f"A second careful look suggests these could fit: {summary}. Open them, buy "
            "whatever really fits; end without buying only if all of them fail.")


def audit_record_lines(st) -> list[str]:
    snippets: list[str] = []
    for ident in newest_live_ids(st):
        if len(snippets) == 30:
            break
        page = st.pages.get(ident) if ident in st.pages else None
        if not page:
            snippets.append("ROW " + to_json(clip(st.listings.get(ident), 120))[:500])
        else:
            snippets.append("OPENED " + to_json(clip(page, 200))[:1200])
    return snippets


def audit_request(st) -> dict[str, Any] | None:
    seen_records = audit_record_lines(st)
    if not seen_records:
        return None
    prompt = AUDIT_PROMPT
    if no_fit_rule(st):
        prompt = prompt.replace('"reason": "one sentence"}', CLOSEST_FIELD).replace("\n\nReturn:", CLOSEST_CHECK + "\n\nReturn:")
    sections = [
        prompt,
        "Materials - the shopper's own request:\n" + st.query,
        "Materials - requirements that currently apply:\n" + ledger_text(st),
        "Materials - listing records already seen, most recent first:\n" + "\n".join(seen_records),
    ]
    answer = st.llm.ask_json("\n\n".join(sections), [STRONG_LLM, FAST_LLM], 1500)
    st.log.append({"kind": "audit", "got": answer})
    st.audit_got = answer if isinstance(answer, dict) else st.audit_got
    return answer


def audit_matches(st, got: dict[str, Any] | None) -> dict[str, list[str]]:
    by_item = (got or {}).get("fits")
    collected: dict[str, list[str]] = {}
    if isinstance(by_item, dict):
        for number, raw_ids in by_item.items():
            usable = []
            for raw in as_list(raw_ids):
                ident = str(raw)
                if listing_usable(st, ident):
                    usable.append(ident)
            collected[str(number)] = usable[:4]
    return {label: ids for label, ids in collected.items() if ids}


def audit_open_steps(st, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
    array_key = compare_param(st)
    side_tool = st.role_of.get("compare")
    single_tool = attribute_tool(st)
    id_param = single_id_param(st, single_tool)
    if array_key and side_tool and not compare_is_bare(st):
        return [(side_tool, {array_key: rows[:compare_capacity(st)]})]
    if not (single_tool and id_param):
        return None
    return [(single_tool, {id_param: ident}) for ident in rows[:2]]


# ---------- cart building and checkout ----------


def send_coerced(st, tool_name: str, raw_args: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
    problems: list[str] = []
    shaped = coerce(raw_args, st.arg_schemas.get(tool_name), "args", problems)
    if problems:
        return None, None
    return shaped, st.store.act(tool_name, shaped)


def open_listings(st, rows: list[str], reserve: int = 5) -> bool:
    steps = audit_open_steps(st, rows)
    if steps is None or steps_left(st) - reserve < len(steps):
        return False
    known_pages = len(st.pages)
    for tool_name, raw_args in steps:
        shaped, reply = send_coerced(st, tool_name, raw_args)
        if shaped is None:
            return False
        if reply is None or st.done:
            break
        observe_result(st, tool_name, reply.get("observation"))
    gained = len(st.pages) - known_pages
    st.log.append({"kind": "audit_open", "rows": rows, "opened": gained})
    return gained > 0


def schema_first_key_of(props: dict[str, Any], wanted: str, plain_only: bool = False) -> str | None:
    for key, spec in props.items():
        if not isinstance(spec, dict) or wanted not in schema_types(spec):
            continue
        if plain_only and "enum" in spec:
            continue
        return key
    return None


def nested_cart_payload(key: str, inner: dict[str, Any], lines: dict[str, int]) -> list[dict[str, Any]] | None:
    item_key = schema_first_key_of(inner, "string", plain_only=True)
    if not item_key:
        return None
    units_key = schema_first_key_of(inner, "integer")
    entries = []
    for ident, units in lines.items():
        entry = {item_key: ident}
        if units_key:
            entry[units_key] = units
        entries.append(entry)
    return [{key: entries}]


def cart_requests(st, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
    tool_name = st.role_of.get("cart") or ""
    schema = st.arg_schemas.get(tool_name) or {}
    props = schema.get("properties") or {}
    for key, spec in props.items():
        element = spec.get("items") if isinstance(spec, dict) and "array" in schema_types(spec) else None
        if isinstance(element, dict) and isinstance(element.get("properties"), dict):
            return nested_cart_payload(key, element["properties"], lines)
    plain = [k for k, s in props.items() if isinstance(s, dict) and "string" in schema_types(s) and "enum" not in s]
    id_param = next((k for tool, k in st.id_params if tool == tool_name and k in plain), None)
    if not id_param:
        id_param = next((k for k in schema.get("required") or [] if k in plain), None)
    if not id_param and len(plain) == 1:
        id_param = plain[0]
    units_key = schema_first_key_of(props, "integer")
    mode_key = next((k for k, s in props.items()
                     if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
    if not id_param:
        return None
    requests: list[dict[str, Any]] = []
    for ident in drop:
        held = st.basket.get(ident, 1)
        if mode_key:
            request = {id_param: ident, mode_key: props[mode_key]["enum"][1]}
            if units_key:
                request[units_key] = held
        elif units_key and cart_sets_count(st, tool_name):
            request = {id_param: ident, units_key: 0}
        elif units_key and isinstance(props[units_key].get("minimum"), int) and props[units_key]["minimum"] <= -held:
            request = {id_param: ident, units_key: -held}
        else:
            return None
        requests.append(request)
    in_cart = cart_listing_ids(st)
    for ident, target_units in lines.items():
        held = st.basket.get(ident, 0) if ident in in_cart else 0
        if held == target_units:
            continue
        if held and not cart_sets_count(st, tool_name):
            return None
        request = {id_param: ident}
        if mode_key:
            request[mode_key] = props[mode_key]["enum"][0]
        if units_key:
            request[units_key] = target_units
        requests.append(request)
    return requests


def cart_line_kind(st, cat_key: str | None, ident: str) -> str:
    if not cat_key:
        return ident
    return str((st.listings.get(ident) or {}).get(cat_key) or "").strip().lower()


def compact_cart_plan(st, cat_key: str | None) -> dict[str, int]:
    chosen: dict[str, int] = {}
    for number, ids in st.pointer_items.items():
        position = as_count(number)
        candidate = next((ident for ident in ids if not is_sold_out(st, ident)), None)
        if not candidate or not position or not 1 <= position <= len(st.wanted):
            continue
        taken = {cart_line_kind(st, cat_key, other) for other in chosen}
        if cart_line_kind(st, cat_key, candidate) in taken:
            continue
        chosen[candidate] = st.wanted[position - 1].quantity
    return chosen


def order_cart_lines(st, tool_name: str, requests: list[dict[str, Any]]) -> list[str] | None:
    sent: list[str] = []
    for raw_args in requests:
        shaped, reply = send_coerced(st, tool_name, raw_args)
        ok = shaped is not None and reply is not None and not st.done
        if not ok:
            return None
        track_cart(st, tool_name, shaped)
        observe_result(st, tool_name, reply.get("observation"))
        sent.append(to_json(shaped))
    return sent


def retry_still_worth_it(st, lines: dict[str, int]) -> bool:
    refresh_offers(st)
    review_cart(st, list(lines), HARD_SPEND)
    if not st.verdict_by_cart.get(check_signature(st, list(lines))):
        return False
    if steps_left(st) < 1 or st.done:
        return False
    return not any(is_sold_out(st, ident) for ident in lines)


def checkout_with_retry(st, buy_tool: str, buy_args: dict[str, Any], lines: dict[str, int],
                        drop: list[str]) -> list[str]:
    answers: list[str] = []
    # a refused order (price, budget or stock moved) gets one more go after a re-read and a fresh check
    for second_go in (False, True):
        if second_go and not retry_still_worth_it(st, lines):
            break
        proceed, gate_text = process_gate(st, "order")
        if gate_text:
            answers.append(gate_text)
        if not proceed:
            break
        outcome = st.store.act(buy_tool, buy_args)
        st.log.append({"kind": "completed", "lines": lines, "drop": drop, "reply": clip(outcome, 200)})
        if outcome is None:
            break
        st.buy_tries += 1
        accepted = outcome.get("finished")
        if not accepted:
            st.refuse_step = st.steps_used
        observe_result(st, buy_tool, outcome.get("observation"))
        answers.append(show_result(st, outcome.get("observation")))
        if st.done or accepted:
            break
    return answers


def fill_cart(st) -> str:
    if st.cart_completed or not st.confirm_mark or is_sold_out(st, st.confirm_mark):
        return ""
    sent, replies = order_planned_cart(st, reviewed=True)
    if not replies:
        return ""
    return (f"Skipped (Kyo acted directly): the independent check approved {st.confirm_mark}, so Kyo "
            f"built a cart ({'; '.join(sent)}) and ordered it. Store answer: "
            + " | next: ".join(replies))


def audit_order(st) -> str:
    # a decline with too few steps left for the actor to act on a review: one audit of the records already seen,
    # and if it finds a fitting listing for every item, Kyo builds that cart and orders it
    if st.audit_ordered or st.done or st.spend > HARD_SPEND:
        return ""
    st.audit_ordered = True
    # always a fresh look: an earlier audit's leads may since have been opened and failed
    verdict = audit_request(st)
    st.decline_checked = True
    fits = audit_matches(st, verdict)
    if len(fits) < len(st.wanted):
        # the leads it names are only search rows: open them with the steps the order itself does not need, look again
        leads = [str(raw) for raw in as_list((verdict or {}).get("open"))]
        unread = [ident for ident in leads if listing_usable(st, ident) and ident not in st.opened_ids]
        if not unread or not open_listings(st, unread, len(st.wanted) + 1):
            return ""
        verdict = audit_request(st)
        fits = audit_matches(st, verdict)
    if len(fits) < len(st.wanted):
        return ""
    st.pointer_items = fits
    st.confirm_mark = next(iter(fits.values()))[0]
    st.log.append({"kind": "audit_order", "lines": fits})
    # the audit reads loosely ("light" for blonde); the strict review has to pass the cart before Kyo orders it
    sent, replies = order_planned_cart(st, reviewed=True)
    if not replies:
        return ""
    return (f"Skipped (Kyo acted directly): a last look at the listings already read found a fit for every item,"
            f" so Kyo built a cart ({'; '.join(sent)}) and ordered it. Store answer: " + " | next: ".join(replies))


def closest_order(st) -> str:
    # the request's own rule for nothing fitting may ask for a listing anyway: build that cart rather than decline
    if st.closest_tried or st.done or not no_fit_rule(st):
        return ""
    st.closest_tried = True
    verdict = st.audit_got if st.audit_got is not None else audit_request(st)
    closest = audit_matches(st, {"fits": (verdict or {}).get("closest")})
    if not closest:
        return ""
    st.pointer_items = closest
    st.confirm_mark = next(iter(closest.values()))[0]
    st.log.append({"kind": "closest_order", "lines": closest})
    sent, replies = order_planned_cart(st, reviewed=False)
    if not replies:
        return ""
    return (f"Skipped (Kyo acted directly): the request's own rule for nothing fitting (\"{no_fit_rule(st)}\") asks"
            f" for a listing anyway, so Kyo built a cart ({'; '.join(sent)}) and ordered it. Store answer: "
            + " | next: ".join(replies))


def order_planned_cart(st, reviewed: bool) -> tuple[list[str], list[str]]:
    cart_tool, buy_tool = st.role_of.get("cart"), st.role_of.get("buy")
    if not cart_tool or not buy_tool:
        return [], []
    st.cart_completed = True
    cat_key = category_field(st)
    chosen = compact_cart_plan(st, cat_key)
    if st.confirm_mark not in chosen:
        return [], []
    current = cart_listing_ids(st)
    chosen_kinds = {cart_line_kind(st, cat_key, ident) for ident in chosen}
    kept: dict[str, int] = {}
    for ident in current:
        if ident in chosen or is_sold_out(st, ident) or cart_line_kind(st, cat_key, ident) in chosen_kinds:
            continue
        kept[ident] = st.basket.get(ident, 1)
    removed = [ident for ident in current if ident not in chosen and ident not in kept]
    kept_kinds = {cart_line_kind(st, cat_key, ident) for ident in kept}
    if len(chosen) + len(kept_kinds) < len(st.wanted):
        return [], []
    lines = {**kept, **chosen}
    requests = cart_requests(st, lines, removed)
    if not requests or len(requests) + 1 > steps_left(st):
        return [], []
    # these store calls need no LLM turn; one review is allowed even past the soft budget
    if reviewed:
        fetch_attributes(st, list(lines), len(requests) + 3)
        review_cart(st, list(lines), HARD_SPEND)
        if not st.verdict_by_cart.get(check_signature(st, list(lines))):
            return [], []
    sent = order_cart_lines(st, cart_tool, requests)
    if sent is None:
        return [], []
    problems: list[str] = []
    buy_args = coerce(dict(st.plan.get("buy_args") or {}), st.arg_schemas.get(buy_tool), "args", problems)
    if problems:
        return [], []
    return sent, checkout_with_retry(st, buy_tool, buy_args, lines, removed)


def role_call_matches(st, role: str, fixed_key: str, name: str, args: dict[str, Any]) -> bool:
    if name != st.role_of.get(role):
        return False
    pinned = st.plan.get(fixed_key)
    if not isinstance(pinned, dict):
        pinned = {}
    for key, value in pinned.items():
        if args.get(key) != value:
            return False
    return True


def is_checkout(st, name: str, args: dict[str, Any]) -> bool:
    return role_call_matches(st, "buy", "buy_args", name, args)


def is_decline(st, name: str, args: dict[str, Any]) -> bool:
    return role_call_matches(st, "refuse", "refuse_args", name, args)


def offered_tools(st) -> list[dict[str, Any]]:
    remaining = steps_left(st)
    ask_tool = st.role_of.get("ask")
    finishers = {st.role_of.get("buy"), st.role_of.get("refuse")} - {None}
    reserve = finish_reserve(st)
    allowed: set[str] | None
    if remaining <= reserve:
        allowed = set(finishers)
    elif remaining <= reserve + 3:
        allowed = finishers | {st.role_of.get("cart"), st.role_of.get("cart_view")}
    elif remaining <= len(st.wanted) + 5 and st.buy_tries == 0:
        allowed = finishers | {st.role_of.get("cart"), st.role_of.get("cart_view"), st.role_of.get("offer")}
    else:
        allowed = None
    visible = [spec for spec in st.tool_specs if spec["function"]["name"] != ask_tool]
    if allowed is None or not finishers:
        return visible
    narrowed = [spec for spec in visible if spec["function"]["name"] in allowed]
    return narrowed if narrowed else visible


def quiet_args(st, name: str, args: dict[str, Any]) -> dict[str, Any]:
    schema = st.arg_schemas.get(name) or {}
    needed = schema.get("required") or []
    apology = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."
    words_ok = st.plan.get("quiet_decline") is False
    cleaned = dict(args)
    text_fields = [
        (key, spec) for key, spec in (schema.get("properties") or {}).items()
        if isinstance(spec, dict) and "string" in schema_types(spec) and "enum" not in spec
    ]
    for key, spec in text_fields:
        current = cleaned.get(key)
        if current is None and key not in needed:
            continue
        said = str(current or "")
        keep_own_words = bool(said) and words_ok and not any(ident in said for ident in st.listings)
        if not keep_own_words:
            cleaned[key] = apology[: spec.get("maxLength", 200)]
    return cleaned


def row_starts_with(st, row: dict[str, Any], prefix: str) -> bool:
    return any(
        field != st.row_id_key and isinstance(cell, str) and cell.strip().lower().startswith(prefix)
        for field, cell in row.items()
    )


def title_to_id(st, value: Any) -> tuple[Any, str]:
    if not isinstance(value, str):
        return value, ""
    stripped = value.strip()
    if stripped in st.listings:
        return stripped, ""
    if looks_like_id(value, True) or len(stripped) < 8:
        return value, ""
    matches = [ident for ident, row in st.listings.items() if row_starts_with(st, row, stripped.lower())]
    if not matches:
        return value, (f"{value!r} isn't a listing id; use the id shown in the result rows "
                       f"(the {st.row_id_key} field)")
    if len(matches) > 1:
        return value, (f"several listings match the text {value!r} ({', '.join(matches[:6])}); "
                       "send one of the ids")
    return matches[0], ""


def fix_ids(st, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    id_tools = {st.role_of.get(role) for role in ("cart", "open", "offer", "compare")} - {None}
    if name not in id_tools or not st.row_id_key:
        return args, ""
    props = (st.arg_schemas.get(name) or {}).get("properties") or {}
    free = [
        key for key, spec in props.items()
        if isinstance(spec, dict) and ("string" in schema_types(spec) or "array" in schema_types(spec))
        and "enum" not in spec
    ]
    if len(free) != 1:
        return args, ""
    only = free[0]
    # only where the argument is known to carry ids: the cart tool, or a call already seen with one
    if name != st.role_of.get("cart") and (name, only) not in st.id_params:
        return args, ""
    if only not in args:
        return dict(args), ""
    given = args[only]
    if isinstance(given, list):
        repaired = [title_to_id(st, item) for item in given]
    else:
        repaired = [title_to_id(st, given)]
    complaint = next((note for _, note in repaired if note), "")
    if complaint:
        return args, complaint
    result = dict(args)
    result[only] = [item for item, _ in repaired] if isinstance(given, list) else repaired[0][0]
    return result, ""


# ---------- cart bookkeeping ----------


def cart_sets_count(st, name: str) -> bool:
    specs = (st.arg_schemas.get(name, {}).get("properties") or {}).values()
    for spec in specs:
        if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2:
            return False
    return any(isinstance(spec, dict) and "integer" in schema_types(spec) and spec.get("minimum") == 0
               for spec in specs)


def apply_cart_line(current: int, count: int | None, removal: bool, sets: bool) -> int:
    if removal:
        return current - count if count else 0
    if sets:
        return 1 if count is None else count
    return current + (1 if count is None else count)


def numeric_units(value: Any) -> int | None:
    if isinstance(value, str):
        return None
    return as_count(value)


def cart_args_remove(spec: dict[str, Any], args: dict[str, Any]) -> bool:
    hits = []
    for key, value in args.items():
        choices = (spec.get(key) or {}).get("enum")
        hits.append(isinstance(choices, list) and len(choices) >= 2 and value in choices[1:])
    return any(hits)


def cart_lines_named(st, spec: dict[str, Any], args: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
    loose = next((numeric_units(v) for v in args.values() if numeric_units(v) is not None), None)
    found: list[tuple[str, int | None]] = []
    whole_cart = False
    for key, value in args.items():
        if not isinstance(value, list):
            if isinstance(value, str) and value in st.listings:
                found.append((value, loose))
            continue
        element = (spec.get(key) or {}).get("items") or {}
        if not value and isinstance(element, dict) and (element.get("type") == "object" or "properties" in element):
            whole_cart = True
        for line in value:
            if isinstance(line, dict):
                whole_cart = True
                ident = next((v for v in line.values() if isinstance(v, str) and v in st.listings), None)
                units = next((numeric_units(v) for v in line.values() if numeric_units(v) is not None), 1)
                if ident:
                    found.append((ident, units))
            elif isinstance(line, str) and line in st.listings:
                found.append((line, loose))
    return found, whole_cart


def track_cart(st, name: str, args: dict[str, Any]) -> None:
    st.cart_changed = True
    st.cart_changed_step = st.steps_used
    spec = st.arg_schemas.get(name, {}).get("properties") or {}
    removal = cart_args_remove(spec, args)
    touched, replaces_all = cart_lines_named(st, spec, args)
    if replaces_all:
        st.basket = {ident: units for ident, units in touched if units and units > 0}
        return
    sets = cart_sets_count(st, name)
    for ident, units in touched:
        after = apply_cart_line(st.basket.get(ident, 0), units, removal, sets)
        if after <= 0:
            st.basket.pop(ident, None)
        else:
            st.basket[ident] = after


# ---------- result display and batched opens ----------


def show_result(st, observation: Any) -> str:
    rendered = to_json(clip(observation, SHOW_LIMIT))
    if len(rendered) <= RESULT_LIMIT:
        return rendered
    return rendered[:RESULT_LIMIT] + "...(cut)"


def side_by_side_spec(st) -> dict[str, Any]:
    return st.arg_schemas.get(st.role_of.get("compare") or "") or {}


def compare_param(st) -> str | None:
    schema = side_by_side_spec(st)
    array_key = schema_first_key_of(schema.get("properties") or {}, "array")
    if not array_key:
        return None
    if any(key != array_key for key in schema.get("required") or []):
        return None
    return array_key


def compare_capacity(st) -> int:
    props = side_by_side_spec(st).get("properties") or {}
    spec = props.get(compare_param(st) or "") or {}
    limit = spec.get("maxItems")
    # a store can fail above a width lower than its schema states: a miss learned from the replies caps the batch
    if not st.compare_misses:
        return limit if isinstance(limit, int) else max(4, st.compare_hits)
    learned = max(2, min(st.compare_misses - 1, max(st.compare_hits, 2)))
    return min(learned, limit) if isinstance(limit, int) else learned


def mergeable_opens(st, calls: list[dict[str, Any]], opener: str, id_param: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for position, call in enumerate(calls):
        function = call.get("function") or {}
        if function.get("name") != opener:
            continue
        raw = function.get("arguments")
        parsed = parse_json_object(raw) if isinstance(raw, str) else None
        if not parsed or set(parsed) != {id_param}:
            continue
        ident = str(parsed.get(id_param) or "").strip()
        if ident in st.listings and ident not in st.sold_step:
            found.append((position, ident))
    return found[:compare_capacity(st)]


def merge_opens(st, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    opener = st.role_of.get("open")
    joiner = st.role_of.get("compare")
    id_param = single_id_param(st, opener)
    array_key = compare_param(st)
    if not (opener and joiner and id_param and array_key) or opener == joiner:
        return calls
    # a comparison view without attributes would hide what the single opens reveal
    if compare_is_bare(st):
        return calls
    chosen = mergeable_opens(st, calls, opener, id_param)
    if len(chosen) < 2:
        return calls
    positions = [position for position, _ in chosen]
    merged_call = {
        "id": calls[positions[0]]["id"],
        "type": "function",
        "function": {"name": joiner, "arguments": to_json({array_key: [ident for _, ident in chosen]})},
    }
    rebuilt = [call if position not in positions else {"id": call["id"], "joined": True}
               for position, call in enumerate(calls)]
    rebuilt[positions[0]] = merged_call
    st.log.append({"kind": "joined", "ids": [ident for _, ident in chosen]})
    return rebuilt

# ---------- running tool calls ----------

NO_STEP = "Skipped, no step spent: "
HELD_NO_STEP = "On hold, no step spent yet. "


def tool_menu(st) -> list[str]:
    return sorted({spec["function"]["name"] for spec in offered_tools(st)})


def parse_call_args(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not (raw or "").strip():
        return {}
    return parse_json_object(raw)


def vet_call(st, call: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str]:
    spec = call.get("function") or {}
    name = spec.get("name")
    if st.done:
        return name, None, "Skipped: the episode is done."
    if name == st.role_of.get("ask"):
        return name, None, NO_STEP + "messages to the shopper go via Kyo's interviewer alone."
    if name not in st.arg_schemas:
        return name, None, NO_STEP + f"no such tool. Valid: {', '.join(tool_menu(st))}."
    usable = tool_menu(st)
    if name not in usable:
        return name, None, NO_STEP + "not enough steps remain for it; pick: " + ", ".join(usable) + "."
    args = parse_call_args(spec.get("arguments"))
    if args is None:
        return name, None, NO_STEP + "the args were not a single JSON object."
    problems: list[str] = []
    args = coerce(args, st.arg_schemas[name], "args", problems)
    if problems:
        return name, None, NO_STEP + "; ".join(problems[:6])
    args, bad_id = fix_ids(st, name, args)
    if bad_id:
        return name, None, NO_STEP + bad_id + "."
    return name, args, ""


def fit_compare_batch(st, args: dict[str, Any], many: str | None) -> tuple[dict[str, Any], str]:
    picked = args.get(many) if many else None
    if not isinstance(picked, list) or len(picked) <= compare_capacity(st) or not st.compare_misses:
        return args, ""
    room = compare_capacity(st)
    rest = ", ".join(map(str, picked[room:]))
    note = (f"\n[Kyo] Kept the first {room} listings, as the store compares at most {room}"
            f" per call. Still to be shown: {rest}.")
    return {**args, many: picked[:room]}, note


def unit_count_note(st, name: str, args: dict[str, Any], counts: dict[str, int]) -> str:
    # a count field may carry a misleading name (a "line" or "slot" number): when the units it would leave in the
    # cart differ from the item's wanted quantity, say so once before the call goes out
    if len(counts) != 1 or cart_args_remove((st.arg_schemas.get(name) or {}).get("properties") or {}, args):
        return ""
    key, count = next(iter(counts.items()))
    ident = next((value for value in args.values() if isinstance(value, str) and value in st.listings), None)
    pos = line_owner(st, ident) if ident else None
    if pos is None or count <= 0 or (ident, count) in st.count_checked:
        return ""
    wanted = st.wanted[pos].quantity
    sets = cart_sets_count(st, name)
    after = count if sets else st.basket.get(ident, 0) + count
    if after == wanted:
        return ""
    st.count_checked.add((ident, count))
    fix = wanted if sets else max(wanted - st.basket.get(ident, 0), 0)
    return (NO_STEP + f"{key} = {count} would leave {after} unit{'s' * (after != 1)} of {ident} in the cart, but item {pos + 1}"
            f" ({st.wanted[pos].name}) wants {wanted}. {key} is the number of units of that listing, not a line"
            f" or position number, so send the call again with {key} = {fix}"
            + (" (0 takes a listing out of the cart)." if sets else "."))


def gatekeep_call(st, name: str, args: dict[str, Any], target: str | None, ending: bool, batch_size: int,
                  mark: str) -> str:
    roles = st.role_of
    if target and name in (roles.get("open"), roles.get("offer")):
        wanted_id = str(args.get(target))
        if wanted_id in st.sold_step:
            return (NO_STEP + f"{args[target]} was gone by step {st.sold_step[wanted_id]} and will stay unavailable."
                    " Swap that cart line for another fitting listing.")
    if name == roles.get("cart"):
        counts = {key: as_count(value) for key, value in args.items() if not isinstance(value, str)}
        counts = {key: count for key, count in counts.items() if count is not None}
        lowering = any(count < 0 for count in counts.values())
        zeroed = bool(counts) and all(count == 0 for count in counts.values()) and not cart_args_remove(
            (st.arg_schemas.get(name) or {}).get("properties") or {}, args)
        if lowering or zeroed:
            fresh_view = st.cart_viewed_step >= st.cart_changed_step
            shown = to_json(st.cart_snapshot) if fresh_view else ""
            absent = [value for value in args.values()
                      if isinstance(value, str) and value in st.listings and value not in st.basket
                      and value not in shown and value not in st.cart_reply]
            if absent and lowering:
                return (NO_STEP + f"{absent[0]} has no line in the cart, so no units can be removed. The cart: "
                        + cart_summary(st))
            if absent:
                return (NO_STEP + f"{', '.join(counts)} = 0 puts no units of {absent[0]} in the cart: that number"
                        " is the count of units, so send the call again with it set to the units wanted.")
        miscount = unit_count_note(st, name, args, counts)
        if miscount:
            return miscount
    if ending:
        if batch_size > 1:
            return NO_STEP + "read the other results first, then send the finishing call by itself."
        return ""
    if steps_left(st) <= finish_reserve(st) and (roles.get("buy") or roles.get("refuse")):
        return (NO_STEP + "the remaining steps are reserved for checkout (once the shopper approves the basket,"
                " if required) or for ending with no purchase.")
    if name != roles.get("cart") and mark in st.recent_calls:
        return NO_STEP + "this exact call was just made and nothing changed; its reply is above."
    return ""


def stale_view_note(st, name: str, args: dict[str, Any], mark: str, target: str | None, readers: set) -> str:
    if name not in readers:
        return ""
    hit = st.page_cache.get(mark)
    if hit and hit[0] == st.needs_version:
        return (NO_STEP + f"the call made at step {hit[1]}, with nothing changed since, gave this reply:\n"
                + hit[2][:3000])
    if not target or name == st.role_of.get("compare"):
        return ""
    if not already_shown(st, name, str(args.get(target))):
        return ""
    return (NO_STEP + f"{args[target]} has shown all that this view offers since the last change."
            " Record kept: " + listing_record(st, str(args[target])))


def peek_cart_before_checkout(st) -> str:
    view_tool = st.role_of.get("cart_view")
    due = st.cart_changed and st.cart_viewed_step < st.cart_changed_step
    if not (due and view_tool and steps_left(st) > 3):
        return ""
    problems: list[str] = []
    view_args = coerce({}, st.arg_schemas.get(view_tool), "args", problems)
    if problems:
        return ""
    reply = st.store.act(view_tool, view_args)
    if reply is None:
        return ""
    st.cart_snapshot = reply.get("observation")
    st.cart_viewed_step = st.steps_used
    change = observe_result(st, view_tool, st.cart_snapshot)
    if not change:
        return ""
    st.queue = []
    return (HELD_NO_STEP + "Viewing the cart turned up a change: " + change + "\nThe cart: " + cart_summary(st)
            + "\nAllow for it, correct the cart if needed, then check out.")


def line_owner(st, ident: str) -> int | None:
    if len(st.wanted) == 1:
        return 0
    field = category_field(st)
    label = str((st.listings.get(ident) or {}).get(field) or "") if field else ""
    if not label:
        return None
    keys = product_match_keys(st)
    name = key_words(label)
    exact = [pos for pos, (shelf, title, _) in enumerate(keys) if name in (shelf, title)]
    if len(exact) == 1:
        return exact[0]
    loose = [pos for pos, key in enumerate(keys) if label_matches(label, [key])]
    return loose[0] if len(loose) == 1 else None


def over_cap_lines(st) -> list[str]:
    found = []
    for ident in cart_listing_ids(st):
        price = listing_price(st, ident)
        pos = line_owner(st, ident)
        if price is None or pos is None:
            continue
        cap = st.wanted[pos].price_cap
        if cap is not None and price > cap:
            found.append(f"{ident} costs {price:g} a unit, above the cap of {cap:g} for item {pos + 1} ({st.wanted[pos].name})")
    return found


def removal_hint(st) -> str:
    name = st.role_of.get("cart") or ""
    key = schema_first_key_of((st.arg_schemas.get(name) or {}).get("properties") or {}, "integer")
    if not (key and cart_sets_count(st, name)):
        return ""
    return f" With {name}, {key} sets a listing's units: {key} = 0 takes that listing out of the cart."


def split_items(st) -> list[str]:
    lines: dict[int, list[str]] = {}
    for ident in cart_listing_ids(st):
        pos = line_owner(st, ident)
        if pos is not None:
            lines.setdefault(pos, []).append(ident)
    problems = []
    for pos, ids in sorted(lines.items()):
        product = st.wanted[pos]
        if len(ids) > 1:
            problems.append(f"item {pos + 1} ({product.name}, wanted {product.quantity}) is spread over {len(ids)}"
                            " listings: " + ", ".join(ids))
            continue
        units = st.basket.get(ids[0], 0)
        if isinstance(units, (int, float)) and units != product.quantity:
            problems.append(f"item {pos + 1} ({product.name}) wants {product.quantity} units but the cart holds"
                            f" {units:g} of {ids[0]}")
    return problems


def uncovered_items(st) -> list[int]:
    # Kyo's cart log can drift from the store (a remove verb that takes off one unit at a time), so a listing the
    # store's last cart reply still shows counts as carted
    ids = list(dict.fromkeys(cart_listing_ids(st) + re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{5,40}", st.cart_reply)))
    ids = [ident for ident in ids if ident in st.listings]
    if not ids:
        return list(range(len(st.wanted))) if st.removed_ids else []
    if len(st.wanted) < 2:
        return []
    owners = [line_owner(st, ident) for ident in ids]
    if None in owners and len(ids) >= len(st.wanted):
        return []
    return [pos for pos in range(len(st.wanted)) if pos not in owners]


def gap_note(st, late: bool = False) -> str:
    # an item named like another requested item may be the brief splitting one product in two: never push a
    # second line for it
    names = [key_words(product.name) for product in st.wanted]
    gaps = [pos for pos in uncovered_items(st) if names.count(names[pos]) == 1]
    if not gaps or steps_left(st) < 3:
        return ""
    back = [ident for ident in st.removed_ids
            if not is_sold_out(st, ident) and line_owner(st, ident) in gaps]
    if late and not back:
        return ""
    st.gap_holds += 1
    st.log.append({"kind": "gap_held", "items": gaps})
    ids = cart_listing_ids(st)
    named = " or ".join(f"item {pos + 1} ({st.wanted[pos].name})" for pos in gaps)
    if len(ids) + len(gaps) > len(st.wanted):
        named = f"one of the requested items ({len(ids)} lines for {len(st.wanted)} items; likely {named})"
    text = (HELD_NO_STEP + "The cart has no line for " + named
            + ", and an order that leaves out a requested item fails the whole request. ")
    if back:
        text += (f"{', '.join(back)} was in the cart for it before: put it back at the wanted count unless its"
                 " record plainly breaks one of that item's musts, or add a listing that fits better, then check out.")
    else:
        text += "Add that item's best fitting listing you have seen at the wanted count, then check out."
    return text + "\nThe cart: " + cart_summary(st)


def checkout_hold(st) -> str:
    if elapsed(st) >= CLOCK_STOP:
        return ""
    # past the late mark keep only the checks made in code, whose fix is a cart call or two;
    # store re-reads, wider searches and LLM reviews no longer fit in the time left
    late = elapsed(st) >= CLOCK_LATE
    reread = ""
    if not late:
        wider = hold_wider(st)
        if wider:
            return wider + "\nThe cart: " + cart_summary(st)
        reread = refresh_offers(st)
        cart_change = peek_cart_before_checkout(st)
        if cart_change:
            return cart_change
    swap = swap_note(st)
    gone_lines = [ident for ident in cart_listing_ids(st) if ident in st.sold_step] if st.holds < 2 else []
    if swap or gone_lines:
        if not swap:
            st.holds += 1
        reason = swap or f"{', '.join(gone_lines)} in the cart is gone for good: replace that first."
        return HELD_NO_STEP + reason + "\nThe cart: " + cart_summary(st)
    split = split_items(st) if st.shape_holds < 2 else []
    if split:
        st.shape_holds += 1
        return (HELD_NO_STEP + "; ".join(split) + ". Each item is ONE listing carted at its wanted quantity"
                " (3 wanted = 3 units of one listing): keep the best fitting listing per item, bring its count to the"
                " wanted units (a cart tool that adds one unit per call needs one call per unit), remove the rest."
                + removal_hint(st)
                + "\nThe cart: " + cart_summary(st))
    gap = gap_note(st, late) if st.gap_holds < 1 else ""
    if gap:
        return gap
    over = over_cap_lines(st) if st.cap_holds < 2 else []
    if over:
        st.cap_holds += 1
        return (HELD_NO_STEP + "; ".join(over) + ". Swap each for a fitting listing within its cap."
                + "\nThe cart: " + cart_summary(st))
    if late:
        return ""
    chain, stop = follow_trails(st)
    if stop:
        return HELD_NO_STEP + chain + "\nThe cart: " + cart_summary(st)
    looked = fetch_attributes(st, cart_listing_ids(st), finish_reserve(st) + 3) + cheaper_candidates(st)
    verdict = review_cart(st)
    if verdict:
        shown = ""
        if looked:
            shown = ("To check, Kyo read the attributes of these items, one step apiece:\n"
                     + "\n".join(looked) + "\n")
        return "Put on hold. " + chain + shown + verdict + "\nThe cart: " + cart_summary(st)
    if reread and st.spend > SOFT_SPEND:
        return (HELD_NO_STEP + "Kyo has just re-read the cart's listings:\n" + reread
                + "\nTry checkout again if every line still has stock and a price within the budget;"
                " if any fails, swap that line first.")
    return ""


def decline_review(st) -> str:
    if steps_left(st) <= 5:
        return ""
    review = audit_decline(st)
    if review:
        return HELD_NO_STEP + review
    if not st.pointers or steps_left(st) <= 4:
        return ""
    if st.refuse_stops == 0:
        st.refuse_stops = 1
        st.log.append({"kind": "refuse_blocked", "leads": list(st.pointers)})
        return (HELD_NO_STEP + "The review turned up listings that might fit: " + ", ".join(st.pointers)
                + ". Open them as needed, add the best-fitting one to the cart (following the pick rule) and"
                " check out. End with no purchase only if all of them plainly fail a requirement; then name which"
                " one in your text and send the ending call once more.")
    if st.refuse_stops != 1:
        return ""
    st.refuse_stops = 2
    lead = st.pointers[0]
    review_cart(st, [lead], HARD_SPEND)
    if not (st.verdict_by_cart.get(check_signature(st, [lead])) and lead in st.pointers):
        return ""
    st.confirm_mark = lead
    st.log.append({"kind": "refuse_blocked", "confirmed": lead})
    return (HELD_NO_STEP + f"A separate check run on {lead}'s records says it fulfills each requirement."
            " Put it in the cart at the requested quantity and check out; end with no purchase only if its record"
            " plainly breaks a requirement.")


def late_decline_hold(st) -> str:
    if steps_left(st) > 5 or st.late_hold:
        return ""
    lines = [ident for ident in st.staged if not is_sold_out(st, ident)]
    fresh = st.cart_snapshot is not None and st.cart_viewed_step >= st.cart_changed_step
    missing = [ident for ident in lines
               if ident not in cart_listing_ids(st) or (ident in st.removed_ids and not fresh)]
    if len(lines) < len(st.wanted) or len(missing) + 1 > steps_left(st):
        return ""
    st.late_hold = True
    st.log.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
    if missing:
        redo = f"Add {', '.join(missing)} to the cart again at the requested counts and then check out."
    else:
        redo = "Check out the cart you have."
    return (HELD_NO_STEP + "When the store has a listing that fits, ending with no purchase earns nothing, and running"
            " short on steps is no proof that it lacks one. " + redo + " End with no purchase only if your search"
            " showed that some requested item has no listing meeting its requirements; then send the ending call"
            " once again.")


def decline_nudge(st) -> str:
    if st.pointers or st.nudged or steps_left(st) <= st.step_cap // 3:
        return ""
    st.nudged = True
    return (HELD_NO_STEP + f"{steps_left(st)} steps still remain, so before ending try search approaches you have not"
            " used yet: browse each requested item's category and page through it using the token or offset"
            " the store gives back, try one-word value queries and other names for the property, and open listings"
            " that could fit. Only after that end with no purchase.")


def judge_compare_width(st, args: dict[str, Any], many: str | None, observation: Any) -> str:
    picked = args.get(many) if many else None
    if not isinstance(picked, list) or len(picked) <= 1:
        return ""
    dumped = to_json(observation)
    if any(str(ident) in dumped for ident in picked):
        st.compare_hits = max(st.compare_hits, len(picked))
        return ""
    st.compare_misses = min(st.compare_misses or 999, len(picked))
    return (f"\n[Kyo] The comparison view returned nothing for {len(picked)} listings;"
            f" resend up to {compare_capacity(st)} per call.")


def remember_id_params(st, name: str, args: dict[str, Any]) -> None:
    for key, value in args.items():
        values = value if isinstance(value, list) else [value]
        if not values:
            continue
        if all(isinstance(entry, str) and entry in st.listings for entry in values):
            st.id_params.add((name, key))


def note_reader_fields(st, name: str, observation: Any, readers: set) -> None:
    if name == st.role_of.get("compare"):
        found = listing_records(st, observation)
        if found and not st.shows_facts.get(name, False):
            st.shows_facts[name] = any(attribute_block(record) for _, record in found)
    if name in readers and name != st.role_of.get("search"):
        remember_view(st, name, observation)


def learn_single_view(st, name: str, observation: Any) -> None:
    roles = st.role_of
    skipped = (roles.get("cart"), roles.get("buy"), roles.get("refuse"), roles.get("ask"))
    if not single_id_param(st, name) or name in skipped:
        return
    listing = find_id(st, observation) if isinstance(observation, dict) else None
    if not listing:
        return
    if not st.shows_facts.get(name, False):
        st.shows_facts[name] = bool(attribute_block(observation))
    if name == roles.get("open") and not st.shows_facts[name]:
        barred = skipped + (roles.get("search"),)
        substitute = None
        for tool in st.arg_schemas:
            if tool == name or not single_id_param(st, tool):
                continue
            if st.shows_facts.get(tool) is not False and tool not in barred:
                substitute = tool
                break
        if substitute:
            roles["open"] = substitute
    if not st.shows_facts[name] and name != roles.get("open") and not roles.get("offer"):
        roles["offer"] = name
    if name == offers_tool(st):
        st.offer_step[listing] = st.steps_used


def book_cart_call(st, name: str, args: dict[str, Any], observation: Any) -> None:
    track_cart(st, name, args)
    st.cart_reply = to_json(observation)
    st.recent_calls = []
    named = [value for value in args.values() if isinstance(value, str) and value in st.listings]
    if cart_args_remove((st.arg_schemas.get(name) or {}).get("properties") or {}, args):
        st.removed_ids.update(value for value in named if not rows_hold_id(observation, value))
    else:
        st.removed_ids.difference_update(value for value in named if rows_hold_id(observation, value))
    counts = [as_count(value) for value in args.values()
              if not isinstance(value, str) and as_count(value) is not None]
    if counts:
        first = counts[0]
        for value in args.values():
            if not isinstance(value, str) or value not in st.listings:
                continue
            if first <= 0 and not rows_hold_id(observation, value):
                st.removed_ids.add(value)
            elif first > 0:
                st.removed_ids.discard(value)
    kept = [ident for ident in cart_listing_ids(st) if not is_sold_out(st, ident) and ident not in st.removed_ids]
    if len(kept) >= len(st.staged):
        st.staged = kept


def append_followups(st, name: str, args: dict[str, Any], reply: dict[str, Any], observation: Any) -> str:
    extra = ""
    proof = evidence_text(st, observation)
    if proof:
        extra += "\n" + proof
    if name == st.role_of.get("cart"):
        zeroed = any(as_count(value) == 0 for value in args.values() if not isinstance(value, str))
        if zeroed and any(isinstance(value, str) and value in st.listings and rows_hold_id(observation, value)
                          for value in args.values()):
            extra += ("\n[Kyo] A count of 0 did not remove that line: this tool adds the given count,"
                      " so send a negative count to take units away.")
    st.sold_noted = False
    stray = stray_lines(st, observation)
    if stray:
        meaning = apply_note(st, stray)
        if meaning:
            extra += "\n[shop note] " + meaning + ". Adjust your picks to match this before checking out."
    refused = is_checkout(st, name, args) and not reply.get("finished")
    if refused and (st.sold_noted or any(line in shopper_words(st, observation) for line in st.sold_ids)):
        sold = sold_note(st)
        if sold:
            extra += "\n[Kyo rechecked the cart lines after the refusal]\n" + sold
    swap = swap_note(st)
    if swap:
        extra += "\n" + swap
    if st.queue:
        extra += "\n[notes left on Kyo's own calls] " + " | ".join(st.queue)
        st.queue = []
    return extra


def compose_reply(st, name: str, args: dict[str, Any], reply: dict[str, Any], tail_note: str, mark: str,
                  searching: bool, readers: set) -> str:
    observation = reply.get("observation")
    text = show_result(st, observation) + tail_note
    if searching:
        text += cheapest_note(st, [observation] + st.page_results)
    plain_text = text
    text += append_followups(st, name, args, reply, observation)
    if name in readers:
        st.page_cache[mark] = (st.needs_version, st.steps_used, plain_text)
    if reply.get("finished"):
        text += " [episode finished]"
    return text


def handle_tool_call(st, call: dict[str, Any], batch_size: int) -> str:
    if call.get("joined"):
        return "Merged into the comparison call above: Kyo viewed all these listings in one joint step."
    name, args, refusal = vet_call(st, call)
    if refusal:
        return refusal
    roles = st.role_of
    many = compare_param(st) if name == roles.get("compare") else None
    args, notes = fit_compare_batch(st, args, many)
    target = single_id_param(st, name)
    ending = is_checkout(st, name, args) or is_decline(st, name, args)
    mark = name + to_json(args)
    refusal = gatekeep_call(st, name, args, target, ending, batch_size, mark)
    if refusal:
        return refusal
    readers = {roles.get(job) for job in ("search", "open", "offer", "compare")} - {None}
    refusal = stale_view_note(st, name, args, mark, target, readers)
    if refusal:
        return refusal
    picked = args.get(many) if many else None
    if isinstance(picked, list) and picked:
        fresh = [ident for ident in picked if not already_shown(st, name, str(ident))]
        if not fresh:
            records = "\n".join(f"{ident}: {listing_record(st, str(ident))}" for ident in picked)
            return (NO_STEP + "these listings have already shown all this view offers since the last change."
                    " The records:\n" + records[:4000])
        if len(fresh) < len(picked):
            dropped = ", ".join(str(ident) for ident in picked if ident not in fresh)
            notes += f"\n[Kyo] Omitting {dropped}: already fully shown and unchanged since then."
            args = {**args, many: fresh}
            mark = name + to_json(args)
    if ending and steps_left(st) > 2:
        hold = checkout_hold(st) if is_checkout(st, name, args) else decline_review(st)
        if hold:
            return hold
    if name == roles.get("cart") and not cart_listing_ids(st) and steps_left(st) > 3 and gate_pending(st, "choose"):
        passed, said = process_gate(st, "choose")
        st.gate_passed.add(("choose", cart_signature(st) + f"#{st.needs_version}"))
        if not passed:
            return "Put on hold. " + said + "\nThe cart: " + cart_summary(st)
        if said:
            notes = "\n[Kyo] " + said + notes
    if is_decline(st, name, args):
        hold = late_decline_hold(st) or decline_nudge(st)
        if hold:
            return hold
        filled = fill_cart(st) or audit_order(st) or closest_order(st)
        if filled:
            return filled + (" [episode finished]" if st.done else "")
        args = quiet_args(st, name, args)
    if ending and steps_left(st) >= 2:
        passed, said = process_gate(st, "order" if is_checkout(st, name, args) else "end")
        if not passed:
            return "Put on hold. " + said + "\nThe cart: " + cart_summary(st)
        if said:
            notes = "\n[Kyo] " + said + notes
    reply = st.store.act(name, args)
    if reply is None:
        return "The call did not go through (network/server). Retry it."
    observation = reply.get("observation")
    notes += judge_compare_width(st, args, many, observation)
    st.recent_calls = (st.recent_calls + [mark])[-3:]
    remember_id_params(st, name, args)
    absorb_result(st, observation)
    note_reader_fields(st, name, observation, readers)
    non_search = {tool for job, tool in roles.items() if job != "search"}
    searching = name not in non_search and not ending
    paged = read_more_pages(st, name, args, observation) if searching else ""
    learn_single_view(st, name, observation)
    if name == roles.get("cart"):
        book_cart_call(st, name, args, observation)
    if name == roles.get("cart_view"):
        st.cart_snapshot = observation
        st.cart_viewed_step = st.steps_used
    if is_checkout(st, name, args):
        st.buy_tries += 1
        if not reply.get("finished"):
            st.refuse_step = st.steps_used
    return compose_reply(st, name, args, reply, paged + notes, mark, searching, readers)


def run_tool_calls(st, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    batch = merge_opens(st, calls)
    return [(call["id"], handle_tool_call(st, call, len(batch))) for call in batch]


# ---------- forced finish ----------


def episode_over(st) -> bool:
    return st.done or steps_left(st) <= 0


def cart_worth_buying(st) -> bool:
    if st.cart_snapshot is not None and st.cart_viewed_step >= st.cart_changed_step:
        return bool(cart_listing_ids(st))
    return bool(st.basket) or (st.cart_changed and steps_left(st) >= 2)


def send_forced_end(st, tool: str, seed: dict[str, Any], stage: str, kind: str) -> tuple[bool, Any]:
    problems: list[str] = []
    args = coerce(seed, st.arg_schemas.get(tool), "args", problems)
    if problems:
        return False, None
    if steps_left(st) >= 2:
        process_gate(st, stage)
    if episode_over(st):
        return True, None
    reply = st.store.act(tool, args)
    st.log.append({"kind": kind, "reply": clip(reply, 200)})
    return True, reply


def finish_episode(st, again: bool = False) -> str:
    if st.done or st.out_of_funds or steps_left(st) <= 0:
        return ""
    buy_tool = st.role_of.get("buy")
    decline_tool = st.role_of.get("refuse")
    if buy_tool and cart_worth_buying(st):
        sent, reply = send_forced_end(st, buy_tool, dict(st.plan.get("buy_args") or {}), "order", "forced_buy")
        if sent and episode_over(st):
            return ""
        if sent and reply is not None and again:
            st.buy_tries += 1
            st.refuse_step = st.steps_used
            observe_result(st, buy_tool, reply.get("observation"))
            return "checkout was turned down: " + show_result(st, reply.get("observation"))
    if not (buy_tool and cart_worth_buying(st)):
        # the loop ran out with no cart worth buying: before the forced decline, the cart code can build itself
        built = audit_order(st) or closest_order(st)
        if built:
            return "" if st.done else built
    if not decline_tool:
        return ""
    seed = quiet_args(st, decline_tool, dict(st.plan.get("refuse_args") or {}))
    sent, reply = send_forced_end(st, decline_tool, seed, "end", "forced_refuse")
    if not sent or reply is None or st.done:
        return ""
    observe_result(st, decline_tool, reply.get("observation"))
    return "the store would not end without a buy: " + show_result(st, reply.get("observation"))
# ---------- history ----------


def trim_history(st) -> None:
    total = sum(len(to_json(turn.get("content"))) for turn in st.messages)
    if total <= CHAT_LIMIT:
        return
    # each trim rewrites the chat's prefix and so costs one uncached call: cut well below the limit, oldest first,
    # so trims stay rare instead of nibbling one more message on every turn
    tool_turns = [pos for pos, turn in enumerate(st.messages) if turn.get("role") == "tool"]
    for pos in tool_turns[:-6]:
        if total <= 0.6 * CHAT_LIMIT:
            break
        text = st.messages[pos].get("content") or ""
        if len(text) <= 600:
            continue
        found = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', text)
        known = [ident for ident in dict.fromkeys(found) if ident in st.listings]
        cut = text[:400] + f"...(cut down to save space; ids it carried: {', '.join(known[:20])})"
        total -= len(to_json(text)) - len(to_json(cut))
        st.messages[pos]["content"] = cut


def tool_call_list(st, message: dict[str, Any]) -> list[dict[str, Any]]:
    built = []
    for number, raw_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call["function"] if isinstance(raw_call.get("function"), dict) else {}
        raw_args = function.get("arguments")
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            args = parse_json_object(raw_args)
        else:
            args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = str(raw_call.get("id") or f"k{st.turns}_{number}")
        built.append({
            "id": call_id,
            "type": "function",
            "function": {"name": str(function.get("name") or ""), "arguments": to_json(args)},
        })
    return built


def reply_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return to_json(content) if content else ""


def commits_early(st, calls: list[dict[str, Any]]) -> bool:
    if st.interviewed or not st.role_of.get("ask") or free_messages(st) <= 0:
        return False
    final_tools = {st.role_of.get("cart"), st.role_of.get("buy"), st.role_of.get("refuse")} - {None}
    if not any(call["function"]["name"] in final_tools for call in calls):
        return False
    return steps_left(st) > 8


# ---------- driver ----------


class Shopper:
    def __init__(self, problem: dict[str, Any]):
        self.st = State(problem)
        self.st.llm = Brain(self.st)
        self.st.store = Store(self.st)

    def say(self, text: str) -> None:
        self.st.messages.append({"role": "user", "content": text})

    def run(self) -> list[dict[str, Any]]:
        st = self.st
        if st.done:
            return st.log
        parse_request(st)
        st.log.append({"kind": "brief", "brief": st.plan})
        self.open_chat()
        self.work(TURN_LIMIT, HARD_SPEND)
        if not st.done:
            self.close_out()
        st.log.append({
            "kind": "end", "finished": st.done, "steps": st.steps_used,
            "cost": round(st.spend, 5), "calls": st.llm_calls,
        })
        return st.log

    def open_chat(self) -> None:
        st = self.st
        first_view = st.policy.get("opening")
        if first_view is not None:
            absorb_result(st, first_view)
        parts = ["SHOPPER'S REQUEST:\n" + st.query]
        if first_view is not None:
            parts.append("WHAT THE STORE SHOWS FIRST:\n" + show_result(st, first_view))
        parts.append(
            "Begin with a category search for every item and open some listings in full, so that the interviewer "
            "learns which properties these listings really carry.\n\n" + status_text(st)
        )
        st.messages = [{"role": "system", "content": LOOP_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]

    def close_out(self) -> None:
        st = self.st
        refusal = finish_episode(st, again=True)
        second_try = (refusal and not st.done and not st.out_of_funds and not st.llm_broke and steps_left(st) > 1
                      and elapsed(st) < CLOCK_STOP)
        if second_try:
            self.say(
                "[Kyo] Kyo went to end the episode, and " + refusal
                + "\nAct on the store reply, then finish up.\n" + status_text(st)
            )
            self.work(st.turns + 10, HARD_SPEND + 0.01)
        if not st.done:
            finish_episode(st)

    def may_continue(self, turn_cap: int, spend_cap: float) -> bool:
        st = self.st
        if st.done or st.out_of_funds or steps_left(st) <= 0 or elapsed(st) >= CLOCK_STOP:
            return False
        return st.turns < turn_cap and st.spend < spend_cap

    def work(self, turn_cap: int, spend_cap: float) -> None:
        st = self.st
        idle = 0
        while self.may_continue(turn_cap, spend_cap):
            if should_interview(st):
                self.interview_turn()
                continue
            trim_history(st)
            st.turns += 1
            message = st.llm.think(st.messages, [FAST_LLM, BACKUP_LLM], 3000, offered_tools(st))
            if message is None:
                st.empty_replies += 1
                if st.llm_broke or st.llm_calls >= CALL_LIMIT or st.empty_replies > 2:
                    return
                continue
            calls = tool_call_list(st, message)
            text = reply_text(message)
            if calls:
                idle = 0
                if commits_early(st, calls):
                    self.ask_first_turn(text, calls)
                else:
                    self.tool_turn(text, calls)
                continue
            idle += 1
            self.idle_turn(text)
            if idle >= 3:
                return

    def interview_turn(self) -> None:
        summary = run_interview(self.st)
        if summary:
            self.say(summary + "\n" + status_text(self.st))

    def idle_turn(self, text: str) -> None:
        self.st.messages.append({"role": "assistant", "content": text or "(no call)"})
        self.say("Reply with one tool call to go on.\n" + status_text(self.st))

    def ask_first_turn(self, text: str, calls: list[dict[str, Any]]) -> None:
        st = self.st
        st.messages.append({"role": "assistant", "content": text or None, "tool_calls": calls})
        st.messages.extend(
            {"role": "tool", "tool_call_id": call["id"],
             "content": "Skipped, no step spent: the shopper gets asked now."}
            for call in calls
        )
        summary = run_interview(st)
        self.say((summary or "The interviewer found no questions.") + "\n" + status_text(st))

    def tool_turn(self, text: str, calls: list[dict[str, Any]]) -> None:
        st = self.st
        st.messages.append({"role": "assistant", "content": text or None, "tool_calls": calls})
        st.log.append({"kind": "turn", "text": text[:300], "calls": [call["function"] for call in calls]})
        replies = run_tool_calls(st, calls)
        last = len(replies) - 1
        for pos, (call_id, body) in enumerate(replies):
            if pos == last:
                body += "\n" + status_text(st)
            st.messages.append({"role": "tool", "tool_call_id": call_id, "content": body})


def crash_entry(exc: BaseException) -> dict[str, Any]:
    return {"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        shopper = Shopper(problem_data)
    except Exception as exc:
        return [crash_entry(exc)]
    try:
        return shopper.run()
    except Exception as exc:
        shopper.st.log.append(crash_entry(exc))
        try:
            if not shopper.st.done:
                finish_episode(shopper.st)
        except Exception as again:
            shopper.st.log.append(crash_entry(again))
        return shopper.st.log


__all__ = ["agent_main"]
