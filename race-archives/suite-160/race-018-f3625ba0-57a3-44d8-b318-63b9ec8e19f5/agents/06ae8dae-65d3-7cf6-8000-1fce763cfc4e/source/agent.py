from __future__ import annotations

import collections
import itertools
import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT, HEAVY, SPARE = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"

LOOP_PER_STEP, ALL_PER_STEP, LOOP_FLOOR, ALL_FLOOR = 0.9, 1.3, 42, 60
SPEND_LIGHT, SPEND_STOP, SPEND_OPEN = 0.05, 0.055, 0.07
DUTY_SPEND = 0.005
DEADLINE, CLOSING, LAST_CALL = 255.0, 60.0, 30.0
WAITS, WAIT_TOTAL, HOT_SECONDS = (4, 10, 20, 35), 45, 25
THINK = {"walk": (2048, 1024, 16.0), "plan": (1024, 1024, 16.0), "check": (2048, 1024, 16.0), "gate": (2048, 512, 16.0),
         "unmix": (1024, 512, 8.0), "trail": (2048, 2048, 32.0), "brief": (4096, 1024, 32.0), "audit": (4096, 2048, 32.0),
         "swap": (4096, 512, 32.0), "basket": (4096, 1024, 32.0), "read": (0, 512, 8.0)}
QUICK_SECONDS, WALK_SLOW, WALK_LATE = 16.0, 10.0, 120.0
COVERED = 4
FIRST_TRY = 3
WIDE_EACH = 3
SHOW_CHARS, OBS_LIMIT, HISTORY_LIMIT = 450, 15000, 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=30, max_retries=2, retry_delay=1)


READING = """the listing's attribute for a property decides when its key names that same property under that name or a plain synonym (any key spelling; one element of a comma or slash list is enough; case, singular or plural, plain synonyms and unit conversions count; a value that contains the required value's words counts, such as "deep teal" for teal, while a value whose words differ does not, as "teal green" is not "sea green", and an abbreviation or code is not the word it stands for); if that attribute says something else, the listing fails whatever other keys or its title say; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names the property, the wanted value stated under another attribute or in the title or description counts, but a number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures; limits include the edge value and a range must lie wholly inside a limit; a requirement the category label itself states is met by that category, and the category label must match the requested item as a whole, never a neighbouring one; a rule of being unlike another listing's value is met unless the records show that same value, and is unknown while that listing is unread; a value on another listing that is not yet read is unknown, so it neither passes nor fails a listing; the newest record of a listing wins; a listing shown as sold out does not fit when the ledger says in stock only; a listing whose own price times its quantity breaks the budget does not fit, and an order's total over every line (each line on its own when the budget is per item) may not exceed it; equal is fine."""


GUIDE = """You are a careful personal shopper buying for one customer in an online store. You act only by calling the store's tools. A desk works beside you: it keeps the ledger shown after your results, asks the shopper the questions and checks the cart before it is ordered. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing, but each of your turns takes seconds of a short clock. Put the calls you would make anyway and that do not need each other's results into one turn, two or three at a time, in the order they should run: searches for different items or in different wordings, opens of the promising rows, the cart lines of several items. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once the desk knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by the desk's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. The desk carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through the desk's interviewer).

WHAT COUNTS AS A FIT
- A listing meets a requirement by these reading rules: """ + READING + """ The desk adds [evidence] lines after opened listings with the keys whose names match a requirement.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold, for each item, the lowest-priced listing among ALL listings that fit it (for items that must work together, the cheapest pair that works). Find the whole fitting set (page through it: result rows are often in no price order, and the desk reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only on proof: each requested item searched in its own category in several ways (its plain name, other names, the wanted value alone, further pages) and every promising row opened, with none meeting everything. Running short of steps or time proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "call": a short everyday name for it to use when talking to the shopper (one or two words, never "and" or "&"), "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at (a listing outside the order, picked out by its shop, brand or words), else []. A requirement that one requested item fit or match another requested item belongs in "fits", not here.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"fits": a list with one entry per requested item that must work with, fit or suit another requested item, each {"item": that item's number in "items" (1 for the first), "with": the number of the item it must fit}; [] when the request asks for no such pairing.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none); a question the shopper puts to you about the request itself (whether it is clear, whether to say more) is not one.
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"cart_add_args": the fixed arguments that make the "cart" tool put units in (for a tool with a mode argument, that mode), else {}.
"cart_remove_args": the fixed arguments that make the "cart" tool take units off, else {}.
"price_cap": {"tool": the search tool, "arg": its argument that keeps only listings at or below a price} when a search tool has one, else null.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of the category's opened listings carry the key and example values. An unwritten preference is almost always on a property the item's listings carry as a key, though the list shows only the listings opened so far.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by the item's listed key it points at when one fits (a hint about looks points at whichever looks key the item's listings carry), else by its plain name.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key whose one value lists several purchase choices at once (a mix of sizes, colours or pack counts), identifiers such as model or part numbers, and warranty; how many units come in one pack and the services a product comes with are real properties. With several items, put every item's hinted properties first, then share the remaining questions between the items, starting with the item the request settles least about.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: """ + READING + """ Each requested item must appear as a separate cart line at the requested quantity, and nothing else may be in the cart.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule (for the cheapest, a lower order total; for items that must work together, only as a pair that works), the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


AUDIT = """A shopping agent wants to end without buying because it thinks some requested item has nothing that fits. The listing records it has seen are grouped under the item whose kind they are (OTHER for the rest): OPENED records had their page with the attributes read; ROW records were seen only as a search result row, so they can be neither passed nor failed on a property the row does not show. Check every item's records against its requirements with the reading rules given below.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement of that item, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "blocked": {"item number": "for an item with no fitting record, the requirement none of its records meets"}, "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ATTRIBUTES: only its attribute table was read, not its page; ROW: seen only as a search result row).
A record is a link only when it agrees with every detail the description gives of that listing (its shop or seller, its brand, its kind of product, each value named with it). Judge each detail by the record's attribute for that property first; its title or description counts only where no attribute names it, and synonyms and unit conversions agree. Whether the listing is in stock does not matter for a link. A record that contradicts one of them is not the link, even when no other record comes close; one that leaves a detail unshown is only a "maybe".
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below agrees with every detail of that link, else "", "about": how the requirement or the previous page describes that listing, "search": for a link with no id, three to six words that find it with the store's search: its shop or brand name, its kind of product and one or two of the values the description gives it (a part or model code alone when one is given); fewer words when SEARCHES TRIED shows such a search found nothing; else "", "maybe": for a link with no id, the ids of ROW records that could be it but whose attributes must be read to tell (at most 4), else []}], "done": true when every link has an id, each of their pages is OPENED and the last one states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

BASKET = """A shopping agent is about to check out a cart for the request below. Judge which of the opened listing records could go in the order, with these reading rules: """ + READING + """
For each requested item, list every record of its kind that meets all of that item's requirements in force. For each pair under MUST WORK TOGETHER, list the pairs of fitting records that work together by their own words (a model, series or code one record lists for the other, or matching sizes and standards both state); a pair the records say nothing about does not count.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids]}, "pairs": [{"item": item number, "with": the other item's number, "ids": [the item's listing id, the other item's listing id]}], "reason": "one sentence"}"""

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
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str], call: str = ""):
        self.name = name
        self.call = call if call and not re.search(r"\band\b|&", call) else name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class ShoppingAssistant:
    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
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
        self.metered: set[str] = set()
        self.finishing = False
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.hot_until = 0.0
        self.walk_secs: list[float] = []
        self.heavy_paid = 0.0
        self.mode_seen: dict[Any, int] = {}
        self.stray_held: set[str] = set()
        self.basket_marks: set[str] = set()
        self.trace: list[dict[str, Any]] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.fits: list[tuple[int, int]] = []
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
        self.check_said: dict[str, str] = {}
        self.restated: dict[str, int] = {}
        self.better: dict[str, list[str]] = {}
        self.check_mark: dict[str, tuple[int, int]] = {}
        self.audited: set[tuple[int, int, int]] = set()
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.swap_for: dict[str, list[str]] = {}
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.leads: list[str] = []
        self.pushes: dict[int, int] = {}
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.blocked: set[str] = set()
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
        self.trail_searched: dict[str, int] = {}
        self.trailed_early = False
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

    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or self.compare_room() == 0 or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def seconds_left(self) -> float:
        return DEADLINE - (time.monotonic() - self.started)

    def spend_stop(self) -> float:
        return SPEND_STOP if len(self.cart_ids()) >= len(self.items) else SPEND_OPEN

    def seat(self, models: list[str], job: str = "") -> list[str]:
        stop = self.spend_stop()
        if self.cost >= stop + (DUTY_SPEND if job == "gate" else 0.0) or self.seconds_left() < (10.0 if self.finishing else LAST_CALL) or self.no_credit:
            return []
        if (self.cost >= SPEND_LIGHT and job != "audit") or self.cost + self.heavy_paid >= stop:
            models = [m for m in models if m != HEAVY] or [LIGHT]
        models = [m for m in dict.fromkeys(models) if m != SPARE]
        return [SPARE] + models if time.monotonic() < self.hot_until else models + [SPARE]

    def think(self, messages: list[dict[str, Any]], models: list[str], tools: list[dict[str, Any]] | None = None,
              seat: str = "walk") -> dict[str, Any] | None:
        budget, _, seconds = THINK[seat]
        slow = len(self.walk_secs) >= 3 and sum(self.walk_secs[-3:]) / 3 > WALK_SLOW
        wanted = budget > 0 and not (seat == "walk" and (slow or self.seconds_left() < WALK_LATE))
        for model in models:
            if model not in self.seat([model], seat):
                continue
            if self.calls >= max(ALL_FLOOR, int(self.max_steps * ALL_PER_STEP)):
                return None
            self.calls += 1
            tries = 0
            thinking = wanted and model != SPARE and self.seconds_left() >= max(CLOSING, seconds + LAST_CALL + 5)
            while True:
                began = time.monotonic()
                message, status = self.call_model(model, messages, tools, seat, thinking)
                if seat == "walk" and thinking and status in (200, "cut", None):
                    self.walk_secs.append(time.monotonic() - began)
                if message is not None:
                    return message
                if status == "cut" or (status is None and thinking):
                    if model not in self.seat([model], seat) or self.calls >= max(ALL_FLOOR, int(self.max_steps * ALL_PER_STEP)):
                        return None
                    self.calls += 1
                    thinking, tries = False, 1
                    continue
                if status == 429 and model != SPARE:
                    self.hot_until = time.monotonic() + HOT_SECONDS
                    break
                proxy = status == "proxy"
                passing = proxy or status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                pause = (2 ** tries if proxy else WAITS[min(tries, len(WAITS) - 1)]) * (0.8 + 0.4 * random.random())
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + pause > WAIT_TOTAL
                        or self.seconds_left() - pause < LAST_CALL):
                    if proxy:
                        return None
                    break
                self.waited += pause
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                   seat: str, thinking: bool) -> tuple[dict[str, Any] | None, Any]:
        budget, room, seconds = THINK[seat]
        seconds = seconds if thinking or model == HEAVY else min(seconds, QUICK_SECONDS)
        _llm.timeout = max(5, min(int(seconds), int(self.seconds_left() - LAST_CALL / 2)))
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": room + budget if thinking else room}
        body["reasoning"] = {"max_tokens": budget} if thinking else {"effort": "low" if model == HEAVY else "none"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost") if usage.get("is_byok") else None
            bill = sum(float(v) for v in (usage.get("cost"), upstream) if isinstance(v, (int, float)) and not isinstance(v, bool))
            self.cost += bill
            self.heavy_paid = max(self.heavy_paid, bill) if model == HEAVY else self.heavy_paid
            message = data["choices"][0].get("message") or {}
            cut = data["choices"][0].get("finish_reason") == "length" and thinking
            if not cut and (message.get("content") or message.get("tool_calls")):
                return message, 200
            self.trace.append({"kind": "llm_fail", "model": model, "status": "cut" if cut else 0})
            return None, "cut" if cut else 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        elif status == 429 and "<html" in str(error.get("body") or "").lower():
            status = "proxy"
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], seat: str) -> dict[str, Any] | None:
        tried: set[str] = set()
        while model := next((m for m in self.seat(models, seat) if m not in tried), None):
            tried.add(model)
            message = self.think([{"role": "user", "content": prompt}], [model], seat=seat)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            hard = self.seconds_left() + 30
            if hard < 3 or name in self.metered:
                return None
            _env.timeout = max(1, min(30, int(hard / 2)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if isinstance(seen, dict) and seen.get("error") == "budget_exceeded":
                    self.metered.add(name)
                    if name == self.roles.get("ask"):
                        self.msg_used = max(self.msg_used, self.msg_limit)
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
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

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        brief = self.ask_json(prompt, [LIGHT], "brief") or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], "brief") or brief
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
            fixed = self.ask_json(ask, [LIGHT, HEAVY], "unmix") or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts, str(entry.get("call") or "").strip()))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        pairs = [(_count(e.get("item")), _count(e.get("with"))) for e in _items(brief.get("fits")) if isinstance(e, dict)]
        self.fits = [(a, b) for a, b in pairs if a and b and a != b and max(a, b) <= len(self.items)]
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
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
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [LIGHT, HEAVY], "unmix") or {}
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
            refs = {need for _, need in self.refs}
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else
                          " (its value is on a listing not yet found and opened, so whether a listing meets it is unknown: it can neither pass a cart nor rule a listing out nor show that nothing fits until that listing is read)" if m in refs else "")
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
            parts.append("pick: the lowest order total among listings that fit (for items that must work together, the cheapest combination that works)")
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
        for a, b in self.fits:
            parts.append(f"item {a} ({self.items[a - 1].name}) must work with item {b} ({self.items[b - 1].name}): settle item {b} first, then choose item {a} by what the two pages say about fitting"
                         f" (a model, series or code one page lists for the other, or matching sizes and standards); when no item {a} fits, try another item {b} that meets its requirements")
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (the desk does it when you check out)", "end": "at an ending without a purchase (the desk does it then)"}[entry["when"]]
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
        text = f"[desk] step {self.step} of {self.max_steps} used, {left} left; about {max(0, int(self.seconds_left()))} s left."
        turns = max(0, int((self.spend_stop() - self.cost) / max(1e-6, self.cost / max(1, self.calls))))
        if turns <= 10:
            text += f" Model spend is near its limit: about {turns} more turns before the desk must finish without you, so have one fitting line per item in the cart by then."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif self.seconds_left() < CLOSING:
            text += " Time is nearly up: set the cart to the best lines you have read, one per item, and check out now."
        elif left <= self.finish_need() + 2:
            text += " Few steps left: set the cart to the chosen lines and check out now; a first checkout can bring a change (a sold-out line, a new price or budget) that needs a few steps to answer."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        unread = self.unread_items() if not self.asked else []
        if unread:
            text += "\n[before the questions] search and open listings of " + ", ".join(f"item {n} ({self.items[n - 1].name})" for n in unread) + ": the interviewer asks about the properties such listings carry."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += ("\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value."
                     " Find such a listing by its shop or brand, its kind and one or two of the values it is described with] " + " | ".join(lookups[:3]))
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
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
                    old = self.rows[ident]
                    flags = {k: v for k, v in node.items() if k == self.stock_key or not self.stock_key}
                    if any(old.get(k) is True and v is False for k, v in flags.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in flags.items()):
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

    @staticmethod
    def label_fits(label: str, item: Item) -> bool:
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        return bool(label) and (_words(label) in (_words(item.shelf or ""), _words(item.name)) or bool({w for w in _stem(label) if len(w) >= 4} & long))

    def unread_items(self) -> list[int]:
        if len(self.items) < 2:
            return []
        cat_key = self.category_key()
        rows = [self.rows.get(i) or {} for i in self.full]
        texts = [_stem(" ".join(str(v) for v in row.values() if isinstance(v, str))) for row in rows]

        def read(item: Item) -> bool:
            want = {w for w in _stem(item.call) if len(w) >= 3}
            return (bool(item.shelf and cat_key) and any(_words(str(row.get(cat_key) or "")) == _words(item.shelf) for row in rows)
                    or bool(want) and any(want <= text for text in texts))

        return [n for n, item in enumerate(self.items, 1) if not read(item)]

    def key_table(self) -> list[tuple[str, int, str, int, list[str]]]:
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
        matched = {label for label in groups if any(self.label_fits(label, item) for item in self.items)}
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
                table.append((label, group["n"], name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any(self.label_fits(label, item) for item in self.items)]
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

    def kind_of(self, ident: str) -> str:
        cat_key = self.category_key()
        return str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""

    def price_of(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
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
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], "read") or {}
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
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
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
        return ("\n[the desk read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[desk] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. A search for a requirement's value"
                " misses fitting listings that state it in other words or only in their attributes."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
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

    def basket(self) -> str:
        mark = self.cart_mark() + f"#{self.version}"
        cart = self.cart_ids()
        if not self.fits or mark in self.basket_marks or len(cart) != len(self.items):
            return ""
        self.basket_marks.add(mark)
        line_of = {n: next((i for i in cart if n in self.owners(self.kind_of(i))), None) for n in range(1, len(self.items) + 1)}
        pool = {n: [i for i in self.seen_order if i in self.full and self.buyable(i) and n in self.owners(self.kind_of(i))][-8:] for n in line_of}
        if None in line_of.values() or self.left() < 2 * len(self.items) + self.tail() + 1:
            return ""
        prompt = BASKET + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if self.fits:
            prompt += "\n\nMUST WORK TOGETHER:\n" + "\n".join(f"item {a} with item {b}" for a, b in self.fits)
        prompt += "\n\nOPENED RECORDS:\n" + "\n".join(f"item {n}: {i}: {self.record_of(i)}" for n, ids in pool.items() for i in ids)
        got = self.ask_json(prompt, [HEAVY, LIGHT], "basket") or {}
        fits = {n: [str(i) for i in _items((got.get("fits") or {}).get(str(n))) if str(i) in pool[n]] for n in pool} if isinstance(got.get("fits"), dict) else {}
        pairs = {(_count(e.get("item")), _count(e.get("with")), *map(str, _items(e.get("ids"))[:2])) for e in _items(got.get("pairs")) if isinstance(e, dict)}
        if not fits or not all(fits.values()):
            return ""
        cost = lambda combo: sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in combo.items())
        works = lambda combo: all((a, b, combo[a], combo[b]) in pairs or (b, a, combo[b], combo[a]) in pairs for a, b in self.fits)
        each = self.brief.get("budget_scope") == "each"
        within = lambda combo: not isinstance(self.budget, (int, float)) or (all(cost({n: i}) <= self.budget for n, i in combo.items()) if each else cost(combo) <= self.budget)
        combos = [dict(zip(fits, ids)) for ids in itertools.product(*fits.values())]
        combos = [c for c in combos if works(c) and within(c)]
        if not combos:
            return ""
        cheapest = self.brief.get("pick") == "cheapest"
        best = min(combos, key=lambda c: (cost(c), sum(c[n] != line_of[n] for n in c)) if cheapest else (sum(c[n] != line_of[n] for n in c), 0))
        self.trace.append({"kind": "basket", "fits": fits, "pairs": len(pairs), "best": best, "cart": line_of})
        current = dict(line_of)
        if current in combos and (not cheapest or cost(current) <= cost(best)):
            return ""
        return self.put_in({line_of[n]: best[n] for n in best if best[n] != line_of[n]},
                           "The cart's lines do not make the " + ("cheapest working combination" if cheapest else "lines work together") + " among the listings read, so the desk put that basket in")

    def stray_lines(self) -> str:
        cat_key, shelves = self.category_key(), {_stem(item.shelf) for item in self.items if item.shelf}
        label = lambda i: str((self.rows.get(i) or {}).get(cat_key) or "")
        mark = self.cart_mark() + f"#{self.version}"
        if not cat_key or len(shelves) < len(self.items) or not shelves <= {_stem(label(i)) for i in self.seen_order} or mark in self.stray_held:
            return ""
        lines = [i for i in self.cart_ids() if _stem(label(i)) not in shelves]
        if not lines:
            return ""
        self.stray_held.add(mark)
        return (", ".join(f"{i} is sold under {label(i)}" for i in lines) + f", while the requested items are sold under {'; '.join(item.shelf for item in self.items)}."
                " The listing's category decides the kind of product: unless that category is the requested item's own, put in a listing of the item's category; then check out again.")

    def cheaper_rows(self) -> list[str]:
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.search_under(cart)
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

    def search_under(self, cart: list[str]) -> None:
        cap = self.brief.get("price_cap") if isinstance(self.brief.get("price_cap"), dict) else {}
        tool, arg = cap.get("tool"), cap.get("arg")
        schema = self.schemas.get(tool) or {} if isinstance(tool, str) else {}
        props = schema.get("properties") or {}
        target = next((k for k in schema.get("required") or [] if k != arg and "string" in _kinds(props.get(k) or {})), None) or self.single_target(tool)
        if not target or arg not in props:
            return
        for line in cart:
            owners, price = self.owners(self.kind_of(line)), self.price_of(line)
            if len(owners) != 1 or price is None or self.left() <= self.tail() + len(cart) + 4:
                continue
            faults: list[str] = []
            args = _shape({target: self.items[owners[0] - 1].call, arg: price}, self.schemas[tool], "args", faults)
            reply = None if faults else self.act(tool, args)
            if reply is not None:
                self.observe(tool, reply.get("observation"))

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
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
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 1 and not any(i in _dump(seen) for i in picks):
                self.compare_bad = min(self.compare_bad or 999, len(picks))
            if len(self.full) == before and len(picks) > self.compare_room() and self.left() > 12:
                faults = []
                args = _shape({target: picks[: self.compare_room()]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        lanes: list[tuple[int, list[str]]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append((opened, pool[: max(0, WIDE_EACH - opened)]))
        lanes.sort(key=lambda lane: lane[0])
        queue = [lane[k] for k in range(WIDE_EACH) for _, lane in lanes if k < len(lane)]
        self.fill_facts(queue[: 2 * len(self.items) + 2], 3 * len(self.items) + 8)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        if self.unread_items() and self.step < self.max_steps // 3:
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
        unseen = [n for n in self.unread_items() if not self.kind_rows(n)]
        if unseen and self.left() > count + 3 * len(self.items) + 8:
            self.cover(unseen)
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, opened, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r} ({opened} opened):")
                last = label
            listing.append(f"  {index}. {name} (on {carried} of {opened}): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], "plan") or {}
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
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        head = ordered[: count - 1] if count >= 2 else ordered[:count]
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        self.take_heard(self.hear(batch), told)
        rest = [(n, lab) for n, lab in ordered[len(head):] if all(_words(lab) != _words(t.split(":")[0]) for t in self.items[n - 1].told + self.items[n - 1].free)]
        if count >= 2 and self.msgs_free() > 0 and rest:
            batch = self.send([(*rest[0], self.question(*rest[0]))])
            if batch:
                sent += batch
                self.take_heard(self.hear(batch), told)
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "The desk's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
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

    def question(self, number: int, label: str) -> str:
        item = self.items[number - 1]
        labels = collections.Counter(self.kind_of(i) for i in self.kind_rows(number) if i in self.full and self.kind_of(i)).most_common(2)
        name = item.shelf or (labels[0][0] if labels and (len(labels) == 1 or labels[0][1] > labels[1][1]) else item.call)
        return f"What {label} would you like for the {name.lower()}?"

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

    def hear(self, sent: list[tuple[int, str, str]]) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, [LIGHT, HEAVY], "read") or {}

    def gate(self, stage: str) -> tuple[bool, str]:
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
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], "gate")
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, the desk did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            unproven += [{"step": p["step"]} for p in steps[len(shown):]]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(desk: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call and not unproven:
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

    def loose_lines(self, observation: Any) -> list[str]:
        refusal = observation.get("refusal") if isinstance(observation, dict) else None
        lines: list[str] = [str(v) for v in refusal if str(v).strip()] if isinstance(refusal, list) else []
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
        got = self.ask_json(prompt, [LIGHT, SPARE], "read") or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
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

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by the desk's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 15) -> str:
        order = [i for i in reversed(self.seen_order) if i not in skip and not self.sold_out(i)]
        if self.brief.get("pick") == "cheapest":
            bar: dict[str, float] = {}
            for i in skip:
                if self.price_of(i) is not None:
                    bar[self.kind_of(i)] = max(bar.get(self.kind_of(i), 0.0), self.price_of(i) or 0.0)
            if bar:
                order = [i for i in order if self.kind_of(i) in bar and (self.price_of(i) or float("inf")) < bar[self.kind_of(i)]]
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        return "\n".join(_dump(_clip(self.full.get(i) or self.rows.get(i), 160))[:900] for i in order[:limit])

    def check_key(self, ids: list[str], planned: bool = False) -> str:
        return ("plan" if planned else "") + _dump(sorted((i, None if planned else self.cart_lines.get(i)) for i in ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids, lead)
        if key in self.checked:
            if lead or self.verdicts.get(key, True):
                return ""
            if self.restated.get(key, 0) >= 2:
                return self.insist_swap(key)
            self.restated[key] = self.restated.get(key, 0) + 1
            if self.check_mark.get(key) == (len(self.full), len(self.rows)):
                return self.check_said.get(key, "")
        if not ids:
            return ""
        self.checked.add(key)
        self.check_mark[key] = (len(self.full), len(self.rows))
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN:\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [HEAVY, LIGHT], "check")
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is True
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        self.lead_items = {k: v for k, v in ((k, [i for i in v if i not in ids]) for k, v in self.lead_items.items()) if v}
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        self.better[key] = better
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        self.check_said[key] = text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."
        return self.check_said[key]

    def trail(self, holding: bool = True) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        finder = self.roles.get("search")
        words_arg = self.single_target(finder)
        seen: list[str] = []
        lost: list[str] = []
        pages = rounds = 0
        start = self.step
        early = min(int(0.4 * self.max_steps), self.left() - self.msgs_free() - 2 * len(self.items) - 6)
        room = lambda: holding or self.step - start < early
        while pending and rounds < (4 if holding else 8) and self.trails < 14 and room():
            rounds += 1
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                if ident in self.opened or ident in self.full:
                    records.append(("OPENED " if ident in self.opened else "ATTRIBUTES ") + _dump(_clip(self.full.get(ident) or self.rows.get(ident), 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            if self.trail_searched:
                prompt += "\n\nSEARCHES TRIED:\n" + "\n".join(f"{q} -> {n} rows" for q, n in self.trail_searched.items())
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], "trail") or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            look: list[str] = []
            check: list[str] = []
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
                    missing = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    lost.append(f"{need} -> find {str(missing.get('about') or '') or 'the listing it points at'}")
                    words = str(missing.get("search") or "").strip()
                    if words and words.lower() not in self.trail_searched:
                        look.append(words)
                    check += [str(i) for i in _items(missing.get("maybe")) if str(i) in self.rows and str(i) not in self.full and str(i) not in check]
            pending = [p for p in pending if p not in self.ref_values]
            look, check = (look[:2] if finder and words_arg else []), check[:4]
            if not (reach and tool and target or look or check) or self.left() < len(reach) + len(look) + len(check) + 4 or self.finished:
                break
            keep = self.tail() + 4 if holding else max(self.tail() + 4, self.left() - max(0, early - (self.step - start)))
            seen += [f"read the attributes of {line.split(':')[0]}" for line in self.fill_facts(check, keep)]
            for words in look:
                if not room():
                    break
                faults: list[str] = []
                args = _shape({words_arg: words}, self.schemas.get(finder), "args", faults)
                reply = None if faults else self.act(finder, args)
                if reply is None or self.finished:
                    break
                self.trail_searched[words.lower()] = len(self.records_in(reply.get("observation")))
                self.observe(finder, reply.get("observation"))
                seen.append(f"search {words!r}: " + self.show(reply.get("observation"))[:600])
            for ident in reach[: max(0, min(3, (6 if holding else 8) - pages))]:
                if not room():
                    break
                pages += 1
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
            text += "[the desk followed the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = holding and bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def offer_tool(self) -> str | None:
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
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
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        self.swap_for.update({i: picks for i in lost})
        text = f"[desk] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], "swap")
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def decline_hold(self) -> str:
        walk = self.left() - self.tail() > len(self.items) + 3
        chain, hold = self.trail() if walk else ("", False)
        if hold:
            return chain
        review = self.audit()
        if walk and review:
            return review
        lines = [i for i in self.built if not self.sold_out(i)]
        if not walk and not self.late_held and len(lines) >= len(self.items):
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
            if len(missing) + 1 <= self.left():
                self.late_held = True
                self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                return ("Running short of steps does not show that the store has no fitting listing. " + redo
                        + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.")
        open_items = [n for n in range(1, len(self.items) + 1) if not self.lead_items.get(str(n))]
        thin = [n for n in open_items if not self.covered(n)]
        if not walk or not thin:
            return ""
        pushes = self.pushes.get(self.version, 0)
        if pushes < 2:
            self.pushes[self.version] = pushes + 1
            self.trace.append({"kind": "decline_pushed", "thin": thin})
            return ("No listing read so far fits " + "; ".join(f"item {n} ({self.items[n - 1].name}, {self.kind_opened(n)} of its listings opened)" for n in thin)
                    + ". That is too few to show the store has none: search its category in ways not tried yet (its plain one-word name, other names, the wanted value alone, further pages),"
                    " open the promising rows, cart what fits and check out. End without a purchase only after that.")
        self.pushes[self.version] = pushes + 1
        if pushes == 2 and self.cover(thin):
            return self.decline_hold()
        return ""

    def owners(self, label: str) -> list[int]:
        stems = _stem(label)
        exact = [n for n, item in enumerate(self.items, 1) if label and (_words(label) in (_words(item.shelf), _words(item.name))
                 or bool({w for w in _stem(item.call) if len(w) >= 3}) and {w for w in _stem(item.call) if len(w) >= 3} <= stems)]
        return exact or [n for n, item in enumerate(self.items, 1) if self.label_fits(label, item)]

    def kind_rows(self, number: int) -> list[str]:
        cat_key = self.category_key()
        return [i for i in self.seen_order if not cat_key or number in self.owners(str((self.rows.get(i) or {}).get(cat_key) or ""))]

    def kind_opened(self, number: int) -> int:
        return sum(1 for i in self.kind_rows(number) if i in self.full)

    def covered(self, number: int) -> bool:
        rows = [i for i in self.kind_rows(number) if self.buyable(i)]
        return self.kind_opened(number) >= COVERED or (bool(rows) and all(i in self.full for i in rows))

    def cover(self, numbers: list[int]) -> bool:
        before = len(self.full)
        tool = self.roles.get("search")
        target = self.single_target(tool)
        for number in numbers:
            pool = [i for i in self.kind_rows(number) if i not in self.full and self.buyable(i)]
            if not pool and tool and target and self.left() - self.tail() > len(self.items) + 4:
                faults: list[str] = []
                args = _shape({target: self.items[number - 1].call}, self.schemas.get(tool), "args", faults)
                reply = None if faults else self.act(tool, args)
                if reply is not None:
                    self.observe(tool, reply.get("observation"))
                    pool = [i for i in self.kind_rows(number) if i not in self.full and self.buyable(i)]
            self.fill_facts(pool[:COVERED], self.tail() + len(self.items) + 2)
        self.trace.append({"kind": "desk_cover", "items": numbers, "opened": len(self.full) - before})
        return len(self.full) > before

    def audit(self) -> str:
        mark = (self.version, len(self.full), len(self.rows))
        if mark in self.audited:
            return ""
        self.audited.add(mark)
        got = self.audit_call()
        named = self.audit_fits(got)
        if len(named) < len(self.items):
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                self.audited.add((self.version, len(self.full), len(self.rows)))
                got = self.audit_call() or got
                named = self.audit_fits(got)
        blocked = {str(k): str(v) for k, v in ((got or {}).get("blocked") or {}).items() if str(v).strip()} if isinstance((got or {}).get("blocked"), dict) else {}
        if got:
            self.lead_items = named
            self.leads = [i for ids in named.values() for i in ids]
            self.blocked = set(blocked) - set(named)
        if not named:
            return ""
        return ("A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
                + "".join(f"; item {k} has none yet ({v})" for k, v in blocked.items() if k not in named)
                + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails.")

    def audit_call(self) -> dict[str, Any] | None:
        cat_key = self.category_key()
        cart = self.cart_ids()
        groups: dict[int, list[str]] = {}
        for ident in sorted(reversed(self.seen_order), key=lambda i: (i not in cart, i not in self.full)):
            if self.sold_out(ident):
                continue
            label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
            for number in self.owners(label) or [0]:
                groups.setdefault(number, []).append(ident)
        blocks = []
        for number in sorted(groups, key=lambda n: n or len(self.items) + 1):
            ids = groups[number]
            opened = [i for i in ids if i in self.full][: 12 if number else (30 if set(groups) == {0} else 8)]
            rows = [i for i in ids if i not in self.full][: 6 if number else 4]
            blocks.append((f"ITEM {number}: {self.items[number - 1].name}" if number else "OTHER") + "\n"
                          + "\n".join(["OPENED " + _dump(_clip(self.full[i], 200))[:1000] for i in opened] + ["ROW " + _dump(_clip(self.rows[i], 120))[:300] for i in rows]))
        if not blocks:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nREADING RULES: " + READING
        prompt += "\n\nLISTING RECORDS SEEN, BY ITEM:\n" + "\n\n".join(blocks)
        got = self.ask_json(prompt, [HEAVY, LIGHT], "audit")
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
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

    def set_cart(self, plan: dict[str, int], drop: list[str]) -> list[str] | None:
        cart, calls = self.roles.get("cart"), self.cart_calls(plan, drop)
        if not cart or calls is None or len(calls) + 1 > self.left():
            return None
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            reply = None if faults else self.act(cart, args)
            seen = (reply or {}).get("observation")
            if reply is None or self.finished or reply.get("refusal") or (isinstance(seen, dict) and seen.get("error")):
                return None
            before = dict(self.cart_lines)
            self.note_cart(cart, args)
            self.learn_mode(cart, args, seen, before)
            self.observe(cart, seen)
            done.append(_dump(args))
        return done

    def insist_swap(self, key: str) -> str:
        swaps: dict[str, str] = {}
        for better in self.better.get(key, []):
            same = [i for i in self.cart_ids() if i not in swaps and self.kind_of(i) == self.kind_of(better)]
            cheaper = bool(same) and (self.brief.get("pick") != "cheapest" or (self.price_of(better) or float("inf")) < (self.price_of(same[0]) or 0.0))
            if len(same) == 1 and cheaper and len(self.owners(self.kind_of(better))) == 1 and better in self.full and self.buyable(better) and better not in self.cart_ids():
                swaps[same[0]] = better
        return self.put_in(swaps, "The review turned this cart down a third time, so the desk put in the listings it names")

    def put_in(self, swaps: dict[str, str], why: str, sold: bool = False) -> str:
        swaps = {old: new for old, new in swaps.items() if new}
        plan = {i: self.cart_lines.get(i, 1) for i in self.cart_ids()}
        for old, new in swaps.items():
            plan[new] = plan.pop(old)
        passed = swaps and not self.check_cart(list(plan)) and (sold or self.verdicts.get(self.check_key(list(plan), True)) is True)
        if not passed or self.set_cart(plan, [i for i in self.cart_ids() if i not in plan]) is None:
            return ""
        return why + " (" + "; ".join(f"{old} -> {new}" for old, new in swaps.items()) + "). Check out again."

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
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
        add, remove = self.cart_mode(name, True), self.cart_mode(name, False)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if remove:
                calls.append({target: ident, remove[0]: remove[1], **({count: have} if count else {})})
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
            calls.append({target: ident, **(dict([add]) if add else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not cart or not buy:
            return ""
        labelled = bool(self.category_key())
        now = [i for i in self.cart_ids() if not self.sold_out(i)]
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            lead = next((i for i in self.lead_items.get(str(number), []) if not self.sold_out(i)), None)
            held = None if str(number) in self.blocked else next(
                (i for i in now if not labelled or number in self.owners(self.kind_of(i))), None)
            pick = lead or held
            if not pick or pick in plan:
                return ""
            plan[pick] = item.qty
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or len(calls) + 1 > self.left():
            return ""
        self.completed = True
        self.fill_facts(list(plan), len(calls) + 3)
        self.trail(holding=False)
        done = None if self.check_cart(list(plan)) else self.set_cart(plan, drop)
        if done is None:
            return ""
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                if self.check_cart(list(plan)) or self.left() < 1 or self.finished or any(self.sold_out(i) for i in plan):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({"kind": "completed", "lines": plan, "drop": drop, "reply": _clip(reply, 200)})
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
        return ("Not run (the desk acted instead): ending without a purchase needs proof that some requested item has no fitting listing,"
                " and the review found a fitting listing for every item, so the desk set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

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

    def finish_need(self) -> int:
        lines = [i for i in self.cart_ids() if not self.sold_out(i)]
        covered = len({n for i in lines for n in self.owners(self.kind_of(i))[:1]}) if self.category_key() else min(len(lines), len(self.items))
        off = len(self.cart_ids()) - len(lines) + max(0, len(lines) - covered)
        return self.tail() + len(self.items) - covered + off + (0 if self.buys else FIRST_TRY)

    def offered(self, narrow: bool = True) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get("ask")
        keep: set[str] | None = None
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.finish_need() or self.seconds_left() < CLOSING:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        tools = [t for t in self.tools if t["function"]["name"] != ask]
        if keep is not None and ends and narrow:
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
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
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
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def cart_mode(self, name: str, adds: bool) -> tuple[str, Any] | None:
        props = self.schemas.get(name, {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not mode:
            return None
        options, sign = props[mode]["enum"], 1 if adds else -1
        read = self.brief.get("cart_add_args" if adds else "cart_remove_args")
        ranked = [v for v, s in self.mode_seen.items() if s == sign] + ([read[mode]] if isinstance(read, dict) and mode in read else [])
        for value in ranked + (options if adds else options[1:] + options[:1]):
            if value in options and self.mode_seen.get(value, sign) == sign:
                return mode, value
        return None

    def learn_mode(self, name: str, args: dict[str, Any], observation: Any, before: dict[str, int]) -> None:
        props = self.schemas.get(name, {}).get("properties") or {}
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        sent = _count(args.get(count)) if count else None
        if ident is None or sent is None:
            return
        shown: list[int] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                if ident in node.values() and _count(node.get(count)) is not None:
                    shown.append(_count(node[count]) or 0)
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        if len(shown) != 1:
            return
        had, now, mode = before.get(ident, 0), shown[0], self.cart_mode(name, True)
        if mode and mode[0] in args and sent and now != had and now in (had + sent, max(0, had - sent)) and (had == 0 or now != sent):
            self.mode_seen[args[mode[0]]] = 1 if now == had + sent else -1
        if now > 0:
            self.cart_lines[ident] = now
        else:
            self.cart_lines.pop(ident, None)

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        remove = self.cart_mode(name, False)
        removal = bool(remove) and args.get(remove[0]) == remove[1]
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
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        declared = spec["maxItems"] if isinstance(spec.get("maxItems"), int) else None
        least = max(2, spec["minItems"]) if isinstance(spec.get("minItems"), int) else 2
        room = max(4, self.compare_ok) if declared is None else declared
        if self.compare_bad:
            room = min(room, self.compare_bad - 1, max(self.compare_ok, self.compare_bad // 2))
        return room if room >= least else 0

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
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
                out.append((cid, "Not run (no step used): The desk's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Not run (no step used): the desk keeps the steps or time left for the cart and the checkout, so this call is closed now (the desk's rule, not the store's answer about any listing); use: " + ", ".join(sorted(allowed)) + "."))
                continue
            if name in self.metered:
                out.append((cid, f"Not run (no step used): the store's budget for {name} is used up; use the other tools."))
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
                if not room:
                    out.append((cid, "Not run (no step used): this store turns down side-by-side calls of every size its schema allows; open the listings one by one."))
                    continue
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
                    trimmed += f"\n[desk] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if self.is_buy(name, args) and not self.cart_ids() and (not self.cart_touched or (self.cart_view is not None and self.cart_seen_at >= self.cart_at)):
                out.append((cid, "Not run (no step used): the cart is empty. Put a fitting listing for each requested item in the cart first, then check out."))
                continue
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
                    spare = [i for i in self.cart_ids() if i in self.gone and self.kind_of(i) and len(self.owners(self.kind_of(i))) == 1 and any(
                        j != i and j not in self.gone and self.kind_of(j) == self.kind_of(i) for j in self.cart_ids())]
                    if spare and self.set_cart({i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in spare}, spare) is not None:
                        trimmed += f"\n[desk] Took the sold-out {', '.join(spare)} out of the cart: another listing of its kind is in it."
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone]
                    if held and self.held >= 2:
                        swap, held = self.put_in({i: next((n for n in self.swap_for.get(i, []) if self.kind_of(n) == self.kind_of(i) and n in self.full
                                                           and self.buyable(n) and n not in self.cart_ids()), "") for i in held},
                                                 "The cart still held a sold-out line, so the desk put in the listing named to replace it", True), []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    stray = self.stray_lines() or self.basket()
                    if stray:
                        out.append((cid, "Not run yet (no step used). " + stray + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("The desk read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.check_key(self.cart_ids()) not in self.verdicts:
                        out.append((cid, "Not run yet (no step used). The desk re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
            if self.is_refuse(name, args):
                held = self.decline_hold()
                if held:
                    self.trace.append({"kind": "decline_held", "said": held[:200]})
                    out.append((cid, "Not run yet (no step used). " + held))
                    continue
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
                    trimmed = "\n[desk] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    room = self.compare_room()
                    trimmed += f"\n[desk] The side-by-side view showed nothing for {len(args[many])} listings: " + (f"send at most {room} per call." if room else "open the listings one by one.")
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
                before = dict(self.cart_lines)
                self.note_cart(name, args)
                self.learn_mode(name, args, observation, before)
                self.cart_said = _dump(observation)
                self.recent = []
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
                    text += "\n[desk] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[the desk re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on the desk's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        self.finishing = True
        if not self.completed and len(self.cart_ids()) < len(self.items):
            self.audit()
            said = self.complete_cart()
            if self.finished or self.left() <= 0:
                return ""
            if said and again:
                return "the checkout was refused: " + said
        stray = [i for i in self.cart_ids() if self.kind_of(i) and not self.owners(self.kind_of(i))]
        kept = {n for i in self.cart_ids() if i not in stray for n in self.owners(self.kind_of(i))}
        if stray and kept >= set(range(1, len(self.items) + 1)):
            self.set_cart({i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in stray}, stray)
        key = self.check_key(self.cart_ids())
        if self.verdicts.get(key) is False and self.better.get(key) and self.left() >= 2 * min(len(self.better[key]), len(self.items)) + self.tail():
            self.insist_swap(key)
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = len(self.cart_ids() if fresh else self.cart_lines) >= len(self.items)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
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
        self.work(max(LOOP_FLOOR, int(self.max_steps * LOOP_PER_STEP)))
        if not self.finished:
            turned = self.finish(again=True)
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[desk] The assistant tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.seat([LIGHT]):
            if self.refs and not self.trailed_early and self.step >= 3:
                self.trailed_early = True
                text, _ = self.trail(holding=False)
                if text:
                    self.chat.append({"role": "user", "content": "[desk] " + text + "\n" + self.status()})
                continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, self.seat([LIGHT]), self.offered(narrow=False))
            if message is None:
                self.misses += 1
                if not self.seat([LIGHT]) or self.misses > 2:
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
            early = self.unread_items() and self.step < self.max_steps // 3
            committing = ({self.roles.get("buy"), self.roles.get("refuse")} | (set() if early else {self.roles.get("cart")})) - {None}
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
        episode = ShoppingAssistant(problem_data)
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
