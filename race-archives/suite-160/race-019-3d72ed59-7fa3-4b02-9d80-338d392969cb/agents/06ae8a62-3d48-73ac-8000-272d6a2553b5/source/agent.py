from __future__ import annotations

import json
import random
import re
import time
from types import SimpleNamespace
from typing import Any

from src.agent.proxy_client import ProxyClient

QUICK = "anthropic/claude-haiku-5.5"
DEEP = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"
WALKER_SEATS = [QUICK, BACKUP]
QUICK_FIRST = [QUICK, DEEP]
DEEP_FIRST = [DEEP, QUICK]
NOTE_SEATS = [QUICK, BACKUP]

WALK_TURNS = 42
LLM_CALLS = 60
SOFT_SPEND = 0.055
HARD_SPEND = 0.078
RETRY_PAUSES = (4, 10, 20, 35)
PAUSE_BUDGET = 45
VALUE_CHARS = 450
RESULT_CHARS = 15000
HISTORY_CHARS = 150000
GATE_TURNS = 3
SELF_PAGES = 6
CHEAP_READS = 3
STEP_DEFAULT = 30
TIME_LIMIT = 285.0
WRAP_SECONDS = 25.0
MESSAGE_DEFAULT = 3

_proxy = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_store_proxy = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

WALKER_GUIDE = """# Shopping walker - this step: buy for one shopper with the store's own tools
You act for a single shopper inside an online store, and the only way to act is a call to one of the store's tools. The episode scores 1 or 0, and only an exactly right finish scores: either a purchase in which every requested item appears at its requested quantity, every item meets every requirement in force at checkout and the total keeps to the budget, or, when the store really has nothing that meets everything, an ending without a purchase made the way the request asks.

# The store
- Each tool call costs one step and steps run out; thinking is free. This store has its own tool names, argument names and result shapes, so read every description and schema and use the argument names and allowed values exactly as written.
- Searches may be narrow: a category filter, a keyword mode that reads only the first words, pages of a few rows, a token or an offset for the next page. When the answer depends on seeing every match, compare the reported total with the rows seen and page on with the exact token or offset the store returned. Short keyword queries that lead with the value work best (the wanted value, then the product noun). A filtered search with no rows usually has a wrong filter value or parameter: try again with the label spelled the way result rows show it, or with another filter parameter.
- Result rows are only summaries, and stores split a record in different ways: one view may show the attributes (properties), another only the offer and the description, and a side-by-side view may carry either. The ledger names the attributes view as soon as it is known. Judge each requirement on a property from those attributes, so read them for every listing you might buy; a single call that shows several listings saves steps when it shows what you need. Do not open a listing a second time when nothing changed since you read it.
- Questions to the shopper belong to the interviewer: you never message the shopper yourself. Its questions and the answers appear in the conversation and go into the ledger.
- The request may also ask for steps along the way, such as an approval, a message, or something to check or read first; the ledger lists them. The code performs the ones due right before the order or the ending when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them. Do the others with the store's tools (product questions still go through the interviewer).

# What makes a listing fit
- The category label decides what kind of product a listing is: only the label matching the requested item as a whole counts, never a neighbouring label.
- A requirement on a property is met when the listing's attribute for that property states the value. How the key is spelled does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values are compared by their words: case, singular and plural and plain synonyms are all fine, and a value holding the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). Different words do not count: an abbreviation or code is not the word behind it, and "teal green" is not "sea green". Convert units before comparing numbers; limits include their edge value and a range must sit wholly inside a limit.
- An attribute whose key names the requirement's property, by that name or a plain synonym, decides the matter: if it says something else, the listing fails whatever other keys or the title say. A key for another property, even a related one (the colour of a handle is not the colour of the body), decides nothing. After an opened listing the code adds [evidence] lines listing the keys whose names match a requirement.
- An attribute listing several values meets a requirement when the wanted value is among them, whatever else it lists. Store lines about how a listing is sold (a minimum order, a pack count, another pack elsewhere) set no requirement; buy the requested quantity.
- With no attribute for that property, the wanted value stated under another attribute (such as a list of styles or features that includes it) or in the title or description counts. A number of a different kind does not (a weight is no capacity), and neither does a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or an audience in the label), the category meets that requirement.
- A requirement may point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle it first: find that listing (search its category, brand, shop or words) and open its page in full with the single-listing or side-by-side tool, because a search row is not enough; when that page sends you on to a further listing for the property, find and open that one in the same way. The last page holds the value, and the order only counts once every page of that chain is open. When the wanted value is anything except that other listing's value, the rule rules out only listings with exactly that value, and it is never on its own a reason to end without buying.
- When a listing you are about to buy sends you to another page about it, look at that page before buying.
- When one requested item has to fit another (an accessory for a device), pick the two together from the listings' own words: a list of what it fits, or a model or series code both pages carry. When no accessory meeting its own requirements fits the chosen device, move to another device that meets its requirements and that such an accessory fits; never end without buying while that switch is untried.
- Keep to exclusions and closed lists. Wishes ("ideally", "would be nice") only break ties.
- Never buy something the store shows as sold out. The total, price times quantity summed over all lines, may not go over the budget; landing exactly on it is fine.

# Choosing
- When the request asks for the cheapest (lowest total, smallest price), the order has to hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set by paging through it (rows often come in no price order, and the code reads the next pages of a short result itself), check the candidates from the lowest price up and take the first that fits fully. When a search accepts a maximum price, use it: list the category below the price of your best pick so far, so only cheaper candidates are left to check.
- When the request asks for the most or least of a measure, compare every fitting listing on it and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right, and hunting for a better one only wastes steps.
- Cart each item as soon as you have opened a listing meeting all of its requirements (under a cheapest, most or least pick, once the fitting set has been compared), then go on to the next item. Stores often sell out a line soon after it is carted, and carting early leaves steps to replace it.
- Several different items need one cart line each, at the quantity requested for it.

# Changes while you work
- The shopper or the store may change things: a new budget, a requirement added or dropped, a new price, a listing sold out. Such changes appear inside tool results, often when you try to check out (a refusal); a note counts no matter which tool's result shows it, and the ledger shows the requirements in force. The episode continues after a refusal: apply every change, fix the cart and check out again.
- Read a chosen listing's record again shortly before checking out, since price and stock can change without notice.

# Finishing
- Put exactly the chosen lines in the cart and remove anything else, look at the cart once, then check out with a call on its own.
- End without a purchase only once a real search has found no listing that meets everything. Being short of steps proves no such thing: in that case check out the best cart you have. Follow the request's wording for that ending, and keep any remark short and free of listing names.
- Every reply has to call at least one tool. Keep your text brief: what you know, then the next move."""

READER_GUIDE = """# Request reader - this step: turn the shopper's request and the store's tools into one plan
- Answer with ONE JSON object, compact (no indentation) and alone, carrying the keys below.

# Keys
- "items": one entry for each distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when some tool lists labels (copied exactly), otherwise "", "quantity": units of that single listing (an integer, 1 when unstated; a pair or a set sold as one listing counts as 1), "musts": the firm requirements on that product, each written "property: constraint" with the exact value, limit, exclusion or closed list}.
  - Name each property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch") and never trade it for a neighbouring property.
  - A must whose value comes from another listing is ONE must that holds the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): the words describing that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that must and never become musts of this product.
  - Price, budget, stock and quantity never go into musts.
  - Words that only paint the product with a comparison or a mood ("soft as a cloud", "cheerful like a summer morning") belong to "wishes"; a must needs a value the shopper names for one of the product's properties.
  - The story around the request (who will use it, where it will stand, what went wrong before, the occasion behind it) is context and no must, unless it states a property's value or limit outright; words that are part of what the product is called (a season, an audience or a use named together with the product) belong to "name" and never become musts.
  - A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; split it into several items only when each product gets its own quantity or its own requirements.
- "budget": the most the shopper will pay in total, as a number in the request's own unit, or null.
- "budget_unit": that unit in words (for example "cents"), or null.
- "budget_scope": "total" when the limit covers the whole order, "each" when it applies per item, null without a limit.
- "in_stock_only": true unless the request clearly accepts items that cannot be bought now.
- "pick": "cheapest" when the request asks for the lowest price or the smallest total, "most" or "least" when it asks for the extreme of some measure, otherwise "any". Saying money is tight is no pick rule.
- "pick_measure": the measure maximised or minimised for "most"/"least", otherwise "".
- "messages": how many messages the shopper lets the agent send (an integer), or null when unstated.
- "steps": how many steps, moves or tool calls the request allows (an integer), or null.
- "lookups": the musts (copied exactly) whose value has to be read first from another listing the request points at, otherwise [].
- "rules": every instruction the request gives about how to work or how to end, each in 12 words at most.
- "wishes": what the shopper would merely like.
- "ask_first": what the request explicitly says to check with the shopper before choosing (empty when nothing).
- "process": the steps the request demands of the agent besides finding and buying the right items (the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what has to be done, in 15 words at most, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are no steps.
- "quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing about it.
- "roles": the exact tool doing each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; with two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer such as price and stock, when that is another tool than "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
- "buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), otherwise {}.
- "refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, otherwise {}.
"""

PLANNER_GUIDE = """# Question planner - this step: choose which properties the interviewer asks the shopper about
- Shoppers keep a few firm preferences they never wrote down, each on one property of one requested product. A question uncovers a preference only when it names exactly that property, and the shopper answers only about the first property a question names, so each question spends one property.
- You receive the request, the items with what the request already settles for each, and a numbered list of attribute keys found on opened listings of each item's category, with how many of those listings carry each key and some example values. An unwritten preference may sit on a property no listed key names.

# Order of choice
- First, whatever the request itself asks you to check with the shopper (shown under ASK FIRST when present), each named by the property it concerns.
- Next, hinted properties. The story in the request often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, how the product looks or the setting it belongs to, the person meant to use or wear it, or what goes inside it. For every such sentence name the one property it points at and the item it concerns, in the order the story gives them.
  - Use the plain name of a hinted property: a hint about who will wear it points at gender, a hint about its looks at colour or finish (whichever the request leaves open).
- Then the listed keys in rank order: prefer the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys most of the item's listings carry; avoid keys found on only one or two listings.
- Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first and then alternate between the items.

# Answer
- "label" is the property's plain name: for a listed key, every word of the key in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish, never cut to "colour" or "finish"); only the property's name, without example values, brackets or a second property.
- Choose up to COUNT, best first. Reply with a single JSON object and no other text, holding "ask": a list of {"item": item number, "label": ..., "hint": up to 8 words of the story it comes from, or ""}"""

LISTENER_GUIDE = """# Answer listener - this step: read the shopper's answers to the interviewer's questions
- Reply with a single JSON object and no other text, holding these keys:
  - "needs": a list of {"item": item number, "need": "property: constraint", "points": true if the value has to be read from another listing, otherwise false}
  - "drop": a list of {"item": item number, "property": a requirement the shopper now gives up}
  - "free": a list of {"item": item number, "property": ...}
  - "crowded": the numbers of the questions whose answer complains that the message asked about too many things or asks for one question at a time (not answers that merely say they did not follow)
  - "budget": the new total limit as a number counted in the same unit as the LIMIT IN FORCE line below (a figure in cents stays in cents), or null
- "needs" holds every firm preference an answer states, with its exact value or limit and unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which can differ from the one asked; several answers about one property make a single need. Copy the value word for word as the answer gives it, keeping its nouns and units ("at least 3 shelves", never "at least 3").
- An answer may give the value by pointing at another listing (a value like, unlike or close to the one on some shop's product, picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties that describe the other product are no needs of this item.
- "free" holds the property of each question the shopper said they do not mind about.
- Homely or roundabout wording that names a value still states that value."""

NOTE_GUIDE = """# Note reader - this step: a note the store showed while the agent worked (it may come from the shopper or the store)
- Reply with a single JSON object and no other text, holding these keys:
  - "kind": "change" if the note changes the task (the budget, requirements, a price, availability, or a refusal of the order), "error" if it only says a call was malformed or not accepted, "other" for anything else
  - "budget": the new total limit as a number in the request's unit, or null
  - "add": a list of {"item": item number, "need": "property: constraint"}
  - "drop": a list of {"item": item number, "need": the requirement no longer wanted}
  - "gone": the listing ids it calls unavailable
  - "sold_out": true if it says a chosen listing is gone, sold out or no longer available, even without naming it
  - "price_moved": true if it says a price changed
  - "summary": one short sentence
- A requirement that is replaced counts as a drop of the old one together with an add of the new one. Report every change the note makes."""

FIT_RULES = """- A listing's attribute decides a property when its key names that very property: any key spelling, one element of a list is enough, synonyms and unit conversions count, and a value holding the required value's words counts ("deep teal" for teal); limits include their edge.
- When an attribute holds several values, the wanted value being one of them is enough; the values beside it break nothing, even when the shopper says that value is all they need on that front, because that speaks of what the shopper wants, not of how long the listing's list may be.
- What the store writes about how it sells a listing (a minimum order in the title, a pack count, a multipack mentioned elsewhere) asks nothing of the shopper: the requested quantity stands, and a store rule that really applies shows up as a refusal at checkout.
- A key for another property, even a related one (the colour of a handle is not the colour of the body), decides nothing.
- With no attribute naming the property, the wanted value stated under another attribute or in the title or description counts; a bare figure that does not say what it measures counts for nothing.
- A requirement that the category label itself states is met by that category, and the category label must match the item.
- A rule of being unlike another listing's value counts as met unless the records show that same value.
- Listings that are out of stock or break the budget do not fit."""

CART_CHECK_GUIDE = """# Cart checker - this step: the last look before the agent checks out; say from the records shown whether this cart is exactly right
- A requirement is met when the listing's attribute for that property states the value: any key spelling, one element of a list is enough, case, singular and plural and plain synonyms count, a value holding the required value's words counts ("deep teal" for teal), units are converted, limits include the edge and a range must sit wholly inside.
- An attribute whose key names the same property as the requirement decides, and the listing fails if it says something else; a key for another property, even a related one (the colour of a handle is not the colour of the body), decides nothing.
- With no attribute for that property, the wanted value stated under another attribute or in the title or description counts, but a number of another kind does not (a weight is no capacity), nor a bare figure whose text does not say what it measures.
- When an attribute holds several values, the wanted value being one of them is enough; the values beside it break nothing, even when the shopper says that value is all they need on that front, because that speaks of what the shopper wants, not of how long the listing's list may be.
- What the store writes about how it sells a listing (a minimum order in the title, a pack count, a multipack mentioned elsewhere) asks nothing of the shopper: the requested quantity stands, and a store rule that really applies shows up as a refusal at checkout.
- A requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value.
- Each requested item has to be its own cart line at the requested quantity, nothing else may be in the cart, nothing may be shown out of stock, and price times quantity summed over all lines must keep to the budget. A listing's newest record wins.
- Under a cheapest, most or least pick, also weigh the other listings shown: when one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
- Reply with a single JSON object and no other text, holding these keys:
  - "ok": true or false, where false is only for a problem the records clearly show or a requirement no record addresses
  - "problems": a list of short, concrete problems
  - "better": listing ids that should take a cart line's place, best first"""

DECLINE_CHECK_GUIDE = """# Decline checker - this step: the agent wants to end without buying because it believes nothing fits; test that against the listing records shown
""" + FIT_RULES + """
- Records marked ROW were met only as a search result row: their page with the attributes was never read, so they can neither pass nor fail on a property the row does not show.
- Reply with a single JSON object and no other text, holding these keys:
  - "fits": an object mapping each item number to up to 4 listing ids that seem to meet every requirement, lowest price first
  - "open": up to 6 ids of ROW listings of a requested kind and within budget whose page might show that they fit, most promising first
  - "reason": one sentence"""

POINTER_SPLIT_GUIDE = """# Pointer splitter - this step: separate words that only describe a pointed-at listing from the product's own requirements
- A shopper's request was read into firm requirements on a product. Some of them take their value from another listing the request points at, and the words picking out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought.
- Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing.
- Reply with a single JSON object and no other text: {"describe_other": [...]}, listing those requirements copied exactly, or an empty list when each one really applies to the product."""

CHAIN_GUIDE = """# Chain follower - this step: the agent is about to check out; trace each requirement whose value lives on another listing
- Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order only counts once the agent has opened every listing along each such chain in full.
- Follow every requirement's chain from the requirement's own description, link after link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: met only as a search result row, page not read).
- Reply with a single JSON object and no other text, holding "chains": a list with one entry per requirement, each having
  - "need": the requirement copied exactly
  - "links": the chain in order, each {"id": the listing id if a record below is that link, otherwise "", "about": how the requirement or the page before describes that listing}
  - "done": true if the last link is an OPENED page stating the property's value itself, otherwise false
  - "value": the property's value as that last page states it, or ""
- List each chain's links in order and stop at the first link whose page is not OPENED or not found, because its words are still unknown."""

PROCESS_GUIDE = """# Process gate - this step: the agent is about to STAGE, and the request asks for the steps below along the way
- Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it stands. A step is done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart's last change and its reply grants it; questions about a product's properties do no step.
- When a step is still missing, give the single store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool as one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
- Reply with a single JSON object and no other text, holding these keys:
  - "steps": a list of {"step": the step, "done": true or false, "lines": the RECORD line numbers showing it done}
  - "call": {"tool": ..., "args": {...}}, or null
  - "hold": what a reply asks to change before going on, or ""
  - "needs": a list of {"item": item number, "need": "property: constraint"}, one per firm requirement such a reply states"""

SWAP_GUIDE = """# Replacement finder - this step: a listing in the agent's cart has just sold out; find other records that can take its cart line
""" + FIT_RULES + """
- Reply with a single JSON object and no other text, holding these keys:
  - "fits": up to 6 entries {"id": listing id, "price": its price as a number, as the record shows it}, one for each listing that seems to meet every requirement of the sold-out line's item
  - "reason": one sentence"""

SHAPE_GUIDE = """# Result reader - this step: name the parts of a store tool's result by the exact keys the result and the tool's parameters use
- Reply with a single JSON object and no other text, holding these keys (null where the result has no such part):
  - "rows": the key of the list of listings
  - "count": the key with the number of listings matched in all (not the number shown)
  - "next": the key whose value asks for the following page
  - "next_arg": the tool parameter that takes that value
  - "price": the key inside a row that holds the listing's price
  - "stock": the key inside a row that says whether the listing can be bought now"""

MODE_GUIDE = """# Finish modes - this step: one store tool both checks out and ends without a purchase, depending on its arguments
- Reply with a single JSON object and no other text, holding "buy_args" (the fixed arguments that make it check out) and "refuse_args" (the fixed arguments that make it end without buying), giving only the arguments that choose the mode."""

PLAIN_ENDING = "I could not find anything here that meets every one of your requirements, so I am ending without a purchase."


def dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def clip_values(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {key: clip_values(inner, limit, depth + 1) for key, inner in value.items()}
    if isinstance(value, list):
        return [clip_values(inner, limit, depth + 1) for inner in value]
    return value


def load_object(text: str) -> dict[str, Any] | None:
    try:
        found = json.loads(text)
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def parse_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        found = json.loads(body)
        return found if isinstance(found, dict) else None
    except ValueError:
        pass
    start, end = body.find("{"), body.rfind("}")
    return load_object(body[start:end + 1]) if 0 <= start < end else None


def key_words(key: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", spaced).strip().lower()


def word_set(text: str) -> frozenset[str]:
    return frozenset(word[:-1] if len(word) > 3 and word.endswith("s") else word for word in key_words(text).split())


def names_same(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def type_list(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    return [kind for kind in declared or [] if isinstance(kind, str)]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def matches_type(value: Any, kind: str) -> bool:
    checks = {"string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": is_number, "boolean": lambda v: isinstance(v, bool), "array": lambda v: isinstance(v, list),
              "object": lambda v: isinstance(v, dict), "null": lambda v: v is None}
    test = checks.get(kind)
    return test(value) if test else True


def coerce_type(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            figure = float(value)
            return int(figure) if kind == "integer" or figure.is_integer() else figure
        if kind == "string" and is_number(value):
            return str(value)
        if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if kind == "array" and not isinstance(value, list) and value is not None:
            return [value]
    return value


def check_enum(value: Any, options: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(options, list) or value in options:
        return value
    if isinstance(value, str):
        same = [option for option in options if isinstance(option, str) and option.lower() == value.strip().lower()]
        if same:
            value = same[0]
    if value not in options:
        faults.append(f"{where}: allowed values are {options}")
    return value


def check_text(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    longest, shortest = schema.get("maxLength"), schema.get("minLength")
    if isinstance(longest, int) and len(value) > longest:
        value = value[:longest]
    if isinstance(shortest, int) and len(value) < shortest:
        faults.append(f"{where} is too short")
    return value


def check_bounds(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> None:
    if not is_number(value):
        return
    if is_number(schema.get("minimum")) and value < schema["minimum"]:
        faults.append(f"{where} is below {schema['minimum']}")
    if is_number(schema.get("maximum")) and value > schema["maximum"]:
        faults.append(f"{where} is above {schema['maximum']}")


def check_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
        faults.append(f"{where} allows at most {schema['maxItems']} entries")
    if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
        faults.append(f"{where} needs at least {schema['minItems']} entries")
    if isinstance(schema.get("items"), dict):
        return [conform(inner, schema["items"], f"{where}[{index}]", faults) for index, inner in enumerate(value)]
    return value


def check_object(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    fields = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if schema.get("additionalProperties") is False:
        faults.extend(f"{where}.{name} is not an allowed argument" for name in value if name not in fields)
    faults.extend(f"{where}.{name} is required" for name in schema.get("required") or [] if name not in value)
    return {key: conform(inner, fields.get(key), f"{where}.{key}", faults) if key in fields else inner for key, inner in value.items()}


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = type_list(schema)
    if kinds and not any(matches_type(value, kind) for kind in kinds):
        value = coerce_type(value, kinds)
        if not any(matches_type(value, kind) for kind in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    value = check_enum(value, schema.get("enum"), where, faults)
    value = check_text(value, schema, where, faults)
    check_bounds(value, schema, where, faults)
    if isinstance(value, list):
        value = check_list(value, schema, where, faults)
    if isinstance(value, dict):
        value = check_object(value, schema, where, faults)
    return value


def shaped(args: dict[str, Any], schema: Any) -> dict[str, Any] | None:
    faults: list[str] = []
    fixed = conform(args, schema, "args", faults)
    return None if faults else fixed


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def whole_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def item_index(value: Any, size: int) -> int:
    return min(max(1, whole_number(value) or 1), size)


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def id_key_of(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(value, "id" in str(key).lower()) for value in values):
            continue
        values = [value.strip() for value in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ("id" in str(key).lower(), sum(len(value) for value in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def listed_in(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        carries = any(isinstance(entry, dict) and ident in [str(x).strip() for x in entry.values() if isinstance(x, str)] for entry in node)
        return carries or any(listed_in(entry, ident, depth + 1) for entry in node)
    if isinstance(node, dict):
        return any(listed_in(inner, ident, depth + 1) for inner in node.values())
    return False


def attribute_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {key: inner for key, inner in value.items() if isinstance(inner, (str, int, float)) and not isinstance(inner, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def free_text_fields(schema: dict[str, Any], with_lists: bool = False) -> list[str]:
    fields = schema.get("properties") or {}
    wanted = ("string", "array") if with_lists else ("string",)
    return [key for key, spec in fields.items() if isinstance(spec, dict) and any(kind in type_list(spec) for kind in wanted) and "enum" not in spec]


def mode_field(schema: dict[str, Any]) -> str | None:
    fields = schema.get("properties") or {}
    return next((key for key, spec in fields.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2), None)


def count_field(schema: dict[str, Any]) -> str | None:
    fields = schema.get("properties") or {}
    return next((key for key, spec in fields.items() if isinstance(spec, dict) and "integer" in type_list(spec)), None)


def kind_words(*names: str) -> set[str]:
    return {word for name in names for word in word_set(name or "") if len(word) >= 4}


def label_matches(label: str, wanted: list[tuple[str, str, set[str]]]) -> bool:
    plain = key_words(label)
    long_words = kind_words(label)
    return any(plain in (shelf, name) or long_words & words for shelf, name, words in wanted)


def spoken_text(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    pieces: list[str] = []

    def gather(node: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, str) and len(node.split()) >= 2:
            pieces.append(node)
        elif isinstance(node, dict):
            for inner in node.values():
                gather(inner, depth + 1)
        elif isinstance(node, list):
            for inner in node:
                gather(inner, depth + 1)

    gather(observation, 0)
    return " ".join(pieces)[:800] if pieces else dump_json(observation)[:400]


def render(observation: Any) -> str:
    text = dump_json(clip_values(observation, VALUE_CHARS))
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + "...(cut)"


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

    def live(self) -> list[str]:
        return [need for need in self.musts + self.told if need not in self.dropped]


class Store:
    def __init__(self, binding: dict[str, Any], finished: bool, trace: list[dict[str, Any]]):
        self.binding = dict(binding)
        self.step = 0
        self.finished = finished
        self.broke = False
        self.trace = trace

    def send(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _store_proxy.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "store_error", "status": status, "body": body[:200]})
            if status == 409:
                retry = self.after_conflict(body.lower(), body)
                if retry is None:
                    return None
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                self.broke = self.broke or status == 402
                return None
        return None

    def after_conflict(self, low: str, body: str) -> bool | None:
        if "finish" in low or "quarantin" in low or "finaliz" in low:
            self.finished = True
            return None
        expected = re.search(r"expected (\d+)", body)
        if expected:
            self.step = int(expected.group(1)) - 1
            return True
        if "reuse" in low or "different action" in low:
            self.step += 1
            return True
        return None


class Clock:
    def __init__(self, seconds: float):
        self.start = time.monotonic()
        self.seconds = seconds

    def left(self) -> float:
        return self.seconds - (time.monotonic() - self.start)


class Seats:
    def __init__(self, trace: list[dict[str, Any]]):
        self.trace = trace
        self.cost = 0.0
        self.calls = 0
        self.waited = 0
        self.no_credit = False
        self.halted = lambda: False

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= LLM_CALLS:
                return None
            self.calls += 1
            message, status = self.retry_model(model, messages, max_tokens, tools)
            if message is not None:
                return message
            if status == 402:
                self.no_credit = True
                return None
        return None

    def retry_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        tries = 0
        status: Any = 0
        while True:
            message, status = self.call_model(model, messages, max_tokens, tools, plain=tries > 0 and status is None)
            if message is not None:
                return message, status
            passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
            limit = 1 if status in (None, 408) else len(RETRY_PAUSES)
            if not passing or tries >= limit or self.waited + RETRY_PAUSES[tries] > PAUSE_BUDGET:
                return None, status
            pause = RETRY_PAUSES[tries] * (0.8 + 0.4 * random.random())
            self.waited += RETRY_PAUSES[tries]
            tries += 1
            time.sleep(pause)

    @staticmethod
    def cache_marks(model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not model.startswith("anthropic/") or len(messages) <= 1:
            return messages
        marked = [dict(message) for message in messages]
        for index in (0, len(marked) - 1):
            content = marked[index].get("content")
            if isinstance(content, str) and content:
                marked[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return marked

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, plain: bool = False) -> tuple[dict[str, Any] | None, Any]:
        body: dict[str, Any] = {"model": model, "messages": self.cache_marks(model, messages), "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if plain:
            data = _proxy.post("/inference/chat/completions", body)
            result = SimpleNamespace(data=data, error=None if data is not None else {"kind": "network", "status": None})
        else:
            result = _proxy.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if is_number(usage.get("cost")):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "llm_error", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = whole_number(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_error", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            found = parse_object((message or {}).get("content"))
            if found is not None:
                return found
            if self.halted() or self.no_credit or self.calls >= LLM_CALLS:
                return None
        return None


class Episode:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        self.view = environment["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [tool for tool in self.view.get("tools") or [] if isinstance(tool, dict) and isinstance(tool.get("function"), dict) and isinstance(tool["function"].get("name"), str)]
        self.schemas = {tool["function"]["name"]: tool["function"].get("parameters") if isinstance(tool["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for tool in self.tools}
        allowed = whole_number(self.view.get("max_steps"))
        if allowed is None and isinstance(self.view.get("max_steps"), float):
            allowed = int(self.view["max_steps"])
        self.step_cap = allowed if allowed and allowed > 0 else STEP_DEFAULT
        self.trace: list[dict[str, Any]] = []
        self.store = Store(environment["binding"], bool(self.view.get("finished")), self.trace)
        self.seats = Seats(self.trace)
        self.seats.halted = lambda: self.store.broke
        self.missed_turns = 0
        self.walk_turns = 0
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.message_cap = 0
        self.messages_sent = 0
        self.process_steps: list[dict[str, Any]] = []
        self.kept_messages = 0
        self.gate_messages = 0
        self.gate_passed: set[tuple[str, str]] = set()
        self.record_lines: list[str] = []
        self.record_steps: list[int] = []
        self.interviewed = False
        self.rows: dict[str, dict[str, Any]] = {}
        self.detailed: dict[str, dict[str, Any]] = {}
        self.seen_ids: list[str] = []
        self.notes_read: set[str] = set()
        self.change_notes: list[str] = []
        self.version = 0
        self.cart_used = False
        self.cart_changed_at = -1
        self.cart_viewed_at = -1
        self.cart_snapshot: Any = None
        self.cart_units: dict[str, int] = {}
        self.reviewed: set[str] = set()
        self.decline_reviewed = False
        self.offer_read_at: dict[str, int] = {}
        self.sold_at: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_phrases: set[str] = set()
        self.note_says_sold = False
        self.sold_holds = 0
        self.id_params: set[tuple[str, str]] = set()
        self.cart_reply = ""
        self.refused_step = -1
        self.leads: list[str] = []
        self.decline_holds = 0
        self.empty_checkout_held = False
        self.search_pushed = False
        self.fullest_cart: list[str] = []
        self.taken_off: set[str] = set()
        self.late_hold_done = False
        self.leads_by_item: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed_id: str | None = None
        self.self_finish_done = False
        self.id_key: str | None = None
        self.shows_attributes: dict[str, bool] = {}
        self.pending_notes: list[str] = []
        self.checkouts = 0
        self.recent_calls: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened_pages: set[str] = set()
        self.pointers: list[tuple[int, str]] = []
        self.pointer_values: dict[str, str] = {}
        self.chain_rounds = 0
        self.chain_holds = 0
        self.batch_ok = 0
        self.batch_bad = 0
        self.view_fields: dict[str, set[str]] = {}
        self.listing_fields: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.self_pages = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_stamps: set[str] = set()
        self.broad_held = False
        self.self_paged: list[Any] = []
        self.clock = Clock(TIME_LIMIT)
        self.swap_leads: dict[str, list[str]] = {}

    @property
    def step(self) -> int:
        return self.store.step

    @property
    def finished(self) -> bool:
        return self.store.finished

    def steps_left(self) -> int:
        return self.step_cap - self.step

    def spare_messages(self) -> int:
        return self.message_cap - self.messages_sent - max(0, self.kept_messages - self.gate_messages)

    def cart_stamp(self) -> str:
        return dump_json(sorted(self.cart_units.items()))

    def version_stamp(self) -> str:
        return self.cart_stamp() + f"#{self.version}"

    def role(self, job: str) -> str | None:
        return self.roles.get(job)

    def ending_tools(self) -> set[str]:
        return {self.role("buy"), self.role("refuse")} - {None}

    def page_views(self) -> set[str]:
        page = self.role("offer") or self.role("open")
        return {tool for tool in (page, self.role("compare")) if tool}

    def attribute_view(self) -> str | None:
        known = next((tool for tool, shows in self.shows_attributes.items() if shows and self.one_listing_arg(tool)), None)
        if known:
            return known
        tool = self.role("open")
        return tool if self.shows_attributes.get(tool or "") is not False and self.one_listing_arg(tool) else None

    def batch_is_bare(self) -> bool:
        seen = self.shows_attributes.get(self.role("compare") or "")
        separate = self.shows_attributes.get(self.attribute_view() or "") is True or bool(self.role("offer"))
        return seen is False or (seen is None and separate)

    def steps_pending(self, stage: str) -> bool:
        return any(entry["when"] == stage for entry in self.process_steps) and (stage, self.version_stamp()) not in self.gate_passed

    def finish_reserve(self) -> int:
        talk = any(entry["when"] == "order" and entry["messages"] > 0 for entry in self.process_steps)
        return 2 if talk and self.messages_sent < self.message_cap and self.steps_pending("order") else 1

    def ask(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        return self.seats.ask_json(prompt, models, max_tokens)

    def call_store(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        data = self.store.send(name, args)
        if data is None:
            return None
        seen = data.get("observation")
        if name in self.page_views() and not (isinstance(seen, dict) and seen.get("error")):
            for value in args.values():
                self.opened_pages.update(entry.strip() for entry in (value if isinstance(value, list) else [value]) if isinstance(entry, str))
        return data

    def tool_lines(self, skip: set[str] | None = None) -> list[str]:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            if skip and fn["name"] in skip:
                continue
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {dump_json(fn.get('parameters') or {})}")
        return lines

    def read_brief(self) -> None:
        prompt = READER_GUIDE + "\nTOOLS:\n" + "\n".join(self.tool_lines()) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + dump_json(clip_values(self.view.get("opening"), 1500))
        brief = self.ask(prompt, [QUICK], 4000) or {}
        lacking = not as_list(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")
        if not self.seats.no_credit and lacking:
            brief = self.ask(prompt, DEEP_FIRST, 2500) or brief
        self.brief = brief
        self.take_roles(brief)
        self.take_items(brief)
        self.take_pointers(brief)
        self.split_pointer_words()
        self.take_limits(brief)
        self.take_process(brief)

    def take_roles(self, brief: dict[str, Any]) -> None:
        named = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {job: tool for job, tool in named.items() if isinstance(tool, str) and tool in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.role("buy"), self.role("refuse")
        same_mode = not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]
        if not buy or buy != refuse or not same_mode:
            return
        prompt = MODE_GUIDE + f"\nTOOL: {buy}\nSCHEMA: " + dump_json(self.schemas.get(buy))
        modes = self.ask(prompt, QUICK_FIRST, 600) or {}
        if isinstance(modes.get("buy_args"), dict) and isinstance(modes.get("refuse_args"), dict) and modes["buy_args"] != modes["refuse_args"]:
            brief["buy_args"], brief["refuse_args"] = modes["buy_args"], modes["refuse_args"]

    def take_items(self, brief: dict[str, Any]) -> None:
        for entry in as_list(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            wanted = whole_number(entry.get("quantity"))
            musts = [str(must) for must in as_list(entry.get("musts")) if must]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), wanted if wanted and wanted > 0 else 1, musts))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]

    def take_pointers(self, brief: dict[str, Any]) -> None:
        for lookup in as_list(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                prop = key_words(text.split(":")[0])
                text = next((must for item in self.items for must in item.musts if key_words(must.split(":")[0]) == prop), text)
            number = next((index for index, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.pointers:
                self.pointers.append((number, text))

    def take_limits(self, brief: dict[str, Any]) -> None:
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if is_number(budget) and budget > 0:
            self.budget = budget
        cap = whole_number(brief.get("messages"))
        if cap is not None and cap >= 0:
            self.message_cap = cap
        elif self.role("ask"):
            self.message_cap = MESSAGE_DEFAULT
        steps = whole_number(brief.get("steps"))
        if steps is not None and 10 <= steps < self.step_cap:
            self.step_cap = steps

    def take_process(self, brief: dict[str, Any]) -> None:
        for entry in as_list(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process_steps.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                           "messages": max(0, whole_number(entry.get("messages")) or 0)})
        if self.role("ask"):
            talk = [entry["messages"] for entry in self.process_steps if entry["when"] in ("order", "end") and entry["messages"]]
            self.kept_messages = min(max(talk), self.message_cap) if talk else 0

    def split_pointer_words(self) -> None:
        for number, item in enumerate(self.items, 1):
            aimed = [need for index, need in self.pointers if index == number and need in item.musts]
            rest = [must for must in item.musts if must not in aimed]
            if not aimed or not rest:
                continue
            prompt = (POINTER_SPLIT_GUIDE + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(aimed) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            found = self.ask(prompt, DEEP_FIRST, 500) or {}
            moved = [must for must in rest if must in [str(x).strip() for x in as_list(found.get("describe_other"))]]
            if not moved:
                continue
            joined = aimed[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if must == aimed[0] else must for must in item.musts if must not in moved]
            self.pointers = [(index, joined if (index, need) == (number, aimed[0]) else need) for index, need in self.pointers]
            self.trace.append({"kind": "split", "item": number, "moved": moved})

    def item_line(self, number: int, item: Item, many: bool) -> str:
        needs = [need + (f" (read from the page it names: {self.pointer_values[need]})" if need in self.pointer_values else "") for need in item.live()]
        head = f"item {number}: {item.name}" if many else f"item: {item.name}"
        if item.shelf:
            head += f" (the store files it under {item.shelf})"
        text = head + f" x{item.qty}; must have: " + ("; ".join(needs) if needs else "only the kind of product")
        return text + ("; the shopper leaves open: " + ", ".join(item.free) if item.free else "")

    def pick_line(self) -> str:
        pick = self.brief.get("pick")
        if pick == "cheapest":
            return "choice rule: the lowest-priced listing among those that fit"
        if pick in ("most", "least"):
            return f"choice rule: among fitting listings, the {pick} {self.brief.get('pick_measure') or ''}".replace("  ", " ")
        return "choice rule: any listing meeting every requirement"

    def view_line(self) -> str:
        facts = self.attribute_view()
        if not facts or not (self.shows_attributes.get(facts) is True or self.role("offer")):
            return ""
        bare = sorted(tool for tool in self.page_views() | {self.role("offer") or ""} if tool and self.shows_attributes.get(tool) is False)
        pages = sorted(tool for tool in self.page_views() if tool != facts)
        return (f"attributes of a listing: read them with {facts}" + (f" ({', '.join(bare)} carries none)" if bare else "")
                + (f"; {' or '.join(pages)} shows a listing's page, which is what a page check or a chain of pages refers to" if pages else ""))

    def ledger_text(self) -> str:
        many = len(self.items) > 1
        parts = [self.item_line(number, item, many) for number, item in enumerate(self.items, 1)]
        brief = self.brief
        if self.budget is not None:
            scope = "per item" if brief.get("budget_scope") == "each" else "for the whole order"
            parts.append(f"spending limit {scope}: {self.budget} {brief.get('budget_unit') or ''}".replace("  ", " "))
        parts.append(self.pick_line())
        if brief.get("in_stock_only", True) is not False:
            parts.append("only listings that can be bought now")
        parts += [f"nice to have (breaks ties only): {wish}" for wish in as_list(brief.get("wishes"))]
        parts += [f"the request says: {rule}" for rule in as_list(brief.get("rules"))]
        if brief.get("quiet_decline") is True:
            parts.append("when nothing fits: end the errand without naming or proposing a listing")
        stage_words = {"choose": "before the picks are made", "order": "at checkout (the code carries it out when you call the checkout tool)", "end": "when ending without a purchase (the code carries it out then)"}
        parts += [f"required step {stage_words[entry['when']]}: {entry['step']}" for entry in self.process_steps]
        views = self.view_line()
        if views:
            parts.append(views)
        return "\n".join("- " + part for part in parts)

    def pace_line(self) -> str:
        left = self.steps_left()
        if left <= self.finish_reserve() + 1:
            return " LAST STEPS: if every item has its cart line, check out now. Ending without a purchase is right only when your searches proved some item has no fitting listing; a lack of steps proves nothing."
        if left <= 6 or (left <= len(self.items) + 5 and self.checkouts == 0):
            return " Only a few steps remain: put the chosen lines in the cart and check out."
        if self.checkouts == 0 and left <= 12:
            return " The steps are getting short: decide your picks, cart them and check out soon."
        if self.checkouts == 0 and not self.cart_used and self.step >= self.step_cap // 2:
            return " Half the steps are gone and nothing is carted: put each item's fitting listing in the cart now, since lines can still sell out."
        return ""

    def live_needs(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.live()]

    def open_pointers(self) -> list[str]:
        live = {need for _, need in self.live_needs()}
        return [need for _, need in self.pointers if need in live and need not in self.pointer_values]

    def status_text(self) -> str:
        text = f"[status] step {self.step} of {self.step_cap} used, {self.steps_left()} left." + self.pace_line()
        lookups = self.open_pointers()
        if lookups and self.checkouts == 0:
            text += "\n[settle these pointers before anything else: open the listing each one names, then every listing that page sends you to; the final page holds the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[listings the reviews suggest] " + ", ".join(self.leads)
        if self.change_notes:
            text += "\n[what has changed] " + " | ".join(self.change_notes[-4:])
        return text + "\n[ledger]\n" + self.ledger_text()

    def listing_id(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_like_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(inner, dict) for inner in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = id_key_of([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_id_key(self, node: list[Any]) -> bool:
        rows = [entry for entry in node if isinstance(entry, dict) and len(entry) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(looks_like_id(row.get(self.id_key), True) for row in rows):
            return True
        found = id_key_of(rows)
        if found:
            self.id_key = found
            return True
        return False

    def keep_record(self, ident: str, node: dict[str, Any]) -> None:
        if ident not in self.rows:
            self.seen_ids.append(ident)
            self.rows[ident] = {}
        old = self.rows[ident]
        if any(old.get(key) is True and value is False for key, value in node.items()):
            self.unavailable.add(ident)
        elif any(old.get(key) is False and value is True for key, value in node.items()):
            self.unavailable.discard(ident)
        old.update(node)
        if attribute_block(node):
            self.detailed[ident] = dict(old)
        elif ident in self.detailed:
            self.detailed[ident].update(node)

    def take_in(self, observation: Any) -> None:
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.listing_id(node, in_list, depth)
                if ident:
                    self.keep_record(ident, node)
                    return
                for inner in node.values():
                    walk(inner, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_id_key(node)
                for inner in node:
                    walk(inner, depth + 1, rows)

        walk(observation, 0)

    def enum_labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            found.update(option.strip().lower() for option in schema.get("enum") or [] if isinstance(option, str))
            for inner in schema.values():
                if isinstance(inner, dict):
                    walk(inner, depth + 1)

        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def shelf_key(self) -> str | None:
        values: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    values.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.enum_labels()
        best = None
        for key, seen in values.items():
            if present[key] < 0.8 * total or len(seen) > 40:
                continue
            listed = sum(1 for value in seen if value.strip().lower() in known) / len(seen)
            mean = sum(len(value) for value in seen) / len(seen)
            score = (-listed, len(seen), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def wanted_kinds(self) -> list[tuple[str, str, set[str]]]:
        return [(key_words(item.shelf or ""), key_words(item.name), kind_words(item.shelf or "", item.name)) for item in self.items]

    def shelf_of(self, ident: str, key: str | None) -> str:
        return str((self.rows.get(ident) or {}).get(key) or "") if key else ""

    def attribute_groups(self) -> dict[str, dict[str, Any]]:
        key = self.shelf_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_ids:
            record = self.detailed.get(ident)
            if not record:
                continue
            label = str(record.get(key) or "listings") if key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            for field, value in (attribute_block(record) or {}).items():
                name = key_words(field)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = self.wanted_kinds()
        matched = {label for label in groups if label_matches(label, wanted)}
        return {label: group for label, group in groups.items() if label in matched} if matched else groups

    @staticmethod
    def useful_keys(group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(group["keys"].items(), key=lambda pair: -pair[1]["n"]):
            values = list(slot["per"].values())
            if not values:
                continue
            mean = sum(len(value) for value in values) / len(values)
            if mean > 40 or sum(1 for value in values if " / " in value) * 2 >= len(values):
                continue
            if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                continue
            if any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept):
                continue
            kept.append((name, slot))
        return kept[:14]

    def attribute_table(self) -> list[tuple[str, str, int, list[str]]]:
        table = []
        for label, group in sorted(self.attribute_groups().items(), key=lambda pair: -pair[1]["n"]):
            table += [(label, name, slot["n"], [value[:40] for value in slot["values"][:4]]) for name, slot in self.useful_keys(group)]
        return table

    def listings_by_item(self) -> list[list[str]]:
        key = self.shelf_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_ids:
            groups.setdefault(self.shelf_of(ident, key), []).append(ident)
        wanted = self.wanted_kinds()
        matched = [ids for label, ids in groups.items() if label_matches(label, wanted)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

    def one_listing_arg(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or "") or {}
        texts = free_text_fields(schema)
        required = schema.get("required") or []
        target = next((key for key in required if key in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(key != target for key in required):
            return None
        return target

    def price(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((inner for inner in value.values() if is_number(inner)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if is_number(value) else None

    def price_or_top(self, ident: str) -> float:
        figure = self.price(ident)
        return figure if figure is not None else float("inf")

    def is_gone(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.sold_at

    def can_buy(self, ident: str) -> bool:
        if self.is_gone(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (is_number(value) and value <= 0))

    def records_of(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        def from_list(entries: list[Any]) -> list[tuple[str, dict[str, Any]]]:
            return [(ident, row) for row in entries if isinstance(row, dict) for ident in [self.listing_id(row, True)] if ident]

        if isinstance(observation, dict):
            ident = self.listing_id(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += from_list(value)
                elif isinstance(value, dict):
                    for inner in value.values():
                        if isinstance(inner, list):
                            found += from_list(inner)
            return found
        return from_list(observation) if isinstance(observation, list) else []

    def note_shown(self, name: str, observation: Any) -> None:
        for ident, record in self.records_of(observation):
            fields = set(record)
            self.view_fields.setdefault(name, set()).update(fields)
            version, had = self.listing_fields.get(ident, (-1, set()))
            self.listing_fields[ident] = (self.version, (had if version == self.version else set()) | fields)

    def nothing_new(self, name: str, ident: str) -> bool:
        version, had = self.listing_fields.get(ident, (-1, set()))
        fields = self.view_fields.get(name)
        return bool(fields) and version == self.version and fields <= had and ident not in self.sold_at

    def result_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [key for key, value in observation.items() if isinstance(value, list) and len(value) >= 2 and all(isinstance(row, dict) for row in value)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {key: (value[:3] if key in lists else value) for key, value in observation.items()}
        prompt = (SHAPE_GUIDE + "\n\nTOOL: " + name + " | parameters: " + dump_json(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + dump_json(clip_values(sample, 120))[:4000])
        found = self.ask(prompt, QUICK_FIRST, 1500) or {}
        rows = found.get("rows") if found.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        params = (self.schemas.get(name) or {}).get("properties") or {}

        def present(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": present(found.get("count"), observation), "next": present(found.get("next"), observation),
                 "next_arg": present(found.get("next_arg"), params), "price": present(found.get("price"), first), "stock": present(found.get("stock"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "result_shape", "tool": name, "shape": shape})
        return shape

    def page_plan(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.result_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, onward = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if onward is not None and (isinstance(onward, bool) or not isinstance(onward, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], onward

    def sorted_by_price(self, observation: Any) -> bool:
        prices = [self.price(ident) for ident, _ in self.records_of(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def read_rest(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.self_paged = []
        pick = self.brief.get("pick")
        plan = self.page_plan(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, onward = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if onward is None or later or count <= shown:
            return ""
        if pick == "cheapest" and self.sorted_by_price(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.self_pages + pages > SELF_PAGES or self.steps_left() - self.finish_reserve() - pages < 12:
            return (f"\n[code] {count} listings match this search but only {shown} are shown, and not in price order. Since the choice rule weighs every fitting listing, "
                    f"either narrow the search with the requirement's value or go on to the next page with {param}={dump_json(onward)}.")
        texts = []
        for _ in range(pages):
            call = shaped({**args, param: onward}, self.schemas.get(name))
            if call is None:
                break
            reply = self.call_store(name, call)
            if reply is None or self.finished:
                break
            self.self_pages += 1
            seen = reply.get("observation")
            self.take_result(name, seen)
            self.self_paged.append(seen)
            shown_text = render(seen)
            self.read_cache[name + dump_json(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={dump_json(onward)}: {shown_text}")
            after = self.page_plan(name, seen)
            onward = after[3] if after else None
            if onward is None:
                break
        self.trace.append({"kind": "self_pages", "tool": name, "read": len(texts)})
        return ("\n[the remaining pages of this search, read by the code at one step each]\n" + "\n".join(texts)) if texts else ""

    def cheapest_rows(self, observations: list[Any]) -> str:
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {ident for seen in observations for ident, _ in self.records_of(seen)}
        rows = [(self.price(ident), ident) for ident in found if self.can_buy(ident) and self.price(ident) is not None]
        if len(rows) < 2:
            return ""
        return "\n[code] rows here that can be bought, lowest price first: " + ", ".join(f"{ident} ({figure:g})" for figure, ident in sorted(rows)[:6])

    def broad_search_hold(self) -> str:
        if self.broad_held or self.brief.get("pick") != "cheapest" or self.steps_left() < 14 + 2 * len(self.items):
            return ""
        self.broad_held = True
        lines = [f"{ident} at {self.price(ident):g}" for ident in self.cart_listing_ids() if self.price(ident) is not None]
        return (f"Held for now (no step used). The choice rule is the lowest price and {self.steps_left()} steps remain. A store search mostly matches titles,"
                " so searching a requirement's value (which often sits only in the attributes) reaches few fitting listings, and the rows are unordered by price."
                " First search each item a few more times under other product names (the bare one-word name, synonyms, words from fitting titles already seen),"
                " keeping only buyable rows cheaper than the carted line" + (f" ({'; '.join(lines)})" if lines else "")
                + "; read the attributes of the cheapest of them, swap in one that fits, and then check out.")

    def read_attributes(self, ids: list[str], keep: int) -> list[str]:
        tool = self.attribute_view()
        target = self.one_listing_arg(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [ident for ident in ids if ident in self.rows and ident not in self.detailed and not self.is_gone(ident)]:
            if self.steps_left() <= keep or self.finished or self.shows_attributes.get(tool) is False:
                break
            args = shaped({target: ident}, self.schemas.get(tool))
            if args is None:
                break
            reply = self.call_store(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.take_result(tool, seen)
            self.note_shown(tool, seen)
            if isinstance(seen, dict) and self.listing_id(seen):
                self.shows_attributes[tool] = self.shows_attributes.get(tool, False) or bool(attribute_block(seen))
            texts.append(f"{ident}: " + render(seen)[:1500])
        if texts:
            self.trace.append({"kind": "attr_reads", "n": len(texts)})
        return texts

    def cheaper_reads(self) -> list[str]:
        if self.brief.get("pick") != "cheapest":
            return []
        stamp = self.version_stamp()
        key = self.shelf_key()
        if stamp in self.cheaper_stamps or not key:
            return []
        self.cheaper_stamps.add(stamp)
        cart = self.cart_listing_ids()
        picks: list[str] = []
        for line in cart:
            figure = self.price(line)
            kind = self.shelf_of(line, key).strip().lower()
            if figure is None or not kind:
                continue
            pool = [ident for ident in self.seen_ids if ident not in cart and ident not in self.detailed and self.can_buy(ident)
                    and self.shelf_of(ident, key).strip().lower() == kind and self.price_or_top(ident) < figure]
            pool.sort(key=lambda ident: self.price(ident) or 0.0)
            picks += [ident for ident in pool[:2] if ident not in picks]
        return self.read_attributes(picks[:CHEAP_READS], self.finish_reserve() + 4)

    def take_result(self, name: str, observation: Any) -> str:
        self.take_in(observation)
        lines = self.note_lines(observation)
        if not lines:
            return ""
        meaning = self.read_note(lines)
        if meaning:
            self.pending_notes.append(f"(seen on {name}) {meaning}")
        return meaning

    def batch_arg(self) -> str | None:
        schema = self.schemas.get(self.role("compare") or "") or {}
        fields = schema.get("properties") or {}
        target = next((key for key, spec in fields.items() if isinstance(spec, dict) and "array" in type_list(spec)), None)
        if not target or any(key != target for key in schema.get("required") or []):
            return None
        return target

    def batch_room(self) -> int:
        spec = ((self.schemas.get(self.role("compare") or "") or {}).get("properties") or {}).get(self.batch_arg() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.batch_bad:
            return max(2, min(self.batch_bad - 1, max(self.batch_ok, 2)))
        return max(4, self.batch_ok)

    def open_more(self) -> None:
        tool = self.role("compare")
        schema = self.schemas.get(tool or "") or {}
        fields = schema.get("properties") or {}
        target = next((key for key, spec in fields.items() if isinstance(spec, dict) and "array" in type_list(spec)), None)
        if not tool or not target or any(key != target for key in schema.get("required") or []) or self.batch_is_bare():
            self.open_more_singly()
            return
        if self.steps_left() <= 12:
            return
        cap = fields[target].get("maxItems") if isinstance(fields[target].get("maxItems"), int) else 4
        ranked = self.listings_by_item()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [ident for ident in ids if ident not in self.detailed][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        args = shaped({target: picks}, schema)
        if args is None:
            return
        before = len(self.detailed)
        reply = self.call_store(tool, args)
        if reply is None:
            return
        seen = reply.get("observation")
        self.take_result(tool, seen)
        self.note_shown(tool, seen)
        rows = self.records_of(seen)
        if rows:
            self.shows_attributes[tool] = self.shows_attributes.get(tool, False) or any(attribute_block(row) for _, row in rows)
        self.trace.append({"kind": "more_opened", "ids": picks, "opened": len(self.detailed) - before})
        if rows and self.shows_attributes.get(tool) is False:
            self.open_more_singly()
            return
        if len(self.detailed) == before and len(picks) > 2 and self.steps_left() > 12:
            again = shaped({target: picks[:2]}, schema)
            if again is not None:
                reply = self.call_store(tool, again)
                if reply is not None:
                    self.take_result(tool, reply.get("observation"))

    def open_more_singly(self) -> None:
        tool = self.attribute_view()
        if not tool or not self.one_listing_arg(tool):
            return
        lanes: list[list[str]] = []
        for ids in self.listings_by_item():
            opened = sum(1 for ident in ids if ident in self.detailed)
            pool = [ident for ident in ids if ident not in self.detailed and self.can_buy(ident)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=self.price_or_top)
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[turn] for turn in range(2) for lane in lanes if turn < len(lane)]
        self.read_attributes(queue[:5], 22)

    def time_to_ask(self) -> bool:
        if self.interviewed or not self.role("ask") or self.spare_messages() <= 0 or self.steps_left() <= 8:
            return False
        return len(self.detailed) >= 3 * len(self.items) + 1 or self.step >= max(8, self.step_cap // 4)

    def message_call(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.role("ask") or "") or {}
        fields = schema.get("properties") or {}
        texts = free_text_fields(schema)
        required = schema.get("required") or []
        target = next((key for key in required if key in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in required:
            if key in args:
                continue
            options = (fields.get(key) or {}).get("enum")
            if not (isinstance(options, list) and options):
                return None
            args[key] = options[0]
        return shaped(args, schema)

    def settled_lines(self) -> list[str]:
        return [f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing") for number, item in enumerate(self.items, 1)]

    def key_lines(self) -> list[str]:
        lines = []
        last = None
        for index, (label, name, carried, values) in enumerate(self.attribute_table(), 1):
            if label != last:
                lines.append(f"category {label!r}:")
                last = label
            lines.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        return lines

    def planned_labels(self, count: int) -> list[tuple[int, str]]:
        prompt = PLANNER_GUIDE.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(self.settled_lines())
        first = [str(entry) for entry in as_list(self.brief.get("ask_first")) if str(entry).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(self.key_lines()) or "(no full listing records yet)")
        plan = self.ask(prompt, DEEP_FIRST, 1800) or {}
        ordered: list[tuple[int, str]] = []
        for pick in as_list(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, whole_number(pick.get("item") or 1) or 1), len(self.items))
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(key_words(label) != key_words(other) for index, other in ordered if index == number):
                ordered.append((number, label))
        self.trace.append({"kind": "ask_plan", "ask": clip_values(plan.get("ask"), 120)})
        return ordered

    def second_round(self, head: list[tuple[int, str]], ordered: list[tuple[int, str]], batch: list[tuple[int, str, str]], heard: dict[str, Any]) -> tuple[int, str, str] | None:
        crowded = [batch[index - 1] for index in (whole_number(entry) for entry in as_list(heard.get("crowded"))) if index and 1 <= index <= len(batch)]
        for number, text, _ in crowded:
            label = next((name for index, name in head if index == number and self.question_text(index, name) == text), None)
            examples = self.example_values(number, label) if label else []
            if label and len(examples) >= 2:
                return number, label, self.question_text(number, label, examples)
        if len(ordered) > len(head):
            number, label = ordered[len(head)]
            return number, label, self.question_text(number, label)
        return None

    def run_interview(self) -> str:
        self.interviewed = True
        count = self.spare_messages()
        if count <= 0:
            return ""
        self.open_more()
        ordered = self.planned_labels(count)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        head = ordered[: count - 1] if count >= 2 else ordered[:count]
        batch = self.send_questions([(number, label, self.question_text(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        heard = self.read_answers(batch)
        self.apply_answers(heard, told)
        if count >= 2 and self.spare_messages() > 0:
            again = self.second_round(head, ordered, batch, heard)
            if again:
                extra = self.send_questions([again])
                if extra:
                    sent += extra
                    self.apply_answers(self.read_answers(extra, QUICK_FIRST), told)
        if told:
            self.version += 1
        return self.interview_summary(sent, told)

    def interview_summary(self, sent: list[tuple[int, str, str]], told: list[str]) -> str:
        summary = "Questions the interviewer put to the shopper (one step each):\n" + "\n".join(f"Q: {question}\nA: {answer}" for _, question, answer in sent)
        summary += "\nFirm requirements these answers add: " + ("; ".join(told) if told else "none")
        summary += ". Every pick has to meet them; run new searches with these values where that helps."
        aimed = [need for _, need in self.pointers if any(need in entry for entry in told)]
        if aimed:
            summary += (" Their value lives on another listing: " + " | ".join(aimed) + ". Locate that listing and open its full page, then each listing that page"
                        " refers to; the final page gives the value, and an order counts only once all of those pages are open.")
        if self.pending_notes:
            summary += "\n[store notes that arrived in the meantime] " + " | ".join(self.pending_notes)
            self.pending_notes = []
        return summary

    def question_text(self, number: int, label: str, examples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {examples[0]} or {examples[1]}?" if examples else "?")

    def example_values(self, number: int, label: str) -> list[str]:
        want = key_words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.attribute_table():
            if name == want:
                return [value for value in values if value and value not in settled and len(value) <= 30][:2]
        return []

    def send_questions(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.steps_left() <= floor or self.messages_sent >= self.message_cap:
                break
            args = self.message_call(text)
            if not args:
                break
            reply = self.call_store(tool, args)
            if reply is None:
                break
            self.messages_sent += 1
            answer = reply.get("observation")
            words = spoken_text(answer)
            sent.append((number, text, words))
            self.record_lines.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_stamp()})")
            self.record_steps.append(self.step)
            self.trace.append({"kind": "question", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def apply_answers(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in as_list(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = item_index(entry.get("item"), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.pointers:
                self.pointers.append((number, need))
        for entry in as_list(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[item_index(entry.get("item"), len(self.items)) - 1]
                prop = key_words(str(entry["property"]).split(":")[0])
                hit = [need for need in item.live() if key_words(need.split(":")[0]) == prop]
                item.dropped.extend(hit)
                told.extend(f"dropped {need}" for need in hit)
        for entry in as_list(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                self.items[item_index(entry.get("item"), len(self.items)) - 1].free.append(str(entry["property"]))
        budget = heard.get("budget")
        if is_number(budget) and budget > 0:
            self.budget = budget

    def read_answers(self, sent: list[tuple[int, str, str]], models: list[str] | None = None) -> dict[str, Any]:
        prompt = LISTENER_GUIDE + "\n\nITEMS:\n" + "\n".join(f"item {number}: {item.name}" for number, item in enumerate(self.items, 1))
        prompt += f"\n\nLIMIT IN FORCE: {self.budget} {self.brief.get('budget_unit') or ''}".rstrip() if self.budget is not None else "\n\nLIMIT IN FORCE: none"
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{index}. (about item {number}) Q: {question}\nA: {answer}" for index, (number, question, answer) in enumerate(sent, 1))
        return self.ask(prompt, models or DEEP_FIRST, 1500) or {}

    def gate_prompt(self, stage: str, due: list[dict[str, Any]], tools: list[str]) -> str:
        notes = "\n\nSTORE NOTES SEEN:\n" + "\n".join(self.change_notes[-4:]) if self.change_notes else ""
        record = "\n".join(f"{index}. {line}" for index, line in enumerate(self.record_lines, 1)) or "(nothing yet)"
        closing = "order" if stage == "order" else "ending"
        return (PROCESS_GUIDE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {entry['step']}" for entry in due)
                + "\n\nLEDGER:\n" + self.ledger_text() + "\n\nCART NOW: " + self.cart_summary()
                + "\n\nRECORD (messages and step calls so far):\n" + record + notes
                + f"\n\nMESSAGES LEFT: {max(0, self.message_cap - self.messages_sent)}   STEPS LEFT: {self.steps_left()} (the {closing} itself takes one)"
                + "\n\nTOOLS:\n" + "\n".join(tools))

    def unproven_steps(self, stage: str, due: list[dict[str, Any]], found: dict[str, Any]) -> list[dict[str, Any]]:
        shown = [entry for entry in as_list(found.get("steps")) if isinstance(entry, dict)]
        talk = stage == "order" and any(entry["messages"] > 0 for entry in due)
        proof = {index for index in range(1, len(self.record_lines) + 1)
                 if not talk or (index <= len(self.record_steps) and self.record_steps[index - 1] > self.cart_changed_at)}
        return [entry for entry in shown if entry.get("done") is not True or not any((whole_number(line) or 0) in proof for line in as_list(entry.get("lines")))]

    def process_gate(self, stage: str) -> tuple[bool, str]:
        due = [entry for entry in self.process_steps if entry["when"] == stage]
        if not due or self.finished or not self.steps_pending(stage):
            return True, ""
        ending = self.ending_tools()
        tools = self.tool_lines(ending)
        said: list[str] = []
        nudged = False
        for _ in range(GATE_TURNS):
            found = self.ask(self.gate_prompt(stage, due, tools), DEEP_FIRST if nudged else QUICK_FIRST, 2000)
            if found is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.apply_answers({"needs": found.get("needs")}, told)
                hold = str(found.get("hold") or "").strip()
            call = found.get("call") if isinstance(found.get("call"), dict) else None
            self.trace.append({"kind": "process", "stage": stage, "call": clip_values(call, 200), "steps": clip_values(found.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                done = ("The code first carried out the steps the request requires:\n" + "\n".join(said) + "\n") if said else ""
                return False, done + "The shopper's answer changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Adjust the cart to it and check out once more."
            name = str((call or {}).get("tool") or "")
            unproven = self.unproven_steps(stage, due, found)
            if not call and unproven and not said and not nudged and self.role("ask") and self.messages_sent < self.message_cap and self.steps_left() >= 2:
                nudged = True
                self.trace.append({"kind": "process_nudge", "steps": clip_values(unproven, 120)})
                self.record_lines.append(f"(code note: since the cart last changed at step {self.cart_changed_at}, no line shows these as done: " + "; ".join(str(entry.get("step") or "") for entry in unproven)[:200]
                                         + f"; {self.message_cap - self.messages_sent} message(s) and {self.steps_left()} steps remain, so name the call that does it)")
                self.record_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_passed.add((stage, self.version_stamp()))
                break
            if self.steps_left() < 2 or (name == self.role("ask") and self.messages_sent >= self.message_cap):
                break
            args = shaped(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name])
            if args is None:
                break
            if name == self.role("ask"):
                text = next((value for value in args.values() if isinstance(value, str)), "")
                batch = self.send_questions([(1, "", text)], 1)
                if not batch:
                    break
                self.gate_messages += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.call_store(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.take_in(seen)
            self.record_lines.append(f"step {self.step}: {name} {dump_json(args)} -> {render(seen)[:300]} (cart then: {self.cart_stamp()})")
            self.record_steps.append(self.step)
            said.append(f"{name} {dump_json(args)} -> {render(seen)[:400]}")
        return True, ("The code first carried out the steps the request requires:\n" + "\n".join(said)) if said else ""

    def note_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.listing_id(observation):
            for value in observation.values():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(entry, str) for entry in value):
                    lines.extend(entry for entry in value if len(entry.split()) >= 5)
                elif isinstance(value, dict) and not self.listing_id(value, False, 1):
                    lines.extend(inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6)
        return [line for line in lines if line not in self.notes_read]

    def drop_from_note(self, entry: dict[str, Any], changed: list[str]) -> None:
        item = self.items[item_index(entry.get("item"), len(self.items)) - 1]
        target = str(entry["need"]).strip()
        live = item.live()
        hit = [need for need in live if need == target] or [need for need in live if key_words(need.split(":")[0]) == key_words(target.split(":")[0])][:1]
        item.dropped.extend(hit or [target])
        changed.append(f"dropped {target}")

    def add_from_note(self, entry: dict[str, Any], changed: list[str]) -> None:
        item = self.items[item_index(entry.get("item"), len(self.items)) - 1]
        need = str(entry["need"]).strip()
        if need in item.dropped:
            item.dropped.remove(need)
        if need not in item.musts + item.told:
            item.told.append(need)
        changed.append(f"added {need}")

    def read_note(self, lines: list[str]) -> str:
        self.notes_read.update(lines)
        found = self.ask(NOTE_GUIDE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger_text() + "\n\nTHE NOTE:\n" + "\n".join(lines), NOTE_SEATS, 1200) or {}
        gone = [str(entry) for entry in as_list(found.get("gone")) if isinstance(entry, (str, int))]
        self.unavailable.update(ident for ident in gone if ident in self.rows)
        self.note_says_sold = bool(gone) or found.get("sold_out") is True
        if self.note_says_sold and len(lines) == 1:
            self.sold_phrases.update(lines)
        if found.get("kind") not in (None, "change"):
            return ""
        changed: list[str] = []
        budget = found.get("budget")
        if is_number(budget) and budget > 0 and budget != self.budget:
            self.budget = budget
            changed.append(f"budget now {budget}")
        for entry in as_list(found.get("drop")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                self.drop_from_note(entry, changed)
        for entry in as_list(found.get("add")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                self.add_from_note(entry, changed)
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if found.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(found.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent_calls = []
            self.change_notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def evidence_line(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.listing_id(node, False, depth)
                if ident and attribute_block(node) and ident not in ids:
                    ids.append(ident)
                    return
                for inner in node.values():
                    walk(inner, depth + 1)
            elif isinstance(node, list):
                for inner in node:
                    walk(inner, depth + 1)

        walk(observation, 0)
        needs = self.live_needs()
        if not ids or not needs:
            return ""
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = attribute_block(self.detailed.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{key}={str(value)[:50]}" for key, value in block.items() if names_same(word_set(prop), word_set(key))]
                if hits:
                    parts.append((f"(item {number}) " if many else "") + f"{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: attribute keys whose names carry a requirement's property; a key about some other part or aspect settles nothing, and a differently named key can still hold the same property] " + " | ".join(lines)

    def cart_is_fresh(self) -> bool:
        return self.cart_snapshot is not None and self.cart_viewed_at >= self.cart_changed_at

    def cart_summary(self) -> str:
        if self.cart_is_fresh():
            return dump_json(clip_values(self.cart_snapshot, 300))[:3000]
        return "lines by the code's record of the cart calls (listing: units): " + dump_json(self.cart_units)

    def cart_listing_ids(self) -> list[str]:
        if self.cart_is_fresh():
            text = dump_json(self.cart_snapshot)
            return [ident for ident in self.rows if ident in text]
        return list(self.cart_units)

    def record_text(self, ident: str) -> str:
        record = self.detailed.get(ident) or self.rows.get(ident)
        return dump_json(clip_values(record, 400))[:3000] if record else "(no record seen)"

    def other_records(self, skip: list[str], limit: int = 30) -> str:
        order = list(reversed(self.seen_ids))
        if self.brief.get("pick") == "cheapest":
            order.sort(key=lambda ident: (ident not in self.detailed, self.price_or_top(ident)))
        lines = []
        for ident in order:
            if ident in skip or self.is_gone(ident):
                continue
            lines.append(dump_json(clip_values(self.detailed.get(ident) or self.rows.get(ident), 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def verdict_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def cart_review(self, ids: list[str] | None = None, limit: float = SOFT_SPEND) -> str:
        planned = ids is not None
        ids = ids if planned else self.cart_listing_ids()
        key = self.verdict_key(ids)
        if key in self.reviewed or self.seats.cost > limit or not ids:
            return ""
        self.reviewed.add(key)
        prompt = CART_CHECK_GUIDE + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger_text()
        if planned:
            quantity = "; ".join(f"item {number}: quantity {item.qty}" for number, item in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_summary()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{ident}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.is_gone(ident) else "") + self.record_text(ident) for ident in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.other_records(ids, 15)
        found = self.ask(prompt, DEEP_FIRST, 1500)
        self.trace.append({"kind": "cart_check", "cart": ids, "got": found})
        self.verdicts[key] = bool(found) and found.get("ok") is not False
        if not found or found.get("ok") is not False:
            return ""
        self.leads = [ident for ident in self.leads if ident not in ids]
        problems = [str(problem) for problem in as_list(found.get("problems"))][:5]
        better = [str(entry) for entry in as_list(found.get("better")) if str(entry) in self.rows and not self.is_gone(str(entry))][:4]
        text = "An independent review finds the cart not yet right: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Better candidates: " + ", ".join(better)
        return text + ". Correct the cart, or re-read the records to show the review is mistaken, and then check out."

    def chain_records(self) -> list[str]:
        records = []
        for ident in reversed(self.seen_ids):
            record = (self.detailed.get(ident) or self.rows.get(ident)) if ident in self.opened_pages else None
            records.append("OPENED " + dump_json(clip_values(record, 300))[:1400] if record else "ROW " + dump_json(clip_values(self.rows.get(ident), 80))[:300])
            if len(records) >= 60:
                break
        return records

    def walk_chains(self, pending: list[str], found: dict[str, Any]) -> tuple[list[str], list[str]]:
        reach: list[str] = []
        lost: list[str] = []
        for chain in as_list(found.get("chains")):
            if not isinstance(chain, dict):
                continue
            need = next((entry for entry in pending if entry == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
            if not need:
                continue
            links = [link for link in as_list(chain.get("links")) if isinstance(link, dict)]
            ids = [str(link.get("id") or "").strip() for link in links]
            if chain.get("done") is True and links and all(ident in self.opened_pages for ident in ids) and str(chain.get("value") or "").strip():
                self.pointer_values[need] = str(chain["value"]).strip()
                self.version += 1
                continue
            closed = [ident for ident in ids if ident in self.rows and ident not in self.opened_pages]
            if closed:
                reach += [ident for ident in closed if ident not in reach]
            else:
                about = next((str(link.get("about") or "") for link, ident in zip(links, ids) if ident not in self.rows), "") or "the listing it points at"
                lost.append(f"{need} -> find {about}")
        return reach, lost

    def follow_pointers(self) -> tuple[str, bool]:
        pending = self.open_pointers()
        tool = self.role("offer") or self.role("open")
        target = self.one_listing_arg(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.chain_rounds < 4 and self.seats.cost < HARD_SPEND:
            self.chain_rounds += 1
            prompt = CHAIN_GUIDE + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending) + "\n\nRECORDS (newest first):\n" + "\n".join(self.chain_records())
            found = self.ask(prompt, QUICK_FIRST, 1800) or {}
            self.trace.append({"kind": "chain", "pending": pending, "got": clip_values(found, 150)})
            reach, lost = self.walk_chains(pending, found)
            pending = [entry for entry in pending if entry not in self.pointer_values]
            if not reach or not tool or not target or self.steps_left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                args = shaped({target: ident}, self.schemas.get(tool))
                if args is None:
                    break
                reply = self.call_store(tool, args)
                if reply is None or self.finished:
                    break
                self.take_result(tool, reply.get("observation"))
                seen.append(f"{ident}: " + render(reply.get("observation"))[:1500])
        text = ("[pages that requirements point at, opened by the code]\n" + "\n".join(seen) + "\n") if seen else ""
        if self.pointer_values:
            text += "[values taken from the pointed-at pages] " + "; ".join(f"{need} -> {value}" for need, value in self.pointer_values.items()) + "\n"
        hold = bool(lost) and self.chain_holds < 2 and self.steps_left() > 5
        if hold:
            self.chain_holds += 1
            text += ("An order only counts once every page a requirement points at is open in full (a search row does not count), including each further listing such a page refers to. Not found and opened yet: "
                     + " | ".join(lost[:3]) + ". Search by its shop, brand or wording, open it, use the value its page states, and then check out.")
        return text, hold

    def offer_view(self) -> str | None:
        return self.role("offer") or self.role("open")

    def refresh_offers(self) -> str:
        tool = self.offer_view()
        target = self.one_listing_arg(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_step)
        stale = [ident for ident in self.cart_listing_ids() if self.offer_read_at.get(ident, -99) < since and ident not in self.sold_at]
        if not stale or self.steps_left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            args = shaped({target: ident}, self.schemas.get(tool))
            if args is None:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.call_store(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.offer_read_at[ident] = self.step
            self.take_result(tool, observation)
            after = self.rows.get(ident) or {}
            moved = [key for key, value in after.items() if key in before and before[key] != value and not isinstance(value, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {dump_json(clip_values(observation, 300))[:900]}")
        self.trace.append({"kind": "offer_reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out_note(self) -> str:
        lost = [ident for ident in self.cart_listing_ids() if ident not in self.sold_at and self.is_gone(ident)]
        if not lost:
            return ""
        for ident in lost:
            self.sold_at[ident] = self.step
        self.leads = [ident for ident in self.leads if ident not in lost]
        self.version += 1
        picks = self.find_replacements(lost)
        self.keep_swap_leads(lost, picks)
        text = f"[code] No longer for sale: {', '.join(lost)} (the store shows it cannot be bought). It will not return, so do not open it again."
        if picks and self.steps_left() <= 6:
            return text + f" Steps are short: this turn, add {picks[0]} at the requested quantity and take out the sold-out line (two calls), then check out. Other read listings that seem to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Read listings that seem to fit that item instead, lowest price first: {', '.join(picks)}. The choice rule is the lowest price, so unless the whole fitting set is already known, run one search below {picks[0]}'s price (with the search's price ceiling when it has one) before replacing the sold-out line with the cheapest fit."
        if picks:
            return text + f" Read listings that seem to fit that item instead: {', '.join(picks)}. This turn, take the sold-out line out of the cart, add {picks[0]} at the requested quantity, and check out."
        return text + " Take that line out of the cart and add the best other fitting listing, searching for one when nothing read so far fits."

    def item_of(self, ident: str) -> int | None:
        if len(self.items) == 1:
            return 1
        label = self.shelf_of(ident, self.shelf_key())
        matches = [number for number, item in enumerate(self.items, 1) if label and label_matches(label, [(key_words(item.shelf or ""), key_words(item.name), kind_words(item.shelf or "", item.name))])]
        return matches[0] if len(matches) == 1 else None

    def keep_swap_leads(self, lost: list[str], picks: list[str]) -> None:
        for ident in lost:
            number = self.item_of(ident)
            if number and picks:
                self.swap_leads[str(number)] = list(picks)

    def swap_lead_hold(self) -> bool:
        named = {number: [ident for ident in ids if not self.is_gone(ident)] for number, ids in self.swap_leads.items()}
        named = {number: ids for number, ids in named.items() if ids}
        if self.leads or not named:
            return False
        self.leads_by_item = named
        self.leads = [ident for ids in named.values() for ident in ids]
        self.trace.append({"kind": "swap_leads", "leads": list(self.leads)})
        return True

    def locate_sold(self) -> str:
        ids = [ident for ident in self.cart_listing_ids() if ident not in self.sold_at]
        if any(self.is_gone(ident) for ident in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.refresh_offers() if ids else ""

    def find_replacements(self, lost: list[str]) -> list[str]:
        if self.seats.cost > SOFT_SPEND:
            return []
        records = []
        for ident in reversed(self.seen_ids):
            if ident in lost or ident in self.sold_at or self.is_gone(ident):
                continue
            records.append(dump_json(clip_values(self.detailed.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP_GUIDE + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger_text()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{ident}: {self.record_text(ident)}" for ident in lost) + "\n\nCART NOW:\n" + self.cart_summary()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        found = self.ask(prompt, DEEP_FIRST, 1200)
        fits: list[tuple[float, str]] = []
        cart = self.cart_listing_ids()
        for entry in as_list((found or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            figure = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.sold_at and ident not in cart and ident not in [known for _, known in fits]:
                fits.append((float(figure) if is_number(figure) else float("inf"), ident))
        ranked = [ident for _, ident in sorted(fits, key=lambda fit: fit[0])] if self.brief.get("pick") == "cheapest" else [ident for _, ident in fits]
        self.trace.append({"kind": "replacements", "lost": lost, "fits": fits})
        return ranked[:4]

    def decline_review(self) -> str:
        if self.decline_reviewed or self.seats.cost > SOFT_SPEND:
            return ""
        self.decline_reviewed = True
        found = self.review_call()
        named = self.review_fits(found)
        if not named:
            rows = [str(ident) for ident in as_list((found or {}).get("open")) if str(ident) in self.rows and str(ident) not in self.opened_pages and not self.is_gone(str(ident))]
            if rows and self.open_listed(rows):
                found = self.review_call()
                named = self.review_fits(found)
        if not named:
            return ""
        self.leads_by_item = named
        self.leads = [ident for ids in named.values() for ident in ids]
        reason = f" ({found.get('reason')})" if found.get("reason") else ""
        return "An independent review believes these listings could fit: " + "; ".join(f"item {key}: {', '.join(ids)}" for key, ids in named.items()) + reason + ". Open them and buy whichever really fits; end without buying only when every one of them fails."

    def review_call(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.seen_ids):
            if self.is_gone(ident):
                continue
            record = self.detailed.get(ident)
            records.append("OPENED " + dump_json(clip_values(record, 200))[:1200] if record else "ROW " + dump_json(clip_values(self.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = DECLINE_CHECK_GUIDE + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger_text()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        found = self.ask(prompt, DEEP_FIRST, 1500)
        self.trace.append({"kind": "decline_check", "got": found})
        return found

    def review_fits(self, found: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (found or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(key): [str(ident) for ident in as_list(ids) if str(ident) in self.rows and not self.is_gone(str(ident))][:4] for key, ids in fits.items()}
        return {key: ids for key, ids in named.items() if ids}

    def open_listed(self, rows: list[str]) -> bool:
        many, joiner = self.batch_arg(), self.role("compare")
        tool = self.attribute_view()
        target = self.one_listing_arg(tool)
        if many and joiner and not self.batch_is_bare():
            plan = [(joiner, {many: rows[: self.batch_room()]})]
        elif tool and target:
            plan = [(tool, {target: ident}) for ident in rows[:2]]
        else:
            return False
        if self.steps_left() < len(plan) + 5:
            return False
        before = len(self.detailed)
        for name, args in plan:
            fixed = shaped(args, self.schemas.get(name))
            if fixed is None:
                return False
            reply = self.call_store(name, fixed)
            if reply is None or self.finished:
                break
            self.take_result(name, reply.get("observation"))
        self.trace.append({"kind": "decline_opens", "rows": rows, "opened": len(self.detailed) - before})
        return len(self.detailed) > before

    def cart_sets_count(self, name: str) -> bool:
        schema = self.schemas.get(name, {})
        if mode_field(schema):
            return False
        fields = schema.get("properties") or {}
        return any(isinstance(spec, dict) and "integer" in type_list(spec) and spec.get("minimum") == 0 for spec in fields.values())

    def list_cart_plan(self, key: str, spec: Any, lines: dict[str, int]) -> list[dict[str, Any]] | None:
        inner = spec["properties"]
        id_key = next((name for name, sub in inner.items() if isinstance(sub, dict) and "string" in type_list(sub) and "enum" not in sub), None)
        units = next((name for name, sub in inner.items() if isinstance(sub, dict) and "integer" in type_list(sub)), None)
        if not id_key:
            return None
        return [{key: [{id_key: ident, **({units: count} if units else {})} for ident, count in lines.items()]}]

    def cart_plan(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.role("cart") or ""
        schema = self.schemas.get(name) or {}
        fields = schema.get("properties") or {}
        for key, spec in fields.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in type_list(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                return self.list_cart_plan(key, shape, lines)
        texts = free_text_fields(schema)
        target = (next((key for tool, key in self.id_params if tool == name and key in texts), None)
                  or next((key for key in schema.get("required") or [] if key in texts), None) or (texts[0] if len(texts) == 1 else None))
        units = count_field(schema)
        mode = mode_field(schema)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_units.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: fields[mode]["enum"][1], **({units: have} if units else {})})
            elif units and self.cart_sets_count(name):
                calls.append({target: ident, units: 0})
            elif units and isinstance(fields[units].get("minimum"), int) and fields[units]["minimum"] <= -have:
                calls.append({target: ident, units: -have})
            else:
                return None
        cart = self.cart_listing_ids()
        for ident, want in lines.items():
            have = self.cart_units.get(ident, 0) if ident in cart else 0
            if have == want:
                continue
            if have and not self.cart_sets_count(name):
                return None
            calls.append({target: ident, **({mode: fields[mode]["enum"][0]} if mode else {}), **({units: want} if units else {})})
        return calls

    def reviewed_plan(self) -> tuple[dict[str, int], list[str]] | None:
        key = self.shelf_key()

        def kind(ident: str) -> str:
            return self.shelf_of(ident, key).strip().lower() if key else ident

        plan: dict[str, int] = {}
        for number, ids in self.leads_by_item.items():
            index = whole_number(number)
            pick = next((ident for ident in ids if not self.is_gone(ident)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(chosen) for chosen in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed_id not in plan:
            return None
        now = self.cart_listing_ids()
        kinds = {kind(chosen) for chosen in plan}
        keep = {ident: self.cart_units.get(ident, 1) for ident in now if ident not in plan and not self.is_gone(ident) and kind(ident) not in kinds}
        drop = [ident for ident in now if ident not in plan and ident not in keep]
        if len(plan) + len({kind(ident) for ident in keep}) < len(self.items):
            return None
        return {**keep, **plan}, drop

    def finish_reviewed(self) -> str:
        cart, buy = self.role("cart"), self.role("buy")
        if self.self_finish_done or not self.confirmed_id or self.is_gone(self.confirmed_id) or not cart or not buy:
            return ""
        self.self_finish_done = True
        planned = self.reviewed_plan()
        if not planned:
            return ""
        lines, drop = planned
        calls = self.cart_plan(lines, drop)
        if not calls or len(calls) + 1 > self.steps_left():
            return ""
        self.read_attributes(list(lines), len(calls) + 3)
        self.cart_review(list(lines), HARD_SPEND)
        if not self.verdicts.get(self.verdict_key(list(lines))):
            return ""
        done = []
        for args in calls:
            fixed = shaped(args, self.schemas.get(cart))
            if fixed is None:
                return ""
            reply = self.call_store(cart, fixed)
            if reply is None or self.finished:
                return ""
            self.track_cart(cart, fixed)
            self.take_result(cart, reply.get("observation"))
            done.append(dump_json(fixed))
        args = shaped(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy))
        if args is None:
            return ""
        replies = self.checkout_twice(buy, args, lines)
        if not replies:
            return ""
        return ("Not run: the code acted instead. The independent check confirmed " + self.confirmed_id + ", so the code arranged the cart (" + "; ".join(done)
                + ") and checked out. The store answered: " + " | next: ".join(replies))

    def checkout_twice(self, buy: str, args: dict[str, Any], lines: dict[str, int]) -> list[str]:
        replies = []
        for attempt in range(2):
            if attempt:
                self.refresh_offers()
                self.cart_review(list(lines), HARD_SPEND)
                if not self.verdicts.get(self.verdict_key(list(lines))) or self.steps_left() < 1 or self.finished or any(self.is_gone(ident) for ident in lines):
                    break
            go, said = self.process_gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.call_store(buy, args)
            self.trace.append({"kind": "self_checkout", "lines": lines, "reply": clip_values(reply, 200)})
            if reply is None:
                break
            self.checkouts += 1
            if not reply.get("finished"):
                self.refused_step = self.step
            self.take_result(buy, reply.get("observation"))
            replies.append(render(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        return replies

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return name == self.role("buy") and all(args.get(key) == value for key, value in fixed.items())

    def is_ending(self, name: str, args: dict[str, Any]) -> bool:
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return name == self.role("refuse") and all(args.get(key) == value for key, value in fixed.items())

    def tools_now(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        ends = self.ending_tools()
        keep: set[str] | None = None
        if left <= self.finish_reserve():
            keep = set(ends)
        elif left <= self.finish_reserve() + 3:
            keep = set(ends) | {self.role("cart"), self.role("cart_view")}
        elif left <= len(self.items) + 5 and self.checkouts == 0:
            keep = set(ends) | {self.role("cart"), self.role("cart_view"), self.role("offer")}
        tools = [tool for tool in self.tools if tool["function"]["name"] != self.role("ask")]
        if keep is not None and ends:
            tools = [tool for tool in tools if tool["function"]["name"] in keep] or tools
        return tools

    def plain_ending(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        out = dict(args)
        for key, spec in (schema.get("properties") or {}).items():
            if not isinstance(spec, dict) or "string" not in type_list(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = PLAIN_ENDING[: spec.get("maxLength", 200)]
        return out

    def fix_one_id(self, value: Any) -> tuple[Any, str]:
        if isinstance(value, str) and value.strip() in self.rows:
            return value.strip(), ""
        if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
            return value, ""
        low = value.strip().lower()
        hits = [ident for ident, row in self.rows.items() if any(key != self.id_key and isinstance(text, str) and text.strip().lower().startswith(low) for key, text in row.items())]
        if len(hits) == 1:
            return hits[0], ""
        if hits:
            return value, f"the text {value!r} fits several listings ({', '.join(hits[:6])}); give a single listing id"
        return value, f"{value!r} is no listing id; give the id shown in the result rows (field {self.id_key})"

    def fix_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.role(job) for job in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        free = free_text_fields(self.schemas.get(name) or {}, True)
        if len(free) != 1 or (name != self.role("cart") and (name, free[0]) not in self.id_params):
            return args, ""
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [self.fix_one_id(value) for value in args[key]] if isinstance(args[key], list) else [self.fix_one_id(args[key])]
            problem = next((issue for _, issue in fixed if issue), "")
            if problem:
                return args, problem
            out[key] = [value for value, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ""

    def track_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_used = True
        self.cart_changed_at = self.step
        fields = self.schemas.get(name, {}).get("properties") or {}
        removal = any(isinstance((fields.get(key) or {}).get("enum"), list) and len(fields[key]["enum"]) >= 2 and value in fields[key]["enum"][1:] for key, value in args.items())
        loose = next((whole_number(value) for value in args.values() if whole_number(value) is not None and not isinstance(value, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (fields.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((inner for inner in line.values() if isinstance(inner, str) and inner in self.rows), None)
                        count = next((whole_number(inner) for inner in line.values() if whole_number(inner) is not None and not isinstance(inner, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose))
        if whole:
            self.cart_units = {ident: count for ident, count in lines if count and count > 0}
            return
        sets = self.cart_sets_count(name)
        for ident, count in lines:
            have = self.cart_units.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_units[ident] = have
            else:
                self.cart_units.pop(ident, None)

    def merge_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = self.role("open"), self.role("compare"), self.one_listing_arg(self.role("open"))
        many = self.batch_arg()
        if not opener or not joiner or not target or not many or opener == joiner or self.batch_is_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = parse_object(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.sold_at:
                picks.append((index, ident))
        picks = picks[: self.batch_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        merged = {index for index, _ in picks}
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": dump_json({many: [ident for _, ident in picks]})}})
            elif index in merged:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.trace.append({"kind": "merged_opens", "ids": [ident for _, ident in picks]})
        return out

    def parsed_args(self, raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        return parse_object(raw) if (raw or "").strip() else {}

    def guard(self, name: str, args: dict[str, Any], calls: list[dict[str, Any]], ending: bool, mark: str, target: str | None) -> str:
        if target and name in (self.role("open"), self.role("offer")) and str(args.get(target)) in self.sold_at:
            return f"Skipped (no step spent): {args[target]} sold out at step {self.sold_at[str(args[target])]} for good. Fill its cart line with another fitting listing."
        if name == self.role("cart") and any((whole_number(value) or 0) < 0 for value in args.values() if not isinstance(value, str)):
            seen = dump_json(self.cart_snapshot) if self.cart_viewed_at >= self.cart_changed_at else ""
            absent = [value for value in args.values() if isinstance(value, str) and value in self.rows and value not in self.cart_units and value not in seen and value not in self.cart_reply]
            if absent:
                return f"Skipped (no step spent): {absent[0]} is not in the cart, so nothing can be taken off. The cart: " + self.cart_summary()
        if ending and self.is_checkout(name, args) and self.role("cart") and not self.cart_listing_ids() and not self.empty_checkout_held:
            self.empty_checkout_held = True
            picks = ", ".join(self.leads) if self.leads else "the listing that fits"
            return f"Skipped (no step spent): the cart is empty, so the store would refuse this checkout. Cart {picks} at the requested quantity first, then check out."
        if ending and len(calls) > 1:
            return "Skipped (no step spent): make the finishing call alone, once you have read the other results."
        if not ending and self.steps_left() <= self.finish_reserve() and (self.role("buy") or self.role("refuse")):
            return "Skipped (no step spent): the remaining steps are reserved for the checkout (with any approval the basket still needs) or for ending without a purchase."
        if not ending and name != self.role("cart") and mark in self.recent_calls:
            return "Skipped (no step spent): this exact call was just made and nothing has changed since; its result is above."
        readers = self.reader_tools()
        cached = self.read_cache.get(mark)
        if name in readers and cached and cached[0] == self.version:
            return f"Skipped (no step spent): nothing changed since the identical call at step {cached[1]}, which returned:\n" + cached[2][:3000]
        if target and name in readers and name != self.role("compare") and self.nothing_new(name, str(args.get(target))):
            return f"Skipped (no step spent): since the last change {args[target]} has already shown all this view offers. Its record: " + self.record_text(str(args[target]))
        return ""

    def reader_tools(self) -> set[str]:
        return {self.role(job) for job in ("search", "open", "offer", "compare")} - {None}

    def view_cart_now(self) -> str:
        if not (self.cart_used and self.cart_viewed_at < self.cart_changed_at and self.role("cart_view") and self.steps_left() > 3):
            return ""
        view_args = shaped({}, self.schemas.get(self.roles["cart_view"]))
        view = self.call_store(self.roles["cart_view"], view_args) if view_args is not None else None
        if view is None:
            return ""
        self.cart_snapshot = view.get("observation")
        self.cart_viewed_at = self.step
        change = self.take_result(self.roles["cart_view"], self.cart_snapshot)
        if not change:
            return ""
        self.pending_notes = []
        return "Held for now (no step used). The cart view reports a change: " + change + "\nThe cart: " + self.cart_summary() + "\nTake the change into account, repair the cart if needed, then check out."

    def checkout_hold(self) -> str:
        broad = self.broad_search_hold()
        if broad:
            return broad + "\nThe cart: " + self.cart_summary()
        reread = self.refresh_offers()
        changed = self.view_cart_now()
        if changed:
            return changed
        swap = self.sold_out_note()
        held = [ident for ident in self.cart_listing_ids() if ident in self.sold_at] if self.sold_holds < 2 else []
        if swap or held:
            self.sold_holds += 0 if swap else 1
            return "Held for now (no step used). " + (swap or f"{', '.join(held)} in the cart has sold out: replace that line first.") + "\nThe cart: " + self.cart_summary()
        chain, hold = self.follow_pointers()
        if hold:
            return "Held for now (no step used). " + chain + "\nThe cart: " + self.cart_summary()
        read = self.read_attributes(self.cart_listing_ids(), self.finish_reserve() + 3) + self.cheaper_reads()
        verdict = self.cart_review()
        if verdict:
            seen = ("For this check the code read the attributes of these listings (one step each):\n" + "\n".join(read) + "\n") if read else ""
            return "Held for now. " + chain + seen + verdict + "\nThe cart: " + self.cart_summary()
        if reread and self.seats.cost > SOFT_SPEND:
            return "Held for now (no step used). The code has just re-read the carted listings:\n" + reread + "\nCheck out again if each line still shows stock and a price within budget; otherwise replace that line first."
        return ""

    def decline_hold(self) -> str:
        if self.steps_left() <= 5:
            return ""
        review = self.decline_review()
        if review:
            return "Held for now (no step used). " + review
        self.swap_lead_hold()
        if self.leads and self.steps_left() > 4 and self.decline_holds == 0:
            self.decline_holds = 1
            self.trace.append({"kind": "decline_held", "leads": list(self.leads)})
            return ("Held for now (no step used). The review points at listings that seem to fit: " + ", ".join(self.leads) + ". Open them where needed, cart the best fit under the choice rule and check out. "
                    "Only if every one of them clearly breaks a requirement, say which in your text and send the ending call again.")
        if self.leads and self.steps_left() > 4 and self.decline_holds == 1:
            self.decline_holds = 2
            first = self.leads[0]
            self.cart_review([first], HARD_SPEND)
            if self.verdicts.get(self.verdict_key([first])) and first in self.leads:
                self.confirmed_id = first
                self.trace.append({"kind": "decline_held", "confirmed": first})
                return f"Held for now (no step used). An independent check of {first}'s records finds every requirement met. Cart it at the requested quantity and check out; end without buying only if its record clearly breaks a requirement."
        return ""

    def late_decline_hold(self) -> str:
        if self.steps_left() > 5 or self.late_hold_done:
            return ""
        lines = [ident for ident in self.fullest_cart if not self.is_gone(ident)]
        fresh = self.cart_is_fresh()
        cart = self.cart_listing_ids()
        missing = [ident for ident in lines if ident not in cart or (ident in self.taken_off and not fresh)]
        if len(lines) < len(self.items) or len(missing) + 1 > self.steps_left():
            return ""
        self.late_hold_done = True
        self.trace.append({"kind": "late_hold", "lines": lines, "missing": missing})
        redo = f"Return {', '.join(missing)} to the cart at the requested quantities and check out." if missing else "Check out the cart as it stands."
        return ("Held for now (no step used). An ending without a purchase scores nothing while the store holds a fitting listing, and being short of steps does not show it holds none. "
                + redo + " End without buying only if your searches showed that some requested item has no listing meeting its requirements; then send the ending call again.")

    def search_push(self) -> str:
        if self.leads or self.search_pushed or self.steps_left() <= self.step_cap // 3:
            return ""
        self.search_pushed = True
        return (f"Held for now (no step used). {self.steps_left()} steps remain, so try new kinds of search before ending: browse each requested item's category page by page with the token or offset the store gives, query single value words and other names of the property, and open listings that could fit. Only then end without buying.")

    def note_batch(self, many: str | None, args: dict[str, Any], observation: Any) -> str:
        if not (many and isinstance(args.get(many), list) and len(args[many]) > 1):
            return ""
        if not any(str(ident) in dump_json(observation) for ident in args[many]):
            self.batch_bad = min(self.batch_bad or 999, len(args[many]))
            return f"\n[code] The side-by-side view returned nothing for {len(args[many])} listings; send no more than {self.batch_room()} per call."
        self.batch_ok = max(self.batch_ok, len(args[many]))
        return ""

    def learn_views(self, name: str, observation: Any) -> None:
        if not self.one_listing_arg(name) or name in (self.role("cart"), self.role("buy"), self.role("refuse"), self.role("ask")):
            return
        listing = self.listing_id(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        self.shows_attributes[name] = self.shows_attributes.get(name, False) or bool(attribute_block(observation))
        busy = (self.role("cart"), self.role("buy"), self.role("refuse"), self.role("ask"), self.role("search"))
        if name == self.role("open") and not self.shows_attributes[name]:
            other = next((tool for tool in self.schemas if tool != name and self.one_listing_arg(tool) and self.shows_attributes.get(tool) is not False and tool not in busy), None)
            if other:
                self.roles["open"] = other
        if not self.shows_attributes[name] and name != self.role("open") and not self.role("offer"):
            self.roles["offer"] = name
        if name == self.offer_view():
            self.offer_read_at[listing] = self.step

    def after_cart_call(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.track_cart(name, args)
        self.cart_reply = dump_json(observation)
        self.recent_calls = []
        counts = [whole_number(value) for value in args.values() if not isinstance(value, str) and whole_number(value) is not None]
        for value in args.values():
            if counts and isinstance(value, str) and value in self.rows:
                if counts[0] <= 0 and not listed_in(observation, value):
                    self.taken_off.add(value)
                elif counts[0] > 0:
                    self.taken_off.discard(value)
        lines = [ident for ident in self.cart_listing_ids() if not self.is_gone(ident) and ident not in self.taken_off]
        if len(lines) >= len(self.fullest_cart):
            self.fullest_cart = lines

    def after_call(self, name: str, args: dict[str, Any], reply: dict[str, Any], ending: bool, mark: str, many: str | None, trimmed: str) -> str:
        observation = reply.get("observation")
        trimmed += self.note_batch(many, args, observation)
        self.recent_calls = (self.recent_calls + [mark])[-3:]
        self.id_params.update((name, key) for key, value in args.items() if (value if isinstance(value, list) else [value]) and all(isinstance(entry, str) and entry in self.rows for entry in (value if isinstance(value, list) else [value])))
        self.take_in(observation)
        readers = self.reader_tools()
        if name == self.role("compare"):
            rows = self.records_of(observation)
            if rows:
                self.shows_attributes[name] = self.shows_attributes.get(name, False) or any(attribute_block(row) for _, row in rows)
        if name in readers - {self.role("search")}:
            self.note_shown(name, observation)
        plain = {tool for job, tool in self.roles.items() if job != "search"}
        paged = self.read_rest(name, args, observation) if name not in plain and not ending else ""
        self.learn_views(name, observation)
        if name == self.role("cart"):
            self.after_cart_call(name, args, observation)
        if name == self.role("cart_view"):
            self.cart_snapshot = observation
            self.cart_viewed_at = self.step
        if self.is_checkout(name, args):
            self.checkouts += 1
            if not reply.get("finished"):
                self.refused_step = self.step
        text = render(observation) + paged + trimmed
        if name not in plain and not ending:
            text += self.cheapest_rows([observation] + self.self_paged)
        base = text
        proof = self.evidence_line(observation)
        if proof:
            text += "\n" + proof
        if name == self.role("cart") and any(whole_number(value) == 0 for value in args.values() if not isinstance(value, str)):
            if any(isinstance(value, str) and value in self.rows and listed_in(observation, value) for value in args.values()):
                text += "\n[code] That line is still in the cart after a count of 0: this tool adds counts to a line, so units come off with a negative count."
        self.note_says_sold = False
        lines = self.note_lines(observation)
        if lines:
            meaning = self.read_note(lines)
            if meaning:
                text += "\n[store note] " + meaning + ". Bring your picks in line with it before checking out."
        if self.is_checkout(name, args) and not reply.get("finished") and (self.note_says_sold or any(line in spoken_text(observation) for line in self.sold_phrases)):
            seen = self.locate_sold()
            if seen:
                text += "\n[carted lines re-read by the code after the refusal]\n" + seen
        swap = self.sold_out_note()
        if swap:
            text += "\n" + swap
        if self.pending_notes:
            text += "\n[store notes from the code's own calls] " + " | ".join(self.pending_notes)
            self.pending_notes = []
        if name in readers:
            self.read_cache[mark] = (self.version, self.step, base)
        return text + (" [episode finished]" if reply.get("finished") else "")

    def handle_one(self, call: dict[str, Any], calls: list[dict[str, Any]]) -> str:
        allowed = {tool["function"]["name"] for tool in self.tools_now()}
        if call.get("joined"):
            return "See the side-by-side result above: the code opened these listings together in a single step."
        fn = call.get("function") or {}
        name = fn.get("name")
        if self.finished:
            return "Skipped: the episode has ended."
        if name == self.role("ask"):
            return "Skipped (no step spent): messages to the shopper are the interviewer's job."
        if name not in self.schemas:
            return f"Skipped (no step spent): no such tool. Available: {', '.join(sorted(allowed))}."
        if name not in allowed:
            return "Skipped (no step spent): not enough steps remain for that; use one of: " + ", ".join(sorted(allowed)) + "."
        args = self.parsed_args(fn.get("arguments"))
        if args is None:
            return "Skipped (no step spent): the arguments must form a single JSON object."
        faults: list[str] = []
        args = conform(args, self.schemas[name], "args", faults)
        if faults:
            return "Skipped (no step spent): " + "; ".join(faults[:6])
        args, wrong = self.fix_ids(name, args)
        if wrong:
            return "Skipped (no step spent): " + wrong + "."
        trimmed = ""
        many = self.batch_arg() if name == self.role("compare") else None
        if many and isinstance(args.get(many), list) and len(args[many]) > self.batch_room() and self.batch_bad:
            room = self.batch_room()
            trimmed = f"\n[code] Only the first {room} listings went out, the most this store shows side by side. Still unshown: {', '.join(map(str, args[many][room:]))}."
            args = {**args, many: args[many][:room]}
        target = self.one_listing_arg(name)
        ending = self.is_checkout(name, args) or self.is_ending(name, args)
        mark = name + dump_json(args)
        refusal = self.guard(name, args, calls, ending, mark, target)
        if refusal:
            return refusal
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [ident for ident in args[many] if not self.nothing_new(name, str(ident))]
            if not keep:
                return ("Skipped (no step spent): since the last change these listings have shown everything this view offers. Records:\n"
                        + "\n".join(f"{ident}: {self.record_text(str(ident))}" for ident in args[many])[:4000])
            if len(keep) < len(args[many]):
                trimmed += f"\n[code] Dropped {', '.join(str(ident) for ident in args[many] if ident not in keep)} from the call: already shown in full and unchanged."
                args = {**args, many: keep}
                mark = name + dump_json(args)
        if ending and self.steps_left() > 2:
            hold = self.checkout_hold() if self.is_checkout(name, args) else self.decline_hold()
            if hold:
                return hold
        if self.is_ending(name, args):
            late = self.late_decline_hold() or self.search_push()
            if late:
                return late
            done = self.finish_reviewed()
            if done:
                return done + (" [episode finished]" if self.finished else "")
            args = self.plain_ending(name, args)
        if ending and self.steps_left() >= 2:
            go, said = self.process_gate("order" if self.is_checkout(name, args) else "end")
            if not go:
                return "Held for now. " + said + "\nThe cart: " + self.cart_summary()
            if said:
                trimmed = "\n[code] " + said + trimmed
        reply = self.call_store(name, args)
        if reply is None:
            return "The call did not go through (network or server trouble); it can be tried again."
        return self.after_call(name, args, reply, ending, mark, many, trimmed)

    def handle_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        calls = self.merge_opens(calls)
        return [(call["id"], self.handle_one(call, calls)) for call in calls]

    def close_out(self, again: bool = False) -> str:
        if self.finished or self.store.broke or self.steps_left() <= 0:
            return ""
        buy, refuse = self.role("buy"), self.role("refuse")
        fresh = self.cart_is_fresh()
        holding = bool(self.cart_listing_ids()) if fresh else bool(self.cart_units)
        if buy and (holding or (self.cart_used and not fresh and self.steps_left() >= 2)):
            args = shaped(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy))
            if args is not None:
                if self.steps_left() >= 2:
                    self.process_gate("order")
                if self.finished or self.steps_left() <= 0:
                    return ""
                reply = self.call_store(buy, args)
                self.trace.append({"kind": "close_buy", "reply": clip_values(reply, 200)})
                if self.finished or self.steps_left() <= 0:
                    return ""
                if reply is not None and again:
                    self.checkouts += 1
                    self.refused_step = self.step
                    self.take_result(buy, reply.get("observation"))
                    return "the store turned the checkout down: " + render(reply.get("observation"))
        if refuse:
            args = shaped(self.plain_ending(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse))
            if args is not None:
                if self.steps_left() >= 2:
                    self.process_gate("end")
                if self.finished or self.steps_left() <= 0:
                    return ""
                reply = self.call_store(refuse, args)
                self.trace.append({"kind": "close_decline", "reply": clip_values(reply, 200)})
                if reply is not None and not self.finished:
                    self.take_result(refuse, reply.get("observation"))
                    return "the store turned down the ending without a purchase: " + render(reply.get("observation"))
        return ""

    def shorten_history(self) -> None:
        if sum(len(dump_json(message.get("content"))) for message in self.chat) <= HISTORY_CHARS:
            return
        results = [index for index, message in enumerate(self.chat) if message.get("role") == "tool"]
        for index in results[: max(1, len(results) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = [ident for ident in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if ident in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result trimmed; it named these listings: {', '.join(ids)})"

    def opening_text(self) -> str:
        opening = self.view.get("opening")
        text = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            text += "\n\nWHAT THE STORE SHOWS FIRST:\n" + render(opening)
        return text + "\n\nBegin by searching each item's category and opening several listings in full, so that the interviewer learns which properties these listings carry.\n\n" + self.status_text()

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_brief()
        self.trace.append({"kind": "request_read", "brief": self.brief})
        if self.view.get("opening") is not None:
            self.take_in(self.view.get("opening"))
        self.chat = [{"role": "system", "content": WALKER_GUIDE}, {"role": "user", "content": self.opening_text()}]
        self.loop(WALK_TURNS, HARD_SPEND)
        if not self.finished:
            turned = self.close_out(again=True)
            if turned and not self.finished and not self.store.broke and not self.seats.no_credit and self.steps_left() > 1:
                self.chat.append({"role": "user", "content": "[code] When the code tried to finish, " + turned + "\nDeal with what the store says, then finish.\n" + self.status_text()})
                self.loop(self.walk_turns + 10, HARD_SPEND + 0.01)
            if not self.finished:
                self.close_out()
        self.trace.append({"kind": "done", "finished": self.finished, "steps": self.step, "cost": round(self.seats.cost, 5), "calls": self.seats.calls})
        return self.trace

    def turn_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for number, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            raw = fn.get("arguments")
            parsed = raw if isinstance(raw, dict) else (parse_object(raw) if isinstance(raw, str) and raw.strip() else {})
            calls.append({"id": str(call.get("id") or f"k{self.walk_turns}_{number}"), "type": "function",
                          "function": {"name": str(fn.get("name") or ""), "arguments": dump_json(parsed if isinstance(parsed, dict) else {})}})
        return calls

    def ask_before_commit(self, content: str, calls: list[dict[str, Any]]) -> bool:
        committing = {self.role("cart"), self.role("buy"), self.role("refuse")} - {None}
        if self.interviewed or not self.role("ask") or self.spare_messages() <= 0 or self.steps_left() <= 8:
            return False
        if not any(call["function"]["name"] in committing for call in calls):
            return False
        self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for call in calls:
            self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Skipped (no step spent): the shopper is questioned first."})
        summary = self.run_interview()
        self.chat.append({"role": "user", "content": (summary or "The interviewer found nothing to ask.") + "\n" + self.status_text()})
        return True

    def loop(self, turn_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.store.broke and self.steps_left() > 0 and self.walk_turns < turn_cap and self.seats.cost < spend_cap and self.clock.left() > WRAP_SECONDS:
            if self.time_to_ask():
                summary = self.run_interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status_text()})
                continue
            self.shorten_history()
            self.walk_turns += 1
            message = self.seats.think(self.chat, WALKER_SEATS, 3000, self.tools_now())
            if message is None:
                self.missed_turns += 1
                if self.seats.no_credit or self.seats.calls >= LLM_CALLS or self.missed_turns > 2:
                    break
                continue
            calls = self.turn_calls(message)
            content = message.get("content")
            if not isinstance(content, str):
                content = dump_json(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Every reply needs a tool call; make one to go on.\n" + self.status_text()})
                if idle >= 3:
                    break
                continue
            idle = 0
            if self.ask_before_commit(content, calls):
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "walk", "text": content[:300], "calls": [call["function"] for call in calls]})
            replies = self.handle_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status_text()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{"kind": "broke", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "broke", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                episode.close_out()
        except Exception as again:
            episode.trace.append({"kind": "broke", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
