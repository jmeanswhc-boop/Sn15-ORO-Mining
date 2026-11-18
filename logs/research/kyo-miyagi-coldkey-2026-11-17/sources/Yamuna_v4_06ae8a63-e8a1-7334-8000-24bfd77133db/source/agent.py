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

SEATS: dict[str, list[str]] = {
    "reader": [LIGHT],
    "reader_again": [HEAVY, LIGHT],
    "modes": [LIGHT, HEAVY],
    "walker": [LIGHT, SPARE],
    "planner": [HEAVY, LIGHT],
    "listener": [LIGHT, HEAVY],
    "listener_hard": [HEAVY, LIGHT],
    "notes": [LIGHT, SPARE],
    "judge": [HEAVY, LIGHT],
    "review": [HEAVY, LIGHT],
    "swap": [HEAVY, LIGHT],
    "trail": [LIGHT, HEAVY],
    "gate": [LIGHT, HEAVY],
    "gate_again": [HEAVY, LIGHT],
    "shape": [LIGHT, HEAVY],
}
OUTPUT_TOKENS: dict[str, int] = {
    "reader": 4000, "reader_again": 2500, "modes": 600, "walker": 3000, "planner": 1800, "listener": 900, "listener_hard": 1200,
    "notes": 1200, "judge": 1500, "review": 1600, "swap": 1200, "trail": 1800, "gate": 2000, "gate_again": 2000, "shape": 1500,
}

TIME_LIMIT_SECONDS = 300
WRAP_UP_SECONDS = 232
WAIT_STEPS = (4, 9, 18, 30)
WAIT_BUDGET_SECONDS = 45
WALK_TURNS = 44
EXTRA_TURNS = 10
LLM_CALLS_CAP = 64
SPEND_EASY = 0.055
SPEND_STOP = 0.08
RESULT_CHARS = 450
RESULT_LIMIT = 15000
CHAT_LIMIT = 150000
GATE_TRIES = 3
PAGE_BUDGET = 6
CHEAPER_READS = 3
DECLINE_OPENS = 6
JUDGE_HOLDS = 2
SEARCH_AFTER_REVEAL = 2
PLAN_PER_PRODUCT = 5

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


ROLE = "You work inside an online store on behalf of one shopper, and you can only act through the store's own tools."

FIT_RULES = """- A listing's property decides a requirement when the property's name is that requirement's property, whatever its spelling (spaces, underscores, camel case, a prefix).
  - A value meets the requirement when its words carry the wanted value: case, singular or plural and plain synonyms do not matter, one entry of a comma or slash list is enough, and a longer value that contains the wanted words still counts.
  - Different words are a different value: a code or abbreviation is not the word it stands for, and a two-word colour is not another two-word colour.
  - Convert units before comparing numbers; a limit includes its edge; a range has to sit wholly inside a limit.
- A property named for something else, even something close (the strap's colour is not the case's colour), says nothing about the requirement.
- When no property is named for the requirement, the wanted value written in another property, the title or the description counts; a figure of another kind (a weight for a capacity) or a bare number that does not say what it measures does not.
- When the category label itself states the value (an audience or a season in the label), the category meets that requirement.
- The category label has to match the requested product as a whole; a neighbouring category is a different product.
- A requirement to be unlike another listing's value is met unless the records show that same value.
- A sold-out listing never fits, and the order total (price times units over all lines) must stay inside the budget; equal is fine."""


WALKER_GUIDE = ROLE + """ Yamuna's helpers (an interviewer, a reviewer, a judge) work beside you; their notes arrive in the conversation.

# What earns the point
- The episode scores 1 only for an exactly right ending: an order holding every requested product at its requested units, each meeting every requirement in force at checkout, inside the budget.
- Ending without an order is right only in the rare case where the store really holds nothing that fits and the request says how to end then; it almost never earns the point, so when in doubt order the best listing you found.

# How this store behaves
- Each tool call uses one step and steps run out; thinking is free. Use the exact tool names, argument names and allowed values from each schema.
- Search tools can be narrow: few rows per page, a token or offset for the next page, titles matched by their first words. Short queries work best: the wanted value and the product noun. Compare the reported total with the rows seen and page on when the choice depends on seeing them all.
- Search rows are summaries. One view may show a listing's properties and another only its price, stock and description; the ledger names the view that shows properties once Yamuna knows it. Judge every requirement from the properties, so read them for every listing you might buy. Do not reopen a listing when nothing changed.
- You never write to the shopper: Yamuna's interviewer asks the questions and adds the answers to the ledger.
- When the request asks for steps along the way (an approval, a message, something to show or check), the ledger lists them. Steps due right before the order or the ending are done by Yamuna when you call checkout or the ending, so never wait for them; do the others with the store's tools.

# When a listing fits
""" + FIT_RULES + """
- A requirement may take its value from another listing ("the same X as the one sold by <shop> with ..."). Find that listing, open its page in full (a search row is not enough), follow any further listing that page sends you to, and take the value from the last page. When the wanted value is anything but that listing's value, only listings sharing that exact value are out.
- When the listing you are about to buy refers you to another page about itself, read that page first.
- When one product has to fit another (an accessory and its device), choose the two together from their own words (a list of what it fits, a model or series code both pages carry); if no fitting accessory exists for the device you chose, switch devices before giving up.
- Exclusions and closed lists are firm; wishes only break ties.

# Choosing
- Cheapest: the order needs the lowest-priced listing among all fitting ones. Collect the fitting set (page through it; rows come in no price order), check candidates from the cheapest up and take the first that fits fully. Use a maximum-price filter when the search has one.
- Most or least of a measure: compare every fitting listing on it; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: cart it and move on instead of hunting for a better one.
- Cart each product as soon as a fitting listing is read (after comparing, for cheapest, most or least): stores sell carted lines out, and an early cart leaves steps to replace one.
- Several different products need one cart line each, at its own units.

# Changes along the way
- The shopper or the store can change the budget, add or drop a requirement, move a price or sell a listing out. Such notes can sit in any tool's result, often in a refused checkout. The episode goes on: apply the change, fix the cart and check out again.
- Re-read a chosen listing shortly before checkout; price and stock can change silently.

# Finishing
- Leave exactly the chosen lines in the cart, look at the cart once, then call checkout on its own.
- Running short of steps never proves that nothing fits: check out the best cart you have.
- When you do end without an order, follow the request's wording, keep the remark short and name no listing.
- Every reply must call at least one tool; keep your text to what you know and your next move."""


READ_GUIDE = """# Reading a shopping request for an agent
You get the store's tools (name, description, parameters) and the shopper's request. Return ONE compact JSON object and nothing else, with these keys:
- "products": one entry per distinct product to buy, each {"name", "shelf", "units", "musts"}.
  - "name": the product in the request's words. "shelf": the store's category label for it when a tool lists labels (copied exactly), else "".
  - "units": units of that one listing (an integer; 1 when unstated; a pair or set sold as one listing is 1).
  - "musts": firm requirements as "property: constraint" strings with the exact value, limit, exclusion or closed list. Use the request's own word for the property ("for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch") and do not swap in a neighbouring property.
  - A must whose value comes from another listing is ONE string holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"); the words describing that other listing never become musts of this product.
  - Price, budget, stock and units never go into musts. The story (who uses it, where it goes, what went wrong before) is context unless it states a value or limit outright; words that are part of the product's name (a season, audience or use named with it) belong in "name".
  - Products named in one phrase joined by "and" with one quantity (often one store department) are ONE product; split only when each has its own quantity or requirements.
- "budget": the most the shopper pays in total, as a number in the request's unit, or null. "budget_unit": that unit in words, or null. "budget_per": "order" when the limit covers the whole order, "product" when it is per product, null without a limit.
- "stock_only": true unless the request clearly accepts unavailable listings.
- "pick": "cheapest" for the lowest price or total, "most" or "least" for the extreme of a measure, else "any" (remarks that money is tight are not a pick). "measure": the measure for most or least, else "".
- "messages": how many messages the shopper allows (integer) or null. "steps": how many steps or tool calls the request allows, or null.
- "pointers": the musts (copied exactly) whose value must first be read from another listing, else [].
- "rules": every instruction about how to work or end, each in at most 20 words.
- "wishes": things the shopper would merely like.
- "check_with_shopper": things the request explicitly says to check with the shopper before choosing, else [].
- "process": steps the request requires besides finding and buying (an approval, telling or showing the shopper something, reading something first, using a named tool), each {"what": at most 15 words, "stage": "choose" (before choosing), "order" (right before checkout) or "end" (right before ending without an order), "messages": messages to the shopper it takes, 0 when none}; [] when none. Requirements, budget, stock, the message and step limits and the ending when nothing fits are not steps.
- "when_nothing_fits": "decline" when the request says to end without buying (or to refuse, cancel, walk away) if nothing fits; "closest" when it says to order the nearest or best available option then; "" when it says nothing about that case.
- "no_names_on_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null otherwise.
- "tools": the exact tool for each job, or null when none does it:
  - "search" finds listings;
  - "details" shows one listing's properties, attributes or specifications (when two single-listing tools exist, the one whose description promises properties, attributes or specifications, never the one about the offer or description);
  - "page" shows one listing's offer (price, stock, description) when that is another tool than "details";
  - "many" shows several listings at once;
  - "message" writes to the shopper; "cart" changes the cart; "cart_view" shows it; "buy" checks out; "refuse" ends without an order.
- "buy_args" / "refuse_args": the fixed arguments that make "buy" check out or "refuse" end without an order (a mode argument's value), else {}."""


MODES_GUIDE = """One store tool both checks out and ends without an order, depending on its arguments. Return ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without an order}, naming only the arguments that pick the mode."""


PLAN_GUIDE = """# Choosing what to ask a shopper
Shoppers keep a few firm preferences they never wrote down, each on one property of one requested product. A question uncovers one only when it names that exact property, and the shopper answers only about the first property a question names, so each question spends one property.

You get the request, each product with what the request already settles, and per product the property keys seen on its opened listings, with how many listings carry each key, how many different values it takes there, and sample values. A hidden preference can also sit on a property no listed key names.

Rank, for each product separately:
- first, anything the request itself says to check with the shopper (under CHECK WITH THE SHOPPER);
- then hinted properties: a story sentence that points at a preference without its value (something the shopper cannot judge or leaves to you, someone whose taste decides, the looks or the setting, who will use or wear it, what goes inside). Name the one property that sentence is about; when a listed key of the product could settle the hint, use that key, and pick among colour, finish, pattern, style or gender only by what the hint and the listed keys say, never by habit;
- then listed keys: those whose values differ most across the product's listings and that most of them carry first; a key with one value on every listing, or seen on one or two listings only, comes last.
Leave out what the request settles for that product, a second name for a property you already chose, keys listing purchase variants (mixed sizes, colours, pack counts), identifiers, warranty, delivery and other logistics.
"property" is the plain name: for a listed key, all of its words in plain spelling ("lid colour" for lidColour, never shortened); only the name, no values, no brackets, no second property. Never give the same property to two products.
Return ONE JSON object and nothing else: {"ask": [{"product": product number, "property": "...", "hint": "up to 8 story words it comes from, or \\"\\""}]} with up to PER_PRODUCT entries per product, best first within each product."""


LISTEN_GUIDE = """# Reading one reply from a shopper
A shopping agent asked one question; read the reply. Return ONE JSON object and nothing else:
{"needs": [{"product": product number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"product": product number, "property": "a requirement the shopper now gives up"}], "no_need": true when the reply states no preference on what was asked (it does not mind, does not follow, asks to hear it again, or talks about something else without a value), "budget": a new total limit as a number, or null}
- A need is a firm preference stated with its exact value or limit and unit ("at least 30 cm", "oval", "not wicker"). Name it by the property the reply itself names, which may differ from the one asked, and copy the value word for word, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- A value given by pointing at another listing (like, unlike or near the one on some shop's product picked out by a few of its properties) is ONE need on the property the reply names, "points": true, its value the reply's own words from the comparison word to the end of that description, never shortened. The other listing's properties are not needs of this product.
- Folksy or roundabout wording that names a value still states it."""


NOTE_GUIDE = """# A note seen while shopping
The store showed this note during a shopping episode; it may come from the shopper or the store. Return ONE JSON object and nothing else:
{"kind": "change" when it changes the task (budget, requirements, a price, availability, a refused order), "error" when it only says a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit or null, "add": [{"product": product number, "need": "property: constraint"}], "drop": [{"product": product number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone or sold out (named or not), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one; report every change."""


JUDGE_GUIDE = """# Last look before checkout
A shopping agent is about to check out. Look for clear breaks only: things the records shown prove wrong. Reading rules:
""" + FIT_RULES + """
- Each requested product must be its own cart line at its requested units, and nothing else may be in the cart.
- A requirement that no record addresses is not a break; mention it under "unsure" instead.
- Under a cheapest, most or least pick, a listing among the others shown that clearly fits everything and beats the cart's choice on that rule is a break; name it under "better".
- The newest record of a listing wins.
Return ONE JSON object and nothing else: {"breaks": ["short concrete break, naming the record field that shows it"], "unsure": ["requirement no record settles"], "better": [listing ids that should replace a cart line, best first]}"""


REVIEW_GUIDE = """# Before ending without an order
A shopping agent is about to end without buying because it believes nothing fits. Check the records shown against the requirements, with these reading rules:
""" + FIT_RULES + """
- OPENED records show the listing's page; ROW records were only seen as search rows, so a property the row does not show is unknown for them.
Return ONE JSON object and nothing else: {"fits": {"product number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "closest": {"product number": the id of the listing of the right kind and inside the budget that misses the fewest requirements, or ""}, "open": [ROW ids of a requested kind inside the budget whose page could show a fit, most promising first, at most 6], "reason": "one sentence"}"""


SWAP_GUIDE = """# A cart line sold out
A listing in a shopping agent's cart just sold out. Find replacements for that line among the other records shown, reading them this way:
""" + FIT_RULES + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it}, at most 6, for listings that appear to meet every requirement of that line's product], "reason": "one sentence"}"""


TRAIL_GUIDE = """# Following a requirement to the page that holds its value
Some requirements take their value from another listing, and a listing's page can send the reader on to a further listing. The order counts only once every page along such a chain was opened in full. Follow each chain from the requirement's own words, link by link, using the records below (OPENED: that page was opened; ROW: only seen as a search row).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes it}], "done": true when the last link is an OPENED page stating the property's value, else false, "value": that value or ""}]}
List each chain's links in order and stop at the first link that is not OPENED or not found."""


GATE_GUIDE = """# Steps the request wants before STAGE
A shopping agent is about to STAGE. The request asks for the steps below along the way. From the numbered RECORD of what the agent did, decide whether each step due now is done for the cart as it stands. A step is done only when a RECORD line shows it and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions about product properties do not do a step.
If a step is still missing, give the one store call that does it next, with a tool from TOOLS and its arguments. A message to the shopper is one short plain sentence or question about that step alone, naming no property or value the step does not need. Give no call when every due step is done or when the messages or steps left cannot do it.
Return ONE JSON object and nothing else: {"steps": [{"step": "...", "done": true or false, "lines": [RECORD line numbers that show it]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"product": product number, "need": "property: constraint"} for each firm requirement such a reply states]}"""


SHAPE_GUIDE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown) or null, "next": the key holding the value that asks for the next page or null, "next_arg": the parameter that takes that value or null, "price": the key inside a row holding the listing's price or null, "stock": the key inside a row saying whether it can be bought now or null}"""


def as_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def shorten(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    if isinstance(value, list):
        return [shorten(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: shorten(v, limit, depth + 1) for k, v in value.items()}
    return value


def read_json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    for candidate in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
        return float(value)
    return None


def plain(key: Any) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", spaced).strip().lower()


def stems(text: Any) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in plain(text).split())


def same_property(wanted: frozenset[str], key: frozenset[str]) -> bool:
    if not wanted or not key:
        return False
    return wanted <= key or (len(key) >= 2 and key <= wanted)


def property_of(need: str) -> str:
    return plain(str(need).split(":", 1)[0])


def label_key(label: str) -> str:
    return " ".join(sorted(stems(label)))


def type_names(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def matches_type(value: Any, kind: str) -> bool:
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }
    return checks.get(kind, lambda v: True)(value)


def coerce(value: Any, kind: str) -> tuple[bool, Any]:
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return True, int(value)
    if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        number = float(value)
        return True, int(number) if kind == "integer" or number.is_integer() else number
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, str(value)
    if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return True, value.strip().lower() == "true"
    if kind == "array" and value is not None and not isinstance(value, list):
        return True, [value]
    return False, value


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = type_names(schema)
    if kinds and not any(matches_type(value, k) for k in kinds):
        for kind in kinds:
            done, value = coerce(value, kind)
            if done:
                break
        if not any(matches_type(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        if isinstance(value, str):
            same = [o for o in allowed if isinstance(o, str) and o.lower() == value.strip().lower()]
            value = same[0] if same else value
        if value not in allowed:
            faults.append(f"{where}: allowed values are {allowed}")
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            faults.append(f"{where} is below {low}")
        if isinstance(high, (int, float)) and value > high:
            faults.append(f"{where} is above {high}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [conform(v, schema["items"], f"{where}[{n}]", faults) for n, v in enumerate(value)]
    if isinstance(value, dict):
        fields = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            faults.extend(f"{where}.{k} is not an allowed argument" for k in value if k not in fields)
        faults.extend(f"{where}.{k} is required" for k in schema.get("required") or [] if k not in value)
        value = {k: conform(v, fields[k], f"{where}.{k}", faults) if k in fields else v for k, v in value.items()}
    return value


def fields_of(schema: dict[str, Any] | None) -> dict[str, Any]:
    found = (schema or {}).get("properties")
    return found if isinstance(found, dict) else {}


def free_text_fields(schema: dict[str, Any] | None) -> list[str]:
    return [k for k, v in fields_of(schema).items() if isinstance(v, dict) and "string" in type_names(v) and "enum" not in v]


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not 3 <= len(text) <= 64 or " " in text:
        return False
    return named or bool(re.search(r"\d", text))


def choose_id_key(rows: list[dict[str, Any]]) -> str | None:
    best: tuple[tuple[bool, float], str] | None = None
    for key in rows[0]:
        named = "id" in str(key).lower()
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(v, named) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (named, sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def carries_id(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        for value in node:
            if isinstance(value, dict) and ident in [str(x).strip() for x in value.values() if isinstance(x, str)]:
                return True
            if carries_id(value, ident, depth + 1):
                return True
        return False
    if isinstance(node, dict):
        return any(carries_id(v, ident, depth + 1) for v in node.values())
    return False


def property_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best: dict[str, Any] | None = None
    for value in node.values():
        if not isinstance(value, dict) or len(value) < 2:
            continue
        scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
            best = scalars
    return best


def spoken_text(observation: Any) -> str:
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
    return " ".join(found)[:800] if found else as_text(observation)[:400]


def one_property(label: Any) -> str:
    head = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(label or ""))[0]
    return head.strip().rstrip("?.").strip()


def query_words(text: str, limit: int = 6) -> str:
    words = re.findall(r"[\w\-\.']+", str(text))
    return " ".join(words[:limit])


class Line:
    def __init__(self, name: str, shelf: str, units: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.units = units
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.unwanted: list[str] = []

    def live_needs(self) -> list[str]:
        return [n for n in self.musts + self.told if n not in self.dropped]

    def settled_properties(self) -> set[str]:
        return {label_key(property_of(n)) for n in self.live_needs()}


class Brains:
    def __init__(self, trace: list[dict[str, Any]], started: float):
        self.log = trace
        self.started = started
        self.cost = 0.0
        self.calls = 0
        self.waited = 0.0
        self.no_credit = False

    def talk(self, messages: list[dict[str, Any]], seat: str, tools: list[dict[str, Any]] | None = None, models: list[str] | None = None) -> dict[str, Any] | None:
        chain = models or SEATS[seat]
        for place, model in enumerate(chain):
            if self.calls >= LLM_CALLS_CAP or self.no_credit:
                return None
            self.calls += 1
            attempt = 0
            while True:
                last = place == len(chain) - 1 and attempt > 0
                message, status = self.call_once(model, messages, OUTPUT_TOKENS[seat], tools, last)
                if message is not None:
                    return message
                if status == 402:
                    self.no_credit = True
                    return None
                passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                limit = 1 if status in (None, 408) else len(WAIT_STEPS)
                if not passing or attempt >= limit or self.waited + WAIT_STEPS[attempt] > WAIT_BUDGET_SECONDS:
                    break
                if time.monotonic() - self.started + WAIT_STEPS[attempt] > WRAP_UP_SECONDS:
                    break
                pause = WAIT_STEPS[attempt] * (0.8 + 0.4 * random.random())
                self.waited += WAIT_STEPS[attempt]
                attempt += 1
                time.sleep(pause)
        return None

    def call_once(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, plain_post: bool) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            for index in {0, len(sent) - 1}:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if plain_post:
            data = _llm.post("/inference/chat/completions", body)
            failure: dict[str, Any] = {}
        else:
            result = _llm.post_verbose("/inference/chat/completions", body)
            data, failure = result.data, result.error or {}
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.log.append({"kind": "llm_empty", "model": model})
            return None, 0
        status = failure.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = as_int(data["error"].get("code"))
            failure = {"body": data["error"].get("message")}
        self.log.append({"kind": "llm_fail", "model": model, "status": status, "body": str(failure.get("body") or "")[:200]})
        return None, status

    def ask(self, prompt: str, seat: str) -> dict[str, Any] | None:
        for model in SEATS[seat]:
            if self.calls >= LLM_CALLS_CAP or self.no_credit:
                return None
            message = self.talk([{"role": "user", "content": prompt}], seat, None, [model])
            parsed = read_json((message or {}).get("content"))
            if parsed is not None:
                return parsed
        return None


class Store:
    def __init__(self, binding: dict[str, Any], max_steps: int, finished: bool, trace: list[dict[str, Any]]):
        self.binding = binding
        self.max_steps = max_steps
        self.step = 0
        self.finished = finished
        self.broke = False
        self.log = trace

    def left(self) -> int:
        return self.max_steps - self.step

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
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
                return data
            failure = result.error or {}
            status = failure.get("status")
            text = str(failure.get("body") or "")
            self.log.append({"kind": "env_fail", "status": status, "body": text[:200]})
            if status == 409:
                low = text.lower()
                if any(w in low for w in ("finish", "quarantin", "finaliz")):
                    self.finished = True
                    return None
                expected = re.search(r"expected (\d+)", text)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif "reuse" in low or "different action" in low:
                    self.step += 1
                else:
                    return None
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                self.broke = self.broke or status == 402
                return None
        return None


class Episode:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        view = environment["policy_view"]
        self.started = time.monotonic()
        self.log: list[dict[str, Any]] = []
        self.request_text = str(view.get("query") or "")
        self.opening = view.get("opening")
        self.tool_specs = [t for t in view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.params_of = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tool_specs}
        steps = as_int(view.get("max_steps"))
        if steps is None and isinstance(view.get("max_steps"), float):
            steps = int(view["max_steps"])
        self.store = Store(dict(environment["binding"]), steps if steps and steps > 0 else 30, bool(view.get("finished")), self.log)
        self.brains = Brains(self.log, self.started)
        self.reading: dict[str, Any] = {}
        self.jobs: dict[str, str] = {}
        self.lines: list[Line] = []
        self.money_cap: float | None = None
        self.msg_cap = 0
        self.msg_used = 0
        self.asked_steps: list[dict[str, Any]] = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_passed: set[tuple[str, str]] = set()
        self.record_log: list[str] = []
        self.record_steps: list[int] = []
        self.interviewed = False
        self.asked_labels: set[str] = set()
        self.seen_rows: dict[str, dict[str, Any]] = {}
        self.sheets: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.page_opened: set[str] = set()
        self.row_id_field: str | None = None
        self.shows_props: dict[str, bool] = {}
        self.notes_heard: set[str] = set()
        self.change_log: list[str] = []
        self.side_notes: list[str] = []
        self.epoch = 0
        self.basket: dict[str, int] = {}
        self.basket_changed_at = -1
        self.basket_seen_at = -1
        self.basket_view: Any = None
        self.basket_touched = False
        self.basket_reply = ""
        self.best_cart: list[str] = []
        self.taken_off: set[str] = set()
        self.judged: dict[str, bool] = {}
        self.judge_holds = 0
        self.reviewed = False
        self.leads: dict[str, list[str]] = {}
        self.closest: dict[str, str] = {}
        self.decline_holds = 0
        self.decline_opened = False
        self.searched_more = False
        self.late_kept = False
        self.self_ordered = False
        self.offer_checked_at: dict[str, int] = {}
        self.sold_marks: dict[str, int] = {}
        self.off_sale: set[str] = set()
        self.sold_phrases: set[str] = set()
        self.note_sold = False
        self.sold_holds = 0
        self.turned_down_at = -1
        self.checkouts = 0
        self.last_calls: list[str] = []
        self.dialogue: list[dict[str, Any]] = []
        self.pointers: list[tuple[int, str]] = []
        self.pointer_found: dict[str, str] = {}
        self.chain_rounds = 0
        self.chain_holds = 0
        self.many_fine = 0
        self.many_fail = 0
        self.fields_by_tool: dict[str, set[str]] = {}
        self.fields_seen: dict[str, tuple[int, set[str]]] = {}
        self.read_memo: dict[str, tuple[int, int, str]] = {}
        self.extra_pages = 0
        self.result_shapes: dict[str, dict[str, Any] | None] = {}
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.cheaper_marks_done: set[str] = set()
        self.extra_results: list[Any] = []
        self.id_params: set[tuple[str, str]] = set()
        self.basket_results: dict[str, str] = {}
        self.silent_turns = 0
        self.walk_turns = 0

    def steps_left(self) -> int:
        return self.store.left()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def late(self) -> bool:
        return self.elapsed() > WRAP_UP_SECONDS

    def over(self) -> bool:
        return self.store.finished or self.store.broke

    def spare_messages(self) -> int:
        return self.msg_cap - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def basket_mark(self) -> str:
        return as_text(sorted(self.basket.items())) + f"#{self.epoch}"

    def pick(self) -> str:
        return str(self.reading.get("pick") or "any")

    def store_call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        data = self.store.call(name, args)
        if data is not None and name in self.page_views():
            seen = data.get("observation")
            if not (isinstance(seen, dict) and seen.get("error")):
                for value in args.values():
                    self.page_opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
        return data

    # reading the request
    def read_the_request(self) -> None:
        catalogue = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {as_text(t['function'].get('parameters') or {})}" for t in self.tool_specs)
        prompt = READ_GUIDE + "\n\nTOOLS:\n" + catalogue + "\n\nREQUEST:\n" + self.request_text
        if self.opening is not None:
            prompt += "\n\nWHAT THE STORE SHOWED FIRST:\n" + as_text(shorten(self.opening, 1500))
        reading = self.brains.ask(prompt, "reader") or {}
        tools = reading.get("tools") if isinstance(reading.get("tools"), dict) else {}
        if not self.brains.no_credit and (not as_list(reading.get("products")) or not tools.get("buy")):
            reading = self.brains.ask(prompt, "reader_again") or reading
        self.reading = reading
        tools = reading.get("tools") if isinstance(reading.get("tools"), dict) else {}
        self.jobs = {job: tool for job, tool in tools.items() if isinstance(tool, str) and tool in self.params_of}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(reading.get(key), dict):
                reading[key] = {}
        self.split_end_modes()
        for entry in as_list(reading.get("products")):
            if isinstance(entry, dict):
                units = as_int(entry.get("units"))
                self.lines.append(Line(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), units if units and units > 0 else 1,
                                       [str(m) for m in as_list(entry.get("musts")) if m]))
        if not self.lines:
            self.lines = [Line("the requested product", "", 1, [])]
        for pointer in as_list(reading.get("pointers")):
            self.note_pointer(str(pointer).strip())
        budget = as_number(reading.get("budget"))
        self.money_cap = budget if budget and budget > 0 else None
        cap = as_int(reading.get("messages"))
        if cap is not None and cap >= 0:
            self.msg_cap = cap
        elif self.jobs.get("message"):
            self.msg_cap = 3
        steps = as_int(reading.get("steps"))
        if steps is not None and 10 <= steps < self.store.max_steps:
            self.store.max_steps = steps
        for entry in as_list(reading.get("process")):
            if isinstance(entry, dict) and str(entry.get("what") or "").strip():
                stage = str(entry.get("stage") or "").strip().lower()
                self.asked_steps.append({"what": str(entry["what"]).strip(), "stage": stage if stage in ("choose", "order", "end") else "order",
                                     "messages": max(0, as_int(entry.get("messages")) or 0)})
        if self.jobs.get("message"):
            talk = [p["messages"] for p in self.asked_steps if p["stage"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_cap) if talk else 0

    def split_end_modes(self) -> None:
        buy, refuse = self.jobs.get("buy"), self.jobs.get("refuse")
        fixed_buy, fixed_refuse = self.reading["buy_args"], self.reading["refuse_args"]
        if not buy or buy != refuse or (fixed_buy and fixed_refuse and fixed_buy != fixed_refuse):
            return
        got = self.brains.ask(MODES_GUIDE + "\n\nTOOL " + buy + ": " + as_text(self.params_of.get(buy)), "modes") or {}
        if isinstance(got.get("buy_args"), dict) and isinstance(got.get("refuse_args"), dict) and got["buy_args"] != got["refuse_args"]:
            self.reading["buy_args"], self.reading["refuse_args"] = got["buy_args"], got["refuse_args"]

    def note_pointer(self, text: str, number: int | None = None) -> None:
        if not text:
            return
        if number is None:
            if not any(text in line.musts for line in self.lines):
                wanted = property_of(text)
                text = next((m for line in self.lines for m in line.musts if property_of(m) == wanted), text)
            number = next((n for n, line in enumerate(self.lines, 1) if text in line.musts), 1)
        if (number, text) not in self.pointers:
            self.pointers.append((number, text))

    def all_needs(self) -> list[tuple[int, str]]:
        return [(n, need) for n, line in enumerate(self.lines, 1) for need in line.live_needs()]

    def product_line(self, value: Any) -> Line:
        return self.lines[min(max(1, as_int(value) or 1), len(self.lines)) - 1]

    def board(self) -> str:
        parts: list[str] = []
        several = len(self.lines) > 1
        for number, line in enumerate(self.lines, 1):
            needs = [n + (f" (value from the page it points at: {self.pointer_found[n]})" if n in self.pointer_found else "") for n in line.live_needs()]
            head = (f"product {number}: " if several else "product: ") + line.name + (f" (store label: {line.shelf})" if line.shelf else "") + f", {line.units} unit(s)"
            text = head + "; requirements: " + ("; ".join(needs) or "none beyond the kind of product")
            if line.unwanted:
                text += "; shopper does not mind: " + ", ".join(line.unwanted)
            parts.append(text)
        reading = self.reading
        if self.money_cap is not None:
            scope = "for each product" if reading.get("budget_per") == "product" else "in total"
            parts.append(" ".join(f"budget: at most {self.money_cap:g} {reading.get('budget_unit') or ''} {scope}".split()))
        if self.pick() == "cheapest":
            parts.append("pick: the cheapest listing that fits")
        elif self.pick() in ("most", "least"):
            parts.append(" ".join(f"pick: the {self.pick()} {reading.get('measure') or ''} among fitting listings".split()))
        else:
            parts.append("pick: whichever listing fits every requirement")
        if reading.get("stock_only", True) is not False:
            parts.append("in stock only")
        parts += [f"wish (tie-breaker only): {w}" for w in as_list(reading.get("wishes"))]
        parts += [f"rule from the request: {r}" for r in as_list(reading.get("rules"))]
        ending = reading.get("when_nothing_fits")
        if ending == "decline":
            parts.append("if nothing fits: the request says to end without an order")
        elif ending == "closest":
            parts.append("if nothing fits: the request says to order the closest option")
        if reading.get("no_names_on_decline") is True:
            parts.append("when ending without an order: name or suggest no listing")
        for entry in self.asked_steps:
            when = {"choose": "before choosing", "order": "at the order (Yamuna does it when you check out)", "end": "at an ending without an order (Yamuna does it then)"}[entry["stage"]]
            parts.append(f"step the request asks for, {when}: {entry['what']}")
        reader = self.sheet_view()
        if reader and (self.shows_props.get(reader) is True or self.jobs.get("page")):
            bare = sorted(t for t in self.page_views() | {self.jobs.get("page") or ""} if t and self.shows_props.get(t) is False)
            pages = sorted(t for t in self.page_views() if t != reader)
            text = f"listing properties come from {reader}"
            if bare:
                text += f"; {', '.join(bare)} show none"
            if pages:
                text += f"; a listing's page (for page chains and page checks) is {' or '.join(pages)}"
            parts.append(text)
        return "\n".join("- " + p for p in parts)

    def status_line(self) -> str:
        left = self.steps_left()
        text = f"[Yamuna] step {self.store.step} of {self.store.max_steps} used, {left} left."
        if left <= self.finish_room() + 1 or self.late():
            text += " LAST STEPS: check out now when the cart holds a line for every product; running out of steps never shows that nothing fits."
        elif left <= 6 or (left <= len(self.lines) + 5 and self.checkouts == 0):
            text += " Few steps left: set the cart to your chosen lines and check out now."
        elif self.checkouts == 0 and left <= 12:
            text += " Steps are getting low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.checkouts == 0 and not self.basket_touched and self.store.step >= self.store.max_steps // 2:
            text += " Half the steps are gone and the cart is empty: cart each product's fitting listing now."
        live = {need for _, need in self.all_needs()}
        open_pointers = [need for _, need in self.pointers if need in live and need not in self.pointer_found]
        if open_pointers and self.checkouts == 0:
            text += "\n[resolve first: open the listing each of these points at and every listing its page refers to; the last page gives the value] " + " | ".join(open_pointers[:3])
        if self.leads:
            text += "\n[listings the review thinks fit] " + "; ".join(f"product {k}: {', '.join(v)}" for k, v in self.leads.items())
        if self.change_log:
            text += "\n[changes so far] " + " | ".join(self.change_log[-4:])
        return text + "\n[ledger]\n" + self.board()

    def finish_room(self) -> int:
        talk = any(p["stage"] == "order" and p["messages"] > 0 for p in self.asked_steps)
        return 2 if talk and self.msg_used < self.msg_cap and self.gate_open("order") else 1

    def gate_open(self, stage: str) -> bool:
        return any(p["stage"] == stage for p in self.asked_steps) and (stage, self.basket_mark()) not in self.gate_passed

    # tools by job
    def one_id_param(self, tool: str | None) -> str | None:
        schema = self.params_of.get(tool or "") or {}
        texts = free_text_fields(schema)
        required = schema.get("required") or []
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in required):
            return None
        return target

    def many_param(self) -> str | None:
        schema = self.params_of.get(self.jobs.get("many") or "") or {}
        target = next((k for k, v in fields_of(schema).items() if isinstance(v, dict) and "array" in type_names(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def page_views(self) -> set[str]:
        page = self.jobs.get("page") or self.jobs.get("details")
        return {t for t in (page, self.jobs.get("many")) if t}

    def offer_view(self) -> str | None:
        return self.jobs.get("page") or self.jobs.get("details")

    def sheet_view(self) -> str | None:
        known = next((t for t, ok in self.shows_props.items() if ok and self.one_id_param(t)), None)
        if known:
            return known
        tool = self.jobs.get("details")
        return tool if self.shows_props.get(tool or "") is not False and self.one_id_param(tool) else None

    def many_is_bare(self) -> bool:
        seen = self.shows_props.get(self.jobs.get("many") or "")
        return seen is False or (seen is None and (self.shows_props.get(self.sheet_view() or "") is True or bool(self.jobs.get("page"))))

    def many_cap(self) -> int:
        spec = fields_of(self.params_of.get(self.jobs.get("many") or "")).get(self.many_param() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.many_fail:
            return max(2, min(self.many_fail - 1, max(self.many_fine, 2)))
        return max(4, self.many_fine)

    def learn_view_role(self, name: str, observation: Any) -> None:
        if not self.one_id_param(name) or name in {self.jobs.get(j) for j in ("cart", "buy", "refuse", "message", "search")}:
            return
        listing = self.listing_id(observation) if isinstance(observation, dict) else None
        if not listing:
            return
        self.shows_props[name] = self.shows_props.get(name, False) or bool(property_block(observation))
        if name == self.jobs.get("details") and not self.shows_props[name]:
            other = next((t for t in self.params_of if t != name and self.one_id_param(t) and self.shows_props.get(t) is not False
                          and t not in {self.jobs.get(j) for j in ("cart", "buy", "refuse", "message", "search", "cart_view")}), None)
            if other:
                self.log.append({"kind": "roles_swapped", "details": other, "page": name})
                self.jobs["details"] = other
                self.jobs.setdefault("page", name)
        if not self.shows_props[name] and name != self.jobs.get("details") and not self.jobs.get("page"):
            self.jobs["page"] = name
        if name == self.offer_view():
            self.offer_checked_at[listing] = self.store.step

    # listings
    def listing_id(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.row_id_field and looks_like_id(node.get(self.row_id_field), True):
            return node[self.row_id_field].strip()
        if not (in_list or depth == 0 or any(isinstance(v, dict) for v in node.values())):
            return None
        found = choose_id_key([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.seen_rows):
            return node[found].strip()
        return None

    def spot_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.row_id_field and all(looks_like_id(r.get(self.row_id_field), True) for r in rows):
            return True
        found = choose_id_key(rows)
        if found:
            self.row_id_field = found
        return bool(found)

    def soak(self, observation: Any) -> None:
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, list):
                listed = self.spot_rows(node)
                for value in node:
                    walk(value, depth + 1, listed)
                return
            if not isinstance(node, dict):
                return
            ident = self.listing_id(node, in_list, depth)
            if not ident:
                for value in node.values():
                    walk(value, depth + 1)
                return
            if ident not in self.seen_rows:
                self.order.append(ident)
                self.seen_rows[ident] = {}
            old = self.seen_rows[ident]
            if any(old.get(k) is True and v is False for k, v in node.items()):
                self.off_sale.add(ident)
            elif any(old.get(k) is False and v is True for k, v in node.items()):
                self.off_sale.discard(ident)
            old.update(node)
            if property_block(node):
                self.sheets[ident] = dict(old)
            elif ident in self.sheets:
                self.sheets[ident].update(node)

        walk(observation, 0)

    def listings_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        def from_list(values: list[Any]) -> list[tuple[str, dict[str, Any]]]:
            return [(i, r) for r in values if isinstance(r, dict) for i in [self.listing_id(r, True)] if i]

        if isinstance(observation, list):
            return from_list(observation)
        if not isinstance(observation, dict):
            return []
        own = self.listing_id(observation)
        if own:
            return [(own, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found += from_list(value)
            elif isinstance(value, dict):
                for inner in value.values():
                    if isinstance(inner, list):
                        found += from_list(inner)
        return found

    def listed_labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            found.update(o.strip().lower() for o in schema.get("enum") or [] if isinstance(o, str))
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)

        for schema in self.params_of.values():
            walk(schema, 0)
        return found

    def kind_field(self) -> str | None:
        values: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.seen_rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    values.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.seen_rows))
        labels = self.listed_labels()
        ranked = []
        for key, seen in values.items():
            if present[key] < 0.8 * total or len(seen) > 40:
                continue
            listed = sum(1 for v in seen if v.strip().lower() in labels) / len(seen)
            ranked.append(((-listed, len(seen), -present[key], sum(len(v) for v in seen) / len(seen)), key))
        return min(ranked)[1] if ranked else None

    def kind(self, ident: str) -> str:
        key = self.kind_field()
        return str((self.seen_rows.get(ident) or {}).get(key) or "").strip().lower() if key else ""

    def product_for_kind(self, label: str) -> int | None:
        if not label:
            return None
        words = plain(label)
        mine = {w for w in stems(label) if len(w) >= 3}
        scored = []
        for number, line in enumerate(self.lines, 1):
            if words in {plain(line.shelf), plain(line.name)} - {""}:
                return number
            theirs = {w for w in stems(line.shelf) | stems(line.name) if len(w) >= 3}
            scored.append((len(mine & theirs), number))
        scored.sort(reverse=True)
        if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        return scored[0][1]

    def groups(self) -> dict[int, list[str]]:
        found: dict[int, list[str]] = {}
        for ident in self.order:
            number = self.product_for_kind(self.kind(ident)) if self.kind_field() else (1 if len(self.lines) == 1 else None)
            if number:
                found.setdefault(number, []).append(ident)
        return found

    def key_menu(self) -> dict[int, list[tuple[str, int, int, list[str]]]]:
        menu: dict[int, list[tuple[str, int, int, list[str]]]] = {}
        for number, ids in self.groups().items():
            read = [i for i in ids if i in self.sheets]
            slots: dict[str, dict[str, str]] = {}
            for ident in read:
                done: set[str] = set()
                for key, value in (property_block(self.sheets[ident]) or {}).items():
                    name = plain(key)
                    if name and name not in done:
                        done.add(name)
                        slots.setdefault(name, {})[ident] = str(value).strip().lower()
            kept: list[tuple[str, dict[str, str]]] = []
            for name, per in slots.items():
                values = list(per.values())
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                if any(name.startswith(other + " ") and per == spec for other, spec in kept):
                    continue
                kept.append((name, per))
            kept.sort(key=lambda kv: (-min(len(set(kv[1].values())), 4) * len(kv[1]), -len(kv[1])))
            rows = []
            for name, per in kept[:14]:
                distinct = list(dict.fromkeys(per.values()))
                rows.append((name, len(per), len(distinct), [v[:40] for v in distinct[:4]]))
            if rows:
                menu[number] = rows
        return menu

    def price(self, ident: str) -> float | None:
        value = (self.seen_rows.get(ident) or {}).get(self.price_field) if self.price_field else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        return as_number(value)

    def is_gone(self, ident: str) -> bool:
        return ident in self.off_sale or ident in self.sold_marks

    def can_buy(self, ident: str) -> bool:
        if self.is_gone(ident):
            return False
        value = (self.seen_rows.get(ident) or {}).get(self.stock_field) if self.stock_field else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def affordable(self, ident: str, number: int) -> bool:
        price = self.price(ident)
        if price is None or self.money_cap is None:
            return True
        return price * self.lines[number - 1].units <= self.money_cap

    def mark_viewed(self, name: str, observation: Any) -> None:
        for ident, record in self.listings_in(observation):
            keys = set(record)
            self.fields_by_tool.setdefault(name, set()).update(keys)
            version, have = self.fields_seen.get(ident, (-1, set()))
            self.fields_seen[ident] = (self.epoch, (have if version == self.epoch else set()) | keys)

    def nothing_new(self, name: str, ident: str) -> bool:
        version, have = self.fields_seen.get(ident, (-1, set()))
        keys = self.fields_by_tool.get(name)
        return bool(keys) and version == self.epoch and keys <= have and ident not in self.sold_marks

    def record_of(self, ident: str, limit: int = 400) -> str:
        record = self.sheets.get(ident) or self.seen_rows.get(ident)
        return as_text(shorten(record, limit))[:3000] if record else "(no record seen)"

    def render(self, observation: Any) -> str:
        text = as_text(shorten(observation, RESULT_CHARS))
        return text[:RESULT_LIMIT] + "...(cut)" if len(text) > RESULT_LIMIT else text

    # result shapes and paging
    def result_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.result_shapes:
            return self.result_shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.result_shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = SHAPE_GUIDE + "\n\nTOOL: " + name + " | parameters: " + as_text(self.params_of.get(name) or {}) + "\n\nRESULT (first rows only):\n" + as_text(shorten(sample, 120))[:4000]
        got = self.brains.ask(prompt, "shape") or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        params = fields_of(self.params_of.get(name))

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), params), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if shape["count"] and as_int(observation[shape["count"]]) is None:
            shape["count"] = None
        self.result_shapes[name] = shape
        self.price_field = self.price_field or shape["price"]
        self.stock_field = self.stock_field or shape["stock"]
        self.log.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def page_plan(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.result_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), as_int(observation.get(shape["count"])), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or count is None:
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def price_sorted_rows(self, observation: Any) -> bool:
        prices = [self.price(i) for i, _ in self.listings_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def read_more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.extra_results = []
        plan = self.page_plan(name, observation) if self.pick() in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown or (self.pick() == "cheapest" and self.price_sorted_rows(observation)):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.extra_pages + pages > PAGE_BUDGET or self.steps_left() - self.finish_room() - pages < 12:
            return (f"\n[Yamuna] {count} listings matched and {shown} are shown, in no price order. The pick weighs every fitting listing: "
                    f"narrow the search with the requirement's value or page on with {param}={as_text(value)}.")
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = conform({**args, param: value}, self.params_of.get(name), "args", faults)
            if faults:
                break
            reply = self.store_call(name, call)
            if reply is None or self.store.finished:
                break
            self.extra_pages += 1
            seen = reply.get("observation")
            self.soak_and_read(name, seen)
            self.extra_results.append(seen)
            shown_text = self.render(seen)
            self.read_memo[name + as_text(call)] = (self.epoch, self.store.step, shown_text)
            texts.append(f"{param}={as_text(value)}: {shown_text}")
            after = self.page_plan(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.log.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Yamuna read the rest of this search, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_rows(self, observations: list[Any]) -> str:
        if self.pick() != "cheapest" or not self.price_field:
            return ""
        found = {i for seen in observations for i, _ in self.listings_in(seen)}
        priced = sorted((self.price(i), i) for i in found if self.can_buy(i) and self.price(i) is not None)
        if len(priced) < 2:
            return ""
        return "\n[Yamuna] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in priced[:6])

    # notes inside results
    def stray_text(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.listing_id(observation):
            for value in observation.values():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines += [v for v in value if len(v.split()) >= 5]
                elif isinstance(value, dict) and not self.listing_id(value, False, 1):
                    lines += [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6]
        return [line for line in lines if line not in self.notes_heard]

    def read_store_note(self, lines: list[str]) -> str:
        self.notes_heard.update(lines)
        got = self.brains.ask(NOTE_GUIDE + "\n\nPRODUCTS AND REQUIREMENTS NOW:\n" + self.board() + "\n\nTHE NOTE:\n" + "\n".join(lines), "notes") or {}
        gone = [str(g) for g in as_list(got.get("gone")) if isinstance(g, (str, int))]
        self.off_sale.update(g for g in gone if g in self.seen_rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            self.sold_phrases.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed: list[str] = []
        budget = as_number(got.get("budget"))
        if budget and budget > 0 and budget != self.money_cap:
            self.money_cap = budget
            changed.append(f"budget now {budget:g}")
        for entry in as_list(got.get("drop")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                line = self.product_line(entry.get("product"))
                target = str(entry["need"]).strip()
                live = line.live_needs()
                hit = [m for m in live if m == target] or [m for m in live if property_of(m) == property_of(target)][:1]
                line.dropped.extend(hit or [target])
                changed.append(f"dropped {target}")
        for entry in as_list(got.get("add")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                line = self.product_line(entry.get("product"))
                need = str(entry["need"]).strip()
                if need in line.dropped:
                    line.dropped.remove(need)
                if need not in line.musts + line.told:
                    line.told.append(need)
                changed.append(f"added {need}")
        if gone:
            changed.append("no longer sold: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.epoch += 1
            self.last_calls = []
            self.change_log.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def soak_and_read(self, name: str, observation: Any) -> str:
        self.soak(observation)
        lines = self.stray_text(observation)
        if not lines:
            return ""
        meaning = self.read_store_note(lines)
        if meaning:
            self.side_notes.append(f"(seen on {name}) {meaning}")
        return meaning

    def key_hints(self, observation: Any) -> str:
        shown: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.listing_id(node, False, depth)
                if ident and property_block(node):
                    if ident not in shown:
                        shown.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        needs = self.all_needs()
        if not shown or not needs:
            return ""
        several = len(self.lines) > 1
        lines = []
        for ident in shown[:12]:
            block = property_block(self.sheets.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if same_property(stems(prop), stems(k))]
                if hits:
                    parts.append((f"(product {number}) " if several else "") + f"{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: properties whose names carry a requirement's property; one about another part decides nothing, and another name may still hold the same property] " + " | ".join(lines)

    # reading listings in code
    def read_props(self, ids: list[str], keep: int) -> list[str]:
        tool = self.sheet_view()
        target = self.one_id_param(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.seen_rows and i not in self.sheets and not self.is_gone(i)]:
            if self.steps_left() <= keep or self.over() or self.shows_props.get(tool) is False or self.late():
                break
            faults: list[str] = []
            args = conform({target: ident}, self.params_of.get(tool), "args", faults)
            if faults:
                break
            reply = self.store_call(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.soak_and_read(tool, seen)
            self.mark_viewed(tool, seen)
            if isinstance(seen, dict) and self.listing_id(seen):
                self.shows_props[tool] = self.shows_props.get(tool, False) or bool(property_block(seen))
            texts.append(f"{ident}: " + self.render(seen)[:1500])
        if texts:
            self.log.append({"kind": "read_props", "n": len(texts)})
        return texts

    def read_many(self, ids: list[str], keep: int) -> bool:
        tool, target = self.jobs.get("many"), self.many_param()
        if not tool or not target or self.many_is_bare() or len(ids) < 2 or self.steps_left() <= keep:
            return False
        faults: list[str] = []
        args = conform({target: ids[: self.many_cap()]}, self.params_of.get(tool), "args", faults)
        if faults:
            return False
        before = len(self.sheets)
        reply = self.store_call(tool, args)
        if reply is None:
            return False
        seen = reply.get("observation")
        self.soak_and_read(tool, seen)
        self.mark_viewed(tool, seen)
        rows = self.listings_in(seen)
        if rows:
            self.shows_props[tool] = self.shows_props.get(tool, False) or any(property_block(r) for _, r in rows)
        self.log.append({"kind": "read_many", "ids": ids, "opened": len(self.sheets) - before})
        return len(self.sheets) > before

    def unread_rows(self, number: int | None, limit: int) -> list[str]:
        pool: list[str] = []
        groups = self.groups()
        numbers = [number] if number else list(range(1, len(self.lines) + 1))
        for n in numbers:
            ids = groups.get(n) or ([] if groups else list(self.order))
            fresh = [i for i in ids if i not in self.sheets and self.can_buy(i) and self.affordable(i, n)]
            if self.pick() == "cheapest":
                fresh.sort(key=lambda i: self.price(i) if self.price(i) is not None else float("inf"))
            else:
                fresh.reverse()
            pool += [i for i in fresh if i not in pool]
        return pool[:limit]

    def widen(self) -> None:
        if self.kind_field() and len(self.lines) > 1:
            for number in range(1, len(self.lines) + 1):
                if number not in self.groups() and self.steps_left() > 20 and not self.late():
                    self.code_search(query_words(self.lines[number - 1].shelf or self.lines[number - 1].name, 5))
        groups = self.groups()
        lanes = []
        for number in range(1, len(self.lines) + 1):
            ids = groups.get(number) or ([] if groups else list(self.order))
            have = sum(1 for i in ids if i in self.sheets)
            fresh = [i for i in ids if i not in self.sheets and self.can_buy(i) and self.affordable(i, number)]
            if self.pick() == "cheapest":
                fresh.sort(key=lambda i: self.price(i) if self.price(i) is not None else float("inf"))
            lanes.append(fresh[: max(0, min(2, 6 - have))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        if not queue:
            return
        if not self.read_many(queue, 12):
            self.read_props(queue[:5], 22)

    # the interview
    def ready_to_ask(self) -> bool:
        if self.interviewed or not self.jobs.get("message") or self.spare_messages() <= 0 or self.steps_left() <= 8 or self.late():
            return False
        return len(self.sheets) >= 3 * len(self.lines) + 1 or self.store.step >= max(8, self.store.max_steps // 4)

    def msg_args(self, text: str) -> dict[str, Any] | None:
        schema = self.params_of.get(self.jobs.get("message") or "") or {}
        texts = free_text_fields(schema)
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = fields_of(schema).get(key) or {}
            if not (isinstance(spec.get("enum"), list) and spec["enum"]):
                return None
            args[key] = spec["enum"][0]
        faults: list[str] = []
        args = conform(args, schema, "args", faults)
        return None if faults else args

    def send_message(self, text: str, floor: int = 6) -> str | None:
        if self.over() or self.steps_left() <= floor or self.msg_used >= self.msg_cap:
            return None
        args = self.msg_args(text)
        if not args:
            return None
        reply = self.store_call(self.jobs["message"], args)
        if reply is None:
            return None
        self.msg_used += 1
        answer = reply.get("observation")
        words = spoken_text(answer)
        self.record_log.append(f"step {self.store.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {as_text(self.basket)})")
        self.record_steps.append(self.store.step)
        self.log.append({"kind": "ask", "q": text, "a": words[:300]})
        if isinstance(answer, dict) and answer.get("error"):
            return None
        return words

    def plan_questions(self, count: int, menu: dict[int, list[tuple[str, int, int, list[str]]]]) -> dict[int, list[str]]:
        products = [f"product {n}: {line.name} - settled by the request: " + ("; ".join(line.musts) or "nothing") for n, line in enumerate(self.lines, 1)]
        table = []
        for number in range(1, len(self.lines) + 1):
            opened = len([i for i in self.groups().get(number, []) if i in self.sheets])
            table.append(f"product {number} ({opened} listings opened):")
            for name, carried, distinct, samples in menu.get(number, []):
                table.append(f"  - {name} (on {carried}, {distinct} different): " + " | ".join(samples))
            if not menu.get(number):
                table.append("  (no listing of this product opened yet)")
        per = min(PLAN_PER_PRODUCT, count + 2)
        prompt = PLAN_GUIDE.replace("PER_PRODUCT", str(per)) + "\n\nREQUEST:\n" + self.request_text + "\n\nPRODUCTS:\n" + "\n".join(products)
        first = [str(x) for x in as_list(self.reading.get("check_with_shopper")) if str(x).strip()]
        if first:
            prompt += "\n\nCHECK WITH THE SHOPPER (the request asks for these):\n" + "\n".join(first)
        prompt += "\n\nPROPERTY KEYS SEEN:\n" + "\n".join(table)
        got = self.brains.ask(prompt, "planner") or {}
        ranked: dict[int, list[str]] = {}
        for entry in as_list(got.get("ask")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, as_int(entry.get("product")) or 1), len(self.lines))
            label = one_property(entry.get("property"))
            if label and len(label.split()) <= 5 and label_key(label) not in {label_key(x) for x in ranked.get(number, [])}:
                ranked.setdefault(number, []).append(label)
        for number, rows in menu.items():
            spare = [name for name, carried, distinct, _ in rows if distinct >= 2 and carried >= 2]
            ranked.setdefault(number, [])
            ranked[number] += [s for s in spare if label_key(s) not in {label_key(x) for x in ranked[number]}]
        self.log.append({"kind": "plan", "ask": shorten(got.get("ask"), 120), "ranked": ranked})
        return ranked

    def next_question(self, ranked: dict[int, list[str]], found: dict[int, int], tried: dict[int, int]) -> tuple[int, str] | None:
        def candidates(number: int) -> list[str]:
            settled = self.lines[number - 1].settled_properties()
            return [label for label in ranked.get(number, []) if label_key(label) not in self.asked_labels and label_key(label) not in settled]

        open_products = [n for n in range(1, len(self.lines) + 1) if candidates(n)]
        if not open_products:
            return None
        unfound = [n for n in open_products if not found.get(n)]
        pool = unfound or open_products
        number = min(pool, key=lambda n: (found.get(n, 0), tried.get(n, 0), n))
        return number, candidates(number)[0]

    def listen(self, number: int, question: str, reply: str) -> dict[str, Any]:
        products = "\n".join(f"product {n}: {line.name}" for n, line in enumerate(self.lines, 1))
        prompt = LISTEN_GUIDE + "\n\nPRODUCTS:\n" + products + f"\n\nQUESTION (about product {number}): {question}\nREPLY: {reply}"
        got = self.brains.ask(prompt, "listener") or {}
        clear = got.get("no_need") is True or any(isinstance(e, dict) and str(e.get("need") or "").strip() for e in as_list(got.get("needs")))
        if not clear and not self.brains.no_credit:
            got = self.brains.ask(prompt, "listener_hard") or got
        return got

    def take_heard(self, heard: dict[str, Any], told: list[str], asked: int | None = None, label: str = "") -> int:
        several = len(self.lines) > 1
        added = 0
        for entry in as_list(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, as_int(entry.get("product")) or asked or 1), len(self.lines))
            need = str(entry["need"]).strip()
            line = self.lines[number - 1]
            if need not in line.told and need not in line.musts:
                line.told.append(need)
                told.append(f"product {number}: {need}" if several else need)
                added += 1
            self.asked_labels.add(label_key(property_of(need)))
            if entry.get("points") is True:
                self.note_pointer(need, number)
        for entry in as_list(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                line = self.product_line(entry.get("product"))
                gone = property_of(entry["property"])
                hit = [m for m in line.live_needs() if property_of(m) == gone]
                line.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        if heard.get("no_need") is True and asked and label and not added:
            self.lines[asked - 1].unwanted.append(label)
        budget = as_number(heard.get("budget"))
        if budget and budget > 0:
            self.money_cap = budget
        return added

    def question_text(self, number: int, label: str) -> str:
        return f"Which {label} do you want for the {self.lines[number - 1].name}?"

    def interview(self) -> str:
        self.interviewed = True
        if self.spare_messages() <= 0:
            return ""
        self.widen()
        ranked = self.plan_questions(self.spare_messages(), self.key_menu())
        found = {n: 0 for n in range(1, len(self.lines) + 1)}
        tried = {n: 0 for n in range(1, len(self.lines) + 1)}
        told: list[str] = []
        talk: list[str] = []
        new_needs: list[tuple[int, str]] = []
        while self.spare_messages() > 0 and not self.over() and self.steps_left() > 6 and not self.late():
            choice = self.next_question(ranked, found, tried)
            if not choice:
                break
            number, label = choice
            self.asked_labels.add(label_key(label))
            tried[number] += 1
            question = self.question_text(number, label)
            reply = self.send_message(question)
            if reply is None:
                break
            talk.append(f"Q: {question}\nA: {reply}")
            before = len(told)
            heard = self.listen(number, question, reply)
            added = self.take_heard(heard, told, number, label)
            if added:
                for entry in as_list(heard.get("needs")):
                    if isinstance(entry, dict) and str(entry.get("need") or "").strip() and entry.get("points") is not True:
                        target = min(max(1, as_int(entry.get("product")) or number), len(self.lines))
                        found[target] = found.get(target, 0) + 1
                        new_needs.append((target, str(entry["need"]).strip()))
                    elif isinstance(entry, dict) and entry.get("points") is True:
                        found[number] = found.get(number, 0) + 1
            self.log.append({"kind": "heard", "product": number, "label": label, "new": told[before:]})
        if told:
            self.epoch += 1
        seen = self.search_after_reveal(new_needs)
        summary = "Yamuna's interviewer asked the shopper (one step per question):\n" + ("\n".join(talk) or "(no question went out)")
        summary += "\nNew firm requirements: " + ("; ".join(told) if told else "none") + "."
        if told:
            summary += " Apply them to every pick and check the listings you meant to buy against them."
        pointed = [need for _, need in self.pointers if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full and every listing its page"
                        " refers to; the last page gives the value, and the order counts only once all of them are opened.")
        if seen:
            summary += "\n" + seen
        if self.side_notes:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.side_notes)
            self.side_notes = []
        return summary

    def code_search(self, words: str) -> str:
        tool = self.jobs.get("search")
        schema = self.params_of.get(tool or "") or {}
        texts = free_text_fields(schema)
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []) or not words.strip():
            return ""
        faults: list[str] = []
        args = conform({target: words.strip()}, schema, "args", faults)
        mark = tool + as_text(args)
        if faults or mark in self.read_memo:
            return ""
        reply = self.store_call(tool, args)
        if reply is None:
            return ""
        seen = reply.get("observation")
        self.soak_and_read(tool, seen)
        text = self.render(seen)
        self.read_memo[mark] = (self.epoch, self.store.step, text)
        self.last_calls = (self.last_calls + [mark])[-3:]
        return f"[Yamuna searched {as_text(args)}] " + text[:3000] + self.cheapest_rows([seen])

    def search_after_reveal(self, needs: list[tuple[int, str]]) -> str:
        shown = []
        for number, need in needs[:SEARCH_AFTER_REVEAL]:
            if self.steps_left() < 16 or self.over() or self.late():
                break
            value = need.split(":", 1)[1] if ":" in need else need
            text = self.code_search(query_words(value, 4) + " " + query_words(self.lines[number - 1].name, 3))
            if text:
                shown.append(text)
        if shown:
            self.log.append({"kind": "search_after_reveal", "n": len(shown)})
        return "\n".join(shown)

    # the cart as Yamuna knows it
    def basket_fresh(self) -> bool:
        return self.basket_view is not None and self.basket_seen_at >= self.basket_changed_at

    def basket_text(self) -> str:
        if self.basket_fresh():
            return as_text(shorten(self.basket_view, 300))[:3000]
        return "lines by Yamuna's record of the cart calls (listing: units): " + as_text(self.basket)

    def basket_ids(self) -> list[str]:
        if self.basket_fresh():
            text = as_text(self.basket_view)
            return [i for i in self.seen_rows if i in text]
        return list(self.basket)

    def other_records(self, skip: list[str], limit: int) -> str:
        ordered = list(reversed(self.order))
        if self.pick() == "cheapest":
            ordered.sort(key=lambda i: (i not in self.sheets, self.price(i) if self.price(i) is not None else float("inf")))
        lines = []
        for ident in ordered:
            if ident in skip or self.is_gone(ident):
                continue
            lines.append(as_text(shorten(self.sheets.get(ident) or self.seen_rows.get(ident), 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def judge(self, ids: list[str], planned: bool = False, spend_cap: float = SPEND_EASY) -> str:
        key = "|".join(sorted(ids)) + f"#{self.epoch}"
        if not ids or self.brains.cost > spend_cap:
            return ""
        if key in self.judged:
            return "" if self.judged[key] else "held"
        prompt = JUDGE_GUIDE + "\n\nREQUEST:\n" + self.request_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.board()
        if planned:
            prompt += "\n\nCART (planned; judge these as carted at the requested units: " + "; ".join(f"product {n}: {line.units}" for n, line in enumerate(self.lines, 1)) + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.basket_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.is_gone(i) else "") + self.record_of(i) for i in ids)
        if self.pick() in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN:\n" + self.other_records(ids, 15)
        got = self.brains.ask(prompt, "judge")
        breaks = [str(b) for b in as_list((got or {}).get("breaks")) if str(b).strip()][:5]
        better = [str(b) for b in as_list((got or {}).get("better")) if str(b) in self.seen_rows and not self.is_gone(str(b)) and str(b) not in ids][:4]
        self.log.append({"kind": "judge", "cart": ids, "breaks": breaks, "better": better})
        clean = not breaks and not better
        self.judged[key] = clean
        if clean:
            return ""
        text = "The judge found a clear problem with this cart: " + "; ".join(breaks or ["a listing that beats the cart's choice on the pick rule"])
        if better:
            text += ". Better-looking listings: " + ", ".join(better)
        return text + ". Fix the cart (or re-read the records if the judge is wrong), then check out."

    def follow_chains(self) -> tuple[str, bool]:
        live = {need for _, need in self.all_needs()}
        pending = [need for _, need in self.pointers if need in live and need not in self.pointer_found]
        tool = self.offer_view()
        target = self.one_id_param(tool)
        opened_now: list[str] = []
        lost: list[str] = []
        while pending and self.chain_rounds < 4 and self.brains.cost < SPEND_STOP and not self.late():
            self.chain_rounds += 1
            records = []
            for ident in reversed(self.order):
                record = (self.sheets.get(ident) or self.seen_rows.get(ident)) if ident in self.page_opened else None
                records.append(("OPENED " + as_text(shorten(record, 300))[:1400]) if record else ("ROW " + as_text(shorten(self.seen_rows.get(ident), 80))[:300]))
                if len(records) >= 60:
                    break
            prompt = TRAIL_GUIDE + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending) + "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.brains.ask(prompt, "trail") or {}
            self.log.append({"kind": "trail", "pending": pending, "got": shorten(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in as_list(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in as_list(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.page_opened for i in ids) and str(chain.get("value") or "").strip():
                    self.pointer_found[need] = str(chain["value"]).strip()
                    self.epoch += 1
                    continue
                closed = [i for i in ids if i in self.seen_rows and i not in self.page_opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.seen_rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.pointer_found]
            if not reach or not tool or not target or self.steps_left() < len(reach) + 4 or self.over():
                break
            for ident in reach[: max(0, min(3, 6 - len(opened_now)))]:
                faults: list[str] = []
                args = conform({target: ident}, self.params_of.get(tool), "args", faults)
                if faults:
                    break
                reply = self.store_call(tool, args)
                if reply is None or self.over():
                    break
                self.soak_and_read(tool, reply.get("observation"))
                opened_now.append(f"{ident}: " + self.render(reply.get("observation"))[:1500])
        text = ""
        if opened_now:
            text += "[Yamuna opened the pages a requirement points at]\n" + "\n".join(opened_now) + "\n"
        if self.pointer_found:
            text += "[values read from those pages] " + "; ".join(f"{k} -> {v}" for k, v in self.pointer_found.items()) + "\n"
        hold = bool(lost) and self.chain_holds < 2 and self.steps_left() > 5
        if hold:
            self.chain_holds += 1
            text += ("The order counts only once every page a requirement points at is opened in full (a search row is not enough), following each page that sends you on. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Look it up by its shop, brand or wording, open its page, use the value written there, and only then check out.")
        return text, hold

    def refresh_offers(self) -> str:
        tool = self.offer_view()
        target = self.one_id_param(tool)
        if not tool or not target:
            return ""
        since = max(self.store.step - 2, self.turned_down_at)
        stale = [i for i in self.basket_ids() if self.offer_checked_at.get(i, -99) < since and i not in self.sold_marks]
        if not stale or self.steps_left() < len(stale) + 3:
            return ""
        changed = []
        for ident in stale:
            faults: list[str] = []
            args = conform({target: ident}, self.params_of.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.seen_rows.get(ident) or {})
            reply = self.store_call(tool, args)
            if reply is None or self.over():
                break
            seen = reply.get("observation")
            self.offer_checked_at[ident] = self.store.step
            self.soak_and_read(tool, seen)
            after = self.seen_rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                changed.append(f"{ident} changed ({', '.join(moved[:4])}): {as_text(shorten(seen, 300))[:900]}")
        self.log.append({"kind": "reread", "ids": stale, "changed": len(changed)})
        if changed:
            self.epoch += 1
        return "\n".join(changed)

    def sold_line_hint(self) -> str:
        lost = [i for i in self.basket_ids() if i not in self.sold_marks and self.is_gone(i)]
        if not lost:
            return ""
        for ident in lost:
            self.sold_marks[ident] = self.store.step
        self.epoch += 1
        picks = self.stand_ins(lost)
        text = f"[Yamuna] Sold out now: {', '.join(lost)}. It will not come back, so do not open it again."
        if picks and self.steps_left() <= 6:
            return text + f" Few steps left: add {picks[0]} at the requested units and remove the sold-out line in this turn, then check out. Other fits already read: {', '.join(picks[1:]) or 'none'}."
        if picks and self.pick() == "cheapest":
            return text + f" Listings already read that appear to fit instead, cheapest first: {', '.join(picks)}. Unless you have seen the whole fitting set, run one search under {picks[0]}'s price before swapping in the cheapest fit."
        if picks:
            return text + f" Listings already read that appear to fit instead: {', '.join(picks)}. Take the sold-out line out and add {picks[0]} at the requested units, then check out."
        return text + " Remove that line and add the best other fitting listing; search for one if none you read fits."

    def spot_sold(self) -> str:
        ids = [i for i in self.basket_ids() if i not in self.sold_marks]
        if any(self.is_gone(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.off_sale.add(ids[0])
            return ""
        return self.refresh_offers() if ids else ""

    def stand_ins(self, lost: list[str]) -> list[str]:
        if self.brains.cost > SPEND_EASY:
            return []
        records = []
        for ident in reversed(self.order):
            if ident in lost or self.is_gone(ident):
                continue
            records.append(as_text(shorten(self.sheets.get(ident) or self.seen_rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP_GUIDE + "\n\nREQUEST:\n" + self.request_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.board()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.basket_text()
        prompt += "\n\nMORE RECORDS, NEWEST FIRST:\n" + "\n".join(records)
        got = self.brains.ask(prompt, "swap") or {}
        fits: list[tuple[float, str]] = []
        cart = self.basket_ids()
        for entry in as_list(got.get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = as_number(entry.get("price")) if isinstance(entry, dict) else None
            if ident in self.seen_rows and ident not in lost and ident not in self.sold_marks and ident not in cart and ident not in [i for _, i in fits]:
                fits.append((price if price is not None else float("inf"), ident))
        ranked = [i for _, i in sorted(fits)] if self.pick() == "cheapest" else [i for _, i in fits]
        self.log.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def read_cheaper(self) -> list[str]:
        if self.pick() != "cheapest" or not self.kind_field():
            return []
        mark = self.basket_mark()
        if mark in self.cheaper_marks_done:
            return []
        self.cheaper_marks_done.add(mark)
        cart = self.basket_ids()
        picks: list[str] = []
        for line in cart:
            price, kind = self.price(line), self.kind(line)
            if price is None or not kind:
                continue
            pool = [i for i in self.order if i not in cart and i not in self.sheets and self.can_buy(i) and self.kind(i) == kind
                    and (self.price(i) if self.price(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.read_props(picks[:CHEAPER_READS], self.finish_room() + 4)

    # before ending without an order
    def review(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.order):
            if self.is_gone(ident):
                continue
            record = self.sheets.get(ident)
            records.append(("OPENED " + as_text(shorten(record, 200))[:1200]) if record else ("ROW " + as_text(shorten(self.seen_rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = REVIEW_GUIDE + "\n\nREQUEST:\n" + self.request_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.board() + "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.brains.ask(prompt, "review")
        self.log.append({"kind": "review", "got": shorten(got, 150)})
        return got

    def review_lists(self, got: dict[str, Any] | None) -> tuple[dict[str, list[str]], dict[str, str]]:
        fits = (got or {}).get("fits")
        named: dict[str, list[str]] = {}
        if isinstance(fits, dict):
            for key, ids in fits.items():
                kept = [str(i) for i in as_list(ids) if str(i) in self.seen_rows and not self.is_gone(str(i))][:4]
                if kept and as_int(key):
                    named[str(as_int(key))] = kept
        nearest: dict[str, str] = {}
        closest = (got or {}).get("closest")
        if isinstance(closest, dict):
            for key, ident in closest.items():
                if as_int(key) and str(ident) in self.seen_rows and not self.is_gone(str(ident)):
                    nearest[str(as_int(key))] = str(ident)
        return named, nearest

    def check_before_decline(self) -> str:
        if self.reviewed or self.brains.cost > SPEND_STOP:
            return ""
        self.reviewed = True
        if not self.decline_opened:
            self.decline_opened = True
            fresh = self.unread_rows(None, DECLINE_OPENS)
            if fresh and not self.read_many(fresh, self.finish_room() + 6):
                self.read_props(fresh, self.finish_room() + 6)
        got = self.review()
        fits, nearest = self.review_lists(got)
        if not fits:
            rows = [str(i) for i in as_list((got or {}).get("open")) if str(i) in self.seen_rows and str(i) not in self.sheets and not self.is_gone(str(i))]
            if rows and (self.read_many(rows, self.finish_room() + 3) or self.read_props(rows[:2], self.finish_room() + 3)):
                got = self.review()
                fits, nearest = self.review_lists(got)
        self.leads = fits
        self.closest = nearest
        if not fits:
            return ""
        reason = str((got or {}).get("reason") or "").strip()
        return ("A review of the records found listings that appear to fit: " + "; ".join(f"product {k}: {', '.join(v)}" for k, v in fits.items())
                + (f" ({reason})" if reason else "") + ". Cart the best fitting one per product (keeping the pick rule) and check out; end without an order only if each of them clearly fails.")

    # Yamuna builds the cart and orders by itself
    def basket_sets_count(self, name: str) -> bool:
        params = fields_of(self.params_of.get(name))
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in params.values()):
            return False
        return any(isinstance(s, dict) and "integer" in type_names(s) and s.get("minimum") == 0 for s in params.values())

    def basket_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.jobs.get("cart") or ""
        schema = self.params_of.get(name) or {}
        params = fields_of(schema)
        for key, spec in params.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in type_names(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_field = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in type_names(s) and "enum" not in s), None)
                count_field = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in type_names(s)), None)
                if not id_field:
                    return None
                return [{key: [{id_field: i, **({count_field: n} if count_field else {})} for i, n in lines.items()]}]
        texts = free_text_fields(schema)
        target = (next((k for t, k in self.id_params if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in params.items() if isinstance(s, dict) and "integer" in type_names(s)), None)
        mode = next((k for k, s in params.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.basket.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: params[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.basket_sets_count(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(params[count].get("minimum"), int) and params[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        current = self.basket_ids()
        for ident, want in lines.items():
            have = self.basket.get(ident, 0) if ident in current else 0
            if have == want:
                continue
            if have and not self.basket_sets_count(name):
                return None
            calls.append({target: ident, **({mode: params[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def order_myself(self, choice: dict[int, str], why: str) -> str:
        cart, buy = self.jobs.get("cart"), self.jobs.get("buy")
        if self.self_ordered or not cart or not buy or not choice:
            return ""
        self.self_ordered = True
        plan: dict[str, int] = {}
        kinds: set[str] = set()
        for number, ident in sorted(choice.items()):
            if ident and not self.is_gone(ident) and 1 <= number <= len(self.lines) and ident not in plan:
                plan[ident] = self.lines[number - 1].units
                kinds.add(self.kind(ident) or ident)
        now = self.basket_ids()
        keep = {i: self.basket.get(i, 1) for i in now if i not in plan and not self.is_gone(i) and (self.kind(i) or i) not in kinds}
        drop = [i for i in now if i not in plan and i not in keep]
        if not plan:
            return ""
        lines = {**keep, **plan}
        calls = self.basket_calls(lines, drop)
        if not calls or len(calls) + 1 > self.steps_left():
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = conform(args, self.params_of.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.store_call(cart, args)
            if reply is None or self.over():
                return ""
            self.note_basket(cart, args, reply.get("observation"))
            self.soak_and_read(cart, reply.get("observation"))
            done.append(as_text(args))
        faults = []
        args = conform(dict(self.reading.get("buy_args") or {}), self.params_of.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        for attempt in range(2):
            if attempt and (self.steps_left() < 2 or self.over() or any(self.is_gone(i) for i in lines)):
                break
            if attempt:
                self.refresh_offers()
            go, said = self.run_gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.store_call(buy, args)
            self.log.append({"kind": "ordered_myself", "why": why, "lines": lines, "drop": drop, "reply": shorten(reply, 200)})
            if reply is None:
                break
            self.checkouts += 1
            if not reply.get("finished"):
                self.turned_down_at = self.store.step
            self.soak_and_read(buy, reply.get("observation"))
            replies.append(self.render(reply.get("observation")))
            if self.over() or reply.get("finished"):
                break
        if not replies:
            return ""
        return f"Handled by Yamuna instead: {why}, so Yamuna set the cart ({'; '.join(done)}) and checked out. Store reply: " + " | then: ".join(replies)

    def plain_decline(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.params_of.get(name) or {}
        out = dict(args)
        for key, spec in fields_of(schema).items():
            if not isinstance(spec, dict) or "string" not in type_names(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            if not text or any(ident in text for ident in self.seen_rows) or self.reading.get("no_names_on_decline") is not False:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."[: spec.get("maxLength", 200)]
        return out

    # steps the request wants before the order or the ending
    def run_gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.asked_steps if p["stage"] == stage]
        if not steps or self.over() or not self.gate_open(stage):
            return True, ""
        ending = {self.jobs.get("buy"), self.jobs.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {as_text(t['function'].get('parameters') or {})}"
                 for t in self.tool_specs if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_TRIES):
            prompt = (GATE_GUIDE.replace("STAGE", "place the order" if stage == "order" else "end without an order")
                      + "\n\nREQUEST:\n" + self.request_text + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['what']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.board() + "\n\nCART NOW: " + self.basket_text()
                      + "\n\nRECORD:\n" + ("\n".join(f"{n}. {line}" for n, line in enumerate(self.record_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.change_log[-4:]) if self.change_log else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_cap - self.msg_used)}   STEPS LEFT: {self.steps_left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            got = self.brains.ask(prompt, "gate_again" if nudged else "gate")
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.log.append({"kind": "gate", "stage": stage, "call": shorten(call, 200), "hold": hold[:200], "told": told})
            if told or hold:
                self.epoch += 1
                return False, ("Before going on, Yamuna did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper answered with a change: " + "; ".join(told + ([hold] if hold else [])) + ". Bring the cart in line with it and try the checkout again.")
            name = str((call or {}).get("tool") or "")
            shown = [e for e in as_list(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            proof = {n for n in range(1, len(self.record_log) + 1) if not talk or (n <= len(self.record_steps) and self.record_steps[n - 1] > self.basket_changed_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((as_int(n) or 0) in proof for n in as_list(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.jobs.get("message") and self.msg_used < self.msg_cap and self.steps_left() >= 2:
                nudged = True
                self.record_log.append(f"(Yamuna: no line since the cart last changed at step {self.basket_changed_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                       + f" done, and {self.msg_cap - self.msg_used} message(s) and {self.steps_left()} steps are left: give the call that does it)")
                self.record_steps.append(-1)
                continue
            if not call or name not in self.params_of or name in ending:
                if not call:
                    self.gate_passed.add((stage, self.basket_mark()))
                break
            if self.steps_left() < 2 or (name == self.jobs.get("message") and self.msg_used >= self.msg_cap):
                break
            faults: list[str] = []
            args = conform(call.get("args") if isinstance(call.get("args"), dict) else {}, self.params_of[name], "args", faults)
            if faults:
                break
            if name == self.jobs.get("message"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                reply = self.send_message(text, 1)
                if reply is None:
                    break
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {reply}")
                continue
            reply = self.store_call(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.soak(seen)
            self.record_log.append(f"step {self.store.step}: {name} {as_text(args)} -> {self.render(seen)[:300]} (cart then: {as_text(self.basket)})")
            self.record_steps.append(self.store.step)
            said.append(f"{name} {as_text(args)} -> {self.render(seen)[:400]}")
        return True, ("Before going on, Yamuna did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    # the two endings
    def buys_now(self, name: str, args: dict[str, Any]) -> bool:
        fixed = self.reading.get("buy_args") if isinstance(self.reading.get("buy_args"), dict) else {}
        return name == self.jobs.get("buy") and all(args.get(k) == v for k, v in fixed.items())

    def ends_now(self, name: str, args: dict[str, Any]) -> bool:
        fixed = self.reading.get("refuse_args") if isinstance(self.reading.get("refuse_args"), dict) else {}
        return name == self.jobs.get("refuse") and all(args.get(k) == v for k, v in fixed.items())

    def before_checkout(self) -> str:
        reread = self.refresh_offers()
        viewer = self.jobs.get("cart_view")
        if self.basket_touched and self.basket_seen_at < self.basket_changed_at and viewer and self.steps_left() > 3:
            faults: list[str] = []
            view_args = conform({}, self.params_of.get(viewer), "args", faults)
            view = self.store_call(viewer, view_args) if not faults else None
            if view is not None:
                self.basket_view = view.get("observation")
                self.basket_seen_at = self.store.step
                change = self.soak_and_read(viewer, self.basket_view)
                if change:
                    self.side_notes = []
                    return "Held back, no step spent. The cart view shows a change: " + change + "\nCart now: " + self.basket_text() + "\nApply it, correct the cart where needed, then check out."
        swap = self.sold_line_hint()
        held = [i for i in self.basket_ids() if i in self.sold_marks] if self.sold_holds < 2 else []
        if swap or held:
            self.sold_holds += 0 if swap else 1
            return "Held back, no step spent. " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.basket_text()
        chain, hold = self.follow_chains()
        if hold:
            return "Held back, no step spent. " + chain + "\nCart now: " + self.basket_text()
        if self.judge_holds >= JUDGE_HOLDS or self.late():
            return ""
        read = self.read_props(self.basket_ids(), self.finish_room() + 3) + self.read_cheaper()
        verdict = self.judge(self.basket_ids())
        if verdict and verdict != "held":
            self.judge_holds += 1
            seen = ("Yamuna read these listings' properties for the judge (one step each):\n" + "\n".join(read) + "\n") if read else ""
            return "Held back. " + chain + seen + verdict + "\nCart now: " + self.basket_text()
        if reread and self.brains.cost > SPEND_EASY and self.judge_holds == 0:
            self.judge_holds += 1
            return "Held back, no step spent. Yamuna has just re-read the cart's listings:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."
        return ""

    def product_for(self, ident: str, fallback: int | None) -> int | None:
        if self.kind_field():
            number = self.product_for_kind(self.kind(ident))
            if number:
                return number
        return fallback if fallback and 1 <= fallback <= len(self.lines) else None

    def full_choice(self) -> dict[int, str]:
        choice: dict[int, str] = {}
        for ident in self.best_cart:
            number = self.product_for(ident, None)
            if number and number not in choice and not self.is_gone(ident):
                choice[number] = ident
        for key, ident in self.closest.items():
            number = self.product_for(ident, as_int(key))
            if number and number not in choice:
                choice[number] = ident
        for key, ids in self.leads.items():
            for ident in ids:
                number = self.product_for(ident, as_int(key))
                if number and not self.is_gone(ident):
                    choice[number] = ident
                    break
        groups = self.groups()
        for number in range(1, len(self.lines) + 1):
            read = [i for i in groups.get(number, []) if i in self.sheets and self.can_buy(i) and self.affordable(i, number) and i not in choice.values()]
            if number not in choice and read:
                choice[number] = min(read, key=lambda i: self.price(i) if self.price(i) is not None else float("inf"))
        return choice if len(choice) >= len(self.lines) else {}

    def before_decline(self) -> str:
        if self.steps_left() > 5 and not self.late():
            found = self.check_before_decline()
            if found:
                self.decline_holds += 1
                return "Held back, no step spent. " + found
            if not self.leads and not self.searched_more and self.steps_left() > self.store.max_steps // 3:
                self.searched_more = True
                return (f"Held back, no step spent. {self.steps_left()} steps remain. Before giving up, try searches not made so far: the plain one-word name of each product,"
                        " the requirement's value with its other common names, the next pages of results already seen; and read the properties of listings that might fit.")
        if self.reading.get("when_nothing_fits") == "decline" and not self.leads:
            return ""
        choice = self.full_choice()
        if not choice:
            return ""
        why = "a review found fitting listings" if self.leads else "ending without an order almost never earns the point, and these are the closest listings found"
        return self.order_myself(choice, why)

    def menu_now(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        ends = {self.jobs.get("buy"), self.jobs.get("refuse")} - {None}
        keep: set[str] | None = None
        if left <= self.finish_room() or self.late():
            keep = set(ends)
        elif left <= self.finish_room() + 3:
            keep = ends | {self.jobs.get("cart"), self.jobs.get("cart_view")}
        elif left <= len(self.lines) + 5 and self.checkouts == 0:
            keep = ends | {self.jobs.get("cart"), self.jobs.get("cart_view"), self.jobs.get("page")}
        tools = [t for t in self.tool_specs if t["function"]["name"] != self.jobs.get("message")]
        if keep is not None and ends:
            tools = [t for t in tools if t["function"]["name"] in keep] or tools
        return tools

    def fix_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.jobs.get(j) for j in ("cart", "details", "page", "many")} - {None} or not self.row_id_field:
            return args, ""
        params = fields_of(self.params_of.get(name))
        free = [k for k, v in params.items() if isinstance(v, dict) and ("string" in type_names(v) or "array" in type_names(v)) and "enum" not in v]
        if len(free) != 1 or (name != self.jobs.get("cart") and (name, free[0]) not in self.id_params):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.seen_rows:
                return value.strip(), ""
            if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.seen_rows.items() if any(k != self.row_id_field and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"{value!r} matches more than one listing ({', '.join(hits[:6])}); give a single listing id"
            return value, f"{value!r} is no listing id; use the id from the result rows (the {self.row_id_field} field)"

        key = free[0]
        if key not in args:
            return args, ""
        fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
        problem = next((p for _, p in fixed if p), "")
        if problem:
            return args, problem
        return {**args, key: [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]}, ""

    def note_basket(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.basket_touched = True
        self.basket_changed_at = self.store.step
        params = fields_of(self.params_of.get(name))
        removal = any(isinstance((params.get(k) or {}).get("enum"), list) and len(params[k]["enum"]) >= 2 and v in params[k]["enum"][1:] for k, v in args.items())
        loose = next((as_int(v) for v in args.values() if as_int(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (params.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for entry in value:
                    if isinstance(entry, dict):
                        whole = True
                        ident = next((v for v in entry.values() if isinstance(v, str) and v in self.seen_rows), None)
                        count = next((as_int(v) for v in entry.values() if as_int(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(entry, str) and entry in self.seen_rows:
                        lines.append((entry, loose))
            elif isinstance(value, str) and value in self.seen_rows:
                lines.append((value, loose))
        if whole:
            self.basket = {i: c for i, c in lines if c and c > 0}
        else:
            sets = self.basket_sets_count(name)
            for ident, count in lines:
                have = self.basket.get(ident, 0)
                if removal:
                    have = have - count if count else 0
                elif sets:
                    have = count if count is not None else 1
                else:
                    have += count if count is not None else 1
                if have > 0:
                    self.basket[ident] = have
                else:
                    self.basket.pop(ident, None)
        self.basket_reply = as_text(observation)
        self.last_calls = []
        counts = [as_int(v) for v in args.values() if not isinstance(v, str) and as_int(v) is not None]
        for value in args.values():
            if counts and isinstance(value, str) and value in self.seen_rows:
                if counts[0] <= 0 and not carries_id(observation, value):
                    self.taken_off.add(value)
                elif counts[0] > 0:
                    self.taken_off.discard(value)
        current = [i for i in self.basket_ids() if not self.is_gone(i) and i not in self.taken_off]
        if len(current) >= len(self.best_cart):
            self.best_cart = current

    def merge_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target, many = self.jobs.get("details"), self.jobs.get("many"), self.one_id_param(self.jobs.get("details")), self.many_param()
        if not opener or not joiner or not target or not many or opener == joiner or self.many_is_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = read_json(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.seen_rows and ident not in self.sold_marks:
                picks.append((index, ident))
        picks = picks[: self.many_cap()]
        if len(picks) < 2:
            return calls
        joined = {i for i, _ in picks}
        out = []
        for index, call in enumerate(calls):
            if index == picks[0][0]:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": as_text({many: [i for _, i in picks]})}})
            elif index in joined:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.log.append({"kind": "joined", "ids": [i for _, i in picks]})
        return out

    def screen_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.merge_opens(calls)
        readers = {self.jobs.get(j) for j in ("search", "details", "page", "many")} - {None}
        for call in calls:
            cid = call["id"]
            allowed = {t["function"]["name"] for t in self.menu_now()}
            if call.get("joined"):
                out.append((cid, "Merged into the side-by-side call above: one step read these listings together."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            if self.over():
                out.append((cid, "Skipped: the episode has already ended."))
                continue
            if name == self.jobs.get("message"):
                out.append((cid, "Skipped, no step spent: only Yamuna's interviewer writes to the shopper."))
                continue
            if name not in self.params_of:
                out.append((cid, f"Skipped, no step spent: this store has no such tool. Its tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Skipped, no step spent: the steps left are kept for finishing; available now: " + ", ".join(sorted(allowed)) + "."))
                continue
            raw = fn.get("arguments")
            args = raw if isinstance(raw, dict) else (read_json(raw) if (raw or "").strip() else {})
            if args is None:
                out.append((cid, "Skipped, no step spent: the arguments must form a single JSON object."))
                continue
            faults: list[str] = []
            args = conform(args, self.params_of[name], "args", faults)
            if faults:
                out.append((cid, "Skipped, no step spent, the arguments do not fit the schema: " + "; ".join(faults[:6])))
                continue
            args, wrong = self.fix_ids(name, args)
            if wrong:
                out.append((cid, "Skipped, no step spent: " + wrong + "."))
                continue
            extra = ""
            many = self.many_param() if name == self.jobs.get("many") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.many_cap() and self.many_fail:
                room = self.many_cap()
                extra = f"\n[Yamuna] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.one_id_param(name)
            if target and name in (self.jobs.get("details"), self.jobs.get("page")) and str(args.get(target)) in self.sold_marks:
                out.append((cid, f"Skipped, no step spent: {args[target]} went out of stock at step {self.sold_marks[str(args[target])]} and stays gone; put another fitting listing in its place."))
                continue
            if name == self.jobs.get("cart") and any((as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = as_text(self.basket_view) if self.basket_fresh() else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.seen_rows and v not in self.basket and v not in view and v not in self.basket_reply]
                if absent:
                    out.append((cid, f"Skipped, no step spent: {absent[0]} has no line in the cart to reduce. The cart holds: " + self.basket_text()))
                    continue
            buying, refusing = self.buys_now(name, args), self.ends_now(name, args)
            ending = buying or refusing
            if ending and len(calls) > 1:
                out.append((cid, "Skipped, no step spent: a finishing call goes alone, in a turn of its own, once the other results are read."))
                continue
            if not ending and self.steps_left() <= self.finish_room() and (self.jobs.get("buy") or self.jobs.get("refuse")):
                out.append((cid, "Skipped, no step spent: the final steps belong to the checkout or the ending."))
                continue
            mark = name + as_text(args)
            if name == self.jobs.get("cart") and self.basket_results.get(mark) == as_text(sorted(self.basket.items())):
                out.append((cid, "Skipped, no step spent: this exact cart call already ran and the cart still matches its result. The cart holds: " + self.basket_text()
                            + ". Each product needs one line at its requested units: " + "; ".join(f"{line.name}: {line.units}" for line in self.lines) + "."))
                continue
            if not ending and name != self.jobs.get("cart") and mark in self.last_calls:
                out.append((cid, "Skipped, no step spent: an identical call just ran and nothing has changed; scroll up for its result."))
                continue
            cached = self.read_memo.get(mark)
            if name in readers and cached and cached[0] == self.epoch:
                out.append((cid, f"Skipped, no step spent: step {cached[1]} made this call already and nothing has changed; its result was:\n" + cached[2][:3000]))
                continue
            if target and name in readers and name != self.jobs.get("many") and self.nothing_new(name, str(args.get(target))):
                out.append((cid, f"Skipped, no step spent: since the last change {args[target]} has nothing new to show in this view. What Yamuna holds on it: " + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                fresh = [i for i in args[many] if not self.nothing_new(name, str(i))]
                if not fresh:
                    out.append((cid, "Skipped, no step spent: since the last change these listings have nothing new to show in this view. What Yamuna holds on them:\n"
                                + "\n".join(f"{i}: {self.record_of(str(i))}" for i in args[many])[:4000]))
                    continue
                if len(fresh) < len(args[many]):
                    extra += f"\n[Yamuna] Left out {', '.join(str(i) for i in args[many] if i not in fresh)}: shown in full already."
                    args = {**args, many: fresh}
                    mark = name + as_text(args)
            if buying and self.steps_left() > 2:
                hold = self.before_checkout()
                if hold:
                    out.append((cid, hold))
                    continue
            if refusing:
                if self.steps_left() <= 5 and not self.late_kept:
                    kept = [i for i in self.best_cart if not self.is_gone(i)]
                    missing = [i for i in kept if i not in self.basket_ids() or (i in self.taken_off and not self.basket_fresh())]
                    if len(kept) >= len(self.lines) and len(missing) + 1 <= self.steps_left():
                        self.late_kept = True
                        redo = f"Put {', '.join(missing)} back in the cart at the requested units, then check out." if missing else "Check out the cart as it is."
                        out.append((cid, "Held back, no step spent. Walking away now would waste a cart that is already built, and a shortage of steps is no evidence that nothing fits. " + redo))
                        continue
                answer = self.before_decline()
                if answer:
                    out.append((cid, answer + (" [episode finished]" if self.store.finished else "")))
                    continue
                args = self.plain_decline(name, args)
            if ending and self.steps_left() >= 2:
                go, said = self.run_gate("order" if buying else "end")
                if not go:
                    out.append((cid, "Held back. " + said + "\nCart now: " + self.basket_text()))
                    continue
                if said:
                    extra = "\n[Yamuna] " + said + extra
            out.append((cid, self.run_call(name, args, mark, ending, buying, many, readers, extra)))
        return out

    def run_call(self, name: str, args: dict[str, Any], mark: str, ending: bool, buying: bool, many: str | None, readers: set[str | None], extra: str) -> str:
        reply = self.store_call(name, args)
        if reply is None:
            return "The store call failed on the network or server side; it can be sent again."
        observation = reply.get("observation")
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            if not any(str(i) in as_text(observation) for i in args[many]):
                self.many_fail = min(self.many_fail or 999, len(args[many]))
                extra += f"\n[Yamuna] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.many_cap()} per call."
            else:
                self.many_fine = max(self.many_fine, len(args[many]))
        self.last_calls = (self.last_calls + [mark])[-3:]
        self.id_params.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.seen_rows for x in (v if isinstance(v, list) else [v])))
        self.soak(observation)
        if name == self.jobs.get("many"):
            rows = self.listings_in(observation)
            if rows:
                self.shows_props[name] = self.shows_props.get(name, False) or any(property_block(r) for _, r in rows)
        if name in readers - {self.jobs.get("search")}:
            self.mark_viewed(name, observation)
        plain_jobs = {tool for job, tool in self.jobs.items() if job != "search"}
        paged = self.read_more_pages(name, args, observation) if name not in plain_jobs and not ending else ""
        self.learn_view_role(name, observation)
        if name == self.jobs.get("cart"):
            self.note_basket(name, args, observation)
            self.basket_results[mark] = as_text(sorted(self.basket.items()))
        if name == self.jobs.get("cart_view"):
            self.basket_view = observation
            self.basket_seen_at = self.store.step
        if buying:
            self.checkouts += 1
            if not reply.get("finished"):
                self.turned_down_at = self.store.step
        text = self.render(observation) + paged + extra
        if name not in plain_jobs and not ending:
            text += self.cheapest_rows([observation] + self.extra_results)
        base = text
        proof = self.key_hints(observation)
        if proof:
            text += "\n" + proof
        if name == self.jobs.get("cart") and any(as_int(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in self.seen_rows and carries_id(observation, v) for v in args.values()):
                text += "\n[Yamuna] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
        self.note_sold = False
        lines = self.stray_text(observation)
        if lines:
            meaning = self.read_store_note(lines)
            if meaning:
                text += "\n[store note] " + meaning + ". Take this into account in every pick before the checkout."
        if buying and not reply.get("finished") and (self.note_sold or any(line in spoken_text(observation) for line in self.sold_phrases)):
            seen = self.spot_sold()
            if seen:
                text += "\n[Yamuna re-read the cart's lines after the refusal]\n" + seen
        swap = self.sold_line_hint()
        if swap:
            text += "\n" + swap
        if self.side_notes:
            text += "\n[notes seen on Yamuna's own calls] " + " | ".join(self.side_notes)
            self.side_notes = []
        if name in readers:
            self.read_memo[mark] = (self.epoch, self.store.step, base)
        if reply.get("finished"):
            text += " [episode finished]"
        return text

    # ending the episode
    def wrap_up(self, again: bool = False) -> str:
        if self.over() or self.steps_left() <= 0:
            return ""
        buy, refuse = self.jobs.get("buy"), self.jobs.get("refuse")
        holding = bool(self.basket_ids()) if self.basket_fresh() else bool(self.basket)
        if buy and not holding and self.steps_left() >= 3 and not self.late():
            if not self.interviewed and self.jobs.get("message") and self.spare_messages() > 0 and self.steps_left() > 8:
                self.interview()
            if not self.reviewed and self.steps_left() > 5:
                self.check_before_decline()
            if self.leads or self.reading.get("when_nothing_fits") != "decline":
                choice = self.full_choice()
                if choice and self.order_myself(choice, "the walk ended with no cart, so Yamuna ordered the best listings found"):
                    return ""
        if buy and (holding or (self.basket_touched and not self.basket_fresh() and self.steps_left() >= 2)):
            faults: list[str] = []
            args = conform(dict(self.reading.get("buy_args") or {}), self.params_of.get(buy), "args", faults)
            if not faults:
                if self.steps_left() >= 2:
                    self.run_gate("order")
                if self.over() or self.steps_left() <= 0:
                    return ""
                reply = self.store_call(buy, args)
                self.log.append({"kind": "forced_buy", "reply": shorten(reply, 200)})
                if self.over() or self.steps_left() <= 0:
                    return ""
                if reply is not None and again:
                    self.checkouts += 1
                    self.turned_down_at = self.store.step
                    self.soak_and_read(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.render(reply.get("observation"))
        if refuse:
            faults = []
            args = conform(self.plain_decline(refuse, dict(self.reading.get("refuse_args") or {})), self.params_of.get(refuse), "args", faults)
            if not faults:
                if self.steps_left() >= 2:
                    self.run_gate("end")
                if self.over() or self.steps_left() <= 0:
                    return ""
                reply = self.store_call(refuse, args)
                self.log.append({"kind": "forced_refuse", "reply": shorten(reply, 200)})
                if reply is not None and not self.store.finished:
                    self.soak_and_read(refuse, reply.get("observation"))
                    return "ending without an order was refused: " + self.render(reply.get("observation"))
        return ""

    def trim_dialogue(self) -> None:
        if sum(len(as_text(m.get("content"))) for m in self.dialogue) <= CHAT_LIMIT:
            return
        results = [n for n, m in enumerate(self.dialogue) if m.get("role") == "tool"]
        for index in results[: max(1, len(results) // 2)]:
            content = self.dialogue[index].get("content") or ""
            if len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in self.seen_rows][:20]
                self.dialogue[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def play(self) -> list[dict[str, Any]]:
        if self.store.finished:
            return self.log
        self.read_the_request()
        self.log.append({"kind": "reading", "reading": self.reading})
        if self.opening is not None:
            self.soak(self.opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request_text
        if self.opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.render(self.opening)
        intro += ("\n\nBegin by searching for each product and reading the properties of several of its listings, so the interviewer sees which properties"
                  " these listings differ on.\n\n" + self.status_line())
        self.dialogue = [{"role": "system", "content": WALKER_GUIDE}, {"role": "user", "content": intro}]
        self.walk(WALK_TURNS, SPEND_STOP)
        if not self.store.finished:
            turned = self.wrap_up(again=True)
            if turned and not self.over() and not self.brains.no_credit and self.steps_left() > 1 and not self.late():
                self.dialogue.append({"role": "user", "content": "[Yamuna] Yamuna tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status_line()})
                self.walk(self.walk_turns + EXTRA_TURNS, SPEND_STOP + 0.01)
            if not self.store.finished:
                self.wrap_up()
        self.log.append({"kind": "end", "finished": self.store.finished, "steps": self.store.step, "cost": round(self.brains.cost, 5),
                           "calls": self.brains.calls, "seconds": round(self.elapsed(), 1)})
        return self.log

    def walk(self, turn_cap: int, spend_cap: float) -> None:
        idle = 0
        stuck = 0
        while not self.over() and self.steps_left() > 0 and self.walk_turns < turn_cap and self.brains.cost < spend_cap and not self.late():
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.dialogue.append({"role": "user", "content": summary + "\n" + self.status_line()})
                continue
            self.trim_dialogue()
            self.walk_turns += 1
            message = self.brains.talk(self.dialogue, "walker", self.menu_now())
            if message is None:
                self.silent_turns += 1
                if self.brains.no_credit or self.brains.calls >= LLM_CALLS_CAP or self.silent_turns > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (read_json(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"y{self.walk_turns}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": as_text(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            content = content if isinstance(content, str) else (as_text(content) if content else "")
            if not calls:
                idle += 1
                self.dialogue.append({"role": "assistant", "content": content or "(no call)"})
                self.dialogue.append({"role": "user", "content": "Your reply needs a tool call.\n" + self.status_line()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.jobs.get("cart"), self.jobs.get("buy"), self.jobs.get("refuse")} - {None}
            if not self.interviewed and self.jobs.get("message") and self.spare_messages() > 0 and self.steps_left() > 8 and any(c["function"]["name"] in committing for c in calls):
                self.dialogue.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.dialogue.append({"role": "tool", "tool_call_id": call["id"], "content": "Skipped, no step spent: the shopper gets the questions first."})
                summary = self.interview()
                self.dialogue.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status_line()})
                continue
            self.dialogue.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.log.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.screen_calls(calls)
            stuck = stuck + 1 if replies and all(text.startswith("Skipped, no step spent") for _, text in replies) else 0
            hint = ""
            if stuck >= 3:
                fresh = self.unread_rows(None, 5)
                hint = ("\n[Yamuna] Your last calls did not run. " + (f"Read the properties of listings not read yet: {', '.join(fresh)}." if fresh else
                        "Search with other words (the product's plain name, a requirement's value) or cart the best listing read so far."))
            for index, (cid, text) in enumerate(replies):
                self.dialogue.append({"role": "tool", "tool_call_id": cid, "content": text + (hint + "\n" + self.status_line() if index == len(replies) - 1 else "")})
            if stuck >= 6:
                self.log.append({"kind": "stuck"})
                break


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.play()
    except Exception as exc:
        episode.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.store.finished:
                episode.wrap_up()
        except Exception as again:
            episode.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.log


__all__ = ["agent_main"]
