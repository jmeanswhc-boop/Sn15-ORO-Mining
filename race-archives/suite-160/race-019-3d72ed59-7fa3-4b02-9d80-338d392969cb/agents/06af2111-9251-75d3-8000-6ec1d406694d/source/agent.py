"""Indora: buys for one shopper inside an ORO store world.

Code leads the episode. One reading turns the request into items, requirements, relations between items and limits;
the store's tools are mapped from their schemas and learned from their first results. Each item's shelf is swept,
the listings worth reading are opened, and a model judges the opened records against the requirements in batches
(in parallel when several items wait). The cart is chosen in code: every requirement met, fit and condition rules
between items held, values read from the pages a requirement points at, the lowest whole total when the cheapest
wins, the extreme measure when one does. Store notes are read as they come and the plan is redone. Questions to the
shopper go through Indora's asking engine: one item a message, its eight likeliest properties ranked by the model's
guess and then by how many listings carry them, one property a message once the shopper asks for one thing at a
time; it runs on the reading and sending helpers of Cairn v2's interviewer (the public race-17 agent). When the
store's tools cannot be mapped, a model works them in a tool loop with the same ledger.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'

EPISODE_SECONDS = 300.0   # the validator's wall-clock limit for one episode
END_MARGIN = 24.0         # seconds kept free at the end for the finishing calls
MODEL_FLOOR = 7.0         # no model request starts with less room than this
SPEND_SOFT = 0.050        # dollars: above this, judging stays on the light model
SPEND_HARD = 0.075        # dollars: above this, no more model calls except the finish
LIGHT_ROOM = 3200         # tokens: the light model reasons unseen before it answers, so its replies get this much room
CALL_CAP = 90             # model requests per episode
STORE_GAP = 0.22          # seconds between store calls
JUDGE_BATCH = 7           # records per judging request
ROW_CHARS = 260           # description characters kept for a search row
PAGE_CHARS = 700          # description characters kept for an opened page
LATE_ASK = 150.0          # no interview starts after this many seconds
CHAIN_HOPS = 6            # pages one pointer may lead through
FUNCTION_WORDS = frozenset("a an the of to in on at by for and or with from that this your have will into over under each only also just more most very".split())  # never search words

_llm = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=90, max_retries=3, retry_delay=1)

READ = """You turn a shopper's request into a work plan for a shopping agent. Read the request, then return ONE JSON object (compact, nothing else) with these keys.

"items": one entry per distinct product to buy: {"name": the kind of product in the request's words, the whole kind phrase kept ("lamp or light fixture", "mugs & cups"), "category": the store's category label for it copied exactly from the list of labels below when one fits, else "", "qty": units wanted (1 when not stated; a pair or set sold as one listing is 1), "reqs": the firm requirements on that product, each ONE string "property: constraint" with the request's own word for the property and the exact value, list, limit or exclusion, units kept ("finish: matte", "width: at least 40 cm", "colour: red or blue", "material: not plastic", "power draw: 100-200"), "max_price": the most paid for one unit when the request caps each unit, else null}.
 - A property the product phrase itself gives is a requirement too: "boots in the hiking style" is the kind "boots" with "style: hiking", and "a lamp of the floor type" is "lamp" with "type: floor"; the name keeps only the kind.
 - A value is only the words that name it: a closing verb or phrase of the sentence around it is not part of it ("get one where the shade is ivory done" gives "shade: ivory").
 - Never put price, budget, stock or quantity into reqs. Story and context (who it is for, where it will stand, what went wrong before) are not reqs unless they state a property's value outright. Words that are part of the product's name stay in "name".
 - A requirement whose value must be read from another listing is ONE req holding the whole pointer as written ("strap colour: as stated for the belt Pellwyn sells with a brass buckle, through every listing that page refers to"); list it under "refs" too.
"refs": for each such req: {"item": item number, "req": that req copied exactly, "property": the property whose value is read there, "relation": how the wanted value relates to the value read ("same", "different", "at least", "at most", "within 2 cm either side", "goes with", ...), "target": {"category": that listing's category label, "shop": the shop or seller named, "describe": the properties that pick it out, as a list of "property: value" strings, each naming its property}}.
"fits": [{"item": n, "with": m}] for each item that has to work with, fit or suit another requested item the agent picks.
"owned": [{"item": n, "code": the model or code of something the shopper already has that item n must fit}].
"conditions": requirements that hold only in some cases, never repeated in reqs: [{"item": n, "req": "property: constraint" on item n, "if_item": m, "if": "property: constraint" on item m, "applies": "when" (the req holds when item m meets "if") or "unless" (it holds except when item m meets "if")}].
"budget": the most the shopper pays for everything, the number exactly as the request writes it (never converted: "595 cents" is 595), or null. "unit": that unit as the request writes it ("cents", "US dollars"), or null.
"goal": "cheapest" when the lowest total or price wins, "most" or "least" when the highest or lowest of some measure wins, else "any". "measure": that measure's property, else "". "measure_item": its item number, else null.
"messages": messages the agent may send the shopper (integer) or null. "steps": tool calls the request allows (integer) or null.
"in_stock": true unless the request accepts items that cannot be bought now.
"process": steps the request asks for besides finding and buying the items (an approval, telling the shopper something, a check), each {"step": what to do in at most 15 words, "when": "order" (right before checking out), "decline" (right before ending without a purchase) or "choose" (before picking), "messages": messages it takes}; [] when none. Requirements, limits and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives if nothing fits, else false.
"""

TOOLS = """You map a store's tools for a shopping agent. Return ONE JSON object (compact, nothing else):
{"search": the tool that finds listings, "open": the tool that shows one listing in full (its attributes or properties; when two tools show one listing, the one promising attributes or details), "offer": a second single-listing tool showing only price and stock, or null, "compare": a tool showing several listings at once, or null, "cart": the tool that changes the cart, "cart_view": the tool that shows the cart, or null, "buy": the tool that checks out, "refuse": the tool that ends without a purchase, "ask": the tool that sends the shopper a message, or null,
"words_arg": the search parameter taking free words, or null, "category_arg": the search parameter limiting results to a category, or null, "mode_arg": a search parameter choosing between word search and category browsing, or null, "words_mode": its value for word search, or null, "browse_mode": its value for browsing, or null, "page_arg": the search parameter asking for a later page (an offset, page number or token), or null,
"buy_args": fixed arguments the checkout needs ({} when none), "refuse_args": fixed arguments that make the refuse tool end without a purchase ({} when none), "cart_add": fixed arguments that make the cart tool add ({} when none), "cart_remove": fixed arguments that make it remove ({} when none)}
Use exact tool and parameter names and exact allowed values."""

JUDGE = """You check store listings against a shopper's requirements for one product, "PRODUCT". For every listing record and every numbered requirement, answer "y" when the record shows the requirement is met, "n" when it shows it is broken, "u" when the record says nothing either way.
Reading rules:
- An attribute whose key names the requirement's property decides it, in any spelling (spaces, underscores, case, plural) and also when the key adds the product's own name (a lamp's "lamp shade colour" for "shade colour"). If such a key states a value that does not meet the requirement, answer "n" even if the title or description says otherwise. A key for a different property, even a related one (a strap colour for the body's colour, power draw for output power), decides nothing.
- When several keys name the property or one key lists several values ("red, blue"), it is "y" when any one of them meets the requirement; but for a number, a key stating one value decides over a key listing several ("8/10/12" lists the sizes it comes in or fits).
- Words match by their words: case, singular or plural, spelling variants (colour/color), a value holding the wanted word ("dark oak" meets oak) and a value naming a specific kind of the wanted one ("merino" meets wool) count. A different value that only sounds close does not ("not stated" is not "none", "roomy" is not "loose"), nor does a bare code or abbreviation.
- Numbers: convert units (1 in = 2.54 cm, 1 ft = 30.48 cm, 1 lb = 0.4536 kg, 1 oz = 28.35 g, 1 kg = 1000 g); "at least" and "at most" include the edge; a bare number under a key naming the property is in that property's usual unit. A requirement value written as a range ("100-200") is met by that same range written alike or by a single value inside it.
- "not X" is "y" when the stated value is something else, "n" when it is X, "u" when nothing is stated. "A or B" is met by either.
- When no key names the property, the title or description stating the wanted value counts as "y", stating another value as "n"; a figure that does not say what it measures counts for nothing, and a figure for something else (what the product makes once mixed or prepared, a product it goes with) does not count for the product.
- The category label must be the requested kind of product: for a record under another kind, answer "n" for every requirement.
Return ONE JSON object and nothing else: {"r": {"<listing id>": "<one letter per requirement, in order>", ...}, "m": {"<listing id>": "<the MEASURE value as stated, with its unit, or empty>"}} (give "m" only when a MEASURE is named below)."""

CHAIN = """A shopper's requirement takes its value from another listing. You read the pages the agent has opened along the pointer and say what they give. Return ONE JSON object and nothing else:
{"is_target": true when the LAST page shown is the listing being looked for (the POINTER's description: its kind, its shop or seller and every property value given) or the page the previous page sent the reader to, else false,
 "value": the value of PROPERTY that the last page itself states (in its attributes or its note) when it does not send the reader to another listing for that property, else "",
 "next": when the last page's note says to go by another listing for PROPERTY (or for the requirement), {"category": that listing's kind, "shop": its shop or seller, "describe": a list of "property: value" strings that pick it out, each naming its property (a note's "comes in a bottle" is "packaging: bottle"), taken only from the last page's own words for it (a page may send the reader back to a shop seen before, by other values)}, else null,
 "req": once "value" is known, the requirement rewritten with it as "property: constraint" with a concrete value, list, exclusion or numeric range and its unit ("within 2 cm of 30 cm" gives "between 28 and 32 cm", "different from red" gives "not red", "that value or higher" with 50 w gives "at least 50 w"; when the requirement asks for what goes with or pairs with that value and a page says what it pairs with, use the paired value), else ""}"""

PICK = """A shopping agent looks for one particular listing: the DESCRIPTION gives its kind, its shop and some of its property values. Rank the rows below by how likely each is that listing, judging from their titles and text (convert units: 1 in = 2.54 cm, 1 lb = 0.4536 kg; a row that clearly states a different value for a described property is unlikely). Return ONE JSON object and nothing else: {"ids": [row ids, most likely first, at most 6], "words": [one or two single search words its attributes would hold for a described value the rows do not show, written the way listings of this kind write it (a number in the unit they use, e.g. 450 for 1 lb when such listings write grams), most telling first; [] when the rows already show the values]}"""

NOTE = """While a shopping agent works, the store showed the text below (from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when it changes the task (budget, a requirement added, changed or dropped, a price, availability, a refused checkout), "error" when it only says a call was malformed or not accepted, "other" otherwise,
 "budget": the new total limit as a number in the request's unit, or null,
 "add": [{"item": item number, "req": "property: constraint"}], "drop": [{"item": item number, "req": the requirement no longer wanted, copied from the ledger when it is there}],
 "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone or sold out (even without naming it),
 "price_moved": true when it says a price changed, "refused": true when it says the checkout or order was not accepted,
 "summary": "one short sentence"}
A replaced requirement is a drop of the old plus an add of the new."""

GATE = """A shopping agent is about to STAGE. The request asks for the steps below along the way. From the RECORD of what the agent has done, decide whether each step due now is done for the cart as it is now (a step about the cart or order counts only when done after the cart last changed). When a step is missing, give the one store call that does it next, a tool from TOOLS with arguments; a message to the shopper is one short plain sentence about that step only (an approval asks in so many words whether the shopper approves the cart as it is now, naming its lines and total). Give no call when every step is done or nothing left can do it.
Return ONE JSON object and nothing else: {"done": true or false, "call": {"tool": "...", "args": {...}} or null}"""

CONFIRM = """A shopping agent is about to buy the listings below, one per item. Check each against its item's requirements as a store's filter for exactly that requirement would: would the listing show up when a shopper filters by it? An attribute whose key names the property decides; a value that is the wanted one in other words or a specific kind of it passes; a value that adds a word making it another kind or class (a "travel mug" is not a "mug" size, a "mini" version of a size class is not that class) fails; a number converts units and limits include the edge; a figure for something else (what it makes, what it fits) does not count; nothing stated is "u". Return ONE JSON object and nothing else: {"r": {"<listing id>": "<one letter per requirement of its item, y, n or u, in order>"}}"""

AUDIT = """A shopping agent found no cart that meets every requirement and is about to end without buying. Look at the requirements and the listing records it has read and say whether that is right. Return ONE JSON object and nothing else: {"fits": {"<item number>": [ids of records that appear to meet every requirement of that item, at most 4]}, "reason": "one sentence"}. Reading rules: an attribute whose key names a property decides it; any one of several listed values is enough; units convert; limits include the edge; a record of another kind of product never fits."""

LOOP = """You are Indora, buying for one shopper in an online store through the store's tools only. The episode earns 1 only when its finish is exactly right: a checkout holding every requested item at its quantity, each meeting every requirement in force, inside the budget, every line's page opened; or, when the store truly has nothing that fits, ending without a purchase as the request asks.
- Every tool call is one step and steps are limited. Use exact tool and argument names. Search results are summaries: open a listing's page before judging or buying it. Page on with the token or offset the store gives when you need the whole set.
- A requirement is met when the listing's attribute for that property states the value (any key spelling; one of several listed values is enough; units convert; limits include the edge). A key for another property decides nothing.
- A requirement pointing at another listing: find that listing, open its page and follow every listing its page refers you to; the last page gives the value.
- An item that must work with another: choose the two together from the codes the listings name (a model code on one, the same code in the other's text).
- When the cheapest wins, compare every fitting listing and take the lowest total.
- Questions to the shopper are handled by Indora's interviewer; never message the shopper yourself.
- End without a purchase only after real searching shows that something has no fitting listing; running short of steps proves nothing, so then buy the best cart.
- Keep your text short and call one tool or more in every reply."""


# ==================================================================================================================
# Indora's helpers: compact JSON, store results decoded into rows, schema-fitting of call arguments, text and numbers.
# ==================================================================================================================

def jdump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)


def shorten(value: Any, limit: int, depth: int = 0) -> Any:
    """The value with every text cut to `limit` characters (lists to 40 entries), for prompts and traces."""
    if depth > 8:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: shorten(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v, limit, depth + 1) for v in value[:40]]
    return value


_DECODER = json.JSONDecoder()


def parse_json(text: Any) -> dict[str, Any] | None:
    """The JSON object a model's text holds: the whole text, a fenced block, or the outermost braces."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', body, re.S)
    if fenced:
        body = fenced.group(1)
    # an answer may write its object, then a corrected one: the last of the same shape counts
    found, at = [], body.find('{')
    while at >= 0:
        try:
            got, end = _DECODER.raw_decode(body, at)
        except ValueError:
            break
        if isinstance(got, dict):
            found.append(got)
        at = body.find('{', end)
    if found:
        later = [got for got in found[1:] if set(got) & set(found[0])]
        return later[-1] if later else found[0]
    for attempt in (body, body[body.find('{'):body.rfind('}') + 1] if '{' in body else ''):
        if not attempt:
            continue
        try:
            got = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(got, dict):
            return got
    return None


def plain(text: Any) -> str:
    """A key or label as lower-case words: camelCase, snake_case and dotted names read alike."""
    words = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text or ''))
    return re.sub(r'[\s_.\-/]+', ' ', words).strip().lower()


def same_label(text: Any) -> str:
    """A category label for comparing: case, spacing and punctuation dropped ("MacBooks" = "macbooks")."""
    return re.sub(r'[^a-z0-9]+', '', str(text or '').lower())


def tokens(text: Any) -> list[str]:
    return re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', str(text or '').lower())


def stems(text: Any) -> set[str]:
    """Word stems for loose matching: plural -s, -es and British -our folded."""
    out = set()
    for word in tokens(text):
        if len(word) > 4 and re.search(r'(?:ch|sh|ss|x|z)es$', word):
            word = word[:-2]
        elif len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
            word = word[:-1]
        out.add(re.sub(r'our$', 'or', word))
    return out


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == '' or value == {}:
        return []
    return [value]


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r'\s*-?\d+(?:\.0+)?\s*', value):
        return int(float(value))
    return None


def as_number(value: Any) -> float | None:
    """A price or amount as a number: plain numbers, or the first figure of a text ("$1,299.00", "2850 cents")."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ('amount', 'value', 'price', 'cents', 'total'):
            if key in value:
                return as_number(value[key])
        found = [as_number(v) for v in value.values() if isinstance(v, (int, float, str))]
        found = [f for f in found if f is not None]
        return found[0] if len(found) == 1 else None
    hit = re.search(r'-?\d[\d,]*(?:\.\d+)?', str(value))
    if not hit:
        return None
    figure = hit.group(0)
    if re.fullmatch(r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?', figure) or ',' not in figure:
        return float(figure.replace(',', ''))
    return float(figure.replace(',', '.')) if re.fullmatch(r'-?\d+,\d{1,2}', figure) else float(figure.replace(',', ''))


def kinds_of(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    if isinstance(kind, list):
        return [k for k in kind if isinstance(k, str)]
    found = []
    for option in schema.get('anyOf') or schema.get('oneOf') or []:
        found += kinds_of(option)
    return found


def fit_args(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """`value` coerced to what `schema` accepts (numbers from digit texts, one-element lists, enum case), with each
    remaining mismatch written to `faults`; a text over its maxLength is cut at the last whole word."""
    if not isinstance(schema, dict):
        return value
    kinds = kinds_of(schema)

    def ok(v: Any) -> bool:
        checks = {'string': isinstance(v, str), 'integer': isinstance(v, int) and not isinstance(v, bool),
                  'number': isinstance(v, (int, float)) and not isinstance(v, bool), 'boolean': isinstance(v, bool),
                  'array': isinstance(v, list), 'object': isinstance(v, dict), 'null': v is None}
        return not kinds or any(checks.get(k, True) for k in kinds)
    if not ok(value):
        if 'integer' in kinds and as_int(value) is not None:
            value = as_int(value)
        elif 'number' in kinds and as_number(value) is not None and not isinstance(value, (dict, list)):
            value = as_number(value)
        elif 'string' in kinds and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        elif 'boolean' in kinds and str(value).strip().lower() in ('true', 'false'):
            value = str(value).strip().lower() == 'true'
        elif 'array' in kinds and value is not None and not isinstance(value, list):
            value = [value]
        if not ok(value):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        same = [o for o in options if isinstance(o, str) and isinstance(value, str) and plain(o) == plain(value)]
        if same:
            value = same[0]
        else:
            faults.append(f'{where} must be one of {options[:30]}')
    if isinstance(value, str):
        cap = schema.get('maxLength')
        if isinstance(cap, int) and len(value) > cap:
            cut = value[:cap]
            value = cut[:cut.rfind(' ')] if ' ' in cut[cap // 2:] else cut
        low = schema.get('minLength')
        if isinstance(low, int) and len(value) < low:
            faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} takes at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [fit_args(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for extra in [k for k in value if k not in props]:
                faults.append(f'{where}.{extra} is not an argument')
        for need in schema.get('required') or []:
            if need not in value:
                faults.append(f'{where}.{need} is required')
        value = {k: fit_args(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()
                 if schema.get('additionalProperties') is not False or k in props}
    return value


def string_args(schema: Any) -> list[str]:
    props = (schema or {}).get('properties') or {}
    return [k for k, v in props.items() if isinstance(v, dict) and 'string' in kinds_of(v) and 'enum' not in v]


def enum_args(schema: Any) -> dict[str, list[Any]]:
    props = (schema or {}).get('properties') or {}
    return {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list)}


def number_args(schema: Any) -> list[str]:
    props = (schema or {}).get('properties') or {}
    return [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(kinds_of(v))]


ID_HINT = re.compile(r'(?:^|[_ ])(?:id|sku|ref|code|item ?id|listing ?id|product ?id|pid|asin)$|^id$', re.I)
PAIR_TEXT = re.compile(r'^\s*([A-Za-z][A-Za-z0-9 _./&()\'-]{0,40}?)\s*(?::|=)\s*(.*)$', re.S)


def text_record(text: str) -> dict[str, Any] | None:
    """A text spelling a record field by field ("id: p-1 | title: Lamp | price: 120", "a=1; b=2", one field a line);
    None for prose. A field value that is itself "k=v; k=v" becomes a nested record (a listing's facts)."""
    body = text.strip()
    if len(body) < 8:
        return None
    if body[0] in '{[':
        try:
            got = json.loads(body)
            return got if isinstance(got, dict) else None
        except ValueError:
            pass
    best: list[tuple[str, str]] = []
    for joint in (' | ', '\n', ' || ', '; ', ' ; ', '|'):
        parts = [p for p in body.split(joint) if p.strip()]
        if len(parts) < 3:
            continue
        pairs = []
        for part in parts:
            hit = PAIR_TEXT.match(part)
            if hit and len(hit.group(1).split()) <= 5:
                pairs.append((hit.group(1).strip(), hit.group(2).strip()))
            elif pairs:
                pairs[-1] = (pairs[-1][0], pairs[-1][1] + joint + part.strip())
        if len(pairs) >= 3 and len(pairs) * 10 >= len(parts) * 7:
            # the first separator (in this order) that spells a record wins: a field's own text may hold more
            # "k: v" lines or "k=v;" lists than the record has fields
            best = pairs
            break
    if not best:
        return None
    record: dict[str, Any] = {}
    for key, value in best:
        if key in record:
            continue
        inner = facts_text(value)
        record[key] = inner if inner else scalar(value)
    return record


def facts_text(text: str) -> dict[str, Any] | None:
    """"k=v; k=v" or "k: v, k: v" written inside one field, as a dict; None otherwise."""
    if not isinstance(text, str) or len(text) < 5:
        return None
    for joint, mark in ((';', '='), (' | ', '='), (',', '='), (';', ':'), (' | ', ':')):
        parts = [p.strip() for p in text.split(joint) if p.strip()]
        if len(parts) < 2:
            continue
        pairs = [p.split(mark, 1) for p in parts]
        if all(len(p) == 2 and 0 < len(p[0].strip()) <= 40 and len(p[0].split()) <= 5 for p in pairs):
            return {p[0].strip(): scalar(p[1]) for p in pairs}
    return None


def scalar(text: str) -> Any:
    bare = str(text).strip().strip('"').strip()
    low = bare.lower()
    if low in ('true', 'yes'):
        return True if low == 'true' else bare
    if low == 'false':
        return False
    if low in ('none', 'null'):
        return None
    if re.fullmatch(r'-?\d+', bare) and not re.match(r'-?0\d', bare):
        return int(bare)
    if re.fullmatch(r'-?\d+\.\d+', bare):
        return float(bare)
    return bare


def unpack(value: Any, depth: int = 0) -> Any:
    """A store result in one shape: JSON texts parsed, records written as text read into objects, rows given as
    an object keyed by id turned into a list, columns-and-rows tables turned into row objects."""
    if depth > 7:
        return value
    if isinstance(value, str):
        body = value.strip()
        if body[:1] in '{[':
            try:
                return unpack(json.loads(body), depth + 1)
            except ValueError:
                pass
        if '\n' in body and len(body.splitlines()) >= 2:
            rows = [text_record(line) for line in body.splitlines() if line.strip()]
            if len(rows) >= 2 and all(rows):
                return rows
        found = text_record(body)
        return found if found else value
    if isinstance(value, list):
        return [unpack(v, depth + 1) for v in value]
    if isinstance(value, dict):
        out = {}
        heads = next((v for v in value.values() if isinstance(v, list) and len(v) >= 2 and all(isinstance(c, str) for c in v)), None)
        for key, item in value.items():
            if heads and isinstance(item, list) and item and all(isinstance(r, list) and len(r) == len(heads) for r in item):
                out[key] = [dict(zip(heads, r)) for r in item]
            elif isinstance(item, dict) and len(item) >= 2 and all(isinstance(r, dict) for r in item.values()) \
                    and all(re.search(r'\d', str(k)) and ' ' not in str(k) for k in item):
                out[key] = [dict({'id': str(k)}, **unpack(r, depth + 1)) for k, r in item.items()]
            elif isinstance(item, str) and depth >= 1 and not ID_HINT.search(plain(key)):
                facts = facts_text(item) if ('=' in item and len(item) < 3000) else None
                out[key] = facts if facts and len(facts) >= 2 else unpack(item, depth + 1) if item.strip()[:1] in '{[' else item
            else:
                out[key] = unpack(item, depth + 1)
        return out
    return value


def leaves(node: Any, depth: int = 0) -> list[str]:
    """Every text and number a value holds, at any depth."""
    if depth > 8:
        return []
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in ([str(k)] if isinstance(v, (str, int, float)) else []) + leaves(v, depth + 1)]
    if isinstance(node, list):
        return [s for v in node for s in leaves(v, depth + 1)]
    if isinstance(node, bool) or node is None:
        return []
    return [str(node)]


CODE_RE = re.compile(r'\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z][A-Za-z0-9]{1,6}(?:-?[A-Za-z0-9]{1,8})?\b')


def codes_in(text: Any) -> set[str]:
    """Model, part and series codes a text names: a letter first, letters and digits mixed ("QX1920", "KT6-7930",
    "M2"), at least two letters or four characters, upper-cased with hyphens dropped; years and plain units
    ("2024", "65w", "4k", "12v", "8gb") are not codes."""
    out = set()
    for hit in CODE_RE.findall(str(text or '')):
        code = hit.upper().replace('-', '')
        if len(code) < 4 or not re.search(r'[A-Z].*\d|\d.*[A-Z]', code):
            continue
        if re.fullmatch(r'\d+(?:[A-Z]{1,3})', code) or re.fullmatch(r'[A-Z]{1,2}\d{1,2}', code):
            continue
        if sum(c.isalpha() for c in code) < 2 and len(code) < 5:
            continue
        out.add(code)
    return out


def sentences(text: Any) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.;!?])\s+|\n+', str(text or '')) if s.strip()]


# ==============================================================================================================
# INTERVIEWER: helpers of Cairn v2's interviewer (public race-17 agent); planning and list policy: class Asker below.
# ==============================================================================================================
ASK_TURNS = 2
MESSAGE_FIELD = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
NOT_MESSAGE = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
QUIET_WORD = 4
ASK_WORDS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
MIX_LABELS = 3
LIST_SIZE = 12
LIST_MIXED = 8
LIST_STEPS = (1, 3)  # labels in the episode's first and second question, before the full list size
LIST_STEPS_MIXED = (3, 3)  # the same on several items, where each question goes to another item (simulated: 3,3,8 level or above 1,3,8 on 2-3 items)
SCARCE_MESSAGES = 2
ASK_BY = 165.0
QUESTION_BY = 210.0
LATE_BY = 190.0
LATE_STEPS = 8
TABLE_KEYS = 18
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
LOOK_ALIKES = 2  # keys set aside in one question (race 16: one or two)
PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
_FIRST_SPELLING: dict[str, str] = {}


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value


def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()


def _snake(key: str) -> str:
    return _words(key).replace(' ', '_')


def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))


def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out


def _spelled(text: str) -> frozenset[str]:
    """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
    return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in _words(text).split()))


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if kind == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True


def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not any((_fits_kind(value, k) for k in kinds))):
        for kind in kinds:
            if kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((_fits_kind(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
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
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def _soft(fallback: Any) -> Any:
    """A method that adds to the episode but is not needed for it: if it fails, the trace says so and the episode goes on."""

    def wrap(method: Any) -> Any:

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append({'kind': 'soft_crash', 'where': method.__name__, 'error': f'{type(exc).__name__}: {exc}'[:300]})
                return fallback() if callable(fallback) else fallback
        return guarded
    return wrap


def _filler(spec: dict[str, Any]) -> Any:
    """A value for a required field the agent has nothing to say in; None when the schema gives no safe one."""
    if isinstance(spec.get('enum'), list) and spec['enum']:
        return spec['enum'][0]
    for word in ('const', 'default'):
        if word in spec:
            return spec[word]
    kinds = _kinds(spec)
    if 'integer' in kinds or 'number' in kinds:
        return spec['minimum'] if isinstance(spec.get('minimum'), (int, float)) else 1
    if 'boolean' in kinds:
        return False
    return 'question'[:spec.get('maxLength') or 8] if 'string' in kinds else None


def _strings(node: Any) -> list[str]:
    """The texts a call's arguments hold, at any depth."""
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return [node] if isinstance(node, str) else []


def _says(node: Any, text: str) -> bool:
    """Whether a call's arguments carry `text` whole, however deep."""
    if isinstance(node, dict):
        return any((_says(v, text) for v in node.values()))
    if isinstance(node, list):
        return any((_says(v, text) for v in node))
    return node == text


class Interviewer:
    """Cairn v2's questions to the shopper: planning, sending and reading the replies."""

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                slot['plain'][ident] = _FIRST_SPELLING.get(str(value).strip(), text).strip().lower()
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['plain'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        twin['plain'].setdefault(ident, slot['plain'][ident])
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:TABLE_KEYS]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['plain'] if i in slot['plain'] and spec['plain'][i].strip() and slot['plain'][i].strip()]
        same = sum((1 for i in both if agree(spec['plain'][i], slot['plain'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.clock() > ASK_BY:
            return False
        opened = len(self.full)
        if self.loop_calls < ASK_TURNS and not all((self.records_of_item(n) for n in range(1, len(self.items) + 1))):
            return False
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and 'const' not in v]
        pool = [k for k in required if k in texts] or texts
        args: dict[str, Any] = {}
        if pool:
            args[max(pool, key=lambda k: (not re.search(NOT_MESSAGE, _snake(k), re.I), bool(re.search(MESSAGE_FIELD, _snake(k), re.I)), props[k].get('maxLength') or 10 ** 6))] = text
        else:
            for key, spec in props.items():
                inner = next((k for k, v in (spec.get('properties') or {}).items() if isinstance(v, dict) and 'string' in _kinds(v)), None) if isinstance(spec, dict) else None
                if isinstance(spec, dict) and 'array' in _kinds(spec):
                    args[key] = [text]
                    break
                if inner:
                    args[key] = {inner: text}
                    break
        if not args:
            return None
        for key in required:
            if key not in args:
                args[key] = _filler(props.get(key) or {})
                if args[key] is None:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        count = self.msg_limit - self.msg_used
        self.lanes = self.plan_lanes(max(count, self.list_cap()) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        groups = self.groups_for(self.msgs_free()) if self.listing else []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.clock() < QUESTION_BY):
            batch = []
            while groups and self.listing and not batch:
                batch = self.ask_group(groups.pop(0), told)
            batch = batch or self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        chain = self.early_trail()
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "The shopper was asked (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if chain:
            summary += '\n' + chain
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def late_question(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When Cairn is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)):
            return ''
        if self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges():
            return ''
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_lanes(self.list_cap() if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
        told: list[str] = []
        budget = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.budget != budget:
            told.append(f'budget now {self.budget}')
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        summary = f'Cairn asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.value_search([number])
        return summary + ('\n' + searched if searched else '')

    def groups_for(self, free: int) -> list[list[int]]:
        """Which items each of the free messages asks about when the items outnumber them: consecutive shares as even as they
        can be, so no item goes unasked. Race 15: 19 of 90 tasks wanted 4 or 5 items and allowed 3 messages, and an agent naming
        two items in one message drew a need on either of them (10 needs in 37 such messages of one agent)."""
        numbers = [n for n in self.plan_order if n in self.lanes] + [n for n in self.lanes if n not in self.plan_order]
        numbers = [n for n in numbers if self.lanes.get(n)]
        if free <= 0 or len(numbers) <= free:
            return []
        out: list[list[int]] = []
        start = 0
        for slot in range(free):
            size = (len(numbers) - start + (free - slot) - 1) // (free - slot)
            out.append(numbers[start:start + size])
            start += size
        return out

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """One message naming a few planned properties of each item of the group, the same list rules applying: the reply is
        about the first tracked property named, so each item's best keys lead its part and the first item's first label is the
        only one a reply disclosing nothing settles."""
        share = max(1, self.list_now() // len(group)) if self.list_now() < self.list_cap() else max(MIX_LABELS, self.list_cap() // len(group))
        parts: list[tuple[int, list[str]]] = []
        for number in group:
            left = [label for label in self.lanes.get(number) or [] if _words(label) not in self.lane_used.setdefault(number, set()) and (not self.told_about(number, label))]
            if left:
                parts.append((number, left[:share]))
        if not parts:
            return []
        if len(parts) == 1:
            number, labels = parts[0]
            fitted, text = self.fit_ask(number, labels, False)
            if not fitted:
                return []
            batch = self.send([(number, labels[0], text)], floor)
            if batch:
                self.finish_ask(number, fitted, False, batch, told)
            return batch
        text = ''
        while True:
            for size in range(share, 0, -1):
                pieces = [f"{', '.join(labels[:size][:-1])} or {labels[:size][-1]}" if len(labels[:size]) >= 2 else labels[0] for _, labels in parts]
                text = ', and '.join((f'what {piece} for the {self.item_said(number)}' for piece, (number, _) in zip(pieces, parts)))
                text = 'W' + text[1:].replace(' for the ', ' would you like for the ', 1) + '?'
                args = self.message_args(text)
                if args and _says(args, text):
                    parts = [(number, labels[:size]) for number, labels in parts]
                    break
            else:
                parts = parts[:-1]
                if not parts:
                    return []
                continue
            break
        first = parts[0][0]
        batch = self.send([(first, parts[0][1][0], text)], floor)
        if not batch:
            return []
        for number, _ in parts:
            self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        self.list_asks += 1
        had = [len(item.told) for item in self.items]
        heard = self.hear([(0, text, batch[0][2])], (LIGHT, HEAVY))
        heard.pop('free', None)
        self.take_heard(heard, told)
        answered = False
        for number, labels in parts:
            new = self.items[number - 1].told[had[number - 1]:]
            if new:
                answered = True
                self.lane_found[number] = self.lane_found.get(number, 0) + 1
                self.settle(number, labels, False, list(new), False)
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        if crowded and not answered:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': first})
            return batch
        if not answered:
            self.lane_used.setdefault(first, set()).add(_words(parts[0][1][0]))
        self.trace.append({'kind': 'mixed_ask', 'parts': parts, 'answered': answered})
        return batch

    def finish_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        """Read a sent question's answer into the items' needs and settle the labels it named."""
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        self.list_asks += 1
        had = [len(item.told) for item in self.items]
        heard = self.hear(batch, (LIGHT, HEAVY))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        new = [need for item, size in zip(self.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.items, had), 1):
            if len(item.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any((_count(x) == 1 for x in _items(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)

    def list_cap(self) -> int:
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

    def list_now(self) -> int:
        """Labels the next list names: few in the episode's first questions, since nothing shows yet how this shopper takes a list."""
        steps = LIST_STEPS if len(self.items) == 1 else LIST_STEPS_MIXED
        return min(steps[self.list_asks], self.list_cap()) if self.list_asks < len(steps) else self.list_cap()

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= SCARCE_MESSAGES and (not self.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                return (number, left[:self.list_now() if self.listing else 1])
        return (0, [])

    def told_about(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item: the label's words are held by the told
        property's name ("colour" after "shade colour") or hold a told name of two or more words; colour and color,
        singular and plural alike. A one-word told name ("type") hides no longer label ("bulb type")."""
        mine = _spelled(label)
        return any((_names(mine, _spelled(t.split(':', 1)[0])) for t in self.items[number - 1].told))

    def look_alikes(self, number: int, label: str) -> list[str]:
        """Up to two keys of the item a reader may take for `label` as well: keys sharing a word with it, or values written under both on the
        listings read. A word several of the item's keys carry says nothing, and neither does a value of one letter or digit."""
        mine = _words(label)
        seen: dict[str, set[str]] = {}
        for ident in self.item_ids(number):
            for key, value in (_fact_block(self.full.get(ident) or {}) or {}).items():
                seen.setdefault(_words(key), set()).update((w for w in _roots(str(value)) if len(w) > 1))
        counts: dict[str, int] = {}
        for name in seen:
            for word in set(name.split()):
                counts[word] = counts.get(word, 0) + 1
        plain = {w for w in mine.split() if counts.get(w, 0) <= 2}
        scored = []
        for order, (name, values) in enumerate(seen.items()):
            if name != mine and name:
                shared = plain & set(name.split())
                common = seen.get(mine, set()) & values
                if shared or common:
                    scored.append((-(3 * len(shared) + min(len(common), 3)), order, name))
        return [name for _, _, name in sorted(scored)[:LOOK_ALIKES]]

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.item_said(number)}?"
        alike = self.apart.get(labels[0], {}).get(number)
        if alike:
            return f"Apart from the {' and the '.join(alike)}, which {labels[0]} would you like for the {self.item_said(number)}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.ask_text(number, labels[:size], shown)
            args = self.message_args(text)
            if args and _says(args, text):
                return (labels[:size], text)
        return ([], '')

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        for _ in range(LIST_SIZE):
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            if self.follow_up:
                number, labels, shown = self.follow_up.pop(0)
            else:
                number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
                shown = False
            if not labels:
                return []
            fitted, text = self.fit_ask(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                break
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.send([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.finish_ask(number, labels, shown, batch, told)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """Mark what an answer showed about the labels its question named.

        One label: it is asked; a reply asking for a narrower question gets the base's follow-up (one product of an
        item name joining several, the property's head word, or example values). A list: the shopper answers about
        the first property of a message it tracks, and its reply to a message naming none of its properties is the
        same as to a property it does not mind. So a disclosed need settles its label and every label before it, and
        the labels after it lead the item's next list; a reply that discloses nothing settles only the first label
        (a need there would have been answered) and the others come next in reverse order, so whatever the shopper
        does not mind no longer stands in front of them; after a second such reply in a row the labels not yet named lead, since the
        order would otherwise return to the first list's. A reply asking for one question at a time ends lists for
        the episode and the list's first label is asked alone."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if crowded and self.listing:
                self.listing = False
                self.trace.append({'kind': 'lists_off', 'item': number})
            if new or not crowded:
                return
            if number not in self.one_product and self.name_parts(number):
                self.one_product.add(number)
                self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
                self.follow_up.insert(0, (number, labels, shown))
                return
            if not self.apart:
                alike = self.look_alikes(number, labels[0])
                if alike:
                    self.apart.setdefault(labels[0], {})[number] = alike
                    self.trace.append({'kind': 'apart', 'item': number, 'label': labels[0], 'from': alike})
                    self.follow_up.insert(0, (number, labels, False))
            return
        if crowded and not new:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_words(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lane_misses[number] = 0 if hit >= 0 else self.lane_misses.get(number, 0) + 1
        if self.lane_misses[number] and self.lane_misses[number] % 2 == 0:
            ahead = [label for _, _, label in sorted((abs(i - (len(ahead) - 1) / 2), i, label) for i, label in enumerate(ahead))]
            self.lanes[number] = rest + ahead
        else:
            self.lanes[number] = ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            seen.setdefault(_words(key), set()).update((w for v in values for w in _roots(v)))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(_spelled(label) & _spelled(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(seen.get(_words(label), set()) & _roots(value)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        every = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled))]
        keys = [key for key in every if not re.search(NOT_PROPERTY, key) or re.search('\\w[ _]models?\\b', key)]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)] + [key for key in every if key not in keys]

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        self.part_shelves()
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        strays = [label for label in labels if not self.owner_of(label)]
        orphans = [n for n in range(1, len(self.items) + 1) if not any((self.owner_of(label) == n and label not in self.adopted for label in labels))]
        if strays and len(orphans) == 1:
            self.adopted.update({label: orphans[0] for label in strays})
            self.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        known = self.owners_seen.get((label, len(self.items)))
        if known is not None:
            return known
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ''):
                score = 3.0
            elif words == _words(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or '') | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        self.owners_seen[(label, len(self.items))] = best
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1 or _words(label) == _words(item.shelf or ''))

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
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
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _words(label) in own:
                    return own[_words(label)]
                near = [s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))]
                return max(near) if near else 0.0
            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS]:
                if any((_words(key) == _words(other) for other in lane[:3])):
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
                if any((_words(x) in hinted[number] and all((_words(x) != _words(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [x for x in lanes[number] if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def name_parts(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and locks"),
        else []; a single product named with "and" ("salt and pepper grinder", "usb and lightning cables", "his and
        hers towels") gives []."""
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

    def item_said(self, number: int) -> str:
        """The item as a question names it. A name joining plural nouns with "and" is asked with "or" (race 13, 10
        agents: "door handles and locks" drew one-at-a-time in 81 of 155 questions; names joined with "or" in 0 of 50,
        "dresser or cabinet" disclosing a need in 10 of 28). After a one-at-a-time reply about the item, one of the
        products its name joins: the one whose head noun most of the item's own rows name outside their category label
        (with no such rows, the first)."""
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = self.category_key()
        rows = [row for row in self.rows.values() if not cat_key or self.owns(item, str(row.get(cat_key) or ''))]
        named = [_roots(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if _roots(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.item_said(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
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
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        if self.quiet_reply(sent):
            self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
            return {}
        self.hear_reads += 1
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. ({f'about item {n}' if n else 'about several items'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        """Whether an answer needs no reading: it names nothing the questions, the items, the listings' keys or their values name
        and holds no number, and it asks nothing about how to ask. Race 15: 5112 of 5160 non-need replies are such (race 14: 5050 of 5055) and
        the check misses few need replies once the keys count (race 15: 35 of 1355 by question words alone). A reading costs a
        model call, and a race model call about 5-10 s of a 300 s episode. The first answer of an episode is always read: it
        tells whether this shopper takes lists."""
        if self.hear_reads < 1:
            return False
        known: frozenset[str] = frozenset()
        for number, item in enumerate(self.items, 1):
            known |= _spelled(item.name) | _spelled(item.shelf or '')
            for need in item.musts + item.told:
                known |= _spelled(' '.join(re.findall('[A-Za-z0-9]+', need.split(':', 1)[0])))
        for label, name, _, _ in self.key_table():
            known |= _spelled(name) | _spelled(label)
        for record in self.full.values():
            for value in (_fact_block(record) or {}).values():
                known |= {w for w in _spelled(' '.join(re.findall('[A-Za-z0-9]+', str(value)))) if len(w) >= QUIET_WORD}
        for _, question, answer in sent:
            asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', question, flags=re.I)
            mine = known | _spelled(' '.join(re.findall('[A-Za-z0-9]+', asked)))
            said = _spelled(' '.join(re.findall('[A-Za-z0-9]+', answer)))
            if re.search('\\d', answer) or said & mine or re.search(ASK_WORDS, answer, re.I):
                return False
        return True

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
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def part_shelves(self) -> None:
        """An item that shares its shelf with another requested item takes the store label that is its own name, when one is seen
        and no item holds it yet."""
        cat_key = self.category_key()
        labels = sorted({str(row.get(cat_key) or '').strip() for row in self.rows.values()} - {''}) if cat_key else []
        moved = False
        for item in self.items:
            name = _roots(item.name)
            if not name or not item.shelf.strip() or not any((other is not item and _words(other.shelf) == _words(item.shelf) for other in self.items)):
                continue
            if any((other is not item and _roots(other.name) == name for other in self.items)):
                continue
            own = [label for label in labels if _roots(label) == name and _words(label) != _words(item.shelf)]
            if len(own) == 1 and not any((_words(other.shelf) == _words(own[0]) for other in self.items)):
                self.trace.append({'kind': 'shelf_parted', 'item': item.name, 'from': item.shelf, 'to': own[0]})
                item.shelf = own[0]
                moved = True
        if moved:
            self.owners_seen = {}
            self.fit_cached = None



# ==================================================================================================================
# ASKER: Indora's asking engine (Race 19 portfolio division D) on the interviewer's helpers above.
# ==================================================================================================================

ASK_LIST = 8               # properties one message names, all of one item (one each after a crowded reply)
ASK_PLAN = """You plan the questions a shopping assistant sends to a shopper. The shopper holds a few firm values they did not write down, each on one property of one requested product (a hidden need). A message reveals one only when it names that exact property, and every property a message names spends part of it, so the choice and the order of the properties decide whether every hidden need comes out.

You get the request, the items with what the request already settles for each, anything the request says to check with the shopper (ASK FIRST), and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry each key and example values.

Rank each item's properties, the likeliest hidden need first:
0. Anything listed under ASK FIRST, named by the property it is about.
1. The properties the story points at. A sentence that tells what the product must fit, hold, match, last for or look like, or how, where or by whom it will be used, without giving the value, points at the one property that settles it: name that property by the item's listed key when one fits. A sentence that only tells the occasion, the mood or who is buying points at nothing.
2. Then the item's other properties, ranked by your own judgment of how likely it is that this shopper holds a firm value on each: read the request and think about what a buyer of this kind of product chooses between. The listing counts only break ties between properties you judge equally likely, and a property no listing carries never ranks above one the listings carry. A property particular to this kind of product (its own type, format, configuration, packaging or fit, often named after the product) is far more often a firm need than one nearly every product has (its material, colour, style or pattern), which shoppers mostly leave open: rank the particular ones first unless the story points at a general one.
Skip a property the request already settles for that item (a key whose values only repeat a required value is not settled), a second name for a property already chosen, a key whose values list several purchase options at once, identifiers (part numbers, SKUs, a listing's own model code), warranty and other logistics. A size the listings state as one value of their own (a shoe size, a size class), pack sizes and counts are properties like any other; a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}, each item's picks in rank order"""


class Asker(Interviewer):
    """Indora's questions to the shopper: one item a message, its ASK_LIST likeliest properties in rank order, ranked
    by the model's guess first and the opened listings' carried counts second; one property a message once the
    shopper asks for one thing at a time."""

    def list_cap(self) -> int:
        return ASK_LIST

    def list_now(self) -> int:
        return ASK_LIST

    def groups_for(self, free: int) -> list[list[int]]:
        # one item a message: a crowd reader turns away messages on several items, and a first-tracked one answers
        # one need a message anyway
        return []

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values)
                           + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = ASK_PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for other in lanes[number]):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip() and number not in self.plan_order:
                self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_keys(number, table)
            own: dict[str, float] = {}
            read = max(1, self.records_of_item(number))
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def carried_by_some(label: str) -> bool:
                if _words(label) in own:
                    return True
                return any(_names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label)) for k in own)
            lane: list[str] = []
            for pick in lanes[number]:
                if all(_words(spelled(pick)) != _words(other) for other in lane):
                    lane.append(spelled(pick))
            if own:
                # the model's order stands, except that a pick no opened listing of the item carries goes after those
                # the listings carry
                lane = [x for x in lane if carried_by_some(x)] + [x for x in lane if not carried_by_some(x)]
            # then the item's listing keys, in carried order, as the lane's backup
            for key in keys:
                if all(_words(key) != _words(other) for other in lane):
                    lane.append(key)
            lanes[number] = [x for x in lane if not re.search(CATCH_ALL, _words(x))] + [x for x in lane if re.search(CATCH_ALL, _words(x))]
        self.trace.append({'kind': 'plan', 'engine': 'indora-d', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:10] for n, v in lanes.items()}})
        return lanes

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """What an answer showed about the properties its question named. One property: it is asked; turned away as
        too much, it is a decoy to a reader that takes one thing at a time and is never asked again (and lists end).
        A list turned away: lists end for the episode, its first property is asked alone next and the rest follow in
        turn. A list answered: a told need settles its property and every one before it (the reader answers about the
        first property it tracks); with nothing told only the first property is settled and the others keep their
        order and lead the item's lane."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_words(labels[0]))
            if crowded and self.listing:
                self.listing = False
                self.trace.append({'kind': 'lists_off', 'item': number})
            return
        if crowded and not new:
            self.listing = False
            self.trace.append({'kind': 'lists_off', 'item': number})
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        used.update(_words(label) for label in labels[:max(hit, 0) + 1])
        ahead = labels[max(hit, 0) + 1:]
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lanes[number] = ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:max(hit, 0) + 1], 'next': ahead[:4]})

# ==================================================================================================================
# INDORA: the episode (Indora's pipeline). Questions go through the Asker above.
# ==================================================================================================================

ROWS = """A shopping agent searched a store for one product, "PRODUCT", and got the summary rows below (title, price, a little text; no full attributes). For each row say whether opening its page is worth a step: "y" when the row's own words already show every requirement met, "n" only when the row clearly is another kind of product or its own words clearly state a value that breaks a requirement, "?" otherwise. When rows show a category, it decides the kind: a part, cover, refill or accessory filed under the product's own category counts as that product. A requirement a row does not mention is never a reason for "n". Return ONE JSON object and nothing else: {"y": [numbers of the rows marked "y"], "n": [numbers of the rows marked "n"]} (every other row is "?")."""

QUERY = """A shopping agent searches a store for one product and needs more search words. The store's word search QUERY_RULE. Suggest up to COUNT short searches (one or two words each, plain words a listing's title or text would carry: the product's other names, its kinds, words from the titles shown, a required value), different from those already tried, that are likely to list fitting listings not yet seen. Return ONE JSON object and nothing else: {"q": ["...", ...]}"""


class Piece:
    """One requested product as the ledger keeps it. `musts`, `told`, `dropped` and `free` are the names the
    interviewer reads; `label` is the store's category label once known."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str], cap: float | None):
        self.name = name
        self.shelf = shelf
        self.label = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.cap = cap


class Indora(Asker):

    @property
    def budget(self) -> float | None:
        return self._budget

    @budget.setter
    def budget(self, value: float | None) -> None:
        """A changed budget keeps the money unit in force: a new figure over 30 times off the one in force, which
        a hundred times more or less brings within 3 times of it, was read in the other unit (cents as dollars)."""
        old = getattr(self, '_budget', None)
        if value and old and value > 0 and old > 0 and not 1 / 30 <= value / old <= 30:
            for scaled in (value * 100, value / 100):
                if 1 / 3 <= scaled / old <= 3:
                    self.note('budget_unit', read=value, taken=scaled, before=old)
                    value = scaled
                    break
        if old is not None and value != old:
            self.changed_at = max(getattr(self, 'changed_at', 0), getattr(self, 'step', 0))
        self._budget = value

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment.get('policy_view') or {}
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        self.descs = {t['function']['name']: str(t['function'].get('description') or '') for t in self.tools}
        limit = as_int(self.view.get('max_steps'))
        self.max_steps = limit if limit and limit > 0 else 30
        self.started = time.monotonic()
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.trace: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.calls = 0
        self.cost = 0.0
        self.env_at = 0.0
        # the request as read
        self.items: list[Piece] = []
        self._budget: float | None = None
        self.unit = ''
        self.goal = 'any'
        self.measure = ''
        self.measure_item = 0
        self.fits: list[tuple[int, int]] = []
        self.owned: list[tuple[int, str]] = []
        self.conds: list[dict[str, Any]] = []
        self.pointers: list[dict[str, Any]] = []
        self.process: list[dict[str, Any]] = []
        self.quiet_decline = False
        self.in_stock = True
        self.msg_limit = 0
        self.msg_used = 0
        self.reserve = 0
        self.gate_msgs = 0
        # the store's tools as mapped
        self.roles: dict[str, str] = {}
        self.spec: dict[str, Any] = {}
        self.mapped = False
        self.keys: dict[str, str | None] = {}
        self.words_scoped: bool | None = None
        # what the store has shown
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.opened: set[str] = set()
        self.viewed: dict[str, set[str]] = {}
        self.found_by: dict[str, set[int]] = {}
        self.gone: set[str] = set()
        self.shown: dict[str, tuple[int, list[str], Any]] = {}
        self.queries: dict[int, list[str]] = {}
        self.counts: dict[int, int] = {}
        self.screens: dict[tuple[int, str], str] = {}
        self.marks: dict[tuple[int, str, str], str] = {}
        self.values: dict[tuple[int, str], str] = {}
        self.resolved: dict[str, str] = {}
        self.cart: dict[str, int] = {}
        self.cart_view: Any = None
        self.cart_changed = False
        # the step each listing's page was last read, and the last step the store may have changed lines silently
        # (the cart first filled, a budget or requirement change): a line read before it is read again at checkout
        self.page_step: dict[str, int] = {}
        self.lean = False
        self.changed_at = 0
        self.filled = False
        self.notes: list[str] = []
        self.heard_texts: set[str] = set()
        self.version = 0
        self.buys = 0
        self.refusals = 0
        self.decline_held = 0
        self.part_pages: dict[str, str] = {}
        self.exhausted: dict[int, bool] = {}
        self.browse_next: dict[int, Any] = {}
        self.looked: set[str] = set()
        # word searches: the rows each query showed over its pages, the next page, whether it was read to its end
        self.query_rows: dict[str, set[str]] = {}
        self.hit_only: set[str] = set()
        self.learned: set[str] = set()
        self.query_next: dict[str, Any] = {}
        self.query_done: dict[str, bool] = {}
        self.query_count: dict[str, int] = {}
        self.noun_bad = False
        self.scaled = False
        self.confirmed: set[str] = set()
        self.nouns: dict[int, str] = {}
        # Cairn's interviewer reads and keeps these
        self.brief: dict[str, Any] = {'ask_first': []}
        self.refs: list[tuple[int, str]] = []
        self.asked = False
        self.loop_calls = 0
        self.listing = True
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.lane_misses: dict[int, int] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.list_asks = 0
        self.apart: dict[str, dict[int, list[str]]] = {}
        self.one_product: set[int] = set()
        self.plan_order: list[int] = []
        self.hear_reads = 0
        self.late_asked = False
        self.adopted: dict[str, int] = {}
        self.owners_seen: dict[tuple[str, int], int] = {}
        self.fit_cached: Any = None
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.queued: list[str] = []

    # ---------------------------------------------------------------- clock, limits, money

    @property
    def budget(self) -> float | None:
        return self._budget

    @budget.setter
    def budget(self, value: float | None) -> None:
        """A new limit about a hundred times off the one in force is the same money in the other unit (an answer
        saying "$48.98" against a request in cents); it is kept in the unit in force."""
        old = getattr(self, '_budget', None)
        if old and value and value > 0:
            if value * 20 < old and 0.3 <= value * 100 / old <= 4:
                value = value * 100
            elif value > old * 20 and 0.3 <= value / 100 / old <= 4:
                value = value / 100
        self._budget = value

    def left(self) -> int:
        return self.max_steps - self.step

    def clock(self) -> float:
        return time.monotonic() - self.started

    def room(self) -> float:
        """Seconds a model request may still use before the finish must start."""
        return EPISODE_SECONDS - END_MARGIN - self.clock()

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return jdump(sorted(self.cart.items()))

    def judges(self, limit: float = SPEND_SOFT) -> list[str]:
        if self.cost <= limit:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def note(self, kind: str, **data: Any) -> None:
        data['kind'] = kind
        data.setdefault('at', round(self.clock(), 1))
        data.setdefault('step', self.step)
        with self.lock:
            self.trace.append(data)

    # ---------------------------------------------------------------- the model

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None = None, effort: str | None = None) -> tuple[dict[str, Any] | None, Any]:
        body: dict[str, Any] = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        if effort:
            body['reasoning'] = {'effort': effort}
        room = self.room()
        inference = _llm
        if room < 60:
            seconds = max(5, int(room) // 5 * 5)
            with self.lock:
                inference = _llm_short.setdefault(seconds, ProxyClient(timeout=seconds, max_retries=1, retry_delay=1))
        result = inference.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                with self.lock:
                    self.cost += float(usage['cost'])
            choice = data['choices'][0] or {}
            message = choice.get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return message, 200
            return None, 'length' if choice.get('finish_reason') == 'length' else 0
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = as_int(data['error'].get('code')) or status
        return None, status

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """One model answer, trying each model in turn; waits out a short outage, gives up when the clock or the
        call cap says so."""
        for model in models:
            if model.startswith(HEAVY) and self.cost > SPEND_HARD:
                continue
            effort = self.effort_for(model, messages)
            room = max(max_tokens, LIGHT_ROOM) if model.startswith(LIGHT) else max_tokens
            for attempt in range(4):
                with self.lock:
                    if self.calls >= CALL_CAP or self.no_credit:
                        return None
                    self.calls += 1
                if self.room() < MODEL_FLOOR:
                    self.note('model_skipped', room=round(self.room(), 1))
                    return None
                began = time.monotonic()
                message, status = self.call_model(model, messages, room, tools, effort)
                if message is not None:
                    return message
                self.note('model_fail', model=model, status=status, secs=round(time.monotonic() - began, 1))
                if status == 402:
                    self.no_credit = True
                    return None
                if status == 'length' and model.startswith(LIGHT) and room < 2 * LIGHT_ROOM:
                    room = 2 * LIGHT_ROOM
                    continue
                if status == 0:
                    break
                if status in (429, 408, 425, None) or (isinstance(status, int) and status >= 500):
                    pause = (2, 5, 9)[min(attempt, 2)] * (0.8 + 0.4 * random.random())
                    if self.room() - pause < MODEL_FLOOR + 4:
                        break
                    time.sleep(pause)
                    continue
                break
        return None

    @staticmethod
    def effort_for(model: str, messages: list[dict[str, Any]]) -> str | None:
        """Reasoning effort per request: low for the heavy model's record checks, its check before buying and its
        question plan (no unseen reasoning; a replay of recorded requests gave matching answers at roughly half
        the cost). Its brief, its audit before a decline and its pointer readings keep their own reasoning, and so
        does every light-model request."""
        if not model.startswith(HEAVY) or len(messages) != 1:
            return None
        text = str(messages[0].get('content') or '')
        for opening in (JUDGE.split('PRODUCT')[0], CONFIRM[:80], PLAN[:80], ASK_PLAN[:80]):
            if text.startswith(opening):
                return 'low'
        return None

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = parse_json((message or {}).get('content'))
            if got is not None:
                return got
            if self.no_credit or self.calls >= CALL_CAP or self.room() < MODEL_FLOOR:
                return None
        return None

    def together(self, jobs: list[Any]) -> list[Any]:
        """Run model-only jobs (no store calls) side by side; their results in order."""
        if len(jobs) <= 1:
            return [job() for job in jobs]
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
            futures = [pool.submit(job) for job in jobs]
            out = []
            for future in futures:
                try:
                    out.append(future.result())
                except Exception as exc:
                    self.note('job_crash', error=f'{type(exc).__name__}: {exc}'[:300])
                    out.append(None)
            return out

    # ---------------------------------------------------------------- the store

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One store call. Its result comes back decoded (except a shopper's reply) and remembered; none past the
        step limit (a call over it voids the episode)."""
        if self.finished or self.broke or self.left() <= 0:
            return None
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        rest = self.env_at + STORE_GAP - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        for _ in range(4):
            self.env_at = time.monotonic()
            result = _env.post_verbose('/environment/call', payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                if name != self.roles.get('ask'):
                    data['observation'] = unpack(data.get('observation'))
                    self.absorb(name, args, data['observation'])
                if name == self.roles.get('cart') and not self.filled:
                    # stores fire some of their changes on the first turn with a filled cart, often without a word
                    self.filled = True
                    self.changed_at = max(self.changed_at, self.step)
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.note('store_fail', tool=name, status=status, body=body[:200])
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low or 'over' in low:
                    self.finished = True
                    return None
                expected = re.search(r'expected (?:step )?(\d+)', low)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low or 'already' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
            time.sleep(0.6)
        return None

    def call(self, name: str | None, args: dict[str, Any]) -> Any:
        """A store call with its arguments fitted to the schema; the observation, or None when it could not run."""
        if not name or name not in self.schemas or self.finished or self.left() <= 0:
            return None
        faults: list[str] = []
        fitted = fit_args(dict(args), self.schemas[name], 'args', faults)
        if faults:
            self.note('bad_args', tool=name, faults=faults[:4], args=shorten(args, 80))
            return None
        data = self.act(name, fitted)
        if data is None:
            return None
        observation = data.get('observation')
        self.done_log.append(f'step {self.step}: {name} {jdump(fitted)[:200]} -> {jdump(shorten(observation, 80))[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        return observation

    # ---------------------------------------------------------------- what results hold

    def id_of(self, record: Any) -> str | None:
        if not isinstance(record, dict) or len(record) < 2:
            return None
        key = self.keys.get('id')
        if key and isinstance(record.get(key), (str, int)) and not isinstance(record.get(key), bool):
            return str(record[key]).strip()
        for name, value in record.items():
            if ID_HINT.search(plain(name)) and isinstance(value, (str, int)) and not isinstance(value, bool):
                text = str(value).strip()
                if 2 < len(text) <= 64 and ' ' not in text:
                    return text
        return None

    def learn_keys(self, rows: list[dict[str, Any]]) -> None:
        """Which row fields hold a listing's id, price, stock, category label, title and seller: by their names first,
        then by their values (the category key's values are the store's labels)."""
        if not rows:
            return
        sample = rows[:12]
        names = {k for r in sample for k in r}

        def pick(pattern: str, test: Any = None) -> str | None:
            hits = [k for k in names if re.search(pattern, plain(k))]
            hits = [k for k in hits if test is None or all(test(r.get(k)) for r in sample if k in r)]
            hits.sort(key=lambda k: (-sum(1 for r in sample if k in r), len(k)))
            return hits[0] if hits else None
        if not self.keys.get('id'):
            ids = [k for k in names if ID_HINT.search(plain(k)) and all(isinstance(r.get(k), (str, int)) and ' ' not in str(r.get(k)).strip() for r in sample if k in r)]
            ids = [k for k in ids if len({str(r.get(k)) for r in sample}) == len(sample)]
            ids.sort(key=lambda k: (plain(k) != 'id', len(k)))
            if ids:
                self.keys['id'] = ids[0]
        if not self.keys.get('price'):
            self.keys['price'] = pick(r'\b(?:price|cost|amount|list price|sale price|unit price)\b', lambda v: as_number(v) is not None)
        if not self.keys.get('stock'):
            self.keys['stock'] = pick(r'\b(?:stock|in stock|on hand|available|availability|inventory|buyable|sold out)\b')
        if not self.keys.get('cat'):
            labels = {plain(x) for x in self.category_labels()}
            by_value = [k for k in names if labels and all(isinstance(r.get(k), str) and plain(r.get(k)) in labels for r in sample if k in r) and any(k in r for r in sample)]
            self.keys['cat'] = by_value[0] if by_value else pick(r'\b(?:category|categories|shelf|department|section|aisle|group|kind|type of product|product type)\b', lambda v: isinstance(v, str) and len(v) <= 80)
        if not self.keys.get('title'):
            self.keys['title'] = pick(r'\b(?:title|name|headline|label)\b', lambda v: isinstance(v, str)) or None
        if not self.keys.get('shop'):
            self.keys['shop'] = pick(r'\b(?:merchant|seller|shop|store|vendor|sold by|retailer|dealer|outlet)\b', lambda v: isinstance(v, str))

    def row_lists(self, node: Any, depth: int = 0) -> list[list[dict[str, Any]]]:
        if depth > 5:
            return []
        if isinstance(node, list):
            dicts = [r for r in node if isinstance(r, dict)]
            if dicts and len(dicts) * 2 >= len(node):
                self.learn_keys(dicts)
                if all(self.id_of(r) for r in dicts):
                    return [dicts]
            return [found for r in node for found in self.row_lists(r, depth + 1)]
        if isinstance(node, dict):
            return [found for v in node.values() for found in self.row_lists(v, depth + 1)]
        return []

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Remember every listing a result shows; a page tool's single record also counts as read in full."""
        pages = ({self.roles.get(r) for r in ('open', 'offer', 'compare')} | set(self.other_reads())) - {None}
        records: list[dict[str, Any]] = []
        if isinstance(observation, dict) and name in pages:
            inner = observation if self.id_of(observation) else next((v for v in observation.values() if isinstance(v, dict) and self.id_of(v)), None)
            if inner is not None:
                self.learn_keys([inner])
                records = [inner]
        if not records:
            for rows in self.row_lists(observation):
                records += rows
        for record in records:
            ident = self.id_of(record)
            if not ident:
                continue
            old = self.rows.get(ident)
            if old is None:
                self.rows[ident] = {}
                self.seen_order.append(ident)
            self.rows[ident].update(record)
            if name in pages:
                self.full.setdefault(ident, {}).update(self.rows[ident])
                self.full[ident].update(record)
                self.viewed.setdefault(ident, set()).add(name)
                self.page_step[ident] = self.step
                if name == self.roles.get('open') or name == self.roles.get('offer'):
                    self.opened.add(ident)
            elif ident in self.full:
                self.full[ident].update(record)
            if self.keys.get('stock') and self.out_of_stock(record.get(self.keys['stock'])):
                self.gone.add(ident)

    @staticmethod
    def out_of_stock(value: Any) -> bool:
        if value is False:
            return True
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value <= 0
        if isinstance(value, str):
            return bool(re.search(r'^(?:no|none|false|0|n)$|out of stock|sold out|unavailable|not available|backorder', value.strip().lower()))
        return False

    def price(self, ident: str) -> float | None:
        key = self.keys.get('price')
        record = self.full.get(ident) or self.rows.get(ident) or {}
        value = record.get(key) if key else None
        if value is None:
            value = next((v for k, v in record.items() if re.search(r'\bprice\b', plain(k)) and as_number(v) is not None), None)
        return as_number(value)

    def label(self, ident: str) -> str:
        key = self.keys.get('cat')
        record = self.rows.get(ident) or self.full.get(ident) or {}
        return str(record.get(key) or '') if key else ''

    def title(self, ident: str) -> str:
        key = self.keys.get('title')
        record = self.rows.get(ident) or {}
        return str(record.get(key) or '') if key else ''

    def shop(self, ident: str) -> str:
        key = self.keys.get('shop')
        record = self.full.get(ident) or self.rows.get(ident) or {}
        return str(record.get(key) or '') if key else ''

    def text(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        return ' '.join(leaves(record))

    def buyable(self, ident: str) -> bool:
        return ident not in self.gone

    def category_key(self) -> str | None:
        return self.keys.get('cat')

    def category_labels(self) -> list[str]:
        arg = self.spec.get('category_arg')
        tool = self.roles.get('search')
        if arg and tool:
            options = ((self.schemas.get(tool) or {}).get('properties') or {}).get(arg, {}).get('enum')
            if isinstance(options, list):
                return [str(o) for o in options]
        return []

    def owner(self, ident: str) -> list[int]:
        """The items a listing could be bought for: by its category label, else by the searches that showed it."""
        label = same_label(self.label(ident))
        if label:
            own = [n for n, item in enumerate(self.items, 1) if item.label and same_label(item.label) == label]
            if own:
                return own
            if any(item.label for item in self.items):
                # a row of no labelled item's category may still be one of the items without a label
                return sorted(n for n in self.found_by.get(ident, set()) if not self.items[n - 1].label)
        return sorted(self.found_by.get(ident, set()))

    def pool(self, number: int) -> list[str]:
        return [i for i in self.seen_order if number in self.owner(i)]

    # ---------------------------------------------------------------- reading the request and the tools

    def tool_lines(self) -> str:
        return '\n'.join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {jdump(t['function'].get('parameters') or {})}" for t in self.tools)

    def label_options(self) -> list[str]:
        """Every value of a long enum in any tool: the store's category labels when it has a list of them."""
        found: list[str] = []
        for schema in self.schemas.values():
            for key, options in enum_args(schema).items():
                if len(options) >= 4 and all(isinstance(o, str) for o in options):
                    found += [o for o in options if o not in found]
        return found

    def setup(self) -> None:
        labels = self.label_options()
        read = READ + ('\nCATEGORY LABELS: ' + jdump(labels) if labels else '\nThe store lists no category labels.') + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            read += '\n\nTHE STORE SHOWS FIRST:\n' + jdump(shorten(self.view.get('opening'), 600))[:2000]
        mapping = TOOLS + '\n\nTOOLS:\n' + self.tool_lines()
        brief, tools = self.together([lambda: self.ask_json(read, [HEAVY, LIGHT], 4000), lambda: self.ask_json(mapping, [LIGHT, HEAVY], 1500)])
        if not brief or not as_list(brief.get('items')):
            brief = self.ask_json(read, [LIGHT, HEAVY], 4000) or brief or {}
        loose = [str(m) for e in as_list((brief or {}).get('items')) if isinstance(e, dict) for m in as_list(e.get('reqs'))
                 if re.search(r'\b(?:listing|page)s?\b', str(m).partition(':')[2], re.I)]
        if loose and not as_list((brief or {}).get('refs')):
            # a value said to come from another listing, read without its "refs" entry: once more, pointing at it
            again = self.ask_json(read + '\n\nA first reading wrote ' + jdump(loose[:3]) + ' as requirements with an empty "refs": each value read from another listing needs its "refs" entry, with its target (category, shop, describe) taken from the request.', [HEAVY, LIGHT], 4000)
            if again and as_list(again.get('items')) and as_list(again.get('refs')):
                brief = again
        self.note('brief', brief=shorten(brief, 200), tools=tools)
        self.map_tools(tools or {})
        self.read_brief(brief or {})

    def map_tools(self, got: dict[str, Any]) -> None:
        roles = {}
        for role in ('search', 'open', 'offer', 'compare', 'cart', 'cart_view', 'buy', 'refuse', 'ask'):
            name = got.get(role)
            if isinstance(name, str) and name in self.schemas:
                roles[role] = name
        guess = self.guess_roles()
        for role, name in guess.items():
            roles.setdefault(role, name)
        if roles.get('offer') == roles.get('open'):
            roles.pop('offer')
        self.roles = roles
        tool = roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        spec: dict[str, Any] = {}
        for key in ('words_arg', 'category_arg', 'mode_arg', 'page_arg'):
            value = got.get(key)
            spec[key] = value if isinstance(value, str) and value in props else None
        texts = string_args(self.schemas.get(tool or ''))
        enums = enum_args(self.schemas.get(tool or ''))
        if not spec['words_arg']:
            spec['words_arg'] = next((k for k in texts if re.search(r'\b(?:q|query|words?|text|keywords?|terms?|search|phrase|needle)\b', plain(k))), texts[0] if len(texts) == 1 else None)
        if not spec['category_arg']:
            spec['category_arg'] = next((k for k, o in enums.items() if len(o) >= 4), None) or next((k for k in texts if k != spec['words_arg'] and re.search(r'\b(?:category|shelf|department|section|aisle|group|label)\b', plain(k))), None)
        if not spec['page_arg']:
            spec['page_arg'] = next((k for k in props if k not in (spec['words_arg'], spec['category_arg']) and re.search(r'\b(?:page|offset|start|begin|skip|cursor|token|after|from|next)\b', plain(k))), None)
        if not spec['page_arg']:
            # a search's only optional whole number starting at 0 is where its page starts (a size or limit word
            # names a page length instead)
            numbers = [k for k, v in props.items() if isinstance(v, dict) and 'integer' in kinds_of(v) and k not in (schema.get('required') or [])
                       and v.get('minimum', 0) == 0 and not re.search(r'\b(?:size|limit|count|max|per|num|number|results)\b', plain(k))]
            if len(numbers) == 1:
                spec['page_arg'] = numbers[0]
        if not spec['mode_arg']:
            spec['mode_arg'] = next((k for k, o in enums.items() if 2 <= len(o) <= 4 and k != spec['category_arg']), None)
        mode_options = enums.get(spec['mode_arg'] or '', [])
        words_mode = got.get('words_mode') if got.get('words_mode') in mode_options else None
        browse_mode = got.get('browse_mode') if got.get('browse_mode') in mode_options else None
        if spec['mode_arg'] and mode_options:
            words_mode = words_mode or next((o for o in mode_options if re.search(r'word|keyword|text|query|search|term', str(o), re.I)), None)
            browse_mode = browse_mode or next((o for o in mode_options if re.search(r'brows|filter|categor|list|shelf|facet', str(o), re.I) and o != words_mode), None)
            if words_mode is None and browse_mode is not None and len(mode_options) == 2:
                words_mode = next(o for o in mode_options if o != browse_mode)
            if browse_mode is None and words_mode is not None and len(mode_options) == 2:
                browse_mode = next(o for o in mode_options if o != words_mode)
        spec['words_mode'], spec['browse_mode'] = words_mode, browse_mode
        for key in ('buy_args', 'refuse_args', 'cart_add', 'cart_remove'):
            spec[key] = got.get(key) if isinstance(got.get(key), dict) else {}
        if roles.get('buy') and roles.get('buy') == roles.get('refuse'):
            # one tool ends both ways: in each two-way choice the endings take different options
            for key, options in enum_args(self.schemas.get(roles['buy'])).items():
                buy, refuse = spec['buy_args'].get(key), spec['refuse_args'].get(key)
                if len(options) != 2 or (buy in options and refuse in options and buy != refuse):
                    continue
                if buy in options:
                    spec['refuse_args'][key] = next(o for o in options if o != buy)
                elif refuse in options:
                    spec['buy_args'][key] = next(o for o in options if o != refuse)
        self.spec = spec
        self.mapped = bool(roles.get('search') and roles.get('cart') and roles.get('buy') and (roles.get('open') or roles.get('compare')))
        self.note('tools', roles=roles, spec=spec, mapped=self.mapped)

    def guess_roles(self) -> dict[str, str]:
        """Roles read from tool descriptions and schemas, used where the mapping left a gap."""
        out: dict[str, str] = {}
        for name, schema in self.schemas.items():
            words = plain(name) + ' ' + self.descs.get(name, '').lower()
            props = (schema or {}).get('properties') or {}
            arrays = [k for k, v in props.items() if isinstance(v, dict) and 'array' in kinds_of(v)]
            texts = string_args(schema)
            enums = enum_args(schema)
            if re.search(r'search|find|locate|browse|look ?up|query|seek', words) and (texts or enums):
                out.setdefault('search', name)
            elif re.search(r'check ?out|purchase|place (?:the )?order|buy|finali[sz]e|complete (?:the )?order', words) and not re.search(r'no purchase|without|decline|refuse|pass', words):
                out.setdefault('buy', name)
            elif re.search(r'no purchase|without (?:a )?(?:purchase|buying)|decline|refuse|give up|stand down|abstain|pass', words):
                out.setdefault('refuse', name)
            elif re.search(r'shopper|buyer|customer|question|message|ask|chat', words) and texts:
                out.setdefault('ask', name)
            elif re.search(r'cart|basket|bag|trolley', words) and (texts or arrays) and (enums or number_args(schema) or arrays):
                out.setdefault('cart', name)
            elif re.search(r'cart|basket|bag|trolley', words) and not props:
                out.setdefault('cart_view', name)
            elif arrays and re.search(r'compare|several|side|many|batch|line ?up|weigh', words):
                out.setdefault('compare', name)
            elif len(texts) == 1 and re.search(r'view|open|show|detail|page|item|listing|inspect|full|read', words):
                out.setdefault('open', name)
        return out

    def read_brief(self, brief: dict[str, Any]) -> None:
        labels = self.category_labels() or self.label_options()
        for entry in as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            qty = as_int(entry.get('qty')) or 1
            musts = [str(m).strip() for m in as_list(entry.get('reqs')) if str(m).strip()]
            cap = as_number(entry.get('max_price')) if entry.get('max_price') not in (None, '', 0) else None
            name = str(entry.get('name') or 'the product').strip()
            shelf = self.match_label(str(entry.get('category') or ''), name, labels)
            self.items.append(Piece(name, shelf, max(1, qty), musts, cap))
        if not self.items:
            self.items = [Piece('the requested product', '', 1, [], None)]
        count = len(self.items)

        def number(value: Any) -> int:
            n = as_int(value) or 0
            return n if 1 <= n <= count else 0
        budget = as_number(brief.get('budget')) if brief.get('budget') not in (None, '', 0) else None
        self.budget = budget if budget and budget > 0 else None
        self.unit = str(brief.get('unit') or '')
        goal = str(brief.get('goal') or 'any').lower()
        self.goal = goal if goal in ('cheapest', 'most', 'least') else 'any'
        self.measure = str(brief.get('measure') or '')
        self.measure_item = number(brief.get('measure_item')) or (1 if self.goal in ('most', 'least') else 0)
        for pair in as_list(brief.get('fits')):
            if isinstance(pair, dict) and number(pair.get('item')) and number(pair.get('with')) and number(pair.get('item')) != number(pair.get('with')):
                self.fits.append((number(pair['item']), number(pair['with'])))
        for entry in as_list(brief.get('owned')):
            if isinstance(entry, dict) and number(entry.get('item')) and str(entry.get('code') or '').strip():
                self.owned.append((number(entry['item']), str(entry['code']).strip()))
        for entry in as_list(brief.get('conditions')):
            if isinstance(entry, dict) and number(entry.get('item')) and number(entry.get('if_item')) and entry.get('req') and entry.get('if'):
                self.conds.append({'item': number(entry['item']), 'req': str(entry['req']).strip(), 'if_item': number(entry['if_item']),
                                   'if': str(entry['if']).strip(), 'applies': 'unless' if str(entry.get('applies')).lower() == 'unless' else 'when'})
        for entry in as_list(brief.get('refs')):
            if not isinstance(entry, dict) or not number(entry.get('item')):
                continue
            item = self.items[number(entry['item']) - 1]
            req = str(entry.get('req') or '').strip()
            if req not in item.musts:
                prop = plain(str(entry.get('property') or req.split(':')[0]))
                req = next((m for m in item.musts if plain(m.split(':')[0]) == prop), req)
            if not req:
                continue
            if req not in item.musts:
                item.musts.append(req)
            target = entry.get('target') if isinstance(entry.get('target'), dict) else {}
            self.pointers.append({'item': number(entry['item']), 'req': req, 'property': str(entry.get('property') or req.split(':')[0]).strip(),
                                  'relation': str(entry.get('relation') or 'same'), 'target': target, 'pages': [], 'tries': 0, 'done': False})
            self.refs.append((number(entry['item']), req))
        limit = as_int(brief.get('messages'))
        self.msg_limit = limit if limit is not None and limit >= 0 else (3 if self.roles.get('ask') else 0)
        if not self.roles.get('ask'):
            self.msg_limit = 0
        steps = as_int(brief.get('steps'))
        if steps and 8 <= steps < self.max_steps:
            self.max_steps = steps
        self.in_stock = brief.get('in_stock') is not False
        self.quiet_decline = brief.get('quiet_decline') is True
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or 'order').lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('order', 'decline', 'choose') else 'order',
                                     'messages': max(0, as_int(entry.get('messages')) or 0)})
        talk = [p['messages'] for p in self.process if p['messages']]
        self.reserve = min(max(talk), self.msg_limit) if talk else 0

    @staticmethod
    def match_label(given: str, name: str, labels: list[str]) -> str:
        """The store label for an item: the one given when the store has it, else the label sharing most word stems
        with the given label and the item's name; the given text when the store lists none."""
        if not labels:
            return given
        for option in labels:
            if plain(option) == plain(given):
                return option
        wanted = stems(given) | stems(name)
        scored = sorted(((len(wanted & stems(o)) / max(1, len(stems(o))), o) for o in labels), reverse=True)
        return scored[0][1] if scored and scored[0][0] > 0 else given

    # ---------------------------------------------------------------- requirements in force

    def pending_pointer(self, req: str) -> bool:
        return any(p['req'] == req and req not in self.resolved for p in self.pointers)

    def base_reqs(self, number: int) -> list[str]:
        item = self.items[number - 1]
        out = []
        for need in item.musts + item.told:
            if need in item.dropped or self.pending_pointer(need):
                continue
            text = self.resolved.get(need, need)
            if text and text not in out:
                out.append(text)
        return out

    def judged_reqs(self, number: int) -> list[str]:
        out = self.base_reqs(number)
        out += [c['req'] for c in self.conds if c['item'] == number and c['req'] not in out]
        out += [c['if'] for c in self.conds if c['if_item'] == number and c['if'] not in out]
        return out

    def verdict(self, number: int, ident: str) -> str:
        """'y' when every requirement in force is met, 'n' when one is broken, 'u' while one is unread."""
        letters = [self.marks.get((number, ident, r), '?') for r in self.base_reqs(number)]
        if 'n' in letters:
            return 'n'
        if '?' in letters or 'u' in letters:
            return 'u' if '?' not in letters else '?'
        return 'y'

    # ---------------------------------------------------------------- searching

    def search_args(self, number: int, words: str | None, start: Any = None, label: str | None = None) -> dict[str, Any] | None:
        spec = self.spec
        shelf = label if label is not None else (self.items[number - 1].label if number else '')
        args: dict[str, Any] = {}
        if words is None and not (spec.get('category_arg') and shelf):
            words = self.noun(number) if number else (shelf or None)
            if not words:
                return None
        if spec.get('mode_arg'):
            mode = spec.get('words_mode') if words else spec.get('browse_mode')
            if mode is None:
                return None
            args[spec['mode_arg']] = mode
        if words:
            if not spec.get('words_arg'):
                return None
            args[spec['words_arg']] = words
        if spec.get('category_arg') and shelf:
            options = enum_args(self.schemas.get(self.roles['search'])).get(spec['category_arg'])
            if not options or shelf in options:
                args[spec['category_arg']] = shelf
        if start is not None and spec.get('page_arg'):
            args[spec['page_arg']] = start
        schema = self.schemas.get(self.roles['search']) or {}
        props = schema.get('properties') or {}
        for need in schema.get('required') or []:
            if need not in args:
                options = (props.get(need) or {}).get('enum')
                if isinstance(options, list) and options:
                    args[need] = options[0]
                elif 'default' in (props.get(need) or {}):
                    args[need] = props[need]['default']
        faults: list[str] = []
        fitted = fit_args(args, schema, 'args', faults)
        return None if faults else fitted

    def noun(self, number: int) -> str:
        """The item's search noun: of the words its name and store label hold, the one most of its rows' titles
        carry (else the name's last word); kept once a shelf's rows chose it, so its searches stay the same."""
        if number in self.nouns:
            return self.nouns[number]
        word = self.pick_noun(number)
        if len(self.pool(number)) >= 6:
            self.nouns[number] = word
        return word

    def pick_noun(self, number: int) -> str:
        item = self.items[number - 1]
        words = [w for w in re.findall(r"[a-z][a-z'-]{2,}", (item.name + ' ' + (item.label or '')).lower()) if w not in ('and', 'the', 'for', 'with', 'or')]
        if not words:
            return 'product'
        titles = [stems(self.title(i)) for i in self.pool(number)]
        if titles:
            best = max(dict.fromkeys(words), key=lambda w: (sum(1 for s in titles if stems(w) & s), -words.index(w)))
            if any(stems(best) & s for s in titles):
                return best
        return re.findall(r"[a-z][a-z'-]{2,}", item.name.lower())[-1] if re.findall(r"[a-z][a-z'-]{2,}", item.name.lower()) else words[-1]

    def query(self, number: int, value: str) -> str:
        """A two-word search for a value: the value alone when it is two words or the search keeps to the item's
        category, else the value and the item's noun."""
        value = ' '.join(value.split()[:2])
        if self.noun_bad or len(value.split()) > 1:
            return value
        noun = self.noun(number)
        return value if not noun or stems(noun) & stems(value) else f'{value} {noun}'

    def paging(self, observation: Any, args: dict[str, Any], shown: int) -> tuple[int | None, Any]:
        """(how many listings matched in all, the value asking for the next page or None)."""
        count = None
        follow = None
        if isinstance(observation, dict):
            for key, value in observation.items():
                name = plain(key)
                if isinstance(value, bool):
                    continue
                if count is None and isinstance(value, int) and re.search(r'\b(?:count|total|hits|matches|found|results|num)\b', name) and not re.search(r'page|shown|size|limit', name):
                    count = value
                if re.search(r'\b(?:next|token|cursor|continu|more|after|offset)\b', name) and isinstance(value, (str, int)) and value != '':
                    follow = value
        page_arg = self.spec.get('page_arg')
        if follow is None and page_arg and count is not None and shown:
            spec = ((self.schemas.get(self.roles['search']) or {}).get('properties') or {}).get(page_arg) or {}
            if 'integer' in kinds_of(spec):
                now = as_int(args.get(page_arg)) or 0
                follow = now + 1 if re.search(r'page', plain(page_arg)) else now + shown
        if follow is not None and page_arg:
            spec = ((self.schemas.get(self.roles['search']) or {}).get('properties') or {}).get(page_arg) or {}
            top = spec.get('maximum')
            if isinstance(top, (int, float)) and isinstance(follow, (int, float)) and follow > top:
                follow = None
            if count is not None and isinstance(follow, int) and not re.search(r'page', plain(page_arg)) and follow >= count:
                follow = None
        return count, follow

    def search(self, number: int, words: str | None, start: Any = None, label: str | None = None) -> tuple[list[str], int | None, Any] | None:
        """One search for an item (a category browse when `words` is None; item 0 with a `label` searches for no
        item); (ids shown, total, next page)."""
        args = self.search_args(number, words, start, label)
        if args is None:
            return None
        mark = jdump(args)
        if mark in self.shown and self.shown[mark][0] == self.version:
            _, ids, more = self.shown[mark]
            return ids, more[0], more[1]
        observation = self.call(self.roles['search'], args)
        if observation is None:
            return None
        ids = [self.id_of(r) for rows in self.row_lists(observation) for r in rows]
        ids = [i for i in ids if i]
        if number:
            for ident in ids:
                self.found_by.setdefault(ident, set()).add(number)
        count, follow = self.paging(observation, args, len(ids))
        self.shown[mark] = (self.version, ids, (count, follow))
        if words:
            key = words.lower()
            self.query_rows.setdefault(key, set()).update(ids)
            self.query_next[key] = follow
            self.query_done[key] = follow is None
            if count is not None:
                self.query_count[key] = count
        if not number:
            pass
        elif words:
            self.queries.setdefault(number, []).append(words.lower())
            item = self.items[number - 1]
            if item.label and self.keys.get('cat') and ids and self.spec.get('category_arg') in args and not stems(words) & stems(item.name + ' ' + item.label):
                stray = sum(1 for i in ids if same_label(self.label(i)) != same_label(item.label))
                self.words_scoped = stray == 0 if self.words_scoped is None else self.words_scoped and stray == 0
        else:
            if count is not None:
                self.counts[number] = count
            self.browse_next[number] = follow
        self.note('search', item=number, words=words, start=start, rows=len(ids), count=count, next=follow)
        self.notice(observation, 'search')
        return ids, count, follow

    def walk(self, number: int, words: str | None, pages: int, stop_new: int = 3) -> int:
        """A search and up to `pages - 1` further pages while each shows at least `stop_new` new listings of the
        item; the steps it took."""
        spent = 0
        start = None
        for page in range(pages):
            if self.left() <= self.finish_room():
                break
            before = len(self.pool(number))
            got = self.search(number, words, start)
            if got is None:
                break
            spent += 1
            ids, count, follow = got
            fresh = len(self.pool(number)) - before
            if follow is None or not ids or (page and fresh < stop_new):
                break
            start = follow
        return spent

    def coverage(self, number: int) -> float:
        total = self.counts.get(number)
        seen = len(self.pool(number))
        return 1.0 if not total else min(1.0, seen / total)

    def finish_room(self) -> int:
        """Steps kept for the finish: a cart call per item, a look at the cart, the checkout, a spare, the messages
        still to send, and the pages a line sends the reader to. While no full basket exists (lean), only what a
        bare checkout needs is kept: the rest of the reserve would otherwise go unused on a decline."""
        if self.lean:
            return len(self.items) + 2 + sum(1 for p in self.process if p['when'] in ('order', 'decline'))
        notes = len(self.items) if self.fits or self.owned or (self.fit_cached and self.fit_cached[1]['more']) else 0
        second = len(self.items) if self.roles.get('offer') and self.roles.get('open') else 0
        # until the first checkout: a refusal (a change the store fires at checkout) still leaves a cart change per
        # item and the checkout again, within a tenth of the steps
        retry = min(len(self.items) + 2, self.max_steps // 10) if not self.buys else 0
        return len(self.items) + 3 + max(0, self.msgs_free()) + notes + second + retry

    def value_groups(self, number: int) -> list[tuple[str, list[str]]]:
        """The item's requirements that a short stated value can be searched for, each with its values (an "or"
        list gives several, any of which meets it), one or two words each."""
        out: list[tuple[str, list[str]]] = []
        for need in self.base_reqs(number):
            prop, _, value = need.partition(':')
            value = value.strip()
            if not value or re.search(r'\b(?:not|no|never|other than|unlike|except|same|listing|page|whatever)\b', value, re.I):
                continue
            if re.search(r'\d', value) and re.search(r'\b(?:at least|at most|under|over|below|above|less|more|lower|higher|fewer|greater|least|most|max|min|maximum|minimum|between|within|inside|outside|range|than|up to|from)\b', value, re.I):
                continue
            pieces: list[str] = []
            whole = True
            for piece in re.split(r'\s+or\s+|,|/', value):
                piece = re.sub(r'[^\w\s.-]', ' ', piece).strip().lower()
                # a longer value searched by its first two words (the search wants every word, so that shows a
                # superset of its rows), small joining words left out
                piece = ' '.join([w for w in piece.split() if w not in FUNCTION_WORDS][:2])
                if len(piece) < 2:
                    whole = False
                    continue
                if piece not in pieces:
                    pieces.append(piece)
            if pieces and whole:
                out.append((need, pieces))
        return out

    def value_words(self, number: int) -> list[str]:
        """Search words from the item's requirements: short stated values, one or two words."""
        out: list[str] = []
        for _, pieces in self.value_groups(number):
            out += [p for p in pieces if p not in out]
        return out

    def value_queries(self, number: int, piece: str) -> list[str]:
        """The searches that stand for a value: with the item's noun and alone."""
        forms = [self.query(number, piece), ' '.join(piece.split()[:2])]
        crowded = self.crowded_form(number, piece)
        return list(dict.fromkeys(forms + ([crowded] if crowded else [])))

    def crowded_form(self, number: int, piece: str) -> str | None:
        """For a two-word value whose search other categories crowd: its telling word and the item's noun (the
        noun's word need not be in every row's text, so that search only adds rows, it never rules any out)."""
        if self.noun_bad or len(piece.split()) < 2:
            return None
        noun = self.noun(number)
        if not noun or stems(noun) & stems(piece):
            return None
        form = f'{self.key_word(piece)} {noun}'
        self.hit_only.add(form)
        return form

    def complete(self, query: str) -> bool:
        """Whether a search was read to its last match (not cut by the store's last page)."""
        if query in self.hit_only or not self.query_done.get(query):
            return False
        count = self.query_count.get(query)
        return count is None or count <= len(self.query_rows.get(query, ())) + 1

    @staticmethod
    def key_word(piece: str) -> str:
        """A value's most telling single word (for a search of two values): the longest plain word, else the
        longest."""
        words = piece.split()
        if len(words) == 1:
            return piece
        plain_words = [w for w in words if w.isalpha()]
        return max(plain_words or words, key=len)

    def pair_queries(self, number: int) -> list[tuple[str, str, list[list[str]]]]:
        """Searches of two wanted values at once (the store wants every word, so one search shows the rows that
        hold both): the two requirements with the fewest alternatives, every pair of their values, at most 4."""
        groups = [(need, pieces) for need, pieces in self.value_groups(number)]
        if len(groups) < 2 or not self.spec.get('words_arg'):
            return []
        groups = sorted(groups, key=lambda g: len(g[1]))[:2]
        (need_a, pieces_a), (need_b, pieces_b) = groups
        combos = [f'{self.key_word(a)} {self.key_word(b)}' for a in pieces_a for b in pieces_b]
        if len(combos) > 4:
            return []
        return [(need_a, need_b, [[c] for c in combos])]

    def families(self, number: int) -> list[tuple[set[str], list[list[str]]]]:
        """Sets of searches that together show every row meeting some requirements by their values: (those
        requirements, per value the searches that stand for it)."""
        out = [({need}, [self.value_queries(number, p) for p in pieces]) for need, pieces in self.value_groups(number)]
        out += [({a, b}, combos) for a, b, combos in self.pair_queries(number)]
        return out

    def value_hits(self, number: int, ident: str) -> tuple[int, int]:
        """(requirements whose value searches showed this row, value-search sets read to the end without it): the
        store's word search holds the listing's attributes, so a row a finished search of a wanted value never
        showed most likely lacks it."""
        hit: set[str] = set()
        miss = 0
        for needs, alternatives in self.families(number):
            ran = [[q for q in forms if q in self.query_rows] for forms in alternatives]
            if any(ident in self.query_rows[q] for qs in ran for q in qs):
                hit |= needs
            elif all(qs and any(self.exhaustive(q) for q in qs) for qs in ran):
                miss += 1
        return len(hit), miss

    def near(self, number: int, ident: str) -> bool:
        """A row one finished value search never showed while the other values' searches did: its own words may
        write that value another way (an abbreviation in the title), so its row is still read."""
        hit, miss = self.value_hits(number, ident)
        return miss == 1 and hit >= max(1, len(self.value_groups(number)) - 1)

    def ruled_out(self, number: int, ident: str) -> bool:
        """A row the finished value searches leave out, unless it is near and its own words passed the screen."""
        if not self.value_hits(number, ident)[1]:
            return False
        return not (self.screens.get((number, ident)) == 'y' and self.near(number, ident))

    def walk_query(self, number: int, query: str, pages: int) -> int:
        """A search walked on from where an earlier walk of it stopped, while its pages show the item's rows."""
        spent = 0
        start = self.query_next.get(query) if query in self.query_rows else None
        if query in self.query_rows and start is None:
            return 0
        for _ in range(pages):
            if self.left() <= self.finish_room():
                break
            got = self.search(number, query, start)
            if got is None:
                break
            spent += 1
            ids, count, follow = got
            mine = sum(1 for i in ids if number in self.owner(i))
            if follow is None or mine < 2:
                if follow is not None:
                    self.query_next[query] = None
                break
            start = follow
        return spent

    def walk_value(self, number: int, piece: str, pages: int) -> int:
        """Search a wanted value (with the item's noun when the search does not keep to the category), continuing
        where an earlier walk of it stopped, while its pages still show the item's rows; the steps taken."""
        query = self.query(number, piece)
        spent = 0
        start = self.query_next.get(query) if query in self.query_rows else None
        if query in self.query_rows and start is None:
            crowded = self.crowded_form(number, piece)
            return self.walk_query(number, crowded, pages) if crowded and crowded in self.query_rows else 0
        for _ in range(pages):
            if self.left() <= self.finish_room():
                break
            got = self.search(number, query, start)
            if got is None:
                break
            spent += 1
            ids, count, follow = got
            if not ids and start is None and query != piece and not self.noun_bad and self.left() > self.finish_room():
                # the noun may not be in what the search reads: the value alone once
                alone = self.search(number, ' '.join(piece.split()[:2]))
                spent += 1
                if alone and any(number in self.owner(i) for i in alone[0]):
                    self.noun_bad = True
                break
            mine = sum(1 for i in ids if number in self.owner(i))
            if follow is None or mine < 2:
                if follow is not None:
                    self.query_next[query] = None
                    crowded = self.crowded_form(number, piece) if start is None else None
                    if crowded and self.left() > self.finish_room():
                        # rows of other categories fill the value's first page: its word with the item's noun
                        spent += self.walk_query(number, crowded, max(1, pages - spent))
                break
            start = follow
        return spent

    def more_queries(self, number: int, count: int) -> list[str]:
        """New search words for an item, from a model reading the rows seen so far, short and not yet tried."""
        tried = set(self.queries.get(number, []))
        titles = [self.title(i) for i in self.pool(number)][:30]
        rule = 'reads only the first two words of a query and wants all of them' if self.spec.get('words_arg') else 'takes free words'
        prompt = QUERY.replace('QUERY_RULE', rule).replace('COUNT', str(count)) + f'\n\nPRODUCT: {self.items[number - 1].name} (store label: {self.items[number - 1].label or "unknown"})\nREQUIREMENTS: ' + '; '.join(self.base_reqs(number)) + '\nTRIED: ' + jdump(sorted(tried)) + '\nTITLES SEEN:\n' + '\n'.join(titles)
        got = self.ask_json(prompt, [LIGHT], 600) or {}
        out = []
        for query in as_list(got.get('q')):
            text = ' '.join(str(query).split()[:2]).lower().strip()
            if text and text not in tried and text not in out:
                out.append(text)
        return out[:count]

    def title_words(self, number: int) -> list[str]:
        """Words many of the item's rows carry in their titles, most common first, not yet searched."""
        tried = {w for q in self.queries.get(number, []) for w in q.split()}
        counts: dict[str, int] = {}
        for ident in self.pool(number):
            for word in set(re.findall(r'[a-z][a-z-]{3,}', self.title(ident).lower())):
                counts[word] = counts.get(word, 0) + 1
        own = stems(self.items[number - 1].name)
        ranked = sorted((w for w, n in counts.items() if n >= 2 and w not in tried and w not in FUNCTION_WORDS), key=lambda w: (-counts[w], w))
        return ranked

    # ---------------------------------------------------------------- reading listings

    def open_pages(self, ids: list[str], tool: str | None = None, floor: int | None = None) -> int:
        """Open listings' pages (one step each; several at once when a side-by-side tool takes them and no single
        page is required) while more than `floor` steps are left (the finish room by default); the steps it took."""
        tool = tool or self.roles.get('open') or self.roles.get('compare')
        floor = self.finish_room() if floor is None else floor
        target = self.single_arg(tool)
        spent = 0
        todo = [i for i in dict.fromkeys(ids) if i in self.rows and tool not in self.viewed.get(i, set())]
        if not todo or not tool:
            return 0
        if target is None and tool == self.roles.get('compare'):
            many = self.list_arg(tool)
            if not many:
                return 0
            cap = (((self.schemas.get(tool) or {}).get('properties') or {}).get(many) or {}).get('maxItems') or 4
            for start in range(0, len(todo), cap):
                if self.left() <= floor or self.finished:
                    break
                if self.call(tool, {many: todo[start:start + cap]}) is not None:
                    spent += 1
            return spent
        for ident in todo:
            if self.left() <= floor or self.finished:
                break
            if self.call(tool, {target: ident}) is not None:
                spent += 1
        if spent:
            self.note('opened', ids=todo[:spent], tool=tool)
        return spent

    def other_reads(self) -> list[str]:
        """The store's other single-listing read tools: no role of their own, one required text argument of id size
        (a datasheet or property-list view beside the full page). A store may count a line as checked only when it was
        shown through such a view, so cart lines are shown through them too."""
        taken = {v for v in self.roles.values() if isinstance(v, str)}
        out = []
        for name, schema in self.schemas.items():
            props = (schema or {}).get('properties') or {}
            required = (schema or {}).get('required') or []
            if name in taken or len(required) != 1 or not isinstance(props.get(required[0]), dict):
                continue
            arg = props[required[0]]
            if 'string' in kinds_of(arg) and 'enum' not in arg and (arg.get('maxLength') or 1000) <= 100:
                out.append(name)
        return out

    def single_arg(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        texts = string_args(schema)
        required = [k for k in schema.get('required') or [] if k in texts]
        if len(required) == 1:
            return required[0]
        if len(texts) == 1:
            return texts[0]
        named = [k for k in texts if ID_HINT.search(plain(k)) or re.search(r'\b(?:item|listing|product|offer)\b', plain(k))]
        return named[0] if len(named) == 1 else None

    def list_arg(self, tool: str | None) -> str | None:
        props = (self.schemas.get(tool or '') or {}).get('properties') or {}
        return next((k for k, v in props.items() if isinstance(v, dict) and 'array' in kinds_of(v)), None)

    def card(self, ident: str, full: bool = True) -> str:
        """A listing for a prompt: id, label, title, price, stock, then its attributes, description and notes."""
        record = (self.full.get(ident) if full else None) or self.rows.get(ident) or {}
        keys = self.keys
        head = [f'[{ident}]']
        for part, key in (('category', keys.get('cat')), ('title', keys.get('title')), ('price', keys.get('price')), ('seller', keys.get('shop'))):
            if key and record.get(key) not in (None, ''):
                head.append(f'{part}: {record[key]}')
        if ident in self.gone:
            head.append('SOLD OUT')
        lines = [' | '.join(head)]
        skip = {v for v in keys.values() if v}
        block = _fact_block(record)
        if block:
            lines.append('attributes: ' + '; '.join(f'{k}={v}' for k, v in block.items())[:1400])
        for key, value in record.items():
            if key in skip or value in (None, '', [], {}):
                continue
            if block and isinstance(value, dict) and all(value.get(k) == v for k, v in block.items()):
                continue
            if isinstance(value, (dict, list)):
                lines.append(f'{key}: ' + jdump(shorten(value, 120))[:500])
            elif isinstance(value, str):
                lines.append(f'{key}: ' + value[:PAGE_CHARS if full else ROW_CHARS])
            elif not isinstance(value, bool):
                lines.append(f'{key}: {value}')
        return '\n'.join(lines)

    def screen_jobs(self, number: int, ids: list[str]) -> list[Any]:
        """Jobs marking rows of an item worth opening ('y'/'?') or not ('n'), by a model reading their summaries."""
        todo = [i for i in ids if (number, i) not in self.screens and i not in self.full]
        reqs = self.base_reqs(number)
        if not todo or not reqs:
            for ident in todo:
                self.screens[(number, ident)] = '?'
            return []
        jobs = []
        for start in range(0, len(todo), 36):
            batch = todo[start:start + 36]
            prompt = ROWS.replace('PRODUCT', self.items[number - 1].name) + '\n\nREQUIREMENTS:\n' + '\n'.join(f'{n}. {r}' for n, r in enumerate(reqs, 1))
            prompt += f'\n\nROWS ({len(batch)}):\n' + '\n'.join(re.sub(r'^\[[^\]]*\]', f'row {k}:', self.card(i, False)[:520]) for k, i in enumerate(batch, 1))

            def job(batch: list[str] = batch, prompt: str = prompt) -> None:
                got = self.ask_json(prompt, [LIGHT], 300) or {}
                picked = {}
                for letter in ('n', 'y'):
                    for k in as_list(got.get(letter)):
                        k = as_int(k)
                        if k is not None and 1 <= k <= len(batch):
                            picked[batch[k - 1]] = letter
                for ident in batch:
                    self.screens[(number, ident)] = picked.get(ident, '?')
            jobs.append(job)
        return jobs

    def judge_jobs(self, numbers: list[int]) -> list[Any]:
        """Jobs judging every opened listing of these items on each requirement still unjudged, in batches."""
        jobs = []
        models = [LIGHT] + ([HEAVY] if self.cost < SPEND_SOFT else [])
        for number in numbers:
            reqs = self.judged_reqs(number)
            if not reqs:
                continue
            ids = [i for i in self.pool(number) if i in self.full and any((number, i, r) not in self.marks for r in reqs)]
            measure = self.measure if self.goal in ('most', 'least') and number == self.measure_item else ''
            for start in range(0, len(ids), JUDGE_BATCH):
                batch = ids[start:start + JUDGE_BATCH]
                wanted = [r for r in reqs if any((number, i, r) not in self.marks for i in batch)]
                prompt = JUDGE.replace('PRODUCT', self.items[number - 1].name + (f' (store label {self.items[number - 1].label})' if self.items[number - 1].label else ''))
                prompt += '\n\nREQUIREMENTS:\n' + '\n'.join(f'{n}. {r}' for n, r in enumerate(wanted, 1))
                if measure:
                    prompt += f'\n\nMEASURE: {measure}'
                prompt += '\n\nLISTINGS:\n' + '\n\n'.join(self.card(i) for i in batch)

                def job(number: int = number, batch: list[str] = batch, wanted: list[str] = wanted, prompt: str = prompt, measured: bool = bool(measure)) -> None:
                    got = self.ask_json(prompt, models, 900) or {}
                    letters = got.get('r') if isinstance(got.get('r'), dict) else {}
                    for ident in batch:
                        row = re.sub(r'[^ynu?]', '', str(letters.get(ident, '')).lower())
                        if len(row) != len(wanted):
                            continue
                        for req, letter in zip(wanted, row):
                            letter = 'u' if letter == '?' else letter
                            if letter == 'y' and self.word_clash(ident, req):
                                self.note('veto', item=number, id=ident, req=req)
                                letter = 'n'
                            elif letter == 'y' and self.split_value(ident, req):
                                # another word inside the wanted words: a listing stating them as written goes
                                # first, this one stays for a plan that finds none
                                self.note('split', item=number, id=ident, req=req)
                                letter = 'u'
                            self.marks[(number, ident, req)] = letter
                    if measured and isinstance(got.get('m'), dict):
                        for ident, value in got['m'].items():
                            if ident in batch:
                                self.values[(number, ident)] = str(value)
                    self.note('judged', item=number, ids=batch, got=shorten(letters, 40))
                jobs.append(job)
        return jobs

    def word_clash(self, ident: str, req: str) -> bool:
        """A plain-value requirement ("finish: matte", "colour: red or blue") against attributes whose keys name its
        property: True when such keys exist and none of their values shares a word with any wanted value (a model
        reading meaning loosely passes "not stated" for "none"). Limits, ranges, numbers and exclusions are left
        to the model."""
        prop, _, value = req.partition(':')
        value = value.strip()
        if not prop.strip() or not value or re.search(r'\d|\b(?:not|no|never|other than|unlike|at least|at most|under|over|below|above|less|more|between|within|up to|from|same|any|except)\b', value, re.I):
            return False
        block = _fact_block(self.full.get(ident) or {}) or {}
        wanted = stems(prop)
        hits = [str(v) for k, v in block.items() if wanted and wanted <= stems(plain(k))]
        if not hits:
            return False
        options = [stems(o) for o in re.split(r'\s+or\s+|,|/', value) if stems(o)]
        if not options:
            return False
        if any(len(tokens(h)) > 2 for h in hits):
            # a descriptive value may name a kind of the wanted one; the reader's meaning stands
            return False
        held = set().union(*(stems(h) for h in hits))
        # words sharing a stem of four letters or more ("knit" in "knitted") are the same word
        return not any(w == h or (min(len(w), len(h)) >= 4 and (w.startswith(h) or h.startswith(w))) for o in options for w in o for h in held)

    def split_value(self, ident: str, req: str) -> bool:
        """A several-word wanted value given only with other words standing between its words (wanted "oak veneer",
        stated "oak walnut veneer") may name something else: True when an attribute whose key names the property
        holds the words in order with a gap inside a single part, and no such part (nor the title) holds them as one
        run. Limits, bare figures and exclusions are left to the model."""
        prop, _, value = req.partition(':')
        value = value.strip()
        if not prop.strip() or not value or re.search(r'\b\d+(?:\.\d+)?\b|\b(?:not|no|never|other than|unlike|at least|at most|under|over|below|above|less|more|between|within|up to|from|same|any|except)\b', value, re.I):
            return False

        def words(text: str) -> list[str]:
            return [w[:-1] if len(w) > 3 and w.endswith('s') else w for w in re.findall(r'[a-z0-9]+', text.lower())]

        def run_in(want: list[str], have: list[str]) -> bool:
            return any(have[k:k + len(want)] == want for k in range(len(have) - len(want) + 1))

        def gapped_in(want: list[str], have: list[str]) -> bool:
            at = 0
            for w in have:
                if at < len(want) and w == want[at]:
                    at += 1
            return at == len(want)

        options = [words(o) for o in re.split(r'\s+or\s+|,|/', value)]
        options = [o for o in options if len(o) >= 2]
        if not options:
            return False
        block = _fact_block(self.full.get(ident) or {}) or {}
        wanted = stems(prop)
        parts = [words(p) for k, v in block.items() if wanted and wanted <= stems(plain(k)) for p in re.split(r'[,;/|]', str(v))]
        if not parts:
            return False
        title = words(self.title(ident))
        if any(run_in(o, p) for o in options for p in parts + [title]):
            return False
        return any(gapped_in(o, p) for o in options for p in parts)

    def judge(self, numbers: list[int]) -> None:
        self.together(self.judge_jobs(numbers))

    def think_round(self) -> None:
        """One model round for every item at once: new rows screened and opened pages judged, side by side."""
        if self.learned or any(not item.label for item in self.items):
            self.learn_labels()
        jobs = []
        if self.room() > 40:
            plan, _ = self.solve()
            for number in range(1, len(self.items) + 1):
                todo = [i for i in self.pool(number) if (number, i) not in self.screens and i not in self.full and self.buyable(i) and (not self.value_hits(number, i)[1] or self.near(number, i))]
                line = plan.get(number) if plan else None
                if line and self.goal != 'most' and self.goal != 'least':
                    todo = [i for i in todo if (self.price(i) or 0) < (self.price(line) or 0)]
                jobs += self.screen_jobs(number, todo)
        jobs += self.judge_jobs(list(range(1, len(self.items) + 1)))
        self.together(jobs)

    # ---------------------------------------------------------------- relations between items

    def own_part(self, ident: str) -> set[str]:
        """The part or model codes a listing names as its own: codes its title holds in brackets (after "part",
        "model", "ref" or alone)."""
        own: set[str] = set()
        for hit in re.findall(r'[\[(]\s*(?:part|pt|p/n|model|ref|sku)?\.?\s*(?:no\.?|number|#)?\s*([A-Za-z0-9-]{4,})\s*[\])]', self.title(ident), re.I):
            own |= codes_in(hit)
        return own

    @staticmethod
    def code_sentence(line: str, found: set[str]) -> bool:
        """A sentence made mostly of codes ("Goes with AB1200, AB1300.", "Model ref. AB1200.", "Part X also fits:
        Y."): a fit or identity statement, not prose that happens to hold a feature code."""
        words = re.findall(r'[A-Za-z0-9-]+', line)
        return bool(found) and len(words) <= 4 + 2 * len(found)

    @staticmethod
    def fit_line(line: str) -> str:
        """A code sentence with spaced codes joined ("KMB 2030" -> "KMB2030", "KMB 37xx" -> "KMB37xx"), a model
        name written as a word and a number too ("Velto 512" -> "VELTO512")."""
        line = re.sub(r'\b([A-Z]{2,5})[ -](\d{2,6}(?:xx|XX|x|X)*)(?![\w-])', r'\1\2', line)
        return re.sub(r'\b([A-Z][a-z]{2,9}) (\d{2,5})\b(?![.,]?\d|[\w-])', lambda m: m.group(1).upper() + m.group(2), line)

    def fit_texts(self) -> dict[str, Any]:
        """For every listing seen: the codes it names as its own, the codes its short code sentences state (what it
        works with, or for a device, what it is), series it works with (a code ending in "xx", or the codes of a
        sentence about a series), codes it excepts, and, from any listing's sentence about another part ("Part P
        also fits X"), the codes that part works with; "more on part P" sentences point at P's page."""
        state = (len(self.rows), len(self.full))
        if self.fit_cached and self.fit_cached[0] == state:
            return self.fit_cached[1]
        own = {i: self.own_part(i) for i in self.seen_order}
        named: dict[str, set[str]] = {}
        extra: dict[str, set[str]] = {}
        more: dict[str, set[str]] = {}
        series: dict[str, set[str]] = {}
        unless: dict[str, set[str]] = {}
        for ident in self.seen_order:
            mine: set[str] = set()
            record = self.full.get(ident) or self.rows.get(ident) or {}
            for text in leaves(record):
                for line in sentences(text):
                    line = self.fit_line(line)
                    wild = {w.upper().rstrip('X') for w in re.findall(r'\b([A-Za-z]{2,5}\d{1,4})(?:xx|XX|x|X)+\b', line)}
                    listed = set()
                    if re.search(r'\bseries\b', line, re.I):
                        tail = re.split(r'\bseries\b', line, maxsplit=1, flags=re.I)[1]
                        listed = {w.upper().replace('-', '') for w in re.findall(r'\b([A-Za-z]{1,4}-?\d{1,4})\b', tail)}
                    found = codes_in(line) - {w + 'XX' for w in wild}
                    if not (found or wild or listed) or not self.code_sentence(line, found | wild | listed):
                        continue
                    said = ' '.join(re.findall(r'\b(?:part|parts|accessory|item|model)\s*(?:no\.?|number|#)?\s*:?\s*([A-Za-z0-9-]{4,})', line, re.I))
                    subjects = (codes_in(said) - own[ident]) & found
                    if subjects and re.search(r'\b(?:more|see|details?|read|info|look)\b', line, re.I) and found <= subjects | own[ident]:
                        more.setdefault(ident, set()).update(subjects)
                        continue
                    if subjects and re.search(r'\balso\b', line, re.I):
                        for part in subjects:
                            extra.setdefault(part, set()).update(found - subjects - own[ident])
                            if listed or wild:
                                series.setdefault(part, set()).update(listed | wild)
                        continue
                    head, _, after = re.split(r'(?i)\b(except|but not|apart from|other than)\b', line, maxsplit=1)[0], None, ''
                    parts_of = re.split(r'(?i)\b(?:except|but not|apart from|other than)\b', line, maxsplit=1)
                    if len(parts_of) > 1:
                        unless.setdefault(ident, set()).update(codes_in(parts_of[1]))
                        found = found - codes_in(parts_of[1])
                    if wild or listed:
                        series.setdefault(ident, set()).update(wild | listed)
                    mine |= found
            named[ident] = mine - own[ident]
        index = {'own': own, 'named': named, 'extra': extra, 'more': more, 'series': series, 'unless': unless}
        self.fit_cached = (state, index)
        return index

    def works_with(self, part: str, device: str) -> bool | None:
        """Whether a listing works with another, by the codes they carry: True when the part's code sentences name a
        code of the device, the part names a series the device's code starts with (and does not except it), or the
        device names a series the part's own code starts with; False when both carry codes and none match, None
        when either says nothing."""
        index = self.fit_texts()
        own_codes = index['own'].get(part, set())
        mine = index['named'].get(part, set()) | {c for p in own_codes for c in index['extra'].get(p, set())}
        theirs = index['own'].get(device, set()) | index['named'].get(device, set())
        if theirs & index['unless'].get(part, set()):
            return False
        if mine & theirs:
            return True
        part_series = index['series'].get(part, set()) | {c for p in own_codes for c in index['series'].get(p, set())}
        if any(t.startswith(pre) for pre in part_series for t in theirs):
            return True
        takes = index['series'].get(device, set()) | {c for t in theirs for c in index['series'].get(t, set())}
        if any(o.startswith(pre) for pre in takes for o in own_codes):
            return True
        if any(o.startswith(c) and len(c) >= 3 and c != o for c in index['named'].get(device, set()) for o in own_codes):
            return True
        if (mine or part_series) and theirs:
            return False
        if takes and own_codes:
            return False
        return None

    def works_with_code(self, part: str, code: str) -> bool | None:
        index = self.fit_texts()
        mine = index['named'].get(part, set()) | {c for p in index['own'].get(part, set()) for c in index['extra'].get(p, set())}
        wanted = codes_in(code) or {code.upper().replace('-', '')}
        if mine & wanted:
            return True
        return False if mine else None

    def fit_system(self, number: int) -> bool:
        """Whether an item's listings carry fit codes at all (most rows name some), so a silent one does not fit."""
        rows = self.pool(number)
        if not rows:
            return False
        index = self.fit_texts()
        with_codes = sum(1 for i in rows if index['named'].get(i) or any(index['extra'].get(p) for p in index['own'].get(i, set())))
        return with_codes * 2 >= len(rows)

    # ---------------------------------------------------------------- choosing the cart

    def options(self, number: int, loose: bool = False) -> list[str]:
        item = self.items[number - 1]
        out = []
        for ident in self.pool(number):
            if ident not in self.full or not self.buyable(ident):
                continue
            letter = self.verdict(number, ident)
            if letter == 'n' or letter == '?' or (letter == 'u' and not loose):
                continue
            price = self.price(ident)
            if price is None:
                continue
            if item.cap is not None and price > item.cap + 1e-9:
                continue
            out.append(ident)
        out.sort(key=lambda i: self.price(i) or 0.0)
        return out

    def cond_ok(self, picks: dict[int, str], loose: bool) -> bool:
        for cond in self.conds:
            n, m = cond['item'], cond['if_item']
            if n not in picks or m not in picks:
                continue
            probe = self.marks.get((m, picks[m], cond['if']), 'u')
            holds = probe == 'y' if cond['applies'] == 'when' else probe != 'y'
            if holds:
                letter = self.marks.get((n, picks[n], cond['req']), 'u')
                if letter == 'n' or (letter != 'y' and not loose):
                    return False
        return True

    def fit_ok(self, picks: dict[int, str], loose: bool) -> bool:
        for part, device in self.fits:
            if part in picks and device in picks:
                verdict = self.works_with(picks[part], picks[device])
                if verdict is False or (verdict is None and not loose and self.fit_system(part)):
                    return False
        for part, code in self.owned:
            if part in picks:
                verdict = self.works_with_code(picks[part], code)
                if verdict is False or (verdict is None and not loose and self.fit_system(part)):
                    return False
        return True

    def unkeyed(self, number: int, ident: str) -> int:
        """Count of requirements (for item `number`) whose property is named by no attribute key on the read page
        of `ident` (met by its title or a related key only: the weaker evidence)."""
        keys = [stems(plain(k)) for k in (_fact_block(self.full.get(ident) or {}) or {})]
        out = 0
        for need in self.base_reqs(number):
            prop = stems(plain(need.split(':')[0]))
            if prop and not any(k and (prop <= k or k <= prop) for k in keys):
                out += 1
        return out

    def doubt(self, number: int, ident: str) -> int:
        """How many of the item's requirements in force a listing's reading has not confirmed."""
        return sum(1 for r in self.base_reqs(number) if self.marks.get((number, ident, r)) != 'y')

    def fit_doubt(self, picks: dict[int, str]) -> int:
        """How many of a plan's fit relations no code settles."""
        count = sum(1 for part, device in self.fits if part in picks and device in picks and self.works_with(picks[part], picks[device]) is None)
        return count + sum(1 for part, code in self.owned if part in picks and self.works_with_code(picks[part], code) is None)

    def measure_of(self, number: int, ident: str) -> float | None:
        return as_number(self.values.get((number, ident)))

    def solve(self, loose: bool = False) -> tuple[dict[int, str] | None, list[int]]:
        """The best cart from what has been judged: (item -> listing, items with no option at all)."""
        numbers = list(range(1, len(self.items) + 1))
        # with no price to win, a line whose attributes name each wanted property is the safer of two that pass;
        # a loose plan takes the lines with the fewest requirements unread first (that counts for more), then those
        keyed = self.goal == 'any'

        def weight(n: int, ident: str) -> int:
            return (100 * self.doubt(n, ident) if loose else 0) + (self.unkeyed(n, ident) if keyed else 0)
        if loose or keyed:
            choices = {n: sorted(self.options(n, loose), key=lambda i, n=n: (weight(n, i), self.price(i) or 0.0))[:30] for n in numbers}
        else:
            choices = {n: self.options(n)[:30] for n in numbers}
        empty = [n for n in numbers if not choices[n]]
        if empty:
            return None, empty
        if self.goal in ('most', 'least') and self.measure_item in choices:
            sign = -1 if self.goal == 'most' else 1
            ranked = [i for i in choices[self.measure_item] if self.measure_of(self.measure_item, i) is not None]
            ranked.sort(key=lambda i: (sign * (self.measure_of(self.measure_item, i) or 0), self.price(i) or 0))
            choices[self.measure_item] = ranked + [i for i in choices[self.measure_item] if i not in ranked]
        order = sorted(numbers, key=lambda n: len(choices[n]))
        best: list[Any] = [None, None]
        cheapest = {n: min((self.price(i) or 0) * self.items[n - 1].qty for i in choices[n]) for n in numbers}

        def rank(picks: dict[int, str], total: float, doubt: int) -> tuple:
            if self.goal in ('most', 'least') and self.measure_item in picks:
                value = self.measure_of(self.measure_item, picks[self.measure_item])
                sign = -1 if self.goal == 'most' else 1
                return (0 if value is not None else 1, sign * (value or 0), total)
            return (0, doubt, total)
        tried = [0]

        def walk(depth: int, picks: dict[int, str], total: float, doubt: int) -> None:
            tried[0] += 1
            if tried[0] > 20000:
                return
            if depth == len(order):
                if self.budget is not None and total > self.budget + 1e-9:
                    return
                score = rank(picks, total, doubt + (100 * self.fit_doubt(picks) if loose else 0))
                if best[0] is None or score < best[0]:
                    best[0], best[1] = score, dict(picks)
                return
            n = order[depth]
            rest = sum(cheapest[m] for m in order[depth + 1:])
            for ident in choices[n]:
                if ident in picks.values():
                    continue
                cost = (self.price(ident) or 0) * self.items[n - 1].qty
                more = weight(n, ident)
                if self.budget is not None and total + cost + rest > self.budget + 1e-9:
                    if self.goal not in ('most', 'least') and not loose and not keyed:
                        break
                    continue
                if best[0] is not None and self.goal not in ('most', 'least') and (0, doubt + more, total + cost + rest) >= best[0]:
                    break
                picks[n] = ident
                if self.fit_ok(picks, loose) and self.cond_ok(picks, loose):
                    walk(depth + 1, picks, total + cost, doubt + more)
                del picks[n]
        walk(0, {}, 0.0, 0)
        if best[1] is None:
            return None, []
        return best[1], []

    def total(self, plan: dict[int, str]) -> float:
        return sum((self.price(i) or 0) * self.items[n - 1].qty for n, i in plan.items())

    # ---------------------------------------------------------------- notes from the store and the shopper

    def loose_texts(self, node: Any, depth: int = 0) -> list[str]:
        """Sentences a result holds outside its listing records: notices, refusals, changes."""
        if depth > 5:
            return []
        if isinstance(node, str):
            return [node] if len(node.split()) >= 5 else []
        if isinstance(node, dict):
            if self.id_of(node) and len(node) >= 4:
                return []
            return [s for v in node.values() for s in self.loose_texts(v, depth + 1)]
        if isinstance(node, list):
            return [s for v in node for s in self.loose_texts(v, depth + 1)]
        return []

    def notice(self, observation: Any, source: str, force: bool = False) -> dict[str, Any] | None:
        texts = [t for t in self.loose_texts(observation) if t not in self.heard_texts]
        if not texts and not force:
            return None
        self.heard_texts.update(texts)
        if not texts:
            texts = [jdump(shorten(observation, 300))[:1500]]
        ledger = '\n'.join(f'item {n}: {it.name}; requirements: ' + '; '.join(self.base_reqs(n)) for n, it in enumerate(self.items, 1))
        prompt = NOTE + f'\n\nITEMS AND REQUIREMENTS NOW:\n{ledger}\nbudget: {self.budget}\nCART: {jdump(self.cart)}\n\nTHE TEXT (from {source}):\n' + '\n'.join(texts)[:3000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1200) or {}
        self.note('note', source=source, texts=[t[:200] for t in texts], got=shorten(got, 120))
        if got.get('kind') != 'change':
            return got
        changed = False
        budget = as_number(got.get('budget')) if got.get('budget') not in (None, '') else None
        if budget and budget > 0 and self.budget is None:
            budget = self.first_budget(budget)
        if budget and budget > 0 and budget != self.budget:
            self.budget = budget
            changed = True
        for entry in as_list(got.get('drop')):
            if isinstance(entry, dict) and str(entry.get('req') or '').strip():
                number = min(max(1, as_int(entry.get('item')) or 1), len(self.items))
                item = self.items[number - 1]
                target = str(entry['req']).strip()
                live = [m for m in item.musts + item.told if m not in item.dropped]
                hit = [m for m in live if m == target or self.resolved.get(m) == target] or [m for m in live if plain(m.split(':')[0]) == plain(target.split(':')[0])][:1]
                item.dropped.extend(hit or [target])
                changed = True
        for entry in as_list(got.get('add')):
            if isinstance(entry, dict) and str(entry.get('req') or '').strip():
                number = min(max(1, as_int(entry.get('item')) or 1), len(self.items))
                item = self.items[number - 1]
                need = str(entry['req']).strip()
                if need in item.dropped:
                    item.dropped.remove(need)
                if need not in item.musts + item.told:
                    item.musts.append(need)
                changed = True
        for ident in as_list(got.get('gone')):
            if str(ident) in self.rows:
                self.gone.add(str(ident))
                changed = True
        if got.get('sold_out') is True and not as_list(got.get('gone')) and len(self.cart) == 1:
            self.gone.add(next(iter(self.cart)))
            changed = True
        if got.get('sold_out') is True or got.get('price_moved') is True:
            for ident in list(self.cart):
                self.viewed.get(ident, set()).discard(self.roles.get('open') or '')
                self.viewed.get(ident, set()).discard(self.roles.get('offer') or '')
            changed = True
        if changed:
            self.changed_at = max(self.changed_at, self.step)
            self.version += 1
            self.notes.append(str(got.get('summary') or 'the task changed'))
        return got

    # ---------------------------------------------------------------- requirements that point at another listing

    @staticmethod
    def target_mark(target: dict[str, Any]) -> tuple[str, frozenset[str]]:
        describe = target.get('describe')
        values = [str(d) for d in describe] if isinstance(describe, list) else re.split(r'[,;]', str(describe or ''))
        return plain(target.get('shop') or ''), frozenset(plain(v) for v in values if v.strip())

    @staticmethod
    def same_shop(have: Any, want: str) -> bool:
        """Whether a row's shop matches a pointer's shop name, also when the name was read with a word more or less
        ("Acme" for "Acme Home")."""
        have, want = plain(have or ''), plain(want or '')
        if not have or not want:
            return False
        if have == want:
            return True
        a, b = have.split(), want.split()
        return a[0] == b[0] and (set(a) <= set(b) or set(b) <= set(a))

    def target_words(self, target: dict[str, Any], label: str) -> list[str]:
        """Search words for the listing a pointer describes, in the description's order: per value its longest plain
        word (never a unit, the word right after a number, nor a word of the shop's name or the shelf's label), then
        per value its first number of two digits or more (the store's search splits "23.8" at the dot and drops
        one-character terms), then figures glued to a unit."""
        describe = target.get('describe')
        values = [str(d) for d in describe] if isinstance(describe, list) else re.split(r'[,;]', str(describe or ''))
        skip = stems(' '.join(str(target.get(k) or '') for k in ('shop', 'category')) + ' ' + (label or ''))
        words: list[str] = []
        numbers: list[str] = []
        mixed: list[str] = []
        for raw in values:
            value = (raw.partition(':')[2] if ':' in raw and raw.partition(':')[2].strip() else raw).lower()
            terms = [t for t in re.split(r'[\s(),./:;|]+', value) if t]
            plain_words = []
            for k, term in enumerate(terms):
                if term.isdigit():
                    continue
                after_number = k > 0 and terms[k - 1].isdigit()
                if re.match(r'\d', term) and '-' not in term:
                    continue
                if after_number and term.isalpha():
                    continue
                if len(term) >= 3 and term not in FUNCTION_WORDS and not stems(term) & skip:
                    plain_words.append(term)
            if plain_words:
                word = max(plain_words, key=len)
                if word not in words:
                    words.append(word)
            digits = next((t for t in re.split(r'[\s(),./:;|x\u00d7-]+', value) if t.isdigit() and len(t) >= 2), None)
            if digits and digits not in numbers:
                numbers.append(digits)
            # a figure glued to its unit ("3.5mm" is searched as "5mm")
            glued = next((t for t in terms if re.fullmatch(r'\d+[a-z]{1,4}', t) and len(t) >= 2), None)
            if glued and glued not in mixed:
                mixed.append(glued)
        return words + numbers + mixed

    def find_listing(self, target: dict[str, Any], skip: set[str], extra: list[str] | None = None) -> list[str]:
        """Listings that may be the one a pointer describes, best first: rows of its category from its shop, found by
        searches of the shop's name with one described value each (a second value's search narrows a crowded
        first), then by the shelf's pages; rows more of those searches showed come first, then a model's reading of
        the rows against the description; the ones already tried left out. `extra`: further value words to search."""
        labels = self.category_labels() or self.label_options()
        label = self.match_label(str(target.get('category') or ''), '', labels) if target.get('category') else ''
        shop = plain(target.get('shop') or '')
        describe = target.get('describe')
        describe = '; '.join(str(d) for d in describe) if isinstance(describe, list) else str(describe or '')

        def matches() -> list[str]:
            found = []
            for ident in self.seen_order:
                if ident in skip:
                    continue
                if label and self.keys.get('cat') and same_label(self.label(ident)) != same_label(label):
                    continue
                if shop and not self.same_shop(self.shop(ident), shop) and not re.search(r'(?<![a-z0-9])' + re.escape(shop) + r'(?![a-z0-9])', plain(self.text(ident))):
                    continue
                found.append(ident)
            return found
        first = str(target.get('shop') or '').split()[:1]
        valued = [f'{first[0]} {w}'.lower() for w in (self.target_words(target, label) + list(extra or []))] if first else []
        valued = list(dict.fromkeys(q for q in valued if len(q.split()) == 2))

        def score(ident: str) -> int:
            return sum(1 for q in valued if ident in self.query_rows.get(q, ()))
        searched = 0
        if valued and describe and self.spec.get('words_arg'):
            for query in valued:
                if self.left() <= self.finish_room() + 3:
                    break
                if query not in self.query_rows:
                    if searched >= 2:
                        break
                    self.search(0, query, None, label or '')
                    searched += 1
                fresh = [i for i in matches() if i in self.query_rows.get(query, ())]
                if not fresh and query in self.query_rows and not self.complete(query) and self.query_next.get(query) is not None and searched < 2:
                    # every row this search showed was tried: its next page
                    self.search(0, query, self.query_next.get(query), label or '')
                    searched += 1
                    fresh = [i for i in matches() if i in self.query_rows.get(query, ())]
                if fresh and (len(fresh) <= 3 or sum(1 for i in fresh if score(i) >= 2) in range(1, 4)):
                    break
        found = matches()
        if not any(score(i) for i in found) and len(found) < 4 and shop and self.left() > self.finish_room() + 2:
            # the shop's name with a word of the shelf's label (a word search may read the whole store)
            shelf_words = re.findall(r'[a-z][a-z-]{2,}', plain(label or target.get('category') or ''))
            words = str(target.get('shop')).split()[:1] + (shelf_words[-1:] if not self.noun_bad else [])
            query = ' '.join(words)
            start = self.query_next.get(query.lower()) if query.lower() in self.query_rows else None
            if query.lower() in self.query_rows and start is None:
                start = 'done'
            if start != 'done':
                got = self.search(0, query, start, label or '')
                if got and not got[0] and len(words) > 1:
                    got = self.search(0, str(target.get('shop')), None, label or '')
                    words = str(target.get('shop')).split()[:1]
                start = got[2] if got else None
                found = matches()
                for _ in range(2):
                    if start is None or len(found) >= 30 or self.left() <= self.finish_room() + 3:
                        break
                    got = self.search(0, ' '.join(words), start, label or '')
                    start = got[2] if got else None
                    found = matches()
        if not found and label:
            start = None
            for _ in range(4):
                if self.left() <= self.finish_room() + 2:
                    break
                got = self.search(0, None, start, label)
                if not got:
                    break
                found = matches()
                if found or got[2] is None:
                    break
                start = got[2]
        if len(found) > 1 and describe:
            prompt = PICK + f"\n\nDESCRIPTION: a listing of {target.get('category') or 'some kind'} from {target.get('shop') or 'some shop'}, {describe}\n\nROWS:\n" + '\n'.join(self.card(i, False)[:700] for i in sorted(found, key=lambda i: -score(i))[:30])
            got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
            ranked = [str(i) for i in as_list(got.get('ids')) if str(i) in found]
            found = ranked + [i for i in found if i not in ranked]
            hinted = [w for w in (str(x).strip().lower() for x in as_list(got.get('words'))[:2]) if w and len(w.split()) == 1 and len(w) >= 2]
            if first and hinted and not any(score(i) for i in found[:3]) and self.spec.get('words_arg'):
                # the model's words for values the rows do not show (a number in the unit such listings use)
                for word in hinted:
                    query = f'{first[0]} {word}'.lower()
                    if query in self.query_rows or self.left() <= self.finish_room() + 3:
                        continue
                    valued.append(query)
                    self.search(0, query, None, label or '')
                found = found + [i for i in matches() if i not in found]
        found = sorted(found, key=lambda i: -score(i))
        return found[:6]

    def describe_fit(self, ident: str, describe: Any) -> tuple[int, int]:
        """How an opened page stands with a pointer's description ("colour: red, size 4"): (values its attributes
        or its own words state, values an attribute named for that property contradicts)."""
        if isinstance(describe, list):
            pieces = [str(d) for d in describe]
        else:
            pieces = re.split(r'\s*[,;]\s*|\s+and\s+', str(describe or ''))
        def norm(text: Any) -> str:
            return ' '.join(re.sub(r'[_:]+', ' ', str(text or '')).lower().split())
        # keys read as words in any spelling (productForm, product_form); values keep their dots
        block = {plain(k): norm(v) for k, v in (_fact_block(self.full.get(ident) or {}) or {}).items()}
        prose = norm(self.title(ident) + ' ' + ' '.join(leaves(self.full.get(ident) or {})))
        agree = clash = 0
        for raw in pieces:
            prop, colon, rest = str(raw).partition(':')
            piece = norm(raw)
            if not piece:
                continue
            if colon and prop.strip() and rest.strip():
                wanted_key = stems(prop)
                keys = [k for k in block if wanted_key and (wanted_key <= stems(k) or stems(k) <= wanted_key)]
                want = norm(rest)
            else:
                keys = [k for k in block if piece.startswith(k + ' ') and len(piece) > len(k) + 1]
                want = piece[len(max(keys, key=len)):].strip() if keys else piece
            if not keys:
                # no attribute names it: its words in the page's text count for it, never against it
                words = [w for w in tokens(want) if len(w) >= 3]
                if words and all(re.search(r'(?<![a-z0-9])' + re.escape(w), prose) for w in words):
                    agree += 1
                continue
            key = max(keys, key=len)
            have = block[key]
            numbers = re.findall(r'\d+(?:\.\d+)?', want)
            if numbers:
                held = {float(x) for x in re.findall(r'\d+(?:\.\d+)?', have)}
                # "16 x 500 gb" holds 8000 gb
                held |= {float(a) * float(b) for a, b in re.findall(r'(\d+(?:\.\d+)?)\s*[x\u00d7*]\s*(\d+(?:\.\d+)?)', have)}
                same = all(float(x) in held for x in numbers)
                units_want = stems(' '.join(re.findall(r'\d\s*([a-z]+)', want)))
                units_have = stems(' '.join(re.findall(r'\d\s*([a-z]+)', have)))
                if not same and units_want and units_have and not units_want & units_have:
                    # the same figure in another unit (500 ml for 16.9 fl oz): not decided here
                    continue
            else:
                same = bool(stems(want)) and (stems(want) <= stems(have) or stems(have) <= stems(want))
            if same:
                agree += 1
            elif want and re.search(r'(?<![a-z0-9])' + re.escape(want) + r'(?![a-z0-9])', prose) and not block.get(key):
                agree += 1
            else:
                clash += 1
        return agree, clash

    def chase(self, pointer: dict[str, Any]) -> str:
        """Follow one pointer requirement through the pages it leads to until a page states the value; the rewritten
        requirement goes in force. What it found, as text."""
        if pointer['done'] or self.finished or pointer['tries'] >= (3 if self.left() > self.finish_room() + 12 else 2):
            return ''
        pointer['tries'] += 1
        target = dict(pointer.get('at') or pointer['target'])
        pages: list[str] = pointer['pages']
        tried: set[str] = pointer.setdefault('tried', set())
        wanted = len([d for d in (target.get('describe') if isinstance(target.get('describe'), list) else re.split(r'[,;]', str(target.get('describe') or ''))) if str(d).strip()])
        visited = {self.target_mark(pointer['target'])}
        for hop in range(CHAIN_HOPS):
            visited.add(self.target_mark(target))
            if self.left() <= self.finish_room() + 1 or self.room() < 25:
                break
            got = None
            # two looks a hop: when no row the first one opened is the listing, the second searches other values
            for look in range(2):
                ranked = []
                for ident in self.find_listing(target, tried | set(pages))[:5]:
                    if len(ranked) >= 3 and any(r[0] == 0 and r[1] < 0 for r in ranked) and self.left() <= self.finish_room() + 8:
                        break
                    tried.add(ident)
                    self.open_pages([ident], self.roles.get('open') or self.roles.get('offer'))
                    if ident not in self.full:
                        continue
                    agree, clash = self.describe_fit(ident, target.get('describe'))
                    ranked.append((clash, -agree, len(ranked), ident))
                    if clash == 0 and agree >= max(1, wanted):
                        break
                    if self.left() <= self.finish_room() + 2:
                        break
                for clash, minus_agree, _, ident in sorted(ranked):
                    if clash and len(ranked) > 1 and any(r[0] == 0 for r in ranked):
                        continue
                    if clash >= 2 and clash > -minus_agree:
                        # its attributes contradict more of the description than they meet: another listing
                        continue
                    if self.roles.get('offer') and self.roles.get('open'):
                        # a store with two page tools may keep a page's pointers and notes on the second one
                        self.open_pages([ident], self.roles.get('offer'), self.finish_room() - 1)
                    prompt = CHAIN + f"\n\nREQUIREMENT: {pointer['req']}\nPROPERTY: {pointer['property']}\nRELATION: {pointer['relation']}\nLOOKING FOR: a listing of {target.get('category') or 'some kind'} from {target.get('shop') or 'some shop'}, {target.get('describe') or ''}\n\nPAGES OPENED ALONG THE POINTER, IN ORDER (the last one is the page to judge):\n" + '\n\n'.join(self.card(i) for i in pages + [ident])
                    answer = self.ask_json(prompt, [LIGHT, HEAVY], 900) or {}
                    ahead = answer.get('next') if isinstance(answer.get('next'), dict) else None
                    if answer.get('is_target') is True and ahead is not None and ((not str(ahead.get('shop') or '').strip() and target.get('shop')) or self.target_mark(ahead) in visited):
                        # a next listing without its shop cannot be searched for, and one described as a listing
                        # already passed is a misread: the stronger model reads the page again
                        answer = self.ask_json(prompt, [HEAVY], 900) or answer
                    if answer.get('is_target') is True:
                        pages.append(ident)
                        got = answer
                        break
                if got or not ranked or self.left() <= self.finish_room() + 4 or self.room() < 25:
                    break
            if not got:
                break
            value = str(got.get('value') or '').strip()
            if isinstance(got.get('next'), dict) and not value:
                target = got['next']
                pointer['at'] = target
                wanted = len([d for d in (target.get('describe') if isinstance(target.get('describe'), list) else re.split(r'[,;]', str(target.get('describe') or ''))) if str(d).strip()])
                continue
            req = str(got.get('req') or '').strip()
            if req and ':' in req:
                self.resolved[pointer['req']] = req
                pointer['done'] = True
                self.version += 1
                self.note('pointer', req=pointer['req'], now=req, pages=pages)
                return f"{pointer['req']} -> {req}"
            break
        self.note('pointer_open', req=pointer['req'], pages=pages, tried=sorted(tried))
        return ''

    def chase_all(self) -> str:
        live = {need for n in range(1, len(self.items) + 1) for need in self.items[n - 1].musts + self.items[n - 1].told if need not in self.items[n - 1].dropped}
        done = [self.chase(p) for p in self.pointers if not p['done'] and p['req'] in live]
        return '; '.join(d for d in done if d)

    # ---------------------------------------------------------------- the interviewer's hooks

    def widen(self) -> None:
        """Before planning questions: two read pages per item at least, so the planner sees what listings carry."""
        for number in range(1, len(self.items) + 1):
            have = sum(1 for i in self.pool(number) if i in self.full)
            if have >= 2:
                continue
            picks = self.open_order(number)[:2 - have]
            self.open_pages(picks)

    def early_trail(self) -> str:
        return self.chase_all() if self.room() > 60 else ''

    def value_search(self, numbers: list[int]) -> str:
        """After an answer added a requirement: one search for its value, so its listings are in view."""
        for number in numbers:
            told = [t for t in self.items[number - 1].told if t not in self.items[number - 1].dropped]
            if not told or self.left() <= self.finish_room() + 4:
                continue
            value = told[-1].partition(':')[2].strip()
            words = ' '.join(re.sub(r'[^\w\s.-]', ' ', value).split()[:2])
            if words and not re.search(r'\b(?:not|no|at least|at most|under|over|between)\b', value, re.I):
                self.search(number, words)
        return ''

    # ---------------------------------------------------------------- exploring

    def open_order(self, number: int) -> list[str]:
        """The item's unread listings in the order worth opening: those the screen kept, cheapest first when the
        cheapest wins, else the ones whose rows already show the requirements first."""
        item = self.items[number - 1]
        out = []
        groups = len(self.value_groups(number))
        for ident in self.pool(number):
            if ident in self.full or not self.buyable(ident):
                continue
            if self.screens.get((number, ident)) == 'n' and not (groups >= 2 and self.value_hits(number, ident)[0] >= groups):
                continue
            price = self.price(ident)
            if price is not None and ((item.cap is not None and price > item.cap + 1e-9) or (self.budget is not None and price * item.qty > self.budget + 1e-9)):
                continue
            out.append(ident)
        wanted = set(tokens(' '.join(self.value_words(number))))
        fit = {ident: self.fit_rank(number, ident) for ident in out}
        if any(v == 2 for v in fit.values()) and self.fit_system(number):
            out = [i for i in out if fit[i] != 2] + [i for i in out if fit[i] == 2]

        hits = {ident: self.value_hits(number, ident) for ident in out}

        def key(ident: str) -> tuple:
            letter = self.screens.get((number, ident), '?')
            price = self.price(ident)
            price = price if price is not None else float('inf')
            match = len(wanted & set(tokens(self.text(ident))))
            hit, miss = hits[ident]
            if self.goal == 'cheapest':
                return (fit[ident] == 2, miss, -hit, letter != 'y', price, -match)
            return (fit[ident], miss, -hit, letter != 'y', -match, price)
        out.sort(key=key)
        return out

    def fit_rank(self, number: int, ident: str) -> int:
        """How a row stands with the items it must work with: 0 it works with a fitting listing of each of them
        (or the code the shopper owns), 1 nothing says, 2 it works with none of them."""
        worst = 0
        for part, device in self.fits:
            if number not in (part, device):
                continue
            other = device if number == part else part
            partners = self.options(other, loose=True) or [i for i in self.pool(other) if self.verdict(other, i) != 'n'][:25]
            if not partners:
                continue
            says = [self.works_with(ident, d) if number == part else self.works_with(d, ident) for d in partners[:25]]
            rank = 0 if True in says else 2 if says and all(s is False for s in says) else 1
            worst = max(worst, rank)
        for part, code in self.owned:
            if part == number:
                says = self.works_with_code(ident, code)
                worst = max(worst, 0 if says else 2 if says is False else 1)
        return worst

    def settled(self, number: int, plan: dict[int, str] | None) -> bool:
        """Whether more reading cannot improve the item's line. Under the cheapest: nothing unread is cheaper and the
        shelf was seen or searched out. Under most/least: the measure item's fitting rows were read. Under any fit, a
        cheaper fit is still worth a few steps while many are left (a fit is right, the cheapest one never wrong)."""
        line = plan.get(number) if plan else None
        unread = self.open_order(number)
        if line is None:
            return not unread and self.exhausted.get(number, False)
        price = self.price(line) or 0
        cheaper = [i for i in unread if (self.price(i) or 0) < price]
        if self.goal == 'cheapest':
            known = self.known_all(number)
            if known:
                cheaper = [i for i in cheaper if not self.ruled_out(number, i)]
            if cheaper:
                return False
            return known or self.coverage(number) >= 0.95 or self.exhausted.get(number, False)
        if self.goal in ('most', 'least') and number == self.measure_item:
            return not unread or len([i for i in self.pool(number) if i in self.full]) >= 14
        spare = self.left() - self.finish_room() - 4 * len(self.items)
        return not cheaper or spare < 4 or self.room() < 110

    def known_all(self, number: int) -> bool:
        """Whether some set of value searches was read to the end, so every listing that can meet the item was
        shown by them."""
        for _, alternatives in self.families(number):
            ran = [[q for q in forms if q in self.query_rows] for forms in alternatives]
            if all(qs and any(self.exhaustive(q) for q in qs) for qs in ran):
                return True
        return False

    def exhaustive(self, query: str) -> bool:
        """A finished search that shows every row holding its words' value: never a search holding a number, as listings
        write numbers in many ways ("1tb", "1 tb", "1000 gb")."""
        return self.complete(query) and not re.search(r'\d', query)

    def step_share(self, numbers: list[int]) -> int:
        free = self.left() - self.finish_room()
        return max(0, free // max(1, len(numbers)))

    def learn_labels(self) -> None:
        """A store with no list of category labels still shows each row's category: an item without a label takes
        the shown category that names it clearly best (the item's words cover it, and no other comes close), so
        rows of a neighbouring kind ("LED bulbs & strip" for an "LED light bulb") are not bought for it."""
        if not self.keys.get('cat'):
            return
        shown: dict[str, int] = {}
        for ident in self.seen_order:
            label = str(self.label(ident) or '').strip()
            if label:
                shown[label] = shown.get(label, 0) + 1
        if len(shown) < 2:
            return
        def words(text: str) -> set[str]:
            return {w for w in stems(text) if w not in FUNCTION_WORDS}
        for item in self.items:
            if item.label and item.name not in self.learned:
                continue
            wanted = words(item.name + ' ' + (item.shelf or ''))
            # a label must be mostly the item's words, and the more of the item's words it holds the better ("a
            # lamp or light fixture" is "Lamps and Light Fixtures", not "Lamps")
            scored = sorted(((len(wanted & words(o)) / max(1, len(words(o))), len(wanted & words(o)) / max(1, len(wanted)), shown[o], o) for o in shown), key=lambda x: (x[0] >= 0.66, x[0] * x[1], x[2]), reverse=True)
            if scored and scored[0][0] >= 0.66 and (len(scored) == 1 or scored[1][0] * scored[1][1] < scored[0][0] * scored[0][1] or scored[1][0] < 0.66):
                if item.label == scored[0][3]:
                    continue
                if item.label:
                    # a label learned from the first rows gives way to a closer one seen later
                    now = next((x for x in scored if x[3] == item.label), None)
                    if now and now[0] * now[1] >= scored[0][0] * scored[0][1]:
                        continue
                item.label = scored[0][3]
                self.learned.add(item.name)
                self.browse_next.pop(self.items.index(item) + 1, None)
                self.note('label_learned', item=item.name, label=item.label)

    def first_look(self) -> None:
        """Every item: its shelf's first page (its labels and size) and searches of its wanted values, deeper under
        the cheapest; then the most promising rows screened and a few opened and judged, all items together."""
        numbers = list(range(1, len(self.items) + 1))
        for number in numbers:
            if self.left() <= self.finish_room():
                break
            item = self.items[number - 1]
            named = [w for w in re.findall(r"[a-z][a-z'-]{2,}", item.name.lower()) if w not in FUNCTION_WORDS]
            # with no category label, two words of the item's own name find its shelf better than one noun
            self.walk(number, ' '.join(named[:2]) if not item.label and len(named) >= 2 else None, 1)
        self.learn_labels()
        deep = 3 if self.goal == 'cheapest' else 2
        budget = max(2, (self.left() - self.finish_room() - self.msgs_free()) // (3 * len(numbers)))
        for number in numbers:
            spent = 0
            for _, _, combos in self.pair_queries(number):
                for (combo,) in combos:
                    if spent >= budget or self.left() <= self.finish_room() + 2:
                        break
                    spent += self.walk_query(number, combo, min(2, budget - spent))
            for _, pieces in self.value_groups(number)[:3]:
                for piece in pieces[:3]:
                    if spent >= budget or self.left() <= self.finish_room() + 2:
                        break
                    spent += self.walk_value(number, piece, min(deep, budget - spent))
            if not self.value_groups(number) and len(numbers) == 1:
                self.walk(number, None, 2)
        self.money_scale()
        self.screen_all()
        per = 3 if len(numbers) == 1 else 2
        for number in numbers:
            self.open_pages(self.open_order(number)[:per])

    def screen_all(self) -> None:
        jobs = []
        for number in range(1, len(self.items) + 1):
            todo = [i for i in self.pool(number) if (number, i) not in self.screens and i not in self.full and (not self.value_hits(number, i)[1] or self.near(number, i))]
            jobs += self.screen_jobs(number, todo)
        if jobs and self.room() > 40:
            self.together(jobs)

    def hunt(self, number: int, room: int) -> int:
        """More searches for an item, at most `room` steps: its wanted values' searches read further first, then
        the shelf's further pages, then new words (the titles' common words, a model's ideas from the titles
        seen); the steps taken."""
        spent = 0
        for _, _, combos in self.pair_queries(number):
            for (combo,) in combos:
                if spent >= room or self.left() <= self.finish_room():
                    break
                spent += self.walk_query(number, combo, min(3, room - spent))
        for _, pieces in self.value_groups(number):
            for piece in pieces:
                if spent >= room or self.left() <= self.finish_room():
                    break
                spent += self.walk_value(number, piece, min(3, room - spent))
        if spent and self.known_all(number) and self.options(number):
            return spent
        # the shelf browsed on from where it stopped (from its start when never browsed, or since a new label)
        fresh = number not in self.browse_next
        follow = self.browse_next.get(number)
        while spent < room and (fresh or follow is not None) and self.left() > self.finish_room():
            got = self.search(number, None, None if fresh else follow)
            fresh = False
            if not got:
                break
            spent += 1
            follow = got[2]
        tried = set(self.queries.get(number, []))
        for word in [w for w in self.title_words(number) if w not in tried][:4]:
            if spent >= room or self.left() <= self.finish_room():
                break
            query = self.query(number, word) if not stems(word) & stems(self.noun(number)) else word
            if query in tried:
                continue
            tried.add(query)
            spent += self.walk(number, query, 3)
        if spent < room and self.room() > 50:
            for query in [q for q in self.more_queries(number, 3) if q not in tried]:
                if spent >= room or self.left() <= self.finish_room():
                    break
                spent += self.walk(number, query, 2)
        return spent

    def explore(self, numbers: list[int], plan: dict[int, str] | None = None) -> bool:
        """One round of more reading for the items not settled: the linked part pages fit items need, then the next
        rows worth opening (only rows cheaper than the item's line once it has one), searches when none are left."""
        moved = self.link_parts(numbers) > 0
        share = self.step_share(numbers)
        for number in numbers:
            if self.left() <= self.finish_room() or self.finished:
                break
            room = min(share, 6)
            line = plan.get(number) if plan else None
            picks = self.open_order(number)
            if line and not (self.goal in ('most', 'least') and number == self.measure_item):
                picks = [i for i in picks if (self.price(i) or 0) < (self.price(line) or 0)]
            if line and self.known_all(number):
                # rows a finished value search never showed are left out while a cheaper line is sought; with no
                # line yet they stay (a value can be written another way than the search's words)
                picks = [i for i in picks if not self.ruled_out(number, i)]
            picks = picks[:min(room, 4 if self.goal == 'cheapest' or not line else 2)]
            if picks:
                moved |= self.open_pages(picks) > 0
                continue
            if line and self.goal == 'any':
                continue
            if self.room() > 45 and not self.exhausted.get(number):
                spent = self.hunt(number, max(2, min(room, 5)))
                if not spent:
                    self.exhausted[number] = True
                moved |= spent > 0
        return moved

    def link_parts(self, numbers: list[int]) -> int:
        """For items that must work with another: the pages a fitting candidate's "more on part P" sentence points at
        (they can name more devices it fits, and must be read before buying it), when no candidate fits yet; the
        steps taken."""
        spent = 0
        index = self.fit_texts()
        for part, device in self.fits:
            if part not in numbers and device not in numbers:
                continue
            parts = self.options(part, loose=True)[:8]
            devices = self.options(device, loose=True)[:12]
            if not devices or any(self.works_with(p, d) for p in parts for d in devices):
                continue
            for ident in parts + devices[:4]:
                for code in sorted(index['more'].get(ident, set())):
                    if code in self.looked or self.left() <= self.finish_room() + 2 or spent >= 4:
                        continue
                    self.looked.add(code)
                    if self.part_page(code):
                        spent += 1
            # parts of the series a candidate device takes ("Takes part series AB3"): the series code with a
            # wanted value of the part (or its noun); the store's words search reads the code in a part's title
            series = []
            for ident in devices[:3]:
                series += [c for c in sorted(index['series'].get(ident, set())) if c not in series]
            values = [self.key_word(pieces[0]) for _, pieces in self.value_groups(part)[:2]]
            for code in series[:2]:
                for words in (values or [self.noun(part)])[:2]:
                    query = f'{code} {words}'
                    if query in self.looked or self.left() <= self.finish_room() + 2 or spent >= 4:
                        continue
                    self.looked.add(query)
                    spent += self.walk_query(part, query, 2)
            # parts another listing says fit a candidate device ("Part P also fits: <its code>"), not seen yet
            codes = set()
            for ident in (self.options(device)[:4] or devices[:4]):
                codes |= index['own'].get(ident, set()) | index['named'].get(ident, set())
            known = set().union(*index['own'].values()) if index['own'] else set()
            for code in sorted(c for c, fits in index['extra'].items() if fits & codes and c not in known):
                if code in self.looked or self.left() <= self.finish_room() + 2 or spent >= 3:
                    continue
                self.looked.add(code)
                before = len(self.seen_order)
                self.part_page(code)
                spent += 1 if len(self.seen_order) >= before else 0
        return spent

    def wanting(self, plan: dict[int, str] | None, empty: list[int]) -> list[int]:
        numbers = list(range(1, len(self.items) + 1))
        if plan is None and empty:
            return [n for n in empty if not self.settled(n, None)]
        if plan is None:
            tied = {n for pair in self.fits for n in pair} | {c['item'] for c in self.conds} | {c['if_item'] for c in self.conds}
            return [n for n in numbers if (n in tied or self.budget is not None) and not (not self.open_order(n) and self.exhausted.get(n))]
        return [n for n in numbers if not self.settled(n, plan)]

    def work(self) -> tuple[dict[int, str] | None, list[int]]:
        """Read, judge and choose until the cart is settled or steps and time run short; the plan."""
        self.first_look()
        if self.roles.get('ask') and self.msgs_free() > 0 and self.clock() < LATE_ASK and self.left() > self.finish_room() + 4:
            summary = self.interview()
            self.note('interview', summary=summary[:600])
        self.chase_all()
        plan, empty = None, []
        for _ in range(16):
            if self.finished or self.broke or self.room() < 30:
                break
            self.think_round()
            plan, empty = self.solve()
            open_ptr = [p for p in self.pointers if not p['done'] and p['tries'] < 3]
            if open_ptr and self.room() > 50:
                self.chase_all()
            wanting = self.wanting(plan, empty)
            self.note('round', plan=plan, empty=empty, wanting=wanting, left=self.left(), total=self.total(plan) if plan else None)
            if not wanting or self.left() <= self.finish_room():
                break
            if not self.explore(wanting, plan):
                break
        self.judge(list(range(1, len(self.items) + 1)))
        return self.solve()

    # ---------------------------------------------------------------- the finish

    def cart_style(self) -> dict[str, Any]:
        tool = self.roles.get('cart')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        style: dict[str, Any] = {'tool': tool}
        for key, spec in props.items():
            inner = spec.get('items') if isinstance(spec, dict) and 'array' in kinds_of(spec) else None
            if isinstance(inner, dict) and isinstance(inner.get('properties'), dict):
                fields = inner['properties']
                style['lines'] = key
                style['line_id'] = next((k for k, s in fields.items() if isinstance(s, dict) and 'string' in kinds_of(s) and 'enum' not in s), None)
                style['line_qty'] = next((k for k, s in fields.items() if isinstance(s, dict) and {'integer', 'number'} & set(kinds_of(s))), None)
                return style
        style['id'] = self.single_arg(tool) or next(iter(string_args(schema)), None)
        style['qty'] = next((k for k in number_args(schema)), None)
        enums = enum_args(schema)
        add = self.spec.get('cart_add') or {}
        remove = self.spec.get('cart_remove') or {}
        mode = next((k for k in enums if k in add or k in remove), None) or next(iter(enums), None)
        style['mode'] = mode
        if mode:
            options = enums[mode]
            style['add'] = add.get(mode) if add.get(mode) in options else next((o for o in options if re.search(r'add|append|put|insert|plus|set|place|hold', str(o), re.I)), options[0])
            style['remove'] = remove.get(mode) if remove.get(mode) in options else next((o for o in options if re.search(r'remov|delet|drop|take|minus|clear|release|out', str(o), re.I)), options[-1])
        return style

    def set_cart(self, lines: dict[str, int]) -> bool:
        """Make the cart hold exactly `lines` ({listing: units}). A tool taking the whole list gets it in one call. A
        per-listing tool offering add/remove choices adds to what is held, so it is sent only the difference: lines to
        drop are taken out first (so a switch never leaves the old product in), then each line is raised or lowered
        to its count, in calls no larger than the schema allows (one unit a call when it takes no count). A reply
        that shows the cart is taken as what is held."""
        style = self.cart_style()
        tool = style.get('tool')
        if not tool:
            return False
        if style.get('lines'):
            entry = [{style['line_id']: i, **({style['line_qty']: n} if style.get('line_qty') else {})} for i, n in lines.items()]
            if self.call(tool, {style['lines']: entry}) is None:
                return False
            self.cart = dict(lines)
            self.cart_changed = True
            return True
        if not style.get('id'):
            return False
        if not style.get('mode'):
            return self.set_each(style, lines)
        for ident, held in list(self.cart.items()):
            if ident not in lines and held > 0 and not self.cart_change(style, ident, held, 'remove'):
                return False
        for ident, want in lines.items():
            held = self.cart.get(ident, 0)
            if want != held and not self.cart_change(style, ident, abs(want - held), 'add' if want > held else 'remove'):
                return False
        return True

    def count_room(self, style: dict[str, Any]) -> tuple[int, int]:
        """The smallest and largest count one cart call may carry (1 and 1 when the tool takes no count)."""
        spec = ((self.schemas.get(style.get('tool') or '') or {}).get('properties') or {}).get(style.get('qty') or '') or {}
        if not style.get('qty'):
            return 1, 1
        low = max(1, as_int(spec.get('minimum')) or 1)
        return low, max(low, as_int(spec.get('maximum')) or 99)

    def cart_change(self, style: dict[str, Any], ident: str, units: int, verb: str) -> bool:
        """Add or remove `units` of one listing through the add/remove tool, chunked to the schema's limits."""
        low, high = self.count_room(style)
        while units > 0:
            step = max(low, min(units, high))
            args = {style['id']: ident, style['mode']: style[verb]}
            if style.get('qty'):
                args[style['qty']] = step
            observation = self.call(style['tool'], args)
            if observation is None:
                return False
            held = self.cart.get(ident, 0) + (step if verb == 'add' else -step)
            if held > 0:
                self.cart[ident] = held
            else:
                self.cart.pop(ident, None)
            self.take_shown(observation)
            self.cart_changed = True
            self.notice(observation, 'cart')
            units -= step
        return True

    def set_each(self, style: dict[str, Any], lines: dict[str, int]) -> bool:
        """A per-listing tool without an add/remove choice: its count sets the line (0 takes it out)."""
        tool = style['tool']
        for ident, held in list(self.cart.items()):
            if ident in lines or not style.get('qty'):
                continue
            if self.call(tool, {style['id']: ident, style['qty']: 0}) is not None:
                self.cart.pop(ident, None)
                self.cart_changed = True
        for ident, want in lines.items():
            if self.cart.get(ident) == want:
                continue
            args = {style['id']: ident}
            if style.get('qty'):
                args[style['qty']] = want
            observation = self.call(tool, args)
            if observation is None:
                return False
            self.cart[ident] = want
            self.take_shown(observation)
            self.cart_changed = True
            self.notice(observation, 'cart')
        return True

    def cart_calls(self, lines: dict[str, int]) -> int:
        """Store calls needed to bring the cart from what it holds to `lines`."""
        style = self.cart_style()
        if style.get('lines'):
            return 1
        if not style.get('mode'):
            return sum(1 for i in self.cart if i not in lines) + sum(1 for i, n in lines.items() if self.cart.get(i) != n)
        _, high = self.count_room(style)
        out = sum(-(-h // high) for i, h in self.cart.items() if i not in lines and h > 0)
        return out + sum(-(-abs(n - self.cart.get(i, 0)) // high) for i, n in lines.items())

    def held_in(self, observation: Any) -> tuple[dict[str, int], bool] | None:
        """The cart lines a result shows ({listing: units}, zero included) and whether they are the whole cart (the
        lines arrive in a list) or only the lines just changed (a single record); None when it shows no line."""
        held: dict[str, int] = {}
        whole = False
        stack: list[tuple[Any, bool]] = [(observation, False)]
        while stack:
            node, listed = stack.pop()
            if isinstance(node, dict):
                ident = self.id_of(node)
                if ident and ident in self.rows:
                    qty = next((as_int(v) for k, v in node.items() if re.search(r'\b(?:qty|quantity|count|units?|number|pieces)\b', plain(k)) and as_int(v) is not None), None)
                    if qty is not None:
                        held[ident] = qty
                        whole = whole or listed
                stack += [(v, isinstance(v, list)) for v in node.values() if isinstance(v, (dict, list))]
            elif isinstance(node, list):
                stack += [(v, True) for v in node]
        return (held, whole) if held else None

    def take_shown(self, observation: Any) -> None:
        """What a cart call's reply shows replaces what is tracked: all of it when it lists the cart, else its lines."""
        shown = self.held_in(observation)
        if shown is None:
            return
        held, whole = shown
        cart = {} if whole else dict(self.cart)
        for ident, qty in held.items():
            if qty > 0:
                cart[ident] = qty
            else:
                cart.pop(ident, None)
        self.cart = cart

    def look_at_cart(self) -> None:
        tool = self.roles.get('cart_view')
        if not tool or self.left() <= 2:
            return
        observation = self.call(tool, {})
        if observation is None:
            return
        self.cart_view = observation
        self.cart_changed = False
        held: dict[str, int] = {}
        stack = [observation]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                ident = self.id_of(node)
                if ident and ident in self.rows:
                    qty = next((as_int(v) for k, v in node.items() if re.search(r'\b(?:qty|quantity|count|units?|number|pieces)\b', plain(k)) and as_int(v) is not None), None)
                    held[ident] = qty if qty is not None else self.cart.get(ident, 1)
                stack += [v for v in node.values() if isinstance(v, (dict, list))]
            elif isinstance(node, list):
                stack += node
        shown = jdump(observation)
        if held or not any(i in shown for i in self.cart):
            self.cart = {i: n for i, n in held.items() if n > 0}
        self.notice(observation, 'cart view')

    def part_page(self, code: str, floor: int | None = None) -> str | None:
        """The listing whose own code is `code`: among seen rows, else found by a search for the code (while more
        than `floor` steps are left; the finish room by default)."""
        index = self.fit_texts()
        hit = next((i for i, own in index['own'].items() if code in own), None)
        if hit or self.left() <= (self.finish_room() + 1 if floor is None else floor):
            return hit
        self.search(0, code, None, '')
        index = self.fit_texts()
        return next((i for i, own in index['own'].items() if code in own), None)

    def pages_before_order(self, plan: dict[int, str]) -> None:
        """Every line's own page, the page of each part a line sends the reader to, and each pointer's pages."""
        lines = list(plan.values())
        # inside the finish room: what the cart, its view and the checkout need stays free
        self.open_pages(lines, self.roles.get('open') or self.roles.get('offer'), len(lines) + 3)
        if self.roles.get('offer') and self.roles.get('open'):
            self.open_pages(lines, self.roles.get('offer'), len(lines) + 3)
        for tool in self.other_reads():
            self.open_pages(lines, tool, len(lines) + 3)
        index = self.fit_texts()
        for ident in lines:
            for code in sorted(index['more'].get(ident, set())):
                # the part's row shown by a search is enough to read its note; a page is opened only when steps allow
                if self.left() <= len(lines) + 2:
                    return
                page = self.part_page(code, len(lines) + 2)
                if page and page not in self.full and self.left() > len(lines) + 6:
                    self.open_pages([page], self.roles.get('open') or self.roles.get('offer'))

    def gate(self, stage: str) -> None:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished:
            return
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = '\n'.join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {jdump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending)
        for _ in range(2):
            prompt = GATE.replace('STAGE', 'check out' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join(f"- {p['step']}" for p in steps)
            prompt += f'\n\nCART NOW: {jdump(self.cart)} (total {self.cart_total()})\n\nRECORD:\n' + ('\n'.join(self.done_log[-30:]) or '(nothing yet)') + f'\n\nMESSAGES LEFT: {self.msg_limit - self.msg_used}   STEPS LEFT: {self.left()}\n\nTOOLS:\n{tools}'
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.note('gate', stage=stage, got=shorten(got, 150))
            if got.get('done') is True or not call or self.left() < 2:
                return
            name = str(call.get('tool') or '')
            args = call.get('args') if isinstance(call.get('args'), dict) else {}
            if name == self.roles.get('ask'):
                if self.msg_used >= self.msg_limit:
                    return
                text = max(_strings(args), key=len, default='')
                if not text or not self.send([(1, '', text)], 1):
                    return
                self.gate_msgs += 1
                continue
            if name not in self.schemas or name in ending:
                return
            self.call(name, args)

    def cart_total(self) -> float:
        return sum((self.price(i) or 0) * n for i, n in self.cart.items())

    def checkout(self, plan: dict[int, str]) -> str:
        """Put the plan in the cart and buy it: 'done', 'refused' (the store said no and its note was read),
        'changed' (a late answer changed the requirements) or 'failed'."""
        lines: dict[str, int] = {}
        for number, ident in plan.items():
            lines[ident] = lines.get(ident, 0) + self.items[number - 1].qty
        before = {i: (self.price(i), self.buyable(i)) for i in lines}
        self.pages_before_order(plan)
        if self.finished:
            return 'done'
        moved = [i for i in lines if (self.price(i), self.buyable(i)) != before[i]]
        if moved:
            # a page read just now shows another price or no stock: the plan is made again on what it says
            self.note('line_moved', ids=moved)
            self.version += 1
            return 'changed'
        changes = self.cart_calls(lines)
        if changes + 1 > self.left():
            # the cart could not be set and bought within the steps left
            self.note('no_room', changes=changes, left=self.left())
            return 'failed'
        if not self.set_cart(lines):
            return 'failed'
        if self.roles.get('cart_view') and self.left() > 3:
            self.look_at_cart()
            if any(i in self.gone for i in lines):
                return 'refused'
            if self.cart != lines and self.left() > len(lines) + 2:
                self.set_cart(lines)
        if self.reread_lines(lines):
            return 'changed'
        if self.msg_used < self.msg_limit and not self.late_asked and self.left() > 4:
            version = self.version
            late = self.late_question()
            if late or self.version != version:
                self.note('late_answer', text=late[:300])
                return 'changed'
        if self.left() >= 2:
            self.gate('order')
        if self.finished:
            return 'done'
        reply = self.act(self.roles['buy'], dict(self.spec.get('buy_args') or {}))
        self.buys += 1
        if reply is None:
            return 'failed'
        if self.finished or reply.get('finished'):
            self.note('bought', lines=lines, total=self.total(plan))
            return 'done'
        self.refusals += 1
        self.refused_cart = dict(self.cart)
        self.notice(reply.get('observation'), 'checkout', force=True)
        return 'refused'

    def reread_lines(self, lines: dict[str, int]) -> bool:
        """Read again, while steps allow, each cart line whose page was read before the store last may have changed
        it; whether one now shows another price or no stock (then a fresh plan is made)."""
        tool = self.roles.get('open') or self.roles.get('offer')
        stale = [i for i in lines if self.page_step.get(i, -1) < self.changed_at]
        # after the reads there must still be room to swap a line, send the messages due at checkout and check out
        after = len(lines) + 1 + sum(1 for p in self.process if p['when'] == 'order') + 2
        if not tool or not stale or self.left() < len(stale) + after:
            return False
        before = {i: (self.price(i), self.buyable(i)) for i in lines}
        for ident in stale:
            self.viewed.get(ident, set()).discard(tool)
        self.open_pages(stale, tool, after - 1)
        moved = [i for i in lines if (self.price(i), self.buyable(i)) != before[i]]
        if moved:
            self.note('line_moved', ids=moved, late=True)
            self.version += 1
        return bool(moved)

    def quiet_args(self) -> dict[str, Any]:
        tool = self.roles.get('refuse')
        schema = self.schemas.get(tool or '') or {}
        args = dict(self.spec.get('refuse_args') or {})
        props = schema.get('properties') or {}
        for key in string_args(schema):
            if key in args and not any(i in str(args[key]) for i in self.rows):
                continue
            if key in (schema.get('required') or []) or key in args:
                args[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:(props.get(key) or {}).get('maxLength') or 200]
        return args

    def audit(self) -> dict[int, list[str]]:
        """A second reading before a decline: which read listings might meet an item after all."""
        records = [self.card(i)[:800] for i in self.seen_order if i in self.full and self.buyable(i)][:22]
        if not records or self.room() < 25:
            return {}
        ledger = '\n'.join(f'item {n}: {it.name} (label {it.label}), quantity {it.qty}; requirements: ' + '; '.join(self.base_reqs(n)) for n, it in enumerate(self.items, 1))
        prompt = AUDIT + f'\n\nREQUIREMENTS:\n{ledger}\nbudget: {self.budget}\n\nRECORDS:\n' + '\n\n'.join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1200) or {}
        out: dict[int, list[str]] = {}
        for key, ids in (got.get('fits') or {}).items() if isinstance(got.get('fits'), dict) else []:
            number = as_int(re.sub(r'\D', '', str(key)) or 0) or 0
            if 1 <= number <= len(self.items):
                out[number] = [str(i) for i in as_list(ids) if str(i) in self.full]
        self.note('audit', got=shorten(got, 120))
        return out

    def decline(self) -> dict[str, Any] | None:
        if self.finished or not self.roles.get('refuse'):
            return None
        if self.left() >= 2:
            self.gate('decline')
        if self.finished:
            return None
        return self.act(self.roles['refuse'], self.quiet_args())

    def first_budget(self, value: float) -> float:
        """A budget a note sets on a request that had none is in the money of the request's per-item caps: a figure
        over 30 times off the largest cap, which a hundred times more or less brings within 3 times of it, was
        read in the other unit."""
        caps = [item.cap for item in self.items if item.cap]
        if not caps:
            return value
        top = max(caps)
        if 1 / 30 <= value / top <= 30:
            return value
        for scaled in (value * 100, value / 100):
            if 1 / 3 <= scaled / top <= 3:
                self.note('budget_unit', read=value, taken=scaled, cap=top)
                return scaled
        return value

    def money_scale(self) -> None:
        """The budget and per-unit caps in the store's money: a request in dollars against a store pricing in cents
        (or the other way) shows as prices a hundred times off the limits, so the limits are rescaled once."""
        if self.scaled or self.budget is None:
            return
        prices = sorted(p for p in (self.price(i) for i in self.seen_order) if p)
        if len(prices) < 8:
            return
        self.scaled = True
        middle = prices[len(prices) // 2]
        unit = (self.unit or '').lower()
        factor = 1.0
        if middle > 30 * self.budget and not re.search(r'cent', unit):
            factor = 100.0
        elif middle * 30 < self.budget and re.search(r'cent', unit) and prices[-1] * 10 < self.budget:
            factor = 0.01
        if factor != 1.0:
            self._budget = self.budget * factor
            for item in self.items:
                if item.cap is not None:
                    item.cap *= factor
            self.version += 1
            self.note('money_scale', factor=factor, budget=self.budget, middle=middle)

    def confirm(self, plan: dict[int, str], loose: bool = False) -> bool:
        """A second, stronger reading of the lines about to be bought whose item has other options (the cheapest
        of many is often the borderline one): a requirement it says is broken marks the line out; with no price
        to win, one it finds unstated marks the line unproven, so a line that states it goes first. Whether any
        line was marked."""
        todo = {n: i for n, i in plan.items() if i not in self.confirmed and len(self.options(n, loose)) > 1}
        if not todo or self.cost > SPEND_SOFT or self.room() < 40:
            return False
        lines = []
        for number, ident in todo.items():
            reqs = self.base_reqs(number)
            lines.append(f'ITEM {number}: {self.items[number - 1].name}; requirements: ' + '; '.join(f'{k}. {r}' for k, r in enumerate(reqs, 1)) + '\n' + self.card(ident))
        got = self.ask_json(CONFIRM + '\n\n' + '\n\n'.join(lines), [HEAVY, LIGHT], 400) or {}
        letters = got.get('r') if isinstance(got.get('r'), dict) else {}
        out = False
        for number, ident in todo.items():
            self.confirmed.add(ident)
            reqs = self.base_reqs(number)
            row = re.sub(r'[^ynu]', '', str(letters.get(ident, '')).lower())
            if len(row) != len(reqs):
                continue
            for req, letter in zip(reqs, row):
                if letter == 'n':
                    self.marks[(number, ident, req)] = 'n'
                    out = True
                    self.note('confirm_out', item=number, id=ident, req=req)
                elif letter == 'u' and self.goal == 'any' and self.marks.get((number, ident, req)) == 'y':
                    self.marks[(number, ident, req)] = 'u'
                    out = True
                    self.note('confirm_unproven', item=number, id=ident, req=req)
        return out

    def loose_plan(self) -> dict[int, str] | None:
        """The loose plan, its lines read again by the stronger reading first (as a strict plan's are)."""
        plan, _ = self.solve(loose=True)
        for _ in range(2):
            if plan is None or self.room() < 40 or not self.confirm(plan, True):
                break
            plan, _ = self.solve(loose=True)
        return plan

    def finish(self, plan: dict[int, str] | None, empty: list[int]) -> None:
        for attempt in range(4):
            if self.finished or self.broke or self.left() <= 0:
                return
            for _ in range(2):
                if plan is None or not self.confirm(plan):
                    break
                plan, empty = self.solve()
            if plan is None and self.room() > 25:
                rescued = self.audit() if empty else {}
                # the second reading's picks stand as meeting their item (a loose plan would take any cheaper
                # unread-requirement listing over them)
                for number, ids in rescued.items():
                    for ident in ids:
                        for req in self.base_reqs(number):
                            self.marks[(number, ident, req)] = 'y'
                plan, empty = self.solve()
                if plan is None:
                    plan = self.loose_plan()
            if plan is None:
                if empty and self.room() > 45 and attempt == 0:
                    # no full basket yet: the steps kept for a checkout's pages and retries go to the items still
                    # without a listing, for a few rounds while they keep turning up new listings
                    self.lean = True
                    try:
                        for _ in range(3):
                            if self.left() <= self.finish_room() + 2 or self.room() <= 45:
                                break
                            seen = len(self.full)
                            self.explore(empty)
                            self.judge(empty)
                            plan, empty = self.solve()
                            if plan is not None or not empty or len(self.full) == seen:
                                break
                    finally:
                        self.lean = False
                    self.note('lean_search', found=plan is not None, left=self.left())
                    if plan is not None:
                        continue
                    plan = self.loose_plan()
                    if plan is not None:
                        continue
                reply = self.decline()
                if self.finished or reply is None or self.left() <= 2:
                    return
                # the store would not take the decline: its note may change the budget or the requirements
                self.notice(reply.get('observation') if isinstance(reply, dict) else reply, 'decline', force=True)
                self.judge(list(range(1, len(self.items) + 1)))
                plan, empty = self.solve()
                if plan is None:
                    plan = self.loose_plan()
                continue
            result = self.checkout(plan)
            self.note('checkout', result=result, plan=plan)
            if result in ('done',) or self.finished:
                return
            if result == 'failed' and attempt >= 1:
                break
            self.judge(list(range(1, len(self.items) + 1)))
            if self.room() > 40 and self.left() > self.finish_room() + 2:
                plan, empty = self.solve()
                if plan is None or any(not self.settled(n, plan) for n in range(1, len(self.items) + 1)):
                    self.explore([n for n in range(1, len(self.items) + 1) if not plan or not self.settled(n, plan)])
                    self.judge(list(range(1, len(self.items) + 1)))
            plan, empty = self.solve()
            if plan is None:
                plan, _ = self.solve(loose=True)
        if not self.finished and self.cart and self.cart != getattr(self, 'refused_cart', None) and self.left() > 0:
            self.act(self.roles['buy'], dict(self.spec.get('buy_args') or {}))
        if not self.finished:
            self.decline()

    # ---------------------------------------------------------------- a model working the tools, when code cannot

    def ledger(self) -> str:
        lines = []
        for number, item in enumerate(self.items, 1):
            lines.append(f'item {number}: {item.name}' + (f' (store label {item.label})' if item.label else '') + f', quantity {item.qty}; requirements: ' + ('; '.join(self.base_reqs(number)) or 'none beyond the kind'))
        if self.budget is not None:
            lines.append(f'budget: at most {self.budget:g} {self.unit} in total')
        lines.append({'cheapest': 'pick: the lowest total wins', 'most': f'pick: the highest {self.measure}', 'least': f'pick: the lowest {self.measure}'}.get(self.goal, 'pick: any listing that meets everything'))
        for part, device in self.fits:
            lines.append(f'item {part} must work with item {device}')
        for cond in self.conds:
            lines.append(f"item {cond['item']}: {cond['req']} {'when' if cond['applies'] == 'when' else 'unless'} item {cond['if_item']} has {cond['if']}")
        if self.notes:
            lines.append('changes so far: ' + ' | '.join(self.notes[-4:]))
        lines.append(f'steps: {self.step} used of {self.max_steps}')
        return '\n'.join('- ' + line for line in lines)

    def loop(self) -> None:
        ask = self.roles.get('ask')
        offered = [t for t in self.tools if t['function']['name'] != ask]
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        chat: list[dict[str, Any]] = [{'role': 'system', 'content': LOOP},
                                      {'role': 'user', 'content': "SHOPPER'S REQUEST:\n" + self.request + '\n\nLEDGER:\n' + self.ledger() + '\n\nStart by searching for each item and opening its likely listings.'}]
        idle = 0
        pushed = False
        while not self.finished and not self.broke and self.left() > 0 and self.room() > MODEL_FLOOR and self.loop_calls < 45:
            if self.ready_to_ask():
                summary = self.interview()
                chat.append({'role': 'user', 'content': (summary or 'No questions were asked.') + '\n\nLEDGER:\n' + self.ledger()})
                continue
            self.loop_calls += 1
            message = self.think(chat, [LIGHT, SPARE], 2500, offered)
            if message is None:
                break
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                fn = call.get('function') if isinstance(call, dict) and isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                args = raw if isinstance(raw, dict) else (parse_json(raw) or {}) if isinstance(raw, str) else {}
                calls.append({'id': str(call.get('id') or f'h{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': jdump(args)}})
            content = message.get('content') if isinstance(message.get('content'), str) else ''
            if not calls:
                idle += 1
                chat.append({'role': 'assistant', 'content': content or '(no call)'})
                chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.ledger()})
                if idle >= 3:
                    break
                continue
            idle = 0
            chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            for index, call in enumerate(calls):
                name = call['function']['name']
                args = parse_json(call['function']['arguments']) or {}
                if name == ask:
                    text = "Not run: Indora's interviewer handles messages to the shopper."
                elif name in ends and len(calls) > 1:
                    text = 'Not run: send the finishing call on its own.'
                elif name == self.roles.get('refuse') and not pushed and self.left() > self.max_steps // 3:
                    pushed = True
                    text = f'Not run: {self.left()} steps are left. Search in ways not tried yet (other words, the category pages) before ending without a purchase.'
                else:
                    observation = self.call(name, args)
                    if name == self.roles.get('cart') and observation is not None:
                        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
                        if ident:
                            removing = any(isinstance(v, str) and re.search(r'remov|delet|drop|take', v, re.I) for v in args.values())
                            if removing:
                                self.cart.pop(ident, None)
                            else:
                                self.cart[ident] = next((as_int(v) for v in args.values() if as_int(v) is not None and not isinstance(v, str)), 1) or 1
                    text = 'The call could not run (check its arguments).' if observation is None else jdump(shorten(observation, 400))[:7000]
                if index == len(calls) - 1:
                    text += '\n\nLEDGER:\n' + self.ledger()
                chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': text})
            size = sum(len(jdump(m.get('content'))) for m in chat)
            if size > 140000:
                for m in chat[2:len(chat) // 2]:
                    if m.get('role') == 'tool' and len(m.get('content') or '') > 500:
                        m['content'] = m['content'][:400] + '...(older result shortened)'
        if not self.finished and self.left() > 0:
            if self.cart and self.roles.get('buy'):
                self.act(self.roles['buy'], dict(self.spec.get('buy_args') or {}))
            if not self.finished:
                self.decline()

    # ---------------------------------------------------------------- the episode

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        opening = self.view.get('opening')
        if opening is not None:
            opened = unpack(opening)
            self.absorb('', {}, opened)
            self.notice(opened, 'the store at the start')
        if self.mapped:
            try:
                plan, empty = self.work()
            except Exception as exc:
                self.note('crash', where='work', error=f'{type(exc).__name__}: {exc}'[:400])
                plan, empty = self.solve()
            self.finish(plan, empty)
        else:
            self.loop()
        self.note('end', finished=self.finished, steps=self.step, cost=round(self.cost, 5), calls=self.calls)
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Indora(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return list(episode.run())
    except Exception as exc:
        episode.note('crash', error=f'{type(exc).__name__}: {exc}'[:400])
        try:
            if not episode.finished and episode.mapped:
                plan, empty = episode.solve()
                if plan is None:
                    plan, _ = episode.solve(loose=True)
                if plan is not None:
                    episode.checkout(plan)
                if not episode.finished:
                    episode.decline()
        except Exception as again:
            episode.note('crash', error=f'{type(again).__name__}: {again}'[:400])
        return list(episode.trace)


__all__ = ['agent_main']
