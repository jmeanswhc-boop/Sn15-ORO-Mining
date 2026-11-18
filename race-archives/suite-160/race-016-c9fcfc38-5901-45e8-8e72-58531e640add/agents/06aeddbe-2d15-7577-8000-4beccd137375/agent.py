from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
FAST = 'anthropic/claude-haiku-5.5'
STRONG = 'anthropic/claude-sonnet-5.5'
BACKUP = 'openai/gpt-6-luna'
TURN_CAP = 42
CALL_CAP = 60
SOFT_USD = 0.055
HARD_USD = 0.078
STRONG_ROOM = 0.02
BACKOFF = (4, 10, 20, 35)
WAIT_BUDGET = 45
VIEW_CHARS = 450
OBS_CHARS = 15000
CHAT_CHARS = 150000
RITUAL_ROUNDS = 3
PAGE_AUTO = 6
CHEAP_OPENS = 4
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
SYS_BUYER = "You are Forge, a buying operator for one shopper. You act only by calling store tools. The episode scores 1 or 0: a purchase must hold every requested product at the right quantity, each meeting every firm requirement in force at checkout, inside budget; or, when nothing truly fits, end with the no-purchase tool as the request requires.\n\nSTORE HABITS\n- Each tool call spends one step. Read each tool's schema and use exact argument names.\n- Search can filter by category, take short value-first keyword queries, and page with the store's token or offset. When totals exceed the page, page on. Empty filtered results usually mean a wrong filter label — retry with labels from rows.\n- Search rows are thin. Attributes live on the full listing view; open that before trusting a requirement. Prefer a multi-listing view when it shows attributes. Do not re-open an unchanged listing.\n- Shopper messages are handled by Forge's interviewer, not by you. Answers update the mission ledger. Forge may pack several planned properties of one item into one message while the shopper accepts that; if they ask for one question at a time, Forge switches to single-property messages.\n- Process steps listed in the ledger (approval, a check, a report) are run by Forge just before checkout or a no-purchase ending — do not stall on them. If a message remains that the order itself does not need, Forge may ask one more planned question before checkout.\n\nFIT RULES\n- Product kind is the category label matching the requested item as a whole, not a neighbour.\n- A property requirement is met when an attribute whose key names that property states the value (any key spelling; one list element is enough; case, plural, plain synonyms; containing the required words is fine). A key for a different property decides nothing. If no attribute names the property, title/description/other attributes may count — but a bare number that does not say what it measures does not.\n- A category label that itself states a required value meets that requirement.\n- Requirements that point at another listing must be resolved by opening every page in the chain; search rows are not enough.\n- Accessories that must fit a device are chosen from the listings' own compatibility words; if none fit the chosen device, switch device rather than walk away untried.\n- Wishes are tie-breakers. Sold-out lines never buy. Price×qty summed must stay within budget (equality ok).\n\nPICKING\n- Cheapest / lowest total: the order must hold the lowest-priced listing among ALL that fit — page the set, open ascending by price, use max-price filters under the current best.\n- Most / least of a measure: compare every fitter on that measure.\n- Otherwise any full fit is enough. Cart a fit early so there are steps left to replace sell-outs.\n- Multi-item requests need one cart line per item at its quantity.\n\nCHANGES\n- Budget, requirements, price and stock can change mid-run (often on checkout refusal). Apply changes, fix the cart, try again.\n- Re-read offer/stock on chosen lines shortly before checkout.\n\nFINISH\n- Set the cart exactly. If the store has a cart-view tool, call it after the last cart change. Then re-open each cart line's page (price/stock can move), run any report tool the mission requires, and checkout alone.\n- Tell the shopper the basket only when the request or process list asks for it.\n- If a cart line sells out at checkout, remove it immediately and cart the best already-read fit (or the next cheaper fit when the pick is cheapest). Do not end without a purchase while unread or already-opened backups remain.\n- Walk away only after a real search shows no full fit — not because steps ran low.\n- Every assistant turn must include a tool call. Keep prose short."

PROMPT_BRIEF = 'Prepare a shopping mission for Forge. Read the shopper request and store tools. Return ONE compact JSON object and nothing else with keys:\n\n"items": one entry per distinct product to buy, each {"name": kind in the request\'s words, "shelf": exact store category label when tools list labels, else the category word the request uses for that product if clear, else "", "quantity": integer units of that one listing (1 if unset; a pair/set that is one listing is 1), "musts": firm requirements as "property: constraint" using the request\'s property word and exact value/limit/exclusion. A must whose value is read from another listing is ONE must holding that whole description. Never put price, budget, stock or quantity into musts. Story context is not a must unless it states a property outright; product-name words (season/audience in the name) stay in "name". A use the request makes a product property IS a must unless the category label already names it. Whatever the shopper will personally check or verify is a firm must. One "I want ..." with one quantity is ONE item.\n"budget": max total as a number in the request\'s unit, or null.\n"budget_unit": that unit in words, or null.\n"budget_scope": "total"|"each"|null.\n"in_stock_only": true unless out-of-stock is clearly allowed.\n"pick": "cheapest" for lowest price/total, "most"|"least" for an extreme measure, else "any". Tight-money remarks alone are not a pick rule.\n"pick_measure": measure for most/least, else "".\n"messages": allowed shopper messages (int) or null.\n"steps": allowed steps (int) or null.\n"lookups": musts (copied) whose value is read from another listing, else [].\n"rules": how-to-work / how-to-end instructions, each ≤12 words.\n"wishes": nice-to-haves.\n"ask_first": things to check with the shopper before choosing.\n"process": required non-buy steps besides finding/buying (shopper approval, showing them the cart, reading a report, viewing the cart first), each {"step": ≤15 words, "when": "choose"|"order"|"end", "messages": int}; [] if none. Prefer when=order for steps due just before checkout. Do not invent cart-view or line-reread steps — Forge enforces those in code when the store has the tools.\n"quiet_decline": true/false/null for naming alternatives on no-fit.\n"fits": [{"item": n, "with": m}] when one item must work with another chosen item.\n"roles": tool name per job or null: "search","open" (attributes), "offer" (price/stock if separate), "compare","ask","cart" (changes the cart), "cart_view" (reads the cart without changing it), "buy","refuse","report" (account/usage/allowance if any). Prefer distinct tools for cart vs cart_view when both exist.\n"buy_args": fixed args that make buy check out, else {}.\n"refuse_args": fixed args that end without purchase, else {}.'
PROMPT_ASK = 'Rank properties Forge\'s interviewer should ask the shopper about. The interviewer packs several labels into one store message while the shopper accepts list questions; your job is a ranked list of distinct properties per item (Forge picks how many fit each message).\n\nShoppers hold a few unwritten firm preferences, each about one property of one product. A preference is revealed only when a question names that exact property. If they do not mind a property, or do not think about it, that still uses the message — so rank carefully.\n\nYou get the request, settled facts per item, and numbered attribute keys seen on opened listings (counts and sample values). An unwritten preference may use a key not listed.\n\nOrder:\n0. ASK FIRST items from the request, by property.\n1. Hinted properties from the story (who uses it, looks, setting, what is inside) — map to a listed key when one fits; otherwise after the item\'s three most common keys.\n2. Listed keys: favour type/style, form/fit, size/capacity, material, colour/pattern for that product kind; prefer keys on most listings; after five+ reads, skip keys on only one or two.\nSkip settled properties (a requirement already names that property; a key whose values only repeat a required decoration is NOT settled), duplicate names, multi-option catch-alls, ids/models/warranty/logistics. Pack size/count keys are normal properties. A key marked [not the stated ...] is a different property than that requirement. Rank each item alone: hints first, then listed keys.\n"label" is the property\'s plain full name (all key words in plain spelling — "lid colour" for lidColour; never shortened to "colour"); only the property name, no example values, no brackets, no second property.\nUp to COUNT per item, best first. ONE JSON only: {"ask":[{"item":n,"label":"...","hint":"≤8 story words or empty"}]}'
PROMPT_HEAR = 'Read shopper answers for Forge\'s interviewer. ONE JSON only:\n{"needs":[{"item":n,"need":"property: constraint","points":true when value is read from another listing else false}],"drop":[{"item":n,"property":"..."}],"free":[{"item":n,"property":"..."}],"crowded":[question numbers that ask for fewer things or one question at a time],"budget": new total or null}\n- A need is what is demanded, never its opposite. Write "not X" only when X is refused. Wording that refuses to bargain over a value states that value as wanted.\n- Copy values word for word with nouns and units; drop framing around the value ("oval" for "matching oval").\n- Name the need by the property the answer itself names (may differ from the one asked); several answers about one property make one need.\n- Pointing at another listing is ONE need with points:true; copy the comparison text exactly.\n- free: properties the shopper does not mind. If one message named several properties and the answer does not say which, leave free empty (Forge will rotate the remaining labels itself).\n- crowded: mark when the reply wants a shorter or single-property question so Forge can stop packing lists.'
PROMPT_NOTE = 'A store note appeared. ONE JSON only:\n{"kind":"change"|"error"|"other","budget":new total or null,"add":[{"item":n,"need":"property: constraint"}],"drop":[{"item":n,"need":"..."}],"gone":[listing ids unavailable],"sold_out":true when a chosen listing is gone even without naming it,"price_moved":true when a price changed,"summary":"one short sentence"}\nA replaced requirement is drop+add. Report every change.'
PROMPT_CHECK = 'Last review before Forge checks out. Decide from the records whether this cart is exactly right.\nReading: an attribute whose key names the requirement\'s property decides (any spelling, list element ok, synonyms/units, containing words ok, limits inclusive, range wholly inside); a different property key decides nothing; with no naming attribute, title/description/other attributes may count (bare undiagnosed numbers do not); category label matching the item as a whole; unlike-another-listing rules pass unless records show the same value. Each item needs its own cart line at the requested qty; nothing else; nothing out of stock; price×qty total within budget. Newest record wins.\nWhen pick is cheapest/most/least, if another shown listing clearly fits and beats the cart on that rule, the cart is wrong.\nONE JSON: {"ok":true|false,"problems":["..."],"better":[ids best first]}. ok=false only for a problem the records clearly show or a requirement no record addresses.'
READ_RULES = 'an attribute decides when its key names the same property (any spelling, list element ok, synonyms/units, containing words ok, limits inclusive); a different property key decides nothing; with no naming attribute, other attributes/title/description may count (bare undiagnosed numbers count for nothing); category label may meet a value it states and must match the item; unlike-another-listing passes unless records show the same value; out-of-stock and over-budget do not fit.'
PROMPT_AUDIT = 'Forge wants to end without buying. Check listing records against requirements, with these reading rules: an attribute decides when its key names the same property (any spelling, list element ok, synonyms/units, containing words ok, limits inclusive); a different property key decides nothing; with no naming attribute, other attributes/title/description may count (bare undiagnosed numbers count for nothing); category label may meet a value it states and must match the item; unlike-another-listing passes unless records show the same value; out-of-stock and over-budget do not fit.\nROW records are search summaries only — do not pass or fail on properties the row does not show.\nONE JSON: {"fits":{"item number":[ids that appear to meet every requirement, cheapest first, ≤4]},"open":[ROW ids of a requested kind inside budget whose page could show a fit, ≤6],"reason":"one sentence"}'
PROMPT_UNMIX = 'Some requirements take their value from another listing; words that describe that other listing are sometimes mis-read as requirements on the product being bought. From the request, which OTHER REQUIREMENTS only describe that other listing? ONE JSON: {"describe_other":[those requirements copied exactly; empty when all are real]}'
PROMPT_TRAIL = 'Before checkout, some requirements take values from other listings, and pages can chain further. The order counts only when every page along each chain was opened in full.\nONE JSON: {"chains":[{"need":exact requirement,"links":[{"id":listing id or "","about":how the need/previous page describes it,"find":search words when not found}],"done":true when the last link is an OPENED page that states the property value itself,"value":that value or ""}]}\nA record matches a link only when it matches the whole description (shop/brand, kind, AND every property value given). List links in order; stop at the first not OPENED or not found.'
PROMPT_RITUAL = 'Forge is about to STAGE. The request asks for the steps below. From the numbered RECORD of what was done, decide whether each step due before this STAGE is done for the cart as it is now. Done only when a RECORD line shows it after the cart last changed (for cart/order steps) with a granting reply; product-property questions do not count.\nIf a step is missing, give the one store call that does it next. Messages go through the messaging tool: one short sentence about that step alone, naming no product properties the step does not need. No call when every due step is done or messages/steps cannot do it.\nONE JSON: {"steps":[{"step":"...","done":bool,"lines":[RECORD numbers]}],"call":{"tool":"...","args":{...}}|null,"hold":what a reply asks to change or "","needs":[{"item":n,"need":"property: constraint"}]}'
PROMPT_SWAP = 'A cart listing sold out while Forge was working. Find replacements among other records, with these reading rules: an attribute decides when its key names the same property (any spelling, list element ok, synonyms/units, containing words ok, limits inclusive); a different property key decides nothing; with no naming attribute, other attributes/title/description may count (bare undiagnosed numbers count for nothing); category label may meet a value it states and must match the item; unlike-another-listing passes unless records show the same value; out-of-stock and over-budget do not fit.\nONE JSON: {"fits":[{"id":listing id,"price":number as shown} for every listing that meets every requirement of the sold-out line\'s item, ≤6],"reason":"one sentence"}'
PROMPT_SHAPE = 'A store tool returned the result below. Name its parts by exact keys. ONE JSON: {"rows":key of the listings list,"count":key for total matches or null,"next":key for next-page value or null,"next_arg":parameter that takes that value or null,"price":row price key or null,"stock":row buyable key or null}'
PROMPT_ATTR = 'A shopper set a requirement. Below are numbered attribute lines (key = value) from listings of that kind. A key names the requirement\'s property under any spelling or plain synonym; a different property does not. A value meets when it states the requirement (case/plural/synonyms, list element ok, containing words ok, units converted, limits inclusive, range wholly inside, exclusion met by any other value). An abbreviation is not the word it stands for. ONE JSON: {"keys":[exact keys that name the property],"yes":[line numbers whose value meets the requirement under any key]}'
KEEP_KEYS = 3
KEEP_SHARE = 0.0
KEEP_GAP = 0.34
SKIP_PROP = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_PROP = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_N = 6
COMPAT_CODE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_RE = '[A-Z]{1,4}\\d{3,6}'
ATTR_SKIP = '\\d|\\b(?:same as|other than|unlike|listing|page|shown|within|at most|at least|under|over|below|above|less|more|between|up to|maximum|minimum)\\b'
ATTR_NOT = '^\\s*(?:not|no|non|never|except|excluding|exclude|without|anything but)\\b[\\s:-]*'
ATTR_SCORE = {'pass': 4, 'weak': 2, 'conflict': 1, 'unknown': 1, 'differs': 0}
ATTR_STOP = {'a', 'an', 'the', 'of', 'in', 'with', 'and', 'to', 'for', 'on', 'it', 'is'}
ATTR_SHOW = 6
ATTR_LEARNS = 6
ATTR_ROOM = 10
ATTR_LINES = 90
BASKET_KEEP = 1
# Interview packing + wall-clock: keep episodes inside a typical sandbox time budget.
LIST_SIZE = 8
SCARCE_MESSAGES = 2
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 8.0
ASK_BY = 210.0
QUESTION_BY = 230.0
LATE_BY = 190.0
LATE_STEPS = 8
SWAP_TIGHT = 10
SWAP_HUNT = 4
# Min steps to absorb discard + add + checkout after replacing a still-valid cart line.
CHEAP_SWAP_ROOM = 6
WIND_DOWN = 200.0
LAST_TURN = 255.0

def _attr_words(text: Any) -> frozenset[str]:
    words = re.findall('[a-z0-9]+', str(text or '').lower())
    return frozenset(w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w for w in words if w not in ATTR_STOP)

def _attr_alts(value: Any) -> list[frozenset[str]]:
    parts = re.split('\\s+or\\s+|\\s*/\\s*|\\s*,\\s*', str(value or ''))
    return [w for w in (_attr_words(p) for p in parts) if w]

def _attr_holds(alts: list[frozenset[str]], text: Any) -> bool:
    words = _attr_words(text)
    return any(alt <= words for alt in alts)

def _attr_share(a: Any, b: Any) -> bool:
    """Whether two attribute values have a word in common, allowing for word endings (rectangle, rectangular)."""
    left, right = (_attr_words(a), _attr_words(b))
    return any(x == y or (len(x) >= 5 and len(y) >= 5 and x[:5] == y[:5]) for x in left for y in right)

def _compat_codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + COMPAT_CODE + '\\b', str(text or ''))]

def _compat_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_RE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + COMPAT_CODE + ')\\b', text, re.I):
        codes += _compat_codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += _compat_codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_RE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += _compat_codes(hit.group(1))
    also = [(hit.group(1).upper(), _compat_codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_RE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_RE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _shrink(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: _shrink(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_shrink(v, limit, depth + 1) for v in value]
    return value

def _as_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None

def _plain(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _stems(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _plain(text).split()))

def _root_set(text: str) -> set[str]:
    out = set()
    for w in _plain(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _same_prop(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _schema_kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _kind_ok(value: Any, kind: str) -> bool:
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

def _coerce(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _schema_kinds(schema)
    if kinds and (not any((_kind_ok(value, k) for k in kinds))):
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
        if not any((_kind_ok(value, k) for k in kinds)):
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
            value = [_coerce(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _coerce(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _looks_id(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_looks_id(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _id_in_obs(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_id_in_obs(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_id_in_obs(v, ident, depth + 1) for v in node.values()))
    return False

def _attr_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

class GoodsLine:

    def __init__(self, name: str, shelf: str, qty: int, rules: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.rules = rules
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Pulse:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def steps_left(self) -> int:
        return self.L.max_steps - self.L.step

    def clock(self) -> float:
        return time.monotonic() - self.L.started

    def model_room(self) -> float:
        return TASK_SECONDS - TIME_MARGIN - self.clock()

    def basket_keep(self) -> int:
        """Reserve a shopper message for cart reveal only when the request/process asks."""
        if not self.L.roles.get('ask') or not self.L.roles.get('buy'):
            return 0
        if self.L.msg_limit < 2:
            return 0
        if getattr(self.L, 'basket_told', -1) > self.L.cart_at and self.L.Catalog.basket_ids():
            return 0
        watch = getattr(self.L, 'SealWatch', None)
        if watch is None or not watch.basket_reveal_wanted():
            return 0
        return BASKET_KEEP

    def ask_slots(self) -> int:
        kept = max(0, self.L.reserve - self.L.gate_msgs) + self.basket_keep()
        return self.L.msg_limit - self.L.msg_used - kept

    def pick_models(self, limit: float=SOFT_USD) -> list[str]:
        if self.L.cost <= limit and self.L.cost + STRONG_ROOM <= HARD_USD:
            return [STRONG, FAST]
        return [FAST] if self.L.cost <= HARD_USD else []

    def infer(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.L.calls >= CALL_CAP:
                return None
            if self.model_room() < CALL_FLOOR:
                self.L.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                return None
            self.L.calls += 1
            tries = 0
            while True:
                message, status = self.raw_infer(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(BACKOFF)) or self.L.waited + BACKOFF[tries] > WAIT_BUDGET:
                    break
                if self.model_room() - BACKOFF[tries] * 1.2 < CALL_FLOOR:
                    break
                pause = BACKOFF[tries] * (0.8 + 0.4 * random.random())
                self.L.waited += BACKOFF[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.L.no_credit = True
                return None
        return None

    def raw_infer(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.L.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.L.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _as_int(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.L.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def json_query(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.infer([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = _as_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.L.broke or self.L.no_credit or self.L.calls >= CALL_CAP:
                return None
        return None

    def env_step(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.L.binding)
        payload['step'] = self.L.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.L.step = data['step'] if isinstance(data.get('step'), int) else self.L.step + 1
                if data.get('finished'):
                    self.L.finished = True
                seen = data.get('observation')
                if name in self.detail_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.L.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.L.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.L.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.L.step = int(found.group(1)) - 1
                    payload['step'] = self.L.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.L.step += 1
                    payload['step'] = self.L.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.L.broke = True
                return None
        return None

    def render(self, observation: Any) -> str:
        text = _to_json(_shrink(observation, VIEW_CHARS))
        if len(text) > OBS_CHARS:
            text = text[:OBS_CHARS] + '...(cut)'
        return text

    def shrink_history(self) -> None:
        size = sum((len(_to_json(m.get('content'))) for m in self.L.chat))
        if size <= CHAT_CHARS:
            return
        tools = [i for i, m in enumerate(self.L.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.L.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.L.rows][:20]
                self.L.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def basket_sig(self) -> str:
        return _to_json(sorted(self.L.cart_lines.items()))

    def ritual_pending(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.L.process)) and (stage, self.basket_sig() + f'#{self.L.version}') not in self.L.gate_ok

    def seal_reserve(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.L.process))
        need = 1
        if talk and self.L.msg_used < self.L.msg_limit and self.ritual_pending('order'):
            need = 2
        # cart_view + basket message + checkout
        if self.L.roles.get('cart_view') and (self.L.cart_view is None or self.L.cart_seen_at < self.L.cart_at):
            need = max(need, 2)
        if self.basket_keep() and self.L.msg_used < self.L.msg_limit:
            need = max(need, 2)
        return need

    def detail_tools(self) -> set[str]:
        page = self.L.roles.get('offer') or self.L.roles.get('open')
        return {t for t in (page, self.L.roles.get('compare')) if t}

    def attrs_tool(self) -> str | None:
        known = next((t for t, ok in self.L.facts_from.items() if ok and self.L.Catalog.id_arg_of(t)), None)
        if known:
            return known
        tool = self.L.roles.get('open')
        return tool if self.L.facts_from.get(tool or '') is not False and self.L.Catalog.id_arg_of(tool) else None

    def side_has_attrs(self) -> bool | None:
        return self.L.facts_from.get(self.L.roles.get('compare') or '')

    def side_is_bare(self) -> bool:
        seen = self.side_has_attrs()
        return seen is False or (seen is None and (self.L.facts_from.get(self.attrs_tool() or '') is True or bool(self.L.roles.get('offer'))))

class Briefing:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def parse_mission(self) -> None:
        lines = []
        for tool in self.L.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_to_json(fn.get('parameters') or {})}")
        prompt = PROMPT_BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.L.request
        if self.L.view.get('opening') is not None:
            prompt += '\n\nOPENING PROMPT_NOTE FROM THE STORE:\n' + _to_json(_shrink(self.L.view.get('opening'), 1500))
        brief = self.L.Pulse.json_query(prompt, [FAST], 4000) or {}
        if not self.L.no_credit and (not _as_list(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.L.Pulse.json_query(prompt, [STRONG, FAST], 2500) or brief
        elif not self.L.no_credit and self.dup_shelves(brief):
            again = self.L.Pulse.json_query(prompt, [STRONG], 2500) or {}
            if _as_list(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.L.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.L.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.L.schemas}
        self.sniff_roles()
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.L.roles.get('buy'), self.L.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _to_json(self.L.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.L.Pulse.json_query(ask, [FAST, STRONG], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = _as_int(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _as_list(entry.get('musts')) if m]
            self.L.items.append(GoodsLine(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[GoodsLine] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.L.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and _plain(m.shelf) == _plain(item.shelf) and (not any((_plain(a.split(':')[0]) == _plain(b.split(':')[0]) and _plain(a.partition(':')[2]) != _plain(b.partition(':')[2]) for a in m.rules for b in item.rules)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.rules += [m for m in item.rules if m not in twin.rules]
                self.L.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.L.items = merged
        if not self.L.items:
            self.L.items = [GoodsLine('the requested product', '', 1, [])]
        for pair in _as_list(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_as_int(pair.get('item')) or 0), became.get(_as_int(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.L.items)) and (1 <= main <= len(self.L.items)) and ((part, main) not in self.L.bound):
                    self.L.bound.append((part, main))
        for lookup in _as_list(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.rules for item in self.L.items)):
                key = _plain(text.split(':')[0])
                text = next((m for item in self.L.items for m in item.rules if _plain(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.L.items, 1) if text in item.rules), 1)
            if text and (number, text) not in self.L.refs:
                self.L.refs.append((number, text))
        self.strip_pointers()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.L.budget = budget
        limit = _as_int(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.L.msg_limit = limit
        elif self.L.roles.get('ask'):
            self.L.msg_limit = 3
        steps = _as_int(brief.get('steps'))
        if steps is not None and 10 <= steps < self.L.max_steps:
            self.L.max_steps = steps
        for entry in _as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.L.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _as_int(entry.get('messages')) or 0)})
        self.inject_process_heuristics()
        if self.L.roles.get('ask'):
            talk = [p['messages'] for p in self.L.process if p['when'] in ('order', 'end') and p['messages']]
            # always keep room for the pre-checkout basket reveal when the store allows messages
            keep = self.L.Pulse.basket_keep()
            self.L.reserve = min(max([0, *talk, keep]), self.L.msg_limit) if (talk or keep) else 0

    def sniff_roles(self) -> None:
        """Fill missing tool roles from schema names/descriptions when the brief omits them.

        Prefer mutate needles for cart and list/view needles for cart_view so a bag/cart
        list tool is not claimed as the mutate tool.
        """
        taken = {v for v in self.L.roles.values() if v}

        def claim(job: str, *need: str) -> None:
            if self.L.roles.get(job):
                return
            for t in self.L.tools:
                fn = t.get('function') or {}
                name = fn.get('name') or ''
                if name in taken:
                    continue
                blob = ((fn.get('description') or '') + ' ' + name).lower().replace('_', ' ')
                if any(n in blob for n in need):
                    self.L.roles[job] = name
                    taken.add(name)
                    return

        claim('search', 'search', 'find', 'query', 'browse', 'locate listing')
        claim('open', 'attribute', 'spec', 'datasheet', 'detail', 'property', 'describe', 'technical', 'open page')
        claim('offer', 'offer', 'price and stock', 'availability')
        claim('compare', 'compare', 'side', 'several', 'weigh')
        claim('ask', 'message', 'customer', 'shopper', 'ask', 'say_user', 'write_customer', 'ask buyer')
        # mutate cart before view so add/set tools win over list/view tools
        claim('cart', 'set to cart', 'settocart', 'hold to cart', 'hold_to', 'add to cart', 'add_item', 'update cart')
        claim('cart_view', 'baglist', 'bag list', 'view cart', 'cart view', 'review cart', 'cart content', 'show cart', 'bag content', 'peek', 'selection_items', 'list the')
        claim('cart', 'cart', 'bag', 'basket')
        claim('buy', 'checkout', 'purchase', 'buyitem', 'buy item', 'settle', 'order', 'conclude')
        claim('refuse', 'decline', 'refuse', 'no purchase', 'walk away', 'give_up', 'pass order', 'passorder')
        claim('report', 'statusreport', 'status report', 'account_report', 'usage_report', 'allowance', 'report')

    def inject_process_heuristics(self) -> None:
        """Fill process gaps the brief missed when the request text clearly asks for them.

        Cart-view + post-view line rereads are enforced in SealWatch code,
        not as process-list entries (avoids burning ritual steps when the brief already covers them).
        """
        req = (self.L.request or '').lower()
        have = {' '.join(p['step'].lower().split()) for p in self.L.process}

        def add(step: str, when: str, messages: int) -> None:
            key = ' '.join(step.lower().split())
            if key in have:
                return
            self.L.process.append({'step': step, 'when': when, 'messages': messages})
            have.add(key)

        if re.search(r'\b(view|review|look at|check|show|read)\b.{0,40}\b(cart|basket|bag)\b', req):
            add('view the cart contents', 'order', 0)
        if re.search(
            r'\b(confirm|approv|acknowledge|ok with|agree)\b.{0,40}\b(shopper|me|customer|before (you |we )?(order|check|buy))\b',
            req,
        ) or re.search(r'\bbefore (you |we )?(order|check out|checkout|buy)\b.{0,60}\b(ask|tell|show|confirm)\b', req):
            add('confirm the plan with the shopper', 'order', 1 if self.L.roles.get('ask') else 0)
        if re.search(r'\b(account|usage|allowance)\b.{0,30}\b(report|check|read)\b', req) or re.search(
            r'\b(report|check)\b.{0,30}\b(account|usage|allowance)\b', req
        ):
            add('run the store account/usage report', 'order', 0)
        if self.L.roles.get('report') and re.search(r'\b(report|allowance|usage|account|status)\b', req):
            add('call the store report tool before finishing', 'order', 0)

    def dup_shelves(self, brief: dict[str, Any]) -> bool:
        shelves = [_plain(str(e.get('shelf') or '')) for e in _as_list(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def strip_pointers(self) -> None:
        for number, item in enumerate(self.L.items, 1):
            pointers = [need for n, need in self.L.refs if n == number and need in item.rules]
            rest = [m for m in item.rules if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = PROMPT_UNMIX + '\n\nREQUEST:\n' + self.L.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.L.Pulse.json_query(prompt, [STRONG, FAST], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _as_list(got.get('describe_other'))]]
            if not moved:
                continue
            item.rules = [m for m in item.rules if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.rules = [joined if m == pointers[0] else m for m in item.rules]
            self.L.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.L.refs]
            self.L.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def is_pointer_need(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.L.refs] or need in [str(x) for x in _as_list(self.L.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.L.items[number - 1]
        if any(({w for w in _stems(other.name) if len(w) >= 4} & _stems(value) for other in self.L.items if other is not item)):
            return False
        cat_key = self.L.Catalog.cat_field()
        keys: set[str] = set()
        read = 0
        for record in self.L.full.values():
            if cat_key and (not self.label_matches(item, str(record.get(cat_key) or ''))):
                continue
            block = _attr_block(record) or {}
            if block:
                read += 1
                keys |= {_plain(k) for k in block}
        return read >= 3 and (not any((_stems(k) & _stems(prop) for k in keys)))

    def mission_text(self) -> str:
        parts = []
        many = len(self.L.items) > 1
        for number, item in enumerate(self.L.items, 1):
            live = [m for m in item.rules + item.told if m not in item.dropped]
            told = [m for m in live if self.is_pointer_need(number, m)]
            needs = [m + (f' (the page it points at gives: {self.L.ref_values[m]})' if m in self.L.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
            parts.append(text)
        brief = self.L.brief
        if self.L.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.L.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in _as_list(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in _as_list(brief.get('musts')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.L.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Forge does it when you check out)', 'end': 'at an ending without a purchase (Forge does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.L.Pulse.attrs_tool()
        if facts and (self.L.facts_from.get(facts) is True or self.L.roles.get('offer')):
            bare = sorted((t for t in self.L.Pulse.detail_tools() | {self.L.roles.get('offer') or ''} if t and self.L.facts_from.get(t) is False))
            pages = sorted((t for t in self.L.Pulse.detail_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def desk_status(self) -> str:
        left = self.L.Pulse.steps_left()
        text = f'[forge] step {self.L.step} of {self.L.max_steps} used, {left} left.'
        if left <= self.L.Pulse.seal_reserve() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.L.items) + 5 and self.L.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.L.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.L.buys == 0 and (not self.L.cart_touched) and (self.L.step >= self.L.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.L.Pulse.clock() >= WIND_DOWN and left > self.L.Pulse.seal_reserve() + 1:
            text += ' Wall-clock is running low: settle on picks, fill the cart and check out soon.'
        live = {need for _, need in self.L.Catalog.live_needs()}
        lookups = [need for _, need in self.L.refs if need in live and need not in self.L.ref_values]
        if lookups and self.L.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.L.leads:
            text += '\n[review candidates] ' + ', '.join(self.L.leads)
        plan = self.L.Catalog.compat_plan() if self.L.bound and self.L.buys == 0 else ''
        if plan:
            text += '\n' + plan
        wide = self.L.Catalog.neighbour_notes()
        if wide:
            text += '\n' + wide
        table = self.L.AttrBench.attr_table()
        if table:
            text += '\n' + table
        if self.L.notes:
            text += '\n[changes so far] ' + ' | '.join(self.L.notes[-4:])
        text += '\n[ledger]\n' + self.mission_text()
        return text

    def run_interview(self) -> str:
        """Ask the shopper; pack several planned properties per message while list-asks stay on."""
        self.L.asked = True
        if self.L.Pulse.ask_slots() <= 0:
            return ''
        self.L.Catalog.widen_search()
        count = self.L.msg_limit - self.L.msg_used
        plan_n = max(count, LIST_SIZE) if self.L.list_asks else count
        self.L.lanes = self.build_lanes(plan_n)
        self.L.lane_found = {n: 0 for n in self.L.lanes}
        self.L.lane_spent = {n: 0 for n in self.L.lanes}
        self.L.lane_used = {n: set() for n in self.L.lanes}
        self.L.lane_misses = {n: 0 for n in self.L.lanes}
        self.L.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.L.Pulse.ask_slots() > 0 and (not self.L.finished) and (self.L.Pulse.steps_left() > 6) and (self.L.Pulse.clock() < QUESTION_BY):
            batch = self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.L.lane_found if self.L.lane_found[n] > 0]
        searched = self.L.Catalog.value_query(found_items) if found_items else ''
        if told:
            self.L.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "Forge's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.L.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.L.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.L.queued)
            self.L.queued = []
        return summary

    def late_question(self) -> str:
        """If a message remains that checkout itself does not need, ask one more planned property before buy."""
        if self.L.late_asked or not self.L.roles.get('ask') or self.L.msg_used >= self.L.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.L.process)):
            return ''
        if self.L.Pulse.steps_left() < LATE_STEPS or self.L.Pulse.clock() > LATE_BY or not self.L.Pulse.pick_models():
            return ''
        self.L.late_asked = True
        if not self.L.lanes:
            self.L.lanes = self.build_lanes(LIST_SIZE if self.L.list_asks else 1)
            self.L.lane_found = {n: 0 for n in self.L.lanes}
            self.L.lane_spent = {n: 0 for n in self.L.lanes}
            self.L.lane_used = {n: set() for n in self.L.lanes}
            self.L.lane_misses = {n: 0 for n in self.L.lanes}
        told: list[str] = []
        budget = self.L.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.L.trace.append({'kind': 'late_ask', 'skipped': 'no property left or no message sent'})
            return ''
        if self.L.budget != budget:
            told.append(f'budget now {self.L.budget}')
        number, question, answer = batch[0]
        self.L.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.L.version += 1
        summary = (
            f'Forge asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\n'
            'New firm requirements from this answer: ' + '; '.join(told)
            + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        )
        pointed = [need for _, need in self.L.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.L.Catalog.value_query([number])
        return summary + ('\n' + searched if searched else '')

    def should_interview(self) -> bool:
        if self.L.asked or not self.L.roles.get('ask') or self.L.Pulse.ask_slots() <= 0:
            return False
        if self.L.Pulse.steps_left() <= 8 or self.L.Pulse.clock() > ASK_BY:
            return False
        opened = len(self.L.full)
        if opened >= 3 * len(self.L.items) + 1:
            return True
        return self.L.step >= max(8, self.L.max_steps // 4)

    def ask_payload(self, text: str) -> dict[str, Any] | None:
        name = self.L.roles.get('ask')
        schema = self.L.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _schema_kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults: list[str] = []
        args = _coerce(args, schema, 'args', faults)
        return None if faults else args

    def pick_lane(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.L.plan_order)}
        scarce = self.L.msg_limit <= SCARCE_MESSAGES and (not self.L.list_asks)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _plain(label) not in used[number] and (not self.told_about(number, label))]
            if left:
                cap = LIST_SIZE if self.L.list_asks else 1
                return (number, left[:cap])
        return (0, [])

    def lane_prompt(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.L.items[number - 1].name}?"
        return self.phrase_q(number, labels[0], self.sample_vals(number, labels[0]) if shown else None)

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """Longest leading slice of labels the ask tool accepts whole (schema may truncate long text)."""
        for size in range(len(labels), 0, -1):
            text = self.lane_prompt(number, labels[:size], shown)
            args = self.ask_payload(text)
            if args and text in args.values():
                return (labels[:size], text)
        return ([], '')

    def told_about(self, number: int, label: str) -> bool:
        mine = _plain(label)
        for t in self.L.items[number - 1].told:
            prop = _plain(t.split(':', 1)[0])
            if mine == prop or (len(prop.split()) >= 2 and prop in mine) or (len(mine.split()) >= 2 and mine in prop):
                return True
            if _same_prop(_stems(mine), _stems(prop)):
                return True
        return False

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send next multi-property (or single) question, parse answer, settle labels."""
        for _ in range(LIST_SIZE):
            self.L.follow_up = [f for f in self.L.follow_up if not self.told_about(f[0], f[1][0])]
            if self.L.follow_up:
                number, labels, shown = self.L.follow_up.pop(0)
            else:
                number, labels = self.pick_lane(self.L.lanes, self.L.lane_found, self.L.lane_spent, self.L.lane_used)
                shown = False
            if not labels:
                return []
            fitted, text = self.fit_ask(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.fit_ask(number, labels, False)
            if fitted:
                break
            self.L.lane_used.setdefault(number, set()).add(_plain(labels[0]))
            self.L.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.deliver_qs([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.L.lane_spent[number] = self.L.lane_spent.get(number, 0) + 1
        had = [len(item.told) for item in self.L.items]
        heard = self.parse_answers(batch, (FAST, STRONG))
        if _as_list(heard.get('needs')) and self.L.Pulse.pick_models():
            heard = self.parse_answers(batch, (STRONG, FAST)) or heard
        if len(labels) > 1:
            heard.pop('free', None)
        self.apply_answers(heard, told)
        new = [need for item, size in zip(self.L.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.L.items, had), 1):
            if len(item.told) > size:
                self.L.lane_found[n] = self.L.lane_found.get(n, 0) + 1
        crowded = any((_as_int(x) == 1 for x in _as_list(heard.get('crowded'))))
        self.settle(number, labels, shown, new, crowded)
        return batch

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        """After a multi-property answer, mark used labels and rotate so unanswered ones come next."""
        used = self.L.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(_plain(labels[0]))
            if new or not crowded:
                return
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _plain(head) not in used:
                self.L.follow_up.insert(0, (number, [head], False))
            elif not shown and len(self.sample_vals(number, labels[0])) >= 2:
                self.L.follow_up.insert(0, (number, labels, True))
            return
        if crowded and not new:
            self.L.list_asks = False
            self.L.trace.append({'kind': 'lists_off', 'item': number})
            self.L.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_label(labels, new) if new else -1
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((_plain(label) for label in labels[:max(hit, 0) + 1]))
        rest = [label for label in self.L.lanes.get(number) or [] if label not in labels]
        self.L.lane_misses[number] = 0 if hit >= 0 else self.L.lane_misses.get(number, 0) + 1
        self.L.lanes[number] = rest + ahead if self.L.lane_misses[number] and self.L.lane_misses[number] % 2 == 0 else ahead + rest
        self.L.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Earliest label a multi-property reply answered; -1 if none map."""
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(_stems(label) & _stems(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(_root_set(label) & _root_set(value)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def keys_for_lane(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.L.items[number - 1]
        settled = [_stems(m.split(':', 1)[0]) for m in item.rules]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.item_for_label(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.L.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_same_prop(prop, _stems(key)) for prop in settled)) and (not re.search(SKIP_PROP, key))]
        return [key for key in keys if not re.search(CATCH_PROP, key)] + [key for key in keys if re.search(CATCH_PROP, key)]

    def item_for_label(self, label: str) -> int:
        if not label:
            return 0
        words = _plain(label)
        long = _root_set(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.L.items, 1):
            if words == _plain(item.shelf or ''):
                score = 3.0
            elif words == _plain(item.name):
                score = 2.0
            else:
                mine = _root_set(item.shelf or '') | _root_set(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def label_matches(self, item: GoodsLine, label: str) -> bool:
        return bool(label) and item in self.L.items and (self.item_for_label(label) == self.L.items.index(item) + 1)

    def opened_for_item(self, number: int) -> int:
        cat_key = self.L.Catalog.cat_field()
        item = self.L.items[number - 1]
        return sum((1 for record in self.L.full.values() if cat_key and self.label_matches(item, str(record.get(cat_key) or '')) and _attr_block(record)))

    def build_lanes(self, count: int) -> dict[int, list[str]]:
        table = self.L.Catalog.attr_census()
        settled = []
        for number, item in enumerate(self.L.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.rules) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.L.Catalog.reject_key_note(label, name, values) + (' [several features in one: ask last]' if re.search(CATCH_PROP, name) else ''))
        prompt = PROMPT_ASK.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.L.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _as_list(self.L.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.L.Pulse.json_query(prompt, [STRONG, FAST], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.L.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _as_list(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _as_int(pick.get('item') or 1) or 1), len(self.L.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((_plain(label) != _plain(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_plain(label))
                if number not in self.L.plan_order:
                    self.L.plan_order.append(number)
        for number in lanes:
            keys = self.keys_for_lane(number, table)
            read = max(1, self.opened_for_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.item_for_label(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _plain(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _plain(label) in own:
                    return own[_plain(label)]
                near = [s for k, s in own.items() if _same_prop(_stems(label), _stems(k)) or _same_prop(_stems(k), _stems(label))]
                return max(near) if near else 0.0
            hinted[number] = {_plain(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((_plain(spelled(pick)) != _plain(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= KEEP_SHARE and (not re.search(CATCH_PROP, k))][:KEEP_KEYS]:
                if any((_plain(key) == _plain(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _plain(other) not in hinted[number] and share_of(other) + KEEP_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _plain(other) != _plain(key)]
                if weak is not None and len(self.L.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((_plain(x) in hinted[number] and all((_plain(x) != _plain(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((_plain(key) != _plain(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_PROP, _plain(x))] + [x for x in lanes[number] if re.search(CATCH_PROP, _plain(x))]
        self.L.trace.append({'kind': 'plan', 'ask': _shrink(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def phrase_q(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.L.items[number - 1].name}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

    def sample_vals(self, number: int, label: str) -> list[str]:
        want = _plain(label)
        settled = ' '.join(self.L.items[number - 1].rules).lower()
        for _, name, _, values in self.L.Catalog.attr_census():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def deliver_qs(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.L.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.L.finished or self.L.Pulse.steps_left() <= floor or self.L.msg_used >= self.L.msg_limit:
                break
            args = self.ask_payload(text)
            if not args:
                break
            reply = self.L.Pulse.env_step(tool, args)
            if reply is None:
                break
            self.L.msg_used += 1
            answer = reply.get('observation')
            words = self.L.Catalog.obs_speech(answer)
            sent.append((number, text, words))
            self.L.done_log.append(f'step {self.L.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.L.Pulse.basket_sig()})')
            self.L.done_steps.append(self.L.step)
            self.L.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def apply_answers(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.L.items) > 1
        for entry in _as_list(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _as_int(entry.get('item')) or 1), len(self.L.items))
            need = str(entry['need']).strip()
            if need not in self.L.items[number - 1].told:
                self.L.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.L.refs:
                self.L.refs.append((number, need))
        for entry in _as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.L.items[min(max(1, _as_int(entry.get('item')) or 1), len(self.L.items)) - 1]
                gone = _plain(str(entry['property']).split(':')[0])
                hit = [m for m in item.rules + item.told if m not in item.dropped and _plain(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _as_list(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _as_int(entry.get('item')) or 1), len(self.L.items))
                self.L.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.L.budget = heard['budget']

    def parse_answers(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(STRONG, FAST)) -> dict[str, Any]:
        prompt = PROMPT_HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.L.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.L.Pulse.json_query(prompt, list(models), 2500) or {}

    def run_ritual(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.L.process if p['when'] == stage]
        if not steps or self.L.finished or (not self.L.Pulse.ritual_pending(stage)):
            return (True, '')
        ending = {self.L.roles.get('buy'), self.L.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_to_json(t['function'].get('parameters') or {})}" for t in self.L.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(RITUAL_ROUNDS):
            prompt = PROMPT_RITUAL.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.L.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.mission_text() + '\n\nCART NOW: ' + self.L.Catalog.basket_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.L.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.L.notes[-4:]) if self.L.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.L.msg_limit - self.L.msg_used)}   STEPS LEFT: {self.L.Pulse.steps_left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.L.Pulse.json_query(prompt, [STRONG, FAST] if nudged else [FAST, STRONG], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.apply_answers({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.L.trace.append({'kind': 'gate', 'stage': stage, 'call': _shrink(call, 200), 'steps': _shrink(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.L.version += 1
                return (False, ('Before going on, Forge did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _as_list(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.L.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.L.done_steps) and self.L.done_steps[i - 1] > self.L.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_as_int(n) or 0) in proof for n in _as_list(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.L.roles.get('ask') and (self.L.msg_used < self.L.msg_limit) and (self.L.Pulse.steps_left() >= 2):
                nudged = True
                self.L.trace.append({'kind': 'gate_unproven', 'steps': _shrink(unproven, 120)})
                self.L.done_log.append(f'(Forge: no line made since the cart last changed at step {self.L.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.L.msg_limit - self.L.msg_used} message(s) and {self.L.Pulse.steps_left()} steps are left: give the call that does it)')
                self.L.done_steps.append(-1)
                continue
            if not call or name not in self.L.schemas or name in ending:
                if not call:
                    self.L.gate_ok.add((stage, self.L.Pulse.basket_sig() + f'#{self.L.version}'))
                break
            if self.L.Pulse.steps_left() < 2 or (name == self.L.roles.get('ask') and self.L.msg_used >= self.L.msg_limit):
                break
            faults: list[str] = []
            args = _coerce(call.get('args') if isinstance(call.get('args'), dict) else {}, self.L.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.L.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.deliver_qs([(1, '', text)], 1)
                if not batch:
                    break
                self.L.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.L.Pulse.env_step(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.L.Catalog.ingest(seen)
            self.L.done_log.append(f'step {self.L.step}: {name} {_to_json(args)} -> {self.L.Pulse.render(seen)[:300]} (cart then: {self.L.Pulse.basket_sig()})')
            self.L.done_steps.append(self.L.step)
            said.append(f'{name} {_to_json(args)} -> {self.L.Pulse.render(seen)[:400]}')
        return (True, 'Before going on, Forge did the steps the request asks for:\n' + '\n'.join(said) if said else '')

class Catalog:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def extract_id(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.L.id_key and _looks_id(node.get(self.L.id_key), True):
            return node[self.L.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _guess_id_key([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.L.rows):
            return node[found].strip()
        return None

    def learn_id_key(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.L.id_key and all((_looks_id(r.get(self.L.id_key), True) for r in rows)):
            return True
        found = _guess_id_key(rows)
        if found:
            self.L.id_key = found
            return True
        return False

    def ingest(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.extract_id(node, in_list, depth)
                if ident:
                    if ident not in self.L.rows:
                        self.L.seen_order.append(ident)
                        self.L.rows[ident] = {}
                    old = self.L.rows[ident]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.L.unavailable.add(ident)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
                        self.L.unavailable.discard(ident)
                    self.L.rows[ident].update(node)
                    if _attr_block(node):
                        self.L.full[ident] = dict(self.L.rows[ident])
                    elif ident in self.L.full:
                        self.L.full[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_id_key(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def shelf_labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get('enum') or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.L.schemas.values():
            walk(schema, 0)
        return found

    def cat_field(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.L.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.L.rows))
        known = self.shelf_labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def attr_census(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.cat_field()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.L.seen_order:
            record = self.L.full.get(ident)
            if not record:
                continue
            block = _attr_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _plain(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_plain(item.shelf or ''), _plain(item.name), {w for w in _stems(item.shelf or '') | _stems(item.name) if len(w) >= 4}) for item in self.L.items]
        matched = {label for label in groups if any((_plain(label) in (shelf, name) or {w for w in _stems(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.L.Briefing.item_for_label(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.same_spec(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def reject_key_note(self, label: str, name: str, values: list[str]) -> str:
        number = self.L.Briefing.item_for_label(label)
        if not number:
            return ''
        for must in self.L.items[number - 1].rules:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _plain(prop) != name and _stems(prop) & _stems(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def same_spec(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def group_musts(self) -> list[list[str]]:
        cat_key = self.cat_field()
        groups: dict[str, list[str]] = {}
        for ident in self.L.seen_order:
            label = str(self.L.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_plain(item.shelf or ''), _plain(item.name), {w for w in _stems(item.shelf or '') | _stems(item.name) if len(w) >= 4}) for item in self.L.items]
        matched = [ids for label, ids in groups.items() if any((_plain(label) in (shelf, name) or {w for w in _stems(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.L.Briefing.item_for_label(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.L.items))]

    def id_arg_of(self, tool: str | None) -> str | None:
        schema = self.L.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _schema_kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_for(self, ident: str) -> float | None:
        value = (self.L.rows.get(ident) or {}).get(self.L.price_key) if self.L.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def can_buy(self, ident: str) -> bool:
        if self.L.SealWatch.is_dry(ident):
            return False
        value = (self.L.rows.get(ident) or {}).get(self.L.stock_key) if self.L.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def rows_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.extract_id(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.extract_id(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.extract_id(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.extract_id(r, True)] if i]
        return []

    def mark_view(self, name: str, observation: Any) -> None:
        for ident, record in self.rows_in(observation):
            keys = set(record)
            self.L.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.L.view_keys.get(ident, (-1, set()))
            self.L.view_keys[ident] = (self.L.version, (have if version == self.L.version else set()) | keys)

    def view_fresh(self, name: str, ident: str) -> bool:
        version, have = self.L.view_keys.get(ident, (-1, set()))
        keys = self.L.tool_keys.get(name)
        return bool(keys) and version == self.L.version and (keys <= have) and (ident not in self.L.gone)

    def learn_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.L.shapes:
            return self.L.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.L.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = PROMPT_SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _to_json(self.L.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _to_json(_shrink(sample, 120))[:4000]
        got = self.L.Pulse.json_query(prompt, [FAST, STRONG], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.L.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.L.shapes[name] = shape
        self.L.price_key = self.L.price_key or shape['price']
        self.L.stock_key = self.L.stock_key or shape['stock']
        self.L.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def page_meta(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.learn_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def is_price_sorted(self, observation: Any) -> bool:
        prices = [self.price_for(i) for i, _ in self.rows_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def auto_page(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.L.paged_obs = []
        pick = self.L.brief.get('pick')
        plan = self.page_meta(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.is_price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.L.pages_read + 2 <= PAGE_AUTO and (self.L.Pulse.steps_left() - self.L.Pulse.seal_reserve() - 2 >= 12):
            pages = 2
        if pages > 2 or self.L.pages_read + pages > PAGE_AUTO or self.L.Pulse.steps_left() - self.L.Pulse.seal_reserve() - pages < 12:
            return f"\n[forge] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={_to_json(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _coerce({**args, param: value}, self.L.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.L.Pulse.env_step(name, call)
            if reply is None or self.L.finished:
                break
            self.L.pages_read += 1
            seen = reply.get('observation')
            self.after_obs(name, seen)
            self.L.paged_obs.append(seen)
            shown_text = self.L.Pulse.render(seen)
            self.L.read_cache[name + _to_json(call)] = (self.L.version, self.L.step, shown_text)
            texts.append(f'{param}={_to_json(value)}: {shown_text}')
            after = self.page_meta(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.L.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[forge] {count} listings matched and Forge has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[Forge read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def price_cap_miss(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.L.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not self.L.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.rows_in(observation) if (self.price_for(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={_to_json(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[forge] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheap_rank_note(self, observations: list[Any]) -> str:
        if self.L.brief.get('pick') != 'cheapest' or not self.L.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.rows_in(seen)}
        rows = [(self.price_for(i), i) for i in found if self.can_buy(i) and self.price_for(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[forge] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def cheap_widen_hold(self) -> str:
        # When pick is cheapest, push one wider search before sealing if steps remain.
        if self.L.widen_done or self.L.brief.get('pick') != 'cheapest' or self.L.Pulse.steps_left() < 12 + 2 * len(self.L.items):
            return ''
        # A proof-passing cart with a thinning step budget should seal, not widen.
        if self.L.Pulse.steps_left() <= SWAP_TIGHT + 2 and self.L.AttrBench.cart_fully_passes():
            self.L.widen_done = True
            return ''
        self.L.widen_done = True
        lines = [f'{i} at {self.price_for(i):g}' for i in self.basket_ids() if self.price_for(i) is not None]
        return f"Not run yet (no step used). {self.L.Pulse.steps_left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def open_attrs(self, ids: list[str], keep: int) -> list[str]:
        tool = self.L.Pulse.attrs_tool()
        target = self.id_arg_of(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.L.rows and i not in self.L.full and (not self.L.SealWatch.is_dry(i))]:
            if self.L.Pulse.steps_left() <= keep or self.L.finished or self.L.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _coerce({target: ident}, self.L.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.L.Pulse.env_step(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.after_obs(tool, seen)
            self.mark_view(tool, seen)
            if isinstance(seen, dict) and self.extract_id(seen):
                self.L.facts_from[tool] = self.L.facts_from.get(tool, False) or bool(_attr_block(seen))
            texts.append(f'{ident}: ' + self.L.Pulse.render(seen)[:1500])
        if texts:
            self.L.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def hunt_cheaper(self) -> list[str]:
        if self.L.brief.get('pick') != 'cheapest':
            return []
        # Do not open more cheap candidates when a passing cart already needs its seal reserve.
        if self.L.Pulse.steps_left() <= SWAP_TIGHT and self.L.AttrBench.cart_fully_passes():
            return []
        mark = self.L.Pulse.basket_sig() + f'#{self.L.version}'
        cat_key = self.cat_field()
        if mark in self.L.cheap_marks or not cat_key:
            return []
        self.L.cheap_marks.add(mark)
        cart = self.basket_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_for(line)
            kind = str((self.L.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.L.seen_order if i not in cart and i not in self.L.full and self.can_buy(i) and (str((self.L.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.price_for(i) if self.price_for(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.price_for(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.open_attrs(picks[:CHEAP_OPENS], self.L.Pulse.seal_reserve() + 4)

    def after_obs(self, name: str, observation: Any) -> str:
        self.ingest(observation)
        lines = self.note_lines(observation)
        if not lines:
            return ''
        meaning = self.interpret_note(lines)
        if meaning:
            self.L.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen_search(self) -> None:
        tool = self.L.roles.get('compare')
        schema = self.L.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _schema_kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.L.Pulse.side_is_bare():
            self.widen_one()
            return
        if self.L.Pulse.steps_left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.group_musts()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.L.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _coerce({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.L.full)
        reply = self.L.Pulse.env_step(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.after_obs(tool, seen)
            self.mark_view(tool, seen)
            rows = self.rows_in(seen)
            if rows:
                self.L.facts_from[tool] = self.L.facts_from.get(tool, False) or any((_attr_block(r) for _, r in rows))
            self.L.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.L.full) - before})
            if rows and self.L.facts_from.get(tool) is False:
                self.widen_one()
                return
            if len(self.L.full) == before and len(picks) > 2 and (self.L.Pulse.steps_left() > 12):
                faults = []
                args = _coerce({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.L.Pulse.env_step(tool, args)
                    if reply is not None:
                        self.after_obs(tool, reply.get('observation'))

    def widen_one(self) -> None:
        tool = self.L.Pulse.attrs_tool()
        target = self.id_arg_of(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.live_needs() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_N if len(self.L.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.group_musts():
            opened = sum((1 for i in ids if i in self.L.full))
            pool = [i for i in ids if i not in self.L.full and self.can_buy(i)]
            if self.L.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_for(i) if self.price_for(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _to_json(self.L.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.open_attrs(first[:cap][:8], 22)
        self.open_attrs((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def search_skel(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.L.roles.get('search')
        schema = self.L.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _schema_kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for args in reversed(self.L.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.cat_field()) or '').strip().lower() for r in self.L.rows.values()} - {''} if self.cat_field() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.L.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.L.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.L.searched)))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.cat_field()
                shelves = {_plain(item.shelf or '') for item in self.L.items} | {_plain(item.name) for item in self.L.items}
                shelves |= {_plain(str(row.get(cat_key) or '')) for row in self.L.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_plain(v) not in shelves and (not any(({w for w in _stems(v) if len(w) >= 4} & {w for w in _stems(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.L.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        used = [_to_json(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _to_json(a[k]) == best))
                return (tool, keep, field)
        return None

    def need_tokens(self, number: int) -> list[str]:
        out = []
        for n, need in self.live_needs():
            if n != number or need in [m for _, m in self.L.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def value_query(self, numbers: list[int]) -> str:
        plan = self.search_skel()
        if not plan or self.L.Pulse.steps_left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.need_tokens(number)
            item = self.L.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _coerce({**base, field: ' '.join(words)}, self.L.schemas.get(tool), 'args', faults)
                if faults or self.L.finished:
                    break
                reply = self.L.Pulse.env_step(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.after_obs(tool, observation)
                cat_key = self.cat_field()
                found = [i for i, _ in self.rows_in(observation)]
                cap = self.L.budget if isinstance(self.L.budget, (int, float)) else None
                prices = [self.price_for(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.can_buy(i) and (not cat_key or self.L.Briefing.label_matches(item, str((self.L.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_for(i) or 0) <= cap)]
                seen.append(f'{_to_json(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.open_attrs([i for i in rows if i not in self.L.full][:min(3, budget)], self.L.Pulse.seal_reserve() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.L.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Forge searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def obs_speech(self, observation: Any) -> str:
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
        return ' '.join(found)[:800] if found else _to_json(observation)[:400]

    def note_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.extract_id(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.extract_id(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.L.heard]

    def interpret_note(self, lines: list[str]) -> str:
        self.L.heard.update(lines)
        prompt = PROMPT_NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.L.Briefing.mission_text() + '\n\nTHE PROMPT_NOTE:\n' + '\n'.join(lines)
        got = self.L.Pulse.json_query(prompt, [FAST, BACKUP], 2500) or {}
        gone = [str(g) for g in _as_list(got.get('gone')) if isinstance(g, (str, int))]
        self.L.unavailable.update((g for g in gone if g in self.L.rows))
        self.L.note_sold = bool(gone) or got.get('sold_out') is True
        self.L.note_named = any((g in self.L.rows for g in gone))
        if self.L.note_sold and len(lines) == 1:
            self.L.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.L.budget):
            self.L.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _as_list(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _as_list(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.L.items[min(max(1, _as_int(entry.get('item')) or 1), len(self.L.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.rules + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _plain(m.split(':')[0]) == _plain(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.L.items[min(max(1, _as_int(entry.get('item')) or 1), len(self.L.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.rules + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.L.brief.get('pick') == 'cheapest' and (drops or gone or self.L.note_sold or any((c.startswith('budget now') for c in changed))):
            self.L.widen_done = False
            self.L.cheap_marks = set()
        if changed or summary:
            self.L.version += 1
            self.L.recent = []
            self.L.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def live_needs(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.L.items, 1):
            for need in item.rules + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def attr_evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.extract_id(node, False, depth)
                if ident and _attr_block(node) and (ident not in ids):
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.live_needs()
        if not ids or not needs:
            return ''
        many = len(self.L.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _attr_block(self.L.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _same_prop(_stems(prop), _stems(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def basket_text(self) -> str:
        if self.L.cart_view is not None and self.L.cart_seen_at >= self.L.cart_at:
            return _to_json(_shrink(self.L.cart_view, 300))[:3000]
        return "lines by Forge's record of the cart calls (listing: units): " + _to_json(self.L.cart_lines)

    def qty_of(self, ident: str) -> int:
        if self.L.cart_view is not None and self.L.cart_seen_at >= self.L.cart_at:
            stack: list[Any] = [self.L.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_as_int(v) for k, v in node.items() if not isinstance(v, str) and _as_int(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _plain(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.L.cart_lines.get(ident, 1)

    def basket_ids(self) -> list[str]:
        if self.L.cart_view is not None and self.L.cart_seen_at >= self.L.cart_at:
            text = _to_json(self.L.cart_view)
            return [i for i in self.L.rows if i in text]
        return list(self.L.cart_lines)

    def listing_blob(self, ident: str) -> str:
        record = self.L.full.get(ident) or self.L.rows.get(ident)
        return _to_json(_shrink(record, 400))[:3000] if record else '(no record seen)'

    def other_listings(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.L.seen_order))
        if self.L.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.L.full, self.price_for(i) if self.price_for(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.L.SealWatch.is_dry(ident):
                continue
            record = self.L.full.get(ident) or self.L.rows.get(ident)
            lines.append(_to_json(_shrink(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def compat_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.L.seen_order:
            record = self.L.full.get(ident) or self.L.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = _compat_marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def compat_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def ids_for_item(self, number: int) -> list[str]:
        cat_key = self.cat_field()
        item = self.L.items[number - 1]
        return [i for i in self.L.seen_order if cat_key and self.L.Briefing.label_matches(item, str((self.L.rows.get(i) or {}).get(cat_key) or ''))]

    def compat_faults(self) -> list[str]:
        if not self.L.bound:
            return []
        index = self.compat_index()
        cart = self.basket_ids()
        faults = []
        for part, main in self.L.bound:
            devices = [i for i in cart if i in self.ids_for_item(main)]
            parts = [i for i in cart if i in self.ids_for_item(part)]
            if len(devices) != 1 or not parts:
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.compat_set(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.ids_for_item(part) if self.compat_set(i, index) & codes and (not self.L.SealWatch.is_dry(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.L.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.compat_plan(short=True)))
        return faults

    def compat_plan(self, short: bool=False) -> str:
        if not self.L.bound:
            return ''
        index = self.compat_index()
        lines = []
        for main in sorted({m for _, m in self.L.bound}):
            parts = [p for p, m in self.L.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.ids_for_item(p):
                    if self.L.SealWatch.is_dry(i):
                        continue
                    for code in self.compat_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.ids_for_item(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.L.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.L.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def wrong_neighbour(self, number: int, label: str) -> str:
        """A seen category label that covers every word of the item's name when this label does not; '' otherwise."""
        item = self.L.items[number - 1]
        cat_key = self.cat_field()
        wanted = _root_set(item.name)
        if item.shelf.strip() or not cat_key or len(wanted) < 2:
            return ''
        have = len(_root_set(label) & wanted)
        seen = {str(r.get(cat_key) or '') for r in self.L.rows.values()}
        better = [l for l in seen if l and wanted <= _root_set(l)]
        return min(better, key=len) if better and have < len(wanted) else ''

    def neighbour_notes(self) -> str:
        """Says once, as soon as it is seen, which category label covers the whole of what was asked."""
        cat_key = self.cat_field()
        if not cat_key:
            return ''
        said = []
        for number, item in enumerate(self.L.items, 1):
            for label in sorted({str(r.get(cat_key) or '') for r in self.L.rows.values()}):
                whole = self.wrong_neighbour(number, label)
                if label and whole and whole not in self.L.whole_said:
                    self.L.whole_said.add(whole)
                    said.append(f"[category] {whole!r} is the category for the whole of what was asked ({item.name}); {label!r} is a neighbouring one, so buy from {whole!r}: list it with the category search and open its listings.")
        return '\n'.join(said)

class AttrBench:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def owner_num(self, ident: str) -> int:
        """Which requested item a listing belongs to, by its category label; 0 when it belongs to none."""
        cat_key = self.L.Catalog.cat_field()
        label = str((self.L.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        if not label:
            return 1 if len(self.L.items) == 1 else 0
        owner = self.L.Briefing.item_for_label(label)
        if owner or len(self.L.items) != 1:
            return owner
        seen = {str(r.get(cat_key) or '') for r in self.L.rows.values()}
        return 0 if any(self.L.Briefing.item_for_label(other) for other in seen) else 1

    def carrier_keys(self, number: int, prop: str, alts: list[frozenset[str]]) -> set[str]:
        """Keys of a neighbouring property that this store uses to state the wanted value on other listings."""
        words = {w for w in _stems(prop) if len(w) >= 4}
        out: set[str] = set()
        for ident, record in self.L.full.items():
            if self.owner_num(ident) != number:
                continue
            for key, value in (_attr_block(record) or {}).items():
                name = _plain(key)
                if _same_prop(_stems(prop), _stems(key)) or not any(w in name.replace(' ', '') for w in words):
                    continue
                if _attr_holds(alts, value):
                    out.add(name)
        return out

    def carrier_against(self, number: int, prop: str, alts: list[frozenset[str]], block: dict[str, Any]) -> str | None:
        """A neighbouring key that states the wanted value on other listings of this item and something else here."""
        for carrier in self.carrier_keys(number, prop, alts):
            against = next((k for k, v in block.items() if _plain(k) == carrier and not _attr_holds(alts, v)), None)
            if against:
                return against
        return None

    def score_need(self, ident: str, number: int, need: str) -> tuple[str, str]:
        """One requirement on one opened listing: (status, the store's words that show it). Only the store's own record is read.

        One naming attribute that states the value is enough. Another key that also names the same property
        is not a conflict. Conflict is reserved for a neighbouring carrier key that states this value on
        other listings and something else here.
        """
        record = self.L.full.get(ident) or {}
        block = _attr_block(record) or {}
        prop, sep, value = need.partition(':')
        prop, value = prop.strip(), value.strip() if sep else prop.strip()
        if need in [m for _, m in self.L.refs] or self.L.Briefing.is_pointer_need(number, need) or (need not in self.L.attr_learned and re.search(ATTR_SKIP, value, re.I)):
            return ('skip', '')
        learned = self.L.attr_learned.get(need)
        if learned:
            keys = {k: v for k, v in block.items() if _plain(k) in learned['keys'] or _same_prop(_stems(prop), _stems(k))}
            good = [k for k, v in keys.items() if (_plain(k), str(v)) in learned['yes']]
            if good:
                against = None if re.match(ATTR_NOT, value, re.I) else self.carrier_against(number, prop, _attr_alts(value), block)
                if against:
                    return ('conflict', f'{good[0]}={str(block[good[0]])[:40]} but {against}={str(block[against])[:40]}')
                return ('pass', f'{good[0]}={str(block[good[0]])[:50]}')
            if keys:
                first = next(iter(keys))
                return ('differs', f'{first}={str(keys[first])[:40]}')
            other = next((k for k, v in block.items() if (_plain(k), str(v)) in learned['yes']), None)
            if other:
                return ('weak', f'{other}={str(block[other])[:40]}')
        named = {k: v for k, v in block.items() if _same_prop(_stems(prop), _stems(k))}
        negative = bool(re.match(ATTR_NOT, value, re.I))
        alts = _attr_alts(re.sub(ATTR_NOT, '', value, flags=re.I))
        if not alts:
            return ('skip', '')
        cat_key = self.L.Catalog.cat_field()
        label = str(record.get(cat_key) or (self.L.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
        if not negative and label and _attr_holds(alts, label):
            return ('pass', f'category {label[:40]}')
        if negative:
            hit = next((k for k, v in named.items() if _attr_holds(alts, v)), None)
            if hit:
                return ('differs', f'{hit}={str(named[hit])[:40]}')
            return ('pass', f'{next(iter(named))}={str(next(iter(named.values())))[:40]}') if named else ('weak', 'no attribute names it')
        hit = next((k for k, v in named.items() if _attr_holds(alts, v)), None)
        if hit:
            against = self.carrier_against(number, prop, alts, block)
            if against:
                return ('conflict', f'{hit}={str(named[hit])[:40]} but {against}={str(block[against])[:40]}')
            return ('pass', f'{hit}={str(named[hit])[:50]}')
        if named:
            first = next(iter(named))
            return ('differs', f'{first}={str(named[first])[:40]}')
        other = next((k for k, v in block.items() if _attr_holds(alts, v)), None)
        if other:
            return ('weak', f'{other}={str(block[other])[:40]}')
        text = ' '.join(str(v) for k, v in record.items() if isinstance(v, str))
        return ('weak', 'title or description') if _attr_holds(alts, text) else ('unknown', 'no attribute states it')

    def attr_lines(self, number: int) -> list[tuple[str, str]]:
        """Distinct attribute lines on opened listings of an item, those on most listings first."""
        per: dict[tuple[str, str], int] = {}
        carried: dict[str, int] = {}
        for ident, record in self.L.full.items():
            if self.owner_num(ident) != number:
                continue
            for key, value in (_attr_block(record) or {}).items():
                if re.search(SKIP_PROP, _plain(key)):
                    continue
                per[(str(key), str(value))] = per.get((str(key), str(value)), 0) + 1
                carried[_plain(key)] = carried.get(_plain(key), 0) + 1
        return sorted(per, key=lambda line: (-carried[_plain(line[0])], -per[line]))

    def learn_attrs(self) -> None:
        """Have the light model read each requirement against the store's own attribute lines, once per new set of lines."""
        for number, need in self.L.Catalog.live_needs():
            if self.L.attr_asks >= ATTR_LEARNS or not self.L.Pulse.pick_models() or self.L.finished:
                return
            if need in [m for _, m in self.L.refs]:
                continue
            lines = self.attr_lines(number)[:ATTR_LINES]
            known = self.L.attr_learned.get(need) or {'keys': set(), 'yes': set(), 'seen': set()}
            if len(lines) < 2 or all((_plain(k), v) in known['seen'] for k, v in lines):
                continue
            self.L.attr_asks += 1
            shown = '\n'.join(f'{i}. {k} = {v[:70]}' for i, (k, v) in enumerate(lines, 1))
            got = self.L.Pulse.json_query(PROMPT_ATTR + f'\n\nREQUIREMENT: {need}\n\nLINES:\n{shown}', [FAST, STRONG], 1500)
            if not isinstance(got, dict) or not isinstance(got.get('yes'), list):
                continue
            yes = {(_plain(lines[n - 1][0]), lines[n - 1][1]) for n in (_as_int(x) for x in got['yes']) if n and 1 <= n <= len(lines)}
            names = {_plain(k) for k in _as_list(got.get('keys')) if isinstance(k, str)}
            self.L.attr_learned[need] = {'keys': known['keys'] | names, 'yes': known['yes'] | yes, 'seen': known['seen'] | {(_plain(k), v) for k, v in lines}}
            self.L.trace.append({'kind': 'proof_learn', 'need': need[:80], 'keys': sorted(names)[:6], 'yes': len(yes), 'lines': len(lines)})

    def score_listing(self, ident: str, number: int) -> tuple[int, list[str], int]:
        """All requirements of an item on a listing: (lowest score, one note per requirement, how many could be judged)."""
        worst, notes, judged = (4, [], 0)
        for owner, need in self.L.Catalog.live_needs():
            if owner != number:
                continue
            status, shown = self.score_need(ident, number, need)
            if status == 'skip':
                continue
            judged += 1
            worst = min(worst, ATTR_SCORE[status])
            notes.append(f"{need.split(':', 1)[0].strip()} {status}" + (f' ({shown})' if shown else ''))
        return (worst, notes, judged)

    def budget_room(self, number: int, skip: Any=()) -> float | None:
        """How much one unit-line of this item may cost and keep the order inside the budget."""
        if not isinstance(self.L.budget, (int, float)) or isinstance(self.L.budget, bool):
            return None
        if self.L.brief.get('budget_scope') == 'each':
            return float(self.L.budget)
        taken = sum((self.list_price(i) or 0.0) * (self.L.Catalog.qty_of(i) or 0) for i in self.L.Catalog.basket_ids() if i not in skip and self.owner_num(i) != number)
        return float(self.L.budget) - taken

    def list_price(self, ident: str) -> float | None:
        """The listing's price, also before the result shape has named the price key."""
        price = self.L.Catalog.price_for(ident)
        if price is not None:
            return price
        record = self.L.rows.get(ident) or {}
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and re.search('price|cost|amount', _plain(key)):
                return float(value)
        return None

    def ranked_rows(self, number: int, skip: Any=()) -> list[dict[str, Any]]:
        """Opened, buyable listings of an item inside the budget: strongest evidence first, then cheapest."""
        room = self.budget_room(number, skip)
        qty = self.L.items[number - 1].qty
        cat_key = self.L.Catalog.cat_field()
        rows = []
        for ident in self.L.full:
            if ident in skip or not self.L.Catalog.can_buy(ident) or self.owner_num(ident) != number:
                continue
            if cat_key and self.L.Catalog.wrong_neighbour(number, str((self.L.rows.get(ident) or {}).get(cat_key) or '')):
                continue
            price = self.list_price(ident)
            if room is not None and price is not None and price * qty > room + 1e-09:
                continue
            score, notes, judged = self.score_listing(ident, number)
            rows.append({'id': ident, 'price': price, 'score': score, 'notes': notes, 'judged': judged})
        rows.sort(key=lambda r: (-r['score'], r['price'] if r['price'] is not None else float('inf')))
        return rows

    def attr_table(self) -> str:
        """The table the agent reads, shown again only when it has changed."""
        lines = []
        for number, item in enumerate(self.L.items, 1):
            rows = [r for r in self.ranked_rows(number) if r['judged']][:ATTR_SHOW]
            if rows:
                lines.append(f'item {number} ({item.name}):')
                lines += [f"  {r['id']} price {r['price']}: {'; '.join(r['notes'])[:170]}" for r in rows]
        text = '\n'.join(lines)
        if not text or text == self.L.attr_shown:
            return ''
        self.L.attr_shown = text
        return "[proof table: what the store's own attributes say on opened listings; pass = an attribute naming the property states the value, conflict = another attribute, which states this value on other listings, says otherwise here, weak = stated only elsewhere, differs = the named attribute says something else]\n" + text

    def cart_line_passes(self, ident: str) -> bool:
        number = self.owner_num(ident)
        if not number or ident not in self.L.full:
            return False
        score, _notes, judged = self.score_listing(ident, number)
        return bool(judged and score >= 4)

    def remember_passed_cart(self) -> None:
        """Keep proof-passing cart lines as sell-out backups even after a voluntary cheap swap removes them."""
        for ident in self.L.Catalog.basket_ids():
            if (self.L.Catalog.qty_of(ident) or 0) <= 0:
                continue
            if self.L.SealWatch.is_dry(ident):
                continue
            if self.cart_line_passes(ident):
                self.L.passed_cart[ident] = self.L.Catalog.qty_of(ident) or self.L.cart_lines.get(ident, 1) or 1

    def cart_fully_passes(self) -> bool:
        """True when every requested item has a proof-passing, in-stock line in the cart."""
        self.learn_attrs()
        lines = [i for i in self.L.Catalog.basket_ids() if (self.L.Catalog.qty_of(i) or 0) > 0 and not self.L.SealWatch.is_dry(i)]
        if len(lines) < len(self.L.items):
            return False
        covered: set[int] = set()
        for ident in lines:
            number = self.owner_num(ident)
            if number and self.cart_line_passes(ident):
                covered.add(number)
        return covered >= set(range(1, len(self.L.items) + 1))

    def better_than_cart(self) -> list[str]:
        """Better opened listings for the cart's lines, by the store's own attributes. Empty when the cart is the best seen."""
        self.learn_attrs()
        lines = [i for i in self.L.Catalog.basket_ids() if (self.L.Catalog.qty_of(i) or 0) > 0]
        pick = self.L.brief.get('pick')
        faults = []
        if pick in ('most', 'least'):
            return faults
        # A passing cart with few steps left must seal — chasing a cheaper row burns the sell-out reserve.
        tight = self.L.Pulse.steps_left() <= SWAP_TIGHT
        if pick == 'cheapest' and tight and self.cart_fully_passes():
            return faults
        for ident in lines:
            number = self.owner_num(ident)
            if not number or ident not in self.L.full:
                continue
            score, notes, judged = self.score_listing(ident, number)
            if not judged:
                continue
            passing = [r for r in self.ranked_rows(number, lines) if r['score'] == 4 and r['judged']]
            price = self.list_price(ident)
            chosen = None
            if pick == 'cheapest':
                if score >= 4 and self.L.Pulse.steps_left() <= CHEAP_SWAP_ROOM + 2:
                    continue
                cheaper = [r for r in passing if r['price'] is not None and price is not None and r['price'] < price]
                chosen = min(cheaper, key=lambda r: r['price']) if cheaper else (passing and score <= 1 and min(passing, key=lambda r: r['price'] if r['price'] is not None else float('inf'))) or None
            elif score < 4 and passing:
                chosen = passing[0]
            if chosen:
                said = '; '.join(chosen['notes'])[:220]
                faults.append(f"{chosen['id']} (price {chosen['price']}) states every requirement: {said}. The cart's {ident} (price {price}): {'; '.join(notes)[:220]}. Take {ident} out and add {chosen['id']} at the requested quantity")
        return faults

    def thin_cart_attrs(self) -> str:
        """A cart line none of whose requirements' own attributes state the value, with no opened listing better."""
        self.learn_attrs()
        if self.L.brief.get('pick') in ('most', 'least'):
            return ''
        said = []
        for ident in [i for i in self.L.Catalog.basket_ids() if (self.L.Catalog.qty_of(i) or 0) > 0 and i in self.L.full]:
            number = self.owner_num(ident)
            if not number:
                continue
            score, notes, judged = self.score_listing(ident, number)
            if judged and score < 4 and not [r for r in self.ranked_rows(number) if r['score'] == 4 and r['judged']]:
                said.append(f"{ident}: {'; '.join(notes)[:200]}")
        return "No opened listing has an attribute naming the requirement that states it. " + ' | '.join(said) + '.' if said else ''

    def passing_leads(self) -> dict[str, list[str]]:
        """Before a decline: opened listings that state every requirement of every item. Empty unless each item has one."""
        self.learn_attrs()
        named: dict[str, list[str]] = {}
        for number in range(1, len(self.L.items) + 1):
            rows = [r['id'] for r in self.ranked_rows(number) if r['score'] == 4 and r['judged']]
            if not rows:
                return {}
            named[str(number)] = rows[:4]
        return named

    def order_swaps(self, lost: list[str], picks: list[str]) -> list[str]:
        """Sold-out replacements, strongest evidence first; listings the table passes join even when the model missed them."""
        self.learn_attrs()
        numbers = {self.owner_num(i) for i in lost} - {0}
        if len(numbers) != 1:
            return picks
        number = numbers.pop()
        rows = self.ranked_rows(number, set(lost) | set(self.L.Catalog.basket_ids()))
        cat_key = self.L.Catalog.cat_field()
        if cat_key:
            picks = [i for i in picks if not self.L.Catalog.wrong_neighbour(number, str((self.L.rows.get(i) or {}).get(cat_key) or ''))]
        scores = {r['id']: r['score'] for r in rows if r['judged']}
        passing = [r['id'] for r in rows if r['score'] == 4 and r['judged']]
        if passing:
            ordered = list(dict.fromkeys(passing + [i for i in picks if scores.get(i, 0) >= 4]))
            if self.L.brief.get('pick') == 'cheapest':
                return sorted(ordered, key=lambda i: (self.list_price(i) if self.list_price(i) is not None else float('inf'), i))
            return ordered
        merged = list(dict.fromkeys(picks + [r['id'] for r in rows if r['judged'] and r['score'] >= 2]))
        keep = [i for i in merged if scores.get(i, 1) >= 2]
        ordered = keep or merged
        if self.L.brief.get('pick') == 'cheapest':
            return sorted(ordered, key=lambda i: (scores.get(i, 1) < 4, self.list_price(i) if self.list_price(i) is not None else float('inf')))
        return sorted(ordered, key=lambda i: -scores.get(i, 1))

class Basket:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def basket_faults(self) -> list[str]:
        cat_key = self.L.Catalog.cat_field()
        ids = [i for i in self.L.Catalog.basket_ids() if (self.L.Catalog.qty_of(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.L.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.L.items, 1) if any((self.L.Briefing.label_matches(item, str(r.get(cat_key) or '')) for r in self.L.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.L.items, 1) if self.L.Briefing.label_matches(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.L.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            for i in lines:
                whole = self.L.Catalog.wrong_neighbour(n, label[i])
                if whole:
                    faults.append(f"{i} is in the category {label[i]!r}, a neighbouring one; {whole!r} is the category for the whole of what was asked ({item.name}): list that category and pick from it")
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.L.Catalog.qty_of(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
        if len(known) == len(self.L.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.L.SealWatch.is_dry(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.L.Catalog.price_for(i) for i in ids}
        if isinstance(self.L.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.L.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.L.budget < prices[i] <= 10 * self.L.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.L.budget:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.L.Catalog.qty_of(i) for i in ids))
                if self.L.budget + 1e-09 < total <= 10 * self.L.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.L.budget:g}')
        return faults

    def qty_mismatches(self) -> list[tuple[str, int, int]]:
        cat_key = self.L.Catalog.cat_field()
        if not cat_key:
            return []
        out = []
        for ident in self.L.Catalog.basket_ids():
            have = self.L.Catalog.qty_of(ident) or 0
            owners = [n for n, item in enumerate(self.L.items, 1) if self.L.Briefing.label_matches(item, str((self.L.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.L.Catalog.basket_ids() if (self.L.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.L.Briefing.label_matches(self.L.items[owners[0] - 1], str((self.L.rows.get(i) or {}).get(cat_key) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.L.items[owners[0] - 1].qty) and (not self.L.SealWatch.is_dry(ident)):
                out.append((ident, have, self.L.items[owners[0] - 1].qty))
        return out

    def repair_qtys(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.L.roles.get('cart') or ''
        schema = self.L.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _schema_kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.L.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _schema_kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.basket_is_set(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.L.Pulse.steps_left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _coerce(args, schema, 'args', faults)
            if faults:
                break
            reply = self.L.Pulse.env_step(name, args)
            if reply is None or self.L.finished:
                break
            self.track_basket(name, args)
            self.L.cart_said = _to_json(reply.get('observation'))
            self.L.Catalog.after_obs(name, reply.get('observation'))
            done.append(_to_json(args) + ' -> ' + self.L.Pulse.render(reply.get('observation'))[:300])
        self.L.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def review_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.L.version}'

    def review_basket(self, ids: list[str] | None=None, limit: float=SOFT_USD) -> str:
        lead = ids is not None
        ids = ids if lead else self.L.Catalog.basket_ids()
        key = self.review_key(ids)
        if key in self.L.checked or not self.L.Pulse.pick_models(limit) or (not ids):
            return ''
        self.L.checked.add(key)
        prompt = PROMPT_CHECK + '\n\nREQUEST:\n' + self.L.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.L.Briefing.mission_text()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.L.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.L.Catalog.basket_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.L.SealWatch.is_dry(i) else '') + self.L.Catalog.listing_blob(i) for i in ids))
        if self.L.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.L.Catalog.other_listings(ids, 15)
        got = self.L.Pulse.json_query(prompt, self.L.Pulse.pick_models(limit), 3000)
        self.L.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.L.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.L.leads = [i for i in self.L.leads if i not in ids]
        problems = [str(p) for p in _as_list(got.get('problems'))][:5]
        better = [str(b) for b in _as_list(got.get('better')) if str(b) in self.L.rows and (not self.L.SealWatch.is_dry(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def basket_ops(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.L.roles.get('cart') or ''
        schema = self.L.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _schema_kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _schema_kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _schema_kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _schema_kinds(s) and ('enum' not in s)]
        target = next((k for t, k in self.L.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _schema_kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.L.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.basket_is_set(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.L.cart_lines.get(ident, 0) if ident in self.L.Catalog.basket_ids() else 0
            if have == want:
                continue
            if have and (not self.basket_is_set(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def auto_seal(self) -> str:
        cart, buy = (self.L.roles.get('cart'), self.L.roles.get('buy'))
        if self.L.completed or not self.L.confirmed or self.L.SealWatch.is_dry(self.L.confirmed) or (not cart) or (not buy):
            return ''
        self.L.completed = True
        cat_key = self.L.Catalog.cat_field()

        def kind(ident: str) -> str:
            return str((self.L.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.L.lead_items.items():
            index = _as_int(number)
            pick = next((i for i in ids if not self.L.SealWatch.is_dry(i)), None)
            if pick and index and (1 <= index <= len(self.L.items)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.L.items[index - 1].qty
        if self.L.confirmed not in plan:
            return ''
        now = self.L.Catalog.basket_ids()
        keep = {i: self.L.cart_lines.get(i, 1) for i in now if i not in plan and (not self.L.SealWatch.is_dry(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.L.items):
            return ''
        lines = {**keep, **plan}
        calls = self.basket_ops(lines, drop)
        if not calls or len(calls) + 1 > self.L.Pulse.steps_left():
            return ''
        self.L.Catalog.open_attrs(list(lines), len(calls) + 3)
        self.review_basket(list(lines), HARD_USD)
        if not self.L.verdicts.get(self.review_key(list(lines))):
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = _coerce(args, self.L.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.L.Pulse.env_step(cart, args)
            if reply is None or self.L.finished:
                return ''
            self.track_basket(cart, args)
            self.L.Catalog.after_obs(cart, reply.get('observation'))
            done.append(_to_json(args))
        faults = []
        args = _coerce(dict(self.L.brief.get('buy_args') or {}), self.L.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.L.SealWatch.refresh_offers()
                self.review_basket(list(lines), HARD_USD)
                if not self.L.verdicts.get(self.review_key(list(lines))) or self.L.Pulse.steps_left() < 1 or self.L.finished or any((self.L.SealWatch.is_dry(i) for i in lines)):
                    break
            go, said = self.L.Briefing.run_ritual('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.L.Pulse.env_step(buy, args)
            self.L.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': _shrink(reply, 200)})
            if reply is None:
                break
            self.L.buys += 1
            if not reply.get('finished'):
                self.L.refused_at = self.L.step
            self.L.Catalog.after_obs(buy, reply.get('observation'))
            replies.append(self.L.Pulse.render(reply.get('observation')))
            if self.L.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (Forge acted instead): the independent check confirmed ' + self.L.confirmed + ', so Forge set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.L.roles.get('buy'):
            return False
        fixed = self.L.brief.get('buy_args') if isinstance(self.L.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_walkaway(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.L.roles.get('refuse'):
            return False
        fixed = self.L.brief.get('refuse_args') if isinstance(self.L.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def visible_tools(self) -> list[dict[str, Any]]:
        left = self.L.Pulse.steps_left()
        ask = self.L.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.L.roles.get('buy'), self.L.roles.get('refuse')} - {None}
        if left <= self.L.Pulse.seal_reserve():
            keep = set(ends)
        elif left <= self.L.Pulse.seal_reserve() + 3:
            keep = set(ends) | {self.L.roles.get('cart'), self.L.roles.get('cart_view')}
        elif left <= SWAP_TIGHT and self.L.AttrBench.cart_fully_passes():
            # Passing cart + thinning budget: stop searching; seal or replace only if sold out.
            keep = set(ends) | {self.L.roles.get('cart'), self.L.roles.get('cart_view')}
        elif left <= len(self.L.items) + 5 and self.L.buys == 0:
            keep = set(ends) | {self.L.roles.get('cart'), self.L.roles.get('cart_view'), self.L.roles.get('offer')}
        tools = [t for t in self.L.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def mute_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.L.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _schema_kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.L.rows)) or self.L.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def fix_id_args(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.L.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.L.id_key:
            return (args, '')
        props = (self.L.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _schema_kinds(v) or 'array' in _schema_kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.L.roles.get('cart') and (name, free[0]) not in self.L.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.L.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or _looks_id(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.L.rows.items() if any((k != self.L.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.L.id_key} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def basket_is_set(self, name: str) -> bool:
        props = self.L.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _schema_kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def track_basket(self, name: str, args: dict[str, Any]) -> None:
        self.L.cart_touched = True
        self.L.cart_at = self.L.step
        # cart changed — seal gates must re-run (cart view + line reread + report)
        self.L.report_done = False
        self.L.seal_reread_mark = None
        spec = self.L.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((_as_int(v) for v in args.values() if _as_int(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.L.rows), None)
                        count = next((_as_int(v) for v in line.values() if _as_int(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.L.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.L.rows:
                lines.append((value, loose_count))
        if whole:
            self.L.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.basket_is_set(name)
        for ident, count in lines:
            have = self.L.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.L.cart_lines[ident] = have
            else:
                self.L.cart_lines.pop(ident, None)

    def side_arg(self) -> str | None:
        schema = self.L.schemas.get(self.L.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _schema_kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def side_limit(self) -> int:
        spec = ((self.L.schemas.get(self.L.roles.get('compare') or '') or {}).get('properties') or {}).get(self.side_arg() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.L.compare_bad:
            return max(2, min(self.L.compare_bad - 1, max(self.L.compare_ok, 2)))
        return max(4, self.L.compare_ok)

    def batch_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.L.roles.get('open'), self.L.roles.get('compare'), self.L.Catalog.id_arg_of(self.L.roles.get('open')))
        many = self.side_arg()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.L.Pulse.side_is_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _as_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.L.rows) and (ident not in self.L.gone):
                picks.append((index, ident))
        picks = picks[:self.side_limit()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _to_json({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.L.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

class SealWatch:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def follow_chains(self) -> tuple[str, bool]:
        live = {need for _, need in self.L.Catalog.live_needs()}
        pending = [need for _, need in self.L.refs if need in live and need not in self.L.ref_values]
        tool = self.L.roles.get('offer') or self.L.roles.get('open')
        target = self.L.Catalog.id_arg_of(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.L.trails < (6 if self.L.Pulse.steps_left() > 15 else 4) and (self.L.cost < HARD_USD):
            self.L.trails += 1
            records = []
            for ident in reversed(self.L.seen_order):
                record = self.L.full.get(ident) or self.L.rows.get(ident) if ident in self.L.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _to_json(_shrink(record, 300))[:1400])
            for ident in reversed(self.L.seen_order):
                if len(records) >= 60:
                    break
                if ident not in self.L.opened or not (self.L.full.get(ident) or self.L.rows.get(ident)):
                    records.append('ROW ' + _to_json(_shrink(self.L.rows.get(ident), 80))[:300])
            prompt = PROMPT_TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.L.Pulse.json_query(prompt, [FAST] + ([STRONG] if self.L.cost + STRONG_ROOM <= HARD_USD else []), 4500) or {}
            self.L.trace.append({'kind': 'trail', 'pending': pending, 'got': _shrink(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _as_list(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _as_list(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.L.opened for i in ids)) and str(chain.get('value') or '').strip():
                    self.L.ref_values[need] = str(chain['value']).strip()
                    self.L.version += 1
                    continue
                if links and all((i in self.L.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.L.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.L.version += 1
                    continue
                closed = [i for i in ids if i in self.L.rows and i not in self.L.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.L.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.L.ref_values]
            if not reach and gaps and tool and target and (self.L.Pulse.steps_left() > 12) and (self.L.link_searches < 3):
                reach = self.hunt_link(*gaps[0])
                if self.L.trace and self.L.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.L.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.L.Pulse.steps_left() < len(reach) + 4) or self.L.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _coerce({target: ident}, self.L.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.L.Pulse.env_step(tool, args)
                if reply is None or self.L.finished:
                    break
                self.L.Catalog.after_obs(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.L.Pulse.render(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[Forge opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.L.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.L.ref_values.items())) + '\n'
        hold = bool(lost) and self.L.trail_blocks < (3 if self.L.Pulse.steps_left() > 15 else 2) and (self.L.Pulse.steps_left() > 5)
        if hold:
            self.L.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def rows_by_shop(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.L.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if ident not in self.L.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_by_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.L.opened:
            row = self.L.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def hunt_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.rows_by_shop(about, list(self.L.rows)) + self.opened_by_shop(about) for k, v in (self.L.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for ident in self.opened_by_shop(about):
            stack, leaves = ([self.L.full.get(ident) or self.L.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.L.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.L.Catalog.search_skel()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.L.link_seen:
            self.L.link_searches += 1
            known = self.rows_by_shop(about, self.L.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _coerce({**base, field: words}, self.L.schemas.get(tool), 'args', faults)
            if not faults:
                self.L.link_searches += 1
                reply = self.L.Pulse.env_step(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.L.Catalog.after_obs(tool, observation)
                    self.L.link_seen[memo] = [i for i, _ in self.L.Catalog.rows_in(observation)]
                    known = self.rows_by_shop(about, self.L.link_seen[memo])
                    self.L.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _to_json(self.L.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.rows_by_shop(about, list(reversed(self.L.seen_order)))))[:2]

    def stock_tool(self) -> str | None:
        return self.L.roles.get('offer') or self.L.roles.get('open')

    def refresh_offers(self) -> str:
        tool = self.stock_tool()
        target = self.L.Catalog.id_arg_of(tool)
        if not tool or not target:
            return ''
        since = max(self.L.step - 2, self.L.refused_at)
        stale = [i for i in self.L.Catalog.basket_ids() if self.L.offer_seen.get(i, -99) < since and i not in self.L.gone]
        if not stale or self.L.Pulse.steps_left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _coerce({target: ident}, self.L.schemas.get(tool), 'args', faults)
            if faults:
                break
            before = dict(self.L.rows.get(ident) or {})
            reply = self.L.Pulse.env_step(tool, args)
            if reply is None or self.L.finished:
                break
            observation = reply.get('observation')
            self.L.offer_seen[ident] = self.L.step
            self.L.Catalog.after_obs(tool, observation)
            after = self.L.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_to_json(_shrink(observation, 300))[:900]}")
        self.L.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.L.version += 1
        return '\n'.join(lines)

    def ensure_cart_view(self) -> str:
        """Force a fresh cart_view after the last cart change before checkout, when the store has one."""
        tool = self.L.roles.get('cart_view')
        if not tool or self.L.Pulse.steps_left() <= 2:
            return ''
        if self.L.cart_view is not None and self.L.cart_seen_at >= self.L.cart_at:
            return ''
        if not self.L.Catalog.basket_ids() and not self.L.cart_touched:
            return ''
        faults: list[str] = []
        view_args = _coerce({}, self.L.schemas.get(tool), 'args', faults)
        if faults:
            return ''
        view = self.L.Pulse.env_step(tool, view_args)
        if view is None:
            return ''
        self.L.cart_view = view.get('observation')
        self.L.cart_seen_at = self.L.step
        # any prior line reread is stale relative to this new view
        self.L.seal_reread_mark = None
        change = self.L.Catalog.after_obs(tool, self.L.cart_view)
        self.L.trace.append({'kind': 'cart_view_forced', 'changed': bool(change)})
        if change:
            self.L.queued = []
            return 'Looking at the cart showed a change: ' + change + '\nCart now: ' + self.L.Catalog.basket_text() + '\nApply the change, fix the cart if needed, then check out.'
        return ''

    def ensure_basket_reread(self) -> str:
        """Re-open every cart line after the last cart_view so price/stock are current at seal.

        When a cart_view tool exists, require a fresh view first; do not treat opens from before
        that view as enough for checkout.
        """
        ids = [i for i in self.L.Catalog.basket_ids() if (self.L.Catalog.qty_of(i) or 0) > 0 and i not in self.L.gone]
        if not ids:
            return ''
        # When a cart_view tool exists, require a fresh view first; reread is tied to that view.
        if self.L.roles.get('cart_view') and (self.L.cart_view is None or self.L.cart_seen_at < self.L.cart_at):
            return ''
        mark = self.L.Pulse.basket_sig() + f'@view{self.L.cart_seen_at}#v{self.L.version}'
        if self.L.seal_reread_mark == mark:
            return ''
        tool = self.L.roles.get('open') or self.L.roles.get('offer')
        target = self.L.Catalog.id_arg_of(tool) if tool else None
        if not tool or not target:
            self.L.seal_reread_mark = mark
            return ''
        need = len(ids)
        if self.L.Pulse.steps_left() < need + self.L.Pulse.seal_reserve() + 1:
            # still mark what we can; don't block forever when steps are gone
            if self.L.Pulse.steps_left() <= self.L.Pulse.seal_reserve() + 1:
                self.L.seal_reread_mark = mark
                return ''
        lines: list[str] = []
        dry: list[str] = []
        for ident in ids:
            if self.L.Pulse.steps_left() <= self.L.Pulse.seal_reserve() + 1 or self.L.finished:
                break
            faults: list[str] = []
            args = _coerce({target: ident}, self.L.schemas.get(tool), 'args', faults)
            if faults:
                continue
            before = dict(self.L.rows.get(ident) or {})
            reply = self.L.Pulse.env_step(tool, args)
            if reply is None or self.L.finished:
                break
            observation = reply.get('observation')
            self.L.offer_seen[ident] = self.L.step
            self.L.Catalog.after_obs(tool, observation)
            self.L.Catalog.mark_view(tool, observation)
            after = self.L.rows.get(ident) or {}
            spoken = self.L.Catalog.obs_speech(observation).lower()
            if self.is_dry(ident) or any(w in spoken for w in ('sold out', 'out of stock', 'no longer available', 'nobody has any', 'ran dry')):
                self.L.unavailable.add(ident)
                dry.append(ident)
                continue
            for k, v in (after if isinstance(after, dict) else {}).items():
                if any(w in str(k).lower() for w in ('in_stock', 'instock', 'available')) and v is False:
                    self.L.unavailable.add(ident)
                    dry.append(ident)
                    break
            changed = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            lines.append(f'{ident}' + (f' ({", ".join(changed[:3])})' if changed else ''))
        self.L.seal_reread_mark = mark
        self.L.trace.append({'kind': 'basket_reread', 'ids': ids, 'read': lines, 'dry': dry})
        if dry:
            self.L.version += 1
            self.L.seal_reread_mark = None
            return (
                f'[forge] After viewing the cart, re-read found unavailable lines: {", ".join(dry)}. '
                'Remove/replace them, view the cart again, re-open each remaining line, then check out.\nCart now: '
                + self.L.Catalog.basket_text()
            )
        return ''

    def basket_reveal_wanted(self) -> bool:
        """True only when the request or process list asks to show/confirm the cart."""
        req = (self.L.request or '').lower()
        process_wants = any(
            re.search(r'\b(cart|basket|bag|show|tell|confirm)\b', p.get('step', ''), re.I)
            for p in self.L.process
            if p.get('when') == 'order'
        )
        req_wants = bool(re.search(r'\b(show|tell|confirm|read).{0,40}\b(cart|basket|bag|order)\b', req))
        return bool(process_wants or req_wants)

    def show_basket_message(self) -> str:
        """Show the shopper the final cart before ordering when the request/process asks.

        Only when the request or process list asks to show/confirm the cart.
        """
        if not self.L.roles.get('ask') or self.L.finished:
            return ''
        if self.L.msg_used >= self.L.msg_limit or self.L.Pulse.steps_left() < 2:
            return ''
        if getattr(self.L, 'basket_told', -1) > self.L.cart_at and self.L.Catalog.basket_ids():
            return ''
        ids = self.L.Catalog.basket_ids()
        if not ids:
            return ''
        if not self.basket_reveal_wanted():
            return ''
        parts = []
        total = 0.0
        priced = True
        for ident in ids:
            qty = self.L.Catalog.qty_of(ident) or self.L.cart_lines.get(ident, 1) or 1
            row = self.L.full.get(ident) or self.L.rows.get(ident) or {}
            title = ''
            for k, v in row.items():
                if isinstance(v, str) and any(w in str(k).lower() for w in ('title', 'name', 'head')) and len(v.strip()) > 2:
                    title = v.strip()[:80]
                    break
            pr = self.L.Catalog.price_for(ident)
            bit = f'{qty}× {title or ident} ({ident})'
            if pr is not None:
                bit += f' @ {pr:g}'
                total += pr * qty
            else:
                priced = False
            parts.append(bit)
        text = 'Here is the cart I am about to order: ' + '; '.join(parts)
        if priced and parts:
            text += f'. Total {total:g}.'
        text += ' Proceeding to checkout.'
        payload = self.L.Briefing.ask_payload(text)
        if not payload:
            return ''
        reply = self.L.Pulse.env_step(self.L.roles['ask'], payload)
        if reply is None:
            return ''
        self.L.msg_used += 1
        self.L.gate_msgs += 1
        self.L.basket_told = self.L.step
        obs = reply.get('observation')
        self.L.Catalog.after_obs(self.L.roles['ask'], obs)
        spoken = self.L.Catalog.obs_speech(obs)
        self.L.done_log.append(f'step {self.L.step}: basket reveal -> {spoken[:240]}')
        self.L.done_steps.append(self.L.step)
        self.L.trace.append({'kind': 'basket_told', 'ids': ids, 'text': text[:240]})
        if spoken and re.search(r'\b(instead|actually|change|drop|add|budget|don\'t|do not|cancel)\b', spoken, re.I):
            self.L.version += 1
            return (
                f'Showed the shopper the cart. Their reply may change the task: {spoken[:400]}\nCart now: '
                + self.L.Catalog.basket_text()
                + '\nApply any change, then check out.'
            )
        return ''

    def run_report_tool(self) -> str:
        """Call the store report tool once per cart version when mapped and steps allow."""
        tool = self.L.roles.get('report')
        if not tool or self.L.Pulse.steps_left() < 2:
            return ''
        mark = self.L.Pulse.basket_sig() + f'@{self.L.version}'
        if getattr(self.L, 'report_mark', None) == mark or getattr(self.L, 'report_done', False):
            return ''
        # Prefer when process/request mentions it; otherwise still once if steps are comfortable
        req = (self.L.request or '').lower()
        wanted = any(
            re.search(r'\b(report|allowance|usage|account|status)\b', p.get('step', ''), re.I)
            for p in self.L.process
        ) or bool(re.search(r'\b(report|allowance|usage|account|status)\b', req))
        if not wanted and self.L.Pulse.steps_left() < self.L.Pulse.seal_reserve() + 4:
            return ''
        faults: list[str] = []
        args = _coerce({}, self.L.schemas.get(tool), 'args', faults)
        if faults:
            return ''
        reply = self.L.Pulse.env_step(tool, args)
        self.L.report_done = True
        self.L.report_mark = mark
        if reply is None:
            return ''
        obs = reply.get('observation')
        self.L.Catalog.after_obs(tool, obs)
        self.L.done_log.append(f'step {self.L.step}: {tool} -> {self.L.Pulse.render(obs)[:240]}')
        self.L.done_steps.append(self.L.step)
        self.L.trace.append({'kind': 'report', 'tool': tool, 'obs': _shrink(obs, 120)})
        return ''

    def pre_seal_stock(self) -> str:
        """Re-read each cart line's offer/stock before checkout. Catches silent sell-outs."""
        tool = self.stock_tool()
        target = self.L.Catalog.id_arg_of(tool) if tool else None
        ids = [i for i in self.L.Catalog.basket_ids() if (self.L.Catalog.qty_of(i) or 0) > 0 and i not in self.L.gone]
        mark = self.L.Pulse.basket_sig() + f'@{self.L.version}'
        if not tool or not target or not ids:
            return ''
        if self.L.stock_checked == mark:
            return ''
        if self.L.Pulse.steps_left() < len(ids) + self.L.Pulse.seal_reserve() + 2:
            return ''
        self.L.stock_checked = mark
        dry: list[str] = []
        moved: list[str] = []
        for ident in ids:
            faults: list[str] = []
            args = _coerce({target: ident}, self.L.schemas.get(tool), 'args', faults)
            if faults:
                continue
            before = dict(self.L.rows.get(ident) or {})
            reply = self.L.Pulse.env_step(tool, args)
            if reply is None or self.L.finished:
                break
            observation = reply.get('observation')
            self.L.offer_seen[ident] = self.L.step
            self.L.Catalog.after_obs(tool, observation)
            after = self.L.rows.get(ident) or {}
            if self.is_dry(ident):
                dry.append(ident)
                continue
            spoken = self.L.Catalog.obs_speech(observation).lower()
            if any((w in spoken for w in ('sold out', 'out of stock', 'no longer available', 'nobody has any', 'ran dry'))):
                self.L.unavailable.add(ident)
                dry.append(ident)
                continue
            # also treat explicit in_stock/false
            for k, v in (after if isinstance(after, dict) else {}).items():
                if any(w in str(k).lower() for w in ('in_stock', 'instock', 'available')) and v is False:
                    self.L.unavailable.add(ident)
                    dry.append(ident)
                    break
            if ident in dry:
                continue
            changed = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if changed:
                moved.append(f"{ident} ({', '.join(changed[:4])})")
        self.L.trace.append({'kind': 'stock_gate', 'ids': ids, 'dry': dry, 'moved': moved})
        if dry:
            self.L.version += 1
            return f"[forge] Stock check before checkout: {', '.join(dry)} is unavailable now. Remove that line and replace it with a listing that still meets every requirement (page the category if needed), then check out."
        if moved:
            return f"[forge] Stock check before checkout found updates on {', '.join(moved)}. Confirm the cart still meets every requirement and the budget, then check out."
        return ''

    def is_dry(self, ident: str) -> bool:
        return ident in self.L.unavailable or ident in self.L.gone

    def dry_swap_note(self) -> str:
        lost = [i for i in self.L.Catalog.basket_ids() if i not in self.L.gone and self.is_dry(i)]
        if not lost:
            return ''
        for ident in lost:
            self.L.gone[ident] = self.L.step
        if self.L.brief.get('pick') == 'cheapest' and self.L.Pulse.steps_left() >= SWAP_TIGHT:
            self.L.pages_read = 0
        self.L.leads = [i for i in self.L.leads if i not in lost]
        self.L.version += 1
        picks = self.swap_candidates(lost)
        if not picks:
            self.hunt_after_dry(lost)
            picks = self.swap_candidates(lost)
        text = f"[forge] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        tight = self.L.Pulse.steps_left() <= SWAP_TIGHT
        if picks and tight:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.L.brief.get('pick') == 'cheapest' and self.L.Pulse.steps_left() > SWAP_TIGHT:
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart. Search its category again and page through results (token/offset), open candidates that state every requirement in their attributes, then add the best fit and check out.'

    def hunt_after_dry(self, lost: list[str]) -> list[str]:
        """Open unread same-item candidates (and one value search) so sell-out recovery has backups."""
        if getattr(self.L, 'swap_hunt_done', False) or self.L.Pulse.steps_left() <= self.L.Pulse.seal_reserve() + 3:
            return []
        self.L.swap_hunt_done = True
        numbers = {self.L.AttrBench.owner_num(i) for i in lost} - {0}
        if not numbers:
            numbers = set(range(1, len(self.L.items) + 1))
        before = set(self.L.full)
        if self.L.Pulse.steps_left() > self.L.Pulse.seal_reserve() + SWAP_HUNT + 4:
            try:
                self.L.Catalog.value_query(sorted(numbers))
            except Exception:
                pass
        picks: list[str] = []
        for number in sorted(numbers):
            room = None
            if isinstance(self.L.budget, (int, float)):
                qty = self.L.items[number - 1].qty
                room = self.L.budget / max(1, qty)
            pool = []
            for ident in self.L.seen_order:
                if ident in self.L.full or self.is_dry(ident) or ident in self.L.gone:
                    continue
                if self.L.AttrBench.owner_num(ident) != number:
                    continue
                if not self.L.Catalog.can_buy(ident):
                    continue
                price = self.L.Catalog.price_for(ident)
                if room is not None and price is not None and price > room + 1e-9:
                    continue
                pool.append(ident)
            pool.sort(key=lambda i: self.L.Catalog.price_for(i) if self.L.Catalog.price_for(i) is not None else float('inf'))
            picks += [i for i in pool[:SWAP_HUNT] if i not in picks]
        opened = self.L.Catalog.open_attrs(picks[:SWAP_HUNT], self.L.Pulse.seal_reserve() + 3)
        found = [i for i in picks if i in self.L.full and i not in before]
        self.L.trace.append({'kind': 'swap_hunt', 'lost': lost, 'opened': found, 'read': len(opened)})
        return found

    def strong_swaps(self, lost: list[str], picks: list[str]) -> list[str]:
        """Keep replacements that the attribute table fully passes; fall back to picks if none do."""
        self.L.AttrBench.learn_attrs()
        numbers = {self.L.AttrBench.owner_num(i) for i in lost} - {0}
        if len(numbers) != 1:
            return picks
        number = numbers.pop()
        # Prefer previously carted proof-passing backups still believed in stock.
        backups = [
            i for i in self.L.passed_cart
            if i not in lost and not self.is_dry(i) and i not in self.L.gone
            and self.L.AttrBench.owner_num(i) == number
        ]
        merged = list(dict.fromkeys(backups + list(picks)))
        scored = []
        for ident in merged:
            if ident in lost or self.is_dry(ident) or ident in self.L.gone:
                continue
            if ident not in self.L.full:
                continue
            score, _notes, judged = self.L.AttrBench.score_listing(ident, number)
            if judged and score >= 4:
                scored.append(ident)
        if not scored:
            need = [i for i in merged if i not in self.L.full][:SWAP_HUNT]
            if need and self.L.Pulse.steps_left() > self.L.Pulse.seal_reserve() + len(need) + 2:
                self.L.Catalog.open_attrs(need, self.L.Pulse.seal_reserve() + 2)
            for ident in merged:
                if ident in lost or self.is_dry(ident) or ident not in self.L.full:
                    continue
                score, _notes, judged = self.L.AttrBench.score_listing(ident, number)
                if judged and score >= 4 and ident not in scored:
                    scored.append(ident)
        if self.L.brief.get('pick') == 'cheapest' and scored:
            return sorted(scored, key=lambda i: self.L.Catalog.price_for(i) if self.L.Catalog.price_for(i) is not None else float('inf'))
        if scored:
            return list(dict.fromkeys([i for i in backups if i in scored] + scored))
        return merged or picks

    def recover_sold_out(self) -> str:
        """Code-side sell-out recovery: drop dry lines, cart a proof-passing backup, then try checkout."""
        lost = [i for i in self.L.Catalog.basket_ids() if i not in self.L.gone and self.is_dry(i)]
        if not lost:
            lost = [i for i, step in self.L.gone.items() if step == self.L.step]
        if not lost:
            return ''
        if getattr(self.L, 'swap_forced', 0) >= 2:
            return ''
        for ident in lost:
            self.L.gone[ident] = self.L.step
        seed = [i for i in self.L.passed_cart if i not in lost and not self.is_dry(i) and i not in self.L.gone]
        picks = self.strong_swaps(lost, seed + self.swap_candidates(lost))
        if not picks:
            self.hunt_after_dry(lost)
            picks = self.strong_swaps(lost, seed + self.swap_candidates(lost))
        if not picks:
            return ''
        force = bool(self.L.unavailable) or self.L.buys > 0 or getattr(self.L, 'swap_forced', 0) > 0
        if self.L.Pulse.steps_left() > SWAP_TIGHT and not force:
            return ''
        owner = self.L.AttrBench.owner_num(lost[0]) or self.L.AttrBench.owner_num(picks[0]) or 1
        qty = self.L.items[owner - 1].qty if 1 <= owner <= len(self.L.items) else 1
        # One requested product: keep a single replacement line (do not stack two of the same item).
        if len(self.L.items) == 1:
            lines = {picks[0]: qty}
            drop = [i for i in self.L.Catalog.basket_ids() if i != picks[0]] + [i for i in lost if i != picks[0]]
        else:
            now = [i for i in self.L.Catalog.basket_ids() if not self.is_dry(i) and i not in self.L.gone]
            keep = {i: (self.L.Catalog.qty_of(i) or self.L.cart_lines.get(i, 1) or 1) for i in now if i != picks[0]}
            lines = {**keep, picks[0]: qty}
            drop = [i for i in self.L.Catalog.basket_ids() if i not in lines] + [i for i in lost if i not in lines]
        drop = list(dict.fromkeys(drop))
        calls = self.L.Basket.basket_ops(lines, drop)
        if not calls:
            return ''
        # Prefer leaving a step for checkout, but still swap when only the cart ops fit.
        if len(calls) > self.L.Pulse.steps_left():
            return ''
        cart = self.L.roles.get('cart')
        buy = self.L.roles.get('buy')
        if not cart:
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = _coerce(args, self.L.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.L.Pulse.env_step(cart, args)
            if reply is None or self.L.finished:
                return ''
            self.L.Basket.track_basket(cart, args)
            self.L.Catalog.after_obs(cart, reply.get('observation'))
            done.append(_to_json(args))
        self.L.swap_forced = getattr(self.L, 'swap_forced', 0) + 1
        self.L.version += 1
        self.L.seal_reread_mark = None
        self.L.report_done = False
        self.L.leads = [picks[0]] + [i for i in picks[1:] if i not in lost]
        self.L.trace.append({'kind': 'swap_forced', 'lost': lost, 'pick': picks[0], 'calls': done})
        said = (
            f"Not run (Forge acted instead): {', '.join(lost)} sold out, so Forge set the cart to "
            f"{picks[0]} at qty {qty} ({'; '.join(done)}). Cart now: " + self.L.Catalog.basket_text()
        )
        # Seal immediately when we still have a checkout step — stops the buyer stacking extra lines.
        if buy and self.L.Pulse.steps_left() >= 1 and not self.L.finished:
            # skip heavy seal rituals when almost out of steps; still require a fresh cart_view if cheap
            if self.L.roles.get('cart_view') and self.L.Pulse.steps_left() >= 3:
                viewed = self.ensure_cart_view()
                if viewed and 'unavailable' in viewed.lower():
                    return said + '. ' + viewed
            faults = []
            args = _coerce(dict(self.L.brief.get('buy_args') or {}), self.L.schemas.get(buy), 'args', faults)
            if not faults:
                reply = self.L.Pulse.env_step(buy, args)
                self.L.buys += 1
                self.L.trace.append({'kind': 'swap_checkout', 'pick': picks[0], 'reply': _shrink(reply, 200)})
                if reply is not None:
                    self.L.Catalog.after_obs(buy, reply.get('observation'))
                    said += ' Forge then checked out: ' + self.L.Pulse.render(reply.get('observation'))[:400]
                    if reply.get('finished') or self.L.finished:
                        said += ' [episode finished]'
                        return said
                    # checkout refused again — mark dry if needed and stop so another recover can run
                    self.L.refused_at = self.L.step
                    spoken = self.L.Catalog.obs_speech(reply.get('observation')).lower()
                    if any(w in spoken for w in ('sold out', 'unavailable', 'ran dry', 'all gone', 'no longer')):
                        self.L.unavailable.add(picks[0])
                        said += f' (replacement {picks[0]} also unavailable).'
        return said + '. Do not add extra lines; check out this cart as it is unless a line is sold out.'

    def detect_dry(self) -> str:
        ids = [i for i in self.L.Catalog.basket_ids() if i not in self.L.gone]
        if any((self.is_dry(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.L.unavailable.add(ids[0])
            return ''
        return self.refresh_offers() if ids else ''

    def swap_candidates(self, lost: list[str]) -> list[str]:
        if not self.L.Pulse.pick_models():
            return self.L.AttrBench.order_swaps(lost, [])[:4]
        records = []
        for ident in reversed(self.L.seen_order):
            if ident in lost or ident in self.L.gone or self.is_dry(ident):
                continue
            records.append(_to_json(_shrink(self.L.full.get(ident) or self.L.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return self.L.AttrBench.order_swaps(lost, [])[:4]
        prompt = PROMPT_SWAP + '\n\nREQUEST:\n' + self.L.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.L.Briefing.mission_text()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.L.Catalog.listing_blob(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.L.Catalog.basket_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.L.Pulse.json_query(prompt, self.L.Pulse.pick_models(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _as_list((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.L.rows and ident not in lost and (ident not in self.L.gone) and (ident not in self.L.Catalog.basket_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.L.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.L.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return self.L.AttrBench.order_swaps(lost, ranked)[:4]

    def decline_audit(self) -> str:
        if self.L.audited or not self.L.Pulse.pick_models():
            return ''
        self.L.audited = True
        got = self.audit_query()
        named = self.audit_hits(got)
        if not named:
            rows = [str(i) for i in _as_list((got or {}).get('open')) if str(i) in self.L.rows and str(i) not in self.L.opened and (not self.is_dry(str(i)))]
            if rows and self.open_candidates(rows):
                got = self.audit_query()
                named = self.audit_hits(got)
        if not named:
            return ''
        self.L.lead_items = named
        self.L.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def audit_query(self) -> dict[str, Any] | None:
        records = []
        for ident in reversed(self.L.seen_order):
            if self.is_dry(ident):
                continue
            record = self.L.full.get(ident) if ident in self.L.full else None
            records.append('OPENED ' + _to_json(_shrink(record, 200))[:1200] if record else 'ROW ' + _to_json(_shrink(self.L.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = PROMPT_AUDIT + '\n\nREQUEST:\n' + self.L.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.L.Briefing.mission_text()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.L.Pulse.json_query(prompt, self.L.Pulse.pick_models(), 4000)
        self.L.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_hits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _as_list(v) if str(i) in self.L.rows and (not self.is_dry(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_candidates(self, rows: list[str]) -> bool:
        many, joiner = (self.L.Basket.side_arg(), self.L.roles.get('compare'))
        tool = self.L.Pulse.attrs_tool()
        target = self.L.Catalog.id_arg_of(tool)
        if many and joiner and (not self.L.Pulse.side_is_bare()):
            plan = [(joiner, {many: rows[:self.L.Basket.side_limit()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.L.Pulse.steps_left() < len(plan) + 5:
            return False
        before = len(self.L.full)
        for name, args in plan:
            faults: list[str] = []
            args = _coerce(args, self.L.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.L.Pulse.env_step(name, args)
            if reply is None or self.L.finished:
                break
            self.L.Catalog.after_obs(name, reply.get('observation'))
        self.L.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.L.full) - before})
        return len(self.L.full) > before

class Conductor:
    def __init__(self, loop: "ForgeLoop") -> None:
        self.L = loop

    def dispatch(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.L.Basket.batch_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.L.Basket.visible_tools()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Forge opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.L.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.L.roles.get('ask'):
                out.append((cid, "Not run (no step used): Forge's interviewer handles every message to the shopper."))
                continue
            if name not in self.L.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _as_obj(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            faults: list[str] = []
            args = _coerce(args, self.L.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.L.Basket.fix_id_args(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            many = self.L.Basket.side_arg() if name == self.L.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.L.Basket.side_limit()) and self.L.compare_bad:
                room = self.L.Basket.side_limit()
                trimmed = f"\n[forge] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.L.Catalog.id_arg_of(name)
            if target and name in (self.L.roles.get('open'), self.L.roles.get('offer')) and (str(args.get(target)) in self.L.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.L.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.L.roles.get('cart') and any(((_as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = _to_json(self.L.cart_view) if self.L.cart_seen_at >= self.L.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.L.rows and (v not in self.L.cart_lines) and (v not in view) and (v not in self.L.cart_said)]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.L.Catalog.basket_text()))
                    continue
            # Do not drop a proof-passing cart line when too few steps remain to recover from another sell-out.
            if name == self.L.roles.get('cart') and self.L.Pulse.steps_left() <= CHEAP_SWAP_ROOM + 2:
                cart_ids = [v for v in args.values() if isinstance(v, str) and v in self.L.rows]
                counts = [_as_int(v) for v in args.values() if not isinstance(v, str) and _as_int(v) is not None]
                mode = next((str(v).lower() for v in args.values() if isinstance(v, str) and str(v).lower() in ('discard', 'remove', 'exclude', 'delete', 'drop')), '')
                removing = bool(mode) or (counts and counts[0] is not None and counts[0] <= 0)
                target_id = cart_ids[0] if cart_ids else ''
                if removing and target_id and target_id in self.L.passed_cart and (not self.L.SealWatch.is_dry(target_id)) and target_id not in self.L.gone:
                    if self.L.AttrBench.cart_line_passes(target_id) or self.L.AttrBench.cart_fully_passes():
                        out.append((cid, f'Not run (no step used): {target_id} already states every requirement and only {self.L.Pulse.steps_left()} steps remain — not enough to recover if a replacement sells out. Check out the cart as it is. Cart now: ' + self.L.Catalog.basket_text()))
                        continue
            ending = self.L.Basket.is_checkout(name, args) or self.L.Basket.is_walkaway(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.L.Pulse.steps_left() <= self.L.Pulse.seal_reserve() and (self.L.roles.get('buy') or self.L.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _to_json(args)
            if not ending and name != self.L.roles.get('cart') and (mark in self.L.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.L.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.L.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.L.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.L.roles.get('compare')) and self.L.Catalog.view_fresh(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.L.Catalog.listing_blob(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.L.Catalog.view_fresh(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.L.Catalog.listing_blob(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[forge] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _to_json(args)
            if ending and self.L.Pulse.steps_left() > 2:
                if self.L.Basket.is_checkout(name, args):
                    wider = self.L.Catalog.cheap_widen_hold()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    late = self.L.Briefing.late_question()
                    if late:
                        out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    viewed = self.L.SealWatch.ensure_cart_view()
                    if viewed:
                        out.append((cid, 'Not run yet (no step used). ' + viewed))
                        continue
                    proved = self.L.SealWatch.ensure_basket_reread()
                    if proved:
                        out.append((cid, 'Not run yet (no step used). ' + proved))
                        continue
                    self.L.SealWatch.run_report_tool()
                    if self.L.finished:
                        out.append((cid, 'Episode finished during the report step.'))
                        continue
                    reread = self.L.SealWatch.refresh_offers()
                    if self.L.note_sold:
                        self.L.SealWatch.detect_dry()
                    stocked = self.L.SealWatch.pre_seal_stock()
                    if stocked:
                        out.append((cid, 'Not run yet (no step used). ' + stocked + '\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    told = self.L.SealWatch.show_basket_message()
                    if told:
                        out.append((cid, 'Not run yet (no step used). ' + told))
                        continue
                    recovered = self.L.SealWatch.recover_sold_out()
                    if recovered:
                        out.append((cid, recovered))
                        continue
                    swap = self.L.SealWatch.dry_swap_note()
                    held = [i for i in self.L.Catalog.basket_ids() if i in self.L.gone] if self.L.held < 2 else []
                    if swap or held:
                        self.L.held += 0 if swap else 1
                        # when steps are tight, try one more code-side swap after the note built picks
                        if self.L.Pulse.steps_left() <= SWAP_TIGHT:
                            recovered = self.L.SealWatch.recover_sold_out()
                            if recovered:
                                out.append((cid, recovered))
                                continue
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    chain, hold = self.L.SealWatch.follow_chains()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    misfit = self.L.Catalog.compat_faults() if self.L.fit_holds < 2 and self.L.Pulse.steps_left() > self.L.Pulse.seal_reserve() + 3 else []
                    if misfit:
                        self.L.fit_holds += 1
                        self.L.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    faults = self.L.Basket.basket_faults() if self.L.line_holds < 2 and self.L.Pulse.steps_left() > self.L.Pulse.seal_reserve() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.L.Basket.repair_qtys(self.L.Basket.qty_mismatches())
                        faults = self.L.Basket.basket_faults() if fixed else faults
                        if fixed and (not faults):
                            self.L.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Forge set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.L.Catalog.basket_text()))
                            continue
                    if faults:
                        self.L.line_holds += 1
                        self.L.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.L.Catalog.basket_text()))
                        continue
                    thin = self.L.AttrBench.thin_cart_attrs() if not self.L.attr_pressed and self.L.Pulse.steps_left() > ATTR_ROOM else ''
                    if thin:
                        self.L.attr_pressed = True
                        self.L.trace.append({'kind': 'proof_thin', 'said': thin[:300]})
                        out.append((cid, 'Not run yet (no step used). ' + thin + f' {self.L.Pulse.steps_left()} steps are left: open more listings of this kind first, searching with the wanted value, then check out.\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    # Allow one extra proof hold when pick is extreme (cheapest/most/least)
                    hold_budget = 3 if self.L.brief.get('pick') in ('cheapest', 'most', 'least') else 2
                    better = self.L.AttrBench.better_than_cart() if self.L.attr_holds < hold_budget and self.L.Pulse.steps_left() > self.L.Pulse.seal_reserve() + 3 else []
                    if better:
                        self.L.attr_holds += 1
                        self.L.trace.append({'kind': 'proof_hold', 'faults': [f[:300] for f in better[:3]]})
                        out.append((cid, "Not run yet (no step used). The store's own attributes show a better fit than the cart: " + ' | '.join(better[:3]) + '. Then check out.\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    read = self.L.Catalog.open_attrs(self.L.Catalog.basket_ids(), self.L.Pulse.seal_reserve() + 3) + self.L.Catalog.hunt_cheaper()
                    verdict = self.L.Basket.review_basket()
                    if verdict:
                        seen = "Forge read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.L.Catalog.basket_text()))
                        continue
                    if reread and self.L.cost > SOFT_USD:
                        out.append((cid, "Not run yet (no step used). Forge re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.L.Pulse.steps_left() > 5:
                    review = self.L.SealWatch.decline_audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if not self.L.leads:
                        named = self.L.AttrBench.passing_leads()
                        if named:
                            self.L.lead_items = named
                            self.L.leads = [i for ids in named.values() for i in ids]
                            self.L.trace.append({'kind': 'proof_leads', 'leads': list(self.L.leads)})
                    # Block a premature refuse when proof still sees fitting leads
                    if self.L.leads and self.L.Pulse.steps_left() > 4 and (self.L.refuse_blocks == 0):
                        self.L.refuse_blocks = 1
                        self.L.trace.append({'kind': 'refuse_blocked', 'leads': list(self.L.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.L.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.L.leads and self.L.Pulse.steps_left() > 4 and (self.L.refuse_blocks == 1):
                        self.L.refuse_blocks = 2
                        first = self.L.leads[0]
                        self.L.Basket.review_basket([first], HARD_USD)
                        owner = self.L.AttrBench.owner_num(first)
                        scored = self.L.AttrBench.score_listing(first, owner) if owner and first in self.L.full else (2, [], 0)
                        if scored[2] and scored[0] <= 1:
                            self.L.trace.append({'kind': 'proof_unconfirmed', 'id': first, 'notes': scored[1][:4]})
                            self.L.leads = [i for i in self.L.leads if i != first]
                        elif self.L.verdicts.get(self.L.Basket.review_key([first])) and first in self.L.leads:
                            self.L.confirmed = first
                            self.L.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
                    if self.L.leads and self.L.Pulse.steps_left() > 6 and self.L.refuse_blocks == 2:
                        self.L.refuse_blocks = 3
                        out.append((cid, 'Not run yet (no step used). Fitting leads remain (' + ', '.join(self.L.leads[:4]) + '). Do not end without a purchase yet: cart the best fit and check out, or open one more page of its category first.'))
                        continue
            if self.L.Basket.is_walkaway(name, args) and self.L.Pulse.steps_left() <= 5 and (not self.L.late_held):
                lines = [i for i in self.L.built if not self.L.SealWatch.is_dry(i)]
                fresh = self.L.cart_view is not None and self.L.cart_seen_at >= self.L.cart_at
                missing = [i for i in lines if i not in self.L.Catalog.basket_ids() or (i in self.L.dropped and (not fresh))]
                if len(lines) >= len(self.L.items) and len(missing) + 1 <= self.L.Pulse.steps_left():
                    self.L.late_held = True
                    self.L.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.L.Basket.is_walkaway(name, args) and (not self.L.leads) and (not self.L.pushed) and (self.L.Pulse.steps_left() > self.L.max_steps // 3):
                self.L.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.L.Pulse.steps_left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.L.Basket.is_walkaway(name, args):
                # sell-out does not prove the store has no fit — hunt and cart a backup first
                if (self.L.gone or self.L.unavailable) and self.L.Pulse.steps_left() > 2:
                    lost = list(self.L.gone) or list(self.L.unavailable)
                    self.L.SealWatch.hunt_after_dry(lost[:3])
                    recovered = self.L.SealWatch.recover_sold_out()
                    if recovered:
                        out.append((cid, recovered))
                        continue
                    picks = self.L.SealWatch.swap_candidates(lost[:3])
                    if picks and getattr(self.L, 'swap_refuse_block', 0) < 1:
                        self.L.swap_refuse_block = 1
                        self.L.leads = picks[:4]
                        out.append((cid, 'Not run yet (no step used). A cart line sold out earlier, but listings that still appear to fit remain: ' + ', '.join(picks[:4]) + '. Add the best one at the requested quantity and check out; end without a purchase only if each of them fails.'))
                        continue
                done = self.L.Basket.auto_seal()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.L.finished else '')))
                    continue
                args = self.L.Basket.mute_ids(name, args)
            if ending and self.L.Pulse.steps_left() >= 2:
                go, said = self.L.Briefing.run_ritual('order' if self.L.Basket.is_checkout(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.L.Catalog.basket_text()))
                    continue
                if said:
                    trimmed = '\n[forge] ' + said + trimmed
            reply = self.L.Pulse.env_step(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _to_json(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.L.compare_bad = min(self.L.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[forge] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.L.Basket.side_limit()} per call.'
                else:
                    self.L.compare_ok = max(self.L.compare_ok, len(args[many]))
            self.L.recent = (self.L.recent + [mark])[-3:]
            self.L.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.L.rows for x in (v if isinstance(v, list) else [v])))))
            self.L.Catalog.ingest(observation)
            if name == self.L.roles.get('compare'):
                rows = self.L.Catalog.rows_in(observation)
                if rows:
                    self.L.facts_from[name] = self.L.facts_from.get(name, False) or any((_attr_block(r) for _, r in rows))
            if name in readers - {self.L.roles.get('search')}:
                self.L.Catalog.mark_view(name, observation)
                self.L.AttrBench.remember_passed_cart()
            plain = {tool for job, tool in self.L.roles.items() if job != 'search'}
            paged = self.L.Catalog.auto_page(name, args, observation) if name not in plain and (not ending) else ''
            if self.L.Catalog.id_arg_of(name) and name not in (self.L.roles.get('cart'), self.L.roles.get('buy'), self.L.roles.get('refuse'), self.L.roles.get('ask')):
                listing = self.L.Catalog.extract_id(observation) if isinstance(observation, dict) else None
                if listing:
                    self.L.facts_from[name] = self.L.facts_from.get(name, False) or bool(_attr_block(observation))
                    if name == self.L.roles.get('open') and (not self.L.facts_from[name]):
                        other = next((t for t in self.L.schemas if t != name and self.L.Catalog.id_arg_of(t) and (self.L.facts_from.get(t) is not False) and (t not in (self.L.roles.get('cart'), self.L.roles.get('buy'), self.L.roles.get('refuse'), self.L.roles.get('ask'), self.L.roles.get('search')))), None)
                        if other:
                            self.L.roles['open'] = other
                    if not self.L.facts_from[name] and name != self.L.roles.get('open') and (not self.L.roles.get('offer')):
                        self.L.roles['offer'] = name
                    if name == self.L.SealWatch.stock_tool():
                        self.L.offer_seen[listing] = self.L.step
            if name == self.L.roles.get('cart'):
                self.L.Basket.track_basket(name, args)
                self.L.cart_said = _to_json(observation)
                self.L.recent = []
                counts = [_as_int(v) for v in args.values() if not isinstance(v, str) and _as_int(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and (value in self.L.rows):
                        if counts[0] <= 0 and (not _id_in_obs(observation, value)):
                            self.L.dropped.add(value)
                        elif counts[0] > 0:
                            self.L.dropped.discard(value)
                self.L.AttrBench.remember_passed_cart()
                lines = [i for i in self.L.Catalog.basket_ids() if not self.L.SealWatch.is_dry(i) and i not in self.L.dropped]
                if len(lines) >= len(self.L.built):
                    self.L.built = lines
            if name == self.L.roles.get('search') and self.L.Catalog.rows_in(observation):
                self.L.searched = (self.L.searched + [dict(args)])[-8:]
            if name == self.L.roles.get('cart_view'):
                self.L.cart_view = observation
                self.L.cart_seen_at = self.L.step
            if self.L.Basket.is_checkout(name, args):
                self.L.buys += 1
                if not reply.get('finished'):
                    self.L.refused_at = self.L.step
            text = self.L.Pulse.render(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.L.Catalog.cheap_rank_note([observation] + self.L.paged_obs) + self.L.Catalog.price_cap_miss(name, args, observation)
            base = text
            proof = self.L.Catalog.attr_evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.L.roles.get('cart') and any((_as_int(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.L.rows and _id_in_obs(observation, v) for v in args.values())):
                    text += '\n[forge] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.L.note_sold = False
            lines = self.L.Catalog.note_lines(observation)
            if lines:
                meaning = self.L.Catalog.interpret_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.L.Basket.is_checkout(name, args) and (not reply.get('finished'))
            # Sell-out wording may sit on observation, reply.refusal, or only in rendered tool text.
            spoken = (self.L.Catalog.obs_speech(observation) + ' ' + self.L.Catalog.obs_speech(reply) + ' ' + text).lower()
            again = refused and any((line in spoken for line in self.L.sold_lines))
            loose = self.L.note_sold and (refused or (not self.L.note_named and (not self.L.Basket.is_checkout(name, args))))
            dry_words = (
                'sold out', 'unavailable', 'ran dry', 'all gone', 'no longer', "trail's closed",
                'out of stock', 'biscuits at a church', 'chase a ghost', 'apparently all gone',
            )
            if refused and self.L.Catalog.basket_ids() and (not self.L.finished):
                if any(w in spoken for w in dry_words) or again or loose:
                    for ident in list(self.L.Catalog.basket_ids()):
                        self.L.unavailable.add(ident)
                    self.L.note_sold = True
                    seen = self.L.SealWatch.detect_dry()
                    if seen:
                        text += "\n[Forge re-read the cart's lines after the refusal]\n" + seen
                elif loose or again:
                    seen = self.L.SealWatch.detect_dry()
                    if seen:
                        text += "\n[Forge re-read the cart's lines after the refusal]\n" + seen
            recovered = self.L.SealWatch.recover_sold_out()
            if recovered:
                text += '\n' + recovered
            else:
                swap = self.L.SealWatch.dry_swap_note()
                if swap:
                    text += '\n' + swap
                recovered = self.L.SealWatch.recover_sold_out()
                if recovered:
                    text += '\n' + recovered
            self.L.note_sold = False
            if self.L.queued:
                text += "\n[notes seen on Forge's own calls] " + ' | '.join(self.L.queued)
                self.L.queued = []
            if name in readers:
                self.L.read_cache[mark] = (self.L.version, self.L.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def force_end(self, again: bool=False) -> str:
        if self.L.finished or self.L.broke or self.L.Pulse.steps_left() <= 0:
            return ''
        buy, refuse = (self.L.roles.get('buy'), self.L.roles.get('refuse'))
        fresh = self.L.cart_view is not None and self.L.cart_seen_at >= self.L.cart_at
        holding = bool(self.L.Catalog.basket_ids()) if fresh else bool(self.L.cart_lines)
        if buy and (holding or (self.L.cart_touched and (not fresh) and (self.L.Pulse.steps_left() >= 2))):
            faults: list[str] = []
            args = _coerce(dict(self.L.brief.get('buy_args') or {}), self.L.schemas.get(buy), 'args', faults)
            if not faults:
                if self.L.Pulse.steps_left() >= 2:
                    self.L.Briefing.run_ritual('order')
                if self.L.finished or self.L.Pulse.steps_left() <= 0:
                    return ''
                reply = self.L.Pulse.env_step(buy, args)
                self.L.trace.append({'kind': 'forced_buy', 'reply': _shrink(reply, 200)})
                if self.L.finished or self.L.Pulse.steps_left() <= 0:
                    return ''
                if reply is not None and again:
                    self.L.buys += 1
                    self.L.refused_at = self.L.step
                    self.L.Catalog.after_obs(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.L.Pulse.render(reply.get('observation'))
        if refuse:
            faults = []
            args = _coerce(self.L.Basket.mute_ids(refuse, dict(self.L.brief.get('refuse_args') or {})), self.L.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.L.Pulse.steps_left() >= 2:
                    self.L.Briefing.run_ritual('end')
                if self.L.finished or self.L.Pulse.steps_left() <= 0:
                    return ''
                reply = self.L.Pulse.env_step(refuse, args)
                self.L.trace.append({'kind': 'forced_refuse', 'reply': _shrink(reply, 200)})
                if reply is not None and (not self.L.finished):
                    self.L.Catalog.after_obs(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.L.Pulse.render(reply.get('observation'))
        return ''

    def execute(self) -> list[dict[str, Any]]:
        if self.L.finished:
            return self.L.trace
        self.L.Briefing.parse_mission()
        self.L.trace.append({'kind': 'brief', 'brief': self.L.brief})
        opening = self.L.view.get('opening')
        if opening is not None:
            self.L.Catalog.ingest(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.L.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.L.Pulse.render(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.L.Briefing.desk_status()
        self.L.chat = [{'role': 'system', 'content': SYS_BUYER}, {'role': 'user', 'content': intro}]
        self.loop_turns(TURN_CAP, HARD_USD)
        if not self.L.finished:
            turned = self.force_end(again=True)
            if turned and (not self.L.finished) and (not self.L.broke) and (not self.L.no_credit) and (self.L.Pulse.steps_left() > 1):
                self.L.chat.append({'role': 'user', 'content': '[forge] Forge tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.L.Briefing.desk_status()})
                self.loop_turns(self.L.loop_calls + 10, HARD_USD + 0.01)
            if not self.L.finished:
                self.force_end()
        self.L.trace.append({'kind': 'end', 'finished': self.L.finished, 'steps': self.L.step, 'cost': round(self.L.cost, 5), 'calls': self.L.calls})
        return self.L.trace

    def loop_turns(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.L.finished and (not self.L.broke) and (self.L.Pulse.steps_left() > 0) and (self.L.loop_calls < loop_cap) and (self.L.cost < spend_cap) and (self.L.Pulse.clock() < LAST_TURN or not self.L.Catalog.basket_ids()):
            if self.L.Briefing.should_interview():
                summary = self.L.Briefing.run_interview()
                if summary:
                    self.L.chat.append({'role': 'user', 'content': summary + '\n' + self.L.Briefing.desk_status()})
                continue
            self.L.Pulse.shrink_history()
            self.L.loop_calls += 1
            message = self.L.Pulse.infer(self.L.chat, [FAST, BACKUP], 3000, self.L.Basket.visible_tools())
            if message is None:
                self.L.misses += 1
                if self.L.no_credit or self.L.calls >= CALL_CAP or self.L.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _as_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.L.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _to_json(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _to_json(content) if content else ''
            if not calls:
                idle += 1
                self.L.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.L.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.L.Briefing.desk_status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.L.roles.get('cart'), self.L.roles.get('buy'), self.L.roles.get('refuse')} - {None}
            if not self.L.asked and self.L.roles.get('ask') and (self.L.Pulse.ask_slots() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.L.Pulse.steps_left() > 8):
                self.L.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.L.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.L.Briefing.run_interview()
                self.L.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.L.Briefing.desk_status()})
                continue
            self.L.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.L.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.dispatch(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.L.Briefing.desk_status()
                self.L.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

class ForgeLoop:
    """Owns episode state; desks collaborate through this loop."""
    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _as_int(self.view.get('max_steps'))
        if steps is None and isinstance(self.view.get('max_steps'), float):
            steps = int(self.view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.finished = bool(self.view.get('finished'))
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
        self.items: list[GoodsLine] = []
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
        self.started = time.monotonic()
        self.list_asks = True
        self.late_asked = False
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.lane_misses: dict[int, int] = {}
        self.follow_up: list[tuple[int, list[str], bool]] = []
        self.one_product: set[int] = set()
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
        self.basket_told = -1
        self.report_done = False
        self.report_mark = None
        self.seal_reread_mark = None
        self.note_named = False
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
        self.cart_said = ''
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.swap_forced = 0
        self.swap_hunt_done = False
        self.swap_refuse_block = 0
        self.passed_cart: dict[str, int] = {}
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
        self.cheap_marks: set[str] = set()
        self.widen_done = False
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.attr_holds = 0
        self.attr_learned: dict[str, dict[str, set[Any]]] = {}
        self.attr_asks = 0
        self.attr_pressed = False
        self.whole_said: set[str] = set()
        self.attr_shown = ''
        self.stock_checked = None
        self.Pulse = Pulse(self)
        self.Briefing = Briefing(self)
        self.Catalog = Catalog(self)
        self.AttrBench = AttrBench(self)
        self.Basket = Basket(self)
        self.SealWatch = SealWatch(self)
        self.Conductor = Conductor(self)


    def execute(self) -> list[dict[str, Any]]:
        return self.Conductor.execute()


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = ForgeLoop(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.execute()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.Conductor.force_end()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace

__all__ = ['agent_main']
