"""Astrolabe: a shopping agent for ORO Bench worlds.

How an episode runs
-------------------
1. Brief: one model call reads the shopper's request into items, stated requirements, budget, the ranking rule
   ("cheapest", "largest ..."), the message allowance and the procedure the shopper lays down.
2. Survey: the loop model searches each item's shelf and opens listings in full; every observation goes through
   a reader that turns dicts, nested envelopes and "key: value | key: value" text rows into one listing store.
3. Interview: once the store shows each item's properties, code writes the questions. A message names up to
   eight properties; the shopper answers the first one it keeps a view on, so the order and the rotation after
   each reply decide what is found. A preference that points at another listing ("same as what Shop X's ...
   gives") is resolved by code: it finds that listing, reads it and follows the listing it names in turn.
4. Choice: code checks every opened listing against every requirement (words, numbers with their units,
   model codes a part must fit) and keeps, per item, the listings that pass, best first by the ranking rule. The
   loop model sees that table in a status block after each step and builds the cart.
5. Checkout: the order goes through a guard that reads the cart back, checks lines, quantities, stock and money,
   and offers the best known fit in place of a failing line. Store notes (sold out, a change of mind) are acted
   on. A decline needs proof that nothing on offer fits.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# models: a cheap one drives the store loop and reads replies; a stronger one plans the interview and reviews
# the order; a third family answers when the first is down
FAST = "anthropic/claude-haiku-5.5"
DEEP = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"

# money per episode (USD): a race run of 90 tasks has a 5 USD key, a qualifying run of 30 a 2 USD key
SPEND_SOFT = 0.030      # past this, the strong model is used only for the order review
SPEND_HARD = 0.045      # past this, the loop stops and code finishes the episode
SPEND_STEP = 0.0003     # long episodes may spend a little more per allowed step
SPEND_CEIL = 0.058

# time per episode (seconds of wall clock); the run's sandbox budget is shared by all tasks
CLOCK_LIMIT = 290.0
CLOCK_FINISH = 245.0    # past this, code finishes the episode
CLOCK_ASK = 170.0       # the interview must start before this

LOOP_TURNS = 46         # loop model turns per episode
MODEL_CALLS = 75        # every model call of the episode
VIEW_CHARS = 1400       # a listing page as the loop model sees it
ROW_CHARS = 230         # a search row as the loop model sees it
OBS_CHARS = 9000        # one observation as the loop model sees it
CHAT_CHARS = 120000     # chat history kept for the loop model
NAMES_PER_MESSAGE = 8   # properties named in one question
PAGES_BEFORE_ASK = 4    # full listings read per item before the interview
WAITS = (2.0, 5.0, 9.0)

_llm = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


# ----------------------------------------------------------------------------------------------------------
# small text helpers
# ----------------------------------------------------------------------------------------------------------

def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def _json_obj(text: Any) -> dict[str, Any] | None:
    """The first JSON object in a model reply (code fences and prose around it are skipped)."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or "{" not in text:
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    start = body.find("{")
    while start >= 0:
        depth, quoted, escaped = 0, False, False
        for at in range(start, len(body)):
            ch = body[at]
            if quoted:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = False
                continue
            if ch == '"':
                quoted = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        got = json.loads(body[start: at + 1])
                    except ValueError:
                        break
                    return got if isinstance(got, dict) else None
        start = body.find("{", start + 1)
    return None


def _snake(name: str) -> str:
    """camelCase and kebab-case to words: 'cookingStyle' -> 'cooking style'."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return re.sub(r"[_\-./]+", " ", text)


def _key(name: Any) -> str:
    """A property name folded for comparison: 'cooking_style', 'cookingStyle' and 'Cooking Style' are one key."""
    return " ".join(re.findall(r"[a-z0-9]+", _snake(str(name)).lower()))


def _words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", str(text or "").lower())


def _stem(word: str) -> str:
    """A crude singular: enough to treat 'cases' and 'case', 'batteries' and 'battery' as one word."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and re.search(r"(ss|sh|ch|x|z)es$", w):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    return w


def _stems(text: Any) -> set[str]:
    return {_stem(w) for w in _words(text) if not w.isdigit() or len(w) > 1}


def _number(value: Any) -> float | None:
    """A price or count as a float: 1299, '1,299', '$12.50', '12.50 USD'; None when there is none."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for name in ("amount", "value", "price", "total"):
            if name in value:
                return _number(value[name])
        return None
    if not isinstance(value, str):
        return None
    found = re.search(r"-?\d[\d,]*(?:\.\d+)?", value.replace(" ", " "))
    if not found:
        return None
    raw = found.group(0)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        raw = raw.replace(",", "")
    elif "," in raw and "." not in raw and re.fullmatch(r"-?\d+,\d{1,2}", raw):
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _quantities(text: Any) -> list[tuple[float, str]]:
    """Every number in a text with the word written right after it: '5000 mAh' -> (5000, 'mah'), '9h' -> (9, 'h').
    Units are read from the text itself, so no list of units is needed."""
    out: list[tuple[float, str]] = []
    for found in re.finditer(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?)\s*([a-zA-Z\"'%]{0,6})", str(text or "")):
        raw = found.group(1)
        raw = raw.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw) else raw.replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        out.append((value, found.group(2).lower()))
    return out


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    return None


def _same(a: Any, b: Any) -> bool:
    return _key(a) == _key(b) and _key(a) != ""


def _overlap(a: Any, b: Any) -> float:
    """Share of the shorter text's word stems found in the other text."""
    x, y = _stems(a), _stems(b)
    if not x or not y:
        return 0.0
    return len(x & y) / min(len(x), len(y))


def _schema_kinds(spec: Any) -> set[str]:
    if not isinstance(spec, dict):
        return set()
    kind = spec.get("type")
    kinds = set(kind) if isinstance(kind, list) else {kind} if isinstance(kind, str) else set()
    for alt in spec.get("anyOf") or spec.get("oneOf") or []:
        kinds |= _schema_kinds(alt)
    return kinds - {None}


# ----------------------------------------------------------------------------------------------------------
# reading what the store returns
# ----------------------------------------------------------------------------------------------------------
# Field names differ from world to world ("list_price" / "price", "merchant" / "sold_by_name", "facts" as a dict or
# as "k=v; k=v" text). Each canonical field is found by the words of its key, never by a fixed key.

_FIELD_WORDS = (
    # (canonical field, words of a key that name it), checked in this order
    ("notes", r"\b(?:know|comments?|remarks?|notes?|print)\b"),
    ("shop", r"\b(?:merchant|seller|sold|shop|store|vendor|retailer)\b"),
    ("brand", r"\b(?:brand|maker|manufacturer|make)\b"),
    ("category", r"\b(?:category|categories|shelf|department|section|aisle|group)\b"),
    ("stock", r"\b(?:stock|hand|available|availability|inventory)\b"),
    ("price", r"\b(?:price|cost)\b"),
    ("title", r"\b(?:title|name|heading)\b"),
    ("facts", r"\b(?:facts|attributes|specs|specifications|properties|details|features|datasheet)\b"),
    ("about", r"\b(?:description|blurb|summary|overview|about|text|body)\b"),
)
_ID_KEY = re.compile(r"(?:^|\s)(?:id|ids|ref|sku|code|number|no)$|^(?:listing|article|product|item)$")
_ID_VALUE = re.compile(r"^[A-Za-z]{1,4}[-_][A-Za-z0-9]{6,40}$")
_TITLE_KEY = re.compile(r"\b(?:title|name|heading)\b")
_LIST_ID = re.compile(r"^[A-Za-z]{0,6}[-_]?[A-Za-z0-9]{4,40}$")


def _scalar(text: str) -> Any:
    low = text.strip().lower()
    if low in ("none", "null", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    return text.strip()


def _pairs(text: Any) -> dict[str, str]:
    """'k=v; k=v' (or 'k: v; k: v') text as a dict; {} when the text is not written that way."""
    out: dict[str, str] = {}
    if not isinstance(text, str) or ("=" not in text and ":" not in text):
        return out
    parts = [p for p in re.split(r";\s+|;$|\n", text) if p.strip()]
    for part in parts:
        found = re.match(r"\s*([^=:;]{1,60}?)\s*[=:]\s*(.*)$", part, re.S)
        if not found:
            continue
        out[found.group(1).strip()] = found.group(2).strip()
    return out if len(out) >= max(1, len(parts) // 2) else {}


def _text_record(text: Any) -> dict[str, Any] | None:
    """A listing written as one text ('ref: p-1 | title: ... | price: 18 | facts: a=b; c=d'), as a dict.
    A part without a 'name:' belongs to the field before it (titles may hold ' | ')."""
    if not isinstance(text, str) or len(text) < 12:
        return None
    parts = re.split(r"\s+\|\s+|\n(?=[A-Za-z][\w .\-]{0,30}:\s)", text.strip())
    if len(parts) < 3:
        return None
    rec: dict[str, Any] = {}
    last = ""
    for part in parts:
        found = re.match(r"\s*([A-Za-z][\w .\-]{0,40}?)\s*:\s?(.*)$", part, re.S)
        if found and (not last or found.group(1).strip().lower() not in rec):
            last = found.group(1).strip()
            rec[last] = found.group(2).strip()
        elif last:
            rec[last] = f"{rec[last]} | {part.strip()}"
    if len(rec) < 3:
        return None
    if not any(_ID_KEY.match(_key(k)) for k in rec) and not any(_TITLE_KEY.match(_key(k)) for k in rec):
        return None
    return {k: (_scalar(v) if isinstance(v, str) else v) for k, v in rec.items()}


def _envelope(value: Any) -> tuple[Any, list[str]]:
    """Split a store note off its payload: a two-key dict whose one value is a list of sentences."""
    notes: list[str] = []
    for _ in range(3):
        if not isinstance(value, dict) or len(value) != 2:
            break
        a, b = list(value)
        hit = None
        for body, side in ((a, b), (b, a)):
            v = value[side]
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and not any(_text_record(x) for x in v):
                hit = (body, side)
                break
            if isinstance(v, str) and len(v) > 20 and " " in v and not _text_record(v) and isinstance(value[body], (dict, list)) \
                    and re.search(r"refus|note|warn|notice|message|alert|news", _key(side)):
                hit = (body, side)
                break
        if not hit:
            break
        body, side = hit
        got = value[side]
        notes += got if isinstance(got, list) else [got]
        value = value[body]
    return value, notes


def _flat_facts(value: Any) -> dict[str, str]:
    """A facts field in any of its shapes as {folded key: text value}."""
    out: dict[str, str] = {}
    if isinstance(value, str):
        value = _pairs(value)
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict):
                name = next((entry[k] for k in entry if re.search(r"name|key|label|property|attribute", _key(k))), None)
                val = next((entry[k] for k in entry if re.search(r"value|text|setting", _key(k))), None)
                if isinstance(name, str):
                    out[_key(name)] = _clip(val, 300) if val is not None else ""
            elif isinstance(entry, str) and ("=" in entry or ":" in entry):
                out.update({_key(k): v for k, v in _pairs(entry).items()})
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if not isinstance(v2, (dict, list)):
                        out.setdefault(_key(k2), "" if v2 is None else str(v2))
                continue
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v if not isinstance(x, (dict, list)))
            out.setdefault(_key(k), "" if v is None else str(v))
    return {k: v for k, v in out.items() if k}


def _listing(raw: Any) -> dict[str, Any] | None:
    """A store record in canonical form: id, title, price, stock, category, shop, brand, facts, notes, about,
    extra (other plain fields). None when the record carries no id."""
    if isinstance(raw, str):
        raw = _text_record(raw)
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {"facts": {}, "extra": {}}
    id_key = next((n for n, v in raw.items() if isinstance(v, (str, int)) and not isinstance(v, bool) and _ID_KEY.search(_key(n))
                   and _LIST_ID.match(str(v).strip())), None)
    if id_key is None:
        id_key = next((n for n, v in raw.items() if isinstance(v, str) and _ID_VALUE.match(v.strip()) and re.search(r"\d", v)
                       and not re.search(r"categor|shelf|section|title|name", _key(n))), None)
    for name, value in raw.items():
        k = _key(name)
        if not k:
            continue
        if name == id_key and "id" not in out:
            out["id"] = str(value).strip()
            continue
        field = next((f for f, words in _FIELD_WORDS if re.search(words, k)), None)
        if field == "facts" and isinstance(value, (dict, list, str)) and not (isinstance(value, str) and not _pairs(value)):
            out["facts"].update(_flat_facts(value))
            out["has_facts"] = True
        elif field in ("notes", "about"):
            if isinstance(value, str) and value.strip() and value.strip().lower() not in ("none", "null"):
                out[field] = (out.get(field, "") + " " + value).strip()
        elif field == "stock" and field not in out and not isinstance(value, (dict, list)):
            num = _number(value) if not isinstance(value, bool) else None
            out["stock"] = value if isinstance(value, bool) else (num > 0 if num is not None else
                                                                  (None if value is None else not re.search(r"\bout\b|\bno\b|sold|back", str(value).lower())))
        elif field == "price" and field not in out and _number(value) is not None and not isinstance(value, bool):
            out["price"] = _number(value)
        elif field in ("shop", "brand", "category", "title") and field not in out and (isinstance(value, str) or value is None):
            if value is not None or field == "brand":
                out[field] = value
        elif isinstance(value, dict) and len(value) >= 2 and sum(1 for v in value.values() if not isinstance(v, (dict, list))) >= max(2, len(value) * 2 // 3):
            # a map of plain values under a key of its own naming ('traits_map', 'spec_sheet'): the listing's facts
            out["facts"].update(_flat_facts(value))
            out["has_facts"] = True
        elif not isinstance(value, (dict, list)) and value is not None and str(value).strip():
            out["extra"][k] = value
    if "id" not in out:
        return None
    return out


def _records(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Every listing inside an observation (lists of dicts, lists of text rows, one page as text or dict)."""
    found: list[dict[str, Any]] = []
    if depth > 5:
        return found
    if isinstance(value, str):
        rec = _listing(value)
        return [rec] if rec else []
    if isinstance(value, list):
        for entry in value:
            found += _records(entry, depth + 1)
        return found
    if isinstance(value, dict):
        rec = _listing(value)
        if rec and (rec.get("title") or rec.get("price") is not None or rec.get("facts")):
            return [rec]
        for v in value.values():
            if isinstance(v, (list, dict)) or (isinstance(v, str) and " | " in v):
                found += _records(v, depth + 1)
    return found


def _paging(value: Any) -> dict[str, Any]:
    """Hit count and next-page marker of a search result, where the store gives them."""
    out: dict[str, Any] = {}
    if not isinstance(value, dict):
        return out
    for name, v in value.items():
        k = _key(name)
        if isinstance(v, (list, dict)):
            continue
        if re.search(r"\b(?:count|total|hits|matches|found|results)\b", k) and _count(v) is not None and "count" not in out:
            out["count"] = _count(v)
        elif re.search(r"\b(?:next|further|more|cursor|token|continue|continuation)\b", k):
            out["next"] = v
        elif re.search(r"\bpage\b", k) and _count(v) is not None:
            out["page"] = _count(v)
    return out


# ----------------------------------------------------------------------------------------------------------
# what each world tool is for
# ----------------------------------------------------------------------------------------------------------
# Each world names its tools its own way (one says "ask the buyer", another "message the customer"); the role is read from
# its name, description and parameters. Roles: search, page (one listing in full), facts (a listing's property list),
# compare (several listings), ask (message to the shopper), cart (change the cart), cart_view, buy, decline, and
# close (one tool that ends the episode either way through an option).

_BUY_WORDS = r"check ?out|purchase|buy|settle|place (?:the |an )?order|commit|confirm (?:the )?order|pay\b|complete (?:the )?order|finali[sz]e"
_DECLINE_WORDS = r"give up|no purchase|without (?:a )?(?:purchase|buying)|decline|refuse|pass\b|skip|abandon|nothing (?:to )?buy|close out with no|walk away|reject"
_CART_WORDS = r"cart|bag\b|basket|stash|selection|trolley|hold to|keep to|order lines?"
_ASK_WORDS = r"shopper|customer|buyer|user|client|question|ask|say|write|message|reply|note to|chat|talk|contact"
_PAGE_WORDS = r"page|in full|full|open|show|view|describe|detail|single (?:article|listing|item|product)|look at"
_FACTS_WORDS = r"property list|datasheet|attributes|specs|specifications|properties|facts"
_COMPARE_WORDS = r"compare|weigh|side by side|several|multiple|many"
_SEARCH_WORDS = r"search|find|look ?up|locate|query|filter|browse|list|range|catalog|shelf|section|listings"
_TEXT_ARG = r"query|words?|text|search|input|keywords?|terms?|phrase|q$|for input|free"
_CAT_ARG = r"categor|shelf|department|section|group|aisle|range|kind|label"
_PAGE_ARG = r"page|leaf|offset|cursor|token|next"
_CAP_ARG = r"cap|max(?:imum)? ?price|price ?max|budget|spend|upper|ceiling|at most"
_BRAND_ARG = r"brand|maker|manufacturer"
_ID_ARG = r"id$|ids$|code|listing|product|item|article|sku|ref|number|target|chosen|picked"
_QTY_ARG = r"qty|quantity|amount|count|number|how many|units|copies"


def _tool_text(tool: dict[str, Any]) -> tuple[str, str, dict[str, Any], list[str]]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = str(fn.get("name") or "")
    desc = str(fn.get("description") or "")
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = [r for r in params.get("required") or [] if isinstance(r, str)]
    return name, (_snake(name) + " . " + desc).lower(), props, required


def _enum_of(spec: Any) -> list[str]:
    if not isinstance(spec, dict):
        return []
    opts = spec.get("enum")
    if isinstance(opts, list):
        return [str(o) for o in opts]
    for alt in spec.get("anyOf") or spec.get("oneOf") or []:
        got = _enum_of(alt)
        if got:
            return got
    return []


def _roles(tools: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Map roles to tool names. Returns (roles, schemas by tool, extra): extra carries the close tool's option for
    buying and for declining, and the search tools' argument names."""
    roles: dict[str, str] = {}
    schemas: dict[str, dict[str, Any]] = {}
    extra: dict[str, Any] = {"search": {}}
    scored: list[tuple[str, str, dict[str, Any], list[str]]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name, text, props, required = _tool_text(tool)
        if not name:
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        schemas[name] = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        scored.append((name, text, props, required))

    def pnames(props: dict[str, Any]) -> list[str]:
        return [_key(p) for p in props]

    for name, text, props, required in scored:
        keys = list(props)
        strings = [k for k in keys if "string" in _schema_kinds(props[k]) and not _enum_of(props[k])]
        arrays = [k for k in keys if "array" in _schema_kinds(props[k])]
        enums = {k: _enum_of(props[k]) for k in keys if _enum_of(props[k])}
        # a closing tool with an option for buying and one for declining
        for k, opts in enums.items():
            buy = [o for o in opts if re.search(_BUY_WORDS, _snake(o).lower())]
            drop = [o for o in opts if re.search(_DECLINE_WORDS, _snake(o).lower())]
            if buy and drop and len(keys) <= 3 and "close" not in roles:
                roles["close"] = name
                extra["close"] = {"arg": k, "buy": buy[0], "decline": drop[0]}
        if roles.get("close") == name:
            continue
        cart_like = re.search(_CART_WORDS, text) is not None
        # changing the cart: an id with a quantity, or a list of lines
        line_list = any(isinstance(props[a].get("items"), dict) and ((props[a]["items"].get("properties")) or {}) for a in arrays)
        id_arg = [k for k in strings if re.search(_ID_ARG, _key(k))]
        qty_arg = [k for k in keys if {"integer", "number"} & _schema_kinds(props[k]) and re.search(_QTY_ARG, _key(k))]
        cart_verbs = any(re.search(r"\b(?:plus|minus|add|remove|include|discard|append|drop|increase|decrease|put|take|set|delete)\b", _snake(o).lower())
                         for opts in enums.values() for o in opts)
        if (line_list or (id_arg and (qty_arg or enums))) and (cart_like or line_list or (id_arg and qty_arg and (cart_verbs or not enums))) \
                and not re.search(_ASK_WORDS + r"|" + _COMPARE_WORDS, _snake(name).lower()):
            roles.setdefault("cart", name)
            continue
        if not keys or not required and not strings and not arrays:
            if re.search(_BUY_WORDS, text) and not re.search(r"view|show|review|list|look|see|contents", _snake(name).lower()):
                roles.setdefault("buy", name)
            elif re.search(_DECLINE_WORDS, text):
                roles.setdefault("decline", name)
            elif cart_like:
                roles.setdefault("cart_view", name)
            continue
        if re.search(_DECLINE_WORDS, text) and len(strings) <= 1 and not arrays and not re.search(_SEARCH_WORDS, _snake(name).lower()):
            roles.setdefault("decline", name)
            continue
        if re.search(_ASK_WORDS, text) and len(strings) == 1 and not arrays and len(keys) <= 2 \
                and not re.search(_ID_ARG, _key(strings[0])) and not re.search(_SEARCH_WORDS, _snake(name).lower()):
            roles.setdefault("ask", name)
            continue
        id_list = any(re.search(_ID_ARG + r"|listings?|products?|items?|chosen|picked|selected", _key(a)) and
                      "string" in _schema_kinds((props[a] or {}).get("items") or {}) for a in arrays)
        if arrays and not strings and (re.search(_COMPARE_WORDS + r"|" + _PAGE_WORDS + r"|contrast", text) or (id_list and len(keys) == 1)):
            roles.setdefault("compare", name)
            continue
        one_id = len(required) <= 1 and len(strings) == 1 and len(keys) <= 2 and re.search(_ID_ARG, _key(strings[0]))
        if one_id and re.search(_FACTS_WORDS, text) and not re.search(r"in full|full", text):
            roles.setdefault("facts", name)
            continue
        if one_id and re.search(_PAGE_WORDS + "|" + _FACTS_WORDS, text):
            roles.setdefault("page", name)
            continue
        if strings or enums:
            spec: dict[str, Any] = {}
            for k in keys:
                kk = _key(k)
                kinds = _schema_kinds(props[k])
                if k in enums:
                    spec.setdefault("mode", k)
                    spec.setdefault("modes", enums[k])
                elif "string" in kinds and re.search(_CAT_ARG, kk) and "cat" not in spec:
                    spec["cat"] = k
                elif "string" in kinds and re.search(_BRAND_ARG, kk) and "brand" not in spec:
                    spec["brand"] = k
                elif "string" in kinds and re.search(_TEXT_ARG, kk) and "text" not in spec:
                    spec["text"] = k
                elif ({"integer", "number"} & kinds) and re.search(_CAP_ARG, kk) and "cap" not in spec:
                    spec["cap"] = k
                elif re.search(_PAGE_ARG, kk) and "page" not in spec:
                    spec["page"] = k
                    spec["page_max"] = props[k].get("maximum") if isinstance(props[k], dict) else None
            if "text" not in spec and "cat" not in spec and len(strings) == 1:
                spec["text"] = strings[0]
            if spec.get("text") or spec.get("cat"):
                extra["search"][name] = spec
                if spec.get("text"):
                    roles.setdefault("search", name)
                else:
                    roles.setdefault("browse", name)
    if "page" not in roles and "facts" in roles:
        roles["page"] = roles["facts"]
    if "search" not in roles and "browse" in roles:
        roles["search"] = roles["browse"]
    if "close" in roles:
        roles.setdefault("buy", roles["close"])
        roles.setdefault("decline", roles["close"])
    return roles, schemas, extra


def _fit_args(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Arguments trimmed to what a schema accepts: unknown names dropped, numbers and strings coerced, enum values
    matched loosely, lengths and ranges respected."""
    props = (schema or {}).get("properties") if isinstance((schema or {}).get("properties"), dict) else None
    if props is None:
        return dict(args)
    out: dict[str, Any] = {}
    for name, value in args.items():
        if name not in props or value is None:
            continue
        spec = props[name] if isinstance(props[name], dict) else {}
        kinds = _schema_kinds(spec)
        opts = _enum_of(spec)
        if opts:
            pick = next((o for o in opts if str(o) == str(value)), None) or next((o for o in opts if _same(o, value)), None)
            if pick is None:
                continue
            value = pick
        elif "integer" in kinds and not isinstance(value, bool):
            num = _number(value)
            if num is None:
                continue
            value = int(round(num))
        elif "number" in kinds and not isinstance(value, bool):
            num = _number(value)
            if num is None:
                continue
            value = num
        elif "string" in kinds and not isinstance(value, str):
            value = str(value)
        if isinstance(value, str) and isinstance(spec.get("maxLength"), int):
            value = value[: spec["maxLength"]]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(spec.get("maximum"), (int, float)) and value > spec["maximum"]:
                value = type(value)(spec["maximum"])
            if isinstance(spec.get("minimum"), (int, float)) and value < spec["minimum"]:
                value = type(value)(spec["minimum"])
        out[name] = value
    return out


# ----------------------------------------------------------------------------------------------------------
# the episode: environment calls, model calls, money and time
# ----------------------------------------------------------------------------------------------------------

class Core:
    def __init__(self, problem: dict[str, Any]) -> None:
        self.started = time.monotonic()
        env = problem.get("environment") if isinstance(problem.get("environment"), dict) else {}
        self.binding = dict(env.get("binding") or {})
        view = env.get("policy_view") if isinstance(env.get("policy_view"), dict) else {}
        self.view = view
        self.request = str(view.get("query") or problem.get("query") or "")
        self.tools = [t for t in view.get("tools") or [] if isinstance(t, dict)]
        self.max_steps = _count(view.get("max_steps")) or 40
        self.finished = bool(view.get("finished"))
        self.roles, self.schemas, self.tool_extra = _roles(self.tools)
        self.step = 0
        self.trace: list[dict[str, Any]] = []
        self.cost = 0.0
        self.calls = 0
        self.waited = 0.0
        self.broke = False          # the environment refused us for good
        self.no_credit = False      # the inference key is spent
        self.fails = 0
        self.notes_seen: list[str] = []
        self.fresh_notes: list[str] = []
        self.outcome = ""           # "bought" / "declined" once the store accepted an ending
        # longer episodes (several items, more steps) get more loop turns and model calls
        self.turn_cap = max(LOOP_TURNS, self.max_steps)
        self.call_cap = MODEL_CALLS + max(0, self.max_steps - 40)

    # ---- clock, steps, money ----------------------------------------------------------------------------
    def clock(self) -> float:
        return time.monotonic() - self.started

    def left(self) -> int:
        return self.max_steps - self.step

    def spend_cap(self) -> float:
        return min(SPEND_CEIL, SPEND_HARD + SPEND_STEP * max(0, self.max_steps - 40))

    def spent(self, share: float = 1.0) -> bool:
        return self.cost >= self.spend_cap() * share

    def model_room(self) -> float:
        return CLOCK_LIMIT - self.clock()

    # ---- environment --------------------------------------------------------------------------------------
    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One environment step; returns the response (observation, finished) or None when the call was refused."""
        if self.finished or self.broke or self.left() <= 0:
            return None
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                self.trace.append({"kind": "act", "step": self.step, "tool": name, "args": _clip(_dump(args), 300),
                                   "obs": _clip(_dump(data.get("observation")), 400)})
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if re.search(r"finish|final|quarantin|closed|over", low):
                    self.finished = True
                    return None
                expected = re.search(r"expected (?:step )?(\d+)", low)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                self.step += 1
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 403, 404, 422):
                self.broke = status in (401, 403)
                return None
            time.sleep(1.0 + random.random())
        return None

    # ---- models -------------------------------------------------------------------------------------------
    def chat(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int = 1500,
             tools: list[dict[str, Any]] | None = None, effort: str | None = None) -> dict[str, Any] | None:
        """One model reply (tries each model in turn, waits out rate limits); None when none answers."""
        for model in models:
            if self.no_credit or self.calls >= self.call_cap or self.model_room() < 8:
                return None
            self.calls += 1
            for wait in (0.0,) + WAITS:
                if wait:
                    if self.waited + wait > 30 or self.model_room() < wait + 10:
                        break
                    self.waited += wait
                    time.sleep(wait * (0.8 + 0.4 * random.random()))
                message, status = self._call(model, messages, max_tokens, tools, effort)
                if message is not None:
                    return message
                if status == 402:
                    self.no_credit = True
                    return None
                if not (status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)):
                    break
        return None

    def _call(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
              tools: list[dict[str, Any]] | None, effort: str | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith("anthropic/"):
            # the system text and the newest turn carry cache marks: the loop resends a growing chat each turn
            sent = [dict(m) for m in messages]
            for index in sorted({0, len(sent) - 1}):
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if effort:
            body["reasoning"] = {"effort": effort}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = (data["choices"][0] or {}).get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = _count(data["error"].get("code")) or status
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": _clip(error.get("body"), 160)})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1500, effort: str | None = None,
                 system: str | None = None) -> dict[str, Any] | None:
        """A JSON object from the first model that writes one."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        for model in models:
            message = self.chat(messages, [model], max_tokens, None, effort)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.no_credit or self.calls >= self.call_cap:
                return None
        return None


# ----------------------------------------------------------------------------------------------------------
# the listing store
# ----------------------------------------------------------------------------------------------------------

class Store:
    """Every listing seen, merged across search rows and full pages."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: set[str] = set()         # ids seen as a full page (facts / notes shown)
        self.order: list[str] = []          # ids in the order first seen
        self.searches: list[dict[str, Any]] = []
        self.gone: set[str] = set()         # ids the store said are sold out / unavailable

    def add(self, rec: dict[str, Any], full: bool) -> str:
        ident = rec["id"]
        old = self.rows.get(ident)
        if old is None:
            self.rows[ident] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rec.items()}
            self.order.append(ident)
        else:
            for k, v in rec.items():
                if k in ("facts", "extra"):
                    old.setdefault(k, {}).update(v)
                elif v not in (None, "", {}) or k not in old:
                    old[k] = v
        if full and (rec.get("facts") or rec.get("notes") or rec.get("has_facts")):
            self.full.add(ident)
        return ident

    def get(self, ident: str) -> dict[str, Any]:
        return self.rows.get(ident) or {}

    def price(self, ident: str) -> float | None:
        p = self.get(ident).get("price")
        return float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) else None

    def in_stock(self, ident: str) -> bool:
        if ident in self.gone:
            return False
        s = self.get(ident).get("stock")
        return s is not False

    def text_of(self, ident: str) -> str:
        """All words a listing shows: title, facts, notes, description."""
        r = self.get(ident)
        facts = "; ".join(f"{k}: {v}" for k, v in (r.get("facts") or {}).items())
        extra = "; ".join(f"{k}: {v}" for k, v in (r.get("extra") or {}).items() if not re.search(r"price|stock|hand|amount", k))
        return " | ".join(str(x) for x in (r.get("title"), r.get("brand"), r.get("shop"), r.get("category"), facts, extra,
                                             r.get("notes"), r.get("about")) if x)

    def facts(self, ident: str) -> dict[str, str]:
        """A listing's facts with spelling variants of one key folded ('pack type' / 'pack_type' / 'packType')."""
        out: dict[str, str] = {}
        for k, v in (self.get(ident).get("facts") or {}).items():
            out.setdefault(_key(k), str(v))
        return out

    def row_line(self, ident: str) -> str:
        r = self.get(ident)
        stock = "SOLD OUT" if ident in self.gone else ("out of stock" if r.get("stock") is False else "in stock" if r.get("stock") else "")
        bits = [ident, _clip(r.get("title"), 110)]
        if r.get("price") is not None:
            bits.append(f"price {r['price']:g}")
        for name in ("brand", "shop", "category"):
            if r.get(name):
                bits.append(f"{name} {_clip(r[name], 30)}")
        if stock:
            bits.append(stock)
        if ident in self.full:
            bits.append("(page read)")
        line = " | ".join(bits)
        if r.get("about") and ident not in self.full:
            line += " | " + _clip(r["about"], 80)
        return _clip(line, ROW_CHARS)

    def page_text(self, ident: str, limit: int = VIEW_CHARS) -> str:
        r = self.get(ident)
        head = self.row_line(ident)
        facts = self.facts(ident)
        seen: dict[str, str] = {}
        for k, v in facts.items():
            # one line per value spelling: 'pack type' and 'pack_type' carrying the same value show once
            if v not in seen.values() or not any(_overlap(k, s) >= 0.5 for s, sv in seen.items() if sv == v):
                seen[k] = v
        parts = [head, "facts: " + "; ".join(f"{k}={_clip(v, 60)}" for k, v in seen.items())]
        if r.get("notes"):
            parts.append("notes: " + _clip(r["notes"], 600))
        extra = {k: v for k, v in (r.get("extra") or {}).items() if not re.search(r"price|stock|hand|amount|title", k)}
        if extra:
            parts.append("other: " + "; ".join(f"{k}={_clip(v, 60)}" for k, v in extra.items()))
        if r.get("about"):
            parts.append("description: " + _clip(r["about"], 300))
        return _clip("\n".join(parts), limit)

    def key_table(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """For a set of listings: per folded fact key, how many listings carry it and its distinct values."""
        table: dict[str, dict[str, Any]] = {}
        for ident in ids:
            for k, v in self.facts(ident).items():
                if not v or not k:
                    continue
                slot = table.setdefault(k, {"n": 0, "values": {}})
                slot["n"] += 1
                val = _clip(v, 40).lower()
                slot["values"][val] = slot["values"].get(val, 0) + 1
        return table


# ----------------------------------------------------------------------------------------------------------
# reading the request
# ----------------------------------------------------------------------------------------------------------

BRIEF = """You read a shopper's request to an online store assistant and write it down as JSON. Use only what the
request says; do not invent requirements.

Write exactly this JSON object:
{
 "items": [                                   // each separate product requested, listed in the order the request names them
   {"name": "",                               // the product as the shopper names it, a short noun phrase
    "quantity": 1,                            // units of this product: the number the request states, else 1
                                              // (a pair of shoes, a set, a pack is one unit)
    "quantity_words": "",                     // the request's exact words that state that number, "" if none
    "shelf": "",                              // only a store category the request itself names as a range, shelf
                                              // or section ("from the dining room sets range"), else ""
    "must": [                                 // every condition the product itself must meet
      {"property": "",                        // the property, in the request's words ("input voltage", "material")
       "rule": "is",                          // is | is_not | one_of | at_least | at_most | between | mentions
       "value": "",                           // the wanted value in the request's own words, quoted exactly
                                              // ("5V", "polyester", "sleeveless"; never reworded)
       "value2": "",                          // upper end for "between", else ""
       "said": ""}                            // the request's own words for this condition
    ],
    "fits": ""}                               // a device, model or code this product must fit / work with, else ""
 ],
 "budget": null,                              // the money limit: its number exactly as the request writes it (unit as stated), else null
 "budget_scope": "total",                     // "total" for the whole order, "each" when the limit is per unit/item
 "rank": {"rule": "none", "property": "", "said": ""},
     // how to choose among products that fit: cheapest | dearest | most | least | none
     // ("most"/"least" go with the property ranked, e.g. most "battery capacity")
 "message_limit": null,                       // most messages the assistant may send the shopper, else null
 "step_limit": null,                          // most steps / actions allowed, else null
 "in_stock_only": false,                      // true when only items available now may be shown or bought
 "decline_if_none": false,                    // true when the shopper says to report that nothing fits rather than substitute
 "procedure": []                              // any other instruction about HOW to shop (what to check, open, tell,
                                              // or avoid doing), each in the request's own words
}
Notes:
- "Cross off anything with X Y" / "no X" / "not X" is a must with rule is_not.
- A condition stated once for the whole order ("all of it in black") goes in every item's must list.
- A condition that refers to another listing ("same colour as the one from shop Z") is a must whose value is the
  request's own words for that reference.
- The kind of product itself (what it is, the range or category it comes from, e.g. "luxury & formal watches",
  "ready-to-drink UHT milk") is the item's name or shelf, never a must.
- Story, reasons, dates and timing are not conditions. A delivery wish is a must (property "delivery") only when
  it names a delivery option a listing could state, such as "instant delivery".
Write only the JSON."""


class Brief:
    def read_request(self) -> None:
        tools = "\n".join(f"- {t.get('function', t).get('name')}: {_clip(t.get('function', t).get('description'), 160)}" for t in self.tools)
        prompt = BRIEF + "\n\nSTORE TOOLS (for context):\n" + tools + "\n\nREQUEST:\n" + self.request
        got = self.ask_json(prompt, [DEEP, FAST], 2600, "low") or {}
        self.brief_raw = got
        items = []
        for number, raw in enumerate(got.get("items") or [], 1):
            if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
                continue
            qty = _count(raw.get("quantity")) or 1
            said_qty = str(raw.get("quantity_words") or "").strip()
            if qty > 1 and (not said_qty or said_qty.lower() not in self.request.lower()):
                qty = 1     # a number the request does not state is not a quantity
            item = {"n": len(items) + 1, "name": _clip(raw.get("name"), 80), "qty": max(1, min(qty, 50)),
                    "shelf": _clip(raw.get("shelf") or "", 80), "fits": _clip(raw.get("fits") or "", 120)}
            items.append(item)
            for must in raw.get("must") or []:
                if isinstance(must, dict) and re.fullmatch(r"\s*(?:price|cost|budget|total|spend(?:ing)?|availability|stock|in stock|on hand)\s*",
                                                           str(must.get("property") or ""), re.I):
                    continue    # money is the budget's business, availability the stock check's
                if isinstance(must, dict) and str(must.get("property") or must.get("value") or "").strip():
                    self.add_need(item["n"], str(must.get("property") or ""), str(must.get("rule") or "is"),
                                  str(must.get("value") or ""), str(must.get("value2") or ""), "request", str(must.get("said") or ""))
        if not items:
            items = [{"n": 1, "name": _clip(self.request, 60), "qty": 1, "shelf": "", "fits": ""}]
        self.items = items
        budget = _number(got.get("budget"))
        self.budget = budget if budget and budget > 0 else None
        self.budget_each = str(got.get("budget_scope") or "").lower().startswith("each")
        rank = got.get("rank") if isinstance(got.get("rank"), dict) else {}
        self.rank = {"rule": str(rank.get("rule") or "none").lower(), "property": str(rank.get("property") or ""),
                     "said": str(rank.get("said") or "")}
        limit = _count(got.get("message_limit"))
        self.msg_limit = limit if limit is not None and 0 <= limit <= 20 else None
        steps = _count(got.get("step_limit"))
        if steps and 5 <= steps < self.max_steps:
            self.max_steps = steps
        self.stock_only = bool(got.get("in_stock_only"))
        self.decline_ok = bool(got.get("decline_if_none"))
        self.procedure = [_clip(p, 200) for p in got.get("procedure") or [] if isinstance(p, str) and p.strip()][:6]
        self.trace.append({"kind": "brief", "items": self.items, "needs": self.needs, "budget": self.budget,
                           "rank": self.rank, "msgs": self.msg_limit, "steps": self.max_steps})

    def add_need(self, item: int, prop: str, rule: str, value: str, value2: str = "", src: str = "request",
                 said: str = "", pointer: dict[str, Any] | None = None) -> dict[str, Any] | None:
        rule = rule.strip().lower().replace(" ", "_").replace("-", "_")
        if rule not in ("is", "is_not", "one_of", "at_least", "at_most", "between", "mentions", "same_as", "not_same_as", "goes_with"):
            rule = "is"
        prop = _clip(prop, 60).strip()
        value = _clip(value, 160).strip()
        for need in self.needs:
            if need["item"] == item and _same(need["prop"], prop) and _same(need["value"], value) and need["rule"] == rule:
                return None
        need = {"id": len(self.needs) + 1, "item": item, "prop": prop, "rule": rule, "value": value, "value2": _clip(value2, 60),
                "src": src, "said": _clip(said, 240), "pointer": pointer, "dropped": False}
        self.needs.append(need)
        return need

    def live_needs(self, item: int | None = None) -> list[dict[str, Any]]:
        return [n for n in self.needs if not n["dropped"] and (item is None or n["item"] == item)]

    def need_text(self, need: dict[str, Any]) -> str:
        rule = {"is": "=", "is_not": "is not", "one_of": "one of", "at_least": ">=", "at_most": "<=", "between": "between",
                "mentions": "mentions", "same_as": "same as", "not_same_as": "not the same as", "goes_with": "goes with"}[need["rule"]]
        value = need["value"] + (f" and {need['value2']}" if need["rule"] == "between" and need["value2"] else "")
        return f"{need['prop'] or 'listing'} {rule} {value}".strip()


# ----------------------------------------------------------------------------------------------------------
# the interview
# ----------------------------------------------------------------------------------------------------------
# What the shopper answers: it keeps a view on a few properties ("subjects"); some hold a preference it never
# stated, the rest it does not mind about. A message is matched against those subjects: the first subject the
# message asks about is answered (its preference, or "don't mind"); a message naming none of them gets a puzzled
# reply. So a message may name several properties, best guess first, and each reply tells where to go next:
#   preference at position j -> the names before it hold nothing; the names after it are still open, ask them next
#   "don't mind"             -> some named subject is free (which one is unknown): the first name goes, the rest
#                               are asked again last-first, so a subject hidden behind the free one comes first
#   second "don't mind"      -> a subject still hidden lies between the two free ones: the middle names go first
#   puzzled                  -> none of the names is a subject: all of them are settled

PLAN = """You help a store assistant decide which questions to ask a shopper. The shopper has firm preferences on a
few product properties that the request below does not mention; the assistant can ask only a few messages, so the
properties most likely to hold such a preference must come first.

REQUEST:
{request}

ALREADY KNOWN (do not list these): {known}

For each item below you get the properties its store listings carry, with how many listings carry each and some
of their values. For each item, list up to {count} properties in the order you would ask about them: first those a
shopper buying this item for this purpose is most likely to insist on and that differ between listings; leave out
identifiers (model numbers, codes, SKUs), prices, stock, delivery, ratings and properties fixed by the request.
Use the property names exactly as given; you may add up to three properties that are not in the list when the
listing titles show them varying, each as a short property name of one or two words.
{items}
Write only JSON: {{"items": [{{"item": 1, "ask": ["property", "..."]}}]}}"""

LISTEN = """A store assistant asked a shopper about product properties; read the shopper's reply.

ITEMS: {items}
QUESTION: {question}
PROPERTIES ASKED, BY ITEM:
{asked}
REPLY: {reply}

Write only JSON:
{{"kind": "need",            // need: the reply states what it wants for a property
                             // no_preference: it does not mind / leaves the choice to the assistant
                             // confused: it did not follow the question / asks to have it again or simpler
                             // one_at_a_time: it asks for one question at a time / fewer questions
                             // other: anything else
 "item": 1,                  // number of the item the reply is about (from ITEMS), null if unclear
 "property": "",             // the property the reply is about, in the reply's own words
 "matched": "",              // the asked property (exactly as listed above) the reply answers, "" if none
 "rule": "is",               // is | is_not | one_of | at_least | at_most | between | same_as | not_same_as | goes_with
 "value": "",                // the wanted value in the reply's words ("plastic bottles", "16 gb", "3g")
 "value2": "",               // upper end for between
 "pointer": null,            // when the value is taken from another listing: {{"shop": "", "category": "", "identify": ""}}
                             // shop = whose listing, category = what kind of product it is, identify = the words
                             // that single that listing out (e.g. "size 50 g and product form gel")
 "budget": null}}            // a new money limit if the reply sets one
Reading rules: "only X will do", "anything but X is ruled out", "has to read X", "X is not up for trade", "won't budge
off X", "locked on X" all mean rule "is" with value X. "nothing other than X" is "is". "anything other than X" or
"owt other than X" means "is_not" (for a listing reference: "not_same_as"). "same as what the listing ... gives"
is "same_as"; "anything that goes with what the listing ... gives" is "goes_with". "at most / no more than / up to"
is at_most; "at least / no less than" is at_least."""


class Interview:
    def item_label(self, number: int) -> str:
        """How a question names an item: the store's own shelf name for it when its listings agree on one, else
        the request's name. A name joining several plural kinds with 'and' is one choice, so 'or' joins them."""
        item = self.items[number - 1]
        cats = [str(self.store.get(i).get("category") or "") for i in self.item_ids(number)]
        cats = [c for c in cats if c]
        name = item["name"]
        if cats:
            best = max(set(cats), key=cats.count)
            if cats.count(best) >= max(2, len(cats) // 2):
                name = best.lower()
        parts = [p for p in re.split(r"\s*(?:,|&|\band\b)\s*", name) if p.strip()]
        if len(parts) >= 2 and all(re.search(r"[a-z]{3}s$", p.strip(), re.I) for p in parts):
            name = re.sub(r"\s*(?:&|\band\b)\s*", " or ", name)
        return name

    def candidate_keys(self, number: int) -> dict[str, dict[str, Any]]:
        ids = [i for i in self.item_ids(number) if i in self.store.full]
        table = self.store.key_table(ids)
        known = {_key(n["prop"]) for n in self.live_needs(number)}
        out: dict[str, dict[str, Any]] = {}
        for k, slot in sorted(table.items(), key=lambda kv: -kv[1]["n"]):
            k = re.sub(r"^(?:[a-z]{1,2} )+", "", k)     # store prefixes ('fa pattern', 'm top neckline') are not words
            if not k or re.search(r"\b(?:id|sku|code|ean|upc|isbn|url|link|image|warranty policy|options?)\b", k):
                continue
            if k in out:
                continue
            if any(_same(k, q) or (len(_stems(q)) and _stems(q) <= _stems(k)) for q in known):
                continue
            # a key spelt twice ('pack type' / 'pack_type' fold to one; 'storage type gr' carries the same values as
            # 'storage type'): keep the plainer spelling
            twin = next((o for o in out if _stems(o) <= _stems(k) and set(out[o]["values"]) == set(slot["values"])), None)
            if twin:
                continue
            out[k] = slot
        return out

    def plan_questions(self) -> dict[int, list[str]]:
        """Per item, the properties to ask about, most likely first (the strong model orders them)."""
        blocks, cand = [], {}
        for item in self.items:
            n = item["n"]
            keys = self.candidate_keys(n)
            cand[n] = keys
            listed = "; ".join(f"{k} ({v['n']}: {', '.join(list(v['values'])[:4])})" for k, v in list(keys.items())[:30])
            titles = " | ".join(_clip(self.store.get(i).get("title"), 70) for i in self.item_ids(n)[:6])
            blocks.append(f"ITEM {n}: {item['name']} (x{item['qty']})\n  properties: {listed or '(no listing read yet)'}\n  some titles: {titles}")
        known = "; ".join(f"item {n['item']}: {self.need_text(n)}" for n in self.live_needs()) or "nothing"
        prompt = PLAN.format(request=self.request, known=known, count=14, items="\n".join(blocks))
        got = self.ask_json(prompt, [DEEP, FAST] if not self.spent(0.5) else [FAST], 2400, "low") or {}
        plan: dict[int, list[str]] = {}
        for entry in got.get("items") or []:
            if not isinstance(entry, dict):
                continue
            n = _count(entry.get("item"))
            if n is None or not 1 <= n <= len(self.items):
                continue
            names = []
            for name in entry.get("ask") or []:
                k = _key(name)
                if k and k not in names and not any(_same(k, x) for x in names):
                    names.append(k)
            plan[n] = names[:16]
        for item in self.items:
            n = item["n"]
            # the model's order first; the most carried keys it left out after it
            rest = [k for k in cand.get(n, {}) if k not in plan.get(n, [])]
            plan[n] = (plan.get(n) or []) + rest[: max(0, 16 - len(plan.get(n) or []))]
        self.trace.append({"kind": "plan", "plan": plan})
        return plan

    def question(self, pairs: list[tuple[int, str]]) -> str:
        groups: list[tuple[int, list[str]]] = []
        for number, label in pairs:
            if groups and groups[-1][0] == number:
                groups[-1][1].append(label)
            else:
                groups.append((number, [label]))

        def join(labels: list[str]) -> str:
            return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} or {labels[-1]}"
        if len(groups) == 1:
            number, labels = groups[0]
            if len(labels) == 1:
                return f"Which {labels[0]} would you like for the {self.item_label(number)}?"
            return f"What {join(labels)} would you like for the {self.item_label(number)}?"
        parts = [f"what {join(ls)} for the {self.item_label(n)}" for n, ls in groups]
        text = "; ".join(parts[:-1]) + "; and " + parts[-1] + "?"
        return "Would you tell me " + text

    def next_pairs(self, queue: list[tuple[int, str]], front: list[tuple[int, str]], room: int) -> list[tuple[int, str]]:
        settled = self.settled
        pairs: list[tuple[int, str]] = []
        for pair in front + queue:
            if len(pairs) >= room:
                break
            if pair in settled or pair in pairs:
                continue
            if any(p[0] == pair[0] and (_same(p[1], pair[1]) or _stems(p[1]) == _stems(pair[1])) for p in pairs):
                continue
            pairs.append(pair)
        # one item's names stay together, items in the order they first appear
        order = list(dict.fromkeys(n for n, _ in pairs))
        return [(n, l) for n in order for m, l in pairs if m == n]

    def global_queue(self, plan: dict[int, list[str]]) -> list[tuple[int, str]]:
        """Interleave the items' lists: rank 1 of every item, then rank 2, ...; items whose listings carry more
        varied properties come first within a rank."""
        def spread(n: int) -> int:
            return sum(1 for v in self.candidate_keys(n).values() if len(v["values"]) >= 2)
        items = sorted(plan, key=lambda n: (-spread(n), n))
        out: list[tuple[int, str]] = []
        for rank in range(max((len(v) for v in plan.values()), default=0)):
            for n in items:
                if rank < len(plan[n]):
                    out.append((n, plan[n][rank]))
        return out

    def msgs_left(self) -> int:
        limit = self.msg_limit if self.msg_limit is not None else 3
        return max(0, limit - self.msg_used - self.msg_reserve)

    # ---- choosing each message ---------------------------------------------------------------------------
    # Which properties the shopper keeps a view on, and which of those hold a preference, is unknown. Sampled
    # guesses ("worlds") start from the planner's order (earlier = more likely kept) and every reply removes the
    # worlds it rules out: a preference for property P means P was the first kept property the message named and
    # P holds; "don't mind" means the first kept one named holds nothing; a puzzled reply means none was kept. The
    # next message is built name by name to make it most likely that its first kept property holds a preference.
    # Names the shopper's reader takes for one property ('length' / 'skirt length') count as one.

    def name_groups(self, pairs: list[tuple[int, str]]) -> dict[tuple[int, str], int]:
        """Names the shopper's reader takes for one property share a group: one name is the other plus words of the
        item's own name or shelf ('length' / 'skirt length', 'material' / 'clothing material')."""
        roots: list[set[str]] = []
        owners: list[int] = []
        out: dict[tuple[int, str], int] = {}
        item_words: dict[int, set[str]] = {}
        for item in getattr(self, "items", []) or []:
            words = _stems(item["name"]) | _stems(item.get("shelf") or "")
            for c in getattr(self, "item_cats", {}).get(item["n"], set()):
                words |= _stems(c)
            item_words[item["n"]] = words
        for pair in pairs:
            st = _stems(pair[1])
            mine = item_words.get(pair[0], set())

            def same(r: set[str]) -> bool:
                if not st or not r:
                    return False
                if st == r:
                    return True
                small, big = (st, r) if len(st) < len(r) else (r, st)
                return small <= big and (big - small) <= mine
            hit = next((g for g, r in enumerate(roots) if owners[g] == pair[0] and same(r)), None)
            if hit is None:
                roots.append(set(st))
                owners.append(pair[0])
                hit = len(roots) - 1
            out[pair] = hit
        return out

    def worlds(self, pairs: list[tuple[int, str]], groups: dict[tuple[int, str], int], history: list[tuple[list[tuple[int, str]], str, int | None]],
               count: int = 0) -> list[tuple[int, int, float]]:
        """Every world of up to four kept properties among the first sixteen groups, with its prior weight,
        that agrees with every reply so far: (kept bits, held bits, weight)."""
        rank: dict[int, int] = {}
        for at, pair in enumerate(pairs):
            rank.setdefault(groups[pair], at)
        names = sorted(rank, key=lambda g: rank[g])[:16]
        weight = {g: 0.96 ** i for i, g in enumerate(names)}
        size_prior = {1: 0.10, 2: 0.35, 3: 0.35, 4: 0.20}
        msgs = [([groups[p] for p in sent if groups.get(p) in weight], kind, answered) for sent, kind, answered in history]
        out: list[tuple[int, int, float]] = []
        n = len(names)

        def check(kept: int, held: int) -> bool:
            for seq, kind, answered in msgs:
                first = next((g for g in seq if kept >> names_at[g] & 1), None)
                if kind == "need":
                    if first is None or not (held >> names_at[first] & 1) or (answered is not None and answered != first):
                        return False
                elif kind == "free":
                    if first is None or held >> names_at[first] & 1:
                        return False
                elif kind == "lost":
                    if first is not None:
                        return False
            return True
        names_at = {g: i for i, g in enumerate(names)}

        def walk(start: int, chosen: list[int]) -> None:
            if chosen:
                k = len(chosen)
                base = size_prior[k]
                for g in chosen:
                    base *= weight[names[g]]
                kept = sum(1 << g for g in chosen)
                for hmask in range(1 << k):
                    held = sum(1 << chosen[j] for j in range(k) if hmask >> j & 1)
                    if check(kept, held):
                        out.append((kept, held, base))
            if len(chosen) == 4:
                return
            for g in range(start, n):
                walk(g + 1, chosen + [g])
        walk(0, [])
        self._world_names = names_at
        return out

    def best_message(self, pairs: list[tuple[int, str]], groups: dict[tuple[int, str], int], worlds: list[tuple[int, int, float]],
                     found: set[int], asked: set[int], room: int) -> list[tuple[int, str]]:
        """Greedy: add the name that most raises the chance that the message's first kept property holds a
        preference not yet found."""
        at = getattr(self, "_world_names", {})
        cands = [p for p in pairs if groups[p] not in found and groups[p] in at]
        if not worlds:
            fresh = [p for p in cands if groups[p] not in asked]
            return list(dict.fromkeys(fresh + cands))[:room]
        total = sum(w for _, _, w in worlds) or 1.0
        found_bits = sum(1 << at[g] for g in found if g in at)
        chosen: list[tuple[int, str]] = []
        used: set[int] = set()
        # worlds where no name chosen so far is kept: only these can still be won by a later name
        open_worlds = worlds
        won = 0.0
        while len(chosen) < room:
            pick, gain = None, 0.0
            for p in cands:
                g = groups[p]
                if g in used:
                    continue
                bit = 1 << at[g]
                v = sum(w for kept, held, w in open_worlds if kept & bit and held & bit and not found_bits & bit)
                if v > gain + 1e-12:
                    pick, gain = p, v
            if pick is None:
                break
            chosen.append(pick)
            used.add(groups[pick])
            won += gain
            bit = 1 << at[groups[pick]]
            open_worlds = [x for x in open_worlds if not x[0] & bit]
        self.trace.append({"kind": "ask_odds", "p": round(won / total, 3), "worlds": len(worlds)}) if hasattr(self, "trace") else None
        return chosen

    def interview(self) -> str:
        """Ask the shopper; returns a summary for the loop model."""
        self.asked = True
        if not self.roles.get("ask") or self.msgs_left() <= 0:
            return ""
        plan = self.plan_questions()
        pairs = self.global_queue(plan)
        groups = self.name_groups(pairs)
        room = NAMES_PER_MESSAGE
        history: list[tuple[list[tuple[int, str]], str, int | None]] = []
        found: set[int] = set()
        asked: set[int] = set()
        sent: list[tuple[str, str]] = []
        told: list[str] = []
        last: list[tuple[int, str]] = []
        while self.msgs_left() > 0 and not self.finished and self.left() > 4 and self.clock() < CLOCK_FINISH - 20:
            worlds = self.worlds(pairs, groups, history)
            msg = self.best_message(pairs, groups, worlds, found, asked, room)
            if msg == last and len(msg) > 1:
                msg = msg[1:]
            if not msg:
                break
            last = list(msg)
            # one item's names together, items in the order they first appear
            order = list(dict.fromkeys(n for n, _ in msg))
            msg = sorted(msg, key=lambda p: order.index(p[0])) if len(order) > 1 else msg
            text = self.question(msg)
            answer = self.say(text)
            if answer is None:
                break
            sent.append((text, answer))
            asked.update(groups[p] for p in msg)
            got = self.listen(msg, text, answer)
            kind = str(got.get("kind") or "other")
            self.trace.append({"kind": "ask_state", "worlds": len(worlds), "msg": [l for _, l in msg], "reply": kind})
            if kind == "one_at_a_time" and room > 1 and len(msg) > 1:
                room = 1
                continue
            if kind == "need":
                at = self.answered_at(msg, got)
                owner = msg[at][0] if at >= 0 else self.owner_of(got, msg)
                need = self.take_reply_need(owner, got, msg[at][1] if at >= 0 else "", answer)
                if need:
                    told.append(f"item {owner}: {self.need_text(need)}" if len(self.items) > 1 else self.need_text(need))
                g = groups[msg[at]] if at >= 0 else None
                if g is not None:
                    found.add(g)
                history.append((msg, "need", g))
            elif kind == "no_preference":
                history.append((msg, "free", None))
            elif kind == "confused":
                history.append((msg, "lost", None))
            budget = _number(got.get("budget"))
            if budget and budget > 0:
                self.budget = budget
                told.append(f"budget now {budget:g}")
        if not sent:
            return ""
        lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in sent)
        summary = "[interview] Astrolabe asked the shopper (each message used one step):\n" + lines
        summary += "\nWhat the answers add to the requirements: " + ("; ".join(told) if told else "none") + "."
        return summary

    def say(self, text: str) -> str | None:
        """Send one message to the shopper; returns its reply text."""
        tool = self.roles.get("ask")
        if not tool:
            return None
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        arg = next((k for k, v in props.items() if "string" in _schema_kinds(v)), None) or next(iter(props), "message")
        reply = self.act(tool, _fit_args({arg: text}, self.schemas.get(tool)))
        if reply is None:
            return None
        self.msg_used += 1
        obs = reply.get("observation")
        body, notes = _envelope(obs)
        self.take_notes(notes)
        if isinstance(body, dict):
            words = [v for v in body.values() if isinstance(v, str)]
            answer = " ".join(words) if words else _dump(body)
        else:
            answer = str(body)
        self.dialogue.append({"q": text, "a": answer})
        self.trace.append({"kind": "ask", "q": text, "a": _clip(answer, 300)})
        return answer

    def listen(self, pairs: list[tuple[int, str]], question: str, answer: str) -> dict[str, Any]:
        groups: dict[int, list[str]] = {}
        for n, label in pairs:
            groups.setdefault(n, []).append(label)
        asked = "\n".join(f"item {n}: " + "; ".join(ls) for n, ls in groups.items())
        items = "; ".join(f"{i['n']} = {i['name']} ({self.item_label(i['n'])})" for i in self.items)
        prompt = LISTEN.format(items=items, question=question, asked=asked, reply=answer)
        got = self.ask_json(prompt, [FAST, BACKUP], 1800, "low") or {}
        self.trace.append({"kind": "heard", "got": _clip(_dump(got), 400)})
        return got

    def answered_at(self, pairs: list[tuple[int, str]], got: dict[str, Any]) -> int:
        """Where in the message the answered property sits; -1 when it matches no asked name."""
        item = _count(got.get("item"))
        many = len({n for n, _ in pairs}) > 1
        for field in ("matched", "property"):
            said = _key(got.get(field) or "")
            if not said:
                continue
            for at, (n, label) in enumerate(pairs):
                if (not many or item is None or item == n) and _same(label, said):
                    return at
            for at, (n, label) in enumerate(pairs):
                if (not many or item is None or item == n) and (_stems(label) <= _stems(said) or _stems(said) <= _stems(label)):
                    return at
        return -1

    def owner_of(self, got: dict[str, Any], pairs: list[tuple[int, str]]) -> int:
        item = _count(got.get("item"))
        if item and 1 <= item <= len(self.items):
            return item
        return pairs[0][0] if pairs else 1

    def take_reply_need(self, owner: int, got: dict[str, Any], asked: str, answer: str) -> dict[str, Any] | None:
        prop = str(asked or got.get("property") or "").strip()     # the asked store name matches the listings best
        value = str(got.get("value") or "").strip()
        rule = str(got.get("rule") or "is")
        pointer = got.get("pointer") if isinstance(got.get("pointer"), dict) and any(got["pointer"].values()) else None
        if not value and not pointer:
            return None
        if pointer:
            # the relation to the pointed listing, from the reply's own words
            low = answer.lower()
            if re.search(r"goes? (?:together )?with|pairs? with|matches? with", low):
                rule = "goes_with"
            elif re.search(r"other than|anything but|not the same|different from", low) and not re.search(r"rules? out anything but", low):
                rule = "not_same_as"
            elif re.search(r"\btops\b|nowt higher|no higher|at most|no more than|or less\b|up to", low):
                rule = "at_most"
            elif re.search(r"nowt lower|no lower|at least|no less than|or more\b", low):
                rule = "at_least"
            elif rule in ("is", "one_of", "") or re.search(r"same as|spot on", low):
                rule = "same_as"
        need = self.add_need(owner, prop or asked, rule, value or "(see the listing it points to)", str(got.get("value2") or ""),
                             "shopper", answer, pointer)
        if need:
            self.found_on[owner] = self.found_on.get(owner, 0) + 1
            if pointer:
                self.pointer_queue.append(need)
        return need


# ----------------------------------------------------------------------------------------------------------
# which item a listing is for, and whether it meets each requirement
# ----------------------------------------------------------------------------------------------------------

CHECK = """Decide, for one store listing, whether it meets each requirement. Use only what the listing shows; a
requirement the listing says nothing about is "unknown". Values that mean the same thing count as a match
("stainless" / "stainless steel", "2 pack" / "pack of 2"); a different value is "no".

LISTING:
{page}

REQUIREMENTS:
{needs}

Write only JSON: {{"verdicts": [{{"id": 1, "verdict": "yes|no|unknown", "why": "a few words"}}]}}"""


VALUES = """A shopper's requirement for a product: {need}
Store listings state this property with the values below. For each value, does a listing with that value meet
the requirement? Only the same value counts: other spelling, plural, word order, an abbreviation or a list that
contains it ("polyester" in "cotton, polyester"). A different or merely similar value does not ("loose" is not
"baggy", "long" is not "full length"). Answer "unclear" only when the value cannot tell.
VALUES:
{values}
Write only JSON: {{"values": [{{"n": 1, "meets": "yes|no|unclear"}}]}}"""


def _unit(u: str) -> str:
    u = u.lower().strip(" .")
    return u[:-1] if len(u) > 2 and u.endswith("s") else u


def _range_of(text: str) -> list[tuple[float, str]]:
    return [(v, _unit(u)) for v, u in _quantities(text)]


class Checks:
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        x = _stems(a) - {"and", "or", "the", "for", "with"}
        y = _stems(b) - {"and", "or", "the", "for", "with"}
        return len(x & y) / len(x | y) if x and y else 0.0

    def item_shelves(self, number: int) -> set[str]:
        """The store categories an item belongs to: those whose name best matches the item's name and shelf (at
        least half their words shared); else the category most of the item's own search rows came from."""
        cats = {str(r.get("category")) for r in self.store.rows.values() if r.get("category")}
        memo = self.shelf_memo.get(number)
        if memo and memo[0] == len(cats):
            return memo[1]
        item = self.items[number - 1]
        name = f"{item['name']} {item.get('shelf') or ''}"
        scored = {c: max(self._jaccard(name, c), self._jaccard(item["name"], c), self._jaccard(item.get("shelf") or "", c)) for c in cats}
        best = max(scored.values(), default=0.0)
        if best >= 0.5:
            out = {c for c, v in scored.items() if v >= best - 1e-9}
        else:
            rows = [i for i, n in self.search_for.items() if n == number]
            counts: dict[str, int] = {}
            for i in rows:
                c = str(self.store.get(i).get("category") or "")
                if c:
                    counts[c] = counts.get(c, 0) + 1
            out = {max(counts, key=counts.get)} if counts else set()
        self.shelf_memo[number] = (len(cats), out)
        return out

    def item_of(self, ident: str) -> int | None:
        """The requested item a listing is for: its category against each item's categories; a listing without a
        category goes by its title words."""
        r = self.store.get(ident)
        cat = str(r.get("category") or "")
        if cat:
            owners = [item["n"] for item in self.items if cat in self.item_shelves(item["n"])]
            if len(owners) == 1:
                return owners[0]
            if len(owners) > 1:
                title = str(r.get("title") or "")
                return max(owners, key=lambda n: _overlap(title, self.items[n - 1]["name"]))
            if not any(self.item_shelves(item["n"]) for item in self.items):
                return 1 if len(self.items) == 1 else self.search_for.get(ident)
            return None
        if len(self.items) == 1:
            return 1
        title = str(r.get("title") or "")
        best = max(self.items, key=lambda it: _overlap(title, it["name"]))
        return best["n"] if _overlap(title, best["name"]) >= 0.5 else self.search_for.get(ident)

    def item_ids(self, number: int, full: bool = False) -> list[str]:
        return [i for i in self.store.order if self.item_of(i) == number and (not full or i in self.store.full)]

    def values_for(self, ident: str, prop: str) -> list[tuple[str, str]]:
        """The listing's (key, value) pairs that state a property: the exact key first, then keys sharing its
        words ('frame material' for 'material'), widest match last."""
        facts = self.store.facts(ident)
        want = _key(prop)
        if not want:
            return []
        exact = [(k, v) for k, v in facts.items() if k == want]
        ws = _stems(want)
        near = [(k, v) for k, v in facts.items() if k != want and ws and (ws <= _stems(k) or (_stems(k) <= ws and len(_stems(k)) >= 1 and len(ws) <= 3))]
        # every key naming the property counts ('style' and 'skirt style'): the store's grader judges the one whose
        # values fit the wanted value
        return exact + near

    def code_verdict(self, ident: str, need: dict[str, Any]) -> str:
        """yes / no / unknown for one requirement, from the listing's own words and numbers."""
        rule = need["rule"]
        value = need.get("resolved") if rule in ("same_as", "not_same_as", "goes_with") else need["value"]
        if rule in ("same_as", "not_same_as", "goes_with"):
            if not value:
                return "unknown"
            rule = {"same_as": "is", "not_same_as": "is_not", "goes_with": "goes_with"}[rule]
            if rule == "goes_with":
                return "unknown"
        if not value:
            return "unknown"
        cat = str(self.store.get(ident).get("category") or "")
        if rule in ("is", "mentions") and cat and _stems(value) and _stems(value) <= _stems(cat):
            return "yes"        # the listing's own shelf names the wanted value ('ready to drink' milk on that shelf)
        pairs = self.values_for(ident, need["prop"])
        text = self.store.text_of(ident)
        if rule == "mentions" or not pairs:
            if rule in ("is", "mentions", "one_of") and value and _stems(value) and _stems(value) <= _stems(self.store.get(ident).get("title") or ""):
                return "yes"
            if rule == "is_not" and not pairs:
                return "unknown" if _stems(value) <= _stems(text) else "yes"
            return "unknown"
        vals = [v for _, v in pairs]
        if rule in ("is", "one_of", "is_not"):
            options = [value] if rule != "one_of" else [o for o in re.split(r"\s*(?:,|/|\bor\b)\s*", value) if o.strip()]
            decider = self.deciding_key(ident, need, options) if len(pairs) > 1 else ""
            if decider and any(k == decider for k, _ in pairs):
                vals = [v for k, v in pairs if k == decider]
            judged = [self.value_judged.get((need["id"], _key(v))) for v in vals]
            hit = any(self.value_match(v, o) for v in vals for o in options) or any(j == "yes" for j in judged)
            # a firm miss: every stated value is judged another one (by the page reader, else plainly by its words)
            miss = not hit and all(j == "no" or (j is None and all(self.value_differs(v, o) for o in options))
                                   for v, j in zip(vals, judged))
            if rule == "is_not":
                return "no" if hit else "yes" if miss else "unknown"
            return "yes" if hit else "no" if miss else "unknown"
        if rule in ("at_most", "at_least", "between"):
            lo_hi = []
            want = _range_of(value)
            if not want:
                return "unknown"
            wv, wu = want[0]
            top = None
            if rule == "between":
                second = _range_of(need.get("value2") or "") or (want[1:2] if len(want) > 1 else [])
                top = second[0] if second else None
            for v in vals:
                nums = _range_of(v)
                same = [x for x, u in nums if u == wu or not wu or not u]
                if same:
                    lo_hi.append((min(same), max(same)))
            if not lo_hi:
                return "unknown"
            ok = []
            for lo, hi in lo_hi:
                if rule == "at_most":
                    ok.append(True if hi <= wv + 1e-9 else False if lo > wv + 1e-9 else None)
                elif rule == "at_least":
                    ok.append(True if lo >= wv - 1e-9 else False if hi < wv - 1e-9 else None)
                else:
                    tv = top[0] if top else None
                    if tv is None:
                        ok.append(None)
                    else:
                        ok.append(True if lo >= wv - 1e-9 and hi <= tv + 1e-9 else False if hi < wv or lo > tv else None)
            if any(x is True for x in ok):
                return "yes"
            if all(x is False for x in ok):
                return "no"
            return "unknown"
        return "unknown"

    def deciding_key(self, ident: str, need: dict[str, Any], options: list[str]) -> str:
        """When several keys name a property ('material' / 'clothing material', 'style' / 'skirt style'), the store
        judges one of them: the key that carries the wanted value on the most opened listings of the item; the
        plain key keeps it unless another carries the value on at least two more listings (or the plain key on none)."""
        number = self.item_of(ident)
        if number is None:
            return ""
        memo_key = (number, need["id"], len(self.store.full))
        if memo_key in self.decider_memo:
            return self.decider_memo[memo_key]
        hits: dict[str, int] = {}
        for other in self.item_ids(number, full=True):
            for k, v in self.values_for(other, need["prop"]):
                if any(self.value_match(v, o) for o in options) or self.value_judged.get((need["id"], _key(v))) == "yes":
                    hits[k] = hits.get(k, 0) + 1
                else:
                    hits.setdefault(k, 0)
        plain = _key(need["prop"])
        pick = ""
        if hits:
            best = max(hits, key=lambda k: (hits[k], k == plain))
            if plain in hits and hits[plain] >= 1 and hits[best] < hits[plain] + 2:
                best = plain
            pick = best if hits[best] >= 1 else ""
        self.decider_memo[memo_key] = pick
        return pick

    @staticmethod
    def value_match(have: str, want: str) -> bool:
        """A stated value carries the wanted one: same words (any order), same numbers with the same unit."""
        h, w = _key(have), _key(want)
        if not h or not w:
            return False
        if h == w:
            return True
        wn, hn = _range_of(want), _range_of(have)
        if wn:
            for x, u in wn:
                if not any(abs(x - y) < 1e-9 and (u == v or not u or not v) for y, v in hn):
                    return False
            rest = {w for w in _stems(re.sub(r"\d+(?:[.,]\d+)?", " ", want)) if len(w) > 2} - {u for _, u in wn}
            return rest <= (_stems(have) | {u for _, u in hn})
        ws, hs = _stems(want), _stems(have)
        if ws and ws <= hs:
            # a list value ('blueberry, strawberry') must carry every wanted word in one of its parts
            parts = [p for p in re.split(r"\s*(?:,|;|/|\||\band\b)\s*", have) if p.strip()]
            return len(parts) <= 1 or len(ws) == 1 or any(ws <= _stems(p) for p in parts)
        return False

    @staticmethod
    def value_differs(have: str, want: str) -> bool:
        """A stated value plainly is another one: other numbers, or each side has words the other lacks."""
        if not _key(have) or Checks.value_match(have, want):
            return False
        wn, hn = _range_of(want), _range_of(have)
        if wn and hn:
            return True
        ws = _stems(want)
        parts = [x for x in re.split(r"\s*(?:,|;|/|\|)\s*", have) if x.strip()] or [have]
        if any(ws & _stems(x) for x in parts):
            # a value sharing a word with the wanted one ('acne' / 'acne care', 'glass bottles' / 'plastic bottles')
            # is for the page reader to judge
            return False
        return True

    def judge_values(self, number: int, need: dict[str, Any]) -> None:
        """The fast model judges, once per new store value, whether it meets a word requirement."""
        if need["rule"] not in ("is", "one_of", "is_not") or not need["value"] or self.spent(0.85) or self.calls >= self.call_cap - 8:
            return
        seen: list[str] = []
        for ident in self.item_ids(number, full=True):
            for _, v in self.values_for(ident, need["prop"]):
                k = _key(v)
                if k and (need["id"], k) not in self.value_judged and v not in seen and len(seen) < 40:
                    seen.append(v)
        fresh = [v for v in seen if not any(self.value_match(v, o) for o in [need["value"]])]
        for v in seen:
            if v not in fresh:
                self.value_judged[(need["id"], _key(v))] = "yes" if need["rule"] != "is_not" else "yes"
        if not fresh:
            return
        self.value_asked.add(need["id"])
        want = self.need_text(need) if need["rule"] != "is_not" else f"{need['prop']} = {need['value']}"
        lines = "\n".join(f"{i + 1}. {_clip(v, 80)}" for i, v in enumerate(fresh))
        got = self.ask_json(VALUES.format(need=want, values=lines), [FAST, BACKUP], 1200, "low") or {}
        for entry in got.get("values") or []:
            if not isinstance(entry, dict):
                continue
            i = _count(entry.get("n"))
            m = str(entry.get("meets") or "").lower()
            if i and 1 <= i <= len(fresh) and m in ("yes", "no", "unclear"):
                self.value_judged[(need["id"], _key(fresh[i - 1]))] = m

    def verdicts(self, ident: str, needs: list[dict[str, Any]], use_model: bool = True) -> dict[int, str]:
        """Per requirement id: yes / no / unknown. Code first; the fast model reads the page for what code cannot
        decide (once per listing and requirement)."""
        out: dict[int, str] = {}
        ask: list[dict[str, Any]] = []
        number = self.item_of(ident)
        if number is not None:
            for need in needs:
                if self.values_for(ident, need["prop"]) and any((need["id"], _key(v)) not in self.value_judged for _, v in self.values_for(ident, need["prop"])):
                    self.judge_values(number, need)
        for need in needs:
            # code verdicts are recomputed (the deciding key may change as more listings are read); the page
            # reader's verdicts are kept
            v = self.code_verdict(ident, need)
            memo = self.verdict_memo.get((ident, need["id"]))
            if v == "unknown" and memo:
                out[need["id"]] = memo
                continue
            out[need["id"]] = v
            if v != "unknown":
                continue
            elif ident in self.store.full and need["rule"] != "goes_with" or (need["rule"] == "goes_with" and need.get("resolved")):
                ask.append(need)
        if ask and use_model and not self.spent(0.9) and self.calls < self.call_cap - 8:
            lines = "\n".join(f"{n['id']}. {self.need_text(n) if n['rule'] not in ('same_as', 'not_same_as', 'goes_with') else self.pointer_text(n)}"
                              for n in ask)
            got = self.ask_json(CHECK.format(page=self.store.page_text(ident, 2400), needs=lines), [FAST, BACKUP], 1500, "low") or {}
            for entry in got.get("verdicts") or []:
                if not isinstance(entry, dict):
                    continue
                nid = _count(entry.get("id"))
                verdict = str(entry.get("verdict") or "").lower()
                if nid in {n["id"] for n in ask} and verdict in ("yes", "no", "unknown"):
                    out[nid] = verdict
                    self.verdict_memo[(ident, nid)] = verdict
        return out

    def pointer_text(self, need: dict[str, Any]) -> str:
        rel = {"same_as": "the same as", "not_same_as": "different from", "goes_with": "goes with"}.get(need["rule"], "")
        if need.get("resolved"):
            return f"{need['prop']} {rel} {need['resolved']} (the value the pointed listing gives)"
        return f"{need['prop']} {rel} the value of the listing it points to (not yet found)"

    def money_ok(self, ident: str, number: int) -> bool:
        if self.budget is None:
            return True
        price = self.store.price(ident)
        if price is None:
            return True
        qty = self.items[number - 1]["qty"]
        return price * (1 if self.budget_each else qty) <= self.budget + 1e-9

    def fits_for(self, number: int, use_model: bool = False) -> list[tuple[str, str]]:
        """Per item, opened listings meeting every live requirement, best first: [(id, 'fits'|'likely')]."""
        needs = self.live_needs(number)
        out: list[tuple[str, str, float]] = []
        stated = {x["id"]: [i for i in self.item_ids(number, full=True) if self.values_for(i, x["prop"])] for x in needs}
        for ident in self.item_ids(number, full=True):
            if not self.store.in_stock(ident) or not self.money_ok(ident, number):
                continue
            if self.code_fault(ident, number):
                continue
            vs = self.verdicts(ident, needs, use_model)
            if any(v == "no" for v in vs.values()):
                continue
            # a requirement this page does not state while other listings of the item do: not a fit
            if any(vs.get(x["id"]) == "unknown" and x["rule"] in ("is", "one_of") and not self.values_for(ident, x["prop"])
                   and any(i != ident for i in stated[x["id"]]) for x in needs):
                continue
            unknown = sum(1 for v in vs.values() if v == "unknown")
            out.append((ident, "fits" if not unknown else "likely", float(unknown)))
        out.sort(key=lambda x: (x[2] > 0, self.rank_key(x[0])))
        return [(i, tag) for i, tag, _ in out]

    def rank_key(self, ident: str) -> tuple[float, ...]:
        rule = self.rank.get("rule")
        price = self.store.price(ident)
        p = price if price is not None else 1e12
        if rule in ("most", "least") and self.rank.get("property"):
            nums = [x for _, v in self.values_for(ident, self.rank["property"]) for x, _ in _range_of(v)]
            if nums:
                val = max(nums) if rule == "most" else min(nums)
                return ((-val if rule == "most" else val), p)
            return (1e12, p)
        if rule == "dearest":
            return (-p,)
        return (p,)


# ----------------------------------------------------------------------------------------------------------
# model codes: which part fits which device
# ----------------------------------------------------------------------------------------------------------
# A device names its own code ("Model number: AB-1230. Also sold as Name 123."); a part names what it fits
# ("Fits: AB-1230, Name 456" / "Fits: every AB12xx model, except AB1230" / "(part CD4567)" / "More on part CD4567").
# Codes are compared without hyphens or case; a family "AB12xx" covers every code that starts "AB12".

_CODE = re.compile(r"\b[A-Z]{2,5}-?\d{2,6}[A-Z]?\b")
_FAMILY = re.compile(r"\b([A-Z]{2,5})-?(\d{0,5})(?:x{1,4}|X{2,4})\b")
_NAME_CODE = re.compile(r"\b[A-Z][a-z]{2,11} \d{2,5}\b")


def _code_key(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+(?=[A-Z(])|\n+", text or "") if s.strip()]


def _code_facts(text: str) -> dict[str, Any]:
    """What a listing text says about codes: own codes, other names it is sold under, what it fits (codes, names,
    families, exceptions) and the part it sends the reader to."""
    own: set[str] = set()
    names: set[str] = set()
    fits: set[str] = set()
    fit_names: set[str] = set()
    families: set[str] = set()
    excepts: set[str] = set()
    more: set[str] = set()
    # "Part AB1234 also fits: ..." speaks for another part: kept aside, it counts only for that part's own listing
    others: dict[str, str] = {}
    def aside(found: re.Match) -> str:
        others[_code_key(found.group(1))] = found.group(2)
        return " "
    text = re.sub(r"\b(?:Part|Model|Item)\s+([A-Z]{2,5}-?\d{2,6}[A-Z]?)\s+(?:also|too)\s+fits:?\s*([^.]*)\.?", aside, text or "")
    clauses: list[str] = []
    for sent in _sentences(text):
        # a code statement may run on from a title without a full stop: split before each statement word
        clauses += [c for c in re.split(r"(?=\b(?:Fits|Compatible with|Works with|Model number|Model no|Part number|Also sold as|More on part)\b)", sent) if c.strip()]
    for sent in clauses:
        low = sent.lower()
        if re.search(r"\b(?:model|part|item|product) (?:number|no\.?|code|ref\.?)\s*[:#]", low) or re.match(r"(?:model|part)\s*:", low):
            own.update(_code_key(c) for c in _CODE.findall(sent))
            if re.search(r"also sold as|also known as|sold as", low):
                tail = re.split(r"(?i)also sold as|also known as|sold as", sent, 1)[-1]
                names.update(_NAME_CODE.findall(tail))
                own.update(_code_key(c) for c in _CODE.findall(tail))
            continue
        if re.search(r"also sold as|also known as", low):
            tail = re.split(r"(?i)also sold as|also known as", sent, 1)[-1]
            names.update(_NAME_CODE.findall(tail))
            continue
        if re.search(r"\b(?:fits|compatible with|works with|goes with|for use with)\b\s*:?", low):
            split = re.split(r"(?i)\b(?:except|excluding|but not)\b", sent, maxsplit=1)
            head, rest = split[0], (split[1] if len(split) > 1 else "")
            if rest:
                excepts.update(_code_key(c) for c in _CODE.findall(rest))
            for prefix, digits in _FAMILY.findall(head):
                families.add(_code_key(prefix + digits))
            body = _FAMILY.sub(" ", head)
            fits.update(_code_key(c) for c in _CODE.findall(body))
            fit_names.update(_NAME_CODE.findall(body))
            continue
        if re.search(r"\bmore on (?:part|model|item)\b", low):
            more.update(_code_key(c) for c in _CODE.findall(sent))
            continue
    for found in re.finditer(r"\((?:part|model|ref\.?|code)\s+([A-Z]{2,5}-?\d{2,6}[A-Z]?)\)", text or ""):
        own.add(_code_key(found.group(1)))
    for code, body in others.items():
        if code in own:
            more_facts = _code_facts("Fits: " + body + ".")
            fits |= more_facts["fits"]
            fit_names |= more_facts["fit_names"]
            families |= more_facts["families"]
    return {"own": own, "names": names, "fits": fits, "fit_names": fit_names, "families": families,
            "except": excepts, "more": more}


class Codes:
    def codes_of(self, ident: str) -> dict[str, Any]:
        memo = self.code_memo.get(ident)
        text = self.store.text_of(ident)
        if memo and memo[0] == len(text):
            return memo[1]
        r = self.store.get(ident)
        got: dict[str, Any] = {}
        for field in ("title", "about", "notes"):
            part = _code_facts(str(r.get(field) or "")) if r.get(field) else None
            if part:
                for k, v in part.items():
                    got.setdefault(k, set()).update(v)
        if not got:
            got = _code_facts("")
        self.code_memo[ident] = (len(text), got)
        return got

    def says_fits(self, ident: str) -> bool:
        c = self.codes_of(ident)
        return bool(c["fits"] or c["fit_names"] or c["families"])

    def fit(self, part: str, device: str) -> bool | None:
        """Does the part fit the device by their codes? None when either side names no code."""
        p, d = self.codes_of(part), self.codes_of(device)
        if not (p["fits"] or p["fit_names"] or p["families"]):
            return None
        own = d["own"]
        names = {_key(n) for n in d["names"]}
        if not own and not names:
            return None
        if own & p["except"]:
            return False
        if own & p["fits"]:
            return True
        if any(o.startswith(f) for o in own for f in p["families"] if f):
            return True
        if names & {_key(n) for n in p["fit_names"]}:
            return True
        return False

    def fit_code(self, part: str, code: str) -> bool | None:
        """Does a part fit a model the request names (a code or a 'Name 123' name)?"""
        p = self.codes_of(part)
        if not (p["fits"] or p["fit_names"] or p["families"]):
            return None
        k = _code_key(code)
        if k in p["except"]:
            return False
        if k in p["fits"] or any(k.startswith(f) for f in p["families"] if f):
            return True
        if _key(code) in {_key(n) for n in p["fit_names"]}:
            return True
        return False

    def fit_pairs(self) -> list[tuple[int, int]]:
        """(part item, device item) pairs: the request says one item must fit another, or the store's listings show
        it (one item's listings name their model code, another's name what they fit)."""
        pairs: list[tuple[int, int]] = []
        for item in self.items:
            target = item.get("fits") or ""
            if not target:
                continue
            for other in self.items:
                if other["n"] != item["n"] and (_overlap(target, other["name"]) >= 0.5 or re.search(rf"\bitem {other['n']}\b", target.lower())):
                    pairs.append((item["n"], other["n"]))
        if len(self.items) > 1:
            for a in self.items:
                for b in self.items:
                    if a["n"] == b["n"] or (a["n"], b["n"]) in pairs or (b["n"], a["n"]) in pairs:
                        continue
                    parts = [i for i in self.item_ids(a["n"]) if self.says_fits(i)]
                    devices = [i for i in self.item_ids(b["n"]) if self.codes_of(i)["own"] and not self.says_fits(i)]
                    if len(parts) >= 2 and len(devices) >= 2 and len(parts) >= len(self.item_ids(a["n"])) // 3:
                        pairs.append((a["n"], b["n"]))
        return pairs

    def stated_codes(self, number: int) -> list[str]:
        """Model codes or names the request says an item must fit."""
        target = self.items[number - 1].get("fits") or ""
        out = [c for c in _CODE.findall(target)] + _NAME_CODE.findall(target)
        return out

    def code_fault(self, ident: str, number: int) -> str:
        """Why a listing cannot be bought for an item by codes: it does not fit a model the request names, or not
        the device already in the cart."""
        for code in self.stated_codes(number):
            if self.fit_code(ident, code) is False:
                return f"does not fit {code}"
        for part_item, device_item in self.fit_pairs():
            if part_item == number:
                device = self.cart_line_for(device_item)
                if device and self.fit(ident, device) is False:
                    return f"does not fit the {self.items[device_item - 1]['name']} in the cart ({device})"
            elif device_item == number:
                part = self.cart_line_for(part_item)
                if part and self.fit(part, ident) is False:
                    return f"the {self.items[part_item - 1]['name']} in the cart ({part}) does not fit it"
        return ""

    def best_basket(self) -> dict[int, str]:
        """Best known pick per item when items must fit each other: the device that the most top parts fit."""
        pairs = self.fit_pairs()
        picks: dict[int, str] = {}
        fits = {item["n"]: [i for i, _ in self.fits_for(item["n"])] for item in self.items}
        if not pairs:
            return {n: v[0] for n, v in fits.items() if v}
        devices = {d for _, d in pairs}
        for d in devices:
            best, score = None, -1
            for dev in fits.get(d, [])[:12]:
                s = sum(1 for p, dd in pairs if dd == d and any(self.fit(part, dev) is not False for part in fits.get(p, [])))
                if s > score:
                    best, score = dev, s
            if best:
                picks[d] = best
        for p, d in pairs:
            dev = picks.get(d)
            choice = next((part for part in fits.get(p, []) if dev is None or self.fit(part, dev) is True), None)
            choice = choice or next((part for part in fits.get(p, []) if dev is None or self.fit(part, dev) is not False), None)
            if choice:
                picks[p] = choice
        for n, v in fits.items():
            if n not in picks and v:
                picks[n] = v[0]
        return picks


# ----------------------------------------------------------------------------------------------------------
# the cart, store notes, and ending the episode
# ----------------------------------------------------------------------------------------------------------

NOTE = """A store assistant got this note from the shop or the shopper while shopping:
NOTE: {note}
CART: {cart}
REQUIREMENTS: {needs}
Write only JSON: {{"kind": "sold_out|dropped|changed|other",
 "listing": "",        // the listing id or title the note is about, if any
 "need_ids": [],       // ids of the requirements the shopper drops or changes (from REQUIREMENTS)
 "new": [{{"item": 1, "property": "", "rule": "is", "value": ""}}],   // a requirement the note adds or the new form of a changed one
 "budget": null}}      // a new money limit if the note sets one"""


REVIEW = """You check a store order before it is placed.

SHOPPER'S REQUEST:
{request}

REQUIREMENTS GATHERED (from the request and the shopper's answers; dropped ones removed):
{needs}

ORDER (one line per requested item):
{lines}

OTHER OPENED LISTINGS THAT MAY FIT:
{others}

For each order line: does the listing satisfy every requirement for its item and is it the kind of product the
shopper asked for? If not, or if one of the other listings fits and is better by the shopper's rule ({rule}), name
it. Judge only from the listing texts shown. Write only JSON:
{{"lines": [{{"item": 1, "ok": true, "replace_with": "", "why": "a few words"}}]}}"""


def _arg_id(args: dict[str, Any], props: dict[str, Any]) -> str | None:
    """The listing id among a call's arguments: an id-named argument that is not an option list, else a value
    shaped like a listing id."""
    for k, v in args.items():
        if isinstance(v, str) and v.strip() and not _enum_of(props.get(k)) and re.search(_ID_ARG, _key(k)) and _LIST_ID.match(v.strip()):
            return v.strip()
    return next((v.strip() for v in args.values() if isinstance(v, str) and _ID_VALUE.match(v.strip())), None)


class Cart:
    # ---- what is in the cart --------------------------------------------------------------------------------
    def cart_ids(self) -> list[str]:
        return [i for i, q in self.cart.items() if q > 0]

    def cart_line_for(self, number: int) -> str | None:
        return next((i for i in self.cart_ids() if self.item_of(i) == number), None)

    def note_cart_call(self, name: str, args: dict[str, Any], obs: Any) -> None:
        """Keep the cart as the agent's own calls change it (a view of the cart overrides this)."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        lists = [v for v in args.values() if isinstance(v, list)]
        if lists:
            new: dict[str, int] = {}
            for line in lists[0]:
                if isinstance(line, dict):
                    ident = _arg_id(line, {})
                    qty = next((_count(v) for k, v in line.items() if _count(v) is not None and re.search(_QTY_ARG, _key(k))), 1)
                    if ident:
                        new[ident] = (qty if qty is not None else 1)
                elif isinstance(line, str) and _LIST_ID.match(line):
                    new[line] = 1
            if re.search(r"set|new|replace|selection|whole", _key(name) + " " + " ".join(_key(k) for k in args)):
                self.cart = new
            else:
                for i, q in new.items():
                    self.cart[i] = self.cart.get(i, 0) + q
            return
        ident = _arg_id(args, props)
        if not ident:
            return
        qty = next((_count(v) for k, v in args.items() if _count(v) is not None and re.search(_QTY_ARG, _key(k))), 1)
        qty = 1 if qty is None else qty
        action = " ".join(str(v).lower() for k, v in args.items() if k in props and _enum_of(props[k]))
        if re.search(r"remove|drop|discard|delete|minus|take|clear|less", action):
            self.cart[ident] = max(0, self.cart.get(ident, 0) - qty) if not re.search(r"remove|delete|clear|discard|drop", action) else 0
        elif re.search(r"set|exact|update", action):
            self.cart[ident] = qty
        else:
            self.cart[ident] = self.cart.get(ident, 0) + qty

    def read_cart_view(self, obs: Any) -> None:
        body, _ = _envelope(obs)
        recs = _records(body)
        if not isinstance(body, (dict, list)):
            return
        new: dict[str, int] = {}
        for rec in recs:
            qty = None
            for k, v in (rec.get("extra") or {}).items():
                if re.search(r"\b(?:qty|quantity|amount|units|count|held|total)\b", k) and _count(v) is not None and not re.search(r"price|cost|due", k):
                    qty = _count(v)
                    break
            new[rec["id"]] = qty if qty is not None else 1
            self.store.add({k: v for k, v in rec.items() if k in ("id", "title", "price")} | {"facts": {}, "extra": {}}, False)
        self.cart = new
        self.cart_seen = self.step

    def view_cart(self) -> None:
        tool = self.roles.get("cart_view")
        if tool and self.left() > 2:
            reply = self.act(tool, {})
            if reply:
                self.absorb(tool, {}, reply.get("observation"))

    def set_cart(self, picks: dict[int, str]) -> bool:
        """Make the cart hold exactly one line per item (the given listing, the item's quantity)."""
        tool = self.roles.get("cart")
        if not tool:
            return False
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        want = {ident: self.items[n - 1]["qty"] for n, ident in picks.items()}
        list_arg = next((k for k, v in props.items() if "array" in _schema_kinds(v)), None)
        if list_arg:
            spec = props[list_arg].get("items") if isinstance(props[list_arg], dict) else {}
            inner = (spec or {}).get("properties") or {}
            id_key = next((k for k, v in inner.items() if "string" in _schema_kinds(v) and re.search(_ID_ARG, _key(k))), None) \
                or next((k for k, v in inner.items() if "string" in _schema_kinds(v)), None)
            qty_key = next((k for k, v in inner.items() if {"integer", "number"} & _schema_kinds(v)), None)
            if id_key:
                lines = [{id_key: i, **({qty_key: q} if qty_key else {})} for i, q in want.items()]
                reply = self.act(tool, {list_arg: lines})
                if reply is None:
                    return False
                self.absorb(tool, {list_arg: lines}, reply.get("observation"))
                return True
            return False
        id_key = next((k for k, v in props.items() if "string" in _schema_kinds(v) and not _enum_of(v) and re.search(_ID_ARG, _key(k))), None)
        qty_key = next((k for k, v in props.items() if {"integer", "number"} & _schema_kinds(v)), None)
        act_key = next((k for k, v in props.items() if _enum_of(v)), None)
        if not id_key:
            return False
        opts = _enum_of(props[act_key]) if act_key else []
        add = next((o for o in opts if re.search(r"add|include|append|plus|put|insert|set", o.lower())), opts[0] if opts else None)
        drop = next((o for o in opts if re.search(r"remove|discard|drop|delete|minus|take", o.lower())), None)
        drops = [i for i in self.cart_ids() if not (i in want and self.cart.get(i, 0) == want[i])]
        adds = [i for i, q in want.items() if self.cart.get(i, 0) != q]
        if drops and (drop is None or self.left() < len(drops) + len(adds) + 1):
            return False    # a cart that cannot be cleared is not rebuilt: two lines for one item fail the order
        for ident in drops:
            have = self.cart.get(ident, 0)
            args = {id_key: ident, act_key: drop, **({qty_key: have} if qty_key else {})}
            reply = self.act(tool, _fit_args(args, self.schemas.get(tool)))
            if reply is None:
                return False
            self.absorb(tool, args, reply.get("observation"))
            self.cart[ident] = 0
        for ident, qty in want.items():
            have = self.cart.get(ident, 0)
            if have == qty:
                continue
            args = {id_key: ident, **({act_key: add} if act_key and add else {}), **({qty_key: qty - have if have < qty else qty} if qty_key else {})}
            reply = self.act(tool, _fit_args(args, self.schemas.get(tool)))
            if reply is None:
                return False
            self.absorb(tool, args, reply.get("observation"))
        return True

    # ---- notes ----------------------------------------------------------------------------------------------
    def take_notes(self, notes: list[str]) -> None:
        for note in notes:
            note = str(note).strip()
            if note and note not in self.notes_seen:
                self.notes_seen.append(note)
                self.fresh_notes.append(note)

    def handle_notes(self) -> str:
        """Act on new store notes: mark sold-out listings, drop or change requirements; returns text for the loop."""
        out: list[str] = []
        while self.fresh_notes:
            note = self.fresh_notes.pop(0)
            needs = "; ".join(f"{n['id']}. item {n['item']}: {self.need_text(n)}" for n in self.live_needs()) or "none"
            cart = "; ".join(f"{i} ({_clip(self.store.get(i).get('title'), 60)})" for i in self.cart_ids()) or "empty"
            got = self.ask_json(NOTE.format(note=note, cart=cart, needs=needs), [FAST, BACKUP], 1200, "low") or {}
            kind = str(got.get("kind") or "other")
            self.trace.append({"kind": "note", "note": _clip(note, 200), "read": _clip(_dump(got), 300)})
            text = f"[store note] {note}"
            if kind == "sold_out":
                gone = self.gone_from_note(str(got.get("listing") or ""), note)
                if not gone and len(self.cart_ids()) > 1 and self.left() > self.reserve() - 1:
                    # the note names no line: the lines' own pages tell which one is out of stock
                    lines = self.cart_ids()
                    for i in lines:
                        self.store.full.discard(i)
                    self.read_full(lines)
                    out_now = [i for i in lines if self.store.get(i).get("stock") is False]
                    gone = out_now[0] if out_now else None
                    for i in out_now[1:]:
                        self.store.gone.add(i)
                if gone:
                    self.store.gone.add(gone)
                    text += f" -> {gone} is no longer available; replace it with the best remaining fit."
            raw_ids = got.get("need_ids")
            raw_ids = raw_ids if isinstance(raw_ids, list) else [raw_ids] if raw_ids is not None else []
            for nid in raw_ids:
                n = next((x for x in self.needs if x["id"] == _count(nid)), None)
                if n and not n["dropped"]:
                    n["dropped"] = True
                    text += f" -> requirement dropped: {self.need_text(n)}."
            for new in got.get("new") or []:
                if isinstance(new, dict) and str(new.get("value") or "").strip():
                    item = _count(new.get("item")) or 1
                    item = item if 1 <= item <= len(self.items) else 1
                    n = self.add_need(item, str(new.get("property") or ""), str(new.get("rule") or "is"), str(new.get("value")), "", "change", note)
                    if n:
                        text += f" -> new requirement: {self.need_text(n)}."
            budget = _number(got.get("budget"))
            if budget and budget > 0:
                self.budget = budget
                text += f" -> budget now {budget:g}."
            self.version += 1
            out.append(text)
        return "\n".join(out)

    def gone_from_note(self, said: str, note: str) -> str | None:
        ids = self.cart_ids()
        hit = next((i for i in ids if i in said or i in note), None)
        if hit:
            return hit
        if said:
            best = max(ids, key=lambda i: _overlap(said, self.store.get(i).get("title") or ""), default=None)
            if best and _overlap(said, self.store.get(best).get("title") or "") >= 0.6:
                return best
        return ids[0] if len(ids) == 1 else None

    # ---- checking an order before it is placed --------------------------------------------------------------
    def order_faults(self) -> tuple[list[str], dict[int, str]]:
        """What is wrong with the cart as an order; and per item the best known replacement."""
        if self.fresh_notes:
            self.handle_notes()
        faults: list[str] = []
        better: dict[int, str] = {}
        basket = self.best_basket()
        lines = self.cart_ids()
        for item in self.items:
            n = item["n"]
            mine = [i for i in lines if self.item_of(i) == n]
            pick = basket.get(n)
            if not mine:
                faults.append(f"item {n} ({item['name']}) has no line in the cart" + (f"; best known fit: {pick}" if pick else ""))
                if pick:
                    better[n] = pick
                continue
            if len(mine) > 1:
                faults.append(f"item {n} has {len(mine)} lines in the cart ({', '.join(mine)}); keep one")
            line = mine[0]
            if self.cart.get(line, 0) != item["qty"]:
                faults.append(f"{line}: quantity {self.cart.get(line, 0)}, the shopper wants {item['qty']}")
            if not self.store.in_stock(line):
                faults.append(f"{line} is not available (sold out / out of stock)")
            vs = self.verdicts(line, self.live_needs(n))
            bad = [self.need_text(x) for x in self.live_needs(n) if vs.get(x["id"]) == "no"]
            if bad:
                faults.append(f"{line} does not meet: " + "; ".join(bad))
            # a requirement the line's page does not state, while other listings of the item do state it, is not met
            unstated = []
            for x in self.live_needs(n):
                if vs.get(x["id"]) != "unknown" or x["rule"] in ("mentions",):
                    continue
                stating = [i for i in self.item_ids(n, full=True) if i != line and self.values_for(i, x["prop"])]
                if stating and not self.values_for(line, x["prop"]):
                    unstated.append(self.need_text(x))
            if unstated:
                faults.append(f"{line} does not state: " + "; ".join(unstated) + " (other listings state this property)")
                bad = bad + unstated
            fault = self.code_fault(line, n)
            if fault:
                faults.append(f"{line} {fault}")
            if not self.money_ok(line, n):
                faults.append(f"{line} costs more than the budget allows")
            if pick and pick != line and not bad and not fault:
                fits = dict(self.fits_for(n))
                if fits.get(pick) == "fits" and self.rank.get("rule") not in (None, "none", "") and \
                        self.rank_key(pick) < self.rank_key(line) and line in fits:
                    faults.append(f"{line} is not the best choice by the shopper's rule ({self.rank['rule']} {self.rank.get('property') or 'price'}): {pick} also fits")
            if (bad or fault or not self.store.in_stock(line) or not self.money_ok(line, n)) and pick and pick != line:
                better[n] = pick
        if not faults and not self.reviewed and lines and not self.spent(0.7) and self.calls < self.call_cap - 6:
            # one read of the whole order by the strong model before the first checkout
            self.reviewed = True
            needs = "\n".join(f"item {x['item']}: {self.need_text(x)}" for x in self.live_needs()) or "none beyond the request"
            shown = "\n\n".join(f"item {self.item_of(i)} line {i} x{self.cart.get(i, 0)}:\n{self.store.page_text(i, 1500)}" for i in lines)
            pool = []
            for item in self.items:
                pool += [i for i, _ in self.fits_for(item["n"]) if i not in lines][:3]
            others = "\n\n".join(f"item {self.item_of(i)} {i}:\n{self.store.page_text(i, 900)}" for i in pool) or "none"
            rule = f"{self.rank.get('rule')} {self.rank.get('property') or ''}".strip() if self.rank.get("rule") not in ("none", "") else "no ranking rule"
            got = self.ask_json(REVIEW.format(request=self.request, needs=needs, lines=shown, others=others, rule=rule),
                                [DEEP, FAST], 1800, "low") or {}
            self.trace.append({"kind": "review", "got": _clip(_dump(got), 500)})
            for entry in got.get("lines") or []:
                if not isinstance(entry, dict) or entry.get("ok") is not False:
                    continue
                n = _count(entry.get("item"))
                swap = str(entry.get("replace_with") or "").strip()
                if n and 1 <= n <= len(self.items):
                    line = self.cart_line_for(n)
                    if swap and swap in self.store.rows and swap != line and self.store.in_stock(swap) and self.item_of(swap) == n:
                        faults.append(f"review: {line} - {_clip(entry.get('why'), 120)}; {swap} fits better")
                        better[n] = swap
                    elif line and not swap:
                        faults.append(f"review: {line} - {_clip(entry.get('why'), 160)}")
        extra = [i for i in lines if self.item_of(i) is None]
        for i in extra:
            faults.append(f"{i} in the cart matches no requested item; remove it")
        if self.budget is not None and not self.budget_each:
            total = sum((self.store.price(i) or 0) * self.cart.get(i, 0) for i in lines)
            if total > self.budget + 1e-9:
                faults.append(f"the cart totals {total:g}, over the budget of {self.budget:g}")
        return faults, better

    def end_buy(self) -> bool:
        """Place the order; True when the store accepted it."""
        tool = self.roles.get("buy")
        if not tool:
            return False
        args: dict[str, Any] = {}
        close = self.tool_extra.get("close")
        if close and tool == self.roles.get("close"):
            args = {close["arg"]: close["buy"]}
        reply = self.act(tool, args)
        if reply is None:
            return False
        self.absorb(tool, args, reply.get("observation"))
        if self.finished:
            self.outcome = "bought"
            return True
        return False

    def end_decline(self, reason: str) -> bool:
        tool = self.roles.get("decline")
        if not tool:
            return False
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        args: dict[str, Any] = {}
        close = self.tool_extra.get("close")
        if close and tool == self.roles.get("close"):
            args[close["arg"]] = close["decline"]
        text_arg = next((k for k, v in props.items() if "string" in _schema_kinds(v) and not _enum_of(v)), None)
        if text_arg:
            args[text_arg] = reason
        reply = self.act(tool, _fit_args(args, self.schemas.get(tool)))
        if reply is None:
            return False
        self.absorb(tool, args, reply.get("observation"))
        if self.finished:
            self.outcome = "declined"
            return True
        return False

    def decline_reason(self) -> str:
        parts = []
        for item in self.items:
            n = item["n"]
            if self.fits_for(n):
                continue
            needs = "; ".join(self.need_text(x) for x in self.live_needs(n)) or "the request"
            parts.append(f"no {item['name']} on offer meets {needs}" + (f" within {self.budget:g}" if self.budget else ""))
        return _clip("Sorry, I could not find a match: " + ("; ".join(parts) or "nothing fits every requirement") + ".", 380)


# ----------------------------------------------------------------------------------------------------------
# finding listings: the opening survey, searches for revealed values, and listing chains
# ----------------------------------------------------------------------------------------------------------
# A preference may take its value from another listing ("same as what Shop X's <kind>, the one with A and B,
# gives, direct or via another listing it names"). That listing's shop comments usually send the reader on ("For
# the storage, refer to Shop Y's laptop running ..."); the first listing in the chain that does not send the
# reader on states the value, or a rule of which values go together. Near twins (same shop and kind, one value
# different) send the same property elsewhere, so a candidate must match every identifying value.

HOP_PICK = """Which candidate listing is the one described?
DESCRIBED: the listing sold by "{shop}" that is a {category}, the one with {identify}
CANDIDATES:
{pages}
It must be the same shop and kind of product, and every identifying value must match (a near twin differs in one;
a listing whose comments say it is not one of them is not it). Write only JSON: {{"target": "candidate id, or empty if none"}}"""

HOP_NEXT = """A store listing's comment sends the reader to another listing for the value of "{prop}":
SENTENCE: {sentence}
Which listing does it point to? shop = the seller named, category = the kind of product, identify = every value
the sentence gives to single it out (e.g. "salicylic acid, lotion, 50 ml").
Write only JSON: {{"shop": "", "category": "", "identify": ""}}"""

HOP_END = """Read this store listing for the value of "{prop}".
LISTING:
{page}
Write only JSON: {{"value": "the value it states for {prop} (facts or comments), or empty",
"pairs": "if it states a rule of which {prop} values go together, the rule's words, else empty"}}"""

PAIRS = """A store states which values of "{prop}" go together: {pairs}
The chain of listings ends at the value "{value}". Which values of "{prop}" go together with "{value}" under this
rule? List only the partners the rule gives (a value pairs with itself only if the rule says so).
Write only JSON: {{"allowed": ["..."]}}"""

_REDIRECT = re.compile(r"\b(?:refer(?:s|red)? to|look (?:to|at)|taken from|take (?:it |that |the \w+ )?from|go(?:es)? by|check(?:s)? (?:the|its)|"
                       r"use(?:s)? the|see the|comes? from|as the reference|is the source|isn't taken|is not taken|sourced? from)\b", re.I)
_PAIR_RULE = re.compile(r"\bgo(?:es)? (?:together )?with\b|\bpairs?\b|\brule\b", re.I)


def _redirect_for(notes: str, prop: str) -> str:
    """The comment sentence that sends the reader elsewhere for a property, or ''."""
    want = _stems(prop)
    for sent in _sentences(notes or ""):
        if not _REDIRECT.search(sent):
            continue
        ss = _stems(sent)
        if want and (want <= ss or (len(want) > 1 and len(want & ss) >= len(want) - 1 and len(want & ss) >= 1)):
            return sent
    return ""


class Finder:
    # ---- searching --------------------------------------------------------------------------------------------
    def search(self, text: str = "", cat: str = "", page: int | None = None, cap: float | None = None,
               item: int | None = None) -> list[str]:
        """Run the store search the way its schema allows; returns the listing ids it showed."""
        specs = self.tool_extra.get("search") or {}
        pick = None
        for name, spec in specs.items():
            if text and spec.get("text"):
                pick = (name, spec)
                break
        if pick is None:
            for name, spec in specs.items():
                if cat and spec.get("cat") and not spec.get("text"):
                    pick = (name, spec)
                    break
        if pick is None and specs:
            pick = next(iter(specs.items()))
        if pick is None:
            return []
        name, spec = pick
        args: dict[str, Any] = {}
        if spec.get("text") and text:
            args[spec["text"]] = text
        if spec.get("cat") and cat:
            args[spec["cat"]] = cat
        if spec.get("mode"):
            modes = spec.get("modes") or []
            want = r"word|text|typed|keyword|search|query" if text else r"browse|field|filter|list|category|shelf"
            mode = next((m for m in modes if re.search(want, m.lower())), None)
            if mode:
                args[spec["mode"]] = mode
        if page and spec.get("page"):
            pspec = ((self.schemas.get(name) or {}).get("properties") or {}).get(spec["page"]) or {}
            if "string" in _schema_kinds(pspec) and not ({"integer", "number"} & _schema_kinds(pspec)):
                # a cursor-paged search: the next-page marker the same search returned last time
                same = [x for x in self.store.searches if x.get("tool") == name and
                        all(x.get("args", {}).get(k) == v for k, v in args.items() if k != spec.get("page"))]
                token = same[-1].get("next") if same else None
                if not token:
                    return []
                args[spec["page"]] = str(token)
            else:
                args[spec["page"]] = page
        if cap is not None and spec.get("cap"):
            args[spec["cap"]] = cap
        if not args:
            return []
        schema = self.schemas.get(name) or {}
        if spec.get("cat") and spec["cat"] not in args and spec["cat"] in (schema.get("required") or []):
            # a search that needs a shelf: the item's known category, else its shelf or name words
            guess = cat
            if not guess and item is not None:
                shelves = sorted(self.item_shelves(item)) if self.store.rows else []
                guess = shelves[0] if shelves else (self.items[item - 1].get("shelf") or self.items[item - 1]["name"])
            if guess:
                args[spec["cat"]] = _clip(guess, 80)
        required = [r for r in schema.get("required") or [] if r not in args]
        if required:
            return []
        reply = self.act(name, _fit_args(args, schema))
        if reply is None:
            return []
        ids = self.absorb(name, args, reply.get("observation"), item)
        return ids

    def read_full(self, ids: list[str]) -> list[str]:
        """Open listings in full, several per step when the store has a side-by-side view."""
        ids = [i for i in dict.fromkeys(ids) if i and i not in self.store.full]
        if not ids:
            return []
        done: list[str] = []
        tool = self.roles.get("compare")
        if tool and len(ids) > 1:
            props = (self.schemas.get(tool) or {}).get("properties") or {}
            arg = next((k for k, v in props.items() if "array" in _schema_kinds(v)), None)
            cap = (props.get(arg) or {}).get("maxItems") if arg else None
            size = cap if isinstance(cap, int) and cap > 0 else 4
            for at in range(0, len(ids), size):
                if self.left() <= self.reserve() or self.finished:
                    break
                chunk = ids[at: at + size]
                reply = self.act(tool, {arg: chunk})
                if reply is None:
                    break
                self.absorb(tool, {arg: chunk}, reply.get("observation"))
                done += [i for i in chunk if i in self.store.full]
            ids = [i for i in ids if i not in self.store.full]
            if not ids:
                return done
        for ident in ids:
            if self.left() <= self.reserve() or self.finished:
                break
            tool = self.roles.get("page")
            # a store whose full page shows no facts keeps them behind a property-list tool: read that instead
            if self.roles.get("facts") and (self.page_facts is not True or not tool):
                tool = self.roles["facts"]
            if not tool:
                return done
            props = (self.schemas.get(tool) or {}).get("properties") or {}
            arg = next((k for k, v in props.items() if "string" in _schema_kinds(v)), None) or next(iter(props), "id")
            reply = self.act(tool, {arg: ident})
            if reply is None:
                break
            self.absorb(tool, {arg: ident}, reply.get("observation"))
            if ident in self.store.full:
                done.append(ident)
        return done

    def open_page(self, ident: str) -> None:
        """A full-page view of one listing with the page tool (a chain's pages are opened this way)."""
        tool = self.roles.get("page")
        if not tool or ident in self.paged or self.left() <= self.reserve():
            return
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        arg = next((k for k, v in props.items() if "string" in _schema_kinds(v)), None) or next(iter(props), "id")
        reply = self.act(tool, {arg: ident})
        if reply is not None:
            self.absorb(tool, {arg: ident}, reply.get("observation"))

    def reserve(self) -> int:
        """Steps kept for building the cart and checking out, with room for one replacement after a sell-out at the
        checkout (a cart that takes one line per call needs a call per line, and one more to take a line out)."""
        tool = self.roles.get("cart")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        whole = any("array" in _schema_kinds(v) for v in props.values())
        per_line = 1 if whole else len(self.items)
        return per_line + (1 if self.roles.get("cart_view") else 0) + 1 + (1 if whole else 2) + 1

    def endgame(self) -> int:
        """When this many steps are left, code builds the cart and places the order (with room for one twin search
        and read after a sell-out)."""
        return self.reserve() + 3

    # ---- the opening survey -----------------------------------------------------------------------------------
    def survey(self) -> None:
        """For each item: one search by its name, then its best rows read in full, so the interview can see the
        properties the item's listings carry."""
        for item in self.items:
            if self.left() <= self.reserve() + 6 or self.finished:
                break
            n = item["n"]
            ids = self.search(text=_clip(item["name"], 60), item=n)
            if len(ids) < 3 and item.get("shelf") and _key(item["shelf"]) != _key(item["name"]):
                ids = self.search(text=_clip(item["shelf"], 60), item=n) or ids
            if not ids:
                continue
            shelves = self.item_shelves(n)
            rows = [i for i in ids if self.item_of(i) == n] or ids
            if not shelves or not rows:
                # the search found no shelf of this name: search by the shelf words of the closest category seen
                rows = ids
            ranked = sorted(rows, key=lambda i: (not self.store.in_stock(i), -sum(1 for x in self.live_needs(n) if _stems(x["value"]) and _stems(x["value"]) <= _stems(self.store.text_of(i)))))
            self.read_full(ranked[:PAGES_BEFORE_ASK])

    # ---- after a revealed preference --------------------------------------------------------------------------
    def value_searches(self, needs: list[dict[str, Any]]) -> None:
        """After the interview: per item with a new preference, search its name with the wanted values (all of the
        item's plain values first, then the newest alone when that finds nothing) and read the best new rows."""
        items = list(dict.fromkeys(n["item"] for n in needs if n["rule"] in ("is", "one_of", "mentions")))
        for number in items:
            if self.left() <= self.reserve() + 4 or self.finished:
                break
            item = self.items[number - 1]
            noun = _clip(item["name"], 40)
            values = [n["value"] for n in self.live_needs(number) if n["rule"] in ("is", "mentions") and n["value"] and len(n["value"]) <= 30]
            newest = [n["value"] for n in needs if n["item"] == number and n["rule"] in ("is", "one_of", "mentions")]
            ids = self.search(text=_clip(" ".join(values[-3:] + [noun]), 60), item=number) if len(values) > 1 else []
            if not ids and newest:
                ids = self.search(text=_clip(f"{newest[-1]} {noun}", 60), item=number)
            fresh = [i for i in ids if i not in self.store.full and self.item_of(i) == number and self.store.in_stock(i) and self.money_ok(i, number)]
            if self.rank.get("rule") == "cheapest":
                fresh.sort(key=lambda i: self.store.price(i) or 1e12)
            else:
                fresh.sort(key=lambda i: -_overlap(" ".join(values), self.store.text_of(i)))
            self.read_full(fresh[:4])

    def cheaper_scan(self, number: int, line: str) -> list[str]:
        """When the shopper wants the cheapest: list the item's shelf below the line's price (a price cap where the
        search takes one, else the item's own search) and read the cheapest unread rows. Returns listings read."""
        price = self.store.price(line)
        if price is None or (number, line) in self.scanned or self.left() <= self.reserve() + 3:
            return []
        self.scanned.add((number, line))
        item = self.items[number - 1]
        cat = str(self.store.get(line).get("category") or "")
        specs = self.tool_extra.get("search") or {}
        capped = next((n for n, sp in specs.items() if sp.get("cap") and (sp.get("cat") or sp.get("text"))), None)
        before = set(self.store.order)
        values = [n["value"] for n in self.live_needs(number) if n["rule"] in ("is", "mentions") and len(n["value"]) <= 30]
        if capped:
            sp = specs[capped]
            cap = max(0, price - 1 if float(price).is_integer() else price - 0.01)
            tries: list[dict[str, Any]] = []
            if sp.get("text"):
                # the wanted values with the item's name, below the line's price
                tries.append({sp["text"]: _clip(" ".join(values[-2:] + [item["name"]]), 60), sp["cap"]: cap, "_mode": r"word|text|typed|keyword"})
            if sp.get("cat") and cat:
                tries.append({sp["cat"]: cat, sp["cap"]: cap, "_mode": r"field|filter|browse|list|category"})
            for args in tries:
                want = args.pop("_mode")
                if sp.get("mode"):
                    mode = next((m for m in sp.get("modes") or [] if re.search(want, m.lower())), None)
                    if mode:
                        args[sp["mode"]] = mode
                if self.left() <= self.reserve() + 3:
                    break
                reply = self.act(capped, _fit_args(args, self.schemas.get(capped)))
                if reply is None:
                    continue
                got_ids = self.absorb(capped, args, reply.get("observation"), number)
                if any(self.item_of(i) == number and (self.store.price(i) or 1e12) < price for i in got_ids):
                    break
        else:
            self.search(text=_clip(" ".join(values[-2:] + [item["name"]]), 60), item=number)
        fresh = [i for i in self.store.order if i not in before or i not in self.store.full]
        fresh = [i for i in fresh if i != line and i not in self.store.full and self.item_of(i) == number and self.store.in_stock(i)
                 and (self.store.price(i) or 1e12) < price]
        fresh.sort(key=lambda i: self.store.price(i) or 1e12)
        got = self.read_full(fresh[:4])
        self.trace.append({"kind": "cheaper", "item": number, "line": line, "price": price, "read": got})
        return got

    # ---- listing chains --------------------------------------------------------------------------------------
    def hop_candidates(self, hop: dict[str, Any], seen: set[str], page: int = 1, tried: set[str] | None = None) -> list[str]:
        """Listings that may be the hop's target: the named shop's listings of the named kind (searched by shop and
        kind; a later page when the first holds no match)."""
        shop = re.sub(r"['\u2019]s$", "", str(hop.get("shop") or "").strip())
        cat = str(hop.get("category") or "").strip()
        ident = str(hop.get("identify") or "")
        tried = tried or set()
        if not shop:
            # the model may leave the seller inside the identifying words: take a shop name the store has shown
            shops = {str(r.get("shop")) for r in self.store.rows.values() if r.get("shop")}
            shop = next((w for w in re.findall(r"[A-Z][A-Za-z]{3,}", ident + " " + cat) if w in shops), "")
            if shop:
                hop["shop"] = shop
        noun = " ".join(_words(re.sub(r"&|\band\b", " ", cat))[:2])

        def known() -> list[str]:
            out = []
            for i in self.store.order:
                r = self.store.get(i)
                if i in seen or i in tried or not shop or _key(r.get("shop") or "") != _key(shop):
                    continue
                if cat and r.get("category") and _overlap(cat, r.get("category")) < 0.34 and _overlap(cat, r.get("title") or "") < 0.34:
                    continue
                out.append(i)
            out.sort(key=lambda i: -_overlap(ident, self.store.text_of(i)))
            return out
        if not shop:
            return []
        # always ask the store (a target may sit only in the search results), merged with listings already known
        self.search(text=f"{shop} {noun}".strip(), page=page if page > 1 else None)
        got = known()
        if not got and page == 1:
            self.search(text=shop)
            got = known()
        return got[:8]

    @staticmethod
    def value_in(text: str, value: str) -> bool:
        """Does a description carry a listing's value (any comma part of it): its numbers with their units, or all
        of its words."""
        have_nums = _range_of(text)
        have = _stems(text)
        for part in re.split(r"\s*[,;/|]\s*", value or ""):
            if not part.strip():
                continue
            nums = _range_of(part)
            if nums:
                # a value with numbers is carried only by the same numbers
                if all(any(abs(x - y) < 1e-9 and (u == w or not u or not w) for y, w in have_nums) for x, u in nums):
                    return True
                continue
            words = {w for w in _stems(part) if not re.search(r"\d", w)}
            if words and words <= have:
                return True
        return False

    def ident_score(self, ident: str, cand: str, before: str = "") -> float:
        """How well a listing matches a hop's identifying words: each of its fact keys the words name must carry
        the listing's own value somewhere in them (values may stand before or after the key); remaining words count
        as values to find among the listing's values and title; a comment denying one of them costs."""
        facts = self.store.facts(cand)
        r = self.store.get(cand)
        text = " " + _key(ident) + " "
        score, named = 0.0, 0
        for k, v in facts.items():
            if len(k) < 3 or not v.strip() or (" " + k + " ") not in text:
                continue
            named += 1
            score += 1.0 if self.value_in(ident, v) else -1.0
        key_stems = set().union(*[_stems(k) for k in facts]) if facts else set()
        values = " ".join(list(facts.values()) + [str(r.get("title") or "")])
        loose = _stems(ident) - key_stems - {"and", "with", "the", "one", "that", "has", "from", "for", "sold", "listing", "product", "a", "an"}
        if loose:
            score += len(loose & _stems(values)) / len(loose) * (1.0 if not named else 0.5)
        notes = str(r.get("notes") or "")
        for denied in re.findall(r"\b(?:not|isn't|is not)\s+(?:a |an |the )?([a-z0-9][\w\- ]{2,30})", notes.lower()):
            if _stems(denied) & loose:
                score -= 1.0
        if before:
            score += _overlap(before, notes) * 0.5
        return score

    def pick_target(self, hop: dict[str, Any], cands: list[str], before: str) -> str:
        """The candidate the hop describes; the fast model settles close calls."""
        ident = str(hop.get("identify") or "")
        scores = {i: self.ident_score(ident, i, before) for i in cands}
        ranked = sorted(cands, key=lambda i: -scores[i])
        if not ranked:
            return ""
        if scores[ranked[0]] > 0 and (len(ranked) == 1 or scores[ranked[0]] - scores[ranked[1]] >= 0.5):
            return ranked[0]
        pages = "\n\n".join(f"[{i}]\n{self.store.page_text(i, 1400)}" for i in ranked[:4])
        got = self.ask_json(HOP_PICK.format(shop=hop.get("shop", ""), category=hop.get("category", ""), identify=ident, pages=pages),
                            [FAST, DEEP], 900, "low") or {}
        target = str(got.get("target") or "").strip()
        return target if target in ranked else (ranked[0] if scores[ranked[0]] > 0.5 else "")

    def resolve_pointer(self, need: dict[str, Any]) -> None:
        hop = dict(need.get("pointer") or {})
        seen: set[str] = set()
        chain: list[str] = []
        before = ""
        for _ in range(6):
            if self.left() <= self.reserve() + 3 or self.finished or self.spent(0.95):
                break
            target, tried = "", set()
            for page in (1, 2, 3):
                cands = self.hop_candidates(hop, seen, page, tried)
                if not cands:
                    break
                tried.update(cands)
                self.read_full(cands)
                target = self.pick_target(hop, cands, before)
                if target or self.left() <= self.reserve() + 4:
                    break
            if not target:
                break
            seen.add(target)
            chain.append(target)
            self.open_page(target)
            notes = str(self.store.get(target).get("notes") or "")
            sentence = _redirect_for(notes, need["prop"])
            if sentence:
                nxt = self.ask_json(HOP_NEXT.format(prop=need["prop"], sentence=sentence), [FAST, DEEP], 700, "low") or {}
                if not str(nxt.get("shop") or "").strip():
                    # the seller left out of the reading: a shop name the store has shown, else this listing's own
                    # shop when the sentence sends the reader to "another product" of it
                    shops = {str(r.get("shop")) for r in self.store.rows.values() if r.get("shop")}
                    named = next((w for w in re.findall(r"[A-Z][A-Za-z]{3,}", sentence) if w in shops), "")
                    nxt["shop"] = named or str(self.store.get(target).get("shop") or "")
                if str(nxt.get("shop") or "").strip():
                    hop, before = nxt, sentence
                    continue
                break
            # the end of the chain: the value from the facts, else the page's words; and any pairing rule
            values = [v for _, v in self.values_for(target, need["prop"]) if v.strip()]
            rule_text = " ".join(x for x in _sentences(notes) if _PAIR_RULE.search(x))
            value = values[0] if values else ""
            pairs = rule_text if need["rule"] == "goes_with" else ""
            if not value or (need["rule"] == "goes_with" and not pairs):
                got = self.ask_json(HOP_END.format(prop=need["prop"], page=self.store.page_text(target, 2000)), [FAST, DEEP], 700, "low") or {}
                value = value or str(got.get("value") or "").strip()
                pairs = pairs or str(got.get("pairs") or "").strip()
            if value:
                need["chain_end"] = value
                rule = need["rule"]
                if rule == "goes_with":
                    allowed: list[str] = []
                    if pairs:
                        allowed = (self.ask_json(PAIRS.format(prop=need["prop"], pairs=pairs, value=value), [FAST, DEEP], 800, "low") or {}).get("allowed") or []
                        allowed = [str(a) for a in allowed if str(a).strip()]
                    need["resolved"] = ", ".join(allowed) if allowed else value
                    need["rule"] = "one_of" if len(allowed) > 1 else "is"
                else:
                    need["resolved"] = value
                    need["rule"] = {"same_as": "is", "not_same_as": "is_not"}.get(rule, rule)
                need["value"] = need["resolved"]
                self.version += 1
            break
        need["chain"] = chain
        self.trace.append({"kind": "chain", "need": need["id"], "chain": chain, "resolved": need.get("resolved")})


# ----------------------------------------------------------------------------------------------------------
# the store loop
# ----------------------------------------------------------------------------------------------------------

GUIDE = """You shop for one customer through the store's tools. The order must hold every product the customer
named, in the number of units they named, and every one of those products must satisfy all of their conditions;
among products that satisfy them, follow the customer's own preference rule (such as lowest price). When the
store has nothing that can satisfy the conditions, close the request without an order, as the customer asked.

Working rules:
- A tool call costs one step and the steps run out; read the status block under each result for what remains.
- Search each item's shelf and open listings in full before choosing: a listing's facts, comments and description
  decide whether it fits; search rows alone do not.
- Do not write to the shopper yourself. The interviewer (code) asks the shopper when the time is right; if you call
  the message tool it runs the interviewer instead. Treat every answer it reports as a firm requirement.
- The status block lists the requirements, and per item the opened listings that meet them ("fits"), best first by
  the shopper's rule. Prefer those. "likely" means nothing contradicts it but a requirement is not stated on the
  page: open or compare to confirm when steps allow.
- A requirement that takes its value from another listing is followed by code; open what it reports.
- Parts and accessories must fit the device by its model code ("Fits: ...", "Model number: ...").
- Only buy what is in stock and within the budget. Put one line per item in the cart at the item's quantity.
- When the shopper wants the cheapest (or most/least of something), look past the first page of results before
  deciding: a cheaper listing that fits wins.
- Check out with the store's purchase tool; code reviews the cart first and tells you what to fix. Store notes
  (sold out, a change of mind) arrive with results: act on them.
Think briefly, then call one tool."""


class Astrolabe(Core, Brief, Interview, Checks, Codes, Cart, Finder):
    def __init__(self, problem: dict[str, Any]) -> None:
        Core.__init__(self, problem)
        self.store = Store()
        self.needs: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.budget: float | None = None
        self.budget_each = False
        self.rank: dict[str, str] = {"rule": "none", "property": "", "said": ""}
        self.msg_limit: int | None = None
        self.msg_used = 0
        self.msg_reserve = 0
        self.stock_only = False
        self.decline_ok = False
        self.procedure: list[str] = []
        self.brief_raw: dict[str, Any] = {}
        self.asked = False
        self.settled: set[tuple[int, str]] = set()
        self.again: set[tuple[int, str]] = set()
        self.quiet: dict[int, int] = {}
        self.found_on: dict[int, int] = {}
        self.pointer_queue: list[dict[str, Any]] = []
        self.dialogue: list[dict[str, str]] = []
        self.item_cache: dict[str, int | None] = {}
        self.shelf_memo: dict[int, tuple[int, set[str]]] = {}
        self.item_cats: dict[int, set[str]] = {}
        self.search_for: dict[str, int] = {}
        self.verdict_memo: dict[tuple[str, int], str] = {}
        self.decider_memo: dict[tuple[int, int, int], str] = {}
        self.value_judged: dict[tuple[int, str], str] = {}
        self.value_asked: set[int] = set()
        self.code_memo: dict[str, tuple[int, dict[str, Any]]] = {}
        self.cart: dict[str, int] = {}
        self.cart_seen = -1
        self.paged: set[str] = set()
        self.scanned: set[tuple[int, str]] = set()
        self.page_facts: bool | None = None   # does the page tool show facts (None: not known yet)
        self.version = 0
        self.guard_rounds = 0
        self.reviewed = False
        self.backed = False
        self.decline_pushed = 0
        self.convo: list[dict[str, Any]] = []
        self.turns = 0

    # ---- every observation goes through here ------------------------------------------------------------------
    def absorb(self, tool: str, args: dict[str, Any], obs: Any, item: int | None = None) -> list[str]:
        body, notes = _envelope(obs)
        self.take_notes(notes)
        if tool == self.roles.get("cart_view"):
            self.read_cart_view(body)
            return []
        if tool == self.roles.get("cart"):
            if not (isinstance(body, dict) and (body.get("error") or body.get("call_failure"))):
                self.note_cart_call(tool, args, body)
            if _records(body) and re.search(r"select|cart|bag|line|due|total", _dump(body).lower()):
                self.read_cart_view(body)
            return []
        if tool in (self.roles.get("ask"), self.roles.get("buy"), self.roles.get("decline"), self.roles.get("close")):
            return []
        full = tool in (self.roles.get("page"), self.roles.get("facts"), self.roles.get("compare"))
        ids = []
        recs = _records(body)
        if full and tool == self.roles.get("page") and recs and self.roles.get("facts") and self.roles.get("facts") != tool:
            # does this store's full page show facts? judged on what the page itself returned
            shown = any(r.get("facts") for r in recs)
            if self.page_facts is None or shown:
                self.page_facts = shown
        for rec in recs:
            ident = self.store.add(rec, full)
            ids.append(ident)
            if item is not None and ident not in self.search_for:
                self.search_for[ident] = item
        if full and tool == self.roles.get("page"):
            self.paged.update(ids)
        if tool not in (self.roles.get("page"), self.roles.get("facts"), self.roles.get("compare")):
            self.store.searches.append({"tool": tool, "args": args, "ids": ids, **_paging(body)})
        return ids

    # ---- what the loop model sees -----------------------------------------------------------------------------
    def render(self, tool: str, args: dict[str, Any], obs: Any, ids: list[str]) -> str:
        body, notes = _envelope(obs)
        if tool in (self.roles.get("page"), self.roles.get("facts"), self.roles.get("compare")) and ids:
            text = "\n\n".join(self.store.page_text(i) for i in ids)
        elif ids:
            paging = _paging(body)
            head = f"{len(ids)} listings shown" + (f" of {paging['count']}" if paging.get("count") is not None else "")
            if paging.get("next"):
                head += f"; more on the next page ({paging['next']})"
            text = head + ":\n" + "\n".join(self.store.row_line(i) for i in ids)
        else:
            text = _clip(_dump(body), 1500)
        if notes:
            text += "\nSTORE NOTE: " + " | ".join(notes)
        return _clip(text, OBS_CHARS)

    def status(self) -> str:
        lines = [f"[status] steps left {self.left()} of {self.max_steps} (at {self.endgame()} left, code places the order with the best fits); messages to the shopper left {self.msgs_left()}"
                 + (f"; budget {self.budget:g}" + (" per unit" if self.budget_each else " in total") if self.budget else "")
                 + (f"; rule: {self.rank['rule']} {self.rank.get('property') or ''}".rstrip() if self.rank.get("rule") not in ("none", "") else "")]
        for item in self.items:
            n = item["n"]
            needs = "; ".join((self.pointer_text(x) if x.get("pointer") and not x.get("resolved") else self.need_text(x))
                              + (" [from the shopper]" if x["src"] == "shopper" else "") for x in self.live_needs(n))
            lines.append(f"item {n}: {item['name']} x{item['qty']}" + (f" (must fit {item['fits']})" if item.get("fits") else "")
                         + f" | requirements: {needs or 'none stated'}")
            fits = self.fits_for(n)
            # the fast model reads the pages of the best few "likely" listings (a few per status, memoised)
            checked = 0
            for i, tag in fits[:4]:
                if tag == "likely" and checked < 2 and not self.spent(0.8):
                    needs = [x for x in self.live_needs(n) if (i, x["id"]) not in self.verdict_memo]
                    if needs:
                        self.verdicts(i, needs, True)
                        checked += 1
            if checked:
                fits = self.fits_for(n)
            if fits:
                lines.append("   fits: " + "; ".join(f"{i} {self.store.price(i) or '?'} {tag}" for i, tag in fits[:5]))
            else:
                read = len(self.item_ids(n, full=True))
                lines.append(f"   fits: none among {read} opened listings")
        if self.fit_pairs():
            basket = self.best_basket()
            if basket:
                lines.append("best basket whose model codes fit together: " + "; ".join(f"item {n}: {i}" for n, i in sorted(basket.items())))
        if self.cart_ids():
            lines.append("cart: " + "; ".join(f"{i} x{self.cart[i]}" for i in self.cart_ids()))
        else:
            lines.append("cart: empty")
        return "\n".join(lines)

    # ---- the loop -------------------------------------------------------------------------------------------
    def offered(self) -> list[dict[str, Any]]:
        return self.tools

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_left() <= 0:
            return False
        if self.clock() > CLOCK_ASK or self.step >= self.max_steps * 0.45:
            return True
        return all(len(self.item_ids(i["n"], full=True)) >= min(PAGES_BEFORE_ASK, max(1, len(self.item_ids(i["n"]))))
                   and self.item_ids(i["n"]) for i in self.items)

    def after_interview(self) -> str:
        """Search for revealed values, follow listing chains; returns text for the loop model."""
        text: list[str] = []
        fresh = [n for n in self.needs if n["src"] == "shopper" and not n.get("searched")]
        for n in fresh:
            n["searched"] = True
        while self.pointer_queue:
            need = self.pointer_queue.pop(0)
            self.resolve_pointer(need)
            if need.get("resolved"):
                text.append(f"[chain] {need['prop']} for item {need['item']}: the pointed listing chain "
                            f"({' -> '.join(need.get('chain') or [])}) ends at {need.get('chain_end')}; requirement now: {self.need_text(need)}")
            else:
                text.append(f"[chain] could not finish the listing chain for {need['prop']} (item {need['item']}); "
                            f"opened {', '.join(need.get('chain') or []) or 'nothing'}")
        self.value_searches([n for n in fresh if not n.get("pointer")])
        return "\n".join(text)

    def run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Run one tool call the loop model asked for; returns the text the model sees."""
        roles = self.roles
        if name == roles.get("ask"):
            if not self.asked:
                summary = self.interview()
                extra = self.after_interview()
                return (summary or "The interviewer had nothing to ask.") + ("\n" + extra if extra else "")
            return "The interviewer already asked the shopper; no more messages are needed. Continue shopping."
        close = self.tool_extra.get("close")
        ending = name in (roles.get("buy"), roles.get("decline"), roles.get("close"), roles.get("cart_view"))
        if self.left() <= self.endgame() and not ending:
            return "Out of steps for shopping: code now places the order with the best fits."
        buying = name == roles.get("buy") and name != roles.get("close") or (close and name == roles.get("close") and str(args.get(close["arg"])) == close["buy"])
        declining = name == roles.get("decline") and name != roles.get("close") or (close and name == roles.get("close") and str(args.get(close["arg"])) == close["decline"])
        if buying:
            return self.try_buy()
        if declining:
            return self.try_decline(args)
        if name not in self.schemas:
            return f"There is no tool named {name}."
        if name == roles.get("page") and self.page_facts is False and roles.get("facts") and roles["facts"] != name:
            # this store's full page shows no facts: the property list is what decides a listing
            ident = _arg_id(args, (self.schemas.get(name) or {}).get("properties") or {})
            if ident and not self.store.facts(ident):
                props = (self.schemas.get(roles["facts"]) or {}).get("properties") or {}
                arg = next((k for k, v in props.items() if "string" in _schema_kinds(v)), None)
                if arg:
                    name, args = roles["facts"], {arg: ident}
        reply = self.act(name, args)
        if reply is None:
            return "The store refused that call (check the arguments)."
        obs = reply.get("observation")
        ids = self.absorb(name, args, obs)
        return self.render(name, args, obs, ids)

    def try_buy(self) -> str:
        if not self.asked and self.ready_to_ask_late():
            summary = self.interview()
            extra = self.after_interview()
            return (summary + ("\n" + extra if extra else "") + "\nThe order was not placed yet: check the cart against these answers, then check out.")
        if self.cart_seen < 0 or self.cart_seen < self.step - 1:
            self.view_cart()
        if self.rank.get("rule") == "cheapest" and self.left() > self.reserve() + 5:
            for item in self.items:
                line = self.cart_line_for(item["n"])
                if line:
                    self.cheaper_scan(item["n"], line)
        if not self.backed:
            # a second fitting listing per item, read before the first checkout: a line often sells out at checkout
            self.backed = True
            for item in self.items:
                if self.left() <= self.reserve() + 5:
                    break
                if len(self.fits_for(item["n"])) < 2:
                    self.widen(item["n"])
        faults, better = self.order_faults()
        if faults and self.guard_rounds < 3 and self.left() > 2:
            self.guard_rounds += 1
            hint = ""
            if better:
                hint = "\nBest known replacements: " + "; ".join(f"item {n}: {i}" for n, i in better.items())
            return "[order review] Not placed yet. Fix these first:\n- " + "\n- ".join(faults) + hint
        ok = self.end_buy()
        if ok:
            return "Order placed."
        text = self.handle_notes()
        faults, _ = self.order_faults()
        if not faults and self.left() > 1 and self.end_buy():
            # the store turned the order down with a note that leaves the cart valid (a new budget the cart is
            # within, a dropped wish): placed again as it stands
            return (text + "\nOrder placed.").strip()
        for _ in range(2):
            # a line sold out or a requirement changed at checkout: code swaps in the best remaining fit and
            # orders again when the cart then passes review
            fixed = self.recover()
            if not fixed:
                break
            faults, _ = self.order_faults()
            if faults or self.left() <= 1:
                text += "\n" + fixed
                break
            if self.end_buy():
                return (text + "\n" + fixed + "\nOrder placed.").strip()
            text += "\n" + fixed + "\n" + self.handle_notes()
        faults, better = self.order_faults()
        hint = ("\nStill wrong:\n- " + "\n- ".join(faults)) if faults else ""
        if better:
            hint += "\nBest known replacements: " + "; ".join(f"item {n}: {i}" for n, i in better.items())
        return (text or "The store did not accept the order.") + hint + "\nFix the cart and check out again."

    def widen(self, number: int, rows: int = 4) -> list[str]:
        """Read more of an item's listings: unread rows first (cheapest first when the rule is cheapest), then the
        next page of the item's search."""
        unread = [i for i in self.item_ids(number) if i not in self.store.full and self.store.in_stock(i) and self.money_ok(i, number)]
        if self.rank.get("rule") == "cheapest":
            unread.sort(key=lambda i: self.store.price(i) or 1e12)
        if not unread and self.left() > self.reserve() + 1:
            item = self.items[number - 1]
            page = 1 + sum(1 for x in self.store.searches if self.search_for.get((x.get("ids") or [""])[0]) == number)
            self.search(text=_clip(item.get("shelf") or item["name"], 60), page=page, item=number)
            unread = [i for i in self.item_ids(number) if i not in self.store.full and self.store.in_stock(i) and self.money_ok(i, number)]
        return self.read_full(unread[:rows])

    def twins(self, gone: str, number: int) -> list[str]:
        """Listings like one that sold out: a search by the leading words of its title."""
        title = str(self.store.get(gone).get("title") or "")
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", title) if len(w) > 2][:5]
        if not words or self.left() <= self.reserve():
            return []
        ids = self.search(text=" ".join(words), item=number)
        if not ids and len(words) > 3:
            ids = self.search(text=" ".join(words[:3]), item=number)
        fresh = [i for i in ids if i != gone and i not in self.store.full and self.store.in_stock(i) and self.item_of(i) == number]
        fresh.sort(key=lambda i: -_overlap(title, self.store.get(i).get("title") or ""))
        got = self.read_full(fresh[:4])
        self.trace.append({"kind": "twins", "gone": gone, "rows": len(ids), "fresh": fresh[:6], "read": got})
        return got

    def recover(self) -> str:
        """Replace cart lines that are gone or fail the requirements with the best remaining fit."""
        changed: list[str] = []
        picks: dict[int, str] = {}
        for item in self.items:
            n = item["n"]
            line = self.cart_line_for(n)
            bad = (line is None or not self.store.in_stock(line) or not self.money_ok(line, n) or bool(self.code_fault(line, n))
                   or any(v == "no" for v in self.verdicts(line, self.live_needs(n)).values()))
            if not bad:
                picks[n] = line
                continue
            fits = [i for i, _ in self.fits_for(n) if i != line]
            if not fits and line:
                self.twins(line, n)
                fits = [i for i, _ in self.fits_for(n) if i != line]
            if not fits:
                self.widen(n)
                fits = [i for i, _ in self.fits_for(n) if i != line]
            if fits and self.rank.get("rule") == "cheapest" and self.left() > self.reserve() + 3:
                # the replacement is the cheapest remaining fit: look below its price first
                self.cheaper_scan(n, fits[0])
                fits = [i for i, _ in self.fits_for(n) if i != line] or fits
            if fits:
                picks[n] = fits[0]
                changed.append(f"item {n}: {line or 'none'} -> {fits[0]}")
            elif line:
                picks[n] = line
        self.trace.append({"kind": "recover", "changed": changed, "picks": picks})
        if not changed:
            return ""
        if not self.set_cart(picks):
            return ""
        if self.roles.get("cart_view") and self.left() > 2:
            self.view_cart()
        return "[recovery] cart changed: " + "; ".join(changed)

    def trace_verdicts(self, why: str) -> None:
        """For debugging runs: each item's opened listings with their verdict per requirement."""
        rows = {}
        for item in self.items:
            n = item["n"]
            needs = self.live_needs(n)
            rows[n] = [(i, self.store.price(i), self.store.in_stock(i), {x["id"]: v for x, v in zip(needs, [self.verdicts(i, [x], False).get(x["id"]) for x in needs])})
                       for i in self.item_ids(n, full=True)[:12]]
        self.trace.append({"kind": "verdicts", "why": why, "needs": [(x["id"], self.need_text(x)) for x in self.live_needs()], "rows": rows})

    def ready_to_ask_late(self) -> bool:
        return bool(self.roles.get("ask")) and self.msgs_left() > 0 and self.left() > self.reserve() + 3

    def try_decline(self, args: dict[str, Any]) -> str:
        if not self.asked and self.ready_to_ask_late():
            summary = self.interview()
            extra = self.after_interview()
            return summary + ("\n" + extra if extra else "") + "\nNot ended yet: look again with these answers."
        basket = self.best_basket()
        covered = all(n["n"] in basket for n in self.items)
        if not covered and self.decline_pushed < 2 and self.left() > self.reserve() + 1:
            # before giving up, read more of each uncovered item's listings
            for item in self.items:
                if item["n"] not in basket and len(self.item_ids(item["n"], full=True)) < 12:
                    self.widen(item["n"])
            basket = self.best_basket()
            covered = all(n["n"] in basket for n in self.items)
            if not covered:
                self.decline_pushed += 1
                return ("[decline review] Not ended yet: code read more listings and none fits so far. Search the "
                        "shelf with other words or the next page, open more listings, and end only when nothing fits.\n" + self.status())
        if covered and self.decline_pushed < 2 and self.left() > self.reserve() + 2:
            self.decline_pushed += 1
            return ("[decline review] Not ended: these opened listings meet every known requirement: "
                    + "; ".join(f"item {n}: {i}" for n, i in basket.items()) + ". Buy them unless a page says otherwise.")
        if covered:
            # known fits for every item: they are ordered rather than declining
            self.finish()
            return "Order placed with the listings that meet every requirement." if self.outcome == "bought" else "Ended."
        self.trace_verdicts("decline")
        reason = next((v for v in args.values() if isinstance(v, str) and len(v) > 10), "") or self.decline_reason()
        if self.end_decline(_clip(reason, 380)):
            return "Ended without a purchase."
        text = self.handle_notes()
        return (text or "The store did not accept ending without a purchase.") + "\nContinue."

    def loop(self) -> None:
        intro = ("SHOPPER'S REQUEST:\n" + self.request + "\n\n" + self.status()
                 + "\n\nStart by searching for each item and opening listings in full; the interviewer asks the shopper when the store shows each item's properties.")
        if self.view.get("opening") is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + _clip(_dump(self.view.get("opening")), 3000)
        self.convo = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        idle = 0
        while not self.finished and not self.broke and self.turns < self.turn_cap and self.left() > self.endgame():
            if self.spent() or self.clock() > CLOCK_FINISH or self.calls >= self.call_cap - 6:
                break
            if self.ready_to_ask():
                summary = self.interview()
                extra = self.after_interview()
                if summary or extra:
                    self.convo.append({"role": "user", "content": "\n".join(x for x in (summary, extra) if x) + "\n\n" + self.status()})
                continue
            self.trim()
            self.turns += 1
            message = self.chat_turn()
            if message is None:
                idle += 1
                if idle >= 3 or self.no_credit:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                fn = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"c{self.turns}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": _dump(parsed or {})}})
            content = message.get("content") if isinstance(message.get("content"), str) else ""
            if not calls:
                idle += 1
                self.convo.append({"role": "assistant", "content": content or "(no call)"})
                self.convo.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.convo.append({"role": "assistant", "content": content or "", "tool_calls": calls})
            for at, call in enumerate(calls):
                name = call["function"]["name"]
                args = _json_obj(call["function"]["arguments"]) or {}
                if self.finished or at >= (3 if self.left() > self.endgame() + 3 else 1):
                    result = "Not run (one call at a time now)."
                else:
                    result = self.run_tool(name, args)
                    notes = self.handle_notes()
                    if notes:
                        result += "\n" + notes
                if at == len(calls) - 1 and not self.finished:
                    result += "\n\n" + self.status()
                self.convo.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    def chat_turn(self) -> dict[str, Any] | None:
        return self.chat(self.convo, [FAST, BACKUP], 2500, self.offered(), "low")

    def trim(self) -> None:
        """Keep the chat under its size: old tool results shrink to a line."""
        size = sum(len(_dump(m.get("content"))) for m in self.convo)
        at = 2
        while size > CHAT_CHARS and at < len(self.convo) - 6:
            m = self.convo[at]
            if m.get("role") == "tool" and len(str(m.get("content"))) > 300:
                size -= len(str(m["content"])) - 200
                m["content"] = _clip(m["content"], 200) + " [older result shortened]"
            at += 1

    # ---- ending when the loop is done -------------------------------------------------------------------------
    def finish(self) -> None:
        """Code ends the episode: the best known basket is ordered (a line that sells out is replaced from what is
        known, or found by a twin search while steps allow); with no fit for some item, the episode is declined. The
        episode is never left without an ending while a step remains."""
        if self.finished:
            return
        if not self.asked and self.ready_to_ask_late():
            self.interview()
            self.after_interview()
        widened: set[int] = set()
        for _ in range(5):
            if self.finished or self.left() <= 0:
                return
            if self.fresh_notes:
                self.handle_notes()
            basket = self.best_basket()
            missing = [i["n"] for i in self.items if i["n"] not in basket]
            if missing and self.left() > self.reserve():
                # an item without a fit: twins of a sold-out line, else more of its shelf
                for n in missing:
                    if n in widened or self.left() <= self.reserve():
                        continue
                    widened.add(n)
                    line = self.cart_line_for(n)
                    if line and not self.store.in_stock(line):
                        self.twins(line, n)
                    if not self.fits_for(n):
                        self.widen(n)
                basket = self.best_basket()
                missing = [i["n"] for i in self.items if i["n"] not in basket]
            if not missing:
                want = {basket[i["n"]]: i["qty"] for i in self.items}
                if {i: q for i, q in self.cart.items() if q > 0} != want:
                    self.set_cart(basket)
                if self.roles.get("cart_view") and self.left() > 2:
                    self.view_cart()
                    if self.fresh_notes:
                        self.handle_notes()
                        if any(not self.store.in_stock(i) for i in self.cart_ids()):
                            continue
                if self.left() > 0 and self.end_buy():
                    return
                self.handle_notes()
                continue
            # some item has no fit: decline (a refused decline may carry a change that lets a listing fit)
            self.trace_verdicts("finish")
            if self.end_decline(self.decline_reason()):
                return
            self.handle_notes()
        if not self.finished and self.left() > 0:
            # last resort: an ending of some kind rather than none
            if self.cart_ids() and all(self.store.in_stock(i) for i in self.cart_ids()) and self.end_buy():
                return
            self.end_decline(self.decline_reason())

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.survey()
        if self.ready_to_ask():
            self.interview()
            self.after_interview()
        self.loop()
        if not self.finished:
            self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "outcome": self.outcome, "steps": self.step,
                           "cost": round(self.cost, 5), "calls": self.calls, "turns": self.turns, "clock": round(self.clock(), 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Astrolabe(problem_data)
    except Exception as exc:  # noqa: BLE001
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:  # noqa: BLE001
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:  # noqa: BLE001
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
