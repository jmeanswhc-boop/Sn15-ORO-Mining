"""Bunny G13: a shopping agent for step-budgeted store worlds.

One ledger holds what the shopper wants: the items, the needs, the way to choose, the budgets and the request's own
steps. Every change to it raises its version. A setup reading fills the ledger and maps the store's tools to their
jobs. A model drives the search one call at a time, while plain code keeps the listing table from every result, sends
the questions to the shopper, checks each listing against the ledger before it goes in the cart, carries out the
request's own steps before the order or before ending without buying, follows changes the shopper or the store
announce, and finishes the trip itself when steps or time run short.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import threading
import time
import unicodedata
from typing import Any

from src.agent.proxy_client import ProxyClient

# ------------------------------------------------------------------------------------------------------------ models
HAIKU, SONNET, GLM = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "z-ai/glm-5.2"
SONNET_FIRST = False               # True puts Sonnet first in the setup reading (flip after an A/B comparison)
SETUP = [SONNET, HAIKU, GLM] if SONNET_FIRST else [HAIKU, SONNET, GLM]   # request read; next model only on a failed reading
CHECK = [HAIKU, GLM]               # listing check of the unclear and clash cells
ESCALATE = SONNET                  # one call per trip for cells Haiku leaves open on a listing about to go in the cart
FINISH_RETRY = [HAIKU, SONNET]     # telling the order and the ending apart when the first reading did not
DRIVER = [HAIKU, GLM, HAIKU]       # the step-by-step loop
LIGHT = [HAIKU, GLM]               # shopper lines, the request's steps, notices, ending words, corrected finishes
TERSE = [GLM, HAIKU]               # very short JSON answers (field roles, finish values, cart arguments): glm is cheapest
STRONG = [SONNET, HAIKU]           # the two readings a trip turns on: the cart before the order, the leads before an ending
HEARING = [SONNET, HAIKU]          # planning the questions and reading the shopper's answers: the wish a trip turns on
# money: a race runs about ninety trips on one inference allowance of about five dollars, and once it is spent every
# later model call fails, so each trip keeps to its share; the strong model is the one whose calls the share decides
PRICES = {HAIKU: (0.10, 0.01, 0.50), SONNET: (2.0, 0.10, 10.0), GLM: (0.006, 0.005, 6.8)}  # dollars per million tokens: sent, read from cache, written
COSTLY = (SONNET,)                 # models a chain passes over once the trip's spend is past its share
TRIP_SHARE = 0.020                 # dollars spent below which a costly model takes its turn in a chain
LATE_SHARE = 0.042                 # ... and for the two readings just before the order or the ending, kept for them
CALL_GUESS = 0.009                 # dollars one costly call is taken to cost before the trip has made one
ROOMY = ("anthropic/",)            # these models reason before answering and need room for it
CACHED = ("anthropic/",)           # these models take cache breakpoints on content blocks
LEAN_CUT = 0.4                     # share of the kept driver turns dropped at once when the chat outgrows its budget
WAITS = (25.0, 15.0, 10.0)         # seconds each model of a chain may take, by its place in the chain
SETUP_WAITS, SETUP_DEADLINE = (40.0, 30.0, 20.0), 60.0   # the request read: per model, and for the whole reading
ROLES_DEADLINE = 15.0              # the field-role read, all models together
ESCALATE_WAIT = 15.0               # the one strong check of a trip
# words of a store tool's own name and description that lean towards placing an order, or towards ending without
# one; used only to tell finishing tools and their mode values apart from the schemas, never on the request
ORDER_STEMS = ("order", "buy", "purchas", "checkout")
END_STEMS = ("declin", "refus", "abstain", "cancel", "end", "without")

# ------------------------------------------------------------------------------------------------------------ budgets
SECONDS = 255.0                     # this trip's own time limit, under the host's
EASE_OFF = 0.25                     # share of the time left from which costly readings stop
HAND_OVER = 0.10                    # share of the time left at which code ends the trip itself
QUESTIONS_DEFAULT = 3               # questions assumed when the request states no message allowance
PAIR_BELOW = 0                      # pairing (two properties while the allowance is small) is off: a shopper given few messages turns such a question back
LIST_KEYS = 8                       # properties one question may name, likeliest first
LIST_FROM_CAP = 3                   # ... only with this many messages allowed: shoppers given fewer turn a list back, and that costs a message
KIT_SEARCHES = 3                    # searches per cart line for a listing of a tied item naming that line's model
LIST_LEAD = 3                       # list places counted as the front: a property named there is not led with again
OPENED_BEFORE_ASKING = 6            # listings of the item's label read with their own property sheet before planning
ASK_BY_STEP = 8                     # ... or this step (a quarter of the cap when that is later), whichever comes first
RESERVE_BASE, RESERVE_PER_ITEM = 5, 1   # finish reserve: the order, a note after it and a swap, plus 1 per extra item
COUNT_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
               "single", "pair", "couple", "dozen"}     # a count written as a word
GUARD_LIMIT = 2                     # each guard fires at most this often in one state (ledger version and cart)
GUARD_TRIP_LIMIT = 6                # ... and at most this often over the whole trip
GROUP_READ = 4                      # listings one compare call is first asked to read together
HUNT_ROUNDS = 3                     # searches per learned need for a listing that states it (each may fetch its next page)
SEARCH_STREAK = 5                   # searches in a row with rows left unread before the driver is asked to open some first
RAW_FINISH_TRIES = 1                # one mended call after a finish that neither ends nor is refused
FREE_READS = 20                     # repeated reads answered from the stored record, without a step
NEAR_KEY = " [near key]"            # marks a need met only under a key that spells the property otherwise
LOOSE = " [loose]"                  # marks a clash code could not settle that only fails once a model has read it
READ_NO = " [read]"                 # marks a need a model reading broke (code saw nothing against it)
NO_FIELD = " [no field anywhere]"    # marks a stated need whose property no opened listing of the item carries at all
NO_FIELD_PEERS = 3                  # ... counted once this many OTHER listings of the item are opened with their sheets
RESULT_CHARS = 15000
_proxy = ProxyClient(timeout=60, max_retries=1)
GIVE_UP_STATUS = (400, 401, 402, 403, 404, 409, 413, 422)
OPS = ("is", "is not", "at most", "at least", "one of", "has")
MOMENTS = ("before_add", "before_order", "before_decline", "after_change", "any")
ACTIONS = ("message", "open", "cart", "compare", "other")


def _post(path: str, body: dict[str, Any], until: float) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Send one request; while the gateway is busy or the answer is lost, send it again after a pause set by the
    request's size, until `until`. Each attempt's own timeout never runs past `until`."""
    pause, attempt = 1.0, 0
    offset = (len(json.dumps(body, default=str)) % 991) / 991.0
    while True:
        _proxy.timeout = max(3, min(60, int(until - time.monotonic())))
        result = _within(path, body, until)
        if result is None:
            return None, {"status": None, "kind": "deadline"}
        if result.data is not None:
            return result.data, None
        error = result.error or {}
        if error.get("status") in GIVE_UP_STATUS:
            return None, error
        attempt += 1
        wait = pause * (0.5 + ((offset + 0.618 * attempt) % 1.0))
        if time.monotonic() + wait >= until:
            return None, error
        time.sleep(wait)
        pause = min(pause * 2, 8.0)


def _within(path: str, body: dict[str, Any], until: float) -> Any:
    """One proxy call that is given up at `until` even when its reply trickles in: it runs in a worker thread and is
    waited for no longer than that. Returns the proxy's result, or None when the deadline came first."""
    box: dict[str, Any] = {}
    def work() -> None:
        try:
            box["result"] = _proxy.post_verbose(path, json_data=body)
        except Exception as exc:  # a failed call is an error result, retried like any other
            box["result"] = type("Failed", (), {"data": None, "error": {"status": None, "kind": type(exc).__name__}})()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(max(0.5, until - time.monotonic()))
    return box.get("result")


def _tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _alike(a: Any, b: Any) -> bool:
    return bool(_tokens(a)) and _tokens(a) == _tokens(b)


def _dump(value: Any, limit: int = RESULT_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + " ...[cut]"


def _trim(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value[:limit] + ("..." if len(value) > limit else "")
    if isinstance(value, list):
        return [_trim(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, limit) for k, v in value.items()}
    return value


def _object_in(text: str) -> dict[str, Any] | None:
    body = re.sub(r"^```[a-z]*|```$", "", (text or "").strip(), flags=re.M).strip()
    for candidate in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _strings(value: Any) -> list[str]:
    return [str(x) for x in value if str(x).strip()] if isinstance(value, list) else []


def _texts(value: Any) -> list[str]:
    """A reading's free text, whether it came back as one string or as a list of them."""
    return [value] if isinstance(value, str) and value.strip() else _strings(value)


def _figures(text: str) -> list[float]:
    """Numbers as a text writes them: a comma before exactly three digits groups thousands, any other comma or point
    marks decimals."""
    out = []
    for raw in re.findall(r"\d+(?:[.,]\d+)*", text):
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
            raw = raw.replace(",", "")
        elif raw.count(",") == 1 and "." not in raw:
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def _as_text_says(number: float, text: str) -> float:
    """A figure a model read from `text`: kept when the text writes it; put back to the written figure when the model
    moved the decimal point; otherwise the model's number stands (the text may spell it in words)."""
    written = _figures(text)
    if any(abs(w - number) < 1e-6 for w in written):
        return number
    for w in written:
        for scale in (100.0, 0.01, 1000.0, 0.001):
            if abs(w - number * scale) < 1e-6:
                return w
    return number


def _d(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _key_words(key: str) -> list[str]:
    return _tokens(re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " "))


def _is_id_key(key: str) -> bool:
    last = (_key_words(key) or [""])[-1]
    return last.endswith("id") and len(last) <= 4


def _owner(name: Any) -> str:
    """A seller's name without a trailing possessive."""
    return re.sub(r"['\u2019]s?$", "", str(name or "").strip())


def _fold(text: Any) -> str:
    return " ".join(_tokens(text))


def _keyfold(key: Any) -> str:
    """A field key as plain words: one spelling for camelCase, snake_case and spaced keys."""
    return " ".join(_key_words(str(key or "")))


def _coded(values: list[str]) -> bool:
    """Values that name the listing rather than describe it: one long letters-then-digits token per value (part or
    model numbers), nearly every listing its own."""
    distinct = list(dict.fromkeys(re.sub(r"[\-_./]", "", str(v).strip().lower()) for v in values if _fold(v)))
    codes = [v for v in distinct if v.isalnum() and len(v) >= 6 and re.search(r"[a-z].*\d", v)]
    return len(values) >= 3 and len(codes) * 10 >= 8 * len(distinct) and len(distinct) * 10 >= 8 * len(values)


def _no_choice(values: list[str]) -> bool:
    """Values a shopper does not choose between: most list several options at once ("a / b / c"), or nearly every
    listing has its own value of three or more words (a model's name, a list of what it suits), or none says
    anything."""
    said = [str(v).strip() for v in values if _fold(v)]
    if not said:
        return True
    def mixed(v: str) -> bool:                 # parts of several kinds: some with figures, some without
        parts = re.split(r"\s+/\s+|\s*\|\s*", v)
        return len(parts) >= 3 and any(re.search(r"\d", x) for x in parts) and any(not re.search(r"\d", x) for x in parts)
    lists = [v for v in said if mixed(v)]
    if len(lists) * 2 > len(said):
        return True
    distinct = list(dict.fromkeys(_fold(v) for v in said))
    long = [v for v in distinct if len(v.split()) >= 3]
    return len(said) >= 4 and len(distinct) * 10 >= 8 * len(said) and len(long) * 10 >= 7 * len(distinct)


def _prop_name(key: str) -> str:
    """A field key as a plain property name: its words, from the first word that carries meaning."""
    words = [w for part in str(key).split(".") for w in _key_words(part)]
    lead = next((i for i, w in enumerate(words[:-1]) if len(w) > 2 or w.isdigit()), max(0, len(words) - 1))
    return " ".join(words[lead:])


def _key_set(text: Any, keep: set[str] = frozenset()) -> set[str]:
    """A key's content words, plural folded, for telling two spellings of one property apart from two properties;
    short words count when `keep` names them (a property's own short words are usually its name, not glue)."""
    words = _keyfold(text).split()
    out = {w.rstrip("s") for w in words if len(w) > 2 or w.isdigit() or w in keep}
    return out or set(words)


def _field(sheet: dict[str, str], prop: str) -> str | None:
    """A sheet's value for a property: the field with the same folded name, else the only field whose name just adds
    or drops words."""
    key = _keyfold(prop)
    if not key:
        return None
    shown = next((v for k, v in sheet.items() if _keyfold(k) == key or _fold(k) == _fold(prop)), None)
    if shown is not None:
        return shown
    short = {w for w in _keyfold(prop).split() if len(w) <= 2}
    want = _key_set(prop, short)
    near = [v for k, v in sheet.items() if _key_set(k, short) and want
            and (_key_set(k, short) <= want or want <= _key_set(k, short))]
    return near[0] if len(near) == 1 else None


def _ids_in(value: Any, known: set[str]) -> list[str]:
    """Listing ids that a call's arguments name, in order, however deeply the arguments nest them."""
    out: list[str] = []
    if isinstance(value, dict):
        for v in value.values():
            out += _ids_in(v, known)
    elif isinstance(value, list):
        for v in value:
            out += _ids_in(v, known)
    elif isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) in known:
        out.append(str(value))
    return list(dict.fromkeys(out))


def _has_list(value: Any) -> bool:
    return isinstance(value, dict) and any(isinstance(v, list) and v and isinstance(v[0], dict) for v in value.values())


def _all_texts(value: Any) -> list[str]:
    """Every non-empty string a value holds, however deeply dicts and lists nest it."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [x for v in value.values() for x in _all_texts(v)]
    if isinstance(value, list):
        return [x for v in value for x in _all_texts(v)]
    return []


def _has_entries(value: Any) -> bool:
    """True when a value holds a non-empty list anywhere (entries of some kind)."""
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return any(_has_entries(v) for v in value.values())
    return False


def _sentences(value: Any) -> list[str]:
    """Sentences a value holds: a string of four words or more, or a list of strings with such sentences."""
    if isinstance(value, str):
        return [value] if len(value.split()) >= 4 else []
    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
        return [x for x in value if len(x.split()) >= 4]
    return []


_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9_ ]{0,39}?)\s*(?::\s|:$)(.*)", re.S)
_INNER_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9_ ]{0,39}?)\s*=(.*)", re.S)


def _split_pairs(text: str, sep: str, pair: "re.Pattern[str]") -> list[list[str]] | None:
    """`text` cut at `sep` into key/value pairs; a piece that does not open with a key belongs to the value before
    it (a title may hold the separator itself). None unless the text opens with a key."""
    out: list[list[str]] = []
    for piece in text.split(sep):
        m = pair.fullmatch(piece.strip())
        if m:
            out.append([m.group(1).strip(), m.group(2).strip()])
        elif out:
            out[-1][1] += sep + piece
        else:
            return None
    return out


def _scalar(text: str) -> Any:
    """A value written as text: the words a store uses for none, yes and no, and plain numbers, become those."""
    if text in ("None", "null", ""):
        return None
    if text in ("True", "False", "true", "false"):
        return text.lower() == "true"
    if re.fullmatch(r"-?(?:0|[1-9][0-9]{0,14})", text):
        return int(text)                       # a figure written with a leading zero is a code: it stays text
    if re.fullmatch(r"-?(?:0|[1-9][0-9]{0,14})\.[0-9]+", text):
        return float(text)
    return text


def _record_of_line(text: Any) -> dict[str, Any] | None:
    """A record a store writes as text, "key: value" pairs cut by " | ", by new lines or by "; ", as a dict; a value
    that is itself a run of "key=value; key=value" becomes a dict of those. None for any other text."""
    if not isinstance(text, str) or len(text) < 12:
        return None
    for sep in (" | ", "\n", "; "):
        if sep not in text:
            continue
        pairs = _split_pairs(text.strip(), sep, _PAIR)
        if not pairs:
            continue
        merged: list[list[str]] = []
        for key, value in pairs:               # a key met again sits inside a value (a blurb quoting one): it joins it
            if merged and any(key == k for k, _ in merged):
                merged[-1][1] += "%s%s: %s" % (sep, key, value)
            else:
                merged.append([key, value])
        if len(merged) < 3:
            continue
        record: dict[str, Any] = {}
        for key, value in merged:
            inner = _split_pairs(value, "; ", _INNER_PAIR) if "=" in value else None
            if inner and len(inner) >= 2:
                record[key] = {k: v.strip() for k, v in inner}
            else:
                record[key] = _scalar(value)
        return record
    return None


def _cells(line: str, sep: str) -> list[str]:
    body = line.strip()
    if sep == "|":
        body = body.strip("|")
    return [c.strip() for c in body.split(sep)]


def _table(lines: list[str], seps: tuple[str, ...] = ("|", ",", "\t")) -> list[dict[str, Any]] | None:
    """Rows a store writes as a table, a header line of names and then one line per row, cells cut by "|" (a
    rule line of dashes may follow the header) or by ","; None unless every row has the header's number of cells."""
    lines = [x for x in lines if isinstance(x, str) and x.strip()]
    if len(lines) < 3:
        return None
    for sep in seps:
        head = _cells(lines[0], sep)
        if len(head) < 3 or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_ ]{0,39}", h) for h in head) or len(set(head)) < len(head):
            continue
        body = [x for x in lines[1:] if not re.fullmatch(r"[\s|:+-]*-{3,}[\s|:+-]*", x.strip())]
        rows = [_cells(x, sep) for x in body]
        if len(rows) >= 2 and all(len(r) == len(head) for r in rows):
            return [{h: _scalar(v) for h, v in zip(head, r)} for r in rows]
    return None


def _as_records(value: Any) -> Any:
    """A result with the records it writes as text (pairs, tables, JSON inside a string) written out as dicts, at any
    depth; a wrapper whose only part is such a record is the record itself."""
    if isinstance(value, str):
        body = value.strip()
        if body[:1] in ("{", "[") and body[-1:] in ("}", "]"):
            try:
                return _as_records(json.loads(body))
            except ValueError:
                pass
        record = _record_of_line(value)
        if record is not None:
            return record
        if "=" in body and "; " in body:
            block = _split_pairs(body, "; ", _INNER_PAIR)
            if block and len(block) >= 2:
                return {k: _scalar(v.strip()) for k, v in block}
        if "\n" in body:
            rows = _table(body.split("\n"))
            if rows is not None:
                return rows
            lines = [x for x in body.split("\n") if x.strip()]
            if len(lines) >= 3 and all(x.count("|") == 1 for x in lines):
                kept = [[c.strip() for c in x.split("|")] for x in lines[1:] if not re.fullmatch(r"[\s|:+-]*-{3,}[\s|:+-]*", x.strip())]
                return {a: _scalar(b) for a, b in kept}
        return value
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(v, dict) and set(v) == {"name", "value"} for v in value):
            return {str(v["name"]): v["value"] for v in value}
        rows = _table(value, ("|", "\t")) if value and all(isinstance(v, str) for v in value) else None
        return rows if rows is not None else [_as_records(v) for v in value]
    if isinstance(value, dict):
        out = {k: _as_records(v) for k, v in value.items()}
        if len(out) == 1:
            key, only = next(iter(out.items()))
            if isinstance(only, dict) and isinstance(value[key], str) \
                    and (_key_words(key) or [""])[0] not in ("error", "errors"):
                return only                    # an error stays an error, whatever its text looks like
        return out
    return value


def _money(value: Any) -> float | None:
    """A price as a number: a plain number, or text that is one figure written as money: with a currency sign or a
    three-letter currency code after it, or with decimals ("$14.95", "14.95 USD", "14.95"); None otherwise."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    m = re.fullmatch(r"\s*([^\w\s]{1,3}\s?)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?(\s?[^\w\s]{1,2}|\s[A-Z]{3})?\s*", value)
    if not m:
        return None
    marks = (m.group(1) or "") + (m.group(4) or "")
    sign = any(unicodedata.category(ch) == "Sc" for ch in marks)
    code = bool(re.fullmatch(r"\s*[A-Z]{3}\s*", m.group(4) or ""))
    if marks.strip() and not (sign or code):
        return None                            # some other mark or unit beside the figure: not money
    if not (sign or code or m.group(3)):
        return None                            # a bare whole number written as text: an id, a year, a count
    return float(m.group(2).replace(",", "") + (m.group(3) or ""))


def _code_like(value: Any) -> bool:
    """A short token with no spaces that holds a digit: how a store writes a listing's own reference."""
    return isinstance(value, str) and 3 <= len(value) <= 64 and not re.search(r"\s", value) and bool(re.search(r"\d", value))


def _notes_beside(observation: Any) -> tuple[Any, list[str]]:
    """A result that carries notes holds, beside the result, lists of sentences (whatever the keys are called), or is
    a two-part wrapper of one sentence and the result; the rest is the result itself, unwrapped when only one part
    remains. A listing's own record (a dict with an id) is never split."""
    if isinstance(observation, dict) and len(observation) >= 2 \
            and not any(_is_id_key(k) and isinstance(v, (str, int)) for k, v in observation.items()):
        lists = [k for k, v in observation.items() if isinstance(v, list) and v
                 and all(isinstance(x, str) for x in v) and any(len(x.split()) >= 4 for x in v)]
        if lists and len(lists) < len(observation):
            rest = {k: v for k, v in observation.items() if k not in lists}
            inner = next(iter(rest.values())) if len(rest) == 1 else rest
            return inner, [x for k in lists for x in observation[k] if len(x.split()) >= 4]
        if len(observation) == 2:
            said = [k for k, v in observation.items() if isinstance(v, str) and len(v.split()) >= 4
                    and re.search(r"[.!?]\s*$", v.strip())]
            other = [k for k in observation if k not in said]
            if len(said) == 1 and len(other) == 1 and isinstance(observation[other[0]], (dict, list)):
                return observation[other[0]], [observation[said[0]]]
    return observation, []


def _roles_from_rows(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Field roles from the rows' shape alone, when no reading gives them: the id-shaped key; the one key that is a
    yes/no in every row (stock); the one number in every row (price); of the short text keys, the one with the
    fewest values (category) and the one with the longest values (title)."""
    rows = [e for e in entries if isinstance(e, dict)]
    if len(rows) < 2:
        return {}
    keys = [k for k in rows[0] if all(k in r for r in rows)]
    def every(test: Any) -> list[str]:
        return [k for k in keys if all(test(r[k]) for r in rows)]
    out: dict[str, str] = {}
    ids = [k for k in keys if _is_id_key(k) and all(isinstance(r[k], (str, int)) for r in rows)]
    if not ids:
        # no key named as an id: the key whose value is a distinct reference token in every row
        ids = [k for k in keys if all(_code_like(r[k]) and _money(r[k]) is None for r in rows)
               and len({r[k] for r in rows}) == len(rows)]
        ids = ids if len(ids) == 1 else []     # two such keys: the shape alone cannot tell which one names the listing
    flags = every(lambda v: isinstance(v, bool))
    numbers = [k for k in every(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)) if k not in ids]
    # a key written as money with a sign or a code in every row is the price, before any plain number (a rating, a
    # count of units in stock)
    marked = [k for k in every(lambda v: isinstance(v, str) and _money(v) is not None and bool(re.search(r"[^\d\s.,]", v)))
              if k not in ids]
    if len(marked) == 1:
        numbers = marked
    elif len(numbers) != 1:
        # no single number in every row: the price is the one key written as money in every row (a currency sign
        # or code, or decimals), else, among several numbers, the one with decimals
        money = [k for k in every(lambda v: _money(v) is not None) if k not in ids and k not in numbers]
        if len(money) == 1:
            numbers = money
        elif len(numbers) > 1:
            cents = [k for k in numbers if any(isinstance(r[k], float) for r in rows)]
            numbers = cents if len(cents) == 1 else numbers
    texts = [k for k in every(lambda v: isinstance(v, str) and 0 < len(v) <= 150) if k not in ids and k not in numbers]
    if ids:
        out["id"] = ids[0]
    if len(flags) == 1:
        out["in_stock"] = flags[0]
    if len(numbers) == 1:
        out["price"] = numbers[0]
    if texts:
        spread = {k: len({r[k] for r in rows}) for k in texts}
        width = {k: sum(len(r[k]) for r in rows) / len(rows) for k in texts}
        title = max(texts, key=lambda k: (width[k], spread[k]))
        out["title"] = title
        labels = [k for k in texts if k != title and width[k] <= 60 and spread[k] < len(rows)]
        if labels:
            out["category"] = min(labels, key=lambda k: (spread[k], -width[k]))
    return out


def _where(exc: BaseException) -> str:
    """An exception's type and the line it came from, for the trip's own record."""
    tb, line = exc.__traceback__, 0
    while tb is not None:
        line, tb = tb.tb_lineno, tb.tb_next
    return "%s at line %d" % (type(exc).__name__, line)


def _fault(observation: Any) -> str:
    """A result made of one code-like token (no spaces; containing a digit or "_") is a fault."""
    inner = _notes_beside(observation)[0]
    if isinstance(inner, dict) and len(inner) == 1:
        value = next(iter(inner.values()))
        if isinstance(value, str) and value and len(value) <= 40 and not re.search(r"\s", value) \
                and re.search(r"[\d_]", value):
            return value
    return ""


def _failed(observation: Any) -> str:
    """A fault, or a result whose leading key names an error: the call did not do its job."""
    problem = _fault(observation)
    if problem:
        return problem
    inner = _notes_beside(observation)[0]
    if isinstance(inner, dict):
        for k, v in inner.items():
            if (_key_words(k) or [""])[0] in ("error", "errors") and v not in (None, False, "", []):
                return str(v)[:80]
    return ""


# ------------------------------------------------------------------------------------------------------------ arguments
def _fit(schema: dict[str, Any], value: Any, label: str, issues: list[str]) -> Any:
    """`value` brought into line with `schema` where that is safe; what cannot be is added to `issues`."""
    if "enum" in schema:
        if value in schema["enum"]:
            return value
        match = next((m for m in schema["enum"] if _alike(m, value)), None)
        if match is None:
            issues.append("%s must be one of %s" % (label, schema["enum"]))
            return value
        return match
    kind = schema.get("type")
    if kind in ("integer", "number"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            issues.append("%s must be a number" % label)
            return value
        if kind == "integer":
            if not number.is_integer():
                issues.append("%s must be a whole number" % label)
                return value
            number = int(number)
        if "minimum" in schema and number < schema["minimum"]:
            issues.append("%s is below %s" % (label, schema["minimum"]))
        if "maximum" in schema and number > schema["maximum"]:
            issues.append("%s is above %s" % (label, schema["maximum"]))
        return number
    if kind == "boolean":
        return value.lower() == "true" if isinstance(value, str) else bool(value)
    if kind == "string":
        text = value if isinstance(value, str) else json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        if "maxLength" in schema and len(text) > schema["maxLength"]:
            head = text[: schema["maxLength"]]
            text = head.rsplit(" ", 1)[0] if " " in head.strip() else head
        if len(text) < schema.get("minLength", 0):
            issues.append("%s is too short" % label)
        return text
    if kind == "array":
        items = value if isinstance(value, list) else [value]
        sub = schema.get("items") or {}
        items = [_fit(sub, v, "%s item" % label, issues) for v in items]
        if len(items) < schema.get("minItems", 0):
            issues.append("%s needs more items" % label)
        if "maxItems" in schema and len(items) > schema["maxItems"]:
            issues.append("%s has too many items" % label)
        return items
    if kind == "object" or "properties" in schema:
        if not isinstance(value, dict):
            issues.append("%s must be an object" % label)
            return value
        props = schema.get("properties") or {}
        out = {}
        for k, v in value.items():
            if v is None:
                continue
            if k in props:
                out[k] = _fit(props[k], v, k, issues)
            elif schema.get("additionalProperties") is False:
                issues.append("there is no argument %r" % k)
            else:
                out[k] = v
        for k in schema.get("required") or []:
            if k not in out:
                issues.append("argument %r is required" % k)
        return out
    return value


# ------------------------------------------------------------------------------------------------------------ prompts
SETUP_READING = """Read the tools of an online store and a shopper's request. Return only one JSON object:
{"tools": {"find": the tool that searches listings,
  "readers": [every tool that shows one listing by its id: its page, its fields, its stock or its price],
  "compare": a tool that lines up several listings side by side, or null,
  "speak": the tool that sends the shopper a message, or null,
  "add": the tool that puts a listing in the cart,
  "cart": the tool that shows the cart, or null,
  "order": the tool that places the order,
  "decline": the tool that ends the trip without buying, or null,
  "add_args": {argument: value} fixed values the add tool needs to put a listing in (never the ones that take it out),
  "order_args": {argument: value} fixed values that make the order tool place the order,
  "decline_args": {argument: value} fixed values that make the decline tool end without buying,
  "find_args": {"words": the argument for search words, "label": the argument naming a category, "page": the argument
   for a page or an offset, "price_cap": the argument for a highest price}, each null when the tool lacks it},
 "items": [{"name": the product as the request names it (a store section's name when the request names the product
  only by the section it is kept in), "qty": units wanted (a pair or a set sold as one listing is one unit), "store_label": the store's own category
  name for it when a tool lists such names, else null, "cap": the highest price for one unit of this item when the
  request states one, else null, "with": the name of another item of this list that the request says this one must
  fit, work with, go with or be for (an accessory, a part or a cover for a device bought alongside), else null}],
 "needs": [{"item": the item's name, "property": the property, "op": one of "is", "is not", "at most", "at least",
  "one of", "has", "value": the needed value as written (a list for "one of"), "when": null, or for a need that holds only
  in some case {"item": the item the case is about (this item, or the item of the list whose choice it depends on),
  "property": the property the case looks at ("price" for what a listing costs), "op": one of the ops above, "value": the
  value that case looks for, "holds": true when the need applies if the case is true, false when it applies unless it
  is true}}],
 "objective": {"kind": "cheapest" | "best" | "any", "criterion": for "best" the property to make highest or lowest,
  else "", "direction": "max" | "min" | ""},
 "budgets": {"money_cap": the most everything together may cost, as a number, or null; "questions_cap": how many
  messages the shopper will take, or null when the request does not say; "max_steps": a step allowance the request
  states, or null},
 "in_stock": true unless the request accepts listings that are not available now,
 "procedures": {"rules": [every sentence of the request about how to shop or what to do at some moment, copied as
  written], "steps": [{"moment": "before_add" | "before_order" | "before_decline" | "after_change" | "any",
  "action": "message" | "open" | "cart" | "compare" | "other", "detail": what to do, in a few words}]},
 "decline_rule": the request's sentence about what to do when nothing fits, copied as written, or null,
 "story_hints": [{"hint": a detail of the shopper's story that bears on what they will want, "property": the listing
  property it bears on}],
 "no_preference": [properties the request says do not matter],
 "references": [{"item": the item's name, "property": the property, "relation": "same" | "compatible" | "not",
  "seller": the seller of the other listing as named, "category": its category as named, "marks": [the values that
  pick that listing out]}]}
Rules: a tool that does two jobs is named for both, with the fixed values that tell them apart. A store section's name
that lists several kinds of goods ("A, B & C", "X and Y parts") is ONE item, the section, bought as one listing from
it, unless the request plainly asks for one of each kind. A seller's or shop's name that says whose listing to look at
for a value ("the same size as the one sold by X") belongs to a reference, never to a second item; a maker's name that
describes the item wanted stays in the item's name or a need. Needs leave out price,
stock, quantity, soft wishes and the words of the item's own name; a range is two needs ("at least" and "at most").
"cheapest" only when the shopper plainly asks for the lowest price; a low limit alone is not that. Background about
the shopper is not a need. A property whose value is not named but tied to what another listing shows (a seller's
listing, picked out by its category and some of its values) goes under "references", never as a need with those
words as its value; the values that pick out the other listing describe that listing, not the item. A step is
something the request asks the assistant to do; what the shopper says they will do
themselves is not one, and what to say when ending without buying is no message step (the ending call carries it).
A need that depends on a case keeps that case in its "when", whatever words introduce it (a condition, an exception, what
may turn out), and each branch of such a rule is a need of its own; a case the request itself already settles is applied
now, and only the need that follows from it is written.
Tool and argument names exactly as the schemas spell them."""

FIELD_ROLES = """Below is a raw result from an online store's tool. Say which keys of its listing entries hold each item
below (exact key spelling, or null when absent), and which top-level keys describe the search as a whole. Return only:
{"id": key, "price": key, "in_stock": key, "category": key, "title": key, "match_count": key with the total number of
matches, "next_page": key whose value is passed to get the next page}"""

FINISH_VALUES = """One store tool both places an order and ends a trip without buying; an argument with fixed choices
selects which. Given the tool's description and that argument's choices, return only {"buy": the choice that places the
order, "none": the choice that ends without buying}."""

CART_ARGS = """Return only the JSON arguments for the store tool below so that one call puts the given listing in the
cart at the given quantity (when the tool sets the whole cart from a list of lines, send just that one line)."""

LISTING_CHECK = """Does one store listing meet a shopper's needs? For each need below decide from the listing's own
records only; later records replace earlier ones. A need is met when the listing has a field for that property that
carries the value: field names vary, one entry of a list is enough, plain synonyms and singular or plural are the same;
a sheet value that adds or changes a word (another colour, finish, size or variant) is a different value, and so is a
broader or a different group; in a title or description a narrower form of the value counts. Title and description count only
for a property the listing has no field for, and a field for some other property is no evidence. Compare figures
as numbers in one unit; a bound is met at its edge. "values_elsewhere" shows what other listings of this store carry
for each property. Return only {"needs": [{"n": the need's number, "fits": "yes" | "no" | "unknown", "why": short}]}."""

SHOPPER_LINES = """During a shopping trip the shopper, or the store on the shopper's behalf, just said something. Given
the request, the current ledger and the question that was asked (if any), return only this JSON object:
{"needs": the complete current list of needs [{"item", "property", "op", "value", "when" (as the ledger has it: the
  case a need depends on, kept whenever the words attach one, whatever words introduce it)}] with the ledger's ops,
 "withdrawn": [properties whose need the shopper dropped],
 "no_preference": [properties the shopper says they do not mind about],
 "objective": {"kind": "cheapest" | "best" | "any", "criterion", "direction"} as it now stands,
 "money_cap": the highest total now allowed (a number), or null when the words do not change it,
 "steps": [new things the shopper asks to be done at some moment: {"moment", "action", "detail"}],
 "events": [{"kind": "price" (a listing's price changed) | "gone" (a listing can no longer be bought) | "other",
  "listing": its id when the words name one, else ""}],
 "answered": true when the words answer the question asked, false when they do not or leave it to us,
 "reply": "answered" (a value or a need is given), "no_need" (the shopper does not mind, or says nothing about it),
  "one_at_a_time" (instead of answering, the shopper complains that too many things were asked and wants each property
  in its own message), or "confused" (the shopper asks to repeat, rephrase or simplify the question instead of answering
  it),
 "references": [{"item", "property", "relation": "same" | "compatible" (it must go with or pair with what that listing
  shows) | "not" (anything but what that listing shows), "seller", "category", "marks": [the values that pick that
  listing out], "described": the shopper's own words describing that listing}] for a property the shopper ties to what
  another listing shows instead of naming its value (such a property goes here only, never under needs)}
Words of insistence around a value never set the comparison, however strongly they are put, even if they sound like a minimum: such a need has "op" "is". Use "at least" or "at most" only when the reply itself says so about the number. A need already in the ledger keeps its op. A plainly named value is firm however casually it is said, even when it comes inside a question, a comparison or an
aside: words that tie a value to a property give a need (words tying it to another listing go under references
instead), and "confused" fits only words that give no value at all. A
need's value is the shopper's own words for it and nothing around them (not the item's name, not the words that tie
the value to its property); a need's property is the one the shopper's words name, spelled as they spell it, even when it differs from the property
asked about. A bound is written with "at most" or "at least" and its figure. Keep every earlier need that was not
withdrawn or replaced. Needs never include price, stock or quantity."""

CHECKLIST = """An assistant on a shopping trip is about to %s. Below are the request's own rules about how to shop,
the steps read from them, the shopper's later words and the calls made since the last change. List only the steps the
rules ask for before this move that have not been done yet. Add nothing the rules do not ask for, and nothing they ask
for at another moment. Return only {"steps": [{"action": "message" | "open" | "cart" | "compare", "about": "cart" |
"listing" | "other", "text": the words to send when the message is more than a summary of the cart, else ""}]}, or
{"steps": []} when nothing is owed."""

RAW_FINISH = """A call to an online store's tool, meant to %s, came back without doing it. Given the tool's schema, the
arguments sent and the result, return only the JSON arguments for one corrected call to the same tool."""

DRIVER_GUIDE = """You work an online store's tools for a shopper who wants to buy the items on their list. Every tool
call uses one step and steps are limited; thinking is free.

Your part is to find listings that fit and to open them. Code does the rest around you:
- it keeps the shopper's needs in a ledger (the STATUS shows it) and sends the questions to the shopper for you;
- before a listing goes in the cart, code checks it against the ledger and says so when it does not fit;
- once every item has a cart line that fits, code looks at the cart, takes the steps the request asks for and places
  the order; when you decide the store has nothing that can work, code ends the trip without buying.

How to work:
1. Search with two or three words: the item's own name plus one needed value. A category argument takes only a name
   the store itself lists.
2. Open promising rows with a tool that shows one listing in full. A listing's own fields decide whether it fits;
   title and description count only for a property the listing has no field for.
3. Put one listing per item in the cart, at that item's quantity. With no rule to take the cheapest or the best, the
   first listing that meets every need is the one to buy: do not look for lower prices.
4. When the shopper or the store announces a change (a new price, a new limit, an item gone, a need dropped), the
   STATUS shows the new ledger; follow its advice.
5. End without buying only after several different searches found nothing that fits within the money limit.
Reply with tool calls; keep any text short."""

DECLINE_WORDS = """A shopping trip ends without buying because nothing in the store meets the shopper's needs. Write
the short message that ends it, the way the shopper's own rule for this case (given below) asks, saying what was
missing when the rule allows it. Name no listing and no listing id. Return only {"text": the message}."""

PLAIN_DECLINE = "Nothing in this store meets every need within the limit, so I am not placing an order."

LAST_LOOK = """A shopping trip is about to end without a purchase. You get the shopper's request, the ledger (items, the
needs stated by the request or the shopper, the money limit, how to choose), the records of the listings opened so far
and the titles of rows seen in searches but not opened. Reply with JSON only:
{"fits": [{"item": the item's name, "listing": the id of an opened listing whose record meets every need of that item,
 the shopper's revealed needs included, within the money limit}],
 "open": [ids of up to three unopened rows whose titles suggest they could meet an item's needs when no opened listing does]}
Name a fit only when the record shows it; a field that names the property decides it, a value that adds a word to the
needed value is another value, a related field decides nothing, the title counts only when no field names the property."""

FINAL_LOOK = """You look over a shopping cart against a shopper's request just before the order goes out. You get the request,
the ledger (items, the needs the request or the shopper stated, the money limit, how to choose between listings that
fit), the records of the listings in the cart and the records of other listings seen in the store. Reply with JSON only:
{"ok": true when every cart line is a listing that meets every need of its item, the shopper's revealed needs
 included, and the way of choosing (cheapest, a measure, or any) does not point at another seen listing instead;
 "problems": [short reasons, each naming the cart line's id and the need it misses],
 "better": [{"item": the item's name, "listing": the id of a seen listing that meets the item's needs better}]}
How to read a record: a field that names the property decides it, in any spelling; a value that adds a word to the
needed value (another colour, finish, size or variant) is another value; a field for a related property decides
nothing; the title and description count only when no field names the property; a listing with no field for a
property the shopper named is weaker than one whose field shows the named value; a bare figure without its measure
says nothing; units are converted; a range counts when it lies wholly inside the needed one. Name a better listing
only when its record shows it meets every need the cart line meets and more, or costs less under a cheapest rule."""

REF_READ = """A shopper's need takes the value of one property from another store listing. Given that property, how it is
tied ("same": the item must show what that listing shows for it; "compatible": it must go together with it; "not": it
must differ), the description of the listing pointed at, and the records of a listing the store returned for that
description, work through these in order and return only:
{"this_is_it": true when the records fit the description (seller, category, values),
 "sends_on": the records' own sentence that sends this property to another listing, in any wording (one that tells the
  reader to look elsewhere for it), or null,
 "points_to": {"seller": the other listing's seller name, or "" when none is given, "category": ..., "marks": [values
  that pick it out], "property": this property's name as that sentence spells it} for that other listing, or null,
 "turns": "same" when the sentence takes the other listing's value as it is, "not" when it says anything but that
  value, "compatible" when it says to pair with that value, or null when no sentence sends the property on,
 "shown": the value these records show for this very property, or null,
 "pairing": the records' own sentence that pairs values of this property (one value goes together with another), or null,
 "value": the value the item must have: null when sends_on is set; else "shown", with the pairing applied when there
  is one (the value paired with the shown one is the answer), or the value that goes with it for a "compatible" tie}
Only what the records say about this very property counts (under its own name, or under "also_called" when that is
given); what they say about other properties does not. A direction to look elsewhere overrides any value the page
shows for the property."""

PAIR_APPLY = """A store listing shows one value for a property and states a sentence that pairs values of that property.
Return only {"value": the value the sentence pairs with the shown one, or the shown value itself when the sentence
pairs nothing with it}."""

RULE_WAY = """Below is a shopper's sentence about what to do when nothing in the store fits. Return only {"way": "end"
when it asks to end without buying (or only to be told so), "other" when it asks to buy some other listing in that
case, or anything else}."""

STATED_AUDIT = """Below are a shopper's request, the items it asks for, and for each item some properties of the store's
listings, each with values of it that the request's own words include. For each property, say whether the request itself states
what that property must, or must not, be for an item. Return only {"needs": [{"item": the item's name, "property": the
property as given, "op": "is" | "is not" | "one of", "value": the value as the store spells it (a list for "one of")}]},
an empty list when the request states none. Background about the shopper and words used in passing are not needs."""

PLAN_QUESTIONS = """You rank what a shopping assistant should name to a shopper. Besides what the request states, the
shopper holds a few firm preferences they did not write down, usually one per product at most, each about one property.
A preference comes out only when a question names that very property, and the property named first is by far the most
likely to be answered. So what matters most is which property goes first.

You get the request, what is already settled per item, hints read from the story, and per item the properties seen on
its opened listings, with how many listings carry each and example values.

Reason from the story. Look for what it implies but leaves open: something that went wrong before, a thing the product
has to fit or work with, who will use it, where it goes, how it has to look, a detail the writer cannot recall or leaves
to you. Each points at a property whose value the story needs and does not give. Rank a property the story bears on above
one that buyers merely care about in general. When the story bears on nothing for an item, rank what most separates one
listing of that product from another for its buyers, preferring properties carried by more listings.

Leave out properties already settled, as well as ids, model names, makers, sellers, warranty and delivery. A property
that only shares a word with a settled one is a different property and stays eligible. Use the listed spelling, in plain
words, one property per entry, and never join two with "and" or "or".

Reply with JSON only: {"ask": [{"item": item number, "label": "property", "hint": "the words of the story it comes
from, or empty"}]} with up to COUNT entries per item, each item's likeliest first."""


# ------------------------------------------------------------------------------------------------------------ evidence

def _text_values(value: Any) -> list[str]:
    """Every string a record holds, keys left out, one per value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _text_values(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _text_values(v)]
    return []


def _in_series(code: tuple[str, str], series: tuple[str, str]) -> bool:
    """True when a plain model code falls in a series written with a trailing run of x ("ABC 20xx" takes "ABC 2030")."""
    word, digits = series
    if "x" not in digits or "x" in code[1] or code[0] != word:
        return False
    stem = digits.rstrip("x")
    return len(code[1]) == len(digits) and code[1].startswith(stem)


LOW_OPENS = 4                       # cheapest rows opened per item once a walk has found them

LOW_PAGES = 5                       # pages one search of a walk may read before the walk moves on to narrower words

LOW_STALL = 3                       # searches in a row without a new row of the item that end a walk

LOW_PLAIN_STEPS = 6                 # walk steps per item when no way of choosing is named (a price rule gets all there are)

SPLIT_PICKS = 8                     # splitting values tried per item after the plain words

AUDIT_OPENS = 4                     # unread rows opened per item before a trip may end without buying

TRIP_CEILING = 0.050                # dollars a trip may spend on models; past it, code finishes the trip alone

RESENDS = 3                         # orders sent again after a note, each after the cart was looked at or changed

CHECKOUT_NOTE = """A shopping assistant was told something by a store while filling a cart or placing an order. Say what the
words mean for the cart. Return only {"kinds": [any of "sold_out" (a listing can no longer be bought), "price_up", "price_down",
"need_dropped" (the shopper no longer asks for something), "need_new" (a new or changed wish), "budget" (a new spending limit),
"other"], "listing": the id of the listing the words name, else ""}."""

def _low_summary(ledger: "Ledger") -> str:
    """The ledger's summary, saying what is bought when the request names no way of choosing."""
    return _low_summary.__wrapped__(ledger).replace("any listing that fits (no price hunting)",
                                                     "the cheapest listing that fits")

SHARED_SHEETS = 4                   # listings read per item when the list has more than two items

TIE_HOPS = 6                        # listings read along one tie, the first one included

TIE_MISSES = 3                      # listings along a tie that say nothing plain before the tie is given up

READ_AGAIN_FLOOR = 20.0             # seconds of slack that a second reading of an empty answer needs

def _when_of(raw: Any, names: list[str]) -> dict[str, Any] | None:
    """The case a need depends on, in the ledger's form, or None when it names no property or no value."""
    if not isinstance(raw, dict):
        return None
    prop, value = str(raw.get("property") or "").strip(), raw.get("value")
    op = next((o for o in OPS if _fold(o) == (_fold(raw.get("op")) or "is")), "is")
    if isinstance(value, list):
        value = [str(x).strip() for x in value if str(x).strip()]
    elif value is not None:
        value = str(value).strip()
    if not prop or value in (None, "", []):
        return None
    said = str(raw.get("item") or "")
    item = next((n for n in names if said and (_alike(n, said) or _fold(n) in _fold(said) or _fold(said) in _fold(n))), "")
    return {"item": item, "property": prop, "op": op, "value": value, "holds": raw.get("holds") is not False}

TIE_SEARCH = """A shopper's need takes its value from a store listing described below. Give the words to search the store
for that listing. A store matches every word, so keep each search short: the seller's or maker's name with the kind of
product, then the name with one distinctive value, then the kind with two distinctive values.
Return only {"searches": ["words", "words", "words"]}."""

def _first_rows(value: Any) -> list[dict[str, Any]] | None:
    """The first list of dicts a result holds, at any depth."""
    if isinstance(value, list):
        return value if value and all(isinstance(x, dict) for x in value) else None
    if isinstance(value, dict):
        for v in value.values():
            rows = _first_rows(v)
            if rows:
                return rows
    return None

class Index:
    """Everything the store has shown. Raw results are kept so the listing table can be rebuilt once the store's field
    roles are known."""

    def __init__(self) -> None:
        self.raw: list[Any] = []
        self.roles: dict[str, str] = {}
        self.records: dict[str, list[str]] = {}      # listing id -> compact records that mention it
        self.rows: dict[str, dict[str, Any]] = {}    # listing id -> price, stock, category, title
        self.opened: set[str] = set()                # listings whose own record was returned
        self.ids: set[str] = set()
        self.category_counts: dict[str, int] = {}
        self.sheet_of: dict[str, dict[str, str]] = {}  # listing id -> its own property sheet
        self.text_of: dict[str, str] = {}             # listing id -> the words its records show
        self.seen_order: list[str] = []              # listing ids in the order first seen
        self.row_keys: set[str] = set()              # plain keys of listing rows in search results
        self.from_search = False

    def take(self, observation: Any, from_search: bool = False) -> None:
        inner, _ = _notes_beside(observation)
        self.raw.append(inner)
        self.from_search = from_search
        self._walk(inner, 0, True)
        self.from_search = False

    def learn_roles(self, roles: dict[str, str]) -> None:
        self.roles = roles
        self.rows, self.category_counts = {}, {}
        for inner in self.raw:
            self._walk(inner, 0, False)

    def id_key(self, value: dict[str, Any]) -> str | None:
        own = self.roles.get("id")
        if own and isinstance(value.get(own), (str, int)) and not isinstance(value.get(own), bool):
            return own
        return next((k for k, v in value.items() if isinstance(v, str) and _is_id_key(k) and 2 < len(v) <= 64), None)

    def _walk(self, value: Any, depth: int, first_time: bool) -> None:
        if isinstance(value, list):
            for v in value:
                self._walk(v, depth + 1, first_time)
            return
        if not isinstance(value, dict):
            return
        key = self.id_key(value)
        if key:
            pid = str(value[key])
            if pid not in self.ids:
                self.seen_order.append(pid)
            self.ids.add(pid)
            if first_time:
                text = _dump(_trim(value, 300), 2000)
                texts = self.records.setdefault(pid, [])
                if text not in texts:
                    texts.append(text)
                if depth == 0 and any(isinstance(v, dict) for v in value.values()):
                    self.opened.add(pid)
            self._row(pid, value)
            if first_time and depth > 0 and self.from_search:
                self.row_keys.update(_prop_name(k) for k, v in value.items()
                                     if isinstance(v, (str, int, float, bool)) and _prop_name(k))
            if first_time:
                words = " ".join(str(v) for v in value.values() if isinstance(v, (str, int, float)))
                self.text_of[pid] = (self.text_of.get(pid, "") + " " + words.lower())[-4000:]
                for v in value.values():
                    if isinstance(v, dict):
                        sheet = self._sheet(v)
                        if sheet:
                            self.sheet_of.setdefault(pid, {}).update(sheet)
                            self.text_of[pid] += " " + " ".join(sheet.values()).lower()
                return_children = [v for v in value.values() if isinstance(v, list)]
            else:
                return_children = [v for v in value.values() if isinstance(v, (dict, list))]
            for v in return_children:
                self._walk(v, depth + 1, first_time)
            return
        elif depth >= 1 and first_time:
            self._sheet(value)
        for v in value.values():
            if isinstance(v, (dict, list)):
                self._walk(v, depth + 1, first_time)

    def _row(self, pid: str, value: dict[str, Any]) -> None:
        row = self.rows.setdefault(pid, {})
        for slot in ("price", "in_stock", "category", "title"):
            field = self.roles.get(slot)
            if not field or field not in value:
                continue
            v = value[field]
            if slot == "price" and _money(v) is not None:
                row["price"] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else _money(v)
            elif slot == "in_stock" and isinstance(v, (bool, int)):
                row["in_stock"] = bool(v)
            elif slot in ("category", "title") and isinstance(v, str):
                if slot == "category" and row.get("category") != v:
                    self.category_counts[v] = self.category_counts.get(v, 0) + 1
                row[slot] = v

    def _sheet(self, value: dict[str, Any]) -> dict[str, str]:
        """A dict made mostly of short plain values is a property sheet."""
        flat = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and len(str(v)) <= 80}
        if len(flat) < max(3, 0.6 * len(value)):
            return {}
        return {_prop_name(k): str(v) for k, v in flat.items() if _prop_name(k)}


# ------------------------------------------------------------------------------------------------------------ ledger
def _need_of(raw: Any, items: list[dict[str, Any]], source: str, version: int) -> dict[str, Any] | None:
    """One need in the ledger's form, or None when it names no property or no value. A need that holds only in some case
    carries that case under "when"."""
    raw = _d(raw)
    prop = str(raw.get("property") or "").strip()
    value = raw.get("value")
    op = _fold(raw.get("op")) or "is"
    op = next((o for o in OPS if _fold(o) == op), "is")
    if isinstance(value, list):
        value = [str(x).strip() for x in value if str(x).strip()]
        if op != "one of" and len(value) == 1:
            value = value[0]
    elif value is not None:
        value = str(value).strip()
    if not prop or value in (None, "", []):
        return None
    said = str(raw.get("item") or "")
    names = [it["name"] for it in items] or ["item"]
    item = next((n for n in names if said and (_alike(n, said) or _fold(n) in _fold(said) or _fold(said) in _fold(n))),
                names[0])
    out = {"item": item, "property": prop, "op": op, "value": value, "source": source, "version": version}
    when = _when_of(raw.get("when"), names)
    if when:
        out["when"] = when
    return out


def _need_key(need: dict[str, Any]) -> tuple[str, str, str]:
    """A need's place in the ledger: item, property and op, so the two bounds of a range are two needs; each case a need
    depends on is a place of its own."""
    op = str(need.get("op") or "is")
    when = need.get("when")
    if when:
        op = "%s if %s %s %s %s %s" % (op, when["holds"], _fold(when["item"]), _fold(when["property"]), when["op"], _value_key(when["value"]))
    return _fold(need.get("item")), _fold(need.get("property")), op


def _value_key(value: Any) -> str:
    """A need's value as compared for changes: case, spacing and punctuation do not count."""
    return json.dumps(sorted(_fold(x) for x in value) if isinstance(value, list) else _fold(value))


def _leaning(text: str) -> int:
    """How a tool's own words lean: each word of placing an order counts one up, each word of ending without one
    counts one down."""
    words = _key_words(str(text or ""))
    return sum(1 for w in words if w.startswith(ORDER_STEMS)) - sum(1 for w in words if w.startswith(END_STEMS))


def _need_text(need: dict[str, Any]) -> str:
    value = need["value"]
    text = "%s %s %s" % (need["property"], need["op"], " or ".join(value) if isinstance(value, list) else value)
    when = need.get("when")
    if when:
        shown = " or ".join(when["value"]) if isinstance(when["value"], list) else when["value"]
        text += " (%s %s%s %s %s)" % ("only when" if when["holds"] else "unless", ("for %s: " % when["item"]) if when["item"] else "",
                                       when["property"], when["op"], shown)
    return text


def _step_of(raw: Any) -> dict[str, Any] | None:
    raw = _d(raw)
    moment = str(raw.get("moment") or "any").strip().lower()
    action = str(raw.get("action") or "other").strip().lower()
    if moment not in MOMENTS:
        moment = "any"
    if action not in ACTIONS:
        action = "other"
    detail = str(raw.get("detail") or "").strip()[:200]
    return {"moment": moment, "action": action, "detail": detail, "done": False}


class Ledger:
    """What the shopper wants, kept current. `version` rises with every change to the needs, the way to choose, the
    budgets, the request's steps, or the price or stock of a listing in play."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []         # {name, qty, store_label, cap}
        self.needs: list[dict[str, Any]] = []
        self.no_preference: list[str] = []
        self.objective: dict[str, str] = {"kind": "any", "criterion": "", "direction": ""}
        self.budgets: dict[str, Any] = {"money_cap": None, "questions_cap": None, "max_steps": None}
        self.in_stock = True
        self.rules: list[str] = []                    # the request's rule sentences, as written
        self.steps: list[dict[str, Any]] = []         # {moment, action, detail, done}
        self.decline_rule: str | None = None
        self.story_hints: list[dict[str, str]] = []
        self.refs: list[dict[str, Any]] = []          # properties tied to another listing: followed by code
        self.version = 0

    def several(self) -> bool:
        return len(self.items) > 1

    def needs_for(self, item: str) -> list[dict[str, Any]]:
        if not self.several():
            return list(self.needs)
        return [n for n in self.needs if _fold(n["item"]) == _fold(item)]

    def signature(self) -> str:
        return json.dumps([sorted(_need_key(n) + (_value_key(n["value"]),) for n in self.needs),
                           self.objective_key(self.objective), self.budgets.get("money_cap"),
                           sorted((s["moment"], s["action"], _fold(s["detail"])) for s in self.steps)], default=str)

    @staticmethod
    def objective_key(objective: dict[str, str]) -> tuple[str, str, str]:
        """The way to choose as compared for changes: the measure and its direction count only for "best"."""
        kind = _fold(objective.get("kind")) or "any"
        if kind != "best":
            return kind, "", ""
        return kind, _fold(objective.get("criterion")), _fold(objective.get("direction"))

    def summary(self) -> str:
        parts = ["items: " + "; ".join("%d x %s%s%s" % (it["qty"], it["name"],
                                                        " [store label %s]" % it["store_label"] if it.get("store_label") else "",
                                                        " (each at most %s)" % it["cap"] if it.get("cap") else "")
                                       for it in self.items)]
        by_item: dict[str, list[str]] = {}
        for n in self.needs:
            by_item.setdefault(n["item"], []).append(_need_text(n))
        if by_item:
            parts.append("needs: " + " | ".join(("%s: " % k if self.several() else "") + "; ".join(v)
                                                for k, v in by_item.items()))
        else:
            parts.append("needs: none beyond the item itself")
        if self.no_preference:
            parts.append("shopper does not mind: " + "; ".join(self.no_preference))
        kind = self.objective.get("kind")
        parts.append("choose: " + ("the cheapest listing that fits" if kind == "cheapest" else
                                   "the fitting listing with the %s %s" % ("highest" if self.objective.get("direction") != "min"
                                                                            else "lowest", self.objective.get("criterion"))
                                   if kind == "best" else "any listing that fits (no price hunting)"))
        if self.budgets.get("money_cap"):
            parts.append("money limit for everything together: %s" % self.budgets["money_cap"])
        if self.in_stock:
            parts.append("must be in stock")
        if self.rules:
            parts.append("the request's own steps: " + " / ".join(self.rules)[:600])
        return "LEDGER v%d | " % self.version + " | ".join(parts)

    def as_material(self) -> dict[str, Any]:
        return {"items": self.items, "needs": [{k: n[k] for k in ("item", "property", "op", "value")} for n in self.needs],
                "no_preference": self.no_preference, "objective": self.objective,
                "money_cap": self.budgets.get("money_cap"), "rules": self.rules,
                "steps": [{k: s[k] for k in ("moment", "action", "detail")} for s in self.steps],
                "references": [{k: r.get(k) for k in ("item", "property", "relation", "seller", "category", "marks")}
                               for r in self.refs if r["status"] == "open"]}


# ------------------------------------------------------------------------------------------------------------ code matcher
def _unit_words(text: Any) -> set[str]:
    """The words of a value with its figures cut away (1.2v -> v)."""
    return {w for w in _tokens(re.sub(r"[\d.,]+", " ", str(text))) if w}


def _unit_like(x: str, y: str) -> bool:
    """One unit word spelled singular or plural, or shortened (in, inches)."""
    return x == y or x.rstrip("s") == y.rstrip("s") or (len(x) >= 2 and len(y) >= 2 and (x.startswith(y) or y.startswith(x)))


def _same_units(shown: str, want: str) -> bool:
    """Two figures can be compared when their unit words agree, or when one of them names no unit."""
    a, b = _unit_words(shown), _unit_words(want)
    return not a or not b or any(_unit_like(x, y) for x in a for y in b)


def _pieces(shown: Any) -> set[str]:
    """The values one field lists, folded."""
    return {_fold(x) for x in re.split(r"[,;/|]", str(shown)) if _fold(x)}


def _cell(shown: str, op: str, value: Any, vocab: set[str]) -> tuple[str, str]:
    """How one field stands on one need: met, against or unclear. A text value is against when it adds a word to the
    named value, or when the field shows a value of the store's own vocabulary for that property and the wanted value
    is one of that vocabulary too."""
    got = _fold(shown)
    wants = [w for w in (_fold(x) for x in (value if isinstance(value, list) else [value])) if w]
    if not got or not wants:
        return "unclear", ""
    pieces = [_fold(x) for x in re.split(r"[,;/|]", str(shown)) if _fold(x)] or [got]
    if op in ("at most", "at least"):
        a, b = _figures(str(shown)), _figures(str(value[0] if isinstance(value, list) else value))
        if len(a) == 1 and len(b) == 1 and _same_units(str(shown), str(value)):
            ok = a[0] <= b[0] + 1e-9 if op == "at most" else a[0] >= b[0] - 1e-9
            return ("met", "shows %s" % shown) if ok else ("against", "shows %s, needs %s %s" % (shown, op, b[0]))
        return "unclear", "shows %s" % shown
    exact = any(w == got or w in pieces for w in wants)
    # the wanted words in a row, as the value writes them; the same words with others between them are another value
    covered = exact or any(" %s " % w in " %s " % x for w in wants for x in [got] + pieces)
    scattered = not covered and any(set(w.split()) <= set(got.split()) for w in wants)
    if op == "is not":
        if exact:
            return "against", "shows %s" % shown
        if not any(_figures(w) for w in wants) and any(set(w.split()) <= set(p.split()) for w in wants for p in pieces):
            return "clash", "shows %s" % shown     # the unwanted value's words sit inside a longer value: a model reads it
        return ("met", "shows %s" % shown) if all(p in vocab for p in pieces) else ("unclear", "shows %s" % shown)
    if covered:
        return "met", "shows %s" % shown
    extra = [x for w in wants if set(w.split()) <= set(got.split()) for x in set(got.split()) - set(w.split())]
    if scattered and op in ("is", "one of") and not any(sep in str(shown) for sep in ",;/|") \
            and extra and all(len(x) >= 3 and x.isalpha() for x in extra):
        # one sheet value that adds a word to the named value is another value of the property (a reordering is not)
        return "against", "shows %s, not %s" % (shown, " / ".join(wants)[:40])
    if scattered and op in ("is", "one of"):
        return "clash", "shows %s%s" % (shown, LOOSE)   # a list of values with the wanted words spread: a model reads it
    if op == "has":
        return "unclear", "shows %s" % shown
    if op in ("is", "one of") and len(_figures(str(shown))) == 1:
        # one figure against one figure in the same measure is decided by code: 1.2 v is not 12 v; every wanted value
        # is tried before the field counts against, and "met" needs the wanted value's own words as well (12 pro max
        # is not 12 mini)
        a, miss = _figures(str(shown))[0], ""
        got_units = _unit_words(shown)
        for w in (str(x) for x in (value if isinstance(value, list) else [value])):
            b = _figures(w)
            if len(b) == 1 and _same_units(str(shown), w):
                if abs(a - b[0]) <= 1e-9 + 0.005 * abs(b[0]) \
                        and all(any(_unit_like(x, y) for y in got_units) for x in _unit_words(w)):
                    return "met", "shows %s" % shown
                miss = miss or w
        if miss and not any(abs(a - _figures(w)[0]) <= 1e-9 + 0.005 * abs(_figures(w)[0])
                            for w in (str(x) for x in (value if isinstance(value, list) else [value])) if len(_figures(w)) == 1):
            return "against", "shows %s, not %s" % (shown, miss)
    if _figures(str(shown)) or any(_figures(w) for w in wants):
        return "unclear", "shows %s" % shown        # figures in words ("under 30 mm") are read, not matched as text
    if all(p in vocab for p in pieces) and any(w in vocab for w in wants):
        # two spellings of one value (grey, gray) go to the strong chain; a plainly different value fails here,
        # word by word: values that share a word but differ in another (hand mixer, stand mixer) are two values
        def near_words(a: str, b: str) -> bool:
            wa, wb = a.split(), b.split()
            return len(wa) == len(wb) and all(x == y or difflib.SequenceMatcher(None, x, y).ratio() >= 0.75
                                              for x, y in zip(wa, wb))
        near = any(near_words(w, p) for w in wants for p in pieces)
        return ("clash" if near else "against"), "shows %s" % shown
    # a plain value the store shows for the property that is not the wanted one is never taken as met unread
    return ("clash", "shows %s%s" % (shown, LOOSE)) if op in ("is", "one of") else ("unclear", "shows %s" % shown)


def _search_key(args: dict[str, Any], page: str | None) -> str:
    """One search's identity: its arguments but the page, folded, with the words in a fixed order (the same words in
    another order bring the same rows)."""
    return "|".join("%s:%s" % (k, " ".join(sorted(_fold(str(v)).split()))) for k, v in sorted(args.items()) if k != page)


# ------------------------------------------------------------------------------------------------------------ the trip
class Trip:
    def __init__(self, problem_data: dict[str, Any]):
        env = _d(_d(problem_data).get("environment"))
        self.binding = _d(env.get("binding"))
        view = _d(env.get("policy_view"))
        self.request = str(view.get("query") or "")
        self.opening = view.get("opening")
        self.tool_list = [t for t in (view.get("tools") or []) if isinstance(t, dict)
                          and isinstance(_d(t.get("function", t)).get("name"), str)]
        self.schemas = {t.get("function", t)["name"]: t.get("function", t) for t in self.tool_list}
        try:
            self.step_cap = int(view.get("max_steps") or 30)
        except (TypeError, ValueError):
            self.step_cap = 30
        self.clock = time.monotonic()
        self.steps = 0
        self.done = bool(view.get("finished"))
        self.stuck = False
        self.out: list[dict[str, Any]] = []
        self.ledger = Ledger()
        self.index = Index()
        self.names: dict[str, str] = {}             # job -> tool name
        self.fixed: dict[str, dict[str, Any]] = {}  # job -> arguments that select it
        self.readers: list[str] = []                # tools that show one listing by its id
        self.page_keys: dict[str, set[str]] = {}    # reader -> the keys every page it returned carried
        self.page_sheet: dict[str, int] = {}        # reader -> the fewest entries a page's nested table held
        self.pair_reads: bool | None = None         # whether the compare tool reads two whole listings at once
        self.word_choices: dict[str, Any] = {}      # the search's choices on the last call whose words found rows
        self.wordless_choices: dict[str, Any] = {}  # its choices on the last call made without words
        self.find_args: dict[str, str] = {}
        self.last_model = ""
        self.roles_tries = 0
        self.talk_sent = 0
        self.talk_hard_cap: int | None = None       # lowered when the store refuses a message
        self.questions_sent = 0
        self.budget_asked = False
        self.asked: list[tuple[int, str]] = []      # (item number, property) asked about
        self.pending_question = ""
        self.pending_text = ""                       # the asked words when other properties rode along
        self.pending_list: list[str] = []            # the properties a list question names, first the one asked
        self.talk_log: list[str] = []
        self.taken: set[tuple[str, str]] = set()
        self.basket: list[str] = []
        self.line_qty: dict[str, int] = {}
        self.held: dict[str, int] = {}               # units of a listing the store's cart reports holding
        self.count_key: str | None = None            # the result field that reports those units, learned from an add
        self.basket_touched = False
        self.add_mark: dict[str, Any] | None = None
        self.basket_seen: Any = None
        self.cart_mark = 0                          # raised by every cart change
        self.cart_viewed = -1                       # cart_mark when the cart was last looked at
        self.fields: dict[str, dict[str, str]] = {}   # listing id -> its short plain fields, from any record
        self.torn: set[str] = set()                   # listings with two keys of one name that share no word
        self.said_by: dict[str, dict[tuple[str, str], set[str]]] = {}  # listing -> (table, name) -> words shown
        self.read_by: dict[str, dict[str, int]] = {}  # listing id -> reader tool -> ledger version when read
        self.checks: dict[tuple[str, int], dict[str, Any]] = {}
        self.facts_seen: dict[str, tuple[Any, Any]] = {}
        self.price_rose: dict[str, float] = {}         # listing id -> the higher price it moved to
        self.gone: set[str] = set()
        self.fired: dict[str, int] = {}
        self.fired_in: dict[tuple[str, int, int], int] = {}    # guard firings per (guard, ledger version, cart mark)
        self.group_size = 0                         # listings per compare read once known (0: not tried yet)
        self.audit_fresh: set[tuple[int, int]] = set()   # (item, listings gone) searched afresh before ending
        self.stated_audited = False                 # the request read once more against the store's own properties
        self.rule_way: str | None = None             # what the request's "nothing fits" sentence asks: "end" or other
        self.keys_seen: dict[str, set[str]] = {}
        self.chat: list[dict[str, Any]] = []
        self.last_looks: list[str] = []
        self.moments_run: set[tuple[str, int]] = set()
        self.raw_tries = 0
        self.page_tokens: dict[str, set[str]] = {}  # search words -> later-page values the store gave for them
        self.resent = 0                             # finishes sent again by code after one came back with a note
        self.commits = 0
        self.commit_key: tuple | None = None
        self.refused_last = False
        self.order_faults = 0                       # orders the store itself would not take (a fault, not a note)
        self.cart_faults, self.fault_mark = 0, None   # the same, for the cart as it stands now
        self.looked: set[tuple] = set()             # (cart, ledger version) pairs the final look has read
        self.swapped_out: set[str] = set()          # lines the final look took out: never swapped back in
        self.version_seen = 0
        self.event_version = 0                      # ledger version of the last notice, refusal or fact change
        self.reread: list[str] = []
        self.hurry = False                          # finishing because time is short: no optional model reads
        self.chain_seen: set[str] = set()           # listings already read along the tie being followed
        self.chain_tried: set[str] = set()          # listings the follower tried to read on the tie being followed
        self.started = self.clock                   # when the setup ended, for the pace of the steps
        self.steps_at_start = 0
        self.paging: dict[str, str] = {}
        self.page_next: dict[str, tuple[dict[str, Any], Any]] = {}   # search words -> (its arguments, next page value)
        self.searches: set[str] = set()             # distinct searches sent (words and narrowing values)
        self.empty_searches: set[str] = set()       # those that brought no rows: the same words again bring none
        self.item_queries: dict[int, set[str]] = {} # item -> distinct searches whose rows served it
        self.item_paged: set[int] = set()           # items for which a later page of a search was read
        self.page_items: dict[str, set[int]] = {}   # search words -> the items its rows served
        self.searched_for: set[int] = set()         # items code searched for by name before planning
        self.read_records: dict[tuple[str, str], str] = {}   # (listing, reader) -> the record that reading gave
        self.free_reads = 0
        self.ask_now = False                        # set when an add came before the questions were ready
        self.takeovers = 0
        self.free: set[tuple[int, str]] = set()     # (item, property) the shopper does not mind about
        self.probed: set[tuple[int, float, int]] = set()  # (item, price, notice count) already looked under
        self.probe_shown: list[str] = []            # the listings the last capped search showed
        self.rechosen: set[tuple[int, int]] = set() # (item, ledger version) whose cheap end was read again
        self.carted: set[str] = set()               # listings our calls put in the store's cart
        self.additive = False                       # (kept off: repeated lines are never assumed to add up)
        self.keeps_unsent = False                   # a cart view showed the store keeps lines a whole-cart call left out
        self.last_sent: set[str] | None = None      # listings our last whole-cart call listed above 0
        self.view_readable = True                   # the last cart view named listings we know
        self.announced: set[int] = set()            # items whose before-add message went out
        self.zero_refused = False                   # the store refused a cart call that carried a line at 0
        self.add_faults: dict[str, int] = {}        # listing -> add calls that failed for it
        self.failed_calls: dict[str, int] = {}      # identical add calls (as sent) that failed
        self.ranked_by_item: dict[int, list[str]] = {}   # the code ranking's properties per item, best first
        self.hunt_rounds: dict[tuple[int, str], int] = {}   # (item, need) -> searches made to find a listing stating it
        self.joint_done: set[tuple[int, tuple[str, ...]]] = set()   # (item, learned needs) searched together once
        self.listed_at: dict[tuple[int, str], int] = {}     # (item, property) -> its frontmost place in a list asked
        self.kit_searches: dict[tuple[int, str], int] = {}  # (item, the other item's cart line) -> code searches made
        self.kit_switched: set[tuple[int, str]] = set()     # (item, line) whose other-item line code already replaced
        self.hunt_read: set[str] = set()            # rows the hunt already tried to read
        self.hunt_batches: dict[tuple[int, str], int] = {}   # (item, need) -> reads of rows already found
        self.single_only = False                    # the shopper asked for one question at a time
        self.reask: tuple[int, str] | None = None   # a property to ask again on its own after a muddled reply
        self.reasked: set[tuple[int, str]] = set()
        self.pending_item = 0                       # the item the question in flight is about
        self.usage: dict[str, dict[str, int]] = {}  # model -> calls, prompt, cached and completion tokens
        self.spent = 0.0                            # dollars this trip's model calls cost so far
        self.call_cost: dict[str, float] = {}       # model -> dollars its last call cost
        self.passed_over = 0                        # costly calls left out to keep the trip in its share
        self.cache_tail: Any = None                 # the last driver message of the previous driver call
        self.lean_cut = 0                           # driver turns left out of the chat (moves in large steps)
        self.escalated = False                      # the one strong-model check of the trip is spent
        self.mode_retried = False                   # the finish mode was read once more at the finish
        self.turn_no = 0                            # driver turns, for our own tool-call ids
        self.unclear_hold = False                   # an add was held once for a listing that settles more needs
        self.queue: list[tuple[int, str]] = []      # planned questions: (item number, property), next first
        self.plan_key: str | None = None            # the needs the queue was planned for (None: not planned yet)
        self.plan_failed = False                    # the planning read gave nothing: code ranking stands in
        self.sheet_readers: set[str] = set()        # readers whose results carried a property sheet
        self.sheetless: set[str] = set()            # readers whose results carried none
        self.prep_reads = 0                         # reads code made to get the questions ready
        self.paged = False                          # a later page of some search was fetched
        self.out_of_moves = False                   # code has nothing left to try but to end
        self.pages_left = 3
        self._init_m_finish()
        self._init_m_interview()

    # -------------------------------------------------------------- time, steps, budgets
    def remaining(self) -> float:
        return SECONDS - (time.monotonic() - self.clock)

    def steps_left(self) -> int:
        return self.step_cap - self.steps

    def easing(self, extra: float = 0.0) -> bool:
        return self.hurry or self.remaining() < SECONDS * EASE_OFF + extra

    def reserve(self) -> int:
        return RESERVE_BASE + RESERVE_PER_ITEM * (max(1, len(self.ledger.items)) - 1)

    def _guard(self, name: str) -> bool:
        """True while the guard `name` may still fire: at most GUARD_LIMIT times in the current state (a new ledger
        version or a cart change opens it again) and GUARD_TRIP_LIMIT times per trip; counts the firing."""
        state = (name, self.ledger.version, self.cart_mark)
        if self.fired_in.get(state, 0) >= GUARD_LIMIT or self.fired.get(name, 0) >= GUARD_TRIP_LIMIT:
            return False
        self.fired_in[state] = self.fired_in.get(state, 0) + 1
        self.fired[name] = self.fired.get(name, 0) + 1
        return True

    def messages_owed(self) -> int:
        """Messages kept back for the request's own steps: one per moment that still owes one. Ending without buying
        owes none when the ending call carries its own words."""
        return self.others_owed() + self.adds_owed()

    def others_owed(self) -> int:
        """Messages kept for the request's own steps at moments other than adding: one per moment."""
        ending_speaks = bool(self.names.get("none") and self.text_arg(self.names["none"]))
        return len({s["moment"] for s in self.ledger.steps if s["action"] == "message" and not s["done"]
                    and s["moment"] != "before_add" and not (s["moment"] == "before_decline" and ending_speaks)})

    def adds_owed(self) -> int:
        """Before-add messages still owed: one per item with no line yet and no message sent for it."""
        if not any(s["action"] == "message" and s["moment"] == "before_add" for s in self.ledger.steps):
            return 0
        have = {i for _, i in self.cart_lines()}
        return sum(1 for n in range(len(self.ledger.items)) if n not in have and n not in self.announced)

    def messages_left(self) -> int:
        """Messages the store will still take: the stated allowance (or the one a refusal showed), else no limit known."""
        caps = [c for c in (self.ledger.budgets.get("questions_cap"), self.talk_hard_cap) if c is not None]
        return max(0, min(caps) - self.talk_sent) if caps else 99


    def listing(self) -> bool:
        """Several properties per message, until the shopper complains about that."""
        return not self.single_only

    def questions_left(self) -> int:
        """Messages open to questions: two when the store allows them, the request's own steps getting what is beyond that."""
        stated = self.ledger.budgets.get("questions_cap")
        left = self.messages_left()
        room = left - self.messages_owed()
        if stated is None:
            room = min(room, QUESTIONS_DEFAULT - self.questions_sent)
        return max(0, room, min(left, 2 - self.questions_sent))

    # -------------------------------------------------------------- model calls
    def _ask(self, chain: list[str], messages: list[dict[str, Any]], tokens: int, tools: list | None = None,
             first_cap: float | None = None, cache: bool = False, caps: tuple[float, ...] | None = None,
             deadline: float | None = None, share: float = TRIP_SHARE) -> Any:
        caps = caps or (((first_cap,) + WAITS[1:]) if first_cap else WAITS)
        n = -1                                    # models of the chain tried so far, less one
        for model in chain:
            if self.remaining() < SECONDS * HAND_OVER * 0.66:
                return None
            if model in COSTLY and self.spent + self.call_cost.get(model, CALL_GUESS) > share:
                self.passed_over += 1                 # the next model of the chain answers in its place
                continue
            if self.spent + self._call_guess(model) > TRIP_CEILING:
                self.passed_over += 1                 # past the ceiling no model is called: code finishes the trip
                continue
            n += 1
            if deadline is not None and deadline - time.monotonic() < 3.0:
                return None
            sent = self._breakpoints(messages) if cache and model.startswith(CACHED) else messages
            frame: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": tokens}
            if tools is not None:
                frame.update({"tools": tools, "tool_choice": "auto"})
            if model.startswith(ROOMY):
                # their reasoning comes out of the same allowance; the driver gets the room c42 gave it
                frame["max_tokens"] = max(tokens, 1600) if tools is not None else max(2 * tokens, 1600)
            else:
                frame["reasoning"] = {"enabled": False}
            wait = max(5.0, min(70.0, self.remaining() - SECONDS * HAND_OVER + 3.0))
            wait = min(wait, caps[min(n, len(caps) - 1)])     # a slow model hands over to the next one in time
            if deadline is not None:
                wait = min(wait, deadline - time.monotonic())
            until = time.monotonic() + wait
            reply, _ = _post("/inference/chat/completions", frame, until)
            self._count(model, reply)
            if reply is None and model in COSTLY:
                self.spent += self.call_cost.get(model, CALL_GUESS)   # a call given up on may still be billed
            try:
                message = reply["choices"][0]["message"]
            except (TypeError, KeyError, IndexError):
                continue
            if message.get("content") or message.get("tool_calls"):
                self.last_model = model
                return message
        return None

    def _breakpoints(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The driver chat with cache breakpoints (Anthropic allows four; three are used): the system text, the last
        message the previous driver call ended on (its prefix was cached then), and the last message now (cached for
        the next call). Marked messages are copies; the chat itself keeps plain strings."""
        marks = {0, len(messages) - 1}
        prev = next((i for i, m in enumerate(messages) if m is self.cache_tail), None)
        if prev is not None and 0 < prev < len(messages) - 1:
            marks.add(prev)
        self.cache_tail = messages[-1] if messages else None
        out = []
        for i, m in enumerate(messages):
            text = m.get("content")
            if i in marks and isinstance(text, str) and text:
                m = dict(m, content=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}])
            out.append(m)
        return out

    def _count(self, model: str, reply: Any) -> None:
        """Tokens sent and read from cache, per model (kept on the trip for the logs), and what the call cost: the
        reply's own figure when it carries one, else its tokens at the model's prices."""
        usage = _d(_d(reply).get("usage"))
        row = self.usage.setdefault(model, {"calls": 0, "prompt": 0, "cached": 0, "completion": 0})
        row["calls"] += 1
        got = {}
        for key, value in (("prompt", usage.get("prompt_tokens")), ("completion", usage.get("completion_tokens")),
                           ("cached", _d(usage.get("prompt_tokens_details")).get("cached_tokens"))):
            if isinstance(value, int):
                row[key] += value
                got[key] = value
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
            sent, cached, written = PRICES.get(model, PRICES[SONNET])
            fresh = max(0, got.get("prompt", 0) - got.get("cached", 0))
            cost = (fresh * sent + got.get("cached", 0) * cached + got.get("completion", 0) * written) / 1e6
        if cost > 0:
            self.spent += cost
            self.call_cost[model] = cost

    def _read(self, chain: list[str], instructions: str, material: Any, tokens: int = 800,
              first_cap: float | None = None, caps: tuple[float, ...] | None = None,
              deadline: float | None = None, share: float = TRIP_SHARE) -> dict[str, Any]:
        message = self._ask(chain, [{"role": "system", "content": instructions},
                                    {"role": "user", "content": json.dumps(material, ensure_ascii=False, default=str)}], tokens,
                            first_cap=first_cap, caps=caps, deadline=deadline, share=share)
        return (_object_in(message.get("content") or "") or {}) if message else {}

    # -------------------------------------------------------------- the store
    def _send(self, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
        action = {"name": name, "args": args}
        until = time.monotonic() + max(3.0, self.remaining() - 2.0)
        result, error = _post("/environment/call", {**self.binding, "step": self.steps + 1, "action": action}, until)
        if result is None:
            said = json.dumps(error or {}, default=str)
            if (error or {}).get("status") == 402:
                self.stuck = True
                return {"error": "the session cannot go on"}, False
            m = re.search(r"expected (\d+)", said)
            if m and int(m.group(1)) - 1 != self.steps:
                self.steps = int(m.group(1)) - 1
                result, error = _post("/environment/call", {**self.binding, "step": self.steps + 1, "action": action}, until)
            elif "finished" in said.lower():
                self.done = True
                return {"error": "the trip has ended"}, False
        if result is None:
            return {"error": "the call did not go through; try again"}, False
        self.steps = result["step"] if isinstance(result.get("step"), int) else self.steps + 1
        self.done = bool(result.get("finished"))
        self.out.append({"role": "assistant", "content": "", "tool_call": {
            "id": "s%d" % self.steps, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}, "environment_result": result})
        return _as_records(result.get("observation")), True

    def _run(self, name: str, args: dict[str, Any], header: str, size: int = 3000) -> tuple[Any, bool]:
        """A call code makes itself: fitted to its schema, sent, taken in, and shown to the driver with `header`."""
        issues: list[str] = []
        args = _fit(self.schemas[name].get("parameters") or {}, args, name, issues)
        if issues:
            return {"error": "; ".join(issues)}, False
        if self.failed_calls.get(name + json.dumps(args, sort_keys=True), 0) >= 2:
            return {"error": "this very call failed twice already"}, False
        observation, sent = self._send(name, args)
        if not sent:
            return observation, False
        notes = self.absorb(name, args, observation)
        self.chat.append({"role": "user", "content": "%s (run for you): %s %s\nRESULT: %s%s" % (
            header, name, json.dumps(args, ensure_ascii=False), _dump(observation, size),
            ("\nnote: " + " ".join(notes)) if notes else "")})
        return observation, True

    # -------------------------------------------------------------- jobs of the tools
    def enum_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        props = (self.schemas.get(name, {}).get("parameters") or {}).get("properties") or {}
        return {k: v for k, v in args.items() if "enum" in (props.get(k) or {})}

    def role(self, name: str, args: Any = None) -> str | None:
        jobs = [j for j, n in self.names.items() if n == name]
        if "buy" in jobs and "none" in jobs:
            given = self.enum_args(name, args) if isinstance(args, dict) else {}
            for job in ("none", "buy"):
                selector = self.enum_args(name, self.fixed.get(job, {}))
                if selector and all(_alike(given.get(k), v) for k, v in selector.items()):
                    return job
            for job, other in (("buy", "none"), ("none", "buy")):
                selector = self.enum_args(name, self.fixed.get(job, {}))
                if selector and any(k in given and not _alike(given[k], v) for k, v in selector.items()):
                    return other
            return "buy"
        for job in ("buy", "none", "add", "cart", "speak", "open", "compare", "find"):
            if job in jobs:
                return job
        return "open" if name in self.readers else None

    def _split_finish(self) -> bool:
        """A tool that both buys and ends without buying must be told apart by an enum argument."""
        if self.names.get("buy") != self.names.get("none"):
            return True
        name = self.names["buy"]
        a, b = self.enum_args(name, self.fixed.get("buy", {})), self.enum_args(name, self.fixed.get("none", {}))
        return any(k in b and not _alike(v, b[k]) for k, v in a.items())

    def _tell_finish_apart(self, chain: list[str] | None = None) -> None:
        name = self.names["buy"]
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        arg = next((k for k in list(params.get("required") or []) + list(props) if len((props.get(k) or {}).get("enum") or []) >= 2), None)
        if not arg:
            return
        got = self._read(chain or TERSE, FINISH_VALUES, {"tool": name, "description": self.schemas[name].get("description", ""),
                                                 "argument": arg, "choices": props[arg]["enum"]}, 300)
        picks = {job: next((m for m in props[arg]["enum"] if _alike(m, got.get(job))), None) for job in ("buy", "none")}
        if picks["buy"] and picks["none"] and picks["buy"] != picks["none"]:
            for job in ("buy", "none"):
                self.fixed.setdefault(job, {})[arg] = picks[job]

    def _mode_by_words(self, name: str) -> None:
        """Mode values of a shared finishing tool from its own words: each value with the clause the description
        gives it, weighed by _leaning. The value leaning clearly towards ordering orders, the one leaning clearly
        away ends; with two values, a clear one decides the other. Nothing is set when no value leans clearly."""
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        arg = next((k for k in list(params.get("required") or []) + list(props)
                    if len((props.get(k) or {}).get("enum") or []) >= 2), None)
        if not arg or self._split_finish():
            return
        values = [str(v) for v in props[arg]["enum"]]
        text = str(self.schemas[name].get("description") or "") + " " + str(props[arg].get("description") or "")
        def lean(v: str) -> int:
            m = re.search(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(v), text)
            clause = re.split(r"[;.]", text[m.end():])[0] if m else ""
            for other in values:                 # a clause ends where the next value begins
                if other != v and re.search(r"\b%s\b" % re.escape(other), clause):
                    clause = clause[:re.search(r"\b%s\b" % re.escape(other), clause).start()]
            return _leaning(v + " " + clause)
        tilt = {v: lean(v) for v in values}
        up = [v for v in values if tilt[v] > 0 and tilt[v] == max(tilt.values())]
        down = [v for v in values if tilt[v] < 0 and tilt[v] == min(tilt.values())]
        buy = up[0] if len(up) == 1 else None
        end = down[0] if len(down) == 1 else None
        if len(values) == 2:
            buy = buy or (next(v for v in values if v != end) if end else None)
            end = end or (next(v for v in values if v != buy) if buy else None)
        if buy and end and buy != end:
            self.fixed.setdefault("buy", {})[arg] = buy
            self.fixed.setdefault("none", {})[arg] = end

    def text_arg(self, name: str) -> str | None:
        params = self.schemas.get(name, {}).get("parameters") or {}
        props = params.get("properties") or {}
        free = [k for k, v in props.items() if v.get("type") == "string" and "enum" not in v]
        required = [k for k in free if k in (params.get("required") or [])]
        if required:
            return required[0]
        return free[0] if len(free) == 1 else None

    def closing_args(self, job: str, text: str) -> dict[str, Any] | None:
        """Arguments for the order or the ending: what setup fixed, the words in the tool's text field, and every other
        required field completed from its schema, so the call can always be made."""
        name = self.names.get(job)
        if not name:
            return None
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        args = dict(self.fixed.get(job, {}))
        shared = self.names.get("buy") == self.names.get("none")
        other = self.fixed.get("buy" if job == "none" else "none", {}) if shared else {}
        if shared:
            # either finish needs its own mode value, or the other one's to steer clear of; never a guess
            modes = [k for k in (params.get("required") or []) if "enum" in (props.get(k) or {})]
            if any(k not in args and k not in other for k in modes):
                return None
        slot = self.text_arg(name)
        if slot:
            args[slot] = text
        for k in params.get("required") or []:
            if k in args:
                continue
            sch = props.get(k) or {}
            if "enum" in sch:
                args[k] = next((m for m in sch["enum"] if not _alike(m, other.get(k))), sch["enum"][0])
        args = self._fill_hole(params, args, text)
        issues: list[str] = []
        args = _fit(params, args, name, issues)
        return None if issues else args

    def _sets_cart(self, name: str) -> bool:
        """A cart tool whose arguments hold a list of line objects sets the whole cart in one call."""
        props = (self.schemas.get(name, {}).get("parameters") or {}).get("properties") or {}
        return any((v or {}).get("type") == "array" and ((v or {}).get("items") or {}).get("type") == "object"
                   for v in props.values())

    def store_labels(self) -> list[str]:
        """Category names the store itself offers: enum values of the tools' arguments, plus those seen on listings."""
        offered: list[str] = []
        for schema in self.schemas.values():
            for sch in ((schema.get("parameters") or {}).get("properties") or {}).values():
                values = (sch or {}).get("enum") or ((sch or {}).get("items") or {}).get("enum") or []
                if len(values) >= 4 and all(isinstance(v, str) for v in values):
                    offered += [v for v in values if v not in offered]
        seen = sorted(self.index.category_counts, key=lambda k: -self.index.category_counts[k])
        return offered + [s for s in seen if s not in offered]

    def label_of(self, idx: int) -> str:
        """The store label an item is bought under: the setup's, when the store knows it; else the known label sharing
        most words with the item's name; empty when none does."""
        if not self.ledger.items:
            return ""
        item = self.ledger.items[min(idx, len(self.ledger.items) - 1)]
        labels = self.store_labels()
        given = item.get("store_label")
        if given:
            hit = next((x for x in labels if _alike(x, given)), None)
            if hit or not labels:
                return hit or str(given)
        said = {w.rstrip("s") for w in _tokens(item.get("name"))}
        best, top = "", (0, 0)
        for label in labels:
            shared = len({w.rstrip("s") for w in _tokens(label)} & said)
            if shared and (shared, self.index.category_counts.get(label, 0)) > top:
                best, top = label, (shared, self.index.category_counts.get(label, 0))
        return best

    def item_of(self, pid: str) -> int:
        """The list item a listing belongs to: by its store label, else by the words its title shares with an item."""
        if not self.ledger.several():
            return 0
        row = self.index.rows.get(pid, {})
        for n in range(len(self.ledger.items)):
            label = self.label_of(n)
            if label and row.get("category") and _alike(row["category"], label):
                return n
        words = set(_tokens(str(row.get("title") or "") + " " + str(row.get("category") or "")
                            + " " + self.index.text_of.get(pid, "")[:300]))
        shared = [len({w.rstrip("s") for w in _tokens(it["name"])} & {w.rstrip("s") for w in words})
                  for it in self.ledger.items]
        return max(range(len(shared)), key=lambda n: shared[n]) if max(shared) > 0 else 0

    # -------------------------------------------------------------- setup
    def setup(self) -> None:
        """One reading of the request and the tools (strong chain); a reading whose tool roles do not hold against
        the real schemas is retried with the next model."""
        tools = "\n".join("%s: %s %s" % (n, s.get("description", ""), _dump(s.get("parameters") or {}, 2500))
                          for n, s in self.schemas.items())
        material: dict[str, Any] = {"tools": tools, "request": self.request}
        if self.opening is not None:
            material["opening"] = _trim(self.opening, 400)
        start, kept = 0, None
        deadline = time.monotonic() + SETUP_DEADLINE
        while start < len(SETUP):
            self.last_model = ""
            got = self._read(SETUP[start:], SETUP_READING, material, 2400, caps=SETUP_WAITS[start:], deadline=deadline)
            problems = self._setup_problems(got)
            if kept is None or len(problems) < kept[0]:
                kept = (len(problems), got)
            if not problems or not self.last_model or self.easing(60.0):
                break                                  # a reading that holds, no model answering, or no time left
            start = SETUP.index(self.last_model) + 1 if self.last_model in SETUP[start:] else start + 1
        try:
            self._take_setup(kept[1] if kept else {})
        except Exception:  # a reading that breaks the mapping counts as no reading
            self.names, self.fixed, self.readers, self.ledger = {}, {}, [], Ledger()
            self._take_ledger({})
        if not self.names.get("buy"):
            self._roles_by_shape()
        if self.opening is not None:
            self.index.take(self.opening)
            self._collect_fields(_notes_beside(self.opening)[0])
        first = "THE SHOPPER'S REQUEST:\n%s" % self.request
        if self.opening is not None:
            first += "\n\nWHAT THE STORE SHOWS FIRST:\n%s" % _dump(self.opening, 4000)
        self.chat = [{"role": "system", "content": DRIVER_GUIDE}, {"role": "user", "content": first + "\n\n" + self.status()}]
        self.version_seen = self.ledger.version
        self.started, self.steps_at_start = time.monotonic(), self.steps

    def _roles_by_shape(self) -> None:
        """No reading held: tool jobs from the schemas alone. A long free text is a message; one short id is a reader;
        an id with a quantity or a mode, or a list of lines, is the cart; one required two-way choice ends the trip
        (its first choice taken as ordering); a free text among other arguments is the search; a list of ids lines
        listings up. A tool with no arguments that shares a word with the cart tool's description shows the cart."""
        def info(name: str) -> tuple[dict[str, Any], list[str], list[str]]:
            params = self.schemas[name].get("parameters") or {}
            props = params.get("properties") or {}
            free = [k for k, v in props.items() if (v or {}).get("type") == "string" and "enum" not in (v or {})]
            return props, list(params.get("required") or []), free
        for name in self.schemas:
            props, req, free = info(name)
            enums = [k for k in req if len((props.get(k) or {}).get("enum") or []) >= 2]
            longest = max((int((props[k] or {}).get("maxLength") or 0) for k in free), default=0)
            if req == free[:1] and len(props) == 1 and longest > 200:
                self.names.setdefault("speak", name)
            elif self._sets_cart(name) or (any(k in free for k in req) and len(req) <= 3
                                           and (self._qty_slot(props) in req or enums) and longest <= 200):
                self.names.setdefault("add", name)
                if enums:
                    self.fixed.setdefault("add", {enums[0]: props[enums[0]]["enum"][0]})
            elif len(enums) == 1 and len(req) == 1 and len(props[enums[0]]["enum"]) == 2 \
                    and len(free) <= 1 and all(k == enums[0] or (k in free and int((props[k] or {}).get("maxLength") or 1000) > 200)
                            for k in props):
                self.names.setdefault("buy", name)
                self.names.setdefault("none", name)
                self.fixed.setdefault("buy", {})
                self.fixed.setdefault("none", {})
            elif len(props) == 1 and req == free[:1] and req:
                if name not in self.readers:
                    self.readers.append(name)
            elif name in self.readers and len(props) > 1:
                self.readers.remove(name)              # a tool with more than its id to say is no reader
            elif len(props) == 1 and (next(iter(props.values())) or {}).get("type") == "array":
                self.names.setdefault("compare", name)
            elif free and len(props) >= 2:
                self.names.setdefault("find", name)
        if self.readers:
            self.names.setdefault("open", self.readers[0])
        bare = [n for n in self.schemas if not ((self.schemas[n].get("parameters") or {}).get("properties"))
                and n not in self.names.values()]
        lean = {n: _leaning(n + " " + str(self.schemas[n].get("description") or "")) for n in bare}
        ups = [n for n in bare if lean[n] > 0 and lean[n] == max(lean.values())]
        downs = [n for n in bare if lean[n] < 0 and lean[n] == min(lean.values())]
        if not self.names.get("buy") and len(ups) == 1:
            self.names["buy"] = ups[0]
        if not self.names.get("none") and len(downs) == 1 and downs[0] != self.names.get("buy"):
            self.names["none"] = downs[0]
        if self.names.get("buy") and self.names.get("buy") == self.names.get("none"):
            self._tell_finish_apart(TERSE)
            if not self._split_finish():
                self._mode_by_words(self.names["buy"])
            if not self._split_finish():
                self._first_choice_orders()
        add = self.names.get("add")
        if add and not self.names.get("cart"):
            said = set(_tokens(self.schemas[add].get("description"))) - set(_tokens(self.request))
            bare = [n for n in self.schemas if not ((self.schemas[n].get("parameters") or {}).get("properties"))
                    and n not in self.names.values() and set(_tokens(self.schemas[n].get("description"))) & said]
            if len(bare) == 1:
                self.names["cart"] = bare[0]
        find = self.names.get("find")
        if find:
            props = info(find)[0]
            words = self.text_arg(find)
            page = next((k for k, v in props.items() if (v or {}).get("type") == "integer"), None) or \
                next((k for k in info(find)[2] if k != words), None)
            label = next((k for k, v in props.items() if len((v or {}).get("enum") or []) >= 4), None)
            self.find_args = {k: v for k, v in (("words", words), ("page", page), ("label", label)) if v}
        if self.names.get("add"):
            self.add_mark = self.enum_args(self.names["add"], self.fixed.get("add", {})) or None

    def _setup_problems(self, r: dict[str, Any]) -> list[str]:
        try:
            return self._setup_faults(r)
        except (TypeError, AttributeError, KeyError, ValueError):
            return ["a reading of the wrong shape"] * 20   # a list or object where a tool's name: never kept

    def _setup_faults(self, r: dict[str, Any]) -> list[str]:
        if not isinstance(r, dict) or not r:
            return ["no reading"]
        t, out = _d(r.get("tools")), []
        for job in ("find", "compare", "speak", "add", "cart", "order", "decline"):
            if t.get(job) not in (None, "") and t.get(job) not in self.schemas:
                out.append("%s names no tool" % job)
        for job in ("add", "order"):
            if t.get(job) not in self.schemas:
                out.append("no %s tool" % job)
        if not isinstance(t.get("readers"), list) or any(x not in self.schemas for x in t.get("readers") or []):
            out.append("readers are not tools")
        for job, key in (("add", "add_args"), ("order", "order_args"), ("decline", "decline_args")):
            name = t.get(job)
            if name in self.schemas:
                props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
                for k, v in _d(t.get(key)).items():
                    if k not in props:
                        out.append("%s has no argument %s" % (name, k))
                    else:
                        _fit(props[k], v, k, out)
        order, decline = t.get("order"), t.get("decline")
        if order and order == decline and order in self.schemas:
            a, b = self.enum_args(order, _d(t.get("order_args"))), self.enum_args(order, _d(t.get("decline_args")))
            if not any(k in b and not _alike(v, b[k]) for k, v in a.items()):
                out.append("order and decline are not told apart")
        if not isinstance(r.get("items"), list) or not r["items"]:
            out.append("no items")
        return out

    def _take_setup(self, r: dict[str, Any]) -> None:
        t = _d(r.get("tools"))
        for job, key in (("find", "find"), ("compare", "compare"), ("speak", "speak"), ("add", "add"), ("cart", "cart"),
                         ("buy", "order"), ("none", "decline")):
            if isinstance(t.get(key), str) and t[key] in self.schemas:
                self.names[job] = t[key]
        for job, key in (("add", "add_args"), ("buy", "order_args"), ("none", "decline_args")):
            name = self.names.get(job)
            if not name:
                continue
            props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
            fixed = {}
            for k, v in _d(t.get(key)).items():
                issues: list[str] = []
                value = _fit(props[k], v, k, issues) if k in props else None
                if k in props and not issues:
                    fixed[k] = value
            self.fixed[job] = fixed
        if self.names.get("add"):
            self.add_mark = self.enum_args(self.names["add"], self.fixed.get("add", {})) or None
        self.readers = [x for x in _strings(t.get("readers")) if x in self.schemas
                        and x not in {self.names.get(j) for j in ("add", "buy", "none", "speak", "find")}]
        if not self.readers:
            # tools that take one listing id and do no other job read one listing
            for name, schema in self.schemas.items():
                params = schema.get("parameters") or {}
                slot = self.text_arg(name)
                if name not in self.names.values() and (params.get("required") or []) == [slot] and slot \
                        and len(params.get("properties") or {}) == 1 \
                        and int(((params.get("properties") or {}).get(slot) or {}).get("maxLength") or 0) <= 200:
                    self.readers.append(name)
        if self.readers:
            self.names["open"] = self.readers[0]
        find = self.names.get("find")
        props = ((self.schemas[find].get("parameters") or {}).get("properties") or {}) if find else {}
        self.find_args = {k: v for k, v in _d(t.get("find_args")).items() if isinstance(v, str) and v in props}
        mapped = [j for j in ("buy", "none") if self.names.get(j)]
        if len(mapped) == 1:
            name = self.names[mapped[0]]
            fprops = (self.schemas[name].get("parameters") or {}).get("properties") or {}
            if any(len((v or {}).get("enum") or []) >= 2 for v in fprops.values()):
                missing = "none" if mapped[0] == "buy" else "buy"
                self.names[missing] = name
                self.fixed.setdefault(missing, {})
        if self.names.get("buy") and not self._split_finish():
            self._tell_finish_apart()
            if not self._split_finish() and not self.easing(60.0):
                self._tell_finish_apart(FINISH_RETRY)
            if not self._split_finish():
                self._mode_by_words(self.names["buy"])
        self._take_ledger(r)

    def _take_ledger(self, r: dict[str, Any]) -> None:
        led, labels = self.ledger, self.store_labels()
        for raw in (r.get("items") if isinstance(r.get("items"), list) else [])[:6]:
            raw = _d(raw)
            try:
                qty = max(1, int(raw.get("qty") or 1))
            except (TypeError, ValueError):
                qty = 1
            try:
                cap = _as_text_says(float(raw["cap"]), self.request) if raw.get("cap") not in (None, "") else None
            except (TypeError, ValueError):
                cap = None
            label = raw.get("store_label") if isinstance(raw.get("store_label"), str) else None
            if label and labels:
                label = next((x for x in labels if _alike(x, label)), None)
            led.items.append({"name": str(raw.get("name") or "item"), "qty": qty, "store_label": label, "cap": cap,
                              "with": raw.get("with") if isinstance(raw.get("with"), str) and raw.get("with").strip() else None})
        if not led.items:
            led.items = [{"name": "item", "qty": 1, "store_label": None, "cap": None}]
        for it in led.items:                       # a tie names another item of the list, never the item itself
            other = it.get("with")
            it["with"] = next((n for n, x in enumerate(led.items) if x is not it and other
                               and (_alike(x["name"], other) or _fold(x["name"]) == _fold(other))), None)
        for raw in r.get("needs") if isinstance(r.get("needs"), list) else []:
            need = _need_of(raw, led.items, "stated", 0)
            if need and _need_key(need) not in {_need_key(n) for n in led.needs}:
                led.needs.append(need)
        obj = _d(r.get("objective"))
        kind = str(obj.get("kind") or "any").lower()
        led.objective = {"kind": kind if kind in ("cheapest", "best", "any") else "any",
                         "criterion": str(obj.get("criterion") or ""),
                         "direction": "min" if str(obj.get("direction") or "").lower() == "min" else "max"}
        if led.objective["kind"] == "best" and not led.objective["criterion"]:
            led.objective["kind"] = "any"
        budgets = _d(r.get("budgets"))
        try:
            if budgets.get("money_cap") not in (None, ""):
                led.budgets["money_cap"] = _as_text_says(float(budgets["money_cap"]), self.request)
        except (TypeError, ValueError):
            pass
        if led.budgets["money_cap"] is None and not led.several() and led.items[0].get("cap"):
            led.budgets["money_cap"] = led.items[0]["cap"] * led.items[0]["qty"]
        for key in ("questions_cap", "max_steps"):
            try:
                if budgets.get(key) not in (None, "") and int(budgets[key]) >= 0:
                    led.budgets[key] = int(budgets[key])
            except (TypeError, ValueError):
                pass
        if led.budgets.get("max_steps") and 8 <= led.budgets["max_steps"] < self.step_cap:
            self.step_cap = led.budgets["max_steps"]
        led.in_stock = r.get("in_stock") is not False
        proc = _d(r.get("procedures"))
        led.rules = _texts(proc.get("rules"))[:8]
        led.steps = [s for s in (_step_of(x) for x in (proc.get("steps") if isinstance(proc.get("steps"), list) else []))
                     if s][:8]
        led.decline_rule = r.get("decline_rule") if isinstance(r.get("decline_rule"), str) and r["decline_rule"].strip() else None
        led.story_hints = [{"hint": str(_d(h).get("hint") or ""), "property": str(_d(h).get("property") or "")}
                           for h in (r.get("story_hints") if isinstance(r.get("story_hints"), list) else [])
                           if str(_d(h).get("property") or "").strip()][:4]
        led.no_preference = _texts(r.get("no_preference"))[:8]
        self._take_refs(r.get("references"))

    # -------------------------------------------------------------- listings
    def _collect_fields(self, value: Any) -> None:
        """Short plain fields of every listing record in a result, under plain property names."""
        if isinstance(value, list):
            for v in value:
                self._collect_fields(v)
            return
        if not isinstance(value, dict):
            return
        key = self.index.id_key(value)
        if key:
            pid = str(value[key])
            own = self.fields.setdefault(pid, {})
            said: dict[tuple[str, str], set[str]] = {}
            kept = self.said_by.setdefault(pid, {})
            def note(name: str, shown: Any, table: str = "") -> None:
                # keys of one name within one table that share no word: the listing contradicts itself
                words = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", str(shown).lower()))
                if words and (table, name) in said and not words & said[(table, name)]:
                    self.torn.add(pid)
                if words:
                    said.setdefault((table, name), words)
                    kept[(table, name)] = said[(table, name)]
            for k, v in value.items():
                if k != key and isinstance(v, (str, int, float, bool)) and len(str(v)) <= 120:
                    name = _prop_name(k)
                    if name:
                        own[name] = str(v)
                        note(name, v)
                elif isinstance(v, dict) and not self.index.id_key(v):
                    # a small property table (too small to count as a sheet) still says what the listing is
                    for k2, v2 in v.items():
                        if isinstance(v2, (str, int, float, bool)) and len(str(v2)) <= 120 and _prop_name(k2):
                            own.setdefault(_prop_name(k2), str(v2))
                            note(_prop_name(k2), v2, str(k))
        for v in value.values():
            if isinstance(v, (dict, list)):
                self._collect_fields(v)

    def fields_of(self, pid: str) -> dict[str, str]:
        out = dict(self.fields.get(pid, {}))
        out.update(self.index.sheet_of.get(pid, {}))
        return out

    def role_names(self) -> set[str]:
        """Property names that only say which listing it is, what it costs or whether it is on hand."""
        return {_prop_name(v) for k, v in self.index.roles.items() if isinstance(v, str)}

    def opened(self, idx: int | None = None) -> list[str]:
        pids = [p for p in self.index.seen_order if p in self.index.opened]
        return pids if idx is None else [p for p in pids if self.item_of(p) == idx]

    def label_pids(self, idx: int) -> list[str]:
        """Opened listings of the item's store label (all opened listings of the item when the label is unknown)."""
        label = self.label_of(idx)
        mine = self.opened(idx)
        if not label:
            return mine
        return [p for p in mine if not self.index.rows.get(p, {}).get("category")
                or _alike(self.index.rows[p]["category"], label)]

    def unread(self, idx: int) -> list[str]:
        """Rows seen for an item, never opened, in stock, under its store label and within its limit."""
        label, cap = self.label_of(idx), self.unit_cap(idx)
        out = []
        for p in self.index.seen_order:
            row = self.index.rows.get(p, {})
            if p in self.index.opened or p in self.gone or self.item_of(p) != idx or row.get("in_stock") is False:
                continue
            if label and row.get("category") and not _alike(row["category"], label):
                continue
            if cap and isinstance(row.get("price"), (int, float)) and row["price"] > cap + 1e-9:
                continue
            out.append(p)
        return out

    def by_mention(self, pids: list[str], idx: int) -> list[str]:
        """Rows in the order to open them: those whose own text names more needed values first (this only orders the
        reading; a listing is never bought on what its text names), then the objective's order."""
        values = [v for n in self.ledger.needs_for(self.ledger.items[idx]["name"]) if n["op"] != "is not"
                  for v in (n["value"] if isinstance(n["value"], list) else [n["value"]])] if self.ledger.items else []
        def named(p: str) -> int:
            words = set(_tokens(self.index.text_of.get(p, "")))
            return sum(1 for v in values if _tokens(v) and set(_tokens(v)) <= words)
        order = self.prefer(pids)
        return sorted(order, key=lambda p: -named(p))

    def line_cap(self, idx: int) -> float | None:
        """The most one unit of an item may cost: its own limit, else what the total leaves after the other lines."""
        own = self.unit_cap(idx)
        cap = self.ledger.budgets.get("money_cap")
        if own or not cap or not self.ledger.several():
            return own
        others = sum((self.index.rows.get(p, {}).get("price") or 0) * self.line_qty.get(p, 1)
                     for p, i in self.cart_lines() if i != idx)
        qty = max(1, int(self.ledger.items[idx].get("qty") or 1))
        return max(0.0, (float(cap) - others) / qty)

    def unit_cap(self, idx: int) -> float | None:
        item = self.ledger.items[idx] if idx < len(self.ledger.items) else {}
        caps = [float(item["cap"])] if item.get("cap") else []
        cap = self.ledger.budgets.get("money_cap")
        if cap and not self.ledger.several():
            caps.append(float(cap) / max(1, int(item.get("qty") or 1)))
        return min(caps) if caps else None

    # -------------------------------------------------------------- the listing check
    def check(self, pid: str, full: bool = True, moment: str = "") -> dict[str, Any]:
        """The listing against the ledger (see _check): a need that holds only in some case counts when that case applies,
        and leaves the listing open while the case cannot be told. For an item that must fit another item of the list,
        also against the listing in the cart for that other item: their records must share a model code."""
        item = self.ledger.items[self.item_of(pid)]["name"] if self.ledger.items else ""
        result, undecided = self._case_check(pid, full, moment)
        if undecided:
            result = dict(result, unclear=list(result.get("unclear") or []) + [_need_text(n) for n in undecided])
        fault = self._tie_fault(pid)
        if fault and result.get("ok"):
            result = dict(result, ok=False, why=list(result.get("why") or []) + [fault], facts=list(result.get("facts") or []) + [fault])
        return result

    def _codes_of(self, pid: str) -> set[tuple[str, str]]:
        """Model codes a listing's records write: a capitalised word or letters with a run of digits ("ABC-2110",
        "Name 211"), a trailing run of x standing for any digit ("ABC31xx"). A plain code written after a series in
        the same sentence, in a later clause, and falling inside that series is one the series leaves out; it comes
        back marked with a leading "!" on its word, so a match against it fails (see _codes_meet)."""
        texts = _text_values(self.index.records.get(pid, []))[:400] + [str(self.index.rows.get(pid, {}).get("title") or "")]
        found: set[tuple[str, str]] = set()
        left_out: set[tuple[str, str]] = set()
        for sentence in (part for text in texts for part in re.split(r"[.;\n|]", text)):
            earlier: list[tuple[str, str]] = []    # series written in an earlier clause of the sentence
            for clause in sentence.split(","):
                here = [(a.upper(), b.lower()) for a, b in
                        re.findall(r"\b([A-Z][A-Za-z]{1,8})[- ]?(\d{2,5}|\d{1,4}[xX]{1,3})\b", clause)]
                for code in here:
                    if "x" not in code[1] and any(_in_series(code, s) for s in earlier):
                        left_out.add(code)
                    else:
                        found.add(code)
                earlier += [c for c in here if "x" in c[1]]
        return (found - left_out) | {("!" + a, b) for a, b in left_out}

    @staticmethod
    def _codes_meet(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> bool:
        """True when a code of one set and a code of the other name the same model (a series takes every model of
        its run), and neither listing leaves that model out."""
        out_a = {(w[1:], d) for w, d in a if w.startswith("!")}
        out_b = {(w[1:], d) for w, d in b if w.startswith("!")}
        for p, m in a:
            for q, n in b:
                if p != q or p.startswith("!"):
                    continue
                if m == n or _in_series((q, n), (p, m)) and (q, n) not in out_a \
                        or _in_series((p, m), (q, n)) and (p, m) not in out_b:
                    return True
                if "x" in m and "x" in n and len(m) == len(n) and (
                        m.rstrip("x").startswith(n.rstrip("x")) or n.rstrip("x").startswith(m.rstrip("x"))):
                    return True                    # two series that share a run of models
        return False

    def _tie_fault(self, pid: str) -> str:
        """For a listing of an item the request ties to another item of the list: why it cannot go with the cart's
        listing of that other item, when the listing names model codes, none meets that line's codes, and another
        opened listing of the item does meet them; else ''."""
        idx = self.item_of(pid)
        if idx >= len(self.ledger.items):
            return ""
        n = self.ledger.items[idx].get("with")
        if not isinstance(n, int) or n == idx or n >= len(self.ledger.items):
            return ""
        line = next((p for p, i in self.cart_lines() if i == n and p != pid), None)
        if not line:
            return ""
        mine, theirs = self._codes_of(pid), self._codes_of(line)
        if not mine or not theirs or self._codes_meet(mine, theirs):
            return ""
        if not any(self._codes_meet(self._codes_of(q), theirs) for q in self.opened(idx) if q != pid and q not in self.gone):
            return ""                              # no opened listing of the item meets them: no evidence to go on
        names = sorted({"%s %s" % c for c in theirs if not c[0].startswith("!")})[:4]
        return "it names none of the model codes of the cart's %s %s (%s); another opened %s does" % (
            self.ledger.items[n]["name"], line, ", ".join(names), self.ledger.items[idx]["name"])

    def _check(self, pid: str, full: bool = True, moment: str = "") -> dict[str, Any]:
        """A listing against the ledger, once per (listing, version). Store facts and a code matcher settle the clear
        cells; only the unclear and clash ones go to the CHECK chain, and code facts stand over any model claim. For a
        listing about to go in the cart (moment "add"), cells left open get one strong-model reading per trip."""
        key = (pid, self.ledger.version)
        cached = self.checks.get(key)
        if cached and (cached.get("nf") != len(self.fields_of(pid))
                       or not cached["full"] and cached.get("seen") != len(self.fields)):
            cached = None          # the listing showed more since, or (code-only reading) the store did
        if cached and (cached["full"] or not full):
            return self._escalate(key, cached) if moment == "add" and self._may_escalate(cached) else cached
        idx = self.item_of(pid)
        row = self.index.rows.get(pid, {})
        item = self.ledger.items[idx] if idx < len(self.ledger.items) else {"name": "item", "qty": 1}
        why: list[str] = []
        if pid in self.gone:
            why.append("no longer available")
        if row.get("in_stock") is False and self.ledger.in_stock:
            why.append("not in stock")
        label = self.label_of(idx)
        if label and row.get("category") and not _alike(row["category"], label):
            why.append("listed under %s, not %s" % (row["category"], label))
        cap = self.unit_cap(idx)
        if cap and isinstance(row.get("price"), (int, float)) and row["price"] > cap + 1e-9:
            why.append("price %s is over the limit of %s a unit" % (row["price"], round(cap, 2)))
        total = self.ledger.budgets.get("money_cap")
        if self.ledger.several() and total and isinstance(row.get("price"), (int, float)) \
                and row["price"] * max(1, int(item.get("qty") or 1)) > float(total) + 1e-9:
            why.append("price %s for %s alone is over the whole limit of %s" % (row["price"], item.get("qty"), total))
        fields = self.fields_of(pid)
        peers = [p for p in self.label_pids(idx) if p != pid]
        cells: list[tuple[dict[str, Any], str, str]] = []
        for need in self.ledger.needs_for(item["name"]):
            shown = _field(fields, need["property"])
            siblings = self._siblings(fields, need["property"])
            if shown is None and siblings:
                shown = siblings[0]
            if shown is None:
                carriers = [p for p in peers if _field(self.fields_of(p), need["property"]) is not None]
                # a property the shopper named is one the store uses: a listing that states nothing for it, while
                # another opened listing of the item does, cannot be shown to fit (its own words may still name it)
                named_by_shopper = need.get("source") == "shopper" and bool(carriers)
                if need["op"] != "is not" and (named_by_shopper or (len(carriers) >= 2 and 2 * len(carriers) > len(peers))):
                    text = _fold(_dump(self.index.records.get(pid, []), 4000))
                    named = any(_fold(v) and _fold(v) in text for v in (need["value"] if isinstance(need["value"], list)
                                                                         else [need["value"]]))
                    cells.append((need, "clash" if named else "against",
                                  "has no %s field, which most listings here carry" % need["property"]))
                elif not carriers and need["op"] != "is not" and need.get("source") not in ("shopper", "reference") \
                        and len([p for p in peers if self.index.sheet_of.get(p) or not self.sheet_readers]) >= NO_FIELD_PEERS \
                        and not any(_fold(v) and _fold(v) in _fold(_dump(self.index.records.get(q, []), 4000))
                                    for q in [pid] + peers
                                    for v in (need["value"] if isinstance(need["value"], list) else [need["value"]])):
                    # the request's reading named a property that no opened listing of the item carries, and no
                    # opened listing's own words name the value either: the store does not describe its goods that
                    # way at all, so a listing's silence on it is no evidence against it (when some listing's words
                    # do name the value, the need is real and the reading of each listing's words stands)
                    cells.append((need, "unclear", "no %s field on any opened listing of the item%s" % (need["property"], NO_FIELD)))
                else:
                    cells.append((need, "unclear", "no %s field" % need["property"]))
                continue
            vocab = {_fold(x) for p in self.fields for x in re.split(r"[,;/|]", str(_field(self.fields_of(p), need["property"]) or ""))
                     if _fold(x)}
            # the key that names the property stands; a sibling key only speaks when that one says nothing clear
            rank = {"met": 0, "unclear": 1, "clash": 1, "against": 2}
            exact = any(_fold(k) == _fold(need["property"]) or _keyfold(k) == _keyfold(need["property"])
                        or _key_set(k) == _key_set(need["property"]) for k in fields)
            state, note = _cell(shown, need["op"], need["value"], vocab)
            if state in ("unclear", "clash") and siblings:
                alt = min((_cell(v, need["op"], need["value"], vocab) for v in siblings), key=lambda c: rank.get(c[0], 1))
                if state == "unclear" or alt[0] == "met":
                    state, note = alt
            if need["op"] in ("is", "one of") and len(siblings) > 1:
                # keys of one property read together: met when one shows the value and the others only narrow it
                # (one key gives a single value, a sibling lists it among others); else the naming key stands
                sets = [(_cell(v, need["op"], need["value"], vocab)[0], _pieces(v)) for v in siblings]
                holding = [got for cell, got in sets if cell == "met"]
                if holding and state != "met" and all(got <= set().union(*holding) for _, got in sets):
                    state, note = "met", "shows %s" % " / ".join(siblings)
                elif holding and state == "against" and not self._key_carries(fields, shown, need["value"]):
                    # the key read first never shows the wanted value in what this store has shown: it may hold
                    # another side of the property; a sibling shows the value, so the model reads it
                    state, note = "clash", "shows %s" % " / ".join(siblings)
            if state != "met" and need["op"] not in ("is not", "at most", "at least") \
                    and self._said_together(need, [shown] + siblings):
                # values a listing gives side by side may be one value spelled twice, or not: the model decides
                state, note = "clash", "this store gives these values side by side"
            if state == "met" and not exact:
                note += NEAR_KEY                   # met only under a key that spells the property otherwise
            cells.append((need, state, note))
        model_done, reached = False, False
        open_ = [n for n, c in enumerate(cells) if c[1] in ("unclear", "clash")]
        clean = not why and not any(c[1] == "against" for c in cells)
        if full and clean and open_ and pid in self.index.opened and not self.easing(30.0):
            model_done = True
            reached = bool(self._settle(pid, item["name"], cells, open_, CHECK))
            open_ = [n for n in open_ if cells[n][1] in ("unclear", "clash")]
        # open cells that never reached a model (time short, or no answer) leave the check short of a full one
        result = self._check_result(key, idx, why, cells, open_, full and (reached or not (clean and open_)), reached)
        return self._escalate(key, result) if moment == "add" and self._may_escalate(result) else result

    def _settle(self, pid: str, item: str, cells: list[tuple[dict[str, Any], str, str]], which: list[int],
                chain: list[str], caps: tuple[float, ...] = (15.0, 10.0)) -> set[int]:
        """The model's reading of the open cells `which`: "yes" meets a need, "no" breaks it, anything else leaves
        it open. Returns the cells the model answered at all."""
        elsewhere = {}
        for n in which:
            prop = cells[n][0]["property"]
            vals = list(dict.fromkeys(str(_field(self.fields_of(p), prop)) for p in self.fields
                                      if p != pid and _field(self.fields_of(p), prop) is not None))
            elsewhere[prop] = vals[:6]
        got = self._read(chain, LISTING_CHECK, {
            "item": item, "listing": pid, "records": self.index.records.get(pid, [])[-4:],
            "needs": [{"n": n, "property": cells[n][0]["property"], "op": cells[n][0]["op"],
                       "value": cells[n][0]["value"], "code_note": cells[n][2]} for n in which],
            "values_elsewhere": elsewhere}, 600, caps=caps)
        seen: set[int] = set()
        for entry in got.get("needs") if isinstance(got.get("needs"), list) else []:
            entry = _d(entry)
            try:
                n = int(entry.get("n"))
            except (TypeError, ValueError):
                continue
            if n in which:
                seen.add(n)
                said = str(entry.get("fits") or "").lower()
                if said == "yes":
                    cells[n] = (cells[n][0], "met", str(entry.get("why") or "")[:80])
                elif said == "no" and not str(cells[n][2]).endswith(NO_FIELD):
                    cells[n] = (cells[n][0], "against", str(entry.get("why") or "")[:80] + READ_NO)
        return seen

    def _check_result(self, key: tuple[str, int], idx: int, facts: list[str], cells: list[tuple[dict[str, Any], str, str]],
                      open_: list[int], full: bool, asked: bool) -> dict[str, Any]:
        # an unsettled clash fails; a loose one (code could not settle it) fails only once a model has read it
        final = [(c[0], "against" if asked or not str(c[2]).endswith(LOOSE) else "unclear", c[2]) if c[1] == "clash"
                 else c for c in cells]
        against = ["%s: %s" % (_need_text(c[0]), c[2]) for c in final if c[1] == "against"]
        result = {"ok": not facts and not against, "why": facts + against, "item": idx, "full": full,
                  "met": sum(1 for c in final if c[1] == "met"),
                  "near": sum(1 for c in final if c[1] == "met" and str(c[2]).endswith(NEAR_KEY)),
                  "unclear": [_need_text(c[0]) for c in final if c[1] == "unclear"], "asked_model": asked,
                  "cells": cells, "open": open_, "facts": facts,
                  "nf": len(self.fields_of(key[0])), "seen": len(self.fields)}
        self.checks[key] = result
        return result

    def _may_escalate(self, result: dict[str, Any]) -> bool:
        """The one strong reading of the trip: for open cells of a listing with no store fact or code clash against it."""
        return bool(result.get("open")) and not self.escalated and not self.easing(40.0 + ESCALATE_WAIT) \
            and self.spent + self.call_cost.get(ESCALATE, CALL_GUESS) <= TRIP_SHARE \
            and not result.get("facts") \
            and not any(c[1] == "against" for c in result.get("cells") or [])

    def _escalate(self, key: tuple[str, int], result: dict[str, Any]) -> dict[str, Any]:
        self.escalated = True
        pid, idx = key[0], result["item"]
        item = self.ledger.items[idx]["name"] if idx < len(self.ledger.items) else "item"
        cells = list(result["cells"])
        seen = self._settle(pid, item, cells, result["open"], [ESCALATE], caps=(ESCALATE_WAIT,))
        still = [n for n in result["open"] if cells[n][1] in ("unclear", "clash")]
        return self._check_result(key, idx, result["facts"], cells, still, result["full"] or bool(seen),
                                  bool(seen) or result.get("asked_model", False))

    def _key_carries(self, fields: dict[str, str], shown: Any, value: Any) -> bool:
        """Whether the key a value was read under shows the wanted value for any listing of the store."""
        keys = {_keyfold(k) for k, v in fields.items() if v == shown}
        wants = {_fold(x) for x in (value if isinstance(value, list) else [value]) if _fold(x)}
        return any(_keyfold(k) in keys and wants & _pieces(v) for p in self.fields for k, v in self.fields_of(p).items())

    @staticmethod
    def _siblings(fields: dict[str, str], prop: str) -> list[str]:
        """Values under every key whose name holds all the property's words (one property, spelled several ways)."""
        words = {w.rstrip("s") for w in _tokens(prop)}
        return [v for k, v in fields.items() if words and words <= {w.rstrip("s") for w in _tokens(k)}]

    def _said_together(self, need: dict[str, Any], shown: list[str]) -> bool:
        """Two values mean the same in this store when one listing gives both side by side under keys of the same
        property (one key says one, a sibling key the other)."""
        wants = {_fold(x) for x in (need["value"] if isinstance(need["value"], list) else [need["value"]]) if _fold(x)}
        seen = {_fold(x) for x in shown if x is not None and _fold(x)}
        for other in self.fields:
            values = {_fold(v) for v in self._siblings(self.fields_of(other), need["property"]) if _fold(v)}
            if values & wants and values & seen:
                return True
        return False

    def solid(self, pid: str, full: bool = True) -> bool:
        """Evidence code may act on by itself: the check passes with no cell left open (a model's "unknown" leaves
        it open)."""
        r = self.check(pid, full=full)
        return bool(r["ok"] and not r["unclear"])

    def replaces(self, new: str, old: str) -> bool:
        """`new` may take `old`'s place by code alone: solid, and it meets every need `old` met."""
        if not self.solid(new):
            return False
        had = {_need_text(c[0]) for c in self.check(old, full=False).get("cells") or [] if c[1] == "met"}
        got = {_need_text(c[0]) for c in self.check(new).get("cells") or [] if c[1] == "met"}
        return had <= got

    def passed(self, idx: int, exclude: set[str] | None = None) -> list[str]:
        """Opened listings of an item that this version's full check passed, in the order the objective prefers."""
        exclude = exclude or set()
        v = self.ledger.version
        pids = [p for p in self.opened(idx) if p not in exclude and p not in self.gone
                and self.checks.get((p, v), {}).get("ok") and self.checks[(p, v)].get("full")]
        return self.prefer(pids)

    def prefer(self, pids: list[str]) -> list[str]:
        """Listings in the order to take them: those that leave no need open first, then by the request's way of choosing;
        with no way named, the fullest evidence first, then the lowest price."""
        kind = self.ledger.objective.get("kind")
        def price(p: str) -> float:
            value = self.index.rows.get(p, {}).get("price")
            return float(value) if isinstance(value, (int, float)) else 1e18
        v = self.ledger.version
        def near(p: str) -> bool:
            # a listing that carries a need only under a near key is the weaker evidence
            return bool(self.checks.get((p, v), {}).get("near"))
        def open_needs(p: str) -> bool:
            # a listing that leaves a need unclear is weaker than one whose record states every need
            return bool(self.check(p, full=False)["unclear"])
        if kind == "best":
            sign = -1.0 if self.ledger.objective.get("direction") != "min" else 1.0
            return sorted(pids, key=lambda p: (self.measure(p) is None, open_needs(p), near(p), sign * (self.measure(p) or 0.0),
                                               price(p)))
        if kind == "cheapest":
            return sorted(pids, key=lambda p: (open_needs(p), price(p), near(p)))
        def stated(p: str) -> int:
            # with no way of choosing named, the listing whose record states the most of what was asked is the safest
            cells = self.checks.get((p, v), {}).get("cells") or []
            return sum(1 for c in cells if len(c) > 1 and c[1] == "met")
        return sorted(pids, key=lambda p: (open_needs(p), near(p), -stated(p), price(p)))

    def measure(self, pid: str) -> float | None:
        shown = _field(self.fields_of(pid), self.ledger.objective.get("criterion") or "")
        figures = _figures(str(shown)) if shown is not None else []
        return figures[0] if figures else None

    def rechoose(self, idx: int, exclude: set[str]) -> str | None:
        """The pick for an item chosen again from scratch under the current ledger: under a cheapest rule the cheap end
        of the item's label is read again first (unread rows within the line's share of the limit, cheapest first);
        then the listing that settles every need, in the objective's order; failing that, any that fits."""
        key = (idx, self.ledger.version)
        if self.ledger.objective.get("kind") == "cheapest" and self.steps_left() > self.reserve() // 2 + 3 \
                and key not in self.rechosen and not self.easing(40.0):
            self.rechosen.add(key)
            cap = self.line_cap(idx)
            cheap = sorted((p for p in self.unread(idx) if isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                            and (not cap or self.index.rows[p]["price"] <= cap + 1e-9)),
                           key=lambda p: self.index.rows[p]["price"])[:2]
            self._open_all(cheap, "CHEAP END READ AGAIN AFTER THE CHANGE", every=False)
        quick = [p for p in self.opened(idx) if p not in exclude and p not in self.gone and self.check(p, full=False)["ok"]]
        for pid in [p for p in self.prefer(quick) if not self.check(p, full=False)["unclear"]][:3]:
            if self.solid(pid):
                return pid
        return None

    def backup(self, idx: int, exclude: set[str]) -> str | None:
        """The listing to swap to for an item: the code check's fitting candidates in the objective's order, the
        first two confirmed by a full check."""
        quick = [p for p in self.opened(idx) if p not in exclude and p not in self.gone and self.check(p, full=False)["ok"]]
        for pid in self.prefer(quick)[:2]:
            if self.check(pid)["ok"]:
                return pid
        return None

    # -------------------------------------------------------------- the cart
    def cart_lines(self) -> list[tuple[str, int]]:
        return [(p, self.item_of(p)) for p in self.basket if p not in self.gone]

    def cart_fits(self) -> bool:
        lines = self.cart_lines()
        if not lines:
            return False
        for n in range(len(self.ledger.items)):
            mine = [p for p, i in lines if i == n]
            if len(mine) != 1 or not self.check(mine[0])["ok"]:
                return False
        return len(lines) == len(self.ledger.items)

    def _line_slots(self, name: str) -> tuple[str | None, str | None, str | None]:
        """For a tool that sets the cart from a list of lines: the list argument, each line's id field, its quantity."""
        props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
        arr = next((k for k, v in props.items() if (v or {}).get("type") == "array"
                    and ((v or {}).get("items") or {}).get("type") == "object"), None)
        if not arr:
            return None, None, None
        sub = props[arr].get("items") or {}
        sprops = sub.get("properties") or {}
        id_slot = next((k for k in list(sub.get("required") or []) + list(sprops)
                        if (sprops.get(k) or {}).get("type") == "string" and "enum" not in (sprops.get(k) or {})), None)
        return arr, id_slot, self._qty_slot(sprops)

    def _add_ids(self, name: str, args: dict[str, Any]) -> list[str]:
        """The listings an add call names: its id argument (each line's id field) only, never other free text."""
        if self._sets_cart(name):
            arr, id_slot, q_slot = self._line_slots(name)
            # a line at quantity 0 is a removal, not a listing being added
            return list(dict.fromkeys(str(_d(x).get(id_slot)) for x in (args.get(arr) or [])
                                      if id_slot and _d(x).get(id_slot) not in (None, "")
                                      and not (q_slot and _d(x).get(q_slot) == 0)))
        slot = self._id_slot(name)
        if not slot:
            props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
            slot = next((k for k, v in args.items() if isinstance(v, str) and v in self.index.ids
                         and "enum" not in (props.get(k) or {})), None)
        value = args.get(slot) if slot else None
        return [str(value)] if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() else []

    def _settles(self, pid: str, needs: list[str]) -> bool:
        """A listing that passes and meets every one of the needs named (by their text) on its own fields."""
        r = self.check(pid, full=False)
        met = {_need_text(c[0]) for c in r.get("cells") or [] if c[1] == "met"}
        return bool(r["ok"]) and set(needs) <= met

    def _is_removal(self, name: str, args: dict[str, Any]) -> bool:
        """An add-tool call that takes a line out: one of add_mark's own mode values differs, or the quantity is 0."""
        mark = self.enum_args(name, args)
        if self.add_mark and any(k in mark and not _alike(mark[k], v) for k, v in self.add_mark.items()):
            return True
        q_slot = self._qty_slot((self.schemas[name].get("parameters") or {}).get("properties") or {})
        return q_slot is not None and args.get(q_slot) == 0

    def _qty_slot(self, props: dict[str, Any]) -> str | None:
        return next((k for k, v in props.items() if (v or {}).get("type") == "integer" and "enum" not in (v or {})), None)

    def _add_args(self, pid: str, qty: int) -> dict[str, Any] | None:
        """Arguments that put one listing in the cart at a quantity, built from the add tool's schema (the model fills
        them only when the schema leaves the shape unclear)."""
        name = self.names.get("add")
        if not name:
            return None
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        args: dict[str, Any] = dict(self.fixed.get("add", {}))
        if self._sets_cart(name):
            arr = next(k for k, v in props.items() if (v or {}).get("type") == "array")
            sub = (props[arr].get("items") or {})
            sprops = sub.get("properties") or {}
            id_slot = next((k for k in (sub.get("required") or []) + list(sprops)
                            if (sprops.get(k) or {}).get("type") == "string" and "enum" not in (sprops.get(k) or {})), None)
            q_slot = self._qty_slot(sprops)
            if not id_slot:
                return self._model_add_args(pid, qty)
            lines = []
            rest = [] if self.additive else [x for x in self.basket if x != pid and x not in self.gone]
            for p in rest + [pid]:
                line = {id_slot: p}
                if q_slot:
                    line[q_slot] = qty if p == pid else self.line_qty.get(p, 1)
                lines.append(line)
            zero_ok = q_slot and ((sprops.get(q_slot) or {}).get("minimum", 0) or 0) <= 0 and not self.zero_refused \
                and self.keeps_unsent
            if zero_ok:                            # a line that can no longer be bought leaves the store's cart too
                lines += [{id_slot: p, q_slot: 0} for p in sorted(self.gone & (self.carted | set(self.basket))) if p != pid]
            args[arr] = lines
        else:
            slot = self._id_slot(name)
            if not slot:
                return self._model_add_args(pid, qty)
            args[slot] = pid
            for k in params.get("required") or []:
                sch = props.get(k) or {}
                if k not in args and k != slot and sch.get("type") == "string" and "enum" not in sch:
                    args[k] = "wanted by the shopper"[: int(sch.get("maxLength") or 40)]
            q_slot = self._qty_slot(props)
            if q_slot:
                args[q_slot] = qty
        for k in params.get("required") or []:
            if k not in args and "enum" in (props.get(k) or {}):
                args[k] = (self.add_mark or {}).get(k, props[k]["enum"][0])
        issues: list[str] = []
        args = _fit(params, args, name, issues)
        return None if issues else args

    def _model_add_args(self, pid: str, qty: int) -> dict[str, Any] | None:
        name = self.names["add"]
        got = self._read(TERSE, CART_ARGS, {"tool": self.schemas[name], "listing": pid, "quantity": qty}, 300)
        issues: list[str] = []
        args = _fit(self.schemas[name].get("parameters") or {}, got, name, issues)
        return None if issues or pid not in _ids_in(args, self.index.ids | {pid}) else args

    def _remove_args(self, pid: str) -> dict[str, Any] | None:
        """Arguments that take one listing out of the cart: the add tool's other mode, a zero quantity, or the cart
        set again without it."""
        name = self.names.get("add")
        if not name:
            return None
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        if self._sets_cart(name):
            arr, id_slot, q_slot = self._line_slots(name)
            sprops = ((props.get(arr) or {}).get("items") or {}).get("properties") or {}
            issues: list[str] = []
            if id_slot and q_slot and ((sprops.get(q_slot) or {}).get("minimum", 0) or 0) <= 0 and not self.zero_refused:
                # the rest of the basket at its quantities and this line at 0: right whether the store replaces the
                # cart with the list or merges the list into it
                lines = [] if self.additive else \
                    [{id_slot: p, q_slot: self.line_qty.get(p, 1)} for p in self.basket if p != pid and p not in self.gone]
                out = _fit(params, {**dict(self.fixed.get("add", {})), arr: lines + [{id_slot: pid, q_slot: 0}]}, name, issues)
                return None if issues else out
            args = self._add_args(pid, 1)
            if args is None:
                return None
            rest = [line for line in args[arr] if pid not in _ids_in(line, {pid})]
            out = _fit(params, {**args, arr: rest}, name, issues)
            return None if issues else out
        args = self._add_args(pid, self.units_held(pid))
        if args is None:
            return None
        for k, v in (self.add_mark or {}).items():
            other = next((m for m in (props.get(k) or {}).get("enum") or [] if not _alike(m, v)), None)
            if other is not None:
                args[k] = other
                return args
        q_slot = self._qty_slot(props)
        if q_slot and (props[q_slot].get("minimum", 0) or 0) <= 0:
            args[q_slot] = 0
            return args
        return None

    def units_held(self, pid: str) -> int:
        """Units of a listing the cart holds: the store's own count when a result reported one, else what we sent."""
        return max(1, int(self.held.get(pid) or self.line_qty.get(pid) or 1))

    def _note_counts(self, name: str, args: dict[str, Any], inner: Any, job: str) -> None:
        """Read the units the store reports per listing. The field is learned from the first plain add: the one
        whole-number field beside the listing's id that equals the quantity sent."""
        records: list[dict[str, Any]] = []

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                if any(isinstance(v, str) and v in self.index.ids for v in x.values()):
                    records.append(x)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(inner)
        props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
        q_slot = self._qty_slot(props)
        if self.count_key is None and job == "add" and q_slot and not self._sets_cart(name) \
                and not self._is_removal(name, args) and isinstance(args.get(q_slot), int):
            sent = args[q_slot]
            for rec in records:
                if args.get(self.text_arg(name) or "") in rec.values():
                    keys = [k for k, v in rec.items() if isinstance(v, int) and not isinstance(v, bool) and v == sent]
                    if len(keys) == 1:
                        self.count_key = keys[0]
        if not self.count_key:
            return
        if job == "cart" and _ids_in(inner, self.index.ids):
            self.held = {}
        if job == "add" and not self._sets_cart(name):
            for pid in self._add_ids(name, args):
                self.held.pop(pid, None)           # a result that leaves a listing out reports nothing held for it
        removal = job == "add" and self._is_removal(name, args)
        for rec in records:
            n = rec.get(self.count_key)
            pid = next((v for v in rec.values() if isinstance(v, str) and v in self.index.ids), None)
            if pid and isinstance(n, int) and not isinstance(n, bool):
                if removal and q_slot and n == args.get(q_slot):
                    continue                       # the units this call took out, not the units left
                self.held[pid] = n

    def cart_summary(self) -> str:
        parts, total = [], 0.0
        for pid in [p for p in self.basket if p not in self.gone]:
            row = self.index.rows.get(pid, {})
            qty = self.line_qty.get(pid, self.ledger.items[self.item_of(pid)]["qty"] if self.ledger.items else 1)
            price = row.get("price")
            if isinstance(price, (int, float)):
                total += price * qty
            parts.append("%d x %s (%s) at %s" % (qty, str(row.get("title") or "listing")[:80], pid,
                                                 price if price is not None else "the listed price"))
        return "Here is the cart before I place the order: %s; total %s." % ("; ".join(parts) or "empty",
                                                                         int(total) if total == int(total) else round(total, 2))

    # -------------------------------------------------------------- after a call
    def _split_notes(self, name: str, observation: Any) -> tuple[Any, list[str]]:
        """Notes found by the shape of a result: sentences beside it, a two-part wrapper, or a key holding sentences
        that this tool never returned before (never inside a listing's own record). Notes on a cart call are also kept for
        the code that re-plans the cart."""
        inner, lines = _notes_beside(observation)
        if isinstance(observation, dict):
            seen = self.keys_seen.get(name)
            record = any(_is_id_key(k) and isinstance(v, (str, int)) for k, v in observation.items())
            if seen is not None and not lines and not record and len(observation) >= 2 and not _failed(observation):
                fresh = [k for k in observation if k not in seen and _sentences(observation[k])]
                if fresh and len(fresh) < len(observation):
                    lines = [s for k in fresh for s in _sentences(observation[k])]
                    rest = {k: v for k, v in observation.items() if k not in fresh}
                    inner = next(iter(rest.values())) if len(rest) == 1 else rest
            if not lines and not _failed(observation):
                self.keys_seen.setdefault(name, set()).update(observation)
        if lines and self.role(name) in ("add", "cart"):
            self.note_queue.append(list(lines))
        return inner, lines

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> list[str]:
        notes: list[str] = []
        job = self.role(name, args)
        inner, lines = self._split_notes(name, observation)
        problem = _failed(observation)
        if job == "speak":
            self.talk_sent += 1
            if problem:
                self.talk_hard_cap = self.talk_sent - 1
                notes.append("the shopper takes no more messages")
        elif problem and "step" in _tokens(problem.replace("_", " ")):
            self.step_cap = min(self.step_cap, self.steps + 1)
        if job in ("speak", "add", "buy", "none"):
            self.last_looks.clear()
        else:
            self.last_looks[:] = self.last_looks[-1:] + [name + json.dumps(args, sort_keys=True)]
        before = {p: (r.get("price"), r.get("in_stock")) for p, r in self.index.rows.items()}
        had_sheet = bool(self.index.sheet_of.get(str(next((v for k, v in args.items() if k == self.text_arg(name)), ""))))
        self.index.take(inner, from_search=job == "find")
        self._learn_fields(inner)
        if job not in ("add", "cart", "buy", "none", "speak"):
            self._collect_fields(inner)            # a cart's own counts are no property of a listing
        pid = next((v for k, v in args.items() if isinstance(v, str) and k == self.text_arg(name)), None)
        if job == "open" and pid and not problem and isinstance(inner, dict):
            if pid in _ids_in(inner, {pid}) or pid in self.index.ids:
                self.index.ids.add(pid)
                self.index.opened.add(pid)
                if pid not in self.index.seen_order:
                    self.index.seen_order.append(pid)
                self.read_by.setdefault(pid, {})[name] = self.ledger.version
                self.read_records[(pid, name)] = _dump(observation, 3000)
                if any(isinstance(v, dict) for v in inner.values()):
                    keys = set(inner)
                    self.page_keys[name] = self.page_keys[name] & keys if name in self.page_keys else keys
                    least = max(len(v) for v in inner.values() if isinstance(v, dict))
                    self.page_sheet[name] = min(self.page_sheet.get(name, least), least)
                if self.index.rows.get(pid, {}).get("in_stock") is False and pid in self.basket:
                    self.gone.add(pid)             # its own record says it cannot be bought now
                if self.index.sheet_of.get(pid):
                    if not had_sheet:
                        self.sheet_readers.add(name)
                elif name not in self.sheet_readers:
                    self.sheetless.add(name)
        elif job is None and pid and pid in self.index.ids and len(args) == 1 and not problem \
                and pid in _ids_in(inner, {pid}) and name not in self.readers:
            self.readers.append(name)              # a tool seen reading one listing by its id is a reader too
            self.index.opened.add(pid)
            self.read_by.setdefault(pid, {})[name] = self.ledger.version
        if job == "find":
            self._watch_paging(args, inner)
            slot = self.find_args.get("words") or self.text_arg(name)
            if not problem and slot and str(args.get(slot) or "").strip() and _ids_in(inner, self.index.ids):
                self.word_choices = self.enum_args(name, args)
            elif not problem and slot and not str(args.get(slot) or "").strip():
                self.wordless_choices = self.enum_args(name, args)
            if not problem:
                page = self.find_args.get("page")
                query = _search_key(args, page)
                self.searches.add(query)
                later = bool(page and args.get(page) not in (None, "", 0, "0", 1, "1"))
                if not later and not _ids_in(inner, self.index.ids):
                    self.empty_searches.add(query)   # a first page with no rows: later pages have none either
                if later:
                    self.paged = True
                served = {self.item_of(p) for p in _ids_in(inner, self.index.ids)
                          if not self.label_of(self.item_of(p)) or _alike(self.index.rows.get(p, {}).get("category"),
                                                                          self.label_of(self.item_of(p)))}
                for n in served:
                    self.item_queries.setdefault(n, set()).add(query)
                    if later:
                        self.item_paged.add(n)
                words = " ".join(str(v) for k, v in args.items() if isinstance(v, str) and k != page)[:60]
                self.page_items[words] = self.page_items.get(words, set()) | served
        if job == "find" and not problem:
            self._one_section()
        if job in ("add", "cart") and not problem:
            self._note_counts(name, args, inner, job)
        if job == "add":
            self._track_add(name, args, problem)
        if job == "cart":
            self.basket_seen = inner
            held = _ids_in(inner, self.index.ids)
            shows = _has_entries(inner)            # an empty cart shows no entries; an unreadable one does
            self.view_readable = bool(held) or not shows
            if held or (not problem and not shows):
                if not held:
                    self.held = {}
                if self.last_sent is not None and set(held) - self.last_sent:
                    self.keeps_unsent = True       # it still holds a line our last whole-cart call left out
                self.basket = held
                self.carted |= set(held)
            self.cart_viewed = self.cart_mark
        self._watch_facts(before)
        refused = job in ("buy", "none") and not self.done and not problem
        self.refused_last = refused and bool(lines or self._refusal_lines(observation))
        if refused:
            lines = lines + [x for x in self._refusal_lines(observation) if x not in lines]
            notes.append("the %s came back refused; the trip goes on" % ("order" if job == "buy" else "ending"))
        said = []
        if job == "speak" and not problem:
            # the reply's words, wherever the result keeps them (codes and single tokens are not words)
            said = [x for x in _all_texts(observation) if re.search(r"[a-zA-Z]{2}", x) and (" " in x or len(x) > 2)]
            lines = list(dict.fromkeys(said + lines))
        question = self.pending_question if job == "speak" else ""
        shown = self.pending_text if job == "speak" else ""
        who = self.pending_item if job == "speak" else -1
        fresh = [x for x in dict.fromkeys(lines) if (who, question, x) not in self.taken]
        self.taken.update((who, question, x) for x in fresh)
        if job == "speak":
            self.pending_question = self.pending_text = ""
        if fresh and self.remaining() > SECONDS * HAND_OVER + 15:
            if job != "speak":
                notes.append("this result carries a note: " + " | ".join(fresh[:4]))
            notes += self._hear(fresh, question, shown)
        elif question and not said and not problem:
            self._no_answer(question)              # the reply held no words at all
        return notes

    def _one_section(self) -> None:
        """Items that are the kinds of goods a store section's name lists, when the request writes every kind that
        name lists, were that section: they become one item, bought as one listing, before any question or cart line.
        Items under different sections, or with needs that differ on one property, stay apart."""
        led = self.ledger
        if not led.several() or self.asked or self.basket_touched:
            return
        def words(text: Any) -> list[str]:
            return [w.rstrip("s") for w in _tokens(text)]
        said = words(self.request)
        heads = {words(it["name"])[-1] for it in led.items if words(it["name"])}
        for label in self.store_labels():
            held = words(label)
            raw = _tokens(label)
            content = [w.rstrip("s") for w in raw if len(w) > 3]
            if len(content) < 3 or not any(w in heads for w in held):
                continue
            # the section's kinds in the label's order and, for a two-part name, the other way round
            joint = next((i for i, w in enumerate(raw) if w in ("and", "or", "&", "with")), None)
            orders = [content]
            if joint is not None:
                left = [w.rstrip("s") for w in raw[:joint] if len(w) > 3]
                right = [w.rstrip("s") for w in raw[joint + 1:] if len(w) > 3]
                if left and right:
                    orders.append(right + left)
            span = next((s for s in (self._span_in_order(o, said, len(held) + 3) for o in orders) if s), None)
            if span is None or any(_figures(w) or w in COUNT_WORDS for w in said[span[0]:span[1] + 1]):
                continue                           # the request does not write the whole section name as one phrase
                                                   # (a count inside the phrase numbers separate goods)
            inside = [it for it in led.items if words(it["name"])
                      and all(w in held for w in words(it["name"]) if len(w) > 3)
                      and any(w in held for w in words(it["name"]))]
            if len(inside) < 2:
                continue
            old = {_fold(it["name"]) for it in inside}
            values: dict[str, str] = {}
            if any(values.setdefault(_fold(n["property"]) + " " + str(n.get("op")), _value_key(n["value"]))
                   != _value_key(n["value"]) for n in led.needs if _fold(n.get("item")) in old):
                continue                           # the kinds want different things: a real list of goods
            # the kinds become one item (the section's), at the first one's place; the others' records fold into it
            name = " ".join(_tokens(label))
            first = min(led.items.index(it) for it in inside)
            caps = {float(it["cap"]) for it in inside if it.get("cap")}
            one = {"name": name, "qty": max(int(it.get("qty") or 1) for it in inside), "store_label": label,
                   "cap": caps.pop() if len(caps) == 1 and all(it.get("cap") for it in inside) else None}
            moved: dict[int, int] = {}
            fresh: list[dict[str, Any]] = []
            for i, it in enumerate(led.items):
                if it in inside:
                    moved[i] = first
                    if i == first:
                        fresh.append(one)
                else:
                    moved[i] = len(fresh)
                    fresh.append(it)
            led.items = fresh
            kept: dict[Any, dict[str, Any]] = {}
            for n in led.needs:
                if _fold(n.get("item")) in old:
                    n["item"] = name
                kept.setdefault(_need_key(n) + (_value_key(n["value"]),), n)
            led.needs = list(kept.values())
            for r in led.refs:
                if _fold(r.get("item")) in old:
                    r["item"] = name
            led.version += 1
            # the per-item records follow the items to their new places
            self.queue, self.plan_key, self.plan_failed, self.ranked_by_item, self.reask = [], None, False, {}, None
            self.searched_for = {moved[i] for i in self.searched_for if i in moved}
            merged: dict[int, set[str]] = {}
            for i, q in self.item_queries.items():
                merged.setdefault(moved.get(i, i), set()).update(q)
            self.item_queries = merged
            self.item_paged = {moved.get(i, i) for i in self.item_paged}
            self.page_items = {k: {moved.get(i, i) for i in v} for k, v in self.page_items.items()}
            self.free = {(moved.get(i, i), p) for i, p in self.free}
            self.pending_item = moved.get(self.pending_item, 0)
            self.announced = {moved.get(i, i) for i in self.announced}
            self.probed = {(moved.get(i, i), *rest) for i, *rest in self.probed}
            self.audit_fresh = {(moved.get(i, i), *rest) for i, *rest in self.audit_fresh}
            self.chat.append({"role": "user", "content": "NOTE: the request named one store section, %s: its kinds are one "
                              "item, bought as one listing from that section.\n%s" % (label, self.status())})
            return

    @staticmethod
    def _span_in_order(kinds: list[str], said: list[str], window: int) -> tuple[int, int] | None:
        """The first stretch of the text (start, end) that carries the words in this order, within the window."""
        for start, w in enumerate(said):
            if w != kinds[0]:
                continue
            at, ok = start, True
            for k in kinds[1:]:
                nxt = next((i for i in range(at + 1, min(len(said), start + window)) if said[i] == k), None)
                if nxt is None:
                    ok = False
                    break
                at = nxt
            if ok:
                return start, at
        return None

    def _refusal_lines(self, observation: Any) -> list[str]:
        out: list[str] = []
        if isinstance(observation, dict):
            for v in observation.values():
                out += _sentences(v) if not isinstance(v, dict) else [x for w in v.values() for x in _sentences(w)]
        return out

    def _track_add(self, name: str, args: dict[str, Any], problem: str) -> None:
        props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
        if problem:
            key = name + json.dumps(args, sort_keys=True)
            self.failed_calls[key] = self.failed_calls.get(key, 0) + 1
            if self._sets_cart(name) and '": 0' in json.dumps(args):
                self.zero_refused = True           # this store will not take a line at 0: lists are resent without them
            elif not self._is_removal(name, args):
                for pid in [p for p in self._add_ids(name, args) if p not in self.basket]:
                    self.add_faults[pid] = self.add_faults.get(pid, 0) + 1
                    if self.add_faults[pid] >= 2:
                        self.gone.add(pid)         # the store will not take it: treated as unavailable
        params = self.schemas[name].get("parameters") or {}
        named = self._add_ids(name, args)
        if not self.add_mark and not problem:
            # the mode of a first add, learned from its required choices only (an optional choice is no mode)
            self.add_mark = {k: v for k, v in self.enum_args(name, args).items() if k in (params.get("required") or [])} or None
        removal = self._is_removal(name, args)
        if not problem and not removal:
            self.basket_touched = True
        q_slot = self._qty_slot(props)
        if self._sets_cart(name):
            if not problem:
                arr, id_slot, line_q = self._line_slots(name)
                self.last_sent = {str(_d(x).get(id_slot)) for x in args.get(arr) or []
                                  if not (line_q and _d(x).get(line_q) == 0)}
                self.basket = []
                for line in args.get(arr) or []:
                    pid = str(_d(line).get(id_slot) or "")
                    qty = _d(line).get(line_q, 1) if line_q else 1
                    if pid and isinstance(qty, int) and qty > 0 and pid not in self.basket:
                        self.basket.append(pid)
                        self.line_qty[pid] = qty
                        self.carted.add(pid)
                    elif pid and qty == 0:
                        self.carted.discard(pid)
        else:
            for pid in named:
                if pid in self.basket and not problem:
                    self.basket.remove(pid)
                if not removal and not problem:
                    self.basket.append(pid)
                    self.line_qty[pid] = int(args.get(q_slot) or 1) if q_slot else 1
                    self.carted.add(pid)
                elif removal and not problem and self.held.get(pid, 0) > 0:
                    self.basket.append(pid)        # the store took fewer units than it held: the line stays
                    self.line_qty[pid] = self.held[pid]
                elif removal and not problem:
                    self.carted.discard(pid)
        self.cart_mark += 1

    def _watch_facts(self, before: dict[str, tuple[Any, Any]]) -> None:
        """A changed price or stock on a listing in play (in the cart or passed) is a change to the ledger."""
        in_play = set(self.basket) | {p for (p, v), r in self.checks.items() if v == self.ledger.version and r.get("ok")}
        for pid in in_play:
            row = self.index.rows.get(pid, {})
            now = (row.get("price"), row.get("in_stock"))
            then = before.get(pid)
            if then is not None and then != now and then != (None, None):
                if isinstance(then[0], (int, float)) and isinstance(now[0], (int, float)) and now[0] > then[0]:
                    self.price_rose[pid] = now[0]
                self.ledger.version += 1
                self.event_version = self.ledger.version
                self.chat.append({"role": "user", "content": "CHANGE: %s now shows price %s, in stock %s (was %s, %s)" % (
                    pid, now[0], now[1], then[0], then[1])})
                return

    def _learn_fields(self, observation: Any) -> None:
        """The first result that lists several entries tells which keys hold the id, price, stock, category and title."""
        if self.index.roles or self.roles_tries >= 2 or self.easing(30.0):
            return
        inner = _notes_beside(observation)[0]
        entries = _first_rows(inner)
        if entries is None:
            return
        self.roles_tries += 1
        sample = entries[:2]
        got = self._read(TERSE, FIELD_ROLES, _trim(sample, 200), 400, deadline=time.monotonic() + ROLES_DEADLINE)
        roles = {k: v for k, v in got.items() if isinstance(v, str) and v} or _roles_from_rows(entries)
        if roles.get("id") or roles.get("price"):
            self.index.learn_roles(roles)

    def _watch_paging(self, args: dict[str, Any], inner: Any) -> None:
        inner = _d(inner)
        total = inner.get(self.index.roles.get("match_count", ""))
        after = inner.get(self.index.roles.get("next_page", ""))
        words = " ".join(str(v) for k, v in args.items() if isinstance(v, str) and k != self.find_args.get("page"))[:60]
        self.paging.pop(words, None)
        self.page_next.pop(words, None)
        if after not in (None, False, ""):
            self.page_tokens.setdefault(self._page_words(args), set()).add(str(after))
            self.page_next[words] = (dict(args), after)
            self.paging[words] = "%s (%s matches; next page: %s=%s)" % (
                words, total if isinstance(total, int) else "more", self.find_args.get("page") or
                self.index.roles.get("next_page"), after)

    def _own_page(self, args: dict[str, Any]) -> dict[str, Any]:
        """A later-page value belongs to the search words that gave it: sent with words no search has paged yet, it
        would skip their first pages, so those words start at their first page."""
        page = self.find_args.get("page")
        if not page or args.get(page) in (None, "", 0, "0", 1, "1"):
            return args
        if self._page_words(args) in self.page_tokens:
            return args
        if not any(str(args.get(page)) in t for t in self.page_tokens.values()):
            return args                            # a value no search gave: left to the store
        find = self.names.get("find") or ""
        params = _d(self.schemas.get(find, {}).get("parameters"))
        if page in (params.get("required") or []):
            slot = _d(_d(params.get("properties")).get(page))
            if "default" in slot or "minimum" in slot:
                return dict(args, **{page: slot.get("default", slot.get("minimum"))})
            return args                            # no first value the schema names: the call goes as written
        return {k: v for k, v in args.items() if k != page}

    def _page_words(self, args: dict[str, Any]) -> str:
        page = self.find_args.get("page")
        return _fold(" ".join(str(v) for k, v in sorted(args.items()) if isinstance(v, str) and k != page))

    # -------------------------------------------------------------- changes
    def _hear(self, lines: list[str], question: str, shown: str = "") -> list[str]:
        """A shopper line or a notice, read against the ledger by the light model (an empty reading is made once more);
        any change raises the version. A reply to a question also moves that question's list on."""
        asked_as = shown or question
        if asked_as:
            self.talk_log.append("asked: " + asked_as)
        material = {"request": self.request, "ledger": self.ledger.as_material(), "recent_talk": self.talk_log[-10:],
                    "question": asked_as or None, "said": lines}
        got = self._read(LIGHT, SHOPPER_LINES, material, 900, first_cap=25.0)
        if not got and self.remaining() > SECONDS * HAND_OVER + READ_AGAIN_FLOOR:
            got = self._read(LIGHT, SHOPPER_LINES, material, 900, first_cap=20.0)
        self.talk_log.append("said: " + " | ".join(lines))
        led = self.ledger
        before = led.signature()
        changes: list[str] = []
        if got:
            heard = {w.rstrip("s") for w in _tokens(" ".join(lines))}
            # a line withdraws only a property it names itself; a muddled reply withdraws nothing
            withdrawn = {_fold(x) for x in _texts(got.get("withdrawn"))
                         if {w.rstrip("s") for w in _tokens(x) if len(w) > 2} & heard}
            new = [n for n in (_need_of(x, led.items, "shopper", led.version + 1)
                               for x in (got.get("needs") if isinstance(got.get("needs"), list) else [])) if n]
            for n in new:                           # a need the ledger already holds keeps its own comparison
                held = next((m for m in led.needs if _fold(m["item"]) == _fold(n["item"]) and _fold(m["property"]) == _fold(n["property"])
                             and str(m["value"]) == str(n["value"])), None)
                if held:
                    n["op"] = held["op"]
            new = [n if _fold(n["property"]) == _fold(question)
                   or any(_fold(m["property"]) == _fold(n["property"]) and _alike(m["item"], n["item"]) for m in led.needs)
                   else dict(n, property=self._key_as_said(n["property"], lines)) for n in new]
            # a line repeating a need that holds only in some case does not turn it into a plain need
            cased = {(_fold(n["item"]), _fold(n["property"]), n["op"], _value_key(n["value"])) for n in led.needs if n.get("when")}
            new = [n for n in new if n.get("when") or (_fold(n["item"]), _fold(n["property"]), n["op"], _value_key(n["value"])) not in cased]
            had = {(_need_key(n), _value_key(n["value"])) for n in led.needs}
            new = [n if (_need_key(n), _value_key(n["value"])) in had else self._value_as_shown(n)
                   for n in self._said_only(new, lines, led.needs)]   # only a need this line adds is trimmed
            marks = {" ".join(_tokens(m)) for r in (got.get("references") or []) if isinstance(r, dict)
                     for m in _texts(r.get("marks"))} - {""}
            # the values that pick out a pointed listing describe that listing, not the item wanted (an answer to
            # the property just asked always stands)
            new = [n for n in new if _fold(n["property"]) == _fold(question)
                   or not ({" ".join(_tokens(_dump(n["value"]))),
                            " ".join(_tokens("%s %s" % (n["property"], _dump(n["value"]))))} & marks)]
            keys = {_need_key(n) for n in new}
            def dropped(n: dict[str, Any]) -> bool:
                # the exact name always goes; a sibling key goes only for a need the shopper gave, and only when no
                # need carries the withdrawn name itself (a stated "pack size" survives a waived "size")
                words = set(_fold(n["property"]).split())
                return any(w == _fold(n["property"]) or (n.get("source") != "stated"
                                                         and not any(_fold(m["property"]) == w for m in led.needs)
                                                         and set(w.split()) <= words) for w in withdrawn if w)
            kept = [n for n in led.needs if _need_key(n) not in keys and not dropped(n)]
            fresh = [n for n in new if (_need_key(n), _value_key(n["value"])) not in had]
            if isinstance(got.get("needs"), list) or withdrawn:
                # an unchanged need keeps its own entry (and wording); only real changes replace one
                same = {(_need_key(n), _value_key(n["value"])): n for n in led.needs}
                led.needs = kept + [same.get((_need_key(n), _value_key(n["value"])), n) for n in new]
            reply = str(got.get("reply") or "").lower()
            many = len(self.talk_now) > 1
            turned = bool(question) and many and reply == "one_at_a_time" and not fresh
            for prop in _texts(got.get("no_preference")):
                # a list frees only the property named first, or one the shopper waives by its own name
                if many and not set(_fold(question).split()) <= set(_fold(prop).split()) \
                        and not (any(_fold(prop) == _fold(x) for x in self.talk_now)
                                 and {w.rstrip("s") for w in _tokens(prop)} <= heard):
                    continue
                if turned:
                    continue
                if _fold(prop) not in {_fold(x) for x in led.no_preference}:
                    led.no_preference.append(prop)
            obj = _d(got.get("objective"))
            kind = str(obj.get("kind") or "").lower()
            if kind in ("cheapest", "any") or (kind == "best" and obj.get("criterion")):
                updated = {"kind": kind, "criterion": str(obj.get("criterion") or ""),
                           "direction": "min" if str(obj.get("direction") or "").lower() == "min" else "max"}
                if led.objective_key(updated) != led.objective_key(led.objective):
                    led.objective = updated
            try:
                written = _figures(" ".join(lines))
                if got.get("money_cap") not in (None, "") and float(got["money_cap"]) > 0:
                    cap = _as_text_says(float(got["money_cap"]), " ".join(lines))
                    if cap != led.budgets.get("money_cap") and any(
                            abs(w - cap) < 1e-6 or w > 0 and abs(cap / w - round(cap / w)) < 1e-9 and 2 <= round(cap / w) <= 10
                            for w in written):
                        led.budgets["money_cap"] = cap
                        changes.append("money limit now %s" % cap)
            except (TypeError, ValueError):
                pass
            for raw in got.get("steps") if isinstance(got.get("steps"), list) else []:
                step = _step_of(raw)
                if step and step["detail"] and all(_fold(step["detail"]) != _fold(s["detail"]) for s in led.steps):
                    led.steps.append(step)
                    led.rules.append(step["detail"])
            for event in got.get("events") if isinstance(got.get("events"), list) else []:
                event = _d(event)
                named = str(event.get("listing") or "").strip()
                named = named if named in self.index.ids else ""
                if event.get("kind") in ("price", "gone"):
                    self.read_records.clear()          # a stored reading may show the old price or stock now
                if event.get("kind") == "price":
                    self.reread = list(dict.fromkeys(self.reread + ([named] if named else [p for p, _ in self.cart_lines()])))
                    changes.append("a price changed; %s read again" % (named or "the cart lines are"))
                elif event.get("kind") == "gone":
                    # the shopper's word is no store fact: the line the words name (else every cart line) is read
                    # again, and only its own record, or the store's refusal to take it, marks it gone
                    target = named or self._named_in(lines)
                    self.reread = list(dict.fromkeys(self.reread + ([target] if target else [p for p, _ in self.cart_lines()])))
                    if target:
                        changes.append("the shopper speaks of %s as gone; its record is read again" % target)
            ties_before = len(led.refs)
            self._take_refs(got.get("references"))
            tied = led.refs[ties_before:]
            for n in fresh:
                i = self._item_number(n["item"])
                if i is not None:
                    self.heard_count[i] = self.heard_count.get(i, 0) + 1
            for r in tied:
                i = self._item_number(r["item"])
                if i is not None:
                    self.heard_count[i] = self.heard_count.get(i, 0) + 1
            if question:
                self.reply_read = True
                said = [n["property"] for n in fresh] + [r["property"] for r in tied]
                listed = self.talk_now or [question]
                way = "turned" if turned and not said else "confused" if reply == "confused" else "no_need" if not said else "answered"
                self._after_reply(self.pending_item, listed, way, said)
                if not said and way != "turned" and way != "confused":
                    self._no_answer(question)
        if led.signature() != before:
            led.version += 1
            changes.insert(0, "the ledger changed")
        if changes and not question:
            self.event_version = led.version
        return ["%s (now: %s)" % ("; ".join(changes), led.summary())] if changes else []

    def _named_in(self, lines: list[str]) -> str:
        """The listing a note's words name: its id, else the one cart line whose whole title the words carry."""
        text = " %s " % _fold(" ".join(lines))
        hit = next((p for p in self.index.ids if len(p) >= 2 and " %s " % _fold(p) in text), "")
        if hit:
            return hit
        titled = [p for p, _ in self.cart_lines() if _fold(self.index.rows.get(p, {}).get("title"))
                  and " %s " % _fold(self.index.rows[p]["title"]) in text]
        return titled[0] if len(titled) == 1 else ""

    def _take_refs(self, raw: Any) -> None:
        """New ties to other listings: one per (item, property); a need the same words set as a literal value goes."""
        led = self.ledger
        for x in raw if isinstance(raw, list) else []:
            x = _d(x)
            prop, seller = str(x.get("property") or "").strip(), _owner(x.get("seller"))
            described = str(x.get("described") or "").strip()
            if not prop or not (seller or described or _texts(x.get("marks"))):
                continue
            item = next((it["name"] for it in led.items if _alike(it["name"], x.get("item"))
                         or _fold(x.get("item")) in _fold(it["name"])), led.items[0]["name"] if led.items else "item")
            if any(_fold(r["property"]) == _fold(prop) and _fold(r["item"]) == _fold(item) for r in led.refs):
                continue
            words = set(_tokens(" ".join(_texts(x.get("marks")))))
            split = [r for r in led.refs if seller and _fold(r["seller"]) == _fold(seller) and _fold(r["property"]) != _fold(prop)
                     and len(r["marks"]) == 1 and set(_tokens(r["marks"][0])) < words]
            for r in split:                        # earlier ties were this pointer's own picking values, split off
                led.refs.remove(r)
                led.needs = [n for n in led.needs if not (n.get("source") == "reference"
                                                          and _fold(n["property"]) == _fold(r["property"]))]
            relation = str(x.get("relation") or "same").lower()
            led.refs.append({"item": item, "property": prop, "relation": relation if relation in ("same", "compatible", "not")
                             else "same", "seller": seller, "category": str(x.get("category") or ""),
                             "marks": _texts(x.get("marks"))[:4], "described": described, "status": "open", "found": []})
            # the pointer's own words are no value: a need that copied them is dropped
            led.needs = [n for n in led.needs if not (_fold(n["property"]) == _fold(prop) and seller
                                                      and _fold(seller) in _fold(_dump(n["value"])))]

    def follow(self, ref: dict[str, Any]) -> None:
        """Follow a property tied to another listing until a listing states its value: find the listing described, read
        it, take the value or the next listing it points to. What was read along the way stays on the tie, so a later
        call goes on from there. A value becomes a plain need (source "reference")."""
        trail = ref.get("trail")
        if trail is None:
            trail = ref["trail"] = {"target": {"seller": ref["seller"], "category": ref["category"], "marks": ref["marks"],
                                               "described": ref.get("described", "")},
                                    "alias": "", "hops": 0, "misses": 0, "seen": set(), "tried": set(),
                                    "relation": ref["relation"]}
        self.chain_seen, self.chain_tried = trail["seen"], trail["tried"]
        ref["status"] = "open"
        while self._tie_room() and ref["status"] == "open":
            if trail["hops"] >= TIE_HOPS or trail["misses"] >= TIE_MISSES:
                self._tie_given_up(ref)
                return
            self._follow_one(ref, trail)

    def _pointer(self, nxt: dict[str, Any], described: Any = None) -> dict[str, Any]:
        """A description of the next listing: its seller, its category (a picking value that names a shelf this store
        shows becomes the category when none was given), and the remaining picking values."""
        marks, category = _texts(nxt.get("marks"))[:5], str(nxt.get("category") or "")
        if not category:
            shelves = {_fold(c) for c in self.index.category_counts} | {_fold(c) for c in self.store_labels()}
            hit = next((m for m in marks if _fold(m) in shelves or " ".join(_tokens(m)) in
                        {" ".join(_tokens(x)) for x in shelves}), None)
            if hit:
                category, marks = hit, [m for m in marks if m != hit]
        return {"seller": _owner(nxt.get("seller")), "category": category, "marks": marks[:4], "described": str(described or "")}

    def _grounded(self, pid: str, value: str, names: list[str], records: list[Any]) -> bool:
        """Whether a listing states `value` for a property: under a field whose key shares a word with one of the
        property's names, or in one sentence of its records that names the property."""
        want, tight = set(_tokens(value)), "".join(_tokens(value))
        keys = [set(w.rstrip("s") for w in _tokens(n)) for n in names if n]
        def states(text: Any) -> bool:
            return want <= set(_tokens(text)) or bool(tight) and tight in "".join(_tokens(text))
        for k, v in self.fields_of(pid).items():
            if states(v) and any(set(w.rstrip("s") for w in _tokens(k)) & ks for ks in keys):
                return True
        for sentence in re.split(r"(?<=[.;!?])\s+|\n", _dump(records, 15000)):
            if states(sentence) and any(ks <= {w.rstrip("s") for w in _tokens(sentence)} for ks in keys):
                return True
        return False

    @staticmethod
    def _seller_in(sentence: str, pointer: dict[str, Any]) -> str:
        """A seller's name in a pointing sentence when the reading left it out: a capitalised word past the first,
        not one of the category's or the picking values' words."""
        taken = set(_tokens(pointer.get("category") or ""))
        words = re.findall(r"[A-Za-z][\w'-]*", sentence)
        for w in words[1:]:
            if w[0].isupper() and w[1:].islower() and len(w) > 2 and _fold(w).rstrip("s").rstrip("'") not in taken:
                return _owner(w)
        return ""

    def _find_pointed(self, target: dict[str, Any]) -> str | None:
        """The listing a description points at: the seller's listings in the category named (searched by the seller's
        name, on that shelf when the search takes one, then with the picking values, then alone) are read, a few at a
        time, until one whose own record carries most of the picking values is open; that one, else the closest."""
        seller = _tokens(_owner(target.get("seller")))
        if not seller:
            return None
        marks = [m for m in _texts(target.get("marks"))]
        wanted = set(_tokens(" ".join(marks)))
        category = str(target.get("category") or "")
        def stem(text: Any) -> set[str]:
            return {w.rstrip("s") for w in _tokens(text) if len(w) > 2}
        def sold_by(p: str) -> bool:
            # the seller is one of the listing's own short fields, never a name its notes mention
            return any(_tokens(_owner(v)) == seller for v in self.fields.get(p, {}).values() if len(str(v)) <= 60)
        def in_category(p: str) -> bool:
            shown = self.index.rows.get(p, {}).get("category")
            return not category or not shown or _alike(shown, category) or bool(stem(shown) & stem(category))
        def overlap(p: str) -> int:
            return len(wanted & set(_tokens(self.index.text_of.get(p, ""))))
        def marks_met(p: str) -> int:
            # a picking value counts only when one field of the listing (its key with its value), or its title,
            # carries all of that value's words: a listing sharing loose words across fields does not qualify
            cells = [set(_tokens("%s %s" % (k, v))) for k, v in self.fields_of(p).items()]
            cells.append(set(_tokens(self.index.rows.get(p, {}).get("title") or "")))
            return sum(1 for m in marks if _tokens(m) and any(set(_tokens(m)) <= c for c in cells))
        def pool() -> list[str]:
            rows = [p for p in self.index.ids if sold_by(p) and p not in self.chain_seen]
            return sorted([p for p in rows if in_category(p)] or rows,
                          key=lambda p: (-marks_met(p), -overlap(p), p not in self.index.opened))
        enough = len([m for m in marks if _tokens(m)])
        tried = self.chain_tried
        shelf = self._shelf_args(category)
        searches = [(target["seller"], shelf), (" ".join([target["seller"]] + marks[:2]), {}), (target["seller"], {})]
        for _ in range(5):
            if self.steps_left() <= self.reserve() + 3 or self.easing(30.0):
                break
            rows = pool()
            hit = next((p for p in rows if p in self.index.opened and marks_met(p) >= enough), None)
            if hit:
                return hit
            unread = [p for p in rows if p not in self.index.opened and p not in tried][:4]
            if unread:
                tried.update(unread)
                self._open_all(unread, "READ FOR A TIED NEED", every=False)
                continue
            args = None
            while searches:
                words, extra = searches.pop(0)
                args = self._search_args(words.strip())
                if args is not None and extra:
                    args.update(extra)
                page = self.find_args.get("page")
                if args is not None and _search_key(args, page) \
                        not in self.searches:
                    break                          # a search this trip already made adds nothing
                args = None
            else:
                break
            if args is None:
                break
            self._run(self.names["find"], args, "SEARCHED FOR A TIED NEED")
        rows = pool()
        opened = [p for p in rows if p in self.index.opened]
        return max(opened, key=marks_met) if opened else (rows[0] if rows else None)

    def _shelf_args(self, category: str) -> dict[str, Any]:
        """The search's own choice of shelf for a category, when one of its choice lists names it."""
        find = self.names.get("find")
        props = ((self.schemas.get(find) or {}).get("parameters") or {}).get("properties") or {} if find else {}
        for key, spec in props.items():
            for value in (spec or {}).get("enum") or []:
                if category and isinstance(value, str) and _fold(value) == _fold(category):
                    return {key: value}
        return {}

    def _word_mode(self, find: str, last: dict[str, Any]) -> dict[str, Any]:
        """A search's arguments with its choices set the way words are searched: those of the last call whose words
        found rows; before any, a required choice used by a call without words is switched to another of its values."""
        params = (self.schemas.get(find) or {}).get("parameters") or {}
        props, required = params.get("properties") or {}, set(params.get("required") or [])
        out = dict(last)
        for k, v in self.wordless_choices.items():
            if k not in required or out.get(k, v) != v:
                continue                           # only a required choice set the way word-less calls set it
            if k in self.word_choices:
                out[k] = self.word_choices[k]
                continue
            others = [x for x in (props.get(k) or {}).get("enum") or [] if x != v]
            if len(others) == 1:
                out[k] = others[0]
        return out

    def _search_args(self, words: str) -> dict[str, Any] | None:
        """Search arguments for some words, built on the last search's own arguments (its page left out)."""
        find = self.names.get("find")
        if not find:
            return None
        params = self.schemas[find].get("parameters") or {}
        props = params.get("properties") or {}
        page = self.find_args.get("page")
        slot = self.find_args.get("words") if self.find_args.get("words") in props else self.text_arg(find)
        if not slot or not words:
            return None
        last = next((json.loads(m["tool_call"]["function"]["arguments"]) for m in reversed(self.out)
                     if m.get("tool_call") and m["tool_call"]["function"]["name"] == find), None) or {}
        required = params.get("required") or []
        # only the search's required choices carry over: a category filter would hide a listing filed elsewhere;
        # they come from the last search whose words found rows, since a listing mode may leave words unread
        last = self._word_mode(find, last)
        args = {k: v for k, v in last.items() if k in required and "enum" in (props.get(k) or {})}
        args[slot] = words
        for k in params.get("required") or []:
            if k not in args and "enum" in (props.get(k) or {}):
                args[k] = props[k]["enum"][0]
        issues: list[str] = []
        args = _fit(params, args, find, issues)
        return None if issues else args

    def _key_as_said(self, prop: str, lines: list[str]) -> str:
        """A need's property as the store spells it: a name no record carries becomes the longest record field name
        the shopper's words spell out in full that holds all of its words (a shortened name reads one field only)."""
        keys = {k for p in set(self.fields) | set(self.index.sheet_of) for k in self.fields_of(p)}
        if not keys or any(_fold(k) == _fold(prop) for k in keys):
            return prop

        def stems(words: list[str]) -> set[str]:
            return {w[:-1] if w.endswith("s") and not w.endswith("ss") else w for w in words}
        text = " %s " % " ".join(_key_words(" ".join(lines)))
        want = stems(_key_words(prop))
        hits = [k for k in keys if _key_words(k) and " %s " % " ".join(_key_words(k)) in text
                and want and want <= stems(_key_words(k))]
        top = [k for k in hits if len(_key_words(k)) == max((len(_key_words(h)) for h in hits), default=0)]
        return top[0] if len({" ".join(_key_words(k)) for k in top}) == 1 else prop

    def _said_only(self, new: list[dict[str, Any]], lines: list[str], old: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The needs a shopper line adds, held to what the store shows: a word the shopper calls us by can ride along
        with a value. A fresh "is" value in text set off by commas loses a piece no opened listing shows for that
        property while another piece is shown; of several fresh "is" needs one line gives for one property, those
        whose value no listing shows go while another is shown. Lists (alternatives) and other ops stay whole."""
        had = {(_need_key(n), _value_key(n["value"])) for n in old}
        fresh = [n for n in new if n["op"] == "is" and isinstance(n["value"], str)
                 and (_need_key(n), _value_key(n["value"])) not in had]
        def seen(prop: str, v: str) -> bool:
            words = set(_fold(v).split())
            vocab = {x for p in self.fields for x in _pieces(_field(self.fields_of(p), prop) or "")}
            return bool(words) and any(words <= set(x.split()) or set(x.split()) <= words for x in vocab)
        def lettered(v: str) -> bool:
            return bool(re.search(r"[a-z]", v.lower()))
        drop: set[int] = set()
        for n in fresh:
            twins = [m for m in fresh if _need_key(m) == _need_key(n)]
            if len(twins) > 1 and all(lettered(m["value"]) for m in twins) and not seen(n["property"], n["value"]) \
                    and any(seen(m["property"], m["value"]) for m in twins):
                drop.add(id(n))
        out: list[dict[str, Any]] = []
        for n in new:
            if id(n) in drop:
                continue
            if any(n is m for m in fresh):
                # a comma before exactly three digits groups a figure; it does not part two values
                parts = [v.strip() for v in re.split(r",(?!\d{3}(?!\d))", n["value"]) if _fold(v)]
                if len(parts) > 1 and all(lettered(v) for v in parts):
                    shown = [v for v in parts if seen(n["property"], v)]
                    if shown and len(shown) < len(parts):
                        n = dict(n, value=", ".join(shown))
            out.append(n)
        return out

    def _value_as_shown(self, need: dict[str, Any]) -> dict[str, Any]:
        """A stated value trimmed to the value the store's sheets show for that property when the shopper's words wrap
        it in the item's or the property's own words: the longest shown value inside the stated one. A stated value
        the sheets show as it is, or wrapped in any other words, stands."""
        if need.get("op") != "is" or not isinstance(need.get("value"), str):
            return need
        said = _fold(need["value"])
        shown = {x for p in self.fields for x in _pieces(_field(self.fields_of(p), need["property"]) or "")}
        if not said or said in shown:
            return need
        wrap = {w.rstrip("s") for w in _tokens(need.get("item", "")) + _tokens(need.get("property", ""))}
        def wrapping(v: str) -> bool:
            extra = [w for w in said.split() if w not in v.split()]
            return bool(extra) and all(w.rstrip("s") in wrap for w in extra)
        inside = sorted(v for v in shown if v and re.search(r"[a-z]", v) and " %s " % v in " %s " % said and wrapping(v))
        return dict(need, value=max(inside, key=len)) if inside else need

    def _no_answer(self, question: str) -> None:
        """A reply that does not answer, or leaves the property to us, records no preference for it, for the item
        the question was about (in a basket the same property may still matter for another item)."""
        idx = self.pending_item
        item = self.ledger.items[idx]["name"] if idx < len(self.ledger.items) else ""
        if not question or question == "budget" or (idx, _fold(question)) in self.free \
                or any(_fold(n["property"]) == _fold(question) for n in self.ledger.needs_for(item)):
            return
        self.free.add((idx, _fold(question)))
        self.ledger.no_preference.append("%s: %s" % (item, question) if self.ledger.several() else question)

    def free_for(self, idx: int) -> set[str]:
        """Properties the shopper does not mind about for one item: its own, and those said of the whole order."""
        item = _fold(self.ledger.items[idx]["name"]) if idx < len(self.ledger.items) else ""
        out = {p for i, p in self.free if i == idx}
        for x in self.ledger.no_preference:
            head, _, tail = str(x).rpartition(":")
            if not head:
                out.add(_fold(x))
            elif _fold(head) == item:
                out.add(_fold(tail))
        return out

    def _kit_fit(self) -> bool:
        """For an item the request ties to another item of the list, once that other item has a line in the cart and
        no opened listing of this item that fits names one of the line's model codes: read rows already seen that
        name one; else search the item's label with each of the line's codes (up to KIT_SEARCHES) and read the rows
        that name it; when that finds none, put in the cart's place for the other item an opened listing of it that
        fits and whose codes an opened listing of this item names. True when it spent a step."""
        if self.steps_left() <= self.reserve() + 6 or self.easing(25.0) or not self.names.get("find"):
            return False
        for idx, it in enumerate(self.ledger.items):
            n = it.get("with")
            if not isinstance(n, int) or n == idx or n >= len(self.ledger.items):
                continue
            line = next((p for p, i in self.cart_lines() if i == n), None)
            theirs = self._codes_of(line) if line else set()
            if not theirs:
                continue
            def meets(q: str, codes: set[tuple[str, str]] = theirs) -> bool:
                return q not in self.gone and self._codes_meet(self._codes_of(q), codes)
            if any(meets(q) and self._case_check(q, False)[0]["ok"] for q in self.opened(idx)):
                continue                           # an opened listing that fits already names the line's model
            key = (idx, line)
            seen = [p for p in self.by_mention(self.unread(idx), idx) if meets(p) and p not in self.hunt_read][:GROUP_READ]
            if seen:
                self.hunt_read.update(seen)
                self._open_all(seen, "READ: THESE NAME THE MODEL OF THE CART'S %s" % self.ledger.items[n]["name"].upper(),
                               every=False)
                return True
            codes = sorted((c for c in theirs if not c[0].startswith("!")), key=lambda c: (-len(c[1]), c))
            done = self.kit_searches.get(key, 0)
            if done < min(KIT_SEARCHES, len(codes)):
                self.kit_searches[key] = done + 1
                word, digits = codes[done]
                args = self._search_args("%s %s %s" % (word, digits, self.label_of(idx) or it["name"]))
                if args is None or _search_key(args, self.find_args.get("page")) in self.searches:
                    return self._kit_fit()
                observation, sent = self._run(self.names["find"], args,
                                              "SEARCHED FOR A %s THAT NAMES THE CART'S %s MODEL" % (
                                                  it["name"].upper(), self.ledger.items[n]["name"].upper()))
                shown = set(self._ids_in(observation))
                rows = [p for p in self.by_mention(self.unread(idx), idx) if p in shown and meets(p)][:GROUP_READ]
                self.hunt_read.update(rows)
                self._open_all(rows, "READ: THESE NAME THE MODEL OF THE CART'S %s" % self.ledger.items[n]["name"].upper(),
                               every=False)
                return sent
            if key in self.kit_switched:
                continue
            self.kit_switched.add(key)
            # no listing of this item names the line's model: another opened listing of the other item that fits,
            # and whose model an opened listing of this item that fits names, takes the line's place
            held = next((p for p, i in self.cart_lines() if i == idx), None)
            mine = [held] if held and self._case_check(held)[0]["ok"] else \
                [q for q in self.opened(idx) if q not in self.gone and self._case_check(q, False)[0]["ok"]]
            alt = [m for m in self.opened(n) if m != line and m not in self.gone
                   and any(self._codes_meet(self._codes_of(q), self._codes_of(m)) for q in mine) and self.check(m)["ok"]]
            if alt and self._swap(line, self.prefer(alt)[0], n):
                self.chat.append({"role": "user", "content": "KIT: no %s names the model of %s, so code put %s in the cart for "
                                  "the %s instead; an opened %s names its model.\n%s" % (
                                      it["name"], line, self.prefer(alt)[0], self.ledger.items[n]["name"], it["name"], self.status())})
                return True
        return False

    def _ids_in(self, observation: Any) -> list[str]:
        """The listing ids of the rows a result shows, in its own order."""
        out: list[str] = []
        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, list):
                for v in node:
                    walk(v, depth + 1)
            elif isinstance(node, dict):
                key = self.index.id_key(node)
                if key and depth > 0:
                    out.append(str(node[key]))
                    return
                for v in node.values():
                    walk(v, depth + 1)
        walk(_notes_beside(observation)[0], 0)
        return list(dict.fromkeys(out))

    def _joint_search(self) -> bool:
        """Once the shopper adds a need to an item: one search that writes the item's wanted values together (the
        values just learned first, single words and figures with units leading), with no product noun when there
        are two or more values; when it shows no unread row of the item, the same without the last stated value.
        The first rows of the item it shows are read. True when it spent a step."""
        for idx, item in enumerate(self.ledger.items):
            needs = self.ledger.needs_for(item["name"])
            learned = [n for n in needs if n.get("source") in ("shopper", "reference")
                       and n["op"] in ("is", "one of", "has", "at least")]
            if not learned:
                continue
            key = (idx, tuple(sorted(" ".join(_need_key(n)) + " " + _value_key(n["value"]) for n in learned)))
            if key in self.joint_done:
                continue
            self.joint_done.add(key)
            if any(all(any(_need_key(c[0]) == _need_key(n) and c[1] == "met"
                           for c in self.check(p, full=False).get("cells") or []) for n in learned)
                   for p in self.opened(idx)):
                continue                           # an opened listing already states every learned value
            def words(n: dict[str, Any]) -> str:
                v = n["value"][0] if isinstance(n["value"], list) and n["value"] else n["value"]
                return str(v).strip() if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""
            told = [w for w in (words(n) for n in learned) if _tokens(w)]
            stated = [w for w in (words(n) for n in needs if n.get("source") not in ("shopper", "reference")
                                  and n["op"] in ("is", "one of", "has")) if _tokens(w)]
            lead = [v for v in told if len(v.split()) == 1 or re.fullmatch(r"\d[\d.,]*\s*[a-z\"']{1,12}", v, re.I)]
            values = list(dict.fromkeys(lead + told + stated))
            if not values:
                continue
            noun = self.label_of(idx) or item["name"]
            tries = [values] if len(values) > 1 else [[values[0], noun]]
            if len(values) > 1:
                tries.append(values[:-1] if stated else [values[0], noun])
            spent = False
            for chosen in tries:
                if self.steps_left() <= self.reserve() + 6 or self.easing(25.0):
                    return spent
                args = self._search_args(" ".join(chosen)[:80])
                if args is None or _search_key(args, self.find_args.get("page")) in self.searches:
                    continue
                observation, sent = self._run(self.names["find"], args, "SEARCHED WITH THE ITEM'S WANTED VALUES TOGETHER")
                spent = spent or sent
                if not sent:
                    return spent
                shown = set(self._ids_in(observation))
                rows = [p for p in self.by_mention(self.unread(idx), idx) if p in shown and p not in self.hunt_read][:3]
                if rows:
                    self.hunt_read.update(rows)
                    self._open_all(rows, "READ FOR THE ITEM'S WANTED VALUES TOGETHER", every=False)
                    return True
            if spent:
                return True
        return False

    def _hunt_revealed(self) -> bool:
        """After an answer or a followed tie adds a need: search until an opened listing settles it, at most
        HUNT_ROUNDS searches per need (the value with the store's label for the item, the value with the item's
        name, the value alone), each followed by reading the rows whose own words name the value. True when it
        spent a step."""
        if self.note_queue and self._take_cart_notes():
            return True                            # a sold-out note moved the ledger: the change is handled next
        if self._walk_due():
            self._low_walk()
            return True
        if self.steps_left() <= self.reserve() + 4 or self.easing(25.0) or not self.names.get("find"):
            return False
        if self._joint_search():
            return True
        for need in self.ledger.needs:
            if need.get("source") not in ("shopper", "reference") or need["op"] not in ("is", "one of", "has"):
                continue
            idx = next((n for n, it in enumerate(self.ledger.items) if _fold(it["name"]) == _fold(need["item"])), 0)
            if idx >= len(self.ledger.items):
                continue
            value = need["value"][0] if isinstance(need["value"], list) and need["value"] else need["value"]
            value = str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""
            if not _tokens(value):
                continue
            key = (idx, _fold(need["property"]) + " " + _fold(value))
            done = self.hunt_rounds.get(key, 0)
            if not value or done >= HUNT_ROUNDS:
                continue
            if any(any(_need_key(c[0]) == _need_key(need) and c[1] == "met" for c in self.check(p, full=False).get("cells") or [])
                   for p in self.opened(idx)):
                self.hunt_rounds[key] = HUNT_ROUNDS
                continue                           # an opened listing states the value: no hunt for this need
            names_value = [p for p in self.by_mention(self.unread(idx), idx) if p not in self.hunt_read
                           and set(_tokens(value)) <= set(_tokens(self.index.text_of.get(p, "")))]
            if names_value and self.hunt_batches.get(key, 0) < 2:   # rows found name the value: read them first
                self.hunt_batches[key] = self.hunt_batches.get(key, 0) + 1
                self.hunt_read.update(names_value[:GROUP_READ])
                self._open_all(names_value[:GROUP_READ], "READ FOR A NEED JUST LEARNED", every=False)
                return True
            item = self.ledger.items[idx]["name"]
            # the value with the store's own label for the item first: the rows that state a learned value sit under
            # that label, and the item's name as the shopper said it often misses them
            tries = ["%s %s" % (value, self.label_of(idx) or item), "%s %s" % (value, item), value]
            self.hunt_rounds[key] = done + 1
            args = self._search_args(tries[done].strip()[:60])
            page = self.find_args.get("page")
            if args is None or _search_key(args, page) in self.searches:
                return self._hunt_revealed()       # this round's words were searched already: the next round
            observation, sent = self._run(self.names["find"], args, "SEARCHED FOR A NEED JUST LEARNED")
            def naming() -> list[str]:
                return [p for p in self.by_mention(self.unread(idx), idx) if p not in self.hunt_read
                        and set(_tokens(value)) <= set(_tokens(self.index.text_of.get(p, "")))][:GROUP_READ]
            rows = naming()
            if sent and not rows and self.steps_left() > self.reserve() + 5 and self.pages_left > 1:
                # the rows that state a learned value often sit on a later page: one more page before the next words
                pieces = self._page_on(self.names["find"], args, observation)
                if pieces:
                    self.chat.append({"role": "user", "content": "\n".join(pieces)})
                    rows = naming()
            self.hunt_read.update(rows)
            self._open_all(rows, "READ FOR A NEED JUST LEARNED", every=False)
            return True
        return False

    def _after_change(self) -> None:
        """On a new version: check the cart lines again; when one no longer fits, name the listing to swap to; a dropped
        need opens the lookout for a cheaper fit again."""
        self.version_seen = self.ledger.version
        if len(self.ledger.needs) < self.needs_seen:
            self.fired.pop("objective", None)      # a dropped need may make a cheaper listing fit
        self.needs_seen = len(self.ledger.needs)
        lines = self.cart_lines()
        said = []
        for pid, idx in lines:
            r = self.check(pid)
            if r["ok"] and pid in self.price_rose:
                def price(p: str) -> float:
                    value = self.index.rows.get(p, {}).get("price")
                    return float(value) if isinstance(value, (int, float)) else 1e18
                below = [p for p in self.passed(idx, {q for q, _ in lines}) if price(p) < self.price_rose[pid]]
                if below:
                    swap = min(below, key=price)
                    said.append("the cart's %s went up to %s; %s fits at %s, so swap it in" % (
                        pid, self.price_rose[pid], swap, price(swap)))
                continue
            if r["ok"]:
                continue                           # a line that still fits after the change stays as it is
            # a line that no longer fits goes to a listing already read that fits, before any fresh look
            swap = self.backup(idx, {p for p, _ in lines}) or self.rechoose(idx, {p for p, _ in lines})
            said.append("the cart's %s no longer fits (%s); %s" % (
                pid, "; ".join(r["why"][:3]),
                "swap it for %s, which fits (price %s)" % (swap, self.index.rows.get(swap, {}).get("price")) if swap
                else "take it out and find a listing that fits"))
        stale = [p for p in self.basket if p in self.gone]
        if stale:
            said.append("the cart still holds %s, which can no longer be bought: it is taken out" % ", ".join(stale))
        if said:
            self.chat.append({"role": "user", "content": "AFTER THE CHANGE: " + " | ".join(said) + "\n" + self.status()})
        elif lines:
            self.chat.append({"role": "user", "content": "AFTER THE CHANGE: the cart still fits the ledger.\n" + self.status()})
        if self.event_version == self.ledger.version and not self.cart_fits():
            self.fired.pop("objective", None)      # the cheapest sweep runs again only for a cart the change broke
            self.unclear_hold = False
        if any(s["moment"] == "after_change" for s in self.ledger.steps):
            self.checklist("after_change")

    def _reread_one(self) -> None:
        pid = self.reread.pop(0)
        reader = self.names.get("open")
        slot = self.text_arg(reader) if reader else None
        if slot and self.steps_left() > 3:
            self._run(reader, {slot: pid}, "READ AGAIN AFTER THE NOTE")

    # -------------------------------------------------------------- questions to the shopper
    def can_ask(self) -> bool:
        # questions go on while the allowance lasts, before or after a line went in the cart
        return bool(self.names.get("speak")) and bool(self.text_arg(self.names["speak"])) \
            and self.questions_left() > 0 and self.steps_left() > self.reserve() + 3 and not self.easing(25.0)

    def sheeted(self, idx: int) -> list[str]:
        """Listings of the item's label read with their own property sheet; all those read, in a store that shows none."""
        with_sheet = [p for p in self.label_pids(idx) if self.index.sheet_of.get(p)]
        return with_sheet or (self.label_pids(idx) if self.sheetless and not self.sheet_readers else [])

    def wanted_sheets(self) -> int:
        """Listings read per item before planning."""
        return OPENED_BEFORE_ASKING if len(self.ledger.items) <= 2 else SHARED_SHEETS

    def ready_to_plan(self) -> bool:
        """Every item has its listings read or nothing more to read, or a third of the steps went by."""
        by_step = max(ASK_BY_STEP + 8, self.step_cap // 3) + 4 * (len(self.ledger.items) - 1)
        by_step = min(by_step, max(2, self.step_cap - self.reserve() - 4 - min(2, self.questions_left())))
        if self.steps >= by_step:
            return True
        return all(len(self.sheeted(n)) >= self.wanted_sheets() or not self.unread(n)
                   for n in range(len(self.ledger.items)))

    def question_due(self) -> tuple[int, str] | None:
        """The next message owed: the item whose turn it is and the property that leads. Pure: it reads and plans
        nothing. An item nothing was heard from goes first; an item whose last message came back unread waits behind
        the others; an item with no property left is not asked about."""
        if not self.can_ask() or self.plan_key is None:
            return None
        self.queue = [(i, q) for i, q in self.queue if self._askable(i, q)]
        items = len(self.ledger.items)
        firsts = {}
        for i, q in self.queue:
            firsts.setdefault(i, q)
        waiting = list(firsts)
        if not waiting:
            return None
        def order(i: int) -> tuple[bool, int, int, int, int]:
            told = self.heard_count.get(i, 0)
            place = self.talk_first.index(i) if i in self.talk_first else len(self.talk_first)
            return (i in self.puzzled_items, told, self.sent_about.get(i, 0), place, i)
        idx = min(waiting, key=order)
        return idx, firsts[idx]

    def _needs_key(self) -> str:
        return json.dumps(sorted(_need_key(n) + (_value_key(n["value"]),) for n in self.ledger.needs))

    def _guessed(self, need: dict[str, Any], idx: int) -> bool:
        """A need read from the request whose value no opened listing of the item shows under that key: a reading of
        the shopper's story in words the store does not use, so the property is still asked (the answer replaces the
        guess; a stated value some listing shows is no guess)."""
        if need.get("source") != "stated" or need["op"] not in ("is", "one of"):
            return False
        want = [_fold(v) for v in (need["value"] if isinstance(need["value"], list) else [need["value"]])]
        want = [w for w in want if w]
        if not want:
            return False
        shown = [_fold(value) for p in (self.label_pids(idx) or self.opened(idx))
                 for prop, value in self.fields_of(p).items()
                 if _alike(prop, need["property"]) or _keyfold(prop) == _fold(need["property"])]
        if not shown:
            return False                           # no listing carries the key: nothing to compare the value with
        return not any(w == s or set(w.split()) <= set(s.split()) or set(s.split()) <= set(w.split())
                       for w in want for s in shown if s)

    def _askable(self, idx: int, prop: str) -> bool:
        """A planned property still worth naming: not asked, not left open, not settled by a need or a tie of this item.
        A key that only adds words to a settled one (a part's own colour beside the item's colour) stays in."""
        key = _fold(prop)
        if not key or len(key.split()) > 3 or (idx, _fold(prop)) in {(i, _fold(q)) for i, q in self.asked}:
            return False
        item = self.ledger.items[idx]["name"] if idx < len(self.ledger.items) else ""
        if any(key == x for x in self.free_for(idx)):
            return False
        roles = [set(_fold(x).split()) for x in self.role_names() if _fold(x)]
        if any(set(key.split()) <= r for r in roles):
            return False
        settled = {f(n["property"]) for n in self.ledger.needs_for(item) if not n.get("when") and not self._guessed(n, idx)
                   for f in (_fold, _keyfold)}
        settled |= {_fold(r["property"]) for r in self.ledger.refs if _fold(r["item"]) == _fold(item)}
        return not any(x and (key == x or set(key.split()) < set(x.split())) for x in settled)

    def prepare_questions(self) -> bool:
        """Get the plan ready: search for an item with no rows by its name, read listings until the item has enough with
        their own property sheet, then plan once. Returns True when it spent a step."""
        if not self.can_ask() or (self.plan_key is not None and not self.plan_failed):
            return False
        cap = self.wanted_sheets() * (len(self.ledger.items) + 1)
        for n in range(len(self.ledger.items)):
            have = self.sheeted(n)
            if len(have) >= self.wanted_sheets() or self.steps_left() <= self.reserve() + 3 + self.questions_left():
                continue
            pages = [p for p in self.label_pids(n) if p not in have]
            rows = self._varied(self.by_mention(pages + self.unread(n), n), have)
            if not rows and self.index.seen_order:
                rows = self._likely_rows(n, have)   # the item's name is not the store's word for it: read what looks closest
            target = self._sheet_read(rows) if self.prep_reads < cap else None
            if target:
                self.prep_reads += 1
                self._run(target[0], {self.text_arg(target[0]): target[1]}, "READ BEFORE ASKING")
                if self.plan_failed:
                    self.plan_key, self.plan_failed = None, False   # a plan that found nothing is made again with more read
                return True
            if not rows and self._name_search(n):
                return True
        if not self.ready_to_plan():
            return False
        if not any(self.opened(n) for n in range(len(self.ledger.items))) and self.steps < max(ASK_BY_STEP, self.step_cap // 4):
            return False                           # nothing read yet: a plan now would have nothing to choose from
        if self.plan_key is not None and self.plan_failed and not any(self.sheeted(n) for n in range(len(self.ledger.items))):
            return False                           # a failed plan is made again only once a sheet has been read
        self.plan_questions()
        return False

    def _varied(self, pids: list[str], have: list[str]) -> list[str]:
        """Rows reordered so that each read before asking is of another product than those already read (a title
        opening not seen yet first): sibling variants of one product share one key set, and the questions are
        planned from the keys the read sheets show."""
        def stem(p: str) -> str:
            return " ".join(_tokens(self.index.rows.get(p, {}).get("title") or "")[:3])
        seen = {stem(p) for p in have}
        fresh, rest = [], []
        for p in pids:
            s = stem(p)
            if s and s not in seen:
                fresh.append(p)
                seen.add(s)
            else:
                rest.append(p)
        return fresh + rest

    def _sheet_read(self, pids: list[str]) -> tuple[str, str] | None:
        """The next (reader, listing) read: readers known to show a sheet, then readers not tried, then the rest; never a
        reader that already read that listing."""
        order = [r for r in self.readers if r in self.sheet_readers] + \
            [r for r in self.readers if r not in self.sheet_readers and r not in self.sheetless] + \
            [r for r in self.readers if r in self.sheetless and r not in self.sheet_readers]
        for pid in pids:
            for reader in order:
                if self.text_arg(reader) and reader not in self.read_by.get(pid, {}):
                    return reader, pid
        return None

    def menu(self, idx: int) -> dict[str, Any]:
        """Per property of the item's opened listings: how many carry it and the values it takes. Keys that name the
        listing, keep several options in one value, or hold free text are left out."""
        pids = self.sheeted(idx) or self.label_pids(idx) or self.opened(idx)
        twins = self._twins(pids)
        taken: dict[str, list[str]] = {}
        for p in pids:
            for prop, value in self._askable_fields(p):
                if prop not in twins:
                    taken.setdefault(prop, []).append(str(value))
        out: dict[str, Any] = {}
        for prop, values in sorted(taken.items(), key=lambda kv: -len(kv[1])):
            if _coded(values) or _no_choice(values) or self._several_at_once(values):
                continue
            distinct = list(dict.fromkeys(values))
            out[prop] = {"listings": len(values), "of": len(pids), "distinct": len(distinct), "values": distinct[:6]}
        return dict(list(out.items())[:30])

    def plan_questions(self) -> None:
        """One reading ranks each item's properties from the request's story; code keeps the names its tables carry and
        adds the rest of each table in its own order. The queue holds every property of every item, item by item."""
        self.plan_key = self._needs_key()
        led = self.ledger
        tables = [self.menu(n) for n in range(len(led.items))]
        self.plan_failed = not any(tables)
        items = [{"item": n + 1, "name": it["name"], "properties": tables[n],
                  "settled": [_need_text(x) for x in led.needs_for(it["name"]) if not self._guessed(x, n)]}
                 for n, it in enumerate(led.items)]
        got = {} if self.plan_failed else self._read(HEARING, PLAN_QUESTIONS.replace("COUNT", str(LIST_KEYS)), {
            "request": self.request, "story_hints": led.story_hints, "items": items,
            "shopper_does_not_mind": led.no_preference}, 1400, first_cap=25.0)
        per_item: dict[int, list[str]] = {n: [] for n in range(len(led.items))}
        for x in [_d(e) for e in got.get("ask")] if isinstance(got.get("ask"), list) else []:
            idx = self._item_number(x.get("item"))
            said = " ".join(_tokens(str(x.get("label") or "").replace("_", " ")))
            # a property the story points at stays in the list even when no opened listing spells it that way
            prop = (self._key_in(tables[idx], said) or (said if said and len(said.split()) <= 3 else None)) if idx is not None else None
            if prop is None or prop in per_item[idx]:
                continue
            per_item[idx].append(prop)
            if str(x.get("hint") or "").strip() and idx not in self.talk_first:
                self.talk_first.append(idx)
        self.rank_questions()
        for n, more in self.ranked_by_item.items():
            per_item[n] += [q for q in more if q not in per_item[n]]
        for n in per_item:
            if not per_item[n]:
                per_item[n] = self._bare_properties(n)
        self.queue = [(n, q) for n, props in per_item.items() for q in props if self._askable(n, q)]

    def asking_ahead(self) -> bool:
        return self.talk_owed()

    def rank_questions(self) -> tuple[int, str] | None:
        """The code's own order of each item's properties (how many listings carry it, then how its values split the
        listings; numeric keys a little lower; a hint from the story first). Returns the first item's first."""
        self.ranked_by_item = {}
        hints = {_fold(h["property"]) for h in self.ledger.story_hints}
        for idx in range(len(self.ledger.items)):
            ranked = []
            for prop, row in self.menu(idx).items():
                if not self._askable(idx, prop):
                    continue
                share = row["listings"] / max(1, row["of"])
                split = 1.0 - sum((row["values"].count(v) / max(1, len(row["values"]))) ** 2 for v in row["values"])
                rank = share * (0.8 if all(_figures(v) for v in row["values"]) else 1.0) + 0.2 * split
                if any(set(_fold(prop).split()) & set(h.split()) for h in hints if h):
                    rank += 10.0
                ranked.append((rank, prop))
            if ranked:
                self.ranked_by_item[idx] = [p for _, p in sorted(ranked, key=lambda r: (-r[0], r[1]))]
        return next(((i, q[0]) for i, q in self.ranked_by_item.items() if q), None)

    def _twins(self, pids: list[str]) -> set[str]:
        """Keys that repeat another key's value on most listings carrying both (the less common one is a twin), and
        keys whose values often join several variants with " / " (a variant label, not one property)."""
        tables = [dict(self._askable_fields(p)) for p in pids]
        count: dict[str, int] = {}
        joined: dict[str, int] = {}
        for t in tables:
            for k, v in t.items():
                count[k] = count.get(k, 0) + 1
                joined[k] = joined.get(k, 0) + (" / " in str(v))
        keys = sorted(count, key=lambda k: (-count[k], k))
        out: set[str] = {k for k in keys if joined[k] * 4 >= count[k] and joined[k]}
        for i, a in enumerate(keys):
            if a in out:
                continue
            for b in keys[i + 1:]:
                both = [t for t in tables if a in t and b in t]
                same = [_fold(t[a]) for t in both if _fold(t[a]) == _fold(t[b]) and _fold(t[a])]
                if len(both) >= 2 and len(same) * 10 >= 6 * len(both) and len(set(same)) >= 2:
                    out.add(b)
        return out

    def _askable_fields(self, pid: str) -> list[tuple[str, str]]:
        """A listing's fields that could be a need: not its id, price, stock, title or label, not long text, and not
        its own price under another key (a fact, not a wish)."""
        skip, own_price, out = self.role_names(), self.index.rows.get(pid, {}).get("price"), []
        sheet = self.index.sheet_of.get(pid)
        source = sheet if sheet else {k: v for k, v in self.fields.get(pid, {}).items() if k not in self.index.row_keys}
        for prop, value in source.items():
            if prop in skip or _keyfold(prop) in skip or len(str(value)) > 60 or _is_id_key(prop.replace(" ", "_")) \
                    or not str(value).strip() \
                    or len(str(value).split()) > 5:
                continue                           # ids, empty values and prose are no property to ask about
            figures = _figures(str(value))
            if isinstance(own_price, (int, float)) and figures and abs(figures[0] - own_price) < 1e-6:
                continue
            if _keyfold(prop) not in {k for k, _ in out}:
                out.append((_keyfold(prop), value))
        return out

    def ask(self, idx: int, prop: str) -> None:
        name = self.names["speak"]
        slot = self.text_arg(name)
        item = self.ledger.items[idx]["name"] if self.ledger.items else "item"
        spec = ((self.schemas[name].get("parameters") or {}).get("properties") or {}).get(slot, {})
        longest = spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else None
        item = re.sub(r",\s*[^,]+$", "", item) if len(item.split(",")[-1].split()) <= 2 and "," in item else item
        listed = [prop] + self._more_properties(idx, prop, LIST_KEYS - 1 if self.listing() else 0)
        while True:
            head, tail = listed[:-1], listed[-1]
            if head:
                text = "For the %s, which %s or %s would you like?" % (item, ", ".join(head), tail)
            else:
                text = "Which %s do you want for the %s?" % (tail, item)
            if not head or longest is None or len(text) <= longest:
                break
            listed.pop()                       # the store takes shorter messages: fewer properties ride along
        issues: list[str] = []
        args = _fit(self.schemas[name].get("parameters") or {}, {slot: text}, name, issues)
        if issues:
            self._after_reply(idx, listed, "no_need", [])
            return
        self.talk_now = listed
        self.pending_question, self.pending_item = prop, idx
        self.pending_text = " or ".join(listed)
        self.pending_list = listed if len(listed) > 1 else []
        self.reply_read = False
        self.sent_about[idx] = self.sent_about.get(idx, 0) + 1
        _, sent = self._run(name, args, "ASKED THE SHOPPER", 800)
        if sent:
            self.questions_sent += 1
            self.chat[-1]["content"] += "\n" + self.status()
        if not self.reply_read:
            self._after_reply(idx, listed, "no_need", [])
        self.pending_question = self.pending_text = ""
        self.pending_list, self.talk_now = [], []

    def _more_properties(self, idx: int, prop: str, count: int) -> list[str]:
        """The item's next planned properties, in order, none sharing a whole name with another in the message."""
        out: list[str] = []
        taken = [set(_fold(prop).split())]
        for i, q in self.queue:
            if len(out) >= count:
                break
            words = set(_fold(q).split())
            if i == idx and words and not any(words == t for t in taken) and self._askable(idx, q):
                out.append(q)
                taken.append(words)
        return out

    # -------------------------------------------------------------- the request's own steps
    def checklist(self, moment: str, pid: str | None = None) -> None:
        """Before an add, the order or ending without buying: the steps the request asks for, once per version. A
        light model lists those not done yet; code carries out the ones it can."""
        # before-add steps recur for every listing added; the others are owed once per version
        key = (moment, self.ledger.version, None) if moment != "before_add" else (moment, self.item_of(pid) if pid else -1)
        if moment == "before_add" and pid and self.item_of(pid) in self.announced:
            return                                 # the item was announced already; a swap within it needs no new word
        led = self.ledger
        stated = [s for s in led.steps if s["moment"] in (moment, "any") and (not s["done"] or s["moment"] == "before_add")]
        if key in self.moments_run or not (led.rules or stated) or self.steps_left() < 3:
            return
        if self.easing(15.0) and not stated:
            return
        self.moments_run.add(key)
        move = {"before_add": "put a listing in the cart", "before_order": "place the order",
                "before_decline": "end the trip without buying", "after_change": "go on after a change"}.get(moment, moment)
        trail = [str(m["tool_call"]["function"]["name"]) + " " + str(m["tool_call"]["function"]["arguments"])[:160]
                 for m in self.out if m.get("tool_call")][-24:]
        got = {} if self.easing(15.0) else \
            self._read(LIGHT, CHECKLIST % move, {"request": self.request, "rules": led.rules,
                                                 "steps": [{k: s[k] for k in ("moment", "action", "detail")} for s in led.steps],
                                                 "talk": self.talk_log[-8:], "calls_so_far": trail,
                                                 "cart": self.cart_summary(), "listing": pid}, 500, first_cap=20.0)
        steps = [_d(x) for x in got.get("steps")] if isinstance(got.get("steps"), list) else \
            [{"action": s["action"], "about": "cart", "text": ""} for s in stated]
        targets = [pid] if pid else [p for p, _ in self.cart_lines()]
        for step in steps[:3]:
            if self.done or self.stuck or self.steps_left() < 2:
                break
            action = str(step.get("action") or "").lower()
            if action == "open":
                self._open_all(targets, "OPENED AS THE REQUEST ASKS")
            elif action == "cart" and self.names.get("cart"):
                self._run(self.names["cart"], dict(self.fixed.get("cart", {})), "CART LOOKED AT AS THE REQUEST ASKS", 1500)
            elif action == "compare" and self.names.get("compare"):
                self._compare(targets)
            elif action == "message" and self.names.get("speak") and self.messages_left() > 0 \
                    and (moment != "before_add" or self.messages_left() > self.others_owed()):
                self._tell(moment, step, pid)
            for s in led.steps:
                if s["moment"] in (moment, "any") and s["action"] == action and s["moment"] != "before_add":
                    s["done"] = True

    def _tell(self, moment: str, step: dict[str, Any], pid: str | None) -> None:
        name = self.names["speak"]
        slot = self.text_arg(name)
        if not slot:
            return
        if moment == "before_add" and pid:
            self.announced.add(self.item_of(pid))
        own = str(step.get("text") or "").strip()
        if moment == "before_add" and pid:
            row = self.index.rows.get(pid, {})
            idx = self.item_of(pid)
            summary = "I am about to put %d x %s (%s) at %s in the cart." % (
                self.ledger.items[idx]["qty"], str(row.get("title") or "listing")[:80], pid, row.get("price"))
        elif moment in ("before_order", "after_change", "any") and self.basket:
            summary = self.cart_summary()
        else:
            summary = ""
        text = (summary + (" " + own if own and own not in summary else "")).strip() if \
            str(step.get("about") or "cart") != "other" or not own else own
        if not text:
            return
        issues: list[str] = []
        args = _fit(self.schemas[name].get("parameters") or {}, {slot: text[:700]}, name, issues)
        if not issues:
            self._run(name, args, "TOLD THE SHOPPER AS THE REQUEST ASKS", 800)

    def _open_all(self, pids: list[str], header: str, every: bool = True) -> None:
        """Each listing read with every reader tool that has not read it since the last notice or change of facts;
        with `every` False, one reading is enough (a listing already opened is left as it is)."""
        def enough(p: str) -> bool:                  # one reading suffices once the listing shows its sheet (or none exists)
            return p in self.index.opened and (bool(self.index.sheet_of.get(p)) or not self.sheet_readers
                                               or any(r in self.read_by.get(p, {}) for r in self.sheet_readers))
        waiting = [p for p in dict.fromkeys(pids) if not (enough(p) and not every)
                   and not any(self.read_by.get(p, {}).get(r, -1) >= max(self.event_version, 0) for r in self.readers)]
        while len(waiting) >= 2 and self.pair_reads is not False and self.steps_left() > 3 and not self.done \
                and not self.easing(12.0):
            size = self.group_size or self._group_cap()
            chunk = waiting[:size]
            if not self._read_pair(chunk, header):
                if len(chunk) > 2 and not self.group_size and self.pair_reads is False:
                    self.group_size, self.pair_reads = 2, None   # the store takes fewer at once: two, from now on
                    continue
                break
            if not self.group_size and len(chunk) == self._group_cap():
                self.group_size = len(chunk)
            waiting = waiting[len(chunk):]
        # the reader that shows the property sheet goes first: it is what settles a need
        order = [r for r in self.readers if r in self.sheet_readers] + [r for r in self.readers if r not in self.sheet_readers]
        for pid in pids:
            if not every and enough(pid):
                continue
            for reader in order[:None if every else 1]:
                slot = self.text_arg(reader)
                if not slot or self.read_by.get(pid, {}).get(reader, -1) >= max(self.event_version, 0) \
                        and reader in self.read_by.get(pid, {}):
                    continue
                if self.steps_left() <= 2 or self.easing(10.0) or self.done:
                    return
                self._run(reader, {slot: pid}, header)

    def _group_cap(self) -> int:
        """Listings a first compare read asks for: GROUP_READ, or fewer when the tool's own schema allows fewer."""
        name = self.names.get("compare")
        props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {} if name else {}
        arr = next((v for v in props.values() if (v or {}).get("type") == "array"), None) or {}
        most = arr.get("maxItems")
        return max(2, min(GROUP_READ, int(most))) if isinstance(most, int) else GROUP_READ

    def _pair_args(self, pids: list[str]) -> tuple[str, dict[str, Any]] | None:
        """The compare call that would read these listings, once a reader's page is known and pairs are not ruled out."""
        name = self.names.get("compare")
        if not name or not self.page_keys or name in self.readers or self.pair_reads is False:
            return None
        params = self.schemas[name].get("parameters") or {}
        arr = next((k for k, v in (params.get("properties") or {}).items() if (v or {}).get("type") == "array"), None)
        if not arr:
            self.pair_reads = False
            return None
        issues: list[str] = []
        args = _fit(params, {arr: list(pids)}, name, issues)
        return None if issues or len(args.get(arr) or []) != len(pids) else (name, args)

    def _read_pair(self, pids: list[str], header: str) -> bool:
        """Two listings read in one step through the compare tool, while it returns each listing's whole record (the
        keys a reader's page carries, with its property sheet). A refusal or a thinner record ends this for the trip."""
        call = self._pair_args(pids)
        if not call:
            return False
        observation, sent = self._run(call[0], call[1], header)
        return sent and self._took_pair(call[0], pids, observation)

    def _took_pair(self, name: str, pids: list[str], observation: Any, reader: str | None = None) -> bool:
        """Marks both listings read when the compare result carries each one's whole record (for `reader`, when the
        driver asked for that reader); else pairs stop."""
        inner = self._split_notes(name, observation)[0]
        found = {}
        def walk(v: Any) -> None:
            if isinstance(v, list):
                for x in v:
                    walk(x)
            elif isinstance(v, dict):
                key = self.index.id_key(v)
                if key and str(v[key]) in pids and any(isinstance(x, dict) for x in v.values()):
                    found[str(v[key])] = v
                else:
                    for x in v.values():
                        walk(x)
        walk(inner)
        if _failed(observation) or set(found) != set(pids):
            self.pair_reads = False
            return False
        for pid, record in found.items():
            tables = [len(v) for v in record.values() if isinstance(v, dict)]
            whole = [r for r, keys in self.page_keys.items() if keys <= set(record)
                     and tables and max(tables) >= self.page_sheet.get(r, 0)]
            if not whole or reader and reader not in whole:
                self.pair_reads = False
                return False
            self.index.opened.add(pid)
            for r in whole:
                self.read_by.setdefault(pid, {})[r] = self.ledger.version
                self.read_records[(pid, r)] = _dump(record, 3000)
        self.pair_reads = True
        return True

    def _compare(self, pids: list[str]) -> None:
        name = self.names["compare"]
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        arr = next((k for k, v in props.items() if (v or {}).get("type") == "array"), None)
        if not arr:
            return
        extra = [p for p in self.passed(self.item_of(pids[0]) if pids else 0) if p not in pids]
        most = int((props[arr] or {}).get("maxItems") or 5)
        issues: list[str] = []
        args = _fit(params, {arr: (pids + extra)[:most]}, name, issues)
        if not issues and len(args.get(arr) or []) >= 1:
            self._run(name, args, "COMPARED AS THE REQUEST ASKS")

    # -------------------------------------------------------------- commit
    def commit_due(self) -> bool:
        return self.cart_fits() and self.commit_key != (self.cart_mark, self.ledger.version) and self.commits < 5 \
            and self.steps_left() >= 2

    def commit(self) -> str:
        """Order a cart whose every line fits: hold once for a cheaper or better fitting listing when the objective
        asks for one, look at the cart, read each line with every reader, carry out the request's steps, order."""
        self.commit_key = (self.cart_mark, self.ledger.version)
        self.commits += 1
        lines = self.cart_lines()
        if not lines:
            return "nothing is in the cart yet; put the listing that fits in the cart first"
        for n in range(len(self.ledger.items)):
            mine = [p for p, i in lines if i == n]
            item = self.ledger.items[n]["name"]
            if len(mine) > 1 and self._guard("cart lines"):
                return "the cart holds %d lines for the %s (%s); keep one and take the others out" % (
                    len(mine), item, ", ".join(mine))
            if not mine and self._guard("cart lines"):
                return "the cart has no line for the %s yet" % item
            for pid in mine:
                r = self.check(pid)
                if not r["ok"] and self._guard("cart fit"):
                    swap = self.backup(n, {p for p, _ in lines})
                    return "the cart's %s does not fit (%s); %s" % (pid, "; ".join(r["why"][:3]),
                                                                    "swap it for %s, which fits" % swap if swap
                                                                    else "find a listing that fits")
                if not r["ok"] and (r.get("facts") or self.backup(n, {p for p, _ in lines})):
                    return self._settle_by_code()  # told twice: a store fact breaks it, or another listing fits
        cap = self.ledger.budgets.get("money_cap")
        spend = self.cart_total(lines)
        if cap and spend > cap + 1e-9 and self._fit_total(lines, float(cap)):
            return "lines were swapped for cheaper listings that fit, to stay within the money limit"
        if cap and spend > cap + 1e-9 and self._guard("total"):
            return "the cart totals %s, over the money limit of %s; swap a line for a cheaper listing that fits" % (spend, cap)
        if cap and spend > cap + 1e-9:
            return self._settle_by_code()
        held = self._objective_hold(lines)
        if held:
            return held
        version = self.ledger.version
        self._drop_gone()
        expected = [p for p, _ in self.cart_lines()]
        if self.names.get("cart") and self.cart_viewed != self.cart_mark and self.steps_left() > 2:
            self._run(self.names["cart"], dict(self.fixed.get("cart", {})), "CART LOOKED AT BEFORE ORDERING", 1500)
            self._restore(expected)
        self._drop_gone()                          # after the view: a gone line it shows leaves the store's cart
        missing = self.missing_items()
        if missing:
            return self._no_line(missing)
        self._open_all([p for p, _ in self.cart_lines()], "READ BEFORE ORDERING")
        while self.reread and self.steps_left() > 3 and not self.easing(15.0):
            self._reread_one()                     # a note named a price change: the new figure decides
        if self.ledger.version == version:
            self.checklist("before_order")
        if self.ledger.version != version and not self.cart_fits():
            return "the ledger changed just before ordering; see the change above"
        if self.over_money():
            if self._guard("total"):
                return "the cart totals %s, over the money limit of %s; swap a line for a cheaper listing that fits" % (
                    self.cart_total(), self.ledger.budgets.get("money_cap"))
            return self._settle_by_code()
        if not self.cart_lines():
            return "the cart turned out empty; put the listing that fits in the cart"
        if self.missing_items():
            return self._no_line(self.missing_items())
        held = self._final_look()
        if held:
            return held                            # the cart was swapped or found wrong: one round for the driver first
        return self.finish("buy")

    def cart_total(self, lines: list[tuple[str, int]] | None = None) -> float:
        return sum(float(self.index.rows.get(p, {}).get("price") or 0) * self.line_qty.get(p, 1)
                   for p, _ in (self.cart_lines() if lines is None else lines))

    def over_money(self) -> bool:
        cap = self.ledger.budgets.get("money_cap")
        return bool(cap) and self.cart_total() > float(cap) + 1e-9

    def _settle_by_code(self) -> str:
        """The driver was told twice and the cart still breaks a need or the money limit: code keeps what fits, fills
        the rest with listings that fit, within the limit, and orders; when that cannot be done, the trip ends through
        the decline path (which, for a list of several items with a buyable line for each, orders that cart)."""
        self._order_what_fits()
        return "the trip is over" if self.done else self.decline()

    def _no_line(self, missing: list[str]) -> str:
        """Items without a cart line: told to the driver twice (shared cart-lines guard); after that code orders what
        fits, and failing that ends through the decline audit."""
        if self._guard("cart lines"):
            lost = [m for m in missing if any(self.item_of(p) == n for p in self.carted
                                              for n, it in enumerate(self.ledger.items) if it["name"] == m)]
            return "the cart has no line for the %s%s; put a listing that fits in the cart for it" % (
                ", ".join(missing), " (the store's cart lost it)" if lost else " yet")
        self._order_what_fits()
        return "the trip is over" if self.done else self.decline()

    def missing_items(self) -> list[str]:
        """Items of the list with no cart line (as far as the last cart view and our calls tell)."""
        have = {i for _, i in self.cart_lines()}
        return [it["name"] for n, it in enumerate(self.ledger.items) if n not in have]

    def _objective_hold(self, lines: list[tuple[str, int]]) -> str:
        """Under a cheapest rule: once, while cheaper in-label rows within the limit are unchecked, read them and say
        when one fits. Under a best rule: once, name a fitting listing that does better on the measure."""
        kind = self.ledger.objective.get("kind")
        # with no way of choosing named, a listing that meets every need the pick meets for less is still the
        # better buy: look under the pick's price too, swapping only by code and saying nothing otherwise
        plain = kind not in ("cheapest", "best")
        if self.fired.get("objective") or self.steps_left() <= (self.reserve() + 6 if plain else 4) \
                or self.easing(20.0):
            return ""
        self.fired["objective"] = 1
        said, moved = [], {}
        for pid, idx in lines:
            item = _fold(self.ledger.items[idx]["name"]) if idx < len(self.ledger.items) else ""
            if plain and (self.steps_left() <= self.reserve() + 6
                          or any(_fold(r["item"]) == item and r["status"] != "resolved" for r in self.ledger.refs)):
                continue                           # short of steps, or a tie still open may be what the pick rests on
            price = self.index.rows.get(pid, {}).get("price")
            if kind != "best" and not isinstance(price, (int, float)):
                self._open_all([pid], "THE PICK'S OWN PAGE, FOR ITS PRICE")
                price = self.index.rows.get(pid, {}).get("price")
            if kind == "best":
                better = [p for p in self.passed(idx, {pid}) if self.measure(p) is not None and (
                    self.measure(pid) is None or (self.measure(p) > self.measure(pid)
                                                  if self.ledger.objective.get("direction") != "min"
                                                  else self.measure(p) < self.measure(pid)))]
                if better:
                    said.append("%s fits and does better on %s (%s against %s for %s)" % (
                        better[0], self.ledger.objective["criterion"], self.measure(better[0]), self.measure(pid), pid))
                continue
            if not isinstance(price, (int, float)):
                continue
            label, cap = self.label_of(idx), self.line_cap(idx)
            below = price
            for turn in range(2 if plain else 4):
                if plain and self.steps_left() <= self.reserve() + 6:
                    break
                # a search capped under the pick's price shows the cheaper listings plain paging may never reach;
                # without one, a search on the item's stated values fills the cheap end once
                probed = self._cap_probe(idx, below)
                if not probed and not turn:
                    self._value_probe(idx, price)
                fresh = sorted(self.index.rows[p]["price"] for p in (self.probe_shown if probed else [])
                               if isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                               and self.index.rows[p]["price"] < below - 1e-9)
                cheaper = sorted((p for p, r in self.index.rows.items() if p != pid and p not in self.gone
                                  and isinstance(r.get("price"), (int, float)) and r["price"] < price - 1e-9
                                  and r.get("in_stock") is not False and (not cap or r["price"] <= cap + 1e-9)
                                  and (not label or not r.get("category") or _alike(r["category"], label))
                                  and self.item_of(p) == idx),
                                 key=lambda p: self.index.rows[p]["price"])
                unread = self.by_mention([p for p in cheaper if p not in self.index.opened][:6], idx)[:3]
                room = max(0, self.steps_left() - self.reserve() // 2 - 2)
                self._open_all(unread[:room], "CHEAPER ROW READ BEFORE ORDERING", every=False)
                opened = [p for p in cheaper if p in self.index.opened]
                if plain:                          # settled by code alone first; no model reads for the rest
                    opened = [p for p in opened if (lambda r: r["ok"] and not r["unclear"] and not r.get("near"))(
                        self.check(p, full=False))]
                fits = [p for p in opened if self.check(p)["ok"]]
                # with no rule named, only a listing whose every need sits under the property's own key, read by
                # code alone, takes the pick's place: one settled from loose text proves too little
                sure = [p for p in fits if self.replaces(p, pid)
                        and not (plain and (self.check(p).get("near") or self.check(p).get("asked_model")
                                            or not self._agrees(p, idx) or not self._as_shown(p, pid, idx)))]
                if sure and self._swap(pid, sure[0], idx):
                    start = moved.get(idx, (pid, price))
                    moved[idx] = start
                    pid, price = sure[0], self.index.rows[sure[0]]["price"]
                    below = price
                    if probed:
                        continue                   # look under the new, lower price too
                elif fits and not plain:
                    said.append("%s fits and costs %s, less than %s for the cart's %s" % (
                        fits[0], self.index.rows[fits[0]]["price"], price, pid))
                elif probed and len(fresh) >= 2:
                    below = fresh[len(fresh) // 2]  # none read fits: look again under the middle of what it showed
                    continue
                elif probed and unread and any(p not in self.index.opened for p in cheaper):
                    continue                       # the capped rows read so far did not fit; read the next cheapest
                break
        for idx, (old, was) in moved.items():
            now = next((p for p, i in self.cart_lines() if i == idx), None)
            if now:
                said.insert(0, "code swapped %s (%s) for %s (%s), which settles every need for less; it is already in "
                               "the cart, leave that line as it is" % (old, was, now, self.index.rows.get(now, {}).get("price")))
        if plain:
            return ""                              # the swaps are done; the order goes ahead
        if said and len(said) == len(moved):
            return "before ordering: " + " | ".join(said)
        if said:
            return "before ordering: " + " | ".join(said) + "; put it in the cart in place of the current line " \
                "(code orders once the cart fits)"
        return ""

    def _as_shown(self, new: str, old: str, idx: int) -> bool:
        """`new` shows each need at least as fully as `old`: every key of `old` that speaks to a need's property
        (its words within the key's, or the key's within its) is a key of `new` too."""
        def keys(pid: str, words: set[str]) -> set[str]:
            out = set()
            for k in self.fields_of(pid):
                name = set(_prop_name(k).split())
                if name and (words <= name or name <= words):
                    out.add(_prop_name(k))
            return out
        for need in self.ledger.needs_for(self.ledger.items[idx]["name"]):
            words = set(_fold(need["property"]).split())
            if words and not keys(old, words) <= keys(new, words):
                return False
        # nor names the item less plainly in its title than `old` does
        def stems(text: Any) -> set[str]:
            return {w.rstrip("s") for w in _tokens(text) if len(w) > 2}
        item = stems(self.ledger.items[idx]["name"])
        def named(pid: str) -> set[str]:
            return item & stems(self.index.rows.get(pid, {}).get("title"))
        return named(old) <= named(new)

    def _agrees(self, pid: str, idx: int) -> bool:
        """A listing that never contradicts itself: no two keys of one name within a table share no word, and for
        each of the item's need properties the tables that name it share a word."""
        if pid in self.torn:
            return False
        said = self.said_by.get(pid, {})
        for need in self.ledger.needs_for(self.ledger.items[idx]["name"]):
            name = _prop_name(need["property"])
            sets = [w for (table, n), w in said.items() if n == name]
            if any(not a & b for a in sets for b in sets):
                return False
        return True

    def _cap_tool(self) -> tuple[str, str] | None:
        """The search that takes a highest price, with that argument: the search tool's own when it has one, else
        another tool doing no other job whose schema has a number without an upper bound."""
        def numeric(props: dict[str, Any], skip: set[str]) -> list[str]:
            return [k for k, v in props.items() if k not in skip and (v or {}).get("type") in ("number", "integer")
                    and "maximum" not in (v or {}) and "enum" not in (v or {})]
        find = self.names.get("find")
        if find:
            props = (self.schemas[find].get("parameters") or {}).get("properties") or {}
            given = self.find_args.get("price_cap")
            if given in props and (props[given] or {}).get("type") in ("number", "integer"):
                return find, given
            slots = numeric(props, {self.find_args.get("page") or ""})
            if len(slots) == 1:
                return find, slots[0]                # one unbounded number beside the words: the price limit
        # another tool only when it browses the store's own shelves (an enum argument naming the item's label)
        busy = set(self.names.values()) | set(self.readers)
        labels = [self.label_of(n) for n in range(len(self.ledger.items))]
        for name, schema in self.schemas.items():
            props = (schema.get("parameters") or {}).get("properties") or {}
            slots = numeric(props, set())
            shelf = any(lab and any(_alike(c, lab) for c in (v or {}).get("enum") or []) for v in props.values()
                        for lab in labels)
            if name not in busy and len(slots) == 1 and shelf:
                return name, slots[0]
        return None

    def _cap_probe(self, idx: int, below: Any) -> bool:
        """One search capped just under a price for an item (once per item and price): listings cheaper than the pick
        come into view however the plain searches page. True when it was sent."""
        found = self._cap_tool()
        if not found or not isinstance(below, (int, float)) or self.steps_left() <= self.reserve() + 4 \
                or self.easing(30.0) or idx >= len(self.ledger.items):
            return False
        key = (idx, float(below), self.event_version)
        if key in self.probed:
            return False
        self.probed.add(key)
        name, slot = found
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        prices = [r["price"] for r in self.index.rows.values() if isinstance(r.get("price"), (int, float))]
        whole = all(float(x).is_integer() for x in prices + [below])
        cap = below - 1 if whole else round(below - 0.01, 2)
        if (props.get(slot) or {}).get("type") == "integer" and not whole:
            cap = math.ceil(below) - 1
        if cap <= 0:
            return False
        item = self.ledger.items[idx]
        needs = [n for n in self.ledger.needs_for(item["name"]) if n["op"] in ("is", "has")]
        value = needs[0]["value"] if needs else ""
        value = value[0] if isinstance(value, list) and value else value
        if name == self.names.get("find"):
            args = self._search_args(("%s %s" % (value, item["name"])).strip()) or self._search_args(item["name"])
            if args is None:
                return False
        else:
            args = {}
            label = self.label_of(idx)
            for k, v in props.items():
                choices = (v or {}).get("enum") or []
                hit = next((c for c in choices if label and _alike(c, label)), None)
                if hit is not None:
                    args[k] = hit
                elif k in (params.get("required") or []) and k != slot:
                    if not choices:
                        return False
                    args[k] = choices[0]
        args[slot] = int(cap) if (props.get(slot) or {}).get("type") == "integer" or whole else cap
        if args[slot] <= 0:
            return False
        observation, sent = self._run(name, args, "LISTINGS UNDER %s (a search capped at a price)" % args[slot])
        self.probe_shown = _ids_in(_notes_beside(observation)[0], self.index.ids) if sent else []
        return sent

    def _value_probe(self, idx: int, price: Any) -> bool:
        """When the store has no search capped at a price: one search on an item's stated values within its own shelf
        (once per item and notice count), sent only while fewer than two cheaper rows of the item are unread."""
        if self._cap_tool() or idx >= len(self.ledger.items) or not isinstance(price, (int, float)) \
                or self.steps_left() <= self.reserve() + 6 or self.easing(30.0):
            return False
        key = (idx, -1.0, self.event_version)
        if key in self.probed:
            return False
        label = self.label_of(idx)
        waiting = [p for p in self.unread(idx) if isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                   and self.index.rows[p]["price"] < price - 1e-9
                   and (not label or not self.index.rows[p].get("category") or _alike(self.index.rows[p]["category"], label))]
        if len(waiting) >= 2:
            return False
        self.probed.add(key)
        item = self.ledger.items[idx]
        values: list[str] = []
        for n in self.ledger.needs_for(item["name"]):
            if n["op"] in ("is", "has"):
                v = n["value"][0] if isinstance(n["value"], list) and n["value"] else n["value"]
                # only a value the store itself shows for that property is worth searching
                vocab = {x for p in self.fields for x in _pieces(_field(self.fields_of(p), n["property"]) or "")}
                words = set(_fold(v).split()) if isinstance(v, str) else set()
                if words and any(words <= set(x.split()) or set(x.split()) <= words for x in vocab) \
                        and v.strip().lower() not in values:
                    values.append(v.strip().lower())
        if not values:
            return False                           # the item's name alone repeats a search already made
        words = " ".join(values[:2] + [item["name"]]).strip()
        args = self._search_args(words)
        if args is None:
            return False
        args.update(self._shelf_args(label or ""))
        page = self.find_args.get("page")
        if _search_key(args, page) in self.searches:
            return False
        return self._run(self.names["find"], args, "LISTINGS FOR THE STATED VALUES, CHEAPER THAN THE PICK")[1]

    def _fit_total(self, lines: list[tuple[str, int]], cap: float) -> bool:
        """Bring the cart within the money limit: the dearest lines first, each swapped for the cheapest listing
        that settles every need within what the other lines leave. True when a swap was made."""
        def cost(p: str) -> float:
            return float(self.index.rows.get(p, {}).get("price") or 0) * self.line_qty.get(p, 1)
        swapped = False
        for pid, idx in sorted(lines, key=lambda x: -cost(x[0])):
            spend = sum(cost(p) for p, _ in self.cart_lines())
            if spend <= cap + 1e-9 or self.steps_left() <= 4:
                break
            share = (cap - (spend - cost(pid))) / max(1, int(self.ledger.items[idx].get("qty") or 1))
            cands = [p for p in self.prefer([q for q in self.opened(idx) if q != pid and q not in self.gone])
                     if isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                     and self.index.rows[p]["price"] <= share + 1e-9 and self.replaces(p, pid)]
            if cands and self._swap(pid, cands[0], idx):
                swapped = True
        return swapped

    def _drop_gone(self) -> None:
        """Take out of the cart the lines that can no longer be bought (they still sit in the store's cart). Our own
        basket follows only a removal call that went through."""
        for pid in [p for p in self.basket if p in self.gone]:
            if self.steps_left() <= 2:
                break
            args = self._remove_args(pid)
            if args and self.steps_left() > 2:
                self._run(self.names["add"], args, "TAKEN OUT: NO LONGER AVAILABLE")

    def _swap(self, old: str, new: str, idx: int) -> bool:
        """Put `new` in the cart in place of `old` for one item (the add tool's own way of taking a line out); the
        request's before-add steps run for `new` first."""
        add = self.names.get("add")
        if not add or self.steps_left() <= 3:
            return False
        self._before_add(new)
        if self.steps_left() <= 3:                 # the removal, the add and the order must still fit
            return False
        args = self._remove_args(old)
        if not args or not self._run(add, args, "TAKEN OUT FOR A CHEAPER FIT")[1] or old in self.basket:
            return False
        args = self._add_args(new, self.ledger.items[idx]["qty"])
        return bool(args) and self._run(add, args, "PUT IN THE CART: A CHEAPER FIT")[1] and new in self.basket

    def _before_add(self, pid: str) -> None:
        """A line code puts in the cart itself still gets the request's before-add steps."""
        if any(s["moment"] == "before_add" for s in self.ledger.steps):
            self.checklist("before_add", pid)

    def _restore(self, expected: list[str]) -> None:
        """After a cart view: put back any line we meant to keep that the store's cart does not hold (only when the
        view named listings we know; otherwise there is nothing to compare)."""
        if not self.view_readable:
            return
        for pid in [p for p in expected if p not in self.basket and p not in self.gone]:
            args = self._add_args(pid, self.line_qty.get(pid, self.ledger.items[self.item_of(pid)]["qty"]))
            if args and self.steps_left() > 2:
                self._run(self.names["add"], args, "PUT BACK: THE CART HAD LOST IT")

    def finish(self, job: str, text: str | None = None) -> str:
        """Order or end without buying. A call that neither ends the trip nor comes back refused gets one mended call
        (the schema's holes filled, else a corrected reading). A note on an order is read for what it changes."""
        name = self.names.get(job)
        text = text or ("Placing the order for the cart." if job == "buy" else self._decline_text())
        args = self.closing_args(job, text)
        if name and args is None and self.names.get("buy") == self.names.get("none") and not self.mode_retried:
            self.mode_retried = True               # one more reading of the mode, then the tool's own words
            self._tell_finish_apart(TERSE)
            if not self._split_finish():
                self._mode_by_words(name)
            args = self.closing_args(job, text)
        if not name or args is None:
            return "the %s tool could not be called" % ("order" if job == "buy" else "ending")
        if job == "buy" and self.note_queue and self._take_cart_notes() and not self.cart_fits():
            return "a note changed the cart just before the order; %s" % self.status()
        while True:
            version = self.ledger.version
            needs_before = {_need_key(n) + (_value_key(n["value"]),) for n in self.ledger.needs}
            observation, sent = self._send(name, args)
            if not sent:
                return "the call did not go through: %s" % _dump(observation, 300)
            self.last_note = _notes_beside(observation)[1] + self._refusal_lines(observation)
            notes = self.absorb(name, args, observation)
            self.chat.append({"role": "user", "content": "%s (run for you): %s %s\nRESULT: %s%s" % (
                "ORDER" if job == "buy" else "ENDED WITHOUT BUYING", name, json.dumps(args, ensure_ascii=False),
                _dump(observation, 2000), ("\nnote: " + " ".join(notes)) if notes else "")})
            if self.done:
                return "the trip is over"
            problem = _failed(observation)
            store_said_no = job == "buy" and bool(problem) and self.names.get("buy") != self.names.get("none") \
                and "step" not in _tokens(problem.replace("_", " "))
            if store_said_no and (self._lines_named(problem) or self.raw_tries >= 1):
                # a well-formed order the store itself would not take: a line can no longer be bought (a fault that
                # names no line gets one corrected call first, below; the same call again is the store's refusal)
                self._order_turned_down(problem)
                return "the store did not take the order: %s; %s" % (problem[:200], self.status())
            grew = bool({_need_key(n) + (_value_key(n["value"]),) for n in self.ledger.needs} - needs_before)
            if self.refused_last and self.resent < RESENDS and self._send_again(job, grew if job == "buy" else self.ledger.version != version):
                self.resent += 1
                args = self.closing_args(job, text) or args
                continue                           # nothing the note said changes the choice: the same finish again
            if self.refused_last:
                return "it came back refused: %s; %s" % (" ".join(notes), self.status())
            if self.raw_tries >= RAW_FINISH_TRIES or self.steps_left() < 1 or self.easing(5.0):
                return "the call did not finish the trip: %s" % _dump(observation, 300)
            self.raw_tries += 1
            fixed = self._mend_finish(name, args, job, observation)
            if fixed is None or fixed == args:
                if store_said_no:
                    self._order_turned_down(problem)
                    return "the store did not take the order: %s; %s" % (problem[:200], self.status())
                return "the call did not finish the trip: %s" % _dump(observation, 300)
            args = fixed

    def _send_again(self, job: str, changed: bool) -> bool:
        """A finish that came back with a note: what happens next depends on what the note says. Sold out: the line changes.
        A price moved: the budget is checked again and the cart usually stays. A need dropped: a cheaper fit is looked for.
        An ending goes again only when still no item has a listing that fits."""
        if self.steps_left() < 1 or self.easing(10.0):
            return False
        if job != "buy":
            return not changed and not any(self.check(p, full=False)["ok"] for n in range(len(self.ledger.items))
                                           for p in self.opened(n))
        kinds, named = self._note_kinds(self.last_note) if self.last_note else ([], "")
        lines = [p for p, _ in self.cart_lines()]
        if {"sold_out", "price_up", "price_down"} & set(kinds):
            self.reread = list(dict.fromkeys(self.reread + lines))
        touched = bool(self.reread)
        while self.reread and self.steps_left() > 3 and not self.easing(10.0):
            self._reread_one()
        held = list(self.basket)
        if self.names.get("cart") and self.steps_left() > 3 and (touched or changed):
            self._run(self.names["cart"], dict(self.fixed.get("cart", {})), "CART LOOKED AT AFTER THE NOTE", 1500)
        out = [p for p in held if self.index.rows.get(p, {}).get("in_stock") is False]
        lost = [p for p in held if self.view_readable and self.basket and p not in self.basket]
        gone = sorted(set(out) | set(lost))
        if gone:
            self.gone |= set(gone)
            self.ledger.version += 1
            self.event_version = self.ledger.version
        elif "sold_out" in kinds:
            self._mark_sold_out(named)
            gone = [p for p in held if p in self.gone]
        if gone:
            self.chat.append({"role": "user", "content": "SOLD OUT AFTER THE NOTE: %s can no longer be bought; the cart is "
                              "planned again.\n%s" % (", ".join(gone), self.status())})
            return self._replan_cart() and not self.done
        self._drop_gone()
        if "need_dropped" in kinds:
            self.fired.pop("objective", None)      # the next commit looks for a listing that now fits for less
            if self.steps_left() > self.reserve() + 3:
                return False
        if self.over_money():
            lines_now = self.cart_lines()
            cap = self.ledger.budgets.get("money_cap")
            if cap and self._fit_total(lines_now, float(cap)) and not self.over_money():
                return self.steps_left() >= 1 and not self.missing_items()
            return False
        if self.missing_items() or self.steps_left() < 1:
            return False
        if "price_up" in kinds and self.ledger.objective.get("kind") != "best":
            self._cheaper_after_rise()             # under a price rule, a cheaper listing already read that fits takes over
        return not changed or self.cart_fits()

    def _record_for(self, pid: str, size: int) -> str:
        """The fullest account of a listing for a strong reading: its row (title, price, stock, label) and every field
        read from its pages, with a stretch of the page text when no field was read."""
        row = self.index.rows.get(pid, {})
        fields = self.fields_of(pid)
        rec = {"id": pid, "title": row.get("title"), "price": row.get("price"), "in_stock": row.get("in_stock"),
               "label": row.get("category"), "fields": fields}
        text = max((self.read_records.get((pid, r), "") for r in self.readers), key=len, default="")
        return _dump(rec, size) + ((" | page: " + text[:size // 2]) if text and len(fields) < 2 else "")

    def _final_look(self) -> str:
        """Before code orders: one strong reading of the cart against the request, with the other listings seen as
        the alternatives; once per cart and ledger version. A line it finds wrong is swapped for the better opened
        listing it names (when that one can still be bought and code finds nothing against it); otherwise the problem
        is told to the driver and the order waits one round. Skipped when time or steps are short."""
        lines = self.cart_lines()
        key = (self.cart_mark, self.ledger.version)
        if not lines or key in self.looked or self.hurry or self.easing(30.0) or self.steps_left() < 4:
            return ""
        self.looked.add(key)
        cart = {p: {"item": self.ledger.items[i]["name"], "units": self.line_qty.get(p, 1),
                    "record": self._record_for(p, 2500)} for p, i in lines}
        others = [p for p in self.prefer([q for q in self.index.opened if q not in cart and q not in self.gone])][:15]
        got = self._read(STRONG, FINAL_LOOK, {
            "request": self.request, "ledger": self.ledger.summary(), "cart": cart,
            "other listings seen": {p: self._record_for(p, 1200) for p in others}}, 700, first_cap=45.0,
            share=LATE_SHARE)
        if not got or str(got.get("ok")).lower() in ("true", "none", ""):
            return ""
        problems = [str(x) for x in got.get("problems") if isinstance(got.get("problems"), list)][:4] \
            if isinstance(got.get("problems"), list) else []
        swapped: list[str] = []
        for x in got.get("better") if isinstance(got.get("better"), list) else []:
            x = _d(x)
            new = str(x.get("listing") or "").strip()
            idx = next((n for n, it in enumerate(self.ledger.items) if _alike(it["name"], x.get("item"))), None)
            if idx is None and len(self.ledger.items) == 1:
                idx = 0
            old = next((p for p, i in lines if i == idx), None)
            if idx is None or not old or new == old or new not in self.index.opened or new in self.gone \
                    or new in self.swapped_out or self.check(new, full=False)["why"] or self.steps_left() < 4:
                continue                           # only an opened, buyable listing code finds nothing against
            if self._swap(old, new, idx):
                self.swapped_out.add(old)
                swapped.append("%s now stands for the %s in place of %s" % (new, self.ledger.items[idx]["name"], old))
        if not swapped and self.steps_left() <= self.reserve() + 6:
            return ""                              # nothing to swap in and little room to look further: the order goes
        said = "the final look found the cart wrong: %s%s" % (
            "; ".join(problems) or "a line does not meet its item's needs",
            ("; " + "; ".join(swapped)) if swapped else "; no better opened listing to swap in")
        return said                                # the caller shows it to the driver with the status

    def _last_look(self) -> str:
        """Before ending without buying: one strong reading of the opened records and the unread rows' titles names,
        per item, a listing that meets every need and rows worth opening; code opens up to three of those rows and,
        when every item has a named listing code finds nothing against, puts them in the cart and orders. Once per
        ledger version and set of opened listings; skipped when time or steps are short."""
        key = ("end", self.ledger.version, len(self.index.opened))
        if key in self.looked or self.hurry or self.easing(30.0) or self.steps_left() < 4 or not self.names.get("add"):
            return ""
        self.looked.add(key)
        opened = {p: self._record_for(p, 1200) for p in self.opened()[-30:] if p not in self.gone}
        unread = {p: str(self.index.rows.get(p, {}).get("title") or "")[:90]
                  for n in range(len(self.ledger.items)) for p in self.by_mention(self.unread(n), n)[:6]}
        got = self._read(STRONG, LAST_LOOK, {
            "request": self.request, "ledger": self.ledger.summary(), "opened listings": opened,
            "rows not opened (id: title)": unread}, 600, first_cap=45.0, share=LATE_SHARE)
        if not got:
            return ""
        to_open = [str(_d(x).get("listing") or x).strip() for x in (got.get("open") if isinstance(got.get("open"), list) else [])]
        to_open = [p for p in to_open if p in unread][:3]
        if to_open and self.steps_left() > 3 + len(self.ledger.items):
            self._open_all(to_open, "OPENED AT THE LAST LOOK", every=False)
        picks: dict[int, str] = {}
        for x in got.get("fits") if isinstance(got.get("fits"), list) else []:
            x = _d(x)
            pid = str(x.get("listing") or "").strip()
            idx = next((i for i, it in enumerate(self.ledger.items) if _alike(it["name"], x.get("item"))), 0 if len(self.ledger.items) == 1 else None)
            if idx is None or idx in picks or pid not in self.index.opened or pid in self.gone or self.check(pid, full=False)["why"]:
                continue
            picks[idx] = pid
        for idx in range(len(self.ledger.items)):   # a row just opened fills an item the reading left without a fit
            if idx not in picks:
                found = next((p for p in to_open if p in self.index.opened and self.item_of(p) == idx
                              and not self.check(p, full=False)["why"]), None)
                if found:
                    picks[idx] = found
        if len(picks) < len(self.ledger.items):
            return "do not end yet: rows were opened at the last look" if to_open else ""
        moves = sum(len([p for p, i in self.cart_lines() if i == idx and p != pid]) + (0 if any(p == pid for p, i in self.cart_lines()) else 1)
                    for idx, pid in picks.items())
        if self.steps_left() < moves + 1:
            return ""                              # the removals, the adds and the order would not all fit
        for idx, pid in picks.items():
            for old in [p for p, i in self.cart_lines() if i == idx and p != pid]:
                args = self._remove_args(old)
                if args and self.steps_left() > 1:
                    self._run(self.names["add"], args, "TAKEN OUT AT THE LAST LOOK")
            if not any(p == pid for p, i in self.cart_lines()):
                args = self._add_args(pid, self.ledger.items[idx]["qty"])
                if args and self.steps_left() > 1:
                    self._run(self.names["add"], args, "PUT IN THE CART AT THE LAST LOOK")
        if self._whole_cart_stands() and self.steps_left() >= 1:
            self.finish("buy")                     # the lines passed the check and the look: the order goes directly
            return "the trip is over" if self.done else "the last look named listings that fit; the order was tried"
        return ""

    def _lines_named(self, text: str) -> list[str]:
        """The cart lines whose id the text carries as a whole word."""
        return [p for p, _ in self.cart_lines() if len(p) >= 2
                and re.search(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(p), text)]

    def _order_turned_down(self, problem: str) -> None:
        """The store itself would not take a well-formed order (a fault, not a shopper's note): the line the fault
        names, or the only line, can no longer be bought; with several lines and no name, each is read again first and
        a second fault on the same cart takes them all out. The ledger version moves so the change is acted on."""
        lines = [p for p, _ in self.cart_lines()]
        named = self._lines_named(problem)
        self.order_faults += 1
        if self.fault_mark != (self.cart_mark, self.ledger.version):
            self.fault_mark, self.cart_faults = (self.cart_mark, self.ledger.version), 0
        self.cart_faults += 1
        if named or len(lines) == 1:
            self.gone |= set(named or lines)
        elif self.cart_faults >= 2:
            self.gone |= set(lines)
        else:
            self.reread = list(dict.fromkeys(self.reread + lines))
            return
        self.ledger.version += 1
        self.chat.append({"role": "user", "content": "THE STORE DID NOT TAKE THE ORDER: %s; %s can no longer be bought "
                          "and leave the cart; another listing that fits takes the place.\n%s" % (
                              problem[:200], ", ".join(sorted(set(named or lines))), self.status())})

    # -------------------------------------------------------------- decline
    def may_decline(self) -> bool:
        """Ending without buying is open only near the end: few steps or little time left, or nothing left to try."""
        return self.steps_left() <= self.reserve() + 3 or self.remaining() < SECONDS * EASE_OFF * 0.6 or self.out_of_moves \
            or self.hurry

    def decline(self) -> str:
        """Ending without buying is the last resort. Search again by values, read unread pages and open the likely rows (the
        audit); order a cart that stands; then buy the closest listing per item, each of its own kind and within the money
        limit. Only an item the store shows nothing of, or a limit no such cart can meet, ends the trip without an order."""
        self._talk_first()
        if self.steps_left() > 4 and not self.easing(20.0) and self.fired.get("decline", 0) < GUARD_TRIP_LIMIT:
            found = self._decline_audit()
            if found and self._guard("decline"):
                return found
        if self.names.get("add") and all(self.passed(n) or self._solid_opened(n) or
                                         any(self.loose_ok(p) for p in self._passing(n))
                                         for n in range(len(self.ledger.items))):
            self._order_what_fits(loose=True)      # every item has a listing that passes: ordering beats ending empty
            if self.done:
                return "the trip is over"
        if self.names.get("add") and self.ledger.several() and self._every_item_held() and not self.over_money():
            self._order_cart_as_is(held=True)      # several items, each with a line that can still be bought
            if self.done:
                return "the trip is over"
        if self.names.get("add") and self._whole_cart_stands():
            self._order_cart_as_is()               # a cart nothing the store showed speaks against goes as it stands
            if self.done:
                return "the trip is over"
        if self.names.get("add"):
            self._open_for_closest()
            if all(self._closest(n) for n in range(len(self.ledger.items))):
                self._order_what_fits(loose=True, closest=True)   # the closest opened listing per item beats an empty trip
                if self.done:
                    return "the trip is over"
        lead = self._last_look()
        if lead:
            return lead
        if self.closing_args("none", PLAIN_DECLINE) is None:
            self._order_what_fits()
            return "the trip is over" if self.done else "ending without buying cannot be told apart from ordering here"
        self.checklist("before_decline")
        return self.finish("none")

    def _every_item_held(self) -> bool:
        """One cart line for every item of the list, none gone or shown out of stock."""
        lines = self.cart_lines()
        return bool(lines) and len(lines) == len(self.ledger.items) and {i for _, i in lines} == set(range(len(self.ledger.items))) \
            and not any(p in self.gone or self.index.rows.get(p, {}).get("in_stock") is False for p, _ in lines)

    def _whole_cart_stands(self) -> bool:
        """One cart line for every item, none broken by a store fact (gone, out of stock, another section, over a
        limit), and the total within the money limit."""
        lines = self.cart_lines()
        if not lines or len(lines) != len(self.ledger.items) or {i for _, i in lines} != set(range(len(self.ledger.items))):
            return False
        if self.over_money():
            return False
        return all(p in self.index.opened and self.check(p, full=False)["ok"] for p, _ in lines)

    def _order_cart_as_is(self, held: bool = False) -> None:
        """Order the cart as it stands: the lines that can no longer be bought leave first, the request's before-order
        steps run while steps allow, then the order goes. With `held`, a cart holding a buyable line for every item
        goes even when a check speaks against a line or the total."""
        self._drop_gone()
        stands = self._every_item_held() if held else self._whole_cart_stands()
        if not stands or self.steps_left() < 1 or not self._guard("cart as is"):
            return
        if self.steps_left() >= 3 + self.messages_owed() and not self.easing(8.0):
            self.checklist("before_order")
        self.finish("buy")


    def _search_more(self, idx: int) -> bool:
        """One more search for an item: the next page of a search that has more rows, else new words (the item's
        name, then with a needed value, then its store label). Built on the last search's own arguments."""
        find = self.names.get("find")
        if not find or self.steps_left() <= 3:
            return False
        params = self.schemas[find].get("parameters") or {}
        props = params.get("properties") or {}
        page = self.find_args.get("page")
        if page in props:
            pages = sorted(self.page_next.items(), key=lambda kv: idx not in self.page_items.get(kv[0], set()))
            for words, (args, after) in pages:
                del self.page_next[words]
                more = dict(args)
                more[page] = after
                if self._run(find, more, "NEXT PAGE BEFORE ENDING")[1]:
                    self.paged = True
                    return True
        last = next((json.loads(m["tool_call"]["function"]["arguments"]) for m in reversed(self.out)
                     if m.get("tool_call") and m["tool_call"]["function"]["name"] == find), None) or {}
        last = self._word_mode(find, last)
        slot = self.find_args.get("words") if self.find_args.get("words") in props else self.text_arg(find)
        if not slot:
            return False
        item = self.ledger.items[idx] if idx < len(self.ledger.items) else {"name": "item"}
        needs = [n for n in self.ledger.needs_for(item["name"]) if n["op"] in ("is", "has", "one of")]
        value = needs[0]["value"] if needs else ""
        value = value[0] if isinstance(value, list) and value else value
        for words in (item["name"], "%s %s" % (value, item["name"]), self.label_of(idx)):
            capped = self._cap_tool()
            args = {k: v for k, v in last.items() if k != page and not (capped and capped == (find, k))}
            args[slot] = str(words).strip()
            for k in params.get("required") or []:
                if k not in args and "enum" in (props.get(k) or {}):
                    args[k] = props[k]["enum"][0]
            key = _search_key(args, page)
            if not args[slot] or key in self.searches:
                continue
            issues: list[str] = []
            args = _fit(params, args, find, issues)
            if not issues and self._run(find, args, "SEARCHED AGAIN BEFORE ENDING")[1]:
                return True
        return False

    def _decline_audit(self) -> str:
        """Before ending: follow open ties, search again by the required values (each item once per state of the needs),
        read unread pages, open the likeliest rows. Names any listing found to fit."""
        said = []
        for tie in [r for r in self.ledger.refs if r["status"] == "open"]:
            if self.steps_left() > self.reserve() + 4 and not self.easing(30.0):
                self.follow(tie)                   # a tie not yet followed is no proof that nothing fits
        for idx, item in enumerate(self.ledger.items):
            key = (idx, self.ledger.version)
            fits = self.passed(idx) or self._solid_opened(idx)
            if not fits and key not in self.audit_walked and self.names.get("find") and self.steps_left() > 6:
                self.audit_walked.add(key)
                self._walk_item(idx, min(8, self.steps_left() - 6))
            fits = self.passed(idx) or self._solid_opened(idx)
            if not fits:
                unread = self.by_mention(self.unread(idx), idx)[:max(0, min(AUDIT_OPENS, self.steps_left() - 4))]
                self._open_all(unread, "OPENED BEFORE ENDING", every=False)
                fits = [p for p in unread if p in self.index.opened and self.solid(p)]
            if fits:
                said.append("%s fits the %s (price %s)" % (fits[0], item["name"], self.index.rows.get(fits[0], {}).get("price")))
        if said:
            return "do not end yet: " + " | ".join(said) + ". Put it in the cart; code orders once the cart fits."
        return ""

    def _passing(self, idx: int) -> list[str]:
        """Opened listings of an item with nothing against them (open cells allowed), fewest open cells first."""
        quick = [p for p in self.prefer(self.opened(idx)) if p not in self.gone and self.check(p, full=False)["ok"]]
        return sorted(quick, key=lambda p: len(self.check(p, full=False)["unclear"]))

    def _solid_opened(self, idx: int) -> list[str]:
        """Opened listings of an item that fit on solid evidence: every one is read by code (free), and those that
        pass with nothing left open count at once; else up to four of the rest, fewest open cells first, get a full
        check."""
        quick = [p for p in self.prefer(self.opened(idx)) if p not in self.gone and self.check(p, full=False)["ok"]]
        sure = [p for p in quick if not self.check(p, full=False)["unclear"]]
        if sure:
            return sure
        quick.sort(key=lambda p: len(self.check(p, full=False)["unclear"]))
        return [p for p in quick[:4] if self.solid(p)]

    def _decline_text(self) -> str:
        """A plain reason, naming no listing, as the decline rule asks."""
        misses: dict[str, int] = {}
        for (pid, v), r in self.checks.items():
            if v == self.ledger.version:
                for why in r.get("why") or []:
                    head = why.split(":")[0] if ":" in why else why
                    if not any(p in head for p in self.index.ids):
                        misses[head] = misses.get(head, 0) + 1
        text = PLAIN_DECLINE
        top = sorted(misses, key=lambda k: -misses[k])[:2]
        if top:
            text += " What the listings I read lacked: %s." % "; ".join(top)
        if self.ledger.decline_rule and not self.easing(10.0):
            got = self._read(LIGHT, DECLINE_WORDS, {"rule": self.ledger.decline_rule, "missing": top,
                                                    "needs": [_need_text(n) for n in self.ledger.needs]}, 300, first_cap=15.0)
            if isinstance(got.get("text"), str) and got["text"].strip():
                text = got["text"].strip()
        for pid in sorted(self.index.ids, key=len, reverse=True):
            if re.search(r"[A-Za-z]", pid) or len(pid) >= 5:       # a short bare number may be a price, not an id
                text = re.sub(r"(?<![\w-])%s(?![\w-])" % re.escape(pid), "", text)
        return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()

    # -------------------------------------------------------------- finish under pressure
    def pressed(self) -> bool:
        """Time to finish: few steps left, the hand-over time reached, or the pace so far (seconds per step since the
        setup) saying the finishing steps would not fit in the time left. A finish forced by time runs in a hurry:
        no optional model reads."""
        if self.steps_left() <= self.reserve():
            return True                            # the finish keeps steps for an order that comes back with a note
        if self._at_ceiling():
            return True                            # the money for this trip is spent: code finishes it from here
        done = self.steps - self.steps_at_start
        pace = (time.monotonic() - self.started) / done if done >= 2 else 0.0
        finishing = (3 + len(self.ledger.items) + self.messages_owed()) * pace + 10.0
        if self.remaining() < SECONDS * HAND_OVER + finishing or self.remaining() < SECONDS * HAND_OVER:
            self.hurry = True
            return True
        return False

    def finish_now(self) -> None:
        """Steps or time are short, or nothing is left to try: order a cart that fits; else end through the decline path,
        which buys the closest cart first. Whatever happens the trip gets an order or a decline while a step is left."""
        self._talk_first()
        self._order_what_fits()
        if not self.done and not self.stuck and self.steps < self.step_cap and not self.may_decline():
            self.out_of_moves = True
        for _ in range(GUARD_LIMIT + 1):
            if self.done or self.stuck or self.steps >= self.step_cap or not self.names.get("none") or not self.may_decline():
                break
            said = self.decline()
            if self.done or not said.startswith("do not end yet"):
                break
            self._order_what_fits()               # the audit found a listing that fits
        self._last_call()

    def _closest(self, idx: int) -> str | None:
        """The opened listing of an item that misses the fewest needs (then leaves the fewest open, then costs least), among
        those of its own kind that can still be bought."""
        def cost(p: str) -> tuple:
            r = self.check(p, full=False)
            price = self.index.rows.get(p, {}).get("price")
            return (len(r.get("why") or []), len(r.get("unclear") or []),
                    price if isinstance(price, (int, float)) else float("inf"))
        return min(self._buyable(idx), key=cost, default=None)

    def _buyable(self, idx: int) -> list[str]:
        """Opened listings of an item that can be bought for it: on hand, of its kind, within its limits."""
        caps = [c for c in (self.unit_cap(idx), self.line_cap(idx) if self.ledger.several() else None) if c is not None]
        def within(p: str) -> bool:
            price = self.index.rows.get(p, {}).get("price")
            return not caps or isinstance(price, (int, float)) and price <= min(caps) + 1e-9
        return [p for p in self.opened(idx) if p not in self.gone and within(p)
                and self.index.rows.get(p, {}).get("in_stock") is not False and self._right_kind(p, idx)]


    def _order_what_fits(self, loose: bool = False, closest: bool = False) -> None:
        """Order what fits: per item keep one cart line that fits and take out the rest; fill each item without one
        with its best opened listing (settling every need first; with `loose`, any that fits; with `closest`, the closest
        that can be bought); then order. A partial cart is never ordered."""
        faults = self.order_faults
        for _ in range(3):
            if self.done or self.stuck or self.steps >= self.step_cap or not self.names.get("add"):
                return
            full = not self.easing(8.0)
            while self.reread and self.steps_left() > 3 and not self.easing(10.0):
                self._reread_one()                 # a note or a fault spoke of a line: its record decides first
            self._drop_gone()
            if self.names.get("cart") and self.cart_viewed != self.cart_mark and self.basket_touched \
                    and self.steps_left() >= 3:
                self._run(self.names["cart"], dict(self.fixed.get("cart", {})), "CART LOOKED AT BEFORE FINISHING", 1500)
            keep: dict[int, str] = {}
            for pid, idx in self.cart_lines():
                if idx not in keep and pid in self.index.opened and self.check(pid, full=full)["ok"]:
                    keep[idx] = pid
            for pid, idx in self.cart_lines():
                if keep.get(idx) != pid and self.steps_left() > 2:
                    args = self._remove_args(pid)
                    if args:
                        self._run(self.names["add"], args, "TAKEN OUT BEFORE ORDERING")
            for idx in range(len(self.ledger.items)):
                if idx in keep or self.steps_left() < 2:
                    continue
                if not self._passing(idx) and self.steps_left() > 5 + (len(self.ledger.items) - len(keep)):
                    # two reads only while the add, the order and an answer to a note still fit after them
                    self._open_all(self.by_mention(self.unread(idx), idx)[:2], "READ TO FILL A LINE", every=False)
                pick = self._best_for(idx, full, loose) or (self._closest(idx) if closest else None)
                if pick:
                    self._before_add(pick)
                if self.steps_left() < 2:              # the add and the order must still fit
                    break
                args = self._add_args(pick, self.ledger.items[idx]["qty"]) if pick else None
                if args:
                    self._run(self.names["add"], args, "PUT IN THE CART TO FINISH")
                    if pick in self.basket:
                        keep[idx] = pick
            if len(keep) < len(self.ledger.items) or self.steps_left() < 1:
                return                             # part of a basket is never the order: no line, no order
            if self.names.get("cart") and self.cart_viewed != self.cart_mark and self.steps_left() >= 3:
                self._run(self.names["cart"], dict(self.fixed.get("cart", {})), "CART LOOKED AT BEFORE ORDERING", 1500)
            self._restore(list(keep.values()))
            self._drop_gone()
            if self.missing_items() or self.steps_left() < 1:
                return                             # an item lost its line: no partial order
            if self.over_money():
                capped = self._fit_total(self.cart_lines(), float(self.ledger.budgets["money_cap"])) and not self.over_money()
                if not capped and not (closest and self._closest_within_limit()):
                    return                         # over the money limit with no cheaper fit: no order
            if self.steps_left() >= 3 + self.messages_owed() and not self.easing(8.0):
                self.checklist("before_order")
            if not closest and self.steps_left() >= 5 and self._final_look():
                continue                           # a line was swapped or found wrong: the next round orders what stands
            before = (self.cart_mark, self.ledger.version, len(self.gone))
            self.finish("buy")
            turned_down = self.order_faults > faults   # the store would not take it: a line is gone, the next round swaps it
            if self.done or not (self.refused_last or turned_down) or (closest and not turned_down):
                return
            if not turned_down and (self.cart_mark, self.ledger.version, len(self.gone)) == before:
                return                             # the same cart under the same ledger would only go out again unchanged
            faults = self.order_faults

    def _best_for(self, idx: int, full: bool, loose: bool) -> str | None:
        quick = self._passing(idx)
        share = self.line_cap(idx) if self.ledger.several() else None
        if share is not None:
            within = [p for p in quick if isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                      and self.index.rows[p]["price"] <= share + 1e-9]
            quick = within or sorted(quick, key=lambda p: self.index.rows.get(p, {}).get("price") or 0)
        pick = next((p for p in quick[:3] if self.solid(p, full=full)), None)
        return pick or (next((p for p in quick if self.loose_ok(p)), None) if loose else None)

    def loose_ok(self, pid: str) -> bool:
        """A listing with open cells may still be ordered, unless an open cell is the need the stated rule for
        "nothing fits" names (that rule is followed)."""
        rule = set(_tokens(self.ledger.decline_rule or ""))
        if not rule:
            return True
        def named(n: dict[str, Any]) -> bool:
            words = _tokens(" ".join(n["value"]) if isinstance(n["value"], list) else n["value"])
            return set(_tokens(n["property"])) <= rule or bool(words) and set(words) <= rule
        # a general "nothing fits" rule does not turn an open cell into a decline: ordering is right far more often
        cells = self.check(pid, full=False).get("cells") or []
        return not any(c[1] in ("unclear", "clash") and named(c[0]) for c in cells)

    # -------------------------------------------------------------- status
    def status(self) -> str:
        left = self.steps_left()
        head = "STATUS step %d/%d (%d left; the last %d are for finishing)" % (self.steps, self.step_cap, left, self.reserve())
        if self.names.get("speak"):
            head += " | messages sent %d" % self.talk_sent
        head += " | about %ds" % max(0, self.remaining() - SECONDS * HAND_OVER)
        lines = [head, self.ledger.summary()]
        v = self.ledger.version
        for idx, item in enumerate(self.ledger.items[:4]):
            shown = []
            for pid in self.prefer(self.opened(idx))[:6]:
                r = self.checks.get((pid, v)) or self.check(pid, full=False)
                row = self.index.rows.get(pid, {})
                state = ("fits" if r.get("full") else "looks right, not checked yet") if r["ok"] else \
                    "does not fit: " + "; ".join(r["why"][:2])
                shown.append("%s | %s | %s" % (pid, row.get("price"), state))
            if shown:
                lines.append("OPENED%s (id | price | check): %s" % (" for the " + item["name"] if self.ledger.several() else "",
                                                                     " ; ".join(shown)))
            unread = self.unread(idx)
            if self.ledger.objective.get("kind") == "cheapest":
                unread = sorted(unread, key=lambda p: self.index.rows.get(p, {}).get("price") or 0)
            if unread:
                lines.append("NOT OPENED YET%s (id | price | title): %s" % (
                    " for the " + item["name"] if self.ledger.several() else "",
                    " ; ".join("%s | %s | %s" % (p, self.index.rows.get(p, {}).get("price"),
                                                 str(self.index.rows.get(p, {}).get("title") or "")[:50]) for p in unread[:6])))
        if self.basket:
            lines.append("CART: " + ", ".join("%s x%d" % (p, self.line_qty.get(p, 1)) for p in self.basket))
        if self.paging:
            lines.append("SEARCHES WITH MORE ROWS: " + " ; ".join(list(self.paging.values())[-3:]))
        if left <= self.reserve():
            lines.append("FINISH NOW: put the listing that fits in the cart (code orders), or end without buying.")
        elif not self.basket and self.steps >= self.step_cap // 2:
            lines.append("Half the steps are gone and the cart is empty: put the listing that fits best so far in the cart "
                         "now; searching on is only worth it for a need no opened listing meets.")
        elif self.question_due():
            lines.append("Questions to the shopper go out next; keep opening listings of the item meanwhile.")
        return "\n".join(lines)

    # -------------------------------------------------------------- tools offered
    def offered(self) -> list[dict[str, Any]]:
        keep = set(self.schemas)
        speak = self.names.get("speak")
        if speak:
            keep.discard(speak)
        if self.steps_left() <= self.reserve():
            jobs = {self.names.get(j) for j in ("add", "cart", "buy", "none")} | set(self.readers)
            if speak and self.messages_owed() and self.messages_left() > 0:
                jobs.add(speak)
                keep.add(speak)
            subset = keep & jobs
            if subset:
                keep = subset
        return [t for t in self.tool_list if t.get("function", t)["name"] in keep]

    # -------------------------------------------------------------- one proposed call
    def screen(self, name: str, raw: Any, allowed: set[str], batch: list[str | None]) -> tuple[dict | None, str]:
        if name not in self.schemas:
            return None, "no tool is called %r; usable now: %s" % (name, ", ".join(sorted(allowed)))
        if name not in allowed:
            if self.names.get("speak") == name:
                return None, "messages to the shopper are composed and sent for you"
            return None, "too few steps remain for that tool; usable now: %s" % ", ".join(sorted(allowed))
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            args = _object_in(raw)
        else:
            args = {}
        if args is None:
            return None, "the arguments are not a readable JSON object"
        issues: list[str] = []
        args = _fit(self.schemas[name].get("parameters") or {}, args, name, issues)
        if issues:
            return None, "the arguments do not fit the tool: " + "; ".join(issues[:6])
        job = self.role(name, args)
        if job == "find":
            args = self._own_page(args)
        if job in ("buy", "none") and len(batch) > 1 and self._guard("alone"):
            return None, "send the order, or the ending, on its own after reading the other results"
        if job not in ("add", "speak", "buy", "none") and name + json.dumps(args, sort_keys=True) in self.last_looks \
                and self._guard("repeat"):
            return None, "an identical call was made a moment ago (see its result); change it or move on"
        if job == "find":
            page = self.find_args.get("page")
            if _search_key(args, page) in self.empty_searches \
                    and self._guard("empty again"):
                return None, "this very search brought no rows earlier in the trip; use other words or another page"
            streak = self._search_streak()
            if streak >= SEARCH_STREAK and any(self.unread(n) for n in range(len(self.ledger.items))) \
                    and self.steps_left() > self.reserve() + 3 and self._guard("open first"):
                # searching on and on while rows sit unread spends the steps the questions and the cheapest sweep
                # need; the rows found so far are read before another search goes out
                return None, "%d searches in a row and their rows are still unread; open the promising rows " \
                    "first (the STATUS names unread rows), then search again if none fits" % streak
        if job == "open" and self.free_reads < FREE_READS:
            pid = str(args.get(self.text_arg(name)) or "")
            if (pid, name) in self.read_records and self.read_by.get(pid, {}).get(name, -1) >= self.ledger.version:
                self.free_reads += 1
                return None, "%s was read with %s at the current ledger version; that record again: %s" % (
                    pid, name, self.read_records[(pid, name)])
        if job == "add" and not self._is_removal(name, args) and self.asking_ahead() \
                and (not self.plan_failed or self._guard("questions first")):
            # while planned questions are owed the add always waits; the guard's cap applies to the code ranking only
            return None, "questions to the shopper go out first; decide again after their replies"
        if job in ("buy", "none") and self.question_due() and self._guard("questions first"):
            return None, "questions to the shopper go out first; decide again after their replies"
        if job == "add":
            why = self._gate_add(name, args)
            if why:
                return None, why
        if job == "buy":
            return None, "DONE:" + self.commit()
        if job == "none":
            return None, "DONE:" + self.decline()
        return args, ""

    def _search_streak(self) -> int:
        """How many searches the driver sent since it last read a listing (code's own calls count as reads too)."""
        n = 0
        for m in reversed(self.out):
            if not m.get("tool_call"):
                continue
            name = m["tool_call"]["function"]["name"]
            job = self.role(name, _object_in(m["tool_call"]["function"].get("arguments")) or {})
            if job == "find":
                n += 1
            elif job in ("open", "compare"):
                break
        return n

    def _gate_add(self, name: str, args: dict[str, Any]) -> str:
        """Before a listing goes in the cart: it is opened when it never was, checked against the ledger, and held
        (guard) when it does not fit; the quantity is set to the item's; the request's before-add steps run."""
        props = (self.schemas[name].get("parameters") or {}).get("properties") or {}
        if self._is_removal(name, args):
            q_slot = self._qty_slot(props)
            pids = self._add_ids(name, args)
            if not self._sets_cart(name) and q_slot and len(pids) == 1 and isinstance(args.get(q_slot), int) \
                    and 0 < args[q_slot] < self.units_held(pids[0]):
                args[q_slot] = self.units_held(pids[0])  # the store takes out only the units named
            return ""                                  # taking a line out
        q_slot = self._qty_slot(props)
        again = [p for p in self._add_ids(name, args) if p in self.basket and p in self.held
                 and self.held[p] == self.ledger.items[self.item_of(p)]["qty"]] if self.ledger.items else []
        if again and not self._sets_cart(name) and self._guard("held already"):
            # this store adds the units sent to the units held: sending the line again doubles it
            return "%s is already in the cart at %d unit(s); to change the units, take it out first and then add it " \
                "at the units wanted" % (again[0], self.held[again[0]])
        pids = [p for p in self._add_ids(name, args) if p not in self.basket or self._sets_cart(name)]
        for pid in pids[:3]:
            if pid not in self.index.opened and self.names.get("open") and self.steps_left() > 3:
                self._open_all([pid], "OPENED BEFORE THE CART", every=False)
            if pid not in self.index.opened:
                if self._guard("unopened"):
                    return "%s was never opened; open it first" % pid
                continue
            r = self.check(pid, moment="add")
            if not r["ok"] and self._guard("add fit"):
                swap = self.backup(r["item"], {pid} | set(self.basket))
                return "%s does not fit (%s); %s, or repeat the call if you are sure" % (
                    pid, "; ".join(r["why"][:3]), "%s fits instead" % swap if swap else "look further")
            if r["ok"] and r["unclear"] and self.event_version and not self.fired.get("replacement"):
                swap = self.rechoose(r["item"], {pid} | set(self.basket))
                if swap and swap != pid and self.solid(swap):
                    self.fired["replacement"] = 1
                    return "after the change %s leaves %s unsettled, while %s settles every need; take %s, or repeat " \
                        "the call if you are sure" % (pid, "; ".join(r["unclear"][:2]), swap, swap)
            if r["ok"] and r["unclear"] and not self.unclear_hold:
                better = [p for p in self.label_pids(r["item"]) if p != pid and p not in self.gone
                          and p not in self.basket and self._settles(p, r["unclear"])]
                if not better and self.steps_left() > self.reserve() + 3 and not self.easing(30.0):
                    # rows not opened yet whose own words carry every unclear need's value are read first
                    item = self.ledger.items[r["item"]]["name"] if r["item"] < len(self.ledger.items) else ""
                    stem = lambda w: w[:-3] + "y" if w.endswith("ies") else w[:-1] if w.endswith("s") and not w.endswith("ss") else w
                    wanted = []                    # per need: the word sets any one of which the row must carry
                    for n in self.ledger.needs_for(item):
                        if _need_text(n) in r["unclear"] and n["op"] in ("is", "one of"):
                            values = n["value"] if isinstance(n["value"], list) else [n["value"]]
                            sets = [{stem(w) for w in _tokens(_dump(v)) if len(w) > 1} for v in values]
                            if all(sets):
                                wanted.append(sets)
                    def names_all(p: str) -> bool:
                        said = {stem(w) for w in _tokens(self.index.text_of.get(p, ""))}
                        return all(any(x <= said for x in sets) for sets in wanted)
                    rows = [p for p in self.unread(r["item"]) if p != pid and wanted and names_all(p)][:2]
                    if rows:
                        self._open_all(rows, "OPENED: ROWS THAT NAME THE UNSETTLED NEEDS", every=False)
                        better = [p for p in rows if p not in self.gone and self._settles(p, r["unclear"])]
                if better:
                    self.unclear_hold = True
                    return "%s leaves %s unsettled, while the opened %s meets %s; take %s, or repeat the call if you " \
                        "are sure" % (pid, "; ".join(r["unclear"][:2]), better[0], "it" if len(r["unclear"]) == 1
                                      else "them", better[0])
        total = self.ledger.budgets.get("money_cap")
        if pids and total and self.ledger.several() and self.fired.get("share", 0) < GUARD_TRIP_LIMIT:
            pid = pids[-1]
            idx = self.item_of(pid)
            price = self.index.rows.get(pid, {}).get("price")
            share = self.line_cap(idx)
            if isinstance(price, (int, float)) and share is not None and price > share + 1e-9:
                fits = [p for p in self.prefer(self.opened(idx)) if p != pid and p not in self.gone
                        and isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                        and self.index.rows[p]["price"] <= share + 1e-9 and self.replaces(p, pid)]
                self.fired["share"] = self.fired.get("share", 0) + 1
                return "%s at %s would take the order over its limit of %s (about %s a unit is left for the %s); %s" % (
                    pid, price, total, round(share, 2), self.ledger.items[idx]["name"],
                    "%s fits within it" % fits[0] if fits else "find a cheaper listing that fits")
        if q_slot and not self._sets_cart(name) and pids:
            qty = self.ledger.items[self.item_of(pids[0])]["qty"]
            if args.get(q_slot) != qty:
                args[q_slot] = qty
        new = [p for p in pids if p not in self.basket]
        if new and any(s["moment"] == "before_add" for s in self.ledger.steps):
            self.checklist("before_add", new[0])
        return ""

    # -------------------------------------------------------------- one model turn
    def _carry_out(self, message: dict[str, Any], calls: list[dict[str, Any]], allowed: set[str]) -> None:
        """Run a turn's proposed calls in order; each gets one tool message, and the last one carries the STATUS."""
        proposed = []
        self.turn_no += 1
        calls = [c for c in calls if isinstance(c, dict)]
        for n, call in enumerate(calls):
            fn = call.setdefault("function", {})
            if not isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.dumps(fn.get("arguments") or {})
            call["id"] = "t%d_%d" % (self.turn_no, n)     # our own ids: unique across turns, safe for every provider
            proposed.append((call["id"], fn.get("name", ""), fn["arguments"]))
        self.chat.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        batch = [self.role(name, _object_in(raw) or {}) for _, name, raw in proposed]
        mark = len(self.chat)
        replies = [{"role": "tool", "tool_call_id": cid, "content": self._carry_one(name, raw, allowed, batch)}
                   for cid, name, raw in proposed]
        if replies:
            replies[-1]["content"] += "\n" + self.status()
        aside = self.chat[mark:]                 # lines added while the calls ran go after their replies
        del self.chat[mark:]
        self.chat.extend(replies)
        self.chat.extend(aside)

    def _carry_one(self, name: str, raw: Any, allowed: set[str], batch: list[str | None]) -> str:
        if self.done:
            return "skipped: the trip is already over"
        args, why = self.screen(name, raw, allowed, batch)
        if args is None and why.startswith("DONE:"):
            return "NOT SENT as proposed; code took over: " + why[5:]
        if args is None:
            return "NOT SENT, no step spent. " + why
        if self.steps_left() <= 1 and self.role(name, args) not in ("buy", "none"):
            return "NOT SENT, no step spent. The last step stays free for finishing."
        mate = self._reading_mate(name, args)
        if mate:
            pair = self._pair_args([args[self.text_arg(name)], mate])
            observation, sent = self._send(*pair) if pair else (None, False)
            if sent:
                notes = self.absorb(pair[0], pair[1], observation)
                if self._took_pair(pair[0], [args[self.text_arg(name)], mate], observation, reader=name):
                    return "\n".join([_dump(observation), "(read together with %s, the next unread listing for the "
                                      "same item, in this one step)" % mate] + (["note: " + " ".join(notes)] if notes else []))
                if self.done:
                    return _dump(observation)
        observation, sent = self._send(name, args)
        if not sent:
            return _dump(observation)
        notes = self.absorb(name, args, observation)
        pieces = [_dump(observation)]
        if self.done:
            pieces.append("(the trip is over)")
        if notes:
            pieces.append("note: " + " ".join(notes))
        if self.role(name, args) == "find" and not self.done and self.ledger.objective.get("kind") == "cheapest":
            pieces += self._page_on(name, args, observation)
        return "\n".join(pieces)

    def _reading_mate(self, name: str, args: dict[str, Any]) -> str | None:
        """For a reader call on one listing: the next unread listing for the same item, to read along in one step."""
        slot = self.text_arg(name)
        pid = args.get(slot) if slot else None
        if self.role(name, args) != "open" or not isinstance(pid, str) or pid not in self.index.ids \
                or self.pair_reads is False or name not in self.page_keys or self.steps_left() <= self.reserve() + 2:
            return None
        idx = self.item_of(pid)
        return next((p for p in self.by_mention(self.unread(idx), idx) if p != pid), None)

    def _page_on(self, name: str, args: dict[str, Any], observation: Any) -> list[str]:
        """Under a cheapest rule a search that reports more rows gets its next page fetched at once (a few per trip),
        so cheaper listings beyond the first page are in view."""
        if self.pages_left <= 0 or self.steps_left() <= self.reserve() + 4 or self.easing(40.0):
            return []
        more = self._later_args(name, args, observation)
        if more is None:
            return []
        observation, sent = self._send(name, more)
        if not sent:
            return []
        self.pages_left -= 1
        notes = self.absorb(name, more, observation)
        return ["NEXT PAGE (fetched for you, %s=%s): %s%s" % (self.find_args.get("page"), more[self.find_args.get("page")],
                                                             _dump(observation), ("\nnote: " + " ".join(notes)) if notes else "")]

    # -------------------------------------------------------------- the trip
    def _can_go(self) -> bool:
        return not (self.done or self.stuck or self.steps >= self.step_cap or self.remaining() < SECONDS * HAND_OVER)

    def _lean(self, budget: int = 90000) -> list[dict[str, Any]]:
        """The chat for the driver: the rules, the plan and the latest turns up to a size; every STATUS line carries
        what was found before. When the kept turns outgrow the budget, the oldest LEAN_CUT share of them is dropped
        at once, so the same prefix (and its cache) holds for many steps between cuts."""
        head, tail = self.chat[:2], self.chat[2:]
        def size(start: int) -> int:
            return sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in tail[start:])
        cut = min(self.lean_cut, max(0, len(tail) - 1))
        while cut < len(tail) - 1 and size(cut) > budget:
            cut += max(1, int(LEAN_CUT * (len(tail) - cut)))
        cut = min(cut, max(0, len(tail) - 1))
        while cut > 0 and tail[cut].get("role") == "tool":
            cut -= 1                                   # a tool reply never opens the kept turns without its call
        self.lean_cut = keep = cut
        if keep == 0:
            return self.chat
        plan = dict(head[-1])
        plan["content"] = str(plan.get("content") or "") + "\n\n(%d earlier turns are left out here; the STATUS lines " \
            "carry what they found.)" % keep
        return head[:-1] + [plan] + tail[keep:]

    def take_over(self) -> None:
        """The driver gave no call three times: code moves the trip on itself. It puts a fitting opened listing in the
        cart (code then orders), else opens unread rows of the item's label, else searches again; with none of these
        left, the trip goes to its finish."""
        self.takeovers += 1
        if self.asking_ahead():                    # questions go out before code puts anything in the cart
            due = self.question_due()
            if due:
                self.ask(*due)
                return
        for idx in range(len(self.ledger.items)):
            if any(i == idx for _, i in self.cart_lines()):
                continue
            quick = [p for p in self.prefer(self.opened(idx)) if p not in self.gone and self.check(p, full=False)["ok"]]
            pick = next((p for p in quick[:2] if self.solid(p)), None)
            args = self._add_args(pick, self.ledger.items[idx]["qty"]) if pick and self.names.get("add") else None
            if args and not self._gate_add(self.names["add"], args):
                self._run(self.names["add"], args, "PUT IN THE CART FOR THE DRIVER")
                return
            unread = self.by_mention(self.unread(idx), idx)[:2]
            if unread and self.steps_left() > 4:
                self._open_all(unread, "OPENED FOR THE DRIVER", every=False)
                return
            if self._search_more(idx):
                return
        self.out_of_moves = True

    def go(self) -> list[dict[str, Any]]:
        if self.done:
            return self.out
        try:
            self.setup()
        except Exception as exc:  # a setup that breaks still leaves a trip to drive: shape roles and a plain chat
            self.out.append({"role": "assistant", "content": "stopped: %s" % _where(exc)})
            if not self.names.get("buy"):
                self._roles_by_shape()
            if not self.chat:
                self.chat = [{"role": "system", "content": DRIVER_GUIDE},
                             {"role": "user", "content": "THE SHOPPER'S REQUEST:\n%s" % self.request}]
        self.strikes, self.turns, errors = 0, 0, 0
        while self._can_go() and not self.out_of_moves:
            try:
                if self._turn():
                    break
            except Exception as exc:  # one failing part must not end the trip: count it, go on, finish after three
                errors += 1
                self.out.append({"role": "assistant", "content": "stopped: %s" % _where(exc)})
                self.chat.append({"role": "user", "content": "Something went wrong on our side; carry on.\n" + self.status()})
                if errors >= 3:
                    self.out_of_moves = True       # the finish may then decline (through the audit) if nothing fits
                    break
        if not self.done and not self.stuck:
            try:
                self.finish_now()
            except Exception as exc:  # the finish itself broke: one plain order or ending, nothing clever
                self.out.append({"role": "assistant", "content": "stopped: %s" % _where(exc)})
                self._plain_finish()
        return self.out

    def _stated_audit(self) -> bool:
        """Needs the first reading of the request may have left out: a store property whose name and one of whose
        values the request both spells, with no need, tie or "does not matter" on it, goes to one light reading.
        True when needs were added (a new ledger version)."""
        def stems(text: Any) -> set[str]:
            return {w.rstrip("s") for w in _tokens(text) if len(w.rstrip("s")) >= 2}
        said = stems(self.request)
        named = stems(" ".join(it["name"] for it in self.ledger.items))
        taken = [stems(x) for x in [n["property"] for n in self.ledger.needs] + list(self.ledger.no_preference)
                 + [r["property"] for r in self.ledger.refs]]
        roles = {_fold(x) for x in self.role_names()}
        found: dict[str, dict[str, set[str]]] = {}          # item name -> property -> values the request spells
        for n, item in enumerate(self.ledger.items):
            for p in sorted(self.label_pids(n)):
                for k, v in self.fields_of(p).items():
                    prop = stems(k)
                    if not prop or not prop <= said or prop <= named or _fold(k) in roles \
                            or any(t and (t <= prop or prop <= t) for t in taken):
                        continue
                    for piece in re.split(r"[,;/|]", str(v)):
                        words = stems(piece)
                        if words and words <= said and not words <= named and not _figures(piece):
                            found.setdefault(item["name"], {}).setdefault(_fold(k), set()).add(_fold(piece))
        if not found:
            return False
        shown = {name: {k: sorted(v)[:6] for k, v in sorted(props.items())[:6]} for name, props in found.items()}
        got = self._read(LIGHT, STATED_AUDIT, {"request": self.request, "items": [it["name"] for it in self.ledger.items],
                                               "properties": shown}, 300, first_cap=15.0)
        led, added = self.ledger, False
        for raw in got.get("needs") if isinstance(got.get("needs"), list) else []:
            need = _need_of(raw, led.items, "audit", led.version + 1)
            if not need:
                continue
            values = need["value"] if isinstance(need["value"], list) else [need["value"]]
            spelled = shown.get(need["item"], {}).get(_fold(need["property"]), [])
            if spelled and all(_fold(x) in spelled for x in values) \
                    and _need_key(need) not in {_need_key(n) for n in led.needs}:
                led.needs.append(need)
                added = True
        if added:
            led.version += 1
        return added

    def _plain_finish(self) -> None:
        """A last finish with no reading at all: order a cart that holds anything, else end without buying."""
        try:
            job = "buy" if self.basket and self.names.get("buy") else "none"
            args = self.closing_args(job, "Placing the order." if job == "buy" else PLAIN_DECLINE)
            if args is not None and not self.done and self.steps < self.step_cap:
                self._send(self.names[job], args)
        except Exception:
            pass

    def _turn(self) -> bool:
        """One pass of the loop: True when the trip should go to its finish."""
        self.turns += 1
        if self.turns > 3 * self.step_cap + 10:
            self.out_of_moves = True
            return True
        if self.strikes >= 3:
            self.strikes = 0
            self.take_over()
            return False
        if self.pressed():
            return True
        if getattr(self, "note_queue", None):
            self._take_cart_notes()
        if self.ledger.version != self.version_seen:
            self._after_change()
            return False
        if not self.stated_audited and not self.basket and not self.easing(30.0) \
                and sum(1 for p in self.index.opened if len(self.fields_of(p)) >= 3) >= 2:
            self.stated_audited = True
            if self._stated_audit():
                return False
        if self.talk_stage():
            return False
        tie = next((r for r in self.ledger.refs if r["status"] == "open"), None)
        if tie and self._tie_room():
            self.follow(tie)
            return False
        if self.reread and self.steps_left() > 4 and not self.easing(20.0):
            self._reread_one()
            return False
        if self._kit_fit():
            return False
        if self._hunt_revealed():
            return False
        walk_due = getattr(self, "_walk_due", None)
        if walk_due and walk_due():
            self._low_walk()
            return False
        if self.commit_due():
            said = self.commit()
            if not self.done:
                self.chat.append({"role": "user", "content": "ORDERING: " + said + "\n" + self.status()})
            return False
        tools = self.offered()
        message = self._ask(DRIVER, self._lean(), 3000, tools=tools, cache=True, caps=WAITS)
        calls = list((message or {}).get("tool_calls") or [])
        if calls:
            self.strikes = 0
            self._carry_out(message, calls, {t.get("function", t)["name"] for t in tools})
            return False
        self.strikes += 1
        if message is not None:
            self.chat.append({"role": "assistant", "content": message.get("content") or ""})
            self.chat.append({"role": "user", "content": "Reply with a tool call.\n" + self.status()})
        return False

    def _init_m_finish(self) -> None:
        if Ledger.summary is not _low_summary:
            _low_summary.__wrapped__ = Ledger.summary
            Ledger.summary = _low_summary
        self.walks_done: set[str] = set()           # needs (ledger needs state) a lowest-price walk already ran for
        self.note_queue: list[list[str]] = []       # note lines from cart calls, read before the next move
        self.note_memo: dict[str, list[str]] = {}   # note words -> kinds (one reading each)
        self.last_note: list[str] = []              # the note lines of the finish just sent
        self.needs_seen = 0                         # needs in the ledger at the last change handled
        self.audit_walked: set[tuple[int, int]] = set()   # (item, ledger version) searched again before ending

    def _walk_key(self) -> str:
        return json.dumps([self._needs_of_item(n) for n in range(len(self.ledger.items))]) + str(self.ledger.objective.get("kind"))

    def _needs_of_item(self, idx: int) -> list[str]:
        return sorted(_need_text(n) for n in self.ledger.needs_for(self.ledger.items[idx]["name"]))

    def _low_room(self) -> int:
        """Steps a walk may use: all there are, less the finish, the messages still owed, opening the cheapest rows and
        putting them in the cart."""
        n = len(self.ledger.items)
        keep = self.reserve() + self.messages_owed() + n * (LOW_OPENS + 1)
        return self.steps_left() - keep

    def _walk_due(self) -> bool:
        """Due once per state of the needs, after the questions to the shopper are planned and sent, while steps remain."""
        if self.ledger.objective.get("kind") == "best" or not self.names.get("find") or not self.ledger.items:
            return False
        if self._walk_key() in self.walks_done or not self.index.seen_order or self.easing(40.0) or self._low_room() < 2:
            return False
        if self.can_ask() and (self.question_due() is not None or (not self.ready_to_plan() and not self.basket)
                               or (self.plan_key is None and not self.plan_failed)):
            return False
        return True

    def _low_walk(self) -> None:
        """Search for the lowest-priced listings by code: for each item from its required values to plain words, then the
        values that split a long result, every page within the room, then the cheapest rows found opened."""
        self.walks_done.add(self._walk_key())
        items = len(self.ledger.items)
        plain = self.ledger.objective.get("kind") != "cheapest"
        room = self._low_room()
        if plain:
            room = min(room, LOW_PLAIN_STEPS * items)
        for idx in range(items):
            share = room // (items - idx)
            used = self._walk_item(idx, share) if share >= 1 else 0
            room -= used
        for idx in range(items):
            if self.steps_left() <= self.reserve() + items or self.easing(25.0):
                break
            self._open_cheapest(idx, LOW_OPENS)

    def _noun_of(self, idx: int) -> str:
        """The item's plainest word for a search: in the shopper's spelling where its store label agrees, else the label's longest word."""
        item = self.ledger.items[idx]
        others = {w for n, it in enumerate(self.ledger.items) if n != idx for w in _tokens(it["name"])}
        own = [w for w in _tokens(item["name"]) if w not in others]
        label = {w.rstrip("s") for w in _tokens(self.label_of(idx))}
        words = [w for w in own if w.rstrip("s") in label]     # the shopper's own spelling, when the label says the same
        words = words or [w for w in _tokens(self.label_of(idx)) if w not in others] or own or _tokens(item["name"])
        return words[-1] if words else ""       # the head of an English product name is its last word

    def _required_values(self, idx: int) -> list[str]:
        """The values the item's needs ask for, as plain search words, each once."""
        out: list[str] = []
        for need in self.ledger.needs_for(self.ledger.items[idx]["name"]):
            if need["op"] not in ("is", "has", "one of"):
                continue
            value = need["value"][0] if isinstance(need["value"], list) and need["value"] else need["value"]
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and _tokens(value):
                text = " ".join(_tokens(value))
                # a bare number finds nothing useful; a long phrase matches nothing when every word must be present
                if text not in out and not text.replace(" ", "").isdigit() and len(text.split()) <= 3:
                    out.append(text)
        return out

    def _ladder_of(self, idx: int) -> list[str]:
        """Search words from the narrowest to the plainest: the required values with the item's word, then fewer of them."""
        values = self._required_values(idx)
        noun = self._noun_of(idx)
        name = self.ledger.items[idx]["name"]
        ladder: list[str] = []
        if values:
            ladder.append(" ".join(values + [name]))
            ladder.append(" ".join(values + [noun]))
            for value in values:
                ladder += ["%s %s" % (value, name), "%s %s" % (value, noun), value]
        ladder += [name, noun, self.label_of(idx)]
        seen: list[str] = []
        for words in ladder:
            words = " ".join(str(words).split()).strip()[:60]
            if words and words.lower() not in seen:
                seen.append(words.lower())
        return seen

    def _rows_of_item(self, idx: int) -> set[str]:
        """Listing rows seen so far that belong to the item."""
        label = self.label_of(idx)
        return {p for p in self.index.seen_order if self.item_of(p) == idx
                and (not label or not self.index.rows.get(p, {}).get("category")
                     or _alike(self.index.rows[p]["category"], label))}

    def _walk_item(self, idx: int, room: int) -> int:
        """One item's walk within `room` steps. Returns the steps used."""
        find = self.names.get("find")
        if not find or room < 1:
            return 0
        used = stall = 0
        page = self.find_args.get("page")
        ladder = self._ladder_of(idx)
        for words in ladder:
            if used >= room or stall >= LOW_STALL or self.done or self.easing(30.0):
                break
            args = self._search_args(words)
            if args is None or _search_key(args, page) in self.searches:
                continue
            spent, fresh = self._walk_pages(find, args, room - used, idx)
            used += spent
            stall = 0 if fresh else stall + 1
        tried = 0
        for value in self._splitting_values(idx):
            if used >= room or stall >= LOW_STALL or tried >= SPLIT_PICKS or self.done or self.easing(30.0):
                break
            words = ("%s %s" % (value, self._noun_of(idx))).strip()[:60]
            args = self._search_args(words)
            if args is None or _search_key(args, page) in self.searches:
                continue
            tried += 1
            spent, fresh = self._walk_pages(find, args, room - used, idx)
            used += spent
            stall = 0 if fresh else stall + 1
        return used + self._rest_of_pages(idx, room - used)

    def _walk_pages(self, find: str, args: dict[str, Any], room: int, idx: int) -> tuple[int, bool]:
        """One search, page by page within `room` and LOW_PAGES; two pages in a row with no new row of the item end it.
        Returns the steps used and whether a new row of the item showed up."""
        steps, dry, any_new = 0, 0, False
        limit = min(room, LOW_PAGES)
        while steps < limit and not self.done and self.steps_left() > self.reserve() and not self.easing(30.0):
            before = len(self._rows_of_item(idx))
            observation, sent = self._run(find, args, "A SEARCH FOR THE LOWEST PRICE (rows kept for you)", 300)
            if not sent:
                break
            steps += 1
            fresh = len(self._rows_of_item(idx)) > before
            any_new = any_new or fresh
            dry = 0 if fresh else dry + 1
            if _failed(observation) or dry >= 2 or self.done:
                break
            more = self._later_args(find, args, observation)
            if more is None:
                break
            args = more
        return steps, any_new

    def _rest_of_pages(self, idx: int, room: int) -> int:
        """Unread pages of the searches that served this item, while the room lasts: the store's own next-page values."""
        find = self.names.get("find")
        page = self.find_args.get("page")
        if not find or not page or room < 1:
            return 0
        used = 0
        for words, (args, after) in sorted(self.page_next.items(), key=lambda kv: idx not in self.page_items.get(kv[0], set())):
            if used >= room or self.done or self.easing(30.0):
                break
            if idx not in self.page_items.get(words, set()):
                continue
            del self.page_next[words]
            more = dict(args)
            more[page] = after
            issues: list[str] = []
            more = _fit(self.schemas[find].get("parameters") or {}, more, find, issues)
            if issues:
                continue
            spent, _ = self._walk_pages(find, more, room - used, idx)
            used += spent
        return used

    def _splitting_values(self, idx: int) -> list[str]:
        """Words that split a long result of this item: short values its listings show, carried by some of them but not
        most, then title words of the same kind. Each is searched together with the item's plain word."""
        rows = sorted(self._rows_of_item(idx))
        if len(rows) < 2:
            return []
        have = set(_tokens(" ".join(self._required_values(idx) + [self._noun_of(idx), self.ledger.items[idx]["name"]])))
        ceiling = max(2, int(len(rows) * 0.6))
        shown: dict[str, int] = {}
        worded: dict[str, int] = {}
        roles = {_fold(x) for x in self.role_names()}
        for pid in rows:
            for key, value in {**self.fields_of(pid), **self.index.sheet_of.get(pid, {})}.items():
                text = " ".join(_tokens(value))
                if not text or _fold(key) in roles or len(text.split()) > 2 or len(text) > 24 or set(text.split()) & have:
                    continue
                shown[text] = shown.get(text, 0) + 1
            for word in set(_tokens(self.index.rows.get(pid, {}).get("title") or "")):
                if len(word) >= 3 and word not in have:
                    worded[word] = worded.get(word, 0) + 1
        fair = lambda n: n <= ceiling                 # a value most rows carry does not split them
        values = sorted((v for v, n in shown.items() if fair(n)), key=lambda v: (-shown[v], v))
        if self.ledger.objective.get("kind") != "cheapest":
            return values
        makers = {w for pid in rows for w in _tokens(self.index.rows.get(pid, {}).get("seller") or "")}
        words = sorted((w for w, n in worded.items() if 2 <= n and fair(n) and w not in makers), key=lambda w: (-worded[w], w))
        return list(dict.fromkeys(values + words))

    def _later_args(self, name: str, args: dict[str, Any], observation: Any) -> dict[str, Any] | None:
        """The arguments of the search's next page, from the store's own page value; a store whose results carry no such
        value but whose search takes a page number is simply asked for the next number."""
        page_arg = self.find_args.get("page")
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        if not page_arg or page_arg not in props:
            return None
        inner = _d(_notes_beside(observation)[0])
        next_key = self.index.roles.get("next_page", "")
        after = inner.get(next_key) if next_key else None
        kind = (props.get(page_arg) or {}).get("type")
        if after in (None, False, ""):
            if next_key or kind != "integer" or not _ids_in(inner, self.index.ids):
                return None
            current = args.get(page_arg) if isinstance(args.get(page_arg), int) else 1
            more = dict(args, **{page_arg: current + 1})
        elif kind == "integer":
            current = args.get(page_arg) if isinstance(args.get(page_arg), int) else 1
            more = dict(args, **{page_arg: after if isinstance(after, int) and not isinstance(after, bool)
                                 and after != args.get(page_arg) else current + 1})
        else:
            more = dict(args, **{page_arg: after})
        issues: list[str] = []
        more = _fit(params, more, name, issues)
        return None if issues else more

    def _open_cheapest(self, idx: int, limit: int) -> None:
        """Open the cheapest unread rows of an item that can be bought, of its own kind and within its limit."""
        cap = self.unit_cap(idx)
        def price(p: str) -> Any:
            return self.index.rows.get(p, {}).get("price")
        wanted = set(_tokens(" ".join(self._required_values(idx))))
        def shows(p: str) -> int:
            return len(wanted & set(_tokens(json.dumps(self.index.rows.get(p, {}), default=str))))
        # a row that shows the required values comes before a merely cheap one: a cheap wrong listing is worth nothing
        rows = sorted((p for p in self.unread(idx) if isinstance(price(p), (int, float)) and self._right_kind(p, idx)
                       and (not cap or price(p) <= cap + 1e-9)), key=lambda p: (-shows(p), price(p)))
        self._open_all(rows[:limit], "CHEAPEST ROWS READ", every=False)

    def _right_kind(self, pid: str, idx: int) -> bool:
        """The listing is of the item's kind: under its store label when both are known, else sharing a word with the item's
        name in its own title or text."""
        row = self.index.rows.get(pid, {})
        label = self.label_of(idx)
        if label and row.get("category"):
            return _alike(row["category"], label)
        def stems(text: Any) -> set[str]:
            return {w.rstrip("s") for w in _tokens(text) if len(w) > 2}
        wanted = stems(self.ledger.items[idx]["name"])
        return not wanted or bool(wanted & stems("%s %s %s" % (row.get("title") or "", row.get("category") or "",
                                                                self.index.text_of.get(pid, "")[:400])))

    def _note_kinds(self, lines: list[str]) -> tuple[list[str], str]:
        """What a store's note says about the cart: its kinds and the listing it names (read once per note)."""
        key = " | ".join(lines)
        if key in self.note_memo:
            cached = self.note_memo[key]
            return cached[0].split(",") if cached[0] else [], cached[1]
        if self._at_ceiling():
            return [], ""                          # no reading left: the records and the cart decide
        got = self._read(LIGHT, CHECKOUT_NOTE, {"said": lines, "cart": self.cart_summary()}, 400, first_cap=15.0)
        kinds = [str(k) for k in (got.get("kinds") if isinstance(got.get("kinds"), list) else [])]
        named = str(got.get("listing") or "").strip()
        named = named if named in self.index.ids else ""
        self.note_memo[key] = [",".join(kinds), named]
        return kinds, named

    def _take_cart_notes(self) -> bool:
        """Notes that came with cart calls, read before the next move: a sold-out note takes its line out of the plan at
        once. True when the ledger moved."""
        queued, self.note_queue = self.note_queue, []
        moved = False
        for lines in queued:
            if self.done or self.easing(5.0):
                break
            kinds, named = self._note_kinds(lines)
            if "sold_out" in kinds and self._mark_sold_out(named):
                moved = True
        return moved

    def _mark_sold_out(self, named: str) -> bool:
        """A note says a listing can no longer be bought: the one it names, else a line whose record says so, else the line
        added last. True when a line is newly out."""
        held = [p for p in self.basket if p not in self.gone]
        if named and named not in held:
            return False                       # a listing outside the cart: nothing in the cart changes
        target = named if named in held else next((p for p in held if self.index.rows.get(p, {}).get("in_stock") is False),
                                                  held[-1] if held else "")
        if not target:
            return False
        self.gone.add(target)
        self.ledger.version += 1
        self.event_version = self.ledger.version
        return True

    def _replan_cart(self) -> bool:
        """Lines were lost to a note: the rest stays, and each item without a line gets the cheapest listing already read
        that fits (read further when none does, then the closest that can be bought). True when the cart can go out again."""
        self._drop_gone()
        for idx in range(len(self.ledger.items)):
            if any(i == idx for _, i in self.cart_lines()) or self.steps_left() < 3:
                continue
            taken = self.gone | set(self.basket)
            pick = self.backup(idx, taken) or self.rechoose(idx, taken)
            if not pick and self.steps_left() > self.reserve() + 2:
                self._walk_item(idx, min(4, self.steps_left() - self.reserve() - 2))
                self._open_cheapest(idx, 2)
                pick = self.backup(idx, taken) or self.rechoose(idx, taken)
            pick = pick or self._closest(idx)
            if pick:
                self._before_add(pick)
                args = self._add_args(pick, self.ledger.items[idx]["qty"]) if self.steps_left() >= 2 else None
                if args:
                    self._run(self.names["add"], args, "PUT IN THE CART AFTER THE NOTE")
        return not self.missing_items() and not self.over_money() and self.steps_left() >= 1

    def _cheaper_after_rise(self) -> None:
        """A price rose: a listing already read that fits every need for less than the raised price takes the line's place
        (under a stated cheapest rule, or when no way of choosing is named)."""
        for pid, idx in self.cart_lines():
            price = self.index.rows.get(pid, {}).get("price")
            if not isinstance(price, (int, float)) or self.steps_left() < 4:
                continue
            below = [p for p in self.passed(idx, {q for q, _ in self.cart_lines()})
                     if isinstance(self.index.rows.get(p, {}).get("price"), (int, float))
                     and self.index.rows[p]["price"] < price - 1e-9 and self.replaces(p, pid)]
            if below:
                best = min(below, key=lambda p: self.index.rows[p]["price"])
                self._swap(pid, best, idx)

    def _fill_hole(self, schema: dict[str, Any], value: Any, text: str, depth: int = 0) -> Any:
        """`value` completed to what its schema requires, from the schema alone: required text gets the words, fixed choices
        their first value, numbers their floor, arrays their minimum of filled entries."""
        if depth > 5:
            return value
        if "enum" in schema:
            return value if value in schema["enum"] else schema["enum"][0]
        kind = schema.get("type")
        if kind == "object" or "properties" in schema:
            out = dict(value) if isinstance(value, dict) else {}
            props = schema.get("properties") or {}
            for k in schema.get("required") or []:
                if out.get(k) in (None, "", [], {}):
                    out[k] = self._fill_hole(props.get(k) or {}, out.get(k), text, depth + 1)
            return out
        if kind == "array":
            items = list(value) if isinstance(value, list) else []
            sub = schema.get("items") or {}
            while len(items) < int(schema.get("minItems") or 0):
                items.append(self._fill_hole(sub, None, text, depth + 1))
            return items
        if kind in ("integer", "number"):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            floor = schema.get("default", schema.get("minimum", 1))
            return floor if isinstance(floor, (int, float)) and not isinstance(floor, bool) else 1
        if kind == "boolean":
            return value if isinstance(value, bool) else True
        if isinstance(value, str) and value:
            return value
        words = text or "ok"
        limit = schema.get("maxLength")
        return words[:limit] if isinstance(limit, int) and limit > 0 else words

    def _mend_finish(self, name: str, args: dict[str, Any], job: str, observation: Any) -> dict[str, Any] | None:
        """A finish the store could not take as sent: first every required field and every optional text field the store's
        words name, from the schema; else one corrected reading. None when nothing new can be sent."""
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        said = set(_tokens(_dump(observation, 600)))
        text = "Placing the order for the cart." if job == "buy" else PLAIN_DECLINE
        fixed = self._fill_hole(params, dict(args), text)
        for k, sch in props.items():
            if k not in fixed and (sch or {}).get("type") == "string" and "enum" not in (sch or {}) \
                    and set(_tokens(k.replace("_", " "))) & said:
                fixed[k] = text
        issues: list[str] = []
        fixed = _fit(params, fixed, name, issues)
        if not issues and fixed != args and self.role(name, fixed) == job:
            return fixed
        move = "place the order" if job == "buy" else "end the trip without buying"
        got = self._read(LIGHT, RAW_FINISH % move, {"tool": self.schemas[name], "sent": args,
                                                     "result": _dump(observation, 1500)}, 400, first_cap=20.0)
        issues = []
        fixed = _fit(params, got, name, issues)
        if got and not issues and self.role(name, fixed) == job:
            return fixed
        return None

    def _talk_first(self) -> None:
        """What is still owed to the shopper goes out before a path that ends the trip, while money and steps allow."""
        owed = getattr(self, "talk_before_ending", None)
        if owed and not self._at_ceiling() and not self.hurry and not self.done:
            try:
                owed()
            except Exception as exc:  # questions must never cost the trip its ending
                self.out.append({"role": "assistant", "content": "stopped: %s" % _where(exc)})

    def _open_for_closest(self) -> None:
        """Each item with nothing of its kind opened yet gets its cheapest unread rows read, while steps allow."""
        for idx in range(len(self.ledger.items)):
            if self._buyable(idx) or self.steps_left() <= 2 + len(self.ledger.items):
                continue
            self._open_cheapest(idx, 2)

    def _closest_within_limit(self) -> bool:
        """Bring a closest cart under the money limit: the dearest line goes to the cheapest listing of its item that can be
        bought, while that helps. True when the cart ends within the limit."""
        cap = self.ledger.budgets.get("money_cap")
        if not cap:
            return True
        tries = 0
        while self.over_money() and tries < len(self.ledger.items) + 2 and self.steps_left() > 3:
            tries += 1
            def cost(p: str) -> float:
                return float(self.index.rows.get(p, {}).get("price") or 0) * self.line_qty.get(p, 1)
            lines = sorted(self.cart_lines(), key=lambda line: -cost(line[0]))
            moved = False
            for pid, idx in lines:
                cheap = sorted((p for p in self._buyable(idx) if p != pid
                                and float(self.index.rows.get(p, {}).get("price") or 0) * self.line_qty.get(pid, 1) < cost(pid)),
                               key=lambda p: self.index.rows[p].get("price") or 0)
                if cheap and self._swap(pid, cheap[0], idx):
                    moved = True
                    break
            if not moved:
                break
        return not self.over_money()

    def _last_call(self) -> None:
        """The trip still has no order and no decline: end it with the plainest call there is. A cart holding a line for
        every item, within the money limit, goes out; anything else ends without buying."""
        if self.done or self.stuck or self.steps >= self.step_cap:
            return
        if self.basket and self.names.get("buy") and self._every_item_held() and not self.over_money():
            self.finish("buy", "Placing the order for the cart.")
        if not self.done and not self.stuck and self.steps < self.step_cap and self.names.get("none"):
            self.finish("none", PLAIN_DECLINE)
        if not self.done and not self.stuck and self.steps < self.step_cap:
            self._plain_finish()

    def _at_ceiling(self) -> bool:
        """True once another call of the driver model would take this trip past its ceiling."""
        return self.spent + self._call_guess(DRIVER[0]) > TRIP_CEILING

    def _call_guess(self, model: str) -> float:
        """What one more call of a model is taken to cost: its own last call, or the dearest call so far when it made none."""
        return self.call_cost.get(model) or max([CALL_GUESS] + list(self.call_cost.values()))

    def _init_m_interview(self) -> None:
        self.heard_count: dict[int, int] = {}    # item -> needs or ties a reply has given
        self.sent_about: dict[int, int] = {}     # item -> messages that named it
        self.puzzled_items: set[int] = set()     # items whose last message was asked to be said again
        self.talk_first: list[int] = []          # items the plan found a story hint for, in its order
        self.open_asked: set[int] = set()        # items that had their one open question
        self.talk_now: list[str] = []            # the properties the message in flight names
        self.reply_read = False                  # the reply to the message in flight was read
        self.name_searches: dict[int, int] = {}  # item -> searches by its name that code made before planning
        self.case_marks: dict[tuple[str, int], str] = {}   # (listing, version) -> the cases its stored check was made under
        self.ledger.as_material = self._ledger_material

    def _ledger_material(self) -> dict[str, Any]:
        """The ledger as the readings see it, each need with the case it depends on."""
        out = Ledger.as_material(self.ledger)
        for shown, need in zip(out["needs"], self.ledger.needs):
            if need.get("when"):
                shown["when"] = need["when"]
        return out

    def _case_applies(self, pid: str, need: dict[str, Any]) -> bool | None:
        """Whether a need applies to a listing: True, False, or None while its case cannot be told yet (the other item's
        line is not in the cart, or the listing does not show the property)."""
        when = need.get("when")
        if not when:
            return True
        target = pid
        if when["item"] and _fold(when["item"]) != _fold(need["item"]):
            other = next((n for n, it in enumerate(self.ledger.items) if _fold(it["name"]) == _fold(when["item"])), None)
            target = next((p for p, i in self.cart_lines() if i == other), None) if other is not None else None
            if target is None:
                return None
        if _fold(when["property"]) == "price":
            shown = self.index.rows.get(target, {}).get("price")
        else:
            shown = _field(self.fields_of(target), when["property"])
        if shown is None:
            return None
        vocab = {_fold(x) for p in self.fields for x in re.split(r"[,;/|]", str(_field(self.fields_of(p), when["property"]) or "")) if _fold(x)}
        state = _cell(str(shown), when["op"], when["value"], vocab)[0]
        if state == "clash":                   # a value shown that differs from the one the case names, spelling aside
            wanted = when["value"] if isinstance(when["value"], list) else [when["value"]]
            state = "met" if any(_alike(str(shown), str(v)) for v in wanted) else "against"
        if state not in ("met", "against"):
            return None
        return (state == "met") == when["holds"]

    def _case_check(self, pid: str, full: bool = True, moment: str = "") -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The listing against the item's needs, with a need that holds only in some case counted only when that case applies.
        Returns the reading and the needs whose case cannot be told yet."""
        item = self.ledger.items[self.item_of(pid)]["name"] if self.ledger.items else ""
        cases = [n for n in self.ledger.needs_for(item) if n.get("when")]
        undecided: list[dict[str, Any]] = []
        if not cases:
            result = self._check(pid, full, moment)
        else:
            standing = {id(n): self._case_applies(pid, n) for n in cases}
            undecided = [n for n in cases if standing[id(n)] is None]
            # the case a need depends on can change with the cart, so a stored reading made under another case is not reused
            key = (pid, self.ledger.version)
            mark = json.dumps(sorted(str(standing[id(n)]) + _need_text(n) for n in cases))
            if self.case_marks.get(key) != mark:
                self.checks.pop(key, None)
                self.case_marks[key] = mark
            held = self.ledger.needs
            self.ledger.needs = [n for n in held if standing.get(id(n), True) is True]
            try:
                result = self._check(pid, full, moment)
            finally:
                self.ledger.needs = held
        return (result, undecided)

    def talk_owed(self) -> bool:
        """True while questions still have to go out before anything is committed."""
        return self.can_ask() and (self.plan_key is None or self.plan_failed or self.question_due() is not None)

    def talk_stage(self) -> bool:
        """One move of the questions owed before the cart, the order or an ending: read a listing, plan, or send one
        message. True when it moved."""
        if self.done or self.stuck or not self.can_ask():
            return False
        if self.prepare_questions():
            return True
        due = self.question_due()
        if due:
            self.ask(*due)
            return True
        return False

    def talk_before_ending(self) -> None:
        """The whole of what is still owed to the shopper and to the ties, run now (for a path about to end the trip)."""
        for _ in range(3 * self.step_cap):
            if not self.talk_stage():
                break
        while not self.done and self._tie_room():
            tie = next((r for r in self.ledger.refs if r["status"] == "open"), None)
            if tie is None:
                break
            self.follow(tie)

    @staticmethod
    def _several_at_once(values: list[str]) -> bool:
        """Most values name three or more options together."""
        parts = [len([x for x in re.split(r"\s*[,;/|]\s*", str(v)) if x]) for v in values]
        return sum(1 for n in parts if n >= 3) * 2 > len(parts)

    def _bare_properties(self, idx: int) -> list[str]:
        """Every property the item's opened listings show, most carried first: what is left when the table holds nothing."""
        counts: dict[str, int] = {}
        for p in self.sheeted(idx) or self.label_pids(idx) or self.opened(idx):
            for prop, _ in self._askable_fields(p):
                counts[prop] = counts.get(prop, 0) + 1
        return sorted(counts, key=lambda k: (-counts[k], k))

    def _name_search(self, n: int) -> bool:
        """A search code makes for an item no rows have come for: its whole name, then its last two words."""
        tries = self.name_searches.get(n, 0)
        words = self.ledger.items[n]["name"].split() if n < len(self.ledger.items) else []
        if tries >= 2 or not words or (tries == 1 and len(words) <= 2) or not self.names.get("find"):
            return False
        self.name_searches[n] = tries + 1
        self.searched_for.add(n)
        args = self._search_args(" ".join(words if tries == 0 else words[-2:]))
        if args is None:
            return False
        return self._run(self.names["find"], args, "SEARCHED FOR AN ITEM BEFORE ASKING")[1]

    def _likely_rows(self, idx: int, have: list[str]) -> list[str]:
        """Unread rows in stock most like the item when none fall under its label: by the words of its name in their text."""
        words = {w.rstrip("s") for w in _tokens(self.ledger.items[idx]["name"])}
        def overlap(p: str) -> int:
            row = self.index.rows.get(p, {})
            text = "%s %s %s" % (row.get("title") or "", row.get("category") or "", self.index.text_of.get(p, "")[:400])
            return len(words & {w.rstrip("s") for w in _tokens(text)})
        pool = [p for p in self.index.seen_order if p not in self.index.opened and p not in self.gone and p not in have
                and self.index.rows.get(p, {}).get("in_stock") is not False and (not self.ledger.several() or self.item_of(p) == idx)]
        return sorted(pool, key=lambda p: -overlap(p))

    def _item_number(self, said: Any) -> int | None:
        """The item a planning entry means: its 1-based number, else its name."""
        led = self.ledger
        try:
            n = int(str(said).strip())
            if 1 <= n <= len(led.items):
                return n - 1
        except ValueError:
            pass
        text = _fold(said)
        return next((n for n, it in enumerate(led.items)
                     if text and (_alike(it["name"], said) or text in _fold(it["name"]) or _fold(it["name"]) in text)),
                    None if led.several() else 0)

    @staticmethod
    def _key_in(table: dict[str, Any], prop: str) -> str | None:
        """The table key a planner's label names: exact, else one whose words contain or are contained in the label's."""
        words = set(_fold(prop).split())
        return next((k for k in table if _fold(k) == _fold(prop)), None) or \
            next((k for k in table if words and (words <= set(_fold(k).split()) or set(_fold(k).split()) <= words)), None)

    @staticmethod
    def _same_prop(a: str, b: str) -> bool:
        x, y = set(_fold(a).split()), set(_fold(b).split())
        return bool(x and y) and (x <= y or y <= x)

    def _after_reply(self, idx: int, listed: list[str], kind: str, heard: list[str]) -> None:
        """Move one item's list on after a reply. A need retires every property up to the one answered; anything else
        retires only the property named first. On a trip for one item, a list that is complained about or asked to be
        said again ends lists for the trip, the next property leading. With several items, a complaint retires nothing
        and the first one is asked again alone, and a request to say it again sends the item behind the others."""
        lone = not self.ledger.several()
        if kind in ("turned", "confused") and not heard and lone and len(listed) > 1:
            self.single_only = True            # this shopper takes one property at a time: stays so, the next one leads
            kind = "no_need"
        elif kind == "turned":
            self.single_only = True
            return
        reached = 0
        if heard:
            reached = max((n for n, p in enumerate(listed) if any(self._same_prop(p, q) for q in heard)), default=0)
            self.puzzled_items.discard(idx)
        elif kind == "confused":
            self.puzzled_items.add(idx)
        for prop in listed[:reached + 1]:
            if (idx, prop) not in self.asked:
                self.asked.append((idx, prop))
        gone = {_fold(p) for p in listed[:reached + 1]}
        self.queue = [(i, q) for i, q in self.queue if not (i == idx and _fold(q) in gone)]

    def _tie_room(self) -> bool:
        return not self.done and not self.stuck and self.steps_left() > self.reserve() + 3 and not self.easing(30.0)

    def _follow_one(self, ref: dict[str, Any], trail: dict[str, Any]) -> None:
        """One listing along a tie."""
        target = trail["target"]
        pid = self._find_pointed(target) or self._find_described(target)
        if pid and pid not in self.index.opened:
            self._open_all([pid], "READ FOR A TIED NEED", every=False)
        if not pid or pid not in self.index.opened:
            trail["misses"] = TIE_MISSES           # nothing to read: ending the walk beats burning steps
            ref["found"].append("no listing of %s was found" % (target.get("seller") or target.get("described") or "the description"))
            return
        self.chain_seen.add(pid)
        trail["hops"] += 1
        self._open_all([pid], "READ FOR A TIED NEED")
        records = self.index.records.get(pid, [])[-4:] + [v for (p, _), v in self.read_records.items() if p == pid][-2:]
        got = self._read(LIGHT, REF_READ, {"property": ref["property"], "also_called": trail["alias"] or None,
                                           "tie": trail["relation"], "described": target, "listing": pid,
                                           "records": records}, 500, first_cap=20.0)
        value = got.get("value")
        nxt = _d(got.get("points_to"))
        if not nxt.get("seller") and got.get("sends_on"):
            nxt["seller"] = self._seller_in(str(got["sends_on"]), nxt)
            gone = set(_tokens(nxt["seller"]))
            nxt["marks"] = [" ".join(w for w in _tokens(m) if w not in gone) for m in _texts(nxt.get("marks"))]
            nxt["marks"] = [m for m in nxt["marks"] if m]
        named = " ".join(_tokens(nxt.get("property")))
        if named and _fold(named) != _fold(ref["property"]):
            trail["alias"] = named
        onward = bool(nxt.get("seller") or nxt.get("marks") or got.get("sends_on"))
        if onward and (got.get("sends_on") or not value):
            value = None                           # the listing sends the property on: its own figure is not the one
            turn = str(got.get("turns") or "same").lower()
            if turn == "not":                      # an "anything but" along the way reverses what the tie asks
                trail["relation"] = "same" if trail["relation"] == "not" else "not"
            elif turn == "compatible" and trail["relation"] != "not":
                trail["relation"] = "compatible"
        elif got.get("pairing") and value is not None and _fold(value) == _fold(got.get("shown")) and trail["relation"] != "not":
            fixed = self._read(LIGHT, PAIR_APPLY, {"property": ref["property"], "shown": got.get("shown"),
                                                   "pairing": got.get("pairing")}, 200, first_cap=12.0)
            if isinstance(fixed.get("value"), str) and fixed["value"].strip():
                value = fixed["value"].strip()
        if isinstance(value, (str, int, float)) and str(value).strip() and not got.get("pairing") \
                and trail["relation"] != "compatible" and not self._grounded(pid, str(value), [ref["property"], trail["alias"]], records):
            value = None                           # a value this listing never states for the property is no reading
            if not onward:
                ref["found"].append("%s states no %s" % (pid, ref["property"]))
                trail["misses"] += 1
                return
        if isinstance(value, (str, int, float)) and str(value).strip():
            self._tie_resolved(ref, trail, pid, str(value).strip())
            return
        if onward:
            ref["found"].append("%s points on to %s" % (pid, nxt.get("seller") or got.get("sends_on")))
            trail["target"] = self._pointer(nxt, got.get("sends_on"))
            trail["misses"] = 0
            return
        ref["found"].append("%s says nothing plain about %s" % (pid, ref["property"]))
        trail["misses"] += 1

    def _tie_resolved(self, ref: dict[str, Any], trail: dict[str, Any], pid: str, value: str) -> None:
        op = "is not" if trail["relation"] == "not" else "is"
        self.ledger.needs.append({"item": ref["item"], "property": ref["property"], "op": op, "value": value,
                                  "source": "reference", "version": self.ledger.version + 1})
        self.ledger.version += 1
        ref["status"] = "resolved"
        self.chat.append({"role": "user", "content": "TIED NEED FOLLOWED: %s's listing %s gives %s %s %s\n%s" % (
            trail["target"].get("seller") or "the listing", pid, ref["property"], op, value, self.status())})

    def _tie_given_up(self, ref: dict[str, Any]) -> None:
        ref["status"] = "unresolved"
        self.chat.append({"role": "user", "content": "TIED NEED NOT SETTLED: %s must %s the listing described; found so "
                          "far: %s. No value is assumed for it.\n%s" % (
                              ref["property"], "differ from" if ref["relation"] == "not" else "go with",
                              "; ".join(ref["found"]) or "nothing", self.status())})

    def _find_described(self, target: dict[str, Any]) -> str | None:
        """A listing found by words alone, for a description with no seller to go by or a seller that came up empty: a
        light reading proposes searches, and rows sharing most words with the description are read."""
        about = " ".join([str(target.get("described") or ""), str(target.get("seller") or ""), str(target.get("category") or "")]
                         + _texts(target.get("marks"))).strip()
        wanted = set(_tokens(about))
        if not wanted or not self.names.get("find"):
            return None
        got = self._read(LIGHT, TIE_SEARCH, {"listing": about}, 200, first_cap=15.0)
        phrases = [str(q).strip() for q in got.get("searches") or [] if str(q).strip()][:3] if isinstance(got.get("searches"), list) else []
        def closest() -> list[str]:
            shared = [(len(wanted & set(_tokens(self.index.text_of.get(p, "")))), p) for p in self.index.ids
                      if p not in self.chain_seen]
            return [p for n, p in sorted(shared, key=lambda s: (-s[0], s[1])) if n >= 2]
        for words in (phrases or [about[:120]]) + [""]:
            if not self._tie_room():
                return None
            if words:
                args = self._search_args(words)
                if args is None or _search_key(args, self.find_args.get("page")) in self.searches:
                    continue
                self._run(self.names["find"], args, "SEARCHED FOR A TIED NEED")
            rows = [p for p in closest() if p not in self.chain_tried][:2]
            if rows:
                self.chain_tried.update(rows)
                self._open_all(rows, "READ FOR A TIED NEED", every=False)
                return next((p for p in rows if p in self.index.opened), None)
        return None

    def _first_choice_orders(self) -> None:
        name = self.names["buy"]
        params = self.schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        arg = next((k for k in list(params.get("required") or []) + list(props) if len((props.get(k) or {}).get("enum") or []) >= 2), None)
        if arg:
            self.fixed.setdefault("buy", {})[arg] = props[arg]["enum"][0]
            self.fixed.setdefault("none", {})[arg] = props[arg]["enum"][1]

    def _id_slot(self, name: str) -> str | None:
        """The argument of a cart call naming the listing: of the required free strings, the one with the smallest stated length (ids rarely state one)."""
        params = self.schemas.get(name, {}).get("parameters") or {}
        props = params.get("properties") or {}
        free = [k for k in params.get("required") or [] if (props.get(k) or {}).get("type") == "string" and "enum" not in (props.get(k) or {})]
        return min(free, key=lambda k: int((props.get(k) or {}).get("maxLength") or 0)) if free else self.text_arg(name)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    trip = None
    try:
        trip = Trip(problem_data)
        return trip.go()
    except Exception as exc:  # end the trip properly whatever went wrong
        if trip is None:
            return []
        try:
            trip.out.append({"role": "assistant", "content": "stopped: %s" % _where(exc)})
            trip.finish_now()
        except Exception:
            trip._plain_finish()
        return trip.out


__all__ = ["agent_main"]
