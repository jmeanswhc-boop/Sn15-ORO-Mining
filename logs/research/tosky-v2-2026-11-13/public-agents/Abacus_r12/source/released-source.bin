"""abacus: a budget-first shopping agent for ORO (SN15).

The organising idea is money.  One model call turns the shopper's request into a plan (products, quantities,
requirements, objective, budget words, message cap, decline rule, tool roles).  One cheap search per product then
shows what that kind of product costs; from the observed medians the agent infers the price unit scale and splits the
shared budget into per-product ceilings (an Allocation).  Every product gets a price Ladder: candidate listings sorted
by price ascending, opened from the bottom rung up, and the first rung whose properties literally satisfy every
requirement becomes the fit.  Code, not the model, has the last word on every requirement: the reader model only
quotes the property it relied on and code re-checks that quote against the listing.

Hidden needs are asked only once every product has a fit, one property per message, choosing a property on which the
fit and the rung above it disagree, so an answer can actually change the pick.  Checkout is one cart line per product,
a cart view against the shared budget, and an order; a refusal is parsed and the cart rebuilt, never re-sent unchanged.
An inference outage never produces a decline: after backoff and model rotation the ladders run on code alone.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter as Tally
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from src.agent.proxy_client import ProxyClient

# --------------------------------------------------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------------------------------------------------

PLANNER_MODELS = ("google/gemini-3.8-flash", "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4.1-flash")
READER_MODELS = ("google/gemini-3.8-flash", "anthropic/claude-haiku-5.5", "z-ai/glm-5.2")
TASK_SECONDS = 270.0          # the platform kills a task at 300 s; everything must be settled before this
CLOSING_SECONDS = 35.0        # kept free for checkout steps once exploration stops
CART_DEADLINE = 150.0         # by now a cart candidate must exist: exploration stops as soon as one does
ORDER_DEADLINE = 200.0        # by now the order sequence starts with the best code-ranked listing
CALL_PATIENCE = 20.0          # a model call never waits longer; a stall rotates to the next model at once
DEGRADED_PATIENCE = 12.0      # after two consecutive stalls every remaining job gets one short attempt
LEAN_SECONDS = 8.0            # above this average answer time, jobs code can do are left to code
FAULTS_TO_BACK = 2            # failed calls after which a model goes to the back of every later chain
STORE_PATIENCE = 30.0         # an environment call never waits longer on the wall clock
# Question ranking, measured on the 60 graded practice menus (tests/menus.py) and the qualifying and race replays:
# presence in quarters of the read listings, spellings, few values, title echo, exact presence, fit vs rung above.
MENU_ORDER = ("tier", "spellings", "fewvals", "echo", "presence", "differs")
MODEL_TOKENS = 1800           # reasoning families burn a small budget before answering; never go below 1500
COMPARE_BATCH = 4             # rungs read in one compare call
READER_BATCH = 6              # listings judged in one reader call
PAGE_LIMIT = 4                # pages per search phrase per product
OFFER_PROBE = 5               # rows priced through the offer tool before the first ladder sort
BEST_WINDOW = 0.05            # for "best": fits within 5 % of the cheapest fit compete on shopper words
QUESTION_DEFAULT = 3          # questions asked when the request states no message cap
SOFT_AFTER = 8                # right-kind reads after which a stated need nobody states stops vetoing
PROVISIONAL_EXTRA = 3         # rungs opened past a fit that rests on kindred keys only, hunting a firm one
SWEEP_ROUNDS = 3              # department sweeps under the current best price for 'cheapest'


# --------------------------------------------------------------------------------------------------------------------
# Wire: the only place that talks HTTP.  Tests swap `open_wire` for an in-memory world.
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class Packet:
    """What a POST came back with: parsed JSON or a description of the failure (never both)."""
    data: Optional[dict]
    error: Optional[dict]


def on_the_clock(post: Callable[[str, dict], Packet], path: str, body: dict, seconds: float) -> Packet:
    """One POST bounded on the wall clock.

    The client's timeout is per socket read: a provider that keeps the connection alive while it works never trips
    it.  The call runs in a daemon thread and is waited for at most `seconds`; a call that has not returned by then
    is abandoned (the thread finishes or dies with the process) and reported as a network stall.
    """
    box: dict[str, Packet] = {}

    def run() -> None:
        try:
            box["packet"] = post(path, body)
        except Exception as error:   # a client fault is a failed call, never the end of the episode
            box["packet"] = Packet(None, {"kind": "network", "status": None, "body": f"{type(error).__name__}: {error}"})

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(max(0.1, seconds))
    if "packet" in box:
        return box["packet"]
    return Packet(None, {"kind": "network", "status": None, "body": f"no answer within {seconds:.0f} s (wall clock)"})


def open_wire(timeout: float) -> Callable[[str, dict], Packet]:
    """Return a poster bound to a fresh ProxyClient with the given timeout; one retry is left to the client."""
    seconds = max(1, int(timeout))
    client = ProxyClient(timeout=seconds, max_retries=1)

    def post(path: str, body: dict) -> Packet:
        result = client.post_verbose(path, body)
        return Packet(result.data, result.error)

    return post


# --------------------------------------------------------------------------------------------------------------------
# Text helpers: small pure functions shared by the matcher, the field lens and the prompts.
# --------------------------------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def stem(token: str) -> str:
    """A deliberately shallow stemmer: plural and -es/-ies endings only, so 'sets' meets 'set' and 'dress' stays."""
    if token.isdigit() or len(token) < 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith(("xes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def grains(text: Any) -> list[str]:
    """Lower-cased stemmed tokens; digits are split from letters so '5v' and '5 V' agree."""
    return [stem(t) for t in _TOKEN.findall(str(text).lower())]


def facets(value: Any) -> list[str]:
    """Split a property value that lists several things ('cans, bottles', 'red/blue', 'a and b')."""
    pieces = re.split(r"[,;/|]|\band\b", str(value).lower())
    return [p.strip() for p in pieces if p.strip()]


def numbers_in(text: Any) -> list[float]:
    out = []
    for raw in _NUMBER.findall(str(text)):
        try:
            out.append(float(raw.replace(",", ".")))
        except ValueError:
            continue
    return out


def same_label(a: str, b: str) -> bool:
    """Two property names mean the same thing when their stemmed token sets are equal ('pack type' ~ 'pack_type')."""
    return set(grains(a)) == set(grains(b)) and bool(grains(a))


def label_overlap(a: str, b: str) -> int:
    """How many non-trivial stemmed tokens two names share."""
    return len({g for g in grains(a) if len(g) > 2} & {g for g in grains(b) if len(g) > 2})


CLOSERS = {"{": "}", "[": "]"}


def json_island(text: str) -> Optional[Any]:
    """Pull the first balanced JSON object or array out of free text, by bracket depth rather than regex."""
    if not isinstance(text, str):
        return None
    brace_at, bracket_at = text.find("{"), text.find("[")
    pairs = list(CLOSERS.items())
    if bracket_at != -1 and (brace_at == -1 or bracket_at < brace_at):
        pairs.reverse()
    for opener, closer in pairs:
        start = text.find(opener)
        while start != -1:
            depth, quoted, escaped = 0, False, False
            for pos in range(start, len(text)):
                ch = text[pos]
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
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            island = text[start:pos + 1]
                            return json.loads(island)
                        except ValueError:
                            break
            start = text.find(opener, start + 1)
    return None


def mended_json(text: str) -> Optional[Any]:
    """The outermost JSON object of a reply, repaired when the reply was cut off at max_tokens: the half-written
    last member is dropped and the open brackets are closed."""
    whole = json_island(text)
    if whole is not None or not isinstance(text, str) or "{" not in text:
        return whole
    body = text[text.index("{"):]
    cut = len(body)
    while cut > 0:
        cut -= 1
        if body[cut] not in ",{[":
            continue
        head = body[:cut + 1].rstrip(",")
        stack, quoted, escaped = [], False, False
        for ch in head:
            if quoted:
                escaped = (ch == "\\") and not escaped
                if ch == '"' and not escaped:
                    quoted = False
                continue
            if ch == '"':
                quoted = True
            elif ch in "{[":
                stack.append(CLOSERS[ch])
            elif ch in "}]" and stack:
                stack.pop()
        if quoted:
            continue
        try:
            return json.loads(head + "".join(reversed(stack)))
        except ValueError:
            continue
    return None


def provider_safe(schema: Any) -> Any:
    """A function schema every provider accepts: an object always has properties, an array always has items, and
    an empty required list is dropped (Gemini rejects all three)."""
    if isinstance(schema, list):
        return [provider_safe(e) for e in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: provider_safe(v) for k, v in schema.items()}
    if out.get("type") == "object" and not out.get("properties"):
        out["properties"] = {"note": {"type": "string"}}
    if out.get("type") == "array" and not isinstance(out.get("items"), dict):
        out["items"] = {"type": "string"}
    if out.get("required") == []:
        out.pop("required")
    return out


def mean_length(values: list) -> float:
    total = 0
    for value in values:
        total += len(value)
    return total / max(1, len(values))


def numbered_needs(needs: list) -> list[str]:
    lines = []
    for i, n in enumerate(needs):
        lines.append(f"{i}. {n.describe()}")
    return lines


def typed_keys(props: dict, *kinds: str) -> list[str]:
    """Parameter names whose JSON schema type is one of `kinds`, in schema order."""
    found = []
    for name in props:
        spec = props[name]
        if isinstance(spec, dict) and spec.get("type") in kinds:
            found.append(name)
    return found


def is_enum(spec: Any) -> bool:
    return isinstance(spec, dict) and bool(spec.get("enum"))


def text_values(mapping: dict) -> list[str]:
    """The string values of a mapping, in order."""
    out = []
    for value in mapping.values():
        if isinstance(value, str):
            out.append(value)
    return out


def as_list(value: Any) -> list:
    """A model may answer one item where a list was asked, or a dict of items: always a list back."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        inner = [v for v in value.values() if isinstance(v, list)]
        return inner[0] if len(inner) == 1 and len(value) == 1 else [value]
    if isinstance(value, str):
        return [piece.strip() for piece in re.split(r",|;", value) if piece.strip()]
    return [value]


def as_bool(value: Any) -> Optional[bool]:
    """true / "true" / "yes" / 1 -> True; false / "false" / "no" / 0 -> False; anything else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "y", "1"):
            return True
        if low in ("false", "no", "n", "0", "none", "null"):
            return False
    return None


def unwrapped(raw: Any, key: str) -> Any:
    """The object that carries `key`: models wrap answers ({"plan": {...}}, [{...}]) or return the bare list."""
    if isinstance(raw, dict):
        if key in raw:
            return raw
        for value in raw.values():
            if isinstance(value, dict) and key in value:
                return value
        if len(raw) == 1:
            only = next(iter(raw.values()))
            if isinstance(only, list):
                return {key: only}
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], dict) and key in raw[0]:
            return raw[0]
        return {key: raw}
    return raw


READER_CHARS = 8000          # listing text per reader call: batches shrink so no listing is ever cut mid-way


def reader_row(rung: Any) -> dict:
    """One listing as the reader sees it; property values are capped so one sprawling sheet cannot crowd out the rest."""
    props = {}
    for key, value in rung.listing.props.items():
        text = str(value)
        props[key] = text if len(text) <= 200 else text[:200] + " …"
    return {"id": rung.listing.ref, "category": rung.listing.category, "properties": props}


def reader_batches(rungs: list) -> list[list]:
    """Rungs grouped for reader calls: at most READER_BATCH per call and at most READER_CHARS of listing text."""
    batches: list[list] = []
    current: list = []
    size = 0
    for rung in rungs:
        length = len(json.dumps(reader_row(rung), ensure_ascii=False, default=str))
        if current and (len(current) >= READER_BATCH or size + length > READER_CHARS):
            batches.append(current)
            current, size = [], 0
        current.append(rung)
        size += length
    if current:
        batches.append(current)
    return batches


def squeeze(obj: Any, limit: int = 6000) -> str:
    """JSON for a prompt, cut at a limit so a verbose listing cannot flood the context."""
    text = json.dumps(obj, default=str, ensure_ascii=False)
    overflow = len(text) - limit
    if overflow > 0:
        text = text[:limit] + " …"
    return text


def as_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def number_in(value: Any) -> Optional[float]:
    """A number however it is written: 450, "450", "450 cents", "$4.50", "1,500"."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    found = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    return float(found.group(0).replace(",", "")) if found else None


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = numbers_in(value)
        return found[0] if len(found) == 1 else None
    return None


def median(values: Iterable[float]) -> Optional[float]:
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


# --------------------------------------------------------------------------------------------------------------------
# Needs and the literal matcher.  The grader reads PROPERTIES literally; so does this code.
# --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Need:
    """One requirement on one property: op is 'is', 'not', 'min' or 'max'."""
    prop: str
    value: str
    op: str = "is"
    soft: bool = False   # a wish (occasion, recipient, intended use): preferred, never a veto
    heard: bool = False  # disclosed by the shopper in answer to a question: always a hard requirement

    @staticmethod
    def from_raw(raw: Any) -> Optional["Need"]:
        if not isinstance(raw, dict):
            return None
        prop, value = str(raw.get("property") or "").strip(), str(raw.get("value") or "").strip()
        op = str(raw.get("op") or "is").strip().lower()
        if op not in ("is", "not", "min", "max"):
            op = "is"
        if not value:
            return None
        soft = bool(as_bool(raw.get("wish"))) or bool(re.search(r"occasion|recipient|purpose|\buse\b|event|wear for", prop.lower()))
        return Need(prop or "any", value, op, soft)

    def describe(self) -> str:
        return {"is": f"{self.prop} = {self.value}", "not": f"{self.prop} is not {self.value}",
                "min": f"{self.prop} at least {self.value}", "max": f"{self.prop} at most {self.value}"}[self.op]


def unfolded_needs(raw: Any) -> list[Need]:
    """One planner requirement as the needs it really states: 'none (sleeveless)' is sleeveless; 'wedge (not block
    heel, not stiletto)' is wedge and two exclusions; 'at least 1 ms' is a lower bound; 'not crew neck' excludes."""
    first = Need.from_raw(raw)
    if first is None:
        return []
    value, op, out = first.value, first.op, []
    inner = re.search(r"\(([^)]*)\)", value)
    if inner:
        value = (value[:inner.start()] + value[inner.end():]).strip(" ,")
        for piece in re.split(r",|;", inner.group(1)):
            piece = re.sub(r"^(?:or|and|e\.g\.?|i\.e\.?)\s+", "", piece.strip(), flags=re.I)
            if piece.lower().startswith("not "):
                out.append(Need(first.prop, piece[4:].strip(), "not", first.soft))
            elif piece and op == "is":
                if value.lower() in ("", "none", "yes", "no") or len(value) > 30:
                    value = piece   # the parenthesis names the value the word stood for
                else:
                    out.append(Need(first.prop, piece, "is", first.soft))   # an alternative on the same property
    low = value.lower()
    if op == "is" and low.startswith("not "):
        op, value = "not", value[4:].strip()
    bound = re.match(r"(at least|minimum of|min\.?|at most|maximum of|max\.?|up to|no more than)\s+(.+)", low)
    if op == "is" and bound:
        op = "min" if bound.group(1).startswith(("at least", "min")) else "max"
        value = value[bound.start(2):].strip()
    if value:
        out.insert(0, Need(first.prop, value, op, first.soft))
    return out


def unstatable(need: Need) -> bool:
    """Stock, price, budget, delivery and conversation limits are checked by code or not at all: no listing property
    states them, so as needs they would veto everything."""
    text = f"{need.prop} {need.value}".lower()
    return bool(re.search(r"stock|avail|on hand|inventory", text) or re.search(r"price|budget|cost|deliver|shipping|messag|step", text))


def value_carries(required: str, actual: Any, ignore: Iterable[str] = ()) -> bool:
    """Does an actual property value literally carry the required value?

    Each facet of the actual value ('pu leather', 'cans') must contain every stemmed token of the requirement.
    Numbers are compared as numbers, with unit tokens required to appear next to them.  'loose' never equals
    'baggy' and 'pu leather' does carry 'leather', which is exactly what the grader does.  Tokens in `ignore`
    (the product kind, the property's own name) may be absent: 'trench coat' under 'jacket coat style' is 'trench'.
    """
    want = grains(required)
    if not want:
        return False
    want_numbers = [float(w) for w in want if re.fullmatch(r"\d+(?:\.\d+)?", w)]
    want_words = set()
    for w in want:
        if not w.replace(".", "", 1).isdigit():
            want_words.add(w)
    trimmed = want_words - set(ignore)
    if trimmed:
        want_words = trimmed
    candidates = facets(actual) + [str(actual).lower()]
    for piece in candidates:
        have = grains(piece)
        have_set = set(have)
        if want_numbers:
            have_numbers = {float(h) for h in have if re.fullmatch(r"\d+(?:\.\d+)?", h)}
            if all(n in have_numbers for n in want_numbers) and want_words <= have_set:
                return True
        elif want_words <= have_set or all(any(near_word(w, h) for h in have_set) for w in want_words):
            return True
    return False


def near_word(a: str, b: str) -> bool:
    """A misspelt word is the same word: five letters or more, same first three, at most one letter changed,
    added or dropped ('tunik' is 'tunic', 'wegde' is 'wedge'); short words and numbers must match exactly."""
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5 or a[:2] != b[:2] or a.isdigit() or b.isdigit() or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return len(diffs) == 2 and diffs[1] == diffs[0] + 1 and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]]
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long_)):
        if long_[:i] + long_[i + 1:] == short:
            return True
    return False


def bound_holds(need: Need, actual: Any) -> bool:
    """'min'/'max' needs compare the first number of the value with the first number of the requirement."""
    limit, found = numbers_in(need.value), numbers_in(actual)
    if not limit or not found:
        return False
    return found[0] >= limit[0] if need.op == "min" else found[0] <= limit[0]


def own_key(name: str, prop: str, kind: str = "") -> bool:
    """The listing key is the shopper's own word for the property: the same words, or the same once the
    product's own words are set aside ('clock_shape' is 'shape' for a clock; 'jeans_fit_type' is not 'fit')."""
    if same_label(name, prop):
        return True
    product = attribute_words(kind, set())
    words = attribute_words(name, product)
    return bool(words) and words == attribute_words(prop, product)


def need_verdict(need: Need, props: dict[str, Any], hint: Optional[str] = None, kind: str = "",
                 category: str = "") -> tuple[Optional[bool], bool]:
    """Decide one need against a listing: (held, firm).

    held is True/False/None (nothing on the listing speaks to the property).  firm is True when the evidence is the
    kind the grader uses without doubt: a property whose NAME is the shopper's property ('fit', not
    'jeans_fit_type'), or the category phrase itself.  A kindred key that holds still counts, but only weakly.
    Negated needs pass when no property anywhere carries the forbidden value; a yes/true need is met by any
    property value that carries the property's own name ('color changing' inside 'clock function').
    """
    stated = {k: v for k, v in props.items() if v not in (None, "")}
    if need.op == "not":
        # the property's own key speaks for it when the listing has one ('material: pu' beside 'material_filter: pu
        # leather'); otherwise kindred keys; with neither, any value carrying the excluded word
        own = [v for k, v in stated.items() if same_label(k, need.prop)]
        kindred = [v for k, v in stated.items() if label_overlap(k, need.prop) > 0]
        scope = own or kindred or list(stated.values())
        return not any(value_carries(need.value, v) for v in scope), True
    if need.op == "is" and category and value_carries(need.value, category):
        return True, True
    if need.op == "is" and need.value.lower() in ("yes", "true"):
        return (True if any(value_carries(need.prop, v) for v in stated.values()) else None), True
    if need.prop == "any" and need.op == "is":
        # a value with no property named: seller option strings ('beige / eu:39') are variants on offer, not facts
        facts_only = [v for v in stated.values() if not variant_like(str(v))]
        return any(value_carries(need.value, v) for v in facts_only), False
    names = [k for k in stated if label_overlap(k, need.prop) > 0]
    if not names and hint:
        names = [k for k in stated if same_label(k, hint)]
    if not names and need.op == "is":
        # the shopper's word for the property may not be the world's ('fabric' for 'material', 'colour' for
        # 'color'): a key whose value carries the wanted value is then the evidence, weakly
        carriers = [k for k, v in stated.items() if value_carries(need.value, v)]
        return (True, False) if carriers else (None, False)
    if not names:
        return None, False
    kind_tokens = grains(kind)
    head = set(kind_tokens[-1:])          # 'coat' in 'trench coat': the noun a value never repeats
    firm_hold = weak_hold = False
    for name in names:
        actual = stated[name]
        key_tokens = set(grains(name))
        ok = bound_holds(need, actual) if need.op in ("min", "max") else any(
            value_carries(need.value, actual, ignore)
            for ignore in (key_tokens, key_tokens | head, key_tokens | set(kind_tokens)))
        if ok and own_key(name, need.prop, kind):
            firm_hold = True
        elif ok:
            weak_hold = True
    if firm_hold:
        return True, True
    if weak_hold:
        return True, False
    return False, any(own_key(n, need.prop, kind) for n in names)


def need_holds(need: Need, props: dict[str, Any], hint: Optional[str] = None) -> Optional[bool]:
    """The verdict alone, for callers that do not rank evidence."""
    return need_verdict(need, props, hint)[0]


def vocabulary_match(text: str, vocabulary: list[str]) -> Optional[str]:
    """A listing value the text literally carries, else a distinctive word of one ('ryzen' from 'amd ryzen 5')."""
    for value in vocabulary:
        if value_carries(value, text):
            return value
    spoken = set(grains(text))
    counts = Tally(g for value in vocabulary for g in set(grains(value)))
    for value in vocabulary:
        for g in grains(value):
            if len(g) > 3 and g in spoken and counts[g] < max(2, len(vocabulary)):
                return g
    return None


_NON_ANSWERS = (
    re.compile(r"any of them|any road|nowt to choose|no real opinion|blank page|choose for"),
    re.compile(r"whichever|all the same|suit me|not picky|no preference|don.t mind"),
    re.compile(r"up to you|open on that|never fretted|pick as you|lose sleep|never given"),
    re.compile(r"no particular|no opinion|indifferent|surprise me|open ground|no lean"),
)


_DIALECT_CUES = (
    r"thank you to find me (.+?)(?:,| as |$)", r"settled in my mind as (.+?)(?:,| hard |$)",
    r"only accept as (.+?)(?:\.|,|$)", r"asking after (.+?) and no other", r"stick to (.+?)(?:,| for |$)",
    r"namely (.+?)(?:\.|,|$)", r"holding out for (.+?)(?:\.|,|$)", r"anything besides (.+?)(?:\.|,|$)",
    r"will serve as [\w' ]+? but (.+?)(?:,|\.|$)", r"only (.+?) will (?:do|rise)", r"nowt else but (.+?)(?:,|\.|$)",
    r"stays fixed: (.+?)(?:,|\.|$)", r"stick to (.+?) when it comes to", r"locked to (.+?)(?:,| because|$)",
    r"ruled out everything but (.+?) under", r"closed, with (.+?) and nothing", r"will do for [\w' ]+? except (.+?)(?:,|\.|$)",
    r"sorted as (.+?)(?:,| and |$)", r"should be (.+?)(?:,| so |$)", r"^where [\w' ]+? is concerned, it'?s (.+?) or",
    r"coming in as (.+?)(?:;|,|$)", r"only accept (.+?)(?:,| the |$)",
)


def dialect_value(reply: str) -> Optional[str]:
    """The value a dialect answer fixes, read from its cue phrase ('I'd thank you to find me satin brass')."""
    text = reply.strip().rstrip(".")
    for cue in _DIALECT_CUES:
        found = re.search(cue, text, re.I)
        if found:
            value = found.group(1).strip(" .,'\"")
            if value and len(value) <= 40 and not value.lower().startswith(("the very same", "anything bar", "what")):
                return value
    return None


def bound_in_answer(value: str) -> tuple[str, str]:
    """'at least 2 drawers' -> ('min', '2 drawers'); 'at most 128 gb' -> ('max', '128 gb'); otherwise ('is', value)."""
    found = re.match(r"\s*at (least|most)\s+(.+)", value, re.I)
    if found:
        op = "min" if found.group(1).lower() == "least" else "max"
        return op, found.group(2).strip()
    return "is", value


REDIRECT = re.compile(r"(?:for|as for|regarding|about) (?:the )?([a-z][a-z \-]{1,30}?), (?:please )?(?:take|read|use|get|check|"
                      r"see|refer|go by|look|consult|defer)\b|([a-z][a-z \-]{1,30}?) (?:must|should|has to|is to) be "
                      r"(?:taken|read|referred|looked up) (?:from|to|on)\b|([a-z][a-z \-]{1,30}?) (?:is|are) (?:given|listed|stated) "
                      r"(?:on|in) (?:the|another|a different)")


def redirected_facts(row: dict) -> list[str]:
    """Facts a listing's own text sends elsewhere ('For color, take it from the ... product from <shop>')."""
    found = []
    for value in row.values():
        if isinstance(value, str) and len(value) < 2000 and re.search(r"take|taken|refer|another|different|not (?:from )?this|"
                                                                      r"go by|rather than|reference|defer", value.lower()):
            for hit in REDIRECT.finditer(value.lower()):
                name = next(g for g in hit.groups() if g).strip()
                name = re.sub(r"^(?:the|its|this item's|this listing's)\s+", "", name)
                if name and len(name.split()) <= 3:
                    found.append(spoken_name(name))
    return found


def spoken_name(name: str) -> str:
    """A property name as a shopper reads it: separators to spaces, words unstemmed ('fa_pattern' -> 'fa pattern')."""
    plain = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name)).split(".")[-1]   # loadCapacity, item.color
    words = [w for w in re.findall(r"[a-z0-9]+", plain.lower()) if not w.isdigit()]
    return " ".join(words) or " ".join(re.findall(r"[a-z0-9]+", str(name).lower())) or name


def sold_out_words(text: str) -> bool:
    """The refusal says the pick can no longer be had."""
    return bool(re.search(r"zero left|sold out|nothing left|none left|nobody has any|ran dry|all gone|dried up", text)
                or re.search(r"can.t be ordered|cannot be ordered|showing unavailable|gone from the shelves|last of those", text))


def price_moved(text: str) -> bool:
    """A refusal about the price of the pick, in either direction ('has come down', 'gone up by a quarter')."""
    priced = re.search(r"charge|price|cost|sum|money wanted|dearer|cheaper", text)
    moved = re.search(r"come down|gone (?:up|down)|went (?:up|down)|risen|rose|lower|higher|dearer|more than|fresh|increase", text)
    return bool(priced and moved) or bool(re.search(r"now costs|fresh sum", text))


def non_answer(reply: str) -> bool:
    """The shopper's standard ways of saying 'no preference' (indifference, never a value)."""
    text = reply.lower()
    return any(p.search(text) for p in _NON_ANSWERS)


def kin(a: str, b: str) -> bool:
    """Two property names that share a distinctive word ('connectivity type' and 'connectivity_speed', 'collar' and
    'collar_type') are treated as one subject when choosing what to ask."""
    return bool({g for g in grains(a) if len(g) >= 5} & {g for g in grains(b) if len(g) >= 5})


def attribute_words(name: str, product: set[str]) -> set[str]:
    """The words that make an attribute name: stemmed, three letters or more, the product's own words left out
    ('fa_pattern' -> {pattern}, 'clothing.material' -> {clothing, material})."""
    words = set()
    for word in re.findall(r"[a-z0-9]+", name.lower().replace("_", " ").replace(".", " ")):
        if len(word) <= 2 or word in product:
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def same_attribute(one: set[str], other: set[str]) -> bool:
    """Two attribute names are one menu entry when the words of the shorter are among the longer's, a word
    standing for any longer word it abbreviates ('max' for 'maximum'); 'pack size' and 'pack type' stay apart."""
    if not one or not other:
        return False
    small, large = (one, other) if len(one) <= len(other) else (other, one)
    for word in small:
        if not any(word == v or (len(word) >= 3 and v.startswith(word)) for v in large):
            return False
    return True


def first_stated(props: dict, names: list[str]) -> Optional[str]:
    """The first non-empty value of any of the names in a listing's properties, as text."""
    for name in names:
        value = props.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def facts_keys(raw: dict) -> set[str]:
    """Keys that sit inside a nested facts block of a listing (as opposed to row-level fields like brand)."""
    return {str(k) for v in raw.values() if isinstance(v, dict) for k in v}


def variant_like(value: str) -> bool:
    """Seller variant strings ('black / int:xl', 'c032#', 'gift box red') are not properties worth a question."""
    return bool(re.search(r"[/#:]|\bint\b|\d\s*x\s*\d", value.lower()))


def category_coverage(kind: str, category: Any) -> int:
    """How many of the kind's words (three letters or more, stemmed) the category carries."""
    want = {g for g in grains(kind) if len(g) > 2}
    have = set(grains(category or ""))
    return len(want & have)


def category_purity(kind: str, category: Any) -> bool:
    """A category made only of the product's own words ('Jeans' for slim fit jeans, not 'Pants & Jeans')."""
    have = {g for g in grains(category or "") if len(g) > 2}
    return bool(have) and have <= {g for g in grains(kind) if len(g) > 2}


def kind_fits(kind: str, category: Any) -> Optional[bool]:
    """Category decides the kind.  'dining set' fits 'Dining Room Sets'; an empty category decides nothing."""
    want = {g for g in grains(kind) if len(g) > 2}
    have = set(grains(category or ""))
    if not want or not have:
        return None
    if want <= have:
        return True
    long_hits = [g for g in want if len(g) > 3 and g in have]
    return True if long_hits else False


# --------------------------------------------------------------------------------------------------------------------
# Listings and the field lens: worlds rename every field, so roles are learned from the shape of the rows.
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class Listing:
    ref: str
    price: Optional[float]
    stock: Optional[bool]
    category: str
    title: str
    props: dict[str, Any]
    raw: dict

    def merge(self, other: "Listing") -> None:
        """A details or offer view adds to what the search row showed; later data wins when present."""
        if other.price is not None:
            self.price = other.price
        if other.stock is not None:
            self.stock = other.stock
        self.category = other.category or self.category
        self.title = other.title or self.title
        self.props.update(other.props)
        self.raw.update(other.raw)


_HINT = {
    "ref": re.compile(r"(^|_)(id|ref|sku|code|key)($|_)|_id$|id$"),
    "price": re.compile(r"price|amount|cost|total|fee"),
    "stock": re.compile(r"stock|avail|hand|sold|inventory|qty|quantity"),
    "category": re.compile(r"categ|aisle|depart|section|shelf|dept|kind|type$|class"),
    "title": re.compile(r"name|title|label"),
    "props": re.compile(r"fact|prop|attr|spec|detail|feature"),
}


def merged_departments(wants: list, browse: Optional["Desk"]) -> list:
    """'bookshelves and wall shelves' read as two products is one department ('bookshelves & wall shelves') when
    the browse tool's enum names it: every word of the wants together lives in one enum value."""
    name = browse.param("department") if browse else None
    spec = browse.schema["properties"].get(name) if name else None
    if len(wants) < 2 or not spec or not is_enum(spec):
        return wants
    def stems(t: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(t).lower())
                if w not in ("and", "or", "the", "of", "for")}
    together = set().union(*(stems(w.kind) for w in wants))
    for value in spec["enum"]:
        if together and together <= stems(value) and len(together) >= len(stems(value)) - 1:
            first = wants[0]
            first.label = first.kind = " and ".join(w.kind for w in wants)
            first.phrases = [w.kind for w in wants] + [p for w in wants for p in w.phrases if p not in [x.kind for x in wants]]
            first.needs = [n for w in wants for n in w.needs]
            first.quantity = max(w.quantity for w in wants)
            return [first]
    return wants


def closest_enum(text: str, values: list) -> Optional[str]:
    """The enum value sharing most word stems with `text` ('bras' -> 'bras, camisoles & slips'); None on no overlap."""
    def stems(t: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(t).lower())
                if w not in ("and", "or", "the", "of", "for")}
    want = stems(text)
    best, score = None, 0
    for value in values:
        shared = len(want & stems(value))
        if shared > score:
            best, score = value, shared
    return best


def offset_param(name: str, spec: dict) -> bool:
    """A start/offset parameter counts rows, not pages."""
    text = snake(name) + " " + str(spec.get("description") or "").lower()
    return bool(re.search(r"offset|start|begin|from|skip|position|first row|list_start", text)) and "page" not in text


def snake(key: str) -> str:
    """ListingIdCode -> listing_id_code: worlds name fields in camel case, the hints are written for words."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()


def hinted(role: str, key: str) -> bool:
    return bool(_HINT[role].search(snake(key)))


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def dict_lists(obj: Any, depth: int = 0) -> list[list[dict]]:
    """Every list of dicts inside an observation, biggest first; the biggest is almost always the result rows."""
    found: list[list[dict]] = []
    if isinstance(obj, list):
        if obj and all(isinstance(e, dict) for e in obj):
            found.append(obj)
        for e in obj:
            if depth < 4:
                found.extend(dict_lists(e, depth + 1))
    elif isinstance(obj, dict):
        for v in obj.values():
            if depth < 4:
                found.extend(dict_lists(v, depth + 1))
    return sorted(found, key=len, reverse=True)


def string_lists(obj: Any, depth: int = 0) -> list[str]:
    """Every list of strings inside a response, flattened: that is where a world puts the shopper's refusal."""
    out: list[str] = []
    if isinstance(obj, list) and obj and all(isinstance(e, str) for e in obj):
        out.extend(obj)
    elif isinstance(obj, dict) and depth < 4:
        for v in obj.values():
            out.extend(string_lists(v, depth + 1))
    elif isinstance(obj, list) and depth < 4:
        for v in obj:
            out.extend(string_lists(v, depth + 1))
    return out


def split_envelope(observation: Any) -> tuple[Any, list[str]]:
    """{<any key>: real result, <any key>: ["shopper words", ...]}: the result and the words, apart.  The keys
    change every race, so only the shape counts: two keys, one a list of sentences, the other the result."""
    if not isinstance(observation, dict) or len(observation) != 2:
        return observation, []
    first, second = observation.values()
    for words, inner in ((first, second), (second, first)):
        if isinstance(words, list) and words and all(isinstance(w, str) for w in words) \
                and isinstance(inner, (dict, list)) and any(" " in w for w in words):
            return inner, list(words)
    return observation, []


LOOSENING = re.compile(r"no longer (?:a )?(?:condition|requirement|must|needed|matters?)|leave out|scrap|"
                       r"widen the net|not fussed|doesn.t matter|does not matter|no need for|drop the|forget (?:about )?the|"
                       r"never mind (?:about )?the|holding to it tighter|any [a-z]+ will do|"
                       r"isn.t worth|not a must|optional now|let (?:that|it) go")


def confirmation_refused(reply: str) -> bool:
    """The shopper's answer to the basket confirmation: anything that is not a refusal or a change is a yes."""
    low = reply.lower().strip()
    if not low or re.search(r"don.t mind|no problem|no worries|not a problem|why not|can.t complain", low):
        return False
    # an acknowledgement ('understood', 'I have read your note') with no stop or change in it is a yes, whatever
    # the shopper says about their kitchen around it ('the dal can wait', 'only half listening')
    if re.search(r"understood|i see it|seen (?:it|your note|that)|read (?:your note|that|it)|taken (?:that|it) on board|"
                 r"carry on|proceed|continue|go ahead|clear to me|accepted|noted|got it|sounds good|please do", low) \
            and not re.search(r"instead|swap|remove|cancel|don.t (?:order|buy|place)|do not (?:order|buy|place)|not yet|"
                              r"hold on|\bstop\b|no longer|leave out|scrap|cheaper|too (?:much|dear|expensive)|over my|"
                              r"sold out|budget|\d+\s*(?:\u00a2|cents?|cts|dollars?)|\bspend\b|rather|change|wrong", low):
        return False
    changes = r"no longer|instead|\bonly\b|budget|\bbut\b|leave out|scrap|swap|cheaper|sold out|over my|forget"
    if re.search(r"\b(?:fine|whatever|up to you|as you (?:see|like|think)|no preference|good|grand|lovely|perfect|"
                 r"go on|go ahead|please do|confirm(?:ed)?|happy)\b", low) and not re.search(changes, low):
        return False
    if re.match(r"(?:yes|yeah|yep|aye|sure|ok|okay|fine|grand|right|perfect|great|lovely|go ahead|please do|"
                r"do it|confirm|that.s (?:fine|grand|right|good|perfect)|sounds good|absolutely|certainly)\b", low) \
            and not re.search(r"\bbut\b|\bhowever\b|no longer|instead|only", low):
        return False
    return bool(re.search(r"^no\b|\bnot\b|n.t\b|\bwait\b|hold on|actually|instead|rather|change|swap|remove|"
                          r"leave out|scrap|drop|no longer|budget|cheaper|too (?:much|dear|expensive)|over my|"
                          r"\bcan only\b|\bonly\b|forget|never mind|sold out|gone", low))


def flat_text(obj: Any) -> str:
    """All strings in an observation joined, for reading replies and refusals."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(flat_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flat_text(v) for v in obj)
    return "" if obj is None else str(obj)


class Lens:
    """Learns which key plays which role in listing rows, structurally first and by name hint as a tie-break.

    Roles: ref (unique short string), price (number), stock (bool or count), category (short string shared by many
    rows), title (distinct string with spaces), props (nested dict of short values).  Scalar string fields that
    are none of those are treated as properties too, since some worlds keep properties flat.
    """

    def __init__(self) -> None:
        self.roles: dict[str, Optional[str]] = dict.fromkeys(_HINT)

    # -- learning ---------------------------------------------------------------------------------------------------

    def learn(self, rows: list[dict]) -> None:
        if not rows:
            return
        keys = [k for k in rows[0] if all(k in r for r in rows)]
        column = {k: [r.get(k) for r in rows] for k in keys}

        def pick(role: str, fits: Callable[[str, list], bool], rank: Callable[[str, list], float]) -> None:
            if self.roles[role] in keys:
                return
            best, best_score = None, None
            for k in keys:
                if k in self.roles.values() or not fits(k, column[k]):
                    continue
                score = (hinted(role, k), rank(k, column[k]))
                if best_score is None or score > best_score:
                    best, best_score = k, score
            if best is not None:
                self.roles[role] = best

        strings = lambda vals: all(isinstance(v, str) for v in vals)
        pick("props", lambda k, vals: all(isinstance(v, dict) for v in vals),
             lambda k, vals: sum(len(v) for v in vals))
        pick("ref", lambda k, vals: strings(vals) and all(v and " " not in v for v in vals)
             and len(set(vals)) == len(vals),
             lambda k, vals: (sum(any(c.isdigit() for c in v) for v in vals), -sum(len(v) for v in vals)))
        pick("price", lambda k, vals: all(is_number(v) for v in vals) and not hinted("stock", k),
             lambda k, vals: sum(vals) / max(1, len(vals)))
        pick("stock", lambda k, vals: all(isinstance(v, bool) for v in vals)
             or (all(is_number(v) for v in vals) and hinted("stock", k)),
             lambda k, vals: 0.0)
        # the title is the most distinctive text column (every listing its own), the category the column whose
        # short values repeat across rows: judged on distinctness first, so a short category is never the title
        pick("title", lambda k, vals: strings(vals) and any(" " in v for v in vals)
             and sum(len(v) for v in vals) / len(vals) < 400 and not hinted("category", k),
             lambda k, vals: (len(set(vals)) / len(vals), min(120.0, mean_length(vals))))
        pick("category", lambda k, vals: strings(vals) and all(0 < len(v) < 60 for v in vals),
             lambda k, vals: -len(set(vals)) if len(vals) > 1 else 0.0)

    # -- reading ----------------------------------------------------------------------------------------------------

    def listing(self, row: dict) -> Optional[Listing]:
        ref_key = self.roles["ref"]
        ref = row.get(ref_key) if ref_key else None
        if ref is None:
            # a single-listing view (details) may not share the search row's shape: fall back to any id-ish string
            for k, v in row.items():
                if isinstance(v, str) and v and " " not in v and hinted("ref", k):
                    ref = v
                    break
        if ref is None:
            return None
        price = self.price_of(row)
        stock = self.stock_of(row)
        category = str(row.get(self.roles["category"]) or "") if self.roles["category"] else ""
        title = str(row.get(self.roles["title"]) or "") if self.roles["title"] else ""
        return Listing(str(ref), price, stock, category, title, self.props_of(row), dict(row))

    def price_of(self, row: dict) -> Optional[float]:
        key = self.roles["price"]
        if key and is_number(row.get(key)):
            return float(row[key])
        for k, v in row.items():
            if hinted("price", k) and not hinted("stock", k):
                if is_number(v):
                    return float(v)
                if isinstance(v, dict):
                    inner = self.price_of(v)
                    if inner is not None:
                        return inner
        return None

    def stock_of(self, row: dict) -> Optional[bool]:
        key = self.roles["stock"]
        if key and key in row:
            v = row[key]
            return bool(v) if isinstance(v, bool) else (v > 0 if is_number(v) else None)
        for k, v in row.items():
            if hinted("stock", k) and not hinted("price", k):
                if isinstance(v, bool):
                    return v
                if is_number(v):
                    return v > 0
                if isinstance(v, str) and v:
                    return not re.search(r"\b(out|sold|no|unavail|0)\b", v.lower())
        return None

    def props_of(self, row: dict) -> dict[str, Any]:
        props: dict[str, Any] = {}
        blocks, any_blocks = [], []
        for k, v in row.items():
            if not isinstance(v, dict):
                continue
            any_blocks.append(v)
            if k == self.roles["props"] or hinted("props", k):
                blocks.append(v)
        for block in blocks or any_blocks:
            for k, v in block.items():
                if isinstance(v, list):
                    if all(isinstance(e, str) for e in v):
                        props[str(k)] = ", ".join(v)
                elif isinstance(v, (str, int, float, bool)):
                    props[str(k)] = v
        taken = set()
        for role in ("ref", "price", "stock", "title", "props"):
            taken.add(self.roles[role])
        for k, v in row.items():
            if k in taken or k in props:
                continue
            if hinted("ref", k) or hinted("price", k):
                continue
            if isinstance(v, str) and 0 < len(v) <= 60:
                props[k] = v
        for fact in redirected_facts(row):
            for key in [k for k in props if spoken_name(k) == fact or fact in spoken_name(k).split(" ")]:
                props.pop(key)   # the seller says this fact lives on another listing: here it is unknown
        return props

    def rows_of(self, observation: Any) -> list[dict]:
        lists = dict_lists(observation)
        if lists:
            return lists[0]
        if isinstance(observation, dict) and self.listing(observation) is not None:
            return [observation]
        return []


# --------------------------------------------------------------------------------------------------------------------
# Desks: one per tool role; each builds schema-valid arguments for its tool.
# --------------------------------------------------------------------------------------------------------------------

ROLES = ("search", "filter", "details", "compare", "offer", "cart_add", "cart_remove", "cart_view", "order", "message", "decline")


def schema_of(tool: dict) -> dict:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    return {"name": str(fn.get("name") or ""), "description": str(fn.get("description") or ""),
            "properties": params.get("properties") if isinstance(params.get("properties"), dict) else {},
            "required": [r for r in params.get("required") or () if isinstance(r, str)]}


@dataclass
class Desk:
    """A tool with the meaning of its parameters: slot name -> parameter name, plus constant arguments."""
    role: str
    name: str
    schema: dict
    slots: dict[str, str] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)   # e.g. the search mode's filter value

    def param(self, slot: str) -> Optional[str]:
        name = self.slots.get(slot)
        return name if name in self.schema["properties"] else None

    @property
    def lines(self) -> Optional[dict]:
        """For a cart tool whose line is nested (an array of line objects, or one line object): where the listing id
        and the count go inside it, and whether one call sets the whole basket."""
        return self.notes.get("lines")

    def build(self, **values: Any) -> dict[str, Any]:
        """Arguments for one call: slot values typed per the schema, constants, and required parameters filled.
        A nested cart line is filled from the item schema: `basket=[(id, count), ...]` (or id/quantity for one
        line) becomes the array of line objects, each with its own id and count field."""
        props = self.schema["properties"]
        args: dict[str, Any] = {}
        for key, value in self.fixed.items():
            if key in props:
                args[key] = value
        nest = self.lines
        if nest is not None:
            single = (values.pop("id", None), values.pop("quantity", None) or 1)
            basket = values.pop("basket", None) or ([single] if single[0] is not None else [])
            objects = [nested_line(nest, *pair) for pair in basket]
            if nest.get("array"):
                args[nest["param"]] = objects
            elif objects:
                args[nest["param"]] = objects[0]
        for slot, value in values.items():
            name = self.param(slot)
            if name and value is not None:
                args[name] = coerce(value, props[name])
        for name in self.schema["required"]:
            if name not in args and name in props:
                args[name] = placeholder(props[name])
        return args


def coerce(value: Any, spec: dict) -> Any:
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    try:
        if kind in ("integer", "number"):
            number = int(float(value)) if kind == "integer" else float(value)
            if is_number(spec.get("minimum")):
                number = max(number, spec["minimum"])
            if is_number(spec.get("maximum")):
                number = min(number, spec["maximum"])
            return number
        if kind == "string":
            if spec.get("enum") and str(value) not in spec["enum"]:
                folded = {str(v).lower().strip(): v for v in spec["enum"]}
                return folded.get(str(value).lower().strip(), closest_enum(str(value), spec["enum"]) or value)
            return str(value)
        if kind == "array":
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]
        if kind == "boolean":
            return bool(value)
    except (TypeError, ValueError):
        return value
    return value


_LINE_ID = re.compile(r"(^|_)(id|ids|ref|sku|code|key|item|items|listing|product|which|what|article|goods|thing)($|_)|id$")
_LINE_COUNT = re.compile(r"qty|quant|count|units?|pieces?|number|num|amount|how_many|many|wanted|times|copies")


def line_fields(spec: Any) -> Optional[dict]:
    """The id field and the count field of a line object schema: the id-like string gets the listing id, the
    integer the count.  Other required fields get their schema placeholder."""
    if not isinstance(spec, dict):
        return None
    props = spec.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    strings = [k for k in typed_keys(props, "string") if not is_enum(props[k])]
    ints = typed_keys(props, "integer", "number")
    if not strings:
        return None
    ident = next((k for k in strings if _LINE_ID.search(snake(k))), None)
    if ident is None:
        ident = next((k for k in strings if not re.search(r"note|comment|text|message|why|reason", snake(k))), strings[0])
    count = next((k for k in ints if _LINE_COUNT.search(snake(k))), ints[0] if ints else None)
    required = [r for r in spec.get("required") or () if isinstance(r, str) and r in props]
    extra = {k: placeholder(props[k]) for k in required if k not in (ident, count)}
    return {"id": ident, "qty": count, "extra": extra, "props": props}


def nested_line(nest: dict, ref: Any, qty: Any) -> dict:
    line = dict(nest.get("extra") or {})
    line[nest["id"]] = str(ref)
    if nest.get("qty"):
        line[nest["qty"]] = coerce(qty if qty is not None else 1, nest["props"][nest["qty"]])
    return line


def nested_cart(props: dict, description: str = "") -> Optional[dict]:
    """A parameter that is an array of line objects (or a single line object) carrying an id and a count: the
    cart line of a nested cart tool.  An array of lines sets the whole basket unless the tool says it adds."""
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("type")
        if kind == "array" and isinstance(spec.get("items"), dict):
            fields = line_fields(spec["items"])
            if fields:
                text = (description + " " + str(spec.get("description") or "")).lower()
                adds = re.search(r"\b(?:add|adds|append|appends|plus|increase|increment)\b", text)
                sets = re.search(r"\b(?:set|sets|replace|replaces|overwrite|whole|entire|full|all lines|becomes|"
                                 r"exactly|clears?|empty list)\b", text)
                return {"param": name, "array": True, "whole": bool(sets or not adds), **fields}
        elif isinstance(spec.get("properties"), dict):
            fields = line_fields(spec)
            if fields and fields["qty"]:
                return {"param": name, "array": False, "whole": False, **fields}
    return None


def lists_ids(spec: Any) -> bool:
    """An array parameter that takes plain ids (the compare tool), never line objects."""
    if not isinstance(spec, dict) or spec.get("type") != "array":
        return False
    items = spec.get("items")
    return not (isinstance(items, dict) and (items.get("type") == "object" or isinstance(items.get("properties"), dict)))


_CODE_VALUE = re.compile(r"^[A-Z]{1,5}[-_ ]?\d{1,5}$")
_FAULT_KEY = re.compile(r"error|issue|fault|problem|fail|reject|denied|invalid|(^|_)code($|_)|(^|_)err($|_)")


def failure_reply(observation: Any) -> bool:
    """A reply that says the call did not happen: an error field, or an opaque code ({"issue_text": "E15"})."""
    if not isinstance(observation, dict) or not 0 < len(observation) <= 3:
        return False
    for key in observation:
        value = observation[key]
        if value in (None, False, "", [], {}):
            continue
        coded = isinstance(value, str) and _CODE_VALUE.match(value.strip())
        if coded or (_FAULT_KEY.search(snake(key)) and isinstance(value, (str, int, bool, dict))):
            return True
    return False


def placeholder(spec: dict) -> Any:
    if "default" in spec:
        return spec["default"]
    if spec.get("enum"):
        return spec["enum"][0]
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind in ("integer", "number") and is_number(spec.get("minimum")):
        return spec["minimum"]
    return {"integer": 1, "number": 1, "boolean": True, "array": [], "object": {}}.get(kind, "")


def guess_desks(tools: list[dict]) -> dict[str, Desk]:
    """Structural fallback when the planner could not assign roles: parameter shapes tell most tools apart.

    A list-typed parameter means compare; a lone id-ish string means details (or offer when its name says price);
    an integer next to an id means the cart add; no parameters at all means cart view or order; the finishing
    tools and the message tool are told apart by their descriptions.
    """
    desks: dict[str, Desk] = {}
    bare: list[tuple[dict, str]] = []   # tools without parameters: view, order and decline, told apart last
    singles: list[tuple[dict, str, str]] = []   # one-id tools: the one about properties is details, a second the offer
    searches: list[tuple[dict, dict[str, str], bool]] = []
    for tool in tools:
        sc = schema_of(tool)
        blob = (snake(sc["name"]) + " " + sc["description"]).lower().replace("_", " ")
        final = bool(re.search(r"\bfinal\b|no more actions|ends? the (?:visit|chat|episode)", blob))
        props = sc["properties"]
        strings = typed_keys(props, "string")
        ints = typed_keys(props, "integer", "number")
        arrays = typed_keys(props, "array")
        ids, free = [], []
        for k in strings:
            if hinted("ref", k):
                ids.append(k)
            if not is_enum(props[k]):   # enums are modes, never free text
                free.append(k)
        cartish = bool(re.search(r"\b(cart|basket|bag|trolley)\b", blob) or re.search(r"\b(hold|add|reserve|keep|put)\b", blob))
        arrays = [k for k in arrays if lists_ids(props[k])]   # an array of line objects is a cart, never a compare
        role, slots = None, {}
        if not props:
            bare.append((sc, blob))
            continue
        nest = nested_cart(props, sc["description"])
        if nest is not None:
            if "cart_add" not in desks:
                desks["cart_add"] = Desk("cart_add", sc["name"], sc, {}, {}, {"lines": nest})
            continue
        if (re.search(r"declin|abort|give up|abstain|refus|without|walk|leave", blob)
                or re.search(r"no purchase|close out|stand down|cancel", blob)) and not ints and not arrays:
            role, slots = "decline", {"text": strings[0] if strings else ""}
        elif re.search(r"message|ask|contact|chat|shopper|customer|note|question", blob) and strings and not ids \
                and not ints and not final:
            role, slots = "message", {"text": strings[0]}
        elif arrays:
            role, slots = "compare", {"ids": arrays[0]}
        elif ints and len(free) == 1 and (ids or cartish):
            role, slots = "cart_add", {"id": free[0], "quantity": ints[0]}   # an id plus a count is the cart line
        elif ids and ints:
            role, slots = "cart_add", {"id": ids[0], "quantity": ints[0]}
        elif ids and re.search(r"price|offer|stock|avail|quote", blob):
            role, slots = "offer", {"id": ids[0]}
        elif ids and len(props) == 1:
            singles.append((sc, ids[0], blob))
            continue
        elif strings:
            slots = search_slots(props, strings, ints)
            searches.append((sc, slots, "query" in slots))
            continue
        if role and role not in desks:
            desks[role] = Desk(role, sc["name"], sc, slots)
    singles.sort(key=lambda e: -len(re.findall(r"spec|detail|feature|propert|attribut|fact|table|everything|full", e[2])))
    for sc, ident, _blob in singles:
        role = "details" if "details" not in desks else "offer"
        if role not in desks:
            desks[role] = Desk(role, sc["name"], sc, {"id": ident})
    # the search is the string tool with free text; a second one with price/brand/range filters is the filter
    for sc, slots, has_query in sorted(searches, key=lambda t: not t[2]):
        role = "search" if "search" not in desks and has_query else "filter"
        if role not in desks:
            if role == "filter":
                slots = {k: v for k, v in slots.items() if k != "query"}
            desks[role] = Desk(role, sc["name"], sc, slots)
    ordering = re.compile(r"order|check ?out|purchas|buy|submit|pay|settle|finali[sz]|complete")
    viewing = re.compile(r"show|view|list|content|peek|display|status|current")
    carting = re.compile(r"cart|basket|bag|trolley|selection|items|lines|held|holding")
    metering = re.compile(r"usage|allowance|quota|budget|remaining|leftover|used so far|limit")
    declining = (re.compile(r"declin|abort|give up|abstain|refus"),
                 re.compile(r"without|stand down|cancel|no purchase|withdraw|no order"))
    def orderliness(entry: tuple[dict, str]) -> int:
        return len(ordering.findall(entry[1])) - len(viewing.findall(entry[1]))

    bare.sort(key=orderliness, reverse=True)
    # of the plain tools left after order and decline, the cart view is the one about the cart, never a meter
    viewish = [e for e in bare if not ordering.search(e[1]) and not any(d.search(e[1]) for d in declining)]
    viewish.sort(key=lambda e: (len(carting.findall(e[1])) - 2 * len(metering.findall(e[1])), len(viewing.findall(e[1]))),
                 reverse=True)
    bare = [e for e in bare if e not in viewish] + viewish
    for sc, blob in bare:
        if any(d.search(blob) for d in declining) and "decline" not in desks:
            role = "decline"
        elif "order" not in desks and (ordering.search(blob) or "cart_view" in desks):
            role = "order"
        else:
            role = "cart_view"
        if role not in desks:
            desks[role] = Desk(role, sc["name"], sc)
    return desks


_ORDERING = re.compile(r"order|check ?out|purchas|buy|submit|pay|settle|finali[sz]|complete|place|confirm|commit")
_DECLINING = re.compile(r"declin|abort|give up|abstain|refus|without|stand down|cancel|no purchase|withdraw|no order|"
                        r"walk away|skip|pass on|forgo|none fit|nothing fits")
_VIEWING = re.compile(r"show|view|list|content|peek|display|status|current|look|what.s in|see")
_CARTING = re.compile(r"cart|basket|bag|trolley|selection|items|lines|held|holding|picks")


_CUE_WEIGHTS = {"order": (2, -3, -1, 0), "decline": (0, 2, 0, 0), "cart_view": (-2, -3, 1, 1)}


def role_cue(role: str, schema: dict) -> int:
    """How much a tool's name and description read like a finishing or cart role (a tool taking cart lines is
    the cart, never the order or the view)."""
    blob = (snake(schema.get("name", "")) + " " + str(schema.get("description") or "")).lower().replace("_", " ")
    nested = nested_cart(schema.get("properties") or {}, str(schema.get("description") or "")) is not None
    if role == "cart_add":
        return 10 * nested
    weights = _CUE_WEIGHTS.get(role, (0, 0, 0, 0))
    hits = [len(cue.findall(blob)) for cue in (_ORDERING, _DECLINING, _VIEWING, _CARTING)]
    return sum(w * h for w, h in zip(weights, hits)) - (5 * nested if role != "decline" else 0)


def search_slots(props: dict, strings: list[str], ints: list[str]) -> dict[str, str]:
    """Which parameter of a search-like tool is the free text, the page token, the price ceiling, the department
    and the brand, read from parameter names and descriptions."""
    slots: dict[str, str] = {}

    def text_of(k: str) -> str:
        spec = props.get(k) if isinstance(props.get(k), dict) else {}
        return (k + " " + str(spec.get("description") or "")).lower()

    for k in strings:
        t = text_of(k)
        if "page" not in slots and re.search(r"page|cursor|token|next|after|marker", t):
            slots["page"] = k
        elif "department" not in slots and re.search(r"depart|categ|section|aisle|dept|shelf|range|class", t):
            slots["department"] = k
        elif "brand" not in slots and re.search(r"brand|marque|maker|manufactur", t):
            slots["brand"] = k
        elif "query" not in slots and re.search(r"keyword|text|query|search|term|phrase|find|words", t):
            slots["query"] = k
    for k in ints:
        t = text_of(k)
        if "price_max" not in slots and re.search(r"price|max|ceiling|cost|budget|under|spend", t):
            slots["price_max"] = k
        elif "page" not in slots and re.search(r"page|offset|start|begin|from|skip|position", t):
            slots["page"] = k
    if "query" not in slots and "department" not in slots and "brand" not in slots:
        for k in strings:
            if k in slots.values() or is_enum(props.get(k)):
                continue
            slots["query"] = k
            break
    return slots


def desks_from_plan(tools: list[dict], assignment: Any) -> dict[str, Desk]:
    """Desks from the planner's tool-role assignment, validated against the schemas; gaps filled structurally."""
    by_name = {schema_of(t)["name"]: schema_of(t) for t in tools}
    desks: dict[str, Desk] = {}
    if isinstance(assignment, dict):
        for role in ROLES:
            spec = assignment.get(role)
            if not isinstance(spec, dict) or spec.get("name") not in by_name:
                continue
            sc = by_name[spec["name"]]
            slots = {}
            for k, v in spec.items():
                if k == "name" or not isinstance(v, str):
                    continue
                if v in sc["properties"]:
                    slots[k] = v
            fixed = spec.get("fixed") if isinstance(spec.get("fixed"), dict) else {}
            fixed = {k: v for k, v in fixed.items() if k in sc["properties"]}
            nest = nested_cart(sc["properties"], sc["description"])
            if nest is not None:
                if role != "cart_add":
                    continue   # a tool taking line objects is the cart, whatever the planner called it
                slots = {}
                fixed = {k: v for k, v in fixed.items() if k != nest["param"]}
            desk = Desk(role, sc["name"], sc, slots, fixed)
            if nest:
                desk.notes["lines"] = nest
            if desk_is_sound(desk):
                desks[role] = desk
    guessed = guess_desks(tools)
    for role in ("order", "decline", "cart_view", "cart_add"):
        # the finishing tools are told apart by their own words: a planner pick that reads less like the role than
        # the structural pick gives way (an 'items in the bag' view is never the order, 'abstain' never the buy)
        mine, theirs = guessed.get(role), desks.get(role)
        if mine is not None and theirs is not None and mine.name != theirs.name \
                and role_cue(role, mine.schema) > role_cue(role, theirs.schema):
            for other, desk in list(desks.items()):
                if desk.name == mine.name and other != role:
                    desks.pop(other)
            desks[role] = mine
    for role, desk in guessed.items():
        if role not in desks and desk.name not in {d.name for d in desks.values()}:
            desks[role] = desk
    if "cart_remove" not in desks and "cart_add" in desks:
        remove = derive_remove(desks["cart_add"])
        if remove is not None:
            desks["cart_remove"] = remove
    for role in ("search", "filter"):
        if role in desks:
            learn_search_filters(desks[role])
    return desks


def learn_search_filters(search: Desk) -> None:
    """Find the search tool's department filter and its keyword/filter mode switch from the schema alone."""
    props = search.schema["properties"]
    if not search.param("department"):
        for name, spec in props.items():
            if isinstance(spec, dict) and spec.get("type") == "string" and name != search.param("query") \
                    and re.search(r"depart|categ|section|aisle|dept|shelf|range", (name + " " + str(spec.get("description") or "")).lower()):
                search.slots["department"] = name
                break
    for name, spec in props.items():
        values = spec.get("enum") if isinstance(spec, dict) else None
        if not isinstance(values, list) or len(values) < 2:
            continue
        filtering = [v for v in values if re.search(r"field|filter|facet|attribute|struct", str(v).lower())]
        if filtering:
            search.slots["mode"] = name
            search.notes["filter_value"] = filtering[0]
            if name not in search.fixed:
                search.fixed[name] = next((v for v in values if v not in filtering), values[0])
            break


def desk_is_sound(desk: Desk) -> bool:
    """Code checks the planner's role map: a search needs a free-text string, a cart line needs an id plus the
    count when the schema has one, details/offer need an id string.  An unsound assignment is dropped so the
    structural guess can fill the role."""
    props = desk.schema["properties"]

    def kind(slot: str) -> Optional[str]:
        name = desk.param(slot)
        spec = props.get(name) if name else None
        return spec.get("type") if isinstance(spec, dict) else None

    if desk.role == "search":
        return kind("query") == "string"
    if desk.role in ("details", "offer"):
        return kind("id") == "string"
    if desk.role == "cart_add" and desk.lines is not None:
        return True
    if desk.role in ("cart_add", "cart_remove"):
        if kind("id") != "string":
            return False
        ints = typed_keys(props, "integer", "number")
        if ints and not desk.param("quantity"):
            desk.slots["quantity"] = ints[0]
        return True
    if desk.role == "compare":
        return kind("ids") == "array" and lists_ids(props.get(desk.param("ids")))
    if desk.role == "message":
        return kind("text") == "string"
    return True


def derive_remove(add: Desk) -> Optional[Desk]:
    """The add tool often carries an enum mode (include/discard, plus/minus): the other value removes a line."""
    for name, spec in add.schema["properties"].items():
        values = spec.get("enum") if isinstance(spec, dict) else None
        if not isinstance(values, list) or len(values) < 2:
            continue
        current = add.fixed.get(name)
        off = [v for v in values if re.search(r"discard|remov|minus|delet|drop|take|out|releas|clear|subtract", str(v).lower())]
        pick = off[0] if off else next((v for v in values if v != current), None)
        if pick is None or pick == current:
            continue
        if current is None:
            add.fixed[name] = next((v for v in values if v != pick), values[0])
        return Desk("cart_remove", add.name, add.schema, dict(add.slots), {**add.fixed, name: pick}, dict(add.notes))
    return None


# --------------------------------------------------------------------------------------------------------------------
# Counter: steps, time and messages in one place.
# --------------------------------------------------------------------------------------------------------------------

class Counter:
    def __init__(self, max_steps: int, message_cap: Optional[int]) -> None:
        self.started = time.monotonic()
        self.max_steps = max_steps
        self.steps = 0
        self.messages = 0
        self.message_cap = message_cap
        self.held_back = 0        # messages kept for the basket confirmation before the order

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def seconds_left(self) -> float:
        return TASK_SECONDS - self.elapsed()

    def steps_left(self) -> int:
        return self.max_steps - self.steps

    def room(self, reserve: int) -> bool:
        """Can one more exploratory step be spent while keeping `reserve` steps and the closing seconds?"""
        return self.steps_left() > reserve and self.seconds_left() > CLOSING_SECONDS and not self.late()

    def late(self) -> bool:
        """Past the cart deadline: exploration, questions and sweeps stop; the best candidate goes to checkout."""
        return self.elapsed() >= CART_DEADLINE

    def urgent(self) -> bool:
        """Past the order deadline: whatever is held gets ordered now."""
        return self.elapsed() >= ORDER_DEADLINE

    def may_message(self) -> bool:
        return self.message_cap is None or self.messages < self.message_cap

    def may_ask(self) -> bool:
        """A question may go out only while the confirmation's message stays free."""
        return self.message_cap is None or self.messages < self.message_cap - self.held_back


# --------------------------------------------------------------------------------------------------------------------
# Oracle: model calls with backoff and rotation.  Never raises; returns None when inference is gone.
# --------------------------------------------------------------------------------------------------------------------

def reasoning_knob(model: str) -> dict:
    family, _slash, _rest = model.partition("/")
    if family in ("openai", "google"):
        return {"reasoning": {"effort": "low"}}
    if family == "anthropic":
        return {}
    return {"reasoning": {"enabled": False}}


class Oracle:
    """Model calls under the network rules: short patience, rotation on any failure, degraded mode after stalls.

    Validator-side inference stalls are the first-order risk; a call never waits longer than CALL_PATIENCE and a
    stalled model is left for the next one at once.  Two consecutive stalls switch to degraded mode, where every
    remaining job gets a single attempt of DEGRADED_PATIENCE so the episode's clock stays with the environment.
    """

    def __init__(self, counter: Counter, trace: list[dict]) -> None:
        self.counter = counter
        self.trace = trace
        self.stalls = 0        # consecutive calls that timed out or never connected
        self.degraded = False
        self.faults: dict[str, int] = {}   # failed calls per model: a model that failed twice goes to the back
        self.job_misses: dict[str, int] = {}   # whole-chain failures per job: twice in a row and code takes over
        self.answer_seconds: list[float] = []

    @property
    def lean(self) -> bool:
        """Answers are slow (over 8 s on average): optional jobs are done by code alone from here on."""
        return self.degraded or (len(self.answer_seconds) >= 1 and
                                 sum(self.answer_seconds) / len(self.answer_seconds) > LEAN_SECONDS)

    def chain(self, models: tuple[str, ...]) -> list[str]:
        """The job's models in order, a failing model moved to the back for the rest of the episode: after two
        failed calls (consecutive or not), or at once after one call that hit its wall-clock cap.  The counts
        never reset within an episode."""
        healthy, failing = [], []
        for model in models:
            (failing if self.faults.get(model, 0) >= FAULTS_TO_BACK else healthy).append(model)
        return healthy + failing

    def ask(self, tag: str, system: str, user: str, models: tuple[str, ...], tokens: int = MODEL_TOKENS) -> Optional[Any]:
        """One question to the first model that answers with JSON, or None."""
        if self.job_misses.get(tag, 0) >= 2:
            return None   # this job failed on every model twice running: code does it from here on
        order = self.chain(models)
        attempts = 1 if self.degraded else len(order)
        for attempt in range(attempts):
            model = order[attempt % len(order)]
            started = self.counter.elapsed()
            budget = self.counter.seconds_left() - CLOSING_SECONDS
            if budget < 6:
                return None
            body = {"model": model, "temperature": 0, "max_tokens": tokens,
                    "messages": [{"role": "system", "content": f"abacus:{tag}\n{system}"},
                                 {"role": "user", "content": user}], **reasoning_knob(model)}
            patience = min(DEGRADED_PATIENCE if self.degraded else CALL_PATIENCE, budget)
            packet = on_the_clock(open_wire(patience), "/inference/chat/completions", body, patience)
            content = self.content_of(packet)
            parsed = mended_json(content) if content else None
            failure = packet.error if packet.data is None else None
            stalled = bool(failure) and failure.get("kind") == "network"
            self.trace.append({"kind": "model", "tag": tag, "model": model, "ok": parsed is not None,
                               "stalled": stalled, "error": None if packet.data else packet.error})
            if parsed is not None:
                self.stalls = 0
                self.job_misses[tag] = 0
                self.answer_seconds.append(self.counter.elapsed() - started)
                return parsed
            self.faults[model] = self.faults.get(model, 0) + 1
            if stalled:
                self.faults[model] = max(self.faults[model], FAULTS_TO_BACK)   # one timeout is enough
                self.stalls += 1
                if self.stalls >= 2 and not self.degraded:
                    self.degraded = True
                    self.job_misses[tag] = self.job_misses.get(tag, 0) + 1
                    return None   # the rest of this job's chain would only burn the clock
        self.job_misses[tag] = self.job_misses.get(tag, 0) + 1
        return None

    @staticmethod
    def content_of(packet: Packet) -> Optional[str]:
        data = packet.data
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(message, dict) and (content is None or (isinstance(content, str) and not content.strip())):
            # some reasoning models leave content empty and put the whole answer in the reasoning text
            for key in ("reasoning_content", "reasoning"):
                if isinstance(message.get(key), str) and "{" in message[key]:
                    content = message[key]
                    break
        if isinstance(content, list):
            pieces = []
            for c in content:
                if isinstance(c, dict):
                    pieces.append(str(c.get("text", "")))
            content = " ".join(pieces)
        return content if isinstance(content, str) and content.strip() else None


# --------------------------------------------------------------------------------------------------------------------
# Plan: what the request asks for, as the planner model (or a code improvisation) reads it.
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class Want:
    """One product line of the request."""
    label: str
    kind: str
    quantity: int
    phrases: list[str]
    needs: list[Need]
    shopper_words: set[str]


@dataclass
class Plan:
    wants: list[Want]
    objective: str                     # cheapest | best | any
    budget: Optional[float]            # in the shopper's own unit
    unit_hint: str                     # cents | whole | unknown
    message_cap: Optional[int]
    step_cap: Optional[int]
    decline_rule: str
    tools: Any
    notify_first: bool = False   # the request asks to be told before the order is placed


PLANNER_SYSTEM = """You turn a shopper's request into a strict JSON plan for a shopping agent. Reply with JSON only.
Schema:
{"products":[{"label":"short name","kind":"product kind as a category word","quantity":1,
   "search":"best free-text search phrase","alternatives":["other phrases","synonyms","requirement values"],
   "requirements":[{"property":"property name","value":"value","op":"is|not|min|max","wish":false}]}],
 "objective":"cheapest|best|any","budget":{"amount":number,"unit_hint":"cents|whole|unknown"}|null,
 "message_cap":int|null,"step_cap":int|null,
 "decline":{"rule":"what the shopper says to do when nothing fits"},"notify_before_order":false,
 "tools":{"search":{"name":"tool","query":"param","page":"param or null","price_max":"param or null",
             "department":"param or null","mode":"enum param switching keyword/filter search or null","fixed":{}},
   "details":{"name":"tool","id":"param"},"compare":{"name":"tool","ids":"param"}|null,
   "offer":{"name":"tool","id":"param"}|null,
   "cart_add":{"name":"tool","id":"param","quantity":"param","fixed":{"param":"constant"}},
   "cart_remove":{"name":"tool","id":"param","quantity":"param or null","fixed":{"param":"constant"}}|null,
   "cart_view":{"name":"tool"},"order":{"name":"tool","fixed":{}},
   "message":{"name":"tool","text":"param"},"decline":{"name":"tool","text":"param"}}}
Rules: one product entry per distinct product the shopper wants (each gets its own cart line); quantity defaults to 1.
A requirement is ONLY an explicit demand on a property ("the material has to be cotton", "style should be wrap",
"leave out crew neck"). Never turn backstory into requirements (an old zipper that broke, what a roommate likes,
the weather); never stock or availability ("on hand"), price or budget, delivery, step or message limits, or the
shopper's own checking ("I'll check the tag myself"). Name the property with the shopper's own word for it ("fit",
"type", "style", "shape", "clock function") and the value with their words ("clock function" = "color changing",
never "color changing" = "yes"). A wish about occasion, recipient or intended use ("for a wedding", "for my sister")
gets "wish": true. 'not X' is op 'not'; 'at least N' is 'min'; 'at most N' is 'max'.
kind is the category the shopper names (e.g. 'routers', 'cola', 'dining room set'). objective is 'cheapest' when the
shopper wants the cheapest fitting item, 'best' when they want the best, else 'any'. budget.amount is the number the
shopper states; unit_hint is 'cents' when they speak in cents/cts, 'whole' for dollars/euros/pesos etc.
message_cap is the stated number of messages or questions allowed. step_cap is the stated step limit.
notify_before_order is true only when the shopper asks to be told or checked with before the order is placed.
tools maps roles to the given tool names and parameter names; 'fixed' holds constants an enum parameter needs (for
example an add/remove mode, or the keyword value of a search mode switch). search.department is the parameter that
filters by department/category when the tool has one. Omit a role with null when no tool plays it."""

READER_SYSTEM = """You read product listings for a shopping agent. For each listing and each numbered requirement, find the
property that speaks to it and QUOTE its exact property name and value. Decide literally: the quoted value must state
the requirement (synonyms do not count; 'pu leather' is leather; a title never counts). Also say whether the listing's
category is the kind of product wanted. Reply with JSON only:
{"verdicts":[{"id":"listing id","kind_ok":true|false,
  "checks":[{"req":0,"property":"quoted property name or null","value":"quoted value or null","ok":true|false}]}]}"""

REPLY_SYSTEM = """A shopper was asked which value of one product property they want. Read their reply, which may be in
dialect ('Nowt else will serve as processor but ryzen' means processor = ryzen) and may name a different property
than the one asked ('My answer on pack type stays fixed: single' means pack type = single). Reply with JSON only:
{"property":"the property the reply is about, in its words","value":"the value they want, in the reply's words, or
null","no_need":true|false,"negated":false}
no_need is true when they are indifferent or give no value. negated is true when they want anything EXCEPT the value
('anything bar black')."""

REFUSAL_SYSTEM = """An order was refused. Read the refusal and extract what changed. Reply with JSON only:
{"sold_out_ids":["listing ids that are unavailable"],"budget":number or null,
 "dropped":[{"product":"label","property":"name"}],
 "added":[{"product":"label","property":"name","value":"value","op":"is|not|min|max"}],"note":"one line"}
budget is the new total budget if the shopper changed it, else null."""


def plan_from_json(raw: Any, query: str) -> Optional[Plan]:
    raw = unwrapped(raw, "products")
    if not isinstance(raw, dict):
        return None
    raw = dict(raw)
    raw["products"] = [p for p in as_list(raw.get("products")) if isinstance(p, dict)]
    if not raw["products"]:
        return None
    shopper_words = {g for g in grains(query) if len(g) > 3}
    wants = []
    for i, p in enumerate(raw["products"]):
        if not isinstance(p, dict):
            continue
        label = str(p.get("label") or p.get("kind") or f"product {i + 1}").strip()
        kind = str(p.get("kind") or label).strip()
        phrases = [str(p.get("search") or kind)]
        for alternative in as_list(p.get("alternatives")):
            if alternative:
                phrases.append(str(alternative))
        needs = []
        for raw_need in as_list(p.get("requirements")):
            for need in unfolded_needs(raw_need):
                if not unstatable(need):
                    needs.append(need)
        quantity = as_int(p.get("quantity"), 1) or 1
        wants.append(Want(label, kind, max(1, quantity), phrases, needs, shopper_words))
    if not wants:
        return None
    budget_raw = raw.get("budget") if isinstance(raw.get("budget"), dict) else {}
    if not budget_raw and raw.get("budget") not in (None, ""):
        budget_raw = {"amount": raw.get("budget")}   # a bare number or "450 cents"
    budget = number_in(budget_raw.get("amount")) if budget_raw else None
    decline = raw.get("decline") if isinstance(raw.get("decline"), dict) else {}
    if not decline and isinstance(raw.get("decline"), str):
        decline = {"rule": raw.get("decline")}
    objective = str(raw.get("objective") or "any").lower()
    return Plan(wants, objective if objective in ("cheapest", "best", "any") else "any",
                budget if budget and budget > 0 else None, str(budget_raw.get("unit_hint") or "unknown"),
                as_int(number_in(raw.get("message_cap"))), as_int(number_in(raw.get("step_cap"))),
                str(decline.get("rule") or raw.get("decline") or ""), raw.get("tools"),
                bool(as_bool(raw.get("notify_before_order"))))


# The request is read by code when the planner call fails.  Shoppers' requests are generated from a small set of
# sentence shapes ("It's X I want", "the material has to be Y", "Cross off anything with Z"); these patterns read
# the product kind and the stated requirements from them, never single words lifted out of the backstory.

_KIND_SHAPES = (
    r"\bit'?s (?:a |an |some )?(.+?) (?:that )?i(?:'m| am)? (?:want|after|wanting)\b",
    r"^(.+?) (?:are|is) what i'?m after",
    r"^(.+?) is the category",
    r"^(.+?) for me\b",
    r"\bi(?:'d| would)? (?:want|like|need)(?: to see)? (.+)",
    r"\bi'?m (?:after|wanting|hunting(?: for)?|looking for) (.+)",
    r"\bi'?ll start with (.+)",
    r"^(?:so )?the thing is (.+)",
    r"^(.+?) (?:are|is) the (?:thing|trail i'?m on)\b",
    r"\b(?:gonna |going to )?start with (.+)",
    r"\bgot (?:a |an )(.+?) hunt on my hands",
    r"\bhere'?s what i want: (.+)",
)
_KIND_STOP = r",| and it| and the | with | where | in (?:a |an |the )?| that | from | for | since | so | range\b| please\b| \(|$"
_NOT_A_KIND = r"\b(?:y'?all|on hand|actually|website|have|that'?d|something|anything|cts|cents|budget|price)\b"

_NEED_SHAPES = (   # (pattern, property group, value group)
    (r"\bthe ([a-z][a-z /\-]{1,30}?)(?:'s| has| have| must| should| is| needs)? ?(?:got to|has to|have to|must|should|needs to) "
     r"(?:be |read |measure |say |land on )(?:a |an )?([a-z0-9%+/.\- ]{1,30}?)"
     r"(?=,| nothing| or bigger| or larger| or more| that'?s| \(| on the | so | and |$)", 1, 2),
    (r"\bit'?s got to (?:be|come in) ([a-z0-9% \-]+?)(?=,| \(| no | and |$)()", 2, 1),
    (r"\bwith (?:a |an )?([a-z\-]+) (pattern|print|finish|sleeves?)\b", 2, 1),
    (r", ([a-z\-]+) (style|fit|cut)\b", 2, 1),
    (r"\bit'?s got to be ([a-z0-9% \-]+?), as far as ([a-z ]+?) go", 2, 1),
    (r"\bit'?s got to be ([a-z0-9% \-]+?), that'?s the ([a-z]+)", 2, 1),
    (r"\bwith (?:a |an )?([a-z0-9 \-]+?) as the ([a-z ]+)", 2, 1),
    (r"\bwhere the ([a-z ]+?) is ([a-z0-9 \-]+)", 1, 2),
    (r"\bthe ([a-z ]+?) on it is ([a-z0-9 \-]+)", 1, 2),
    (r"\bin (?:a |an |the )?([a-z\- ]+?) (style|fit|pack type|colou?r|material)\b", 2, 1),
    (r"\bit'?s the ([a-z\- ]+?) ([a-z]+ type) i want", 2, 1),
    (r"\bwith (?:a |an )([a-z\-]+) (frame|collar|neck|heel|sole)\b", 2, 1),
    (r"\b([a-z]+) is the colou?r\b", None, 1),
    (r"\bfor (?:a |an )([a-z]+) (occasion)\b", 2, 1),
    (r"\bi'?ll take ([a-z\-]+) only\b()", 2, 1),
    (r"\bi (?:want|need|'d like) [a-z' ]+? in ([a-z]+)(?=,|\.| that|$)()", 2, 1),
    (r"\bit'?s (?:a |an )?[a-z' ]+? in ([a-z]+) that i'?m? (?:want|after)()", 2, 1),
    (r"\bwhere the ([a-z ]+?) (?:are|go(?:es)? up to|reads?) ([a-z0-9+/.\- ]+?)(?=,|\.|$)", 1, 2),
)


def sentences(query: str) -> list[str]:
    out = []
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z])", query.replace("\u2019", "'")):
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


def request_kind(query: str) -> Optional[str]:
    """The product kind the request names, read from its 'I want X' sentence shapes."""
    for sentence in sentences(query):
        low = sentence.lower().rstrip(".!? ")
        matches = []
        for order, shape in enumerate(_KIND_SHAPES):
            hit = re.search(shape, low)
            if hit:
                matches.append((hit.start(), order, hit))
        matches.sort(key=lambda m: m[:2])
        for _start, _order, found in matches:
            kind = found.group(1).strip()
            for _ in range(3):   # leading fillers first, then the clause boundary
                kind = re.sub(r"^(?:a pair of|a|an|some|something from|anything from|the|your)\s+", "", kind)
            kind = re.split(_KIND_STOP, kind, maxsplit=1)[0].strip(" ,.")
            if 2 < len(kind) <= 40 and not re.search(r"\b(?:it|this|that|one|something)$", kind) \
                    and not re.search(_NOT_A_KIND, kind):
                return kind
    return None


def request_needs(query: str) -> list[Need]:
    """Requirements stated in the request's usual shapes: 'the X has to be Y', 'Cross off anything with X Y', …"""
    needs: list[Need] = []
    for sentence in sentences(query):
        low = sentence.lower().rstrip(".!? ")
        cross = re.search(r"cross off anything with ([a-z0-9 \-]+)", low)
        if cross:
            words = cross.group(1).split()
            cut = len(words) - 1
            if len(words) > 2 and words[-2].isdigit():
                cut -= 1
            subject = " ".join(words[:cut]) or "any"
            excluded_value = " ".join(words[cut:])
            needs.append(Need(subject, excluded_value, "not"))
            continue
        for shape, prop_group, value_group in _NEED_SHAPES:
            for found in re.finditer(shape, low):
                prop = (found.group(prop_group).strip() if prop_group else "colour") or "any"
                value = re.sub(r"^(?:a|an|the)\s+", "", found.group(value_group).strip(" ,."))
                if value in ("that", "this", "it", "any", "one", "a", "an"):
                    continue
                after = low[found.end():found.end() + 12]
                op = "min" if re.match(r"\s*or (?:bigger|larger|more|higher)", after) else "is"
                need = Need(prop, value, op)
                if value and not unstatable(need) and need not in needs:
                    needs.append(need)
    for alternatives in re.finditer(r"\bon ([a-z ]+?), it'?s ([a-z ]+?) or ([a-z ]+?), nothing else", query.lower()):
        for value in alternatives.groups()[1:]:
            needs.append(Need(alternatives.group(1), value.strip()))
    for excluded in re.finditer(r",? so nothing ([a-z\-]+)\b", query.lower()):
        if needs:   # 'for a formal occasion, so nothing casual': the exclusion rides on the need just stated
            last = needs[-1]
            needs.append(Need(last.prop, excluded.group(1), "not"))
    return needs


def head_words(kind: str) -> list[str]:
    """The content words of a product kind ('jam or marmalade' -> jam, marmalade)."""
    out = []
    for word in re.findall(r"[a-z][a-z\-/']+", kind.lower()):
        if word not in ("or", "and", "of", "for"):
            out.append(word)
    return out


def keyword_phrases(query: str) -> list[str]:
    """Search phrases read from the request without a model: the kind, double-quoted phrases, then the noun phrase
    after an article; apostrophes never delimit a phrase."""
    phrases: list[str] = []

    def add(text: str) -> None:
        text = " ".join(text.lower().split())
        if len(text) >= 3 and text not in phrases and len(text) <= 60:
            phrases.append(text)

    kind = request_kind(query)
    if kind:
        add(kind)
        heads = head_words(kind)
        if len(heads) > 1:
            add(heads[-1])
        for part in kind.split("/") if "/" in kind else ():
            add(part)   # 'cardigan/sweater': each spelling is a search of its own
    for quoted in re.findall(r"[\"\u201c]([^\"\u201d]{3,60})[\"\u201d]", query):
        add(quoted)
    if not phrases:
        for phrase in re.findall(r"\b(?:a|an|some)\s+([A-Za-z][A-Za-z\-]{2,}(?:\s+[A-Za-z][A-Za-z\-]{2,})?)", query):
            add(phrase)
    return phrases or [query[:60]]


def value_phrases(want: Want) -> list[str]:
    """The planner's phrases, with 'kind + required value' phrases right after the first one.

    Keyword search ranks titles and categories, so a property value the shopper requires (a wish included) is worth
    a search of its own next to the kind word ('maternity bottoms formal'); results come cheapest first either way.
    """
    phrases = list(want.phrases[:1])
    heads = head_words(want.kind)
    for need in want.needs:
        if need.op != "is" or len(need.value) > 30:
            continue
        for candidate in (f"{want.kind} {need.value}", f"{heads[-1]} {need.value}" if heads else "",
                          f"{need.value} {heads[0]}" if heads else ""):
            if not candidate:
                continue
            seen_before = False
            for known in phrases + want.phrases:
                if known.lower() == candidate.lower():
                    seen_before = True
            if not seen_before:
                phrases.append(candidate)
    for phrase in want.phrases[1:]:
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def improvised_plan(query: str) -> Plan:
    """Without any model: the kind and stated requirements read from the request's sentence shapes, the budget
    from the number next to a currency word, caps from 'N messages' / 'N steps'."""
    low = query.lower()
    phrases = keyword_phrases(query)
    kind = request_kind(query) or phrases[0]
    shopper_words = {g for g in grains(query) if len(g) > 3}
    needs = request_needs(query)
    want = Want(kind, kind, 1, phrases, needs, shopper_words)
    wants = listed_products(query, shopper_words) or [want]
    money = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:us\s+)?(cents?|cts|\u00a2|dollars?|euros?|rand|pesos?|pounds?)", low) \
        or re.search(r"[$\u20ac\u00a3]\s*(\d[\d,]*(?:\.\d+)?)()", low)
    if money:
        budget = float(money.group(1).replace(",", ""))
        hint = "cents" if re.match(r"c|\u00a2", money.group(2) or "") else "whole"
    else:
        # a bare number is a budget only in a sentence about money ('4 messages', '2 of them' never are)
        amounts = [n for sentence in sentences(query)
                   if re.search(r"spend|budget|afford|ceiling|price|cost|money|pay|over\b", sentence.lower())
                   for n in numbers_in(re.sub(r"\b\d+\s+(?:messages?|questions?|steps?|moves?|of them|pieces|pairs|sets)\b",
                                              "", sentence.lower()))]
        budget, hint = (max(amounts) if amounts else None), "unknown"
    words_to_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    cap = re.search(r"\b(\d+|one|two|three|four|five) (?:messages?|questions?)\b", low)
    message_cap = None
    if cap:
        spoken = cap.group(1)
        message_cap = int(spoken) if spoken.isdigit() else words_to_numbers[spoken]
    steps = re.search(r"\b(\d+) (?:steps|moves)\b", low)
    objective = "any"
    if "cheapest" in low:
        objective = "cheapest"
    elif re.search(r"\bthe best\b", low):
        objective = "best"
    rule = ""
    for sentence in sentences(query):
        if re.search(r"nothing found|come up empty|if nothing|nothing fits", sentence.lower()):
            rule = sentence
            break
    notify = bool(re.search(r"(?:tell|let me know|check with|ask|warn|message) me before (?:you )?(?:check|order|buy|pay|place)", low))
    return Plan(wants, objective, budget, hint, message_cap, int(steps.group(1)) if steps else None, rule, None, notify)


_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a pair of": 1}


def listed_products(query: str, shopper_words: set[str]) -> list[Want]:
    """'I need bed sheets in queen size, two sets of them, and a bra in black': one want per named product, its count
    and the plain 'in <value>' requirement riding on it.  Only a list of two to four short pieces counts."""
    for sentence in sentences(query):
        low = sentence.lower().rstrip(".!? ")
        found = re.search(r"\b(?:i need|i want|i'm after|i am after|i'm looking for|i am looking for|looking for|get me|"
                          r"i'd like|i would like)\s+(.+)", low)
        if not found:
            continue
        listed = re.split(r"\s+(?:that|which|so|because|as|from your|for my|for the|to)\s+|:", found.group(1))[0]
        pieces = [p.strip() for p in re.split(r"[,;]\s*(?:and\s+|plus\s+)?|\s+and\s+|\s+plus\s+", listed) if p.strip()]
        if not pieces or any(re.search(r"\b(?:i|i'm|my|your|range|it|them|me|you|has|have|is|are|be|was|must|"
                                            r"should|will|ready|also)\b", p) and
                                  not re.match(r"(?:two|three|four|five|six|\d+) (?:sets|pairs|packs|units|pieces|of)\b", p)
                                  for p in pieces):
            continue
        wants: list[Want] = []
        for piece in pieces:
            count = re.match(r"(two|three|four|five|six|\d+) (?:sets|pairs|packs|units|pieces|of)\b", piece)
            if count and wants:   # 'two sets of them' counts the product before it
                spoken = count.group(1)
                wants[-1].quantity = int(spoken) if spoken.isdigit() else _COUNTS[spoken]
                continue
            qty = 1
            lead = re.match(r"(a pair of|an?|one|two|three|four|five|six|\d+|some)\s+", piece)
            if lead:
                word = lead.group(1)
                qty = int(word) if word.isdigit() else _COUNTS.get(word, 1)
                piece = piece[lead.end():]
            kind, _sep, rest = re.split(r"\s+(?=(?:in|with|that|which|for|from)\b)", piece, maxsplit=1)[0], "", ""
            rest = piece[len(kind):].strip()
            kind = kind.strip(" ,.")
            if not (1 <= len(kind.split()) <= 3) or re.search(_NOT_A_KIND, kind) or len(kind) < 3:
                return []
            needs = []
            value = re.match(r"in ([a-z0-9\-]+)(?: (size|colou?r))?$", rest)
            if value:
                # no property named: the value is matched through whichever listing fact carries it
                prop = value.group(2) or "any"
                needs.append(Need(prop.replace("colour", "color"), value.group(1)))
            heads = head_words(kind)
            phrases = [kind] + ([heads[-1]] if len(heads) > 1 else [])
            wants.append(Want(kind, kind, qty, phrases, needs, shopper_words))
        if 2 <= len(wants) <= 4 or (len(wants) == 1 and wants[0].quantity > 1):
            return wants
    return []


# --------------------------------------------------------------------------------------------------------------------
# Allocation: the shared budget split into per-product ceilings, rebalanced as fits are confirmed.
# --------------------------------------------------------------------------------------------------------------------

class Allocation:
    def __init__(self, budget: Optional[float], labels: list[str], quantities: dict[str, int]) -> None:
        self.budget = budget                      # in listing units, None when the shopper named none
        self.labels = labels
        self.quantities = quantities
        self.weights: dict[str, float] = dict.fromkeys(labels, 1.0)
        self.fixed: dict[str, float] = {}         # label -> confirmed per-unit price of its fit

    def weigh(self, label: str, typical_price: Optional[float]) -> None:
        if typical_price and typical_price > 0:
            self.weights[label] = typical_price * self.quantities[label]

    def confirm(self, label: str, unit_price: Optional[float]) -> None:
        if unit_price is None:
            self.fixed.pop(label, None)
        else:
            self.fixed[label] = unit_price

    def spent(self) -> float:
        return sum(p * self.quantities[l] for l, p in self.fixed.items())

    def ceiling(self, label: str) -> Optional[float]:
        """Per-unit ceiling for a product: its weighted share of what the confirmed fits leave over."""
        if self.budget is None:
            return None
        if label in self.fixed:
            return (self.budget - self.spent() + self.fixed[label] * self.quantities[label]) / self.quantities[label]
        open_labels = [l for l in self.labels if l not in self.fixed]
        total_weight = sum(self.weights[l] for l in open_labels) or 1.0
        remaining = self.budget - self.spent()
        return max(0.0, remaining * self.weights[label] / total_weight) / self.quantities[label]

    def slack(self, label: str) -> Optional[float]:
        """The hard per-unit limit: everything the other confirmed fits leave, regardless of weights."""
        if self.budget is None:
            return None
        others = sum(p * self.quantities[l] for l, p in self.fixed.items() if l != label)
        return (self.budget - others) / self.quantities[label]


SCALES = (1.0, 100.0, 0.01)   # same unit, shopper whole / listings cents, shopper cents / listings whole


def infer_scale(budget: Optional[float], hint: str, typical_total: Optional[float], fractional: bool = False) -> float:
    """Multiplier from the shopper's unit to listing units: 1, 100 (shopper whole, listings cents) or 0.01.

    The shopper's own unit word is trusted against the shape of the prices, never against their size: search pools
    mix cheap accessories with the real thing, so a median can be far from the budget in either direction.  Cents
    against whole-number prices is 1; cents against prices with decimals is 0.01; whole currency against
    whole-number prices is 100 only when the budget would otherwise sit far below them.  Without a unit word the
    sane-ratio rule decides.
    """
    if not budget or not typical_total:
        return 1.0
    if hint == "cents":
        return 0.01 if fractional else 1.0
    if hint == "whole":
        if fractional:
            return 1.0
        return 100.0 if budget / typical_total < 0.2 and 0.2 <= budget * 100 / typical_total <= 30 else 1.0
    for scale in SCALES:
        ratio = budget * scale / typical_total
        if 0.2 <= ratio <= 30:
            return scale
    return 1.0


# --------------------------------------------------------------------------------------------------------------------
# Ladder: one product's candidates in price order.
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class Rung:
    listing: Listing
    read: bool = False
    fit: Optional[bool] = None
    dead: bool = False
    quotes: dict[int, str] = field(default_factory=dict)   # need index -> property name the reader quoted
    kind_vote: Optional[bool] = None                       # the reader's category verdict
    kind_ok: Optional[bool] = None                         # code's final word on the category
    verdicts: dict[int, Optional[bool]] = field(default_factory=dict)   # need index -> held / failed / unstated
    firm_on: set[int] = field(default_factory=set)         # needs held by the shopper's own property name
    firm: bool = False                                     # a fit the grader cannot dispute

    @property
    def price(self) -> Optional[float]:
        return self.listing.price


class Ladder:
    def __init__(self, want: Want) -> None:
        self.want = want
        self.rungs: dict[str, Rung] = {}
        self.order: list[str] = []            # arrival order, for ties and unpriced rows
        self.phrase_index = 0                 # which search phrase is in use
        self.page = 0                         # pages fetched for the current phrase
        self.cursor: Any = None               # paging token returned by the last result
        self.page_rows = 0                    # rows a full page carries, for offset paging
        self.browse: Optional[str] = None     # the enum department being listed, "done" once it ran dry
        self.filtered = False                 # a price-ceiling search was already issued
        self.exhausted = False
        self.dominant_category = ""
        self.asked: set[str] = set()          # properties already asked about (never repeated)
        self.soft: set[int] = set()           # needs no judged listing can state: they stop vetoing
        self.swept = False                    # the department was enumerated below the fit's price
        self.kind_seen = False                # some listing's category names the shopper's kind
        self.priority: set[str] = set()       # rows a search naming a required value returned
        self.keyword_swept = False            # the phrases were searched again below the fit's price
        self.coverage = 0                     # most kind words any seen category carries

    # -- supply ---------------------------------------------------------------------------------------------------

    def absorb(self, listings: list[Listing]) -> int:
        fresh = 0
        for item in listings:
            rung = self.rungs.get(item.ref)
            if rung is None:
                self.rungs[item.ref] = Rung(item)
                self.order.append(item.ref)
                fresh += 1
            else:
                rung.listing.merge(item)
        cats = Tally(r.listing.category for r in self.rungs.values() if r.listing.category)
        if cats:
            (self.dominant_category, _count), = cats.most_common(1)
        self.kind_seen = any(kind_fits(self.want.kind, c) for c in cats)
        self.coverage = max([category_coverage(self.want.kind, c) for c in cats] or [0])
        return fresh

    def sorted_rungs(self) -> list[Rung]:
        priced = [self.rungs[r] for r in self.order if self.rungs[r].price is not None]
        unpriced = [self.rungs[r] for r in self.order if self.rungs[r].price is None]
        return sorted(priced, key=lambda r: r.price) + unpriced

    def live(self) -> list[Rung]:
        """Rungs that can still be bought; a listing shown out of stock is never one, whatever the request says."""
        return [r for r in self.sorted_rungs() if not r.dead and r.listing.stock is not False]

    def affordable_unread(self, ceiling: Optional[float]) -> list[Rung]:
        """Unread rungs under the ceiling: rows a value-naming search returned first, each group cheapest first."""
        out = []
        for r in self.live():
            if r.read or r.price is None:
                continue
            if ceiling is not None and r.price > ceiling + 1e-9:
                break
            out.append(r)
        named = [r for r in out if r.listing.ref in self.priority]
        return named + [r for r in out if r.listing.ref not in self.priority]

    def unpriced(self) -> list[Rung]:
        return [r for r in self.live() if r.price is None]

    def typical_price(self) -> Optional[float]:
        prices = []
        for r in self.rungs.values():
            if not r.dead and r.price is not None:
                prices.append(r.price)
        return median(prices)

    # -- verdicts ------------------------------------------------------------------------------------------------

    def fits(self, ceiling: Optional[float]) -> list[Rung]:
        out = []
        for r in self.live():
            if r.fit and r.price is not None and (ceiling is None or r.price <= ceiling + 1e-9):
                out.append(r)
        return out

    def best_fit(self, ceiling: Optional[float], objective: str) -> Optional[Rung]:
        """Firm fits before provisional ones; among those, the cheapest, except that within 5 % of it a rung that
        states a soft need (or, for 'best'/'any', carries more of the shopper's words) wins."""
        fits = self.fits(ceiling)
        if not fits:
            return None
        firm = [r for r in fits if r.firm]
        pool = firm or fits
        # wishes rank before price: a fit that states the wish beats one silent on it, which beats one against it
        top = max(self.wish_tier(r) for r in pool)
        pool = [r for r in pool if self.wish_tier(r) == top]
        # at equal coverage, the aisle made only of the product's own words
        pure = [r for r in pool if category_purity(self.want.kind, r.listing.category)]
        pool = pure or pool
        cheapest = pool[0]
        window = [r for r in pool if r.price <= cheapest.price * (1 + BEST_WINDOW) + 1e-9]
        words = (lambda r: 0) if objective == "cheapest" else self.word_score
        best = window[0]
        for r in window[1:]:
            if (self.soft_hits(r), words(r), -r.price) > (self.soft_hits(best), words(best), -best.price):
                best = r
        return best

    def wish_tier(self, rung: Rung) -> int:
        """Sum over the wishes: 2 when the listing states it, 1 when silent, 0 when it states something else."""
        tier = 0
        for i, n in enumerate(self.want.needs):
            if n.soft:
                verdict = rung.verdicts.get(i)
                tier += 2 if verdict is True else (1 if verdict is None else 0)
        return tier

    def soft_hits(self, rung: Rung) -> int:
        """How many wishes and unstated needs this rung actually speaks to."""
        hits = 0
        for i, v in rung.verdicts.items():
            if v is True and (i in self.soft or self.want.needs[i].soft):
                hits += 1
        return hits

    def waived(self, i: int) -> bool:
        """A need that does not veto: unstated everywhere so far, or a wish no read listing has met yet.  Once some
        right-kind listing states the wished value, the wish binds like any requirement."""
        if i in self.soft:
            return True
        if i >= len(self.want.needs) or not self.want.needs[i].soft:
            return False
        for rung in self.rungs.values():
            if rung.read and rung.kind_ok and rung.verdicts.get(i) is True:
                return False
        return True

    def word_score(self, rung: Rung) -> int:
        have = set()
        for v in rung.listing.props.values():
            have.update(grains(v))
        return len(have & self.want.shopper_words)

    def judge(self, rung: Rung) -> None:
        """Code decides: category for the kind, then every need literally, using reader quotes only as synonyms.

        Verdicts are recorded per need (True held, False failed, None unstated) so `close_books` can tell a need
        that listings fail from one that no listing can state at all.
        """
        kind = kind_fits(self.want.kind, rung.listing.category)
        if kind is True and self.coverage >= 2 and category_coverage(self.want.kind, rung.listing.category) < self.coverage:
            kind = False   # 'Casual & Fashion Watches' is not 'luxury formal watch' once 'Luxury & Formal Watches' is seen
        if kind is False and self.kind_seen:
            pass   # the shopper's kind exists as a category in this world: any other category is another kind
        elif kind is not True:
            # the shopper's word may not be the world's category word ('glasses' in 'Eyewear'): the reader's
            # verdict decides, and without a reader the category most of the search results share stands in
            if rung.kind_vote is not None:
                kind = rung.kind_vote
            else:
                kind = (rung.listing.category == self.dominant_category) if rung.listing.category else True
        rung.kind_ok = bool(kind)
        rung.verdicts, rung.firm_on = {}, set()
        for i, need in enumerate(self.want.needs):
            held, firm = need_verdict(need, rung.listing.props, rung.quotes.get(i), self.want.kind, rung.listing.category)
            rung.verdicts[i] = held
            if held and firm:
                rung.firm_on.add(i)
        # two 'is' needs on one property are alternatives the planner split ('closure = zipper' or 'button'):
        # one of them holding satisfies the pair
        for i, need in enumerate(self.want.needs):
            twins = [j for j, other in enumerate(self.want.needs) if j != i and other.op == need.op == "is"
                     and same_label(other.prop, need.prop)]
            if twins and any(rung.verdicts.get(j) for j in twins) and not rung.verdicts.get(i):
                rung.verdicts[i] = True
                if any(j in rung.firm_on for j in twins):
                    rung.firm_on.add(i)
        self.settle_fit(rung)

    def settle_fit(self, rung: Rung) -> None:
        """A fit has no failed need that is not waived; it is firm when every unwaived need rests on its own key."""
        ok = rung.kind_ok is not False
        firm = ok
        for i, v in rung.verdicts.items():
            if self.waived(i):
                continue
            if v is not True:
                ok = False
            if i not in rung.firm_on:
                firm = False
        wishes_met = all(rung.verdicts.get(i) is True for i, n in enumerate(self.want.needs) if n.soft)
        rung.fit, rung.firm = ok, ok and firm and wishes_met

    def close_books(self) -> None:
        """A need that no judged listing states (after a few reads) cannot be checked literally in this world, so
        it stops vetoing; listings that do state it are still preferred.  Then every read rung gets its fit."""
        read = [r for r in self.rungs.values() if r.read]
        right_kind = [r for r in read if r.kind_ok]
        for i, need in enumerate(self.want.needs):
            spoken = [r for r in right_kind if r.verdicts.get(i) is not None]
            if not spoken and len(right_kind) >= SOFT_AFTER and not need.heard:
                self.soft.add(i)
            elif spoken or need.heard:
                self.soft.discard(i)
        for rung in read:
            self.settle_fit(rung)

    def rejudge(self) -> None:
        for rung in self.rungs.values():
            if rung.read:
                self.judge(rung)
        self.close_books()

    def report(self, limit: int = 5) -> list[dict]:
        """Per-need verdicts of the cheapest rungs, for the trace: why nothing fit."""
        out = []
        for r in self.live()[:limit]:
            verdicts = {}
            for i, v in r.verdicts.items():
                if i < len(self.want.needs):
                    verdicts[self.want.needs[i].describe()] = v
            out.append({"ref": r.listing.ref, "price": r.price, "read": r.read, "kind": r.kind_ok, "firm": r.firm,
                        "needs": verdicts})
        return out

    def keys_carrying(self, value: str) -> list[str]:
        """Property names under which read listings show a value (what 'any = gray' is really about: color)."""
        keys: list[str] = []
        for rung in self.rungs.values():
            if not rung.read:
                continue
            for key, shown in rung.listing.props.items():
                if key not in keys and not variant_like(str(shown)) and value_carries(value, shown):
                    keys.append(key)
        return keys

    def drop_value_need(self, need: Need) -> bool:
        """Remove one need (and its alternatives on the same property); quotes are re-indexed by a rejudge."""
        if need.prop != "any":
            return self.drop_need(need.prop)
        before = len(self.want.needs)
        self.want.needs = [n for n in self.want.needs if n is not need]
        if len(self.want.needs) == before:
            return False
        self.soft.clear()
        for rung in self.rungs.values():
            rung.quotes.clear()
        return True

    def drop_need(self, prop: str) -> bool:
        """Remove every need on a property (by name tokens); quotes and verdicts are re-indexed by a rejudge."""
        before = len(self.want.needs)
        kept = []
        for n in self.want.needs:
            if same_label(n.prop, prop) or label_overlap(n.prop, prop) > 0:
                continue
            kept.append(n)
        self.want.needs = kept
        if len(self.want.needs) != before:
            self.soft.clear()
            for rung in self.rungs.values():
                rung.quotes.clear()
            return True
        return False

    def next_above(self, rung: Rung) -> Optional[Rung]:
        """The rung just above a given one in price order (read or not)."""
        live = self.live()
        for i, r in enumerate(live):
            if r is rung:
                return live[i + 1] if i + 1 < len(live) else None
        return None


# --------------------------------------------------------------------------------------------------------------------
# Broker: runs one task end to end.
# --------------------------------------------------------------------------------------------------------------------

class Audit:
    """The last look before an order goes out.

    Every cart line is held against the request once more: the listing's category must be the kind, no stated
    hard need may fail on its properties, the listing must not be shown out of stock, and the lines together must
    stay inside the budget.  Anything found is a reason; an ordinary order with reasons is not sent, a salvage
    order (a refused decline left no better move) goes out with the reasons on record.
    """

    def __init__(self, ladders: list, allocation: Any) -> None:
        self.by_label = {}
        for ladder in ladders:
            self.by_label[ladder.want.label] = ladder
        self.allocation = allocation

    def review(self, fits: dict) -> list[str]:
        reasons: list[str] = []
        spend = 0.0
        for label, rung in fits.items():
            ladder = self.by_label.get(label)
            if ladder is None:
                continue
            listing = rung.listing
            quantity = self.allocation.quantities.get(label, 1)
            if listing.price:
                spend += quantity * listing.price
            if rung.kind_ok is False:
                reasons.append(f"{label}: category {listing.category!r} is not {ladder.want.kind!r}")
            if listing.stock is False:
                reasons.append(f"{label}: {listing.ref} shows no stock")
            for index, need in enumerate(ladder.want.needs):
                if ladder.waived(index):
                    continue
                verdict = rung.verdicts.get(index)
                if verdict is False:
                    reasons.append(f"{label}: {need.describe()} fails on {listing.ref}")
                elif verdict is None and not rung.read:
                    reasons.append(f"{label}: {need.describe()} unchecked on unread {listing.ref}")
        limit = self.allocation.budget
        if limit is not None and spend > limit + 1e-6:
            reasons.append(f"cart {spend:g} over budget {limit:g}")
        return reasons


class WireDown(Exception):
    """The environment endpoint stopped answering; nothing more can be done for this task."""


class Broker:
    def __init__(self, problem: dict) -> None:
        env = problem.get("environment") or {}
        self.binding = dict(env.get("binding") or {})
        view = env.get("policy_view") or {}
        self.query = str(view.get("query") or "")
        self.opening = view.get("opening")
        self.tools = []
        for tool in view.get("tools") or []:
            if isinstance(tool, dict):
                self.tools.append(tool)
        self.max_steps = as_int(view.get("max_steps"), 40) or 40
        self.finished = bool(view.get("finished"))
        self.trace: list[dict] = []
        self.counter = Counter(self.max_steps, None)
        self.oracle = Oracle(self.counter, self.trace)
        self.lens = Lens()
        self.step = 0
        self.plan: Optional[Plan] = None
        self.desks: dict[str, Desk] = {}
        self.ladders: list[Ladder] = []
        self.allocation: Optional[Allocation] = None
        self.refused_carts: set[str] = set()
        self.cart_refs: dict[str, str] = {}   # label -> ref currently in the cart
        self.scale = 1.0                       # shopper units -> listing units
        self.last_cart_view: Any = None         # the cart as the add tool last showed it, when it does
        self.price_refresh = 0                 # price-moved refusals answered by re-holding the lines
        self.notified = False                  # the pre-order notice the request may demand was sent
        self.resent = 0                        # carts sent again after a budget or requirement change
        self.salvaged = False                  # the cart is a last resort after a refused decline
        self.bare_details = 0                  # details answers in a row that carried no property block
        self.last_rows: set[str] = set()       # listing refs the last search returned
        self.asked_now = 0                     # questions sent in the interview
        self.rephrased = 0                     # plain re-asks sent after a rephrase request
        self.heard: list[str] = []             # shopper words that arrived wrapped around an observation
        self.confirmed: Optional[str] = None   # the cart signature the shopper last confirmed
        self.kept: Optional[dict] = None       # picks to re-send unchanged after a loosened condition
        self.basket_stale = False              # the basket may not hold what cart_refs says: set it again
        self.swapped_details = False           # details and offer traded places once
        self.compare_thin = False              # compare rows carry no property block: read through details
        self.details_rich: Optional[bool] = None   # the details tool has shown a property block (or not)
        self.order_failures = 0                # orders answered with a failure code instead of a finish or refusal

    # -- environment plumbing ------------------------------------------------------------------------------------

    def call(self, desk: Desk, args: dict) -> Any:
        """One environment step; resyncs on 409 ('expected N'), retries briefly on transport failure.  The last
        step of the episode is kept for a finishing call: nothing else is sent once only one step is left."""
        if self.finished:
            return None
        finishing = desk.role in ("order", "decline")
        if self.counter.steps_left() <= 0 or (self.counter.steps_left() == 1 and not finishing):
            return None
        for attempt in range(4):
            body = dict(self.binding)
            body["step"] = self.step + 1
            body["action"] = {"name": desk.name, "args": args}
            # an unanswered call is re-sent under the same step number (the world replays a step it already ran)
            packet = on_the_clock(open_wire(STORE_PATIENCE), "/environment/call", body, STORE_PATIENCE)
            if packet.data is not None:
                self.step += 1
                self.counter.steps = self.step
                observation = packet.data.get("observation")
                self.finished = bool(packet.data.get("finished"))
                if not finishing:
                    observation, spoken = split_envelope(observation)
                    if spoken:
                        self.heard.extend(spoken)
                self.trace.append({"kind": "call", "step": self.step, "tool": desk.name, "args": args,
                                   "observation": observation, "finished": self.finished,
                                   "at": round(self.counter.elapsed(), 1)})
                if desk.role == "message":
                    self.counter.messages += 1
                repaired = self.repair(desk, args, observation)
                if repaired is not None and not self.finished:
                    return self.call(desk, repaired)
                return observation
            error = packet.error or {}
            expected = re.search(r"expected\s+(\d+)", str(error.get("body") or ""))
            if error.get("status") == 409 and expected:
                synced = int(expected.group(1))
                self.step = synced - 1
                continue
            self.trace.append({"kind": "wire-error", "tool": desk.name, "error": error})
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        raise WireDown(desk.name)

    @staticmethod
    def repair(desk: Desk, args: dict, observation: Any) -> Optional[dict]:
        """On an invalid-arguments error naming a missing parameter, fill it from the schema; never the same call."""
        if not (isinstance(observation, dict) and "invalid" in str(observation.get("error", "")).lower()):
            return None
        detail = str(observation.get("detail") or observation.get("error") or "")
        named = re.findall(r"\$\.?(\w+)|missing required (\w+)|'(\w+)'", detail)
        for group in named:
            for name in group:
                if name and name in desk.schema["properties"] and name not in args:
                    return {**args, name: placeholder(desk.schema["properties"][name])}
        return None

    @staticmethod
    def errored(observation: Any) -> bool:
        return isinstance(observation, dict) and ((bool(observation.get("error")) and len(observation) <= 2)
                                                  or failure_reply(observation))

    def desk(self, role: str) -> Optional[Desk]:
        return self.desks.get(role)

    def checkout_steps(self) -> int:
        """Steps one checkout needs: a cart line per product, the cart view, the confirmation and the order."""
        confirming = self.desk("message") is not None and self.counter.may_message()
        return len(self.ladders) + (3 if confirming else 2)

    def reserve(self) -> int:
        """Steps exploration leaves untouched: a checkout, one refusal cycle (lines out, prices re-read, lines in,
        view, order) and a closing decline, so a refused first order can still end in a finish."""
        base = len(self.ladders) + 2
        confirm = self.checkout_steps() - base   # the basket confirmation, while a message is left for it
        full = 2 * base + len(self.ladders) + 1
        if self.counter.max_steps >= 3 * full:
            return full + confirm
        return base + 1 + confirm   # a very small step cap: one checkout and the closing step

    def comfort(self) -> int:
        """What optional work (questions, extra reads) must leave untouched: the reserve and a question's worth."""
        return self.reserve() + 1

    # -- planning ---------------------------------------------------------------------------------------------------

    def make_plan(self) -> None:
        user = f"Request:\n{self.query}\n"
        if self.opening is not None:
            user += f"\nOpening observation:\n{squeeze(self.opening, 1500)}\n"
        user += f"\nTools:\n{squeeze([provider_safe(schema_of(t)) for t in self.tools], 7000)}"
        raw = self.oracle.ask("plan", PLANNER_SYSTEM, user, PLANNER_MODELS, tokens=2600)
        self.plan = plan_from_json(raw, self.query) or improvised_plan(self.query)
        if self.plan.step_cap:
            self.counter.max_steps = min(self.max_steps, self.plan.step_cap)
        self.counter.message_cap = self.plan.message_cap
        # the confirmation before the order keeps one message back; a cap of three or fewer is all questions (the
        # qualifying shape), and the confirmation then goes on a message the questions left
        cap = self.plan.message_cap
        self.counter.held_back = 1 if cap is not None and (cap >= 4 or self.plan.notify_first) else 0
        self.desks = desks_from_plan(self.tools, self.plan.tools)
        self.plan.wants = merged_departments(self.plan.wants, self.desk("filter"))
        for want in self.plan.wants:
            want.phrases = value_phrases(want)
        self.ladders = [Ladder(w) for w in self.plan.wants]
        self.trace.append({"kind": "plan", "products": [w.label for w in self.plan.wants],
                           "needs": dict((w.label, numbered_needs(w.needs)) for w in self.plan.wants),
                           "objective": self.plan.objective, "budget": self.plan.budget,
                           "roles": dict((r, d.name) for r, d in self.desks.items())})

    # -- searching ---------------------------------------------------------------------------------------------------

    def search(self, ladder: Ladder, phrase: str, ceiling: Optional[float] = None, paging: bool = False,
               department: Optional[str] = None, desk: Optional[Desk] = None) -> tuple[int, int]:
        """One search step; returns (rows returned, listings new to the ladder).

        With `department` the call is a filter-mode enumeration of that department (the mode enum set to its filter
        value, no keywords): keyword mode answers a department-only search with nothing.
        """
        desk = desk or self.desk("search")
        if desk is None:
            return 0, 0
        values: dict[str, Any] = {"query": phrase} if not department else {"department": department}
        if ceiling is not None and desk.param("price_max"):
            values["price_max"] = math.floor(ceiling)
            ladder.filtered = True
        page_param = desk.param("page")
        if paging and page_param:
            spec = desk.schema["properties"][page_param]
            if ladder.cursor is not None and spec.get("type") == "string":
                values["page"] = ladder.cursor
            elif spec.get("type") in ("integer", "number") and is_number(ladder.cursor):
                values["page"] = ladder.cursor   # the start the world itself handed back: never a page number
            elif spec.get("type") in ("integer", "number") and offset_param(page_param, spec):
                values["page"] = ladder.page * max(1, ladder.page_rows or 10)
            elif spec.get("type") in ("integer", "number"):
                base = as_int(spec.get("minimum"), as_int(spec.get("default"), 1)) or 1
                values["page"] = base + ladder.page
                top = spec.get("maximum")
                if is_number(top) and values["page"] > top:
                    return 0, 0
            else:
                return 0, 0
        args = desk.build(**values)
        if department and desk.param("mode") and desk.notes.get("filter_value") is not None:
            args[desk.param("mode")] = desk.notes["filter_value"]
            args.pop(desk.param("query"), None)
        observation = self.call(desk, args)
        if observation is None or self.errored(observation):
            return 0, 0
        rows = self.lens.rows_of(observation)
        self.lens.learn(rows)
        listings = []
        for r in rows:
            seen_row = self.lens.listing(r)
            if seen_row:
                listings.append(seen_row)
        ladder.page += 1
        fresh_ids = {l.ref for l in listings} - self.last_rows if paging else {l.ref for l in listings}
        self.last_rows = {l.ref for l in listings}
        ladder.page_rows = max(ladder.page_rows or 0, len(rows))
        ladder.cursor = self.paging_token(observation, page_param, desk)
        if paging and not fresh_ids:
            ladder.page = PAGE_LIMIT   # a page that adds no new ids ends the paging
        if self.last_page(observation, len(rows)):
            ladder.page = PAGE_LIMIT
        return len(rows), ladder.absorb(listings)

    @staticmethod
    def last_page(observation: Any, shown: int) -> bool:
        """The results are over when a next-page field is null or false, or a total count says this page held
        everything, whatever the page numbers would allow."""
        if not isinstance(observation, dict):
            return False
        flags = [v for k, v in observation.items() if not isinstance(v, (list, dict))
                 and (re.search(r"next|more|further|continu|cursor", k.lower()) or re.search(r"token|follow|after|marker", k.lower()))]
        if flags and all(not v for v in flags):
            return True
        totals = []
        for k, v in observation.items():
            if is_number(v) and re.search(r"total|count|match|found", k.lower()):
                totals.append(v)
        if not totals:
            return False
        return shown == 0 or min(totals) <= shown

    def page_cap(self, desk: Desk) -> int:
        page_param = desk.param("page")
        spec = desk.schema["properties"][page_param] if page_param else {}
        if is_number(spec.get("maximum")):
            base = as_int(spec.get("minimum"), as_int(spec.get("default"), 1)) or 1
            return max(1, min(PAGE_LIMIT, int(spec["maximum"]) - base + 1))
        return PAGE_LIMIT

    @staticmethod
    def paging_token(observation: Any, page_param: Optional[str], desk: Desk) -> Any:
        """A short string scalar in the result that is neither a row nor an id is the next-page token."""
        if not isinstance(observation, dict) or not page_param:
            return None
        if desk.schema["properties"][page_param].get("type") in ("integer", "number"):
            # a numeric next start ('onward_token': 20) is sent back as is; counts and totals are not starts
            for k, v in observation.items():
                if is_number(v) and re.search(r"token|next|onward|offset|cursor|start|continu|resume|follow", snake(k)) \
                        and not re.search(r"total|count|num|hits|matched|size|found", snake(k)):
                    return int(v)
            return None
        if desk.schema["properties"][page_param].get("type") != "string":
            return None
        scalars = {k: v for k, v in observation.items()
                   if isinstance(v, str) and v and " " not in v and len(v) < 80 and not hinted("ref", k)}
        for k, v in scalars.items():
            if re.search(r"page|cursor|token|next|more", k.lower()) or re.search(r"further|continu|follow|after|marker", k.lower()):
                return v
        return next(iter(scalars.values()), None)

    def resupply(self, ladder: Ladder, ceiling: Optional[float]) -> bool:
        """Find more candidates, one search step at a time, until a step brings a listing the ladder had not seen.

        Order: the next page of the current phrase while pages still return rows, then the same phrase under the
        schema's price ceiling (paged the same way), then each alternative phrase.  Returns False when the ladder
        is exhausted or no step can be spared.
        """
        desk = self.desk("search")
        if desk is None:
            ladder.exhausted = True
        page_param = desk.param("page") if desk else None
        numbered = bool(page_param) and desk.schema["properties"][page_param].get("type") in ("integer", "number")
        cap = self.page_cap(desk) if desk else 0
        while not ladder.exhausted and self.counter.room(self.reserve()):
            phrase = ladder.want.phrases[ladder.phrase_index]
            if page_param and 0 < ladder.page < cap and (numbered or ladder.cursor is not None):
                rows, fresh = self.search(ladder, phrase, ceiling if ladder.filtered else None, paging=True)
                if fresh:
                    return True
                if not rows:
                    ladder.page = PAGE_LIMIT
                continue
            if ceiling is not None and desk.param("price_max") and not ladder.filtered:
                ladder.page, ladder.cursor = 0, None
                if self.search(ladder, phrase, ceiling)[1]:
                    return True
                continue
            if ladder.phrase_index + 1 < len(ladder.want.phrases):
                ladder.phrase_index += 1
                ladder.page, ladder.cursor, ladder.filtered = 0, None, False
                if self.search(ladder, ladder.want.phrases[ladder.phrase_index],
                               ceiling if desk.param("price_max") else None)[1]:
                    return True
                continue
            if self.browse_department(ladder):
                return True
            ladder.exhausted = True
        return False

    def browse_department(self, ladder: Ladder) -> bool:
        """When keyword phrases run dry and a browse tool lists one department from an enum, list the department
        whose name shares the product's words, page by page with the start the world hands back."""
        desk = self.desk("filter")
        name = desk.param("department") if desk else None
        spec = desk.schema["properties"].get(name) if name else None
        if not spec or not is_enum(spec) or ladder.browse == "done":
            return False
        if ladder.browse is None:
            value = closest_enum(ladder.want.kind + " " + ladder.want.label, spec["enum"])
            if value is None:
                ladder.browse = "done"
                return False
            ladder.browse, ladder.page, ladder.cursor = value, 0, None
        while self.counter.room(self.reserve()) and ladder.page < PAGE_LIMIT:
            rows, fresh = self.search(ladder, "", None, paging=ladder.page > 0, department=ladder.browse, desk=desk)
            if not rows:
                break
            if fresh:
                return True
        ladder.browse = "done"
        return False

    # -- reading -----------------------------------------------------------------------------------------------------

    def price_rung(self, rung: Rung) -> None:
        desk = self.desk("offer")
        if desk is None:
            return
        observation = self.call(desk, desk.build(id=rung.listing.ref))
        if observation is None or self.errored(observation):
            rung.dead = True
            return
        seen = self.lens.listing(observation) if isinstance(observation, dict) else None
        if seen is None and isinstance(observation, dict):
            seen = Listing(rung.listing.ref, self.lens.price_of(observation), self.lens.stock_of(observation),
                           "", "", {}, observation)
        if seen is not None:
            rung.listing.merge(seen)
        if rung.listing.price is None:
            rung.dead = True

    def open_rungs(self, ladder: Ladder, rungs: list[Rung]) -> list[Rung]:
        """Fetch properties for the given rungs (compare when several, details otherwise) and judge them."""
        compare, details = self.desk("compare"), self.desk("details")
        opened: list[Rung] = []
        if len(rungs) > 1 and compare is not None and not self.compare_thin:
            observation = self.call(compare, compare.build(ids=[r.listing.ref for r in rungs]))
            rows = self.lens.rows_of(observation) if observation is not None else []
            # compare rows without any property block (race 7: the facts live on the details tool only) do not
            # count as read while the details tool has not shown itself bare too
            thin = bool(rows) and details is not None and self.details_rich is not False \
                and not any(facts_keys(row) for row in rows if isinstance(row, dict))
            if thin:
                self.compare_thin = True
                self.trace.append({"kind": "compare-thin", "tool": compare.name})
            for row in rows:
                seen = self.lens.listing(row)
                rung = ladder.rungs.get(seen.ref) if seen else None
                if rung is not None:
                    rung.listing.merge(seen)
                    if not thin:
                        rung.read = True
                        opened.append(rung)
            rungs = [r for r in rungs if r not in opened]
        for rung in rungs:
            if details is None:
                rung.read = True
                opened.append(rung)
                continue
            if not self.counter.room(self.reserve()) and opened:
                break
            observation = self.call(details, details.build(id=rung.listing.ref))
            if observation is None or self.errored(observation):
                rung.dead = True
                continue
            seen = self.lens.listing(observation) if isinstance(observation, dict) else None
            if seen is None and isinstance(observation, dict):
                seen = Listing(rung.listing.ref, self.lens.price_of(observation), self.lens.stock_of(observation),
                               "", "", self.lens.props_of(observation), observation)
            if seen is not None:
                rung.listing.merge(seen)
            if not facts_keys(observation if isinstance(observation, dict) else {}):
                self.bare_details += 1
                other = self.desk("offer")
                if self.bare_details >= 2 and other is not None and not self.swapped_details:
                    # twice no property block: the other one-id tool is the one that shows properties
                    self.desks["details"], self.desks["offer"] = other, details
                    details, self.bare_details, self.swapped_details = other, 0, True
                elif self.bare_details >= 2 and self.details_rich is None:
                    self.details_rich = False
                    self.compare_thin = False   # no tool shows more than the compare rows: back to batches
            else:
                self.bare_details = 0
                self.details_rich = True
            rung.read = True
            opened.append(rung)
        for rung in opened:
            ladder.judge(rung)
        # the reader is only worth a call where code could not decide: an unstated need or an unconfirmed kind
        unsure = []
        for r in opened:
            undecided = kind_fits(ladder.want.kind, r.listing.category) is not True
            for i, v in r.verdicts.items():
                if v is None and not ladder.want.needs[i].soft:
                    undecided = True
            if undecided:
                unsure.append(r)
        if unsure:
            if not self.oracle.lean:
                self.read_with_model(ladder, unsure)
            for rung in unsure:
                ladder.judge(rung)
        ladder.close_books()
        return opened

    def read_with_model(self, ladder: Ladder, rungs: list[Rung]) -> None:
        """Ask the reader which property each requirement rests on; keep the quotes only where they are real."""
        if not rungs or (not ladder.want.needs and all(
                kind_fits(ladder.want.kind, r.listing.category) is not None for r in rungs)):
            return
        for batch in reader_batches(rungs):
            user = (f"Product wanted: {ladder.want.kind}\nRequirements:\n"
                    + "\n".join(numbered_needs(ladder.want.needs))
                    + "\nListings:\n" + json.dumps([reader_row(r) for r in batch], ensure_ascii=False, default=str))
            raw = self.oracle.ask("read", READER_SYSTEM, user, READER_MODELS, tokens=2600)
            answer = unwrapped(raw, "verdicts")
            verdicts = as_list(answer.get("verdicts")) if isinstance(answer, dict) else []
            for verdict in verdicts:
                if not isinstance(verdict, dict):
                    continue
                rung = ladder.rungs.get(str(verdict.get("id")).strip())
                if rung is None:
                    continue
                vote = as_bool(verdict.get("kind_ok"))
                if vote is not None:
                    rung.kind_vote = vote
                for check in as_list(verdict.get("checks")):
                    if not isinstance(check, dict):
                        continue
                    idx, name = as_int(check.get("req")), check.get("property")
                    if idx is None or not isinstance(name, str) or idx >= len(ladder.want.needs):
                        continue
                    actual = next((k for k in rung.listing.props if same_label(k, name)), None)
                    if actual is None:
                        continue
                    # the quote must be true: the value the reader cites has to be the value the listing shows
                    quoted = str(check.get("value") or "")
                    if quoted and set(grains(quoted)) - set(grains(rung.listing.props[actual])):
                        continue
                    rung.quotes[idx] = actual

    # -- climbing ---------------------------------------------------------------------------------------------------

    def ceiling_for(self, ladder: Ladder) -> Optional[float]:
        assert self.allocation is not None
        return self.allocation.ceiling(ladder.want.label)

    def climb(self, ladder: Ladder) -> Optional[Rung]:
        """Open rungs from the cheapest up until a fit (plus its 5 % window for 'best'); resupply when dry.

        The ladder first works under the product's weighted share of the budget; when that runs dry the ceiling
        relaxes to the slack the other confirmed fits leave, so a product that costs more than its share can still
        be bought when the whole cart fits.
        """
        plan, allocation = self.plan, self.allocation
        assert plan is not None and allocation is not None
        label = ladder.want.label
        relaxed = False
        extra = 0
        while True:
            ceiling = allocation.slack(label) if relaxed else allocation.ceiling(label)
            fit = ladder.best_fit(ceiling, plan.objective)
            unread = ladder.affordable_unread(ceiling)
            if fit is not None and self.counter.late():
                return fit   # past the cart deadline any candidate is the candidate
            cheaper = [r for r in unread if fit is not None and r.price < fit.price]
            if cheaper and self.counter.room(self.reserve()):
                # a sweep can inject rungs below the current fit: they are opened before the fit is trusted
                self.open_rungs(ladder, cheaper[:COMPARE_BATCH] if self.desk("compare") is not None else cheaper[:1])
                continue
            batch_size = COMPARE_BATCH if self.desk("compare") is not None else 1
            if fit is not None and not fit.firm and unread and extra < PROVISIONAL_EXTRA * batch_size \
                    and self.counter.room(self.comfort()):
                # the fit rests on kindred keys, waived needs or an unmet wish: a few more rungs may hold a firm one
                extra += len(unread[:batch_size])
                self.open_rungs(ladder, unread[:batch_size])
                continue
            if fit is not None and not fit.firm and not unread and extra < PROVISIONAL_EXTRA * batch_size \
                    and self.counter.room(self.comfort()):
                if ladder.page >= 2:
                    ladder.page = PAGE_LIMIT   # two pages of one phrase is enough: the next phrase names the value
                if self.resupply(ladder, ceiling):
                    extra += 1   # the next search is part of the hunt for a firm fit
                    continue
            if fit is not None:
                if plan.objective == "cheapest":
                    return fit
                window = [r for r in unread if r.price <= fit.price * (1 + BEST_WINDOW) + 1e-9]
                if not window or not self.counter.room(self.reserve()):
                    return fit
                unread = window
            if not self.counter.room(self.reserve()):
                return fit
            if not unread:
                unpriced = ladder.unpriced()
                if unpriced and self.desk("offer") is not None:
                    self.price_rung(unpriced[0])
                    continue
                if self.resupply(ladder, ceiling):
                    continue
                slack = allocation.slack(label)
                if not relaxed and ceiling is not None and slack is not None and slack > ceiling + 1e-9:
                    relaxed, ladder.exhausted, ladder.filtered = True, False, False
                    continue
                return fit
            batch = unread[:COMPARE_BATCH] if self.desk("compare") is not None else unread[:1]
            self.open_rungs(ladder, batch)

    def observe_prices(self) -> None:
        """One cheap search per product, offers for a few unpriced rows, then the scale and the allocation."""
        plan = self.plan
        assert plan is not None
        for ladder in self.ladders:
            if not self.counter.room(self.reserve()):
                break
            self.search(ladder, ladder.want.phrases[0])
            self.value_search(ladder)
            probes = 0
            for rung in ladder.unpriced():
                if probes >= OFFER_PROBE or self.desk("offer") is None or not self.counter.room(self.reserve()):
                    break
                self.price_rung(rung)
                probes += 1
        typical = {l.want.label: l.typical_price() for l in self.ladders}
        total = 0.0
        fractional = False
        for l in self.ladders:
            usual = typical[l.want.label]
            if usual:
                total += usual * l.want.quantity
            for r in l.rungs.values():
                if r.price is not None and abs(r.price - round(r.price)) > 1e-9:
                    fractional = True
        total = total or None
        scale = infer_scale(plan.budget, plan.unit_hint, total, fractional)
        self.scale = scale
        budget = plan.budget * scale if plan.budget else None
        self.allocation = Allocation(budget, [l.want.label for l in self.ladders],
                                     {l.want.label: l.want.quantity for l in self.ladders})
        for ladder in self.ladders:
            self.allocation.weigh(ladder.want.label, typical[ladder.want.label])
        self.trace.append({"kind": "allocation", "scale": scale, "budget": budget, "typical": typical})

    def run_ladders(self) -> dict[str, Rung]:
        """Climb every ladder, confirming fits into the allocation so later ladders see the freed budget."""
        assert self.allocation is not None
        fits: dict[str, Rung] = {}
        pending = list(self.ladders)
        for _round in range(3):
            still: list[Ladder] = []
            for ladder in pending:
                rung = self.climb(ladder)
                if rung is not None and self.plan is not None and self.plan.objective == "cheapest":
                    rung = self.sweep_cheaper(ladder, rung)
                if rung is not None:
                    fits[ladder.want.label] = rung
                    self.allocation.confirm(ladder.want.label, rung.price)
                else:
                    self.allocation.confirm(ladder.want.label, None)
                    still.append(ladder)
            # another pass only helps when a ceiling moved: a fit confirmed this round frees budget for the rest
            if not still or len(still) == len(pending) or self.allocation.budget is None:
                break
            for ladder in still:
                ladder.exhausted = False
            pending = still
        return fits

    def extras_affordable(self) -> bool:
        """Optional searches (the value-naming search, the keyword sweep) run only with steps for every ladder's
        climb left over and inside the first quarter of the cart deadline (questions come before them)."""
        return self.counter.room(self.comfort() + 2 * len(self.ladders)) and self.counter.elapsed() < CART_DEADLINE / 4

    def value_search(self, ladder: Ladder) -> None:
        """One search naming the product and a required value ('bomber jacket military'): keyword engines need
        every word, so its rows are the ones most likely to state the value; they are opened first."""
        for need in ladder.want.needs:
            if need.op != "is" or need.soft or len(need.value) > 30:
                continue
            if attribute_words(need.value, set()) <= attribute_words(ladder.want.kind, set()):
                continue   # the value only restates the kind
            if attribute_words(need.value, set()) <= attribute_words(ladder.want.phrases[0], set()):
                return     # the first search already named the value
            phrase = f"{ladder.want.kind} {need.value}"
            if not self.extras_affordable():
                return   # extras only when the step budget leaves room for every ladder's climb
            paging = (ladder.page, ladder.cursor)   # the first phrase's paging continues afterwards
            rows, _fresh = self.search(ladder, phrase)
            ladder.page, ladder.cursor = paging
            if rows:
                ladder.priority.update(self.last_rows)
            return

    def sweep_cheaper(self, ladder: Ladder, fit: Rung) -> Rung:
        """Keyword search shows a slice of a department sorted by id, so the cheapest fit may never surface.  When
        the search tool has a department filter, enumerate the fit's category under a ceiling just below the fit's
        price (filter mode, no keywords) and climb again; repeat while the price keeps dropping."""
        desk = next((d for d in (self.desk("filter"), self.desk("search"))
                     if d is not None and d.param("department") and d.param("price_max")), None)
        if fit.price is None:
            return fit
        if desk is None:
            return self.keyword_sweep(ladder, fit)
        category = fit.listing.category or ladder.dominant_category
        if not category or ladder.swept or self.counter.late():
            return fit
        for _round in range(SWEEP_ROUNDS):
            if not self.counter.room(self.comfort()):
                break
            step_down = 1 if float(fit.price).is_integer() else 0.01
            ceiling = fit.price - step_down
            ladder.page, ladder.cursor = 0, None
            rows, fresh = self.search(ladder, "", ceiling, department=category, desk=desk)
            cap = self.page_cap(desk)
            while rows and ladder.page < cap and self.counter.room(self.comfort()):
                more = self.search(ladder, "", ceiling, paging=True, department=category, desk=desk)
                more_rows, more_fresh = more
                rows, fresh = more_rows, fresh + more_fresh
            if not fresh:
                break
            ladder.exhausted = False
            better = self.climb(ladder)
            if better is None or better.price is None or better.price >= fit.price:
                break
            fit = better
        ladder.swept = True
        return self.keyword_sweep(ladder, fit)

    def keyword_sweep(self, ladder: Ladder, fit: Rung) -> Rung:
        """For a cheapest task: the product's own phrases again under a price cap just below the best fit, while
        the price keeps dropping (a keyword engine shows only a slice, and the cheaper fit may sit outside it)."""
        search = self.desk("search")
        if search is None or not search.param("price_max") or ladder.keyword_swept or self.counter.late():
            return fit
        if not self.extras_affordable():
            return fit
        ladder.keyword_swept = True
        for phrase in ladder.want.phrases[:2]:
            if fit.price is None or not self.counter.room(self.comfort()):
                break
            step_down = 1 if float(fit.price).is_integer() else 0.01
            ladder.page, ladder.cursor = 0, None
            _rows, fresh = self.search(ladder, phrase, fit.price - step_down)
            if not fresh:
                continue
            ladder.exhausted = False
            better = self.climb(ladder)
            if better is not None and better.price is not None and better.price < fit.price:
                fit = better
        return fit

    # -- hidden needs -----------------------------------------------------------------------------------------------

    def question_for(self, ladder: Ladder, fit: Rung) -> Optional[tuple[str, list[str]]]:
        """The property most worth one message: never a stated need or a property already asked (aliases included).

        Candidates come from every read listing, not only the fit: a hidden need often sits on a property the
        cheapest fit does not even state ('style' on a shirt whose facts never mention it), and the answer then
        moves the pick.  Ranked by: the fit and the rung above disagree on it (or the fit lacks it), how many keys of
        the listings repeat it under other spellings (worlds duplicate the properties that matter), how many read
        listings state it, and how few distinct values it takes.  Row-level fields beside a facts block, seller
        variant strings and per-listing identifiers (every value different) are never asked.
        """
        plan = self.plan
        assert plan is not None
        above = ladder.next_above(fit)
        unopened_above = above is not None and not above.read and self.desk("details") is not None
        if unopened_above and self.counter.room(self.comfort() + 1):
            self.open_rungs(ladder, [above])
        ranked = self.question_menu(ladder, fit, above)
        if not ranked:
            return None
        self.trace.append({"kind": "candidates", "product": ladder.want.label,
                           "ranked": [[c[1], list(c[0])] for c in ranked[:8]]})
        _, name, distinct, own = ranked[0]
        examples = []
        if own not in (None, ""):
            examples.append(str(own))
        for v in distinct:
            if v != str(own) and len(examples) < 2:
                examples.append(v)
        return name, examples

    @staticmethod
    def question_menu(ladder: Ladder, fit: Rung, above: Optional[Rung]) -> list:
        """Every askable property, best first: (score, name, distinct values, the fit's value).

        Spellings of one attribute ('pattern' / 'fa_pattern', 'material' / 'clothing.material') are one menu entry:
        two names belong together when the words of one, minus the product's own words and two-letter prefixes, are
        among the other's ('pack size' and 'pack type' stay apart).  Entries rank by MENU_ORDER: the share of read
        listings that state the attribute (in quarters), how many spellings the world gives it, how few values it
        takes, how many titles echo its value, the exact count, and whether the fit and the rung above disagree.
        Measured on 60 graded practice menus (tests/menus.py): the best hidden property is first in 26 menus (was
        19) and in the top three in 48 (was 35).  Row-level fields beside a facts block, seller variant strings,
        single-valued attributes the fit states and per-listing identifiers are never asked.
        """
        product = attribute_words(ladder.want.kind + " " + ladder.want.label, set())
        # a requirement that only restates the kind ('type: UHT milk') settles no attribute family
        kind_words = product | attribute_words(ladder.dominant_category, set())
        taken = [n.prop for n in ladder.want.needs
                 if not (attribute_words(n.value, set()) and attribute_words(n.value, set()) <= kind_words)]
        taken += sorted(ladder.asked)
        taken_words = [attribute_words(t, product) for t in taken]
        read = [r for r in ladder.rungs.values() if r.read and r.kind_ok is not False]
        stated: dict[str, list[tuple[Rung, str]]] = {}
        for rung in read:
            facts = facts_keys(rung.listing.raw)
            for name, value in rung.listing.props.items():
                if facts and name not in facts:
                    continue
                shown = "" if value is None else str(value)
                if not shown or len(shown) > 40 or variant_like(shown):
                    continue
                stated.setdefault(name, []).append((rung, shown))
        entries: list[dict] = []
        for name in sorted(stated, key=lambda n: (-len(stated[n]), len(n), n)):
            words = attribute_words(name, product)
            home = None
            for entry in entries:
                if same_attribute(words, entry["words"]):
                    home = entry
                    break
            if home is None:
                entries.append({"words": words, "names": [name]})
            else:
                home["names"].append(name)
        choices = []
        for entry in entries:
            names = entry["names"]
            if any(same_label(n, t) or kin(n, t) for n in names for t in taken):
                continue
            if entry["words"] and any(entry["words"] == w for w in taken_words):
                continue
            by_listing: dict[str, str] = {}
            for n in names:
                for rung, shown in stated[n]:
                    by_listing.setdefault(rung.listing.ref, shown)
            seen = list(by_listing.values())
            distinct = sorted(set(seen))
            own = first_stated(fit.listing.props, names)
            if len(distinct) < 2 and own is not None:
                continue
            coded = False
            for v in distinct:
                if re.search(r"\d", v):
                    coded = True
            if coded and len(distinct) == len(seen) >= 6:
                continue   # every listing different and coded: an identifier (model, sku), not a preference
            other = first_stated(above.listing.props, names) if above is not None and above.read else None
            differs = own is None or (other is not None and other != own)
            echoed = 0   # listings whose title repeats their own value for the attribute
            for rung in read:
                value = first_stated(rung.listing.props, names)
                if value and rung.listing.title and value_carries(value, rung.listing.title):
                    echoed += 1
            signals = {"presence": len(seen), "spellings": len(names), "fewvals": -len(distinct),
                       "differs": differs, "echo": min(echoed, 2), "common": 2 * len(seen) >= len(read),
                       "tier": min(3, (4 * len(seen)) // max(1, len(read)))}
            score = tuple(signals[key] for key in MENU_ORDER)
            choices.append((score, names[0], distinct, own))
        return sorted(choices, key=lambda c: c[0], reverse=True)

    def phrase_question(self, label: str, name: str, examples: list[str]) -> str:
        """One short plain question about one property.  Options appear only when each is one or two plain words;
        raw option strings, units, slashes and parentheses never reach the shopper."""
        prop = spoken_name(name)
        plain = [e.strip().lower() for e in examples
                 if re.fullmatch(r"[a-z][a-z ]{0,18}", e.strip().lower()) and len(e.split()) <= 2]
        if len(set(plain)) == 2:
            return f"For the {label}, which {prop}: {plain[0]} or {plain[1]}?"
        return f"For the {label}, which {prop} do you want?"

    @staticmethod
    def wants_rephrase(reply: str) -> bool:
        """The race shopper asks for simpler words or one question at a time; such a reply answers nothing."""
        text = reply.lower()
        if dialect_value(reply) or re.search(r"\bas (?:a |an )?[a-z0-9][\w\- ]{0,30}, hard as oak", text):
            return False   # an answer that fixes a value is never a request to repeat the question
        cues = (r"\bagain\b|differently|repeat|in turn|one at a time|slice",
                r"tray of question|first of those|only the first|each question",
                r"greek|didn.t follow|plain(?:er|ly| words| english| terms)|in plain|simpler",
                r"lost me|once more|another go|take that in|all at sea|scratching my head",
                r"(?:n.t|not|hard to|couldn.t) understand|swam about|driving at")
        for cue in cues:
            if re.search(cue, text):
                return True
        return False

    def vocabulary(self, ladder: Ladder, name: str) -> list[str]:
        """Values the listings show for a property: the words a dialect answer is checked against."""
        seen: list[str] = []
        for rung in ladder.rungs.values():
            value = rung.listing.props.get(name) if rung.read else None
            if value not in (None, "") and str(value) not in seen:
                seen.append(str(value))
        return seen

    def ask_hidden_needs(self, fits: dict[str, Rung]) -> bool:
        """Ask one property per message, cap respected; returns True when any answer became a requirement.

        One message is kept back for a rephrase while a cap exists; when no rephrase was needed it is spent on a
        last question at the end.
        """
        desk, plan = self.desk("message"), self.plan
        assert plan is not None
        if desk is None or not desk.param("text"):
            return False
        limit = plan.message_cap if plan.message_cap is not None else QUESTION_DEFAULT
        if plan.message_cap is not None:
            limit -= self.counter.held_back   # the basket confirmations before the order have first call on the cap
        spare = 0
        if plan.message_cap is not None and limit >= 2:
            spare = 1
        self.asked_now, self.rephrased, changed = 0, 0, False
        rounds = (self.ladders * max(1, limit), self.ladders)   # the second round spends an unused reserve
        for which, lap in enumerate(rounds):
            if which == 1 and (self.rephrased or not spare):
                break
            for ladder in lap:
                allowed = limit if (self.rephrased or which == 1) else limit - spare
                if self.asked_now >= allowed:
                    break
                if not self.counter.may_ask() or not self.counter.room(self.comfort()) or self.counter.late():
                    return changed
                fit = fits.get(ladder.want.label)
                if fit is None:
                    continue
                outcome = self.interview(desk, ladder, fit)
                if outcome is None:
                    return changed
                changed = changed or outcome
        return changed

    def interview(self, desk: Desk, ladder: Ladder, fit: Rung) -> Optional[bool]:
        """One question (re-asked plainly once on a rephrase request); True when the answer became a need, False
        when not, None when the conversation cannot continue."""
        choice = self.question_for(ladder, fit)
        if choice is not None and self.counter.elapsed() >= CART_DEADLINE - CALL_PATIENCE:
            return None   # a question sent now would be answered after the cart deadline
        if choice is None:
            return False
        name, examples = choice
        ladder.asked.add(name)
        self.asked_now += 1
        reply = self.call(desk, desk.build(text=self.phrase_question(ladder.want.label, name, examples)))
        if reply is None or self.errored(reply):
            return None
        words = flat_text(reply)
        if self.wants_rephrase(words):
            # the same property again, in one plain sentence; the request itself answers nothing
            if not self.counter.may_ask() or not self.counter.room(self.comfort()):
                return None
            self.rephrased += 1
            prop = spoken_name(name)
            reply = self.call(desk, desk.build(text=f"Which {prop} do you want for the {ladder.want.label}?"))
            if reply is None or self.errored(reply):
                return None
            words = flat_text(reply)
            if self.wants_rephrase(words):
                return False
        value, negated, named = self.value_in_reply(words, name, examples, ladder)
        self.trace.append({"kind": "answer", "product": ladder.want.label, "property": named, "asked": name,
                           "value": value, "negated": negated})
        if not value:
            return False
        ladder.asked.add(named)
        op, value = bound_in_answer(value)
        ladder.want.needs.append(Need(named, value, "not" if negated else op, heard=True))
        if not negated:
            # the new value is a search phrase too: a ladder that never saw it may hold no listing stating it
            phrase = f"{ladder.want.phrases[0]} {value}"
            if phrase not in ladder.want.phrases:
                ladder.want.phrases.append(phrase)
            ladder.exhausted = False
        ladder.rejudge()
        return True

    def value_in_reply(self, reply: str, name: str, examples: list[str],
                       ladder: Ladder) -> tuple[Optional[str], bool, str]:
        """(value, negated, property) a reply fixes, checked against what the shopper said and the listings show.

        The shopper often answers about the property the hidden need is really on ('packaging type?' -> 'pack type
        stays fixed: single'), so the property the reply names wins over the one asked.  Dialect replies are read by
        the model when it answers and by code otherwise; a value counts when the shopper literally said it (and is
        normalised to a listing value when one matches), or when it is a listing value, or a distinctive word of
        one, found in the reply.  'anything bar X' negates.
        """
        negated = bool(re.search(r"anything (?:bar|but)|other than|except for|\bbar\b what|nothing like|besides anything", reply.lower()))
        named = self.named_property(reply, ladder) or name
        if non_answer(reply) and not vocabulary_match(reply, self.vocabulary(ladder, name) or examples):
            return None, False, named   # a standard non-answer needs no model call
        quoted = self.quoted_listing_value(reply, ladder, named)
        if quoted is not None:
            return quoted, negated, named   # 'the very same as what's written on the page for ... that X has listed'
        cue = dialect_value(reply)
        if cue:
            exact = next((v for v in self.vocabulary(ladder, named) if same_label(v, cue)), None)
            return exact or cue, negated, named
        vocabulary = self.vocabulary(ladder, named) or self.vocabulary(ladder, name) or examples
        raw = None if self.oracle.lean else self.oracle.ask("reply", REPLY_SYSTEM, f"Property asked: {name}\nListing values: {vocabulary}\n"
                              f"Reply: {reply}", PLANNER_MODELS, tokens=1500)
        raw = unwrapped(raw, "value")
        if isinstance(raw, dict):
            if as_bool(raw.get("no_need")):
                return None, False, named
            negated = negated or bool(as_bool(raw.get("negated")))
            prop = str(raw.get("property") or "").strip()
            if prop and value_carries(prop, reply):
                named = prop
            value = str(raw.get("value") or "").strip()
            if value and value_carries(value, reply):
                exact = next((v for v in vocabulary if same_label(v, value)), None)
                return exact or value, negated, named
            if value:
                checked = vocabulary_match(value, vocabulary)
                if checked:
                    return checked, negated, named
        spoken = vocabulary_match(reply, vocabulary)
        return (spoken, negated, named) if spoken else (None, False, named)

    @staticmethod
    def quoted_listing_value(reply: str, ladder: Ladder, prop: str) -> Optional[str]:
        """The race shopper sometimes points at a listing instead of naming a value: 'what's written on the page
        for the sneakers that Dorsakaan has listed'.  The value is that listing's own value for the property."""
        found = re.search(r"written on the page for .*? that ([A-Za-z][\w\-]+) has listed", reply)
        if not found:
            return None
        who = found.group(1).lower()
        for rung in ladder.rungs.values():
            haystack = " ".join(text_values(rung.listing.raw)).lower()
            if who in haystack or who in rung.listing.title.lower():
                for key, value in rung.listing.props.items():
                    if same_label(key, prop) or label_overlap(key, prop) > 0:
                        return str(value)
        return None

    @staticmethod
    def named_property(reply: str, ladder: Ladder) -> Optional[str]:
        """A listing property name the reply itself mentions (the longest one), if any."""
        names = sorted({k for r in ladder.rungs.values() if r.read for k in r.listing.props})   # deterministic
        said, said_size = None, 0
        for k in names:
            size = len(grains(k))
            if size > said_size and value_carries(k.replace("_", " "), reply):
                said, said_size = k, size
        return said

    # -- checkout ---------------------------------------------------------------------------------------------------

    def cart_signature(self, fits: dict[str, Rung]) -> str:
        return json.dumps(sorted((r.listing.ref, self.allocation.quantities[l] if self.allocation else 1)
                                 for l, r in fits.items()))

    def checkout(self, fits: dict[str, Rung]) -> Optional[Any]:
        """Add one line per product, view the cart, order.  Returns the refusal observation, or None when done."""
        add, view, order = self.desk("cart_add"), self.desk("cart_view"), self.desk("order")
        plan = self.plan
        assert plan is not None and self.allocation is not None
        signature = self.cart_signature(fits)
        whole = add is not None and add.lines is not None and add.lines.get("whole")
        if whole and ({l: r.listing.ref for l, r in fits.items()} != self.cart_refs or self.basket_stale):
            # a cart tool that sets the whole basket gets every line in one call, every time
            observation = self.send_basket(fits)
            if self.finished:
                return None
            if observation is not None and self.errored(observation):
                return observation
        for label, rung in fits.items():
            if whole:
                break
            held = self.cart_refs.get(label)
            if held == rung.listing.ref:
                continue
            if self.cart_refs.get(label):
                self.remove_line(label)
            if add is None:
                break
            qty = self.allocation.quantities[label]
            observation = self.call(add, add.build(id=rung.listing.ref, quantity=qty))
            if self.finished:
                return None
            if self.errored(observation):
                return observation
            self.cart_refs[label] = rung.listing.ref
            self.last_cart_view = observation if self.cart_total(observation) is not None else None
        shown = self.last_cart_view   # a cart tool that answers with the cart saves the view step
        if shown is None and view is not None and self.counter.steps_left() > 1:
            observation = self.call(view, view.build())
            if self.finished:
                return None
        elif shown is not None:
            observation = shown
        if shown is not None or (view is not None and self.counter.steps_left() > 1):
            strays = self.cart_strays(observation, {r.listing.ref for r in fits.values()})
            empty = self.cart_empty(observation) and bool(fits)
            if (strays or empty) and whole and self.counter.steps_left() > 3:
                # the basket view disagrees with the lines just set: set them again (another id field when the
                # view stayed empty, since then the call was read but the line was not)
                self.basket_stale = True
                if empty:
                    self.retarget_lines(add)
                self.send_basket(fits)
                if self.finished:
                    return None
                if view is not None and self.counter.steps_left() > 1:
                    observation = self.call(view, view.build())
            elif strays and self.counter.steps_left() > 2:
                for ref in strays:   # a dead line left in the cart is a guaranteed zero
                    self.discard_ref(ref)
                if view is not None and self.counter.steps_left() > 1:
                    observation = self.call(view, view.build())
            total = self.cart_total(observation)
            limit = self.allocation.budget
            if total is not None and limit is not None and total - limit > 1e-6:
                return {"abacus": "over budget", "total": total}
        if order is None:
            return {"abacus": "no order tool"}
        if not self.cart_refs or set(self.cart_refs) != set(fits):
            return {"abacus": "cart incomplete"}   # an order without every line is a guaranteed miss
        reasons = Audit(self.ladders, self.allocation).review(fits)
        if reasons:
            self.trace.append({"kind": "audit", "reasons": reasons, "salvage": self.salvaged})
            if not self.salvaged:
                return {"abacus": "audit", "reasons": reasons}
        if self.heard:
            # the shopper spoke around an earlier observation: their words are read before anything is ordered
            spoken, self.heard = self.heard, []
            return {"abacus_spoken": spoken}
        message = self.desk("message")
        if self.confirmed != signature and message is not None and message.param("text") \
                and self.counter.may_message() and self.counter.steps_left() > 2 * len(self.ladders) + 3 \
                and self.counter.seconds_left() > CLOSING_SECONDS:
            # the basket as it now stands goes to the shopper once before every order (and again after any edit)
            reply = self.call(message, message.build(text=self.basket_note(fits)))
            self.confirmed, self.notified = signature, True
            if self.finished:
                return None
            words = " ".join(self.heard + [flat_text(reply)]) if reply is not None else " ".join(self.heard)
            self.heard = []
            if reply is not None and not self.errored(reply) and confirmation_refused(words) \
                    and not self.wants_rephrase(words):
                self.confirmed = None
                return {"abacus_spoken": [words], "confirmation": True}
        observation = self.call(order, order.build())
        if self.finished:
            return None
        inner, spoken = split_envelope(observation)
        if observation is None or (self.errored(inner) and not (isinstance(observation, dict) and
                                                                 "observation" in observation)):
            # an opaque code ('E15') is not a refusal: the order did not happen, most likely because the basket
            # is not what was meant.  The lines are set again and the order repeated; the picks stay alive
            self.order_failures += 1
            self.heard.extend(spoken)
            self.cart_refs.clear()
            self.basket_stale = True
            self.last_cart_view = None
            self.trace.append({"kind": "order-failed", "reply": observation, "count": self.order_failures})
            return {"abacus": "order failed"}
        self.refused_carts.add(signature)
        return observation

    def send_basket(self, fits: dict[str, Rung]) -> Any:
        """One call that sets the basket to exactly these lines; on a failure code, once more with another reading
        of the line schema (the other id-like field, or the count left out)."""
        add = self.desk("cart_add")
        assert add is not None and self.allocation is not None
        basket = list(zip((r.listing.ref for r in fits.values()), map(self.allocation.quantities.get, fits)))
        observation = self.call(add, add.build(basket=basket))
        if self.finished:
            return None
        if self.errored(observation) and self.counter.steps_left() > 3 and self.retarget_lines(add):
            observation = self.call(add, add.build(basket=basket))
            if self.finished:
                return None
        if observation is None or self.errored(observation):
            return observation
        self.cart_refs = {l: r.listing.ref for l, r in fits.items()}
        self.basket_stale = False
        self.last_cart_view = observation if self.cart_total(observation) is not None else None
        return observation

    def retarget_lines(self, add: Optional[Desk]) -> bool:
        """Point the line's id at the next string field of the item schema; False when none is left."""
        nest = add.lines if add is not None else None
        if not nest:
            return False
        tried = nest.setdefault("tried", [nest["id"]])
        spare = next(iter(k for k in typed_keys(nest["props"], "string")
                          if not is_enum(nest["props"][k]) and k not in tried), None)
        if spare is None:
            return False
        tried.append(spare)
        nest["id"], nest["extra"] = spare, {k: v for k, v in nest["extra"].items() if k != spare}
        return True

    def cart_empty(self, observation: Any) -> bool:
        """A cart view that shows no lines at all: a total of zero or every list in it empty."""
        if not isinstance(observation, dict) or self.errored(observation):
            return False
        lists = [v for v in observation.values() if isinstance(v, list)]
        return self.cart_total(observation) == 0 or (bool(lists) and not any(lists) and not self.lens.rows_of(observation))

    def basket_note(self, fits: dict[str, Rung]) -> str:
        """Every line with its quantity and unit price, then the one question: shall the order go?"""
        parts = []
        for label, rung in fits.items():
            qty = self.allocation.quantities[label] if self.allocation else 1
            title = re.sub(r"\s+", " ", str(rung.listing.title or label)).strip()
            if len(title) > 48:
                title = title[:48].rsplit(" ", 1)[0]
            price = rung.price if rung.price is not None else rung.listing.price
            cost = f" at {price:g} each" if is_number(price) else ""
            parts.append(f"{qty} x {title}{cost}")
        return f"Your basket: {'; '.join(parts)}. Shall I place this order?"

    def remove_line(self, label: str) -> None:
        ref = self.cart_refs.pop(label, None)
        if ref is not None:
            self.discard_ref(ref, self.allocation.quantities[label] if self.allocation else 1)

    def discard_ref(self, ref: str, qty: int = 1) -> None:
        """Take a listing out of the cart: the remove desk, or the add desk at quantity zero when its schema allows.
        A cart tool that sets the whole basket needs no call here: the next basket it is sent leaves the line out."""
        remove, add = self.desk("cart_remove"), self.desk("cart_add")
        if add is not None and add.lines is not None:
            if not add.lines.get("whole"):
                self.call(add, add.build(basket=[(ref, 0)]))
            self.basket_stale = True
        elif remove is not None:
            self.call(remove, remove.build(id=ref, quantity=qty))
        elif add is not None and add.param("quantity"):
            # a count of zero is the only way out without a remove tool; a schema minimum is overridden on purpose,
            # since a line left behind is a certain miss and a refused call costs one step
            args = add.build(id=ref, quantity=1)
            args[add.param("quantity")] = 0
            self.call(add, args)
        for label, held in list(self.cart_refs.items()):
            if held == ref:
                self.cart_refs.pop(label)

    def cart_strays(self, observation: Any, wanted: set[str]) -> list[str]:
        """Ids in the cart view that are not the current picks."""
        seen = []
        for row in self.lens.rows_of(observation):
            listing = self.lens.listing(row)
            known = self.known_refs()
            ref = listing.ref if listing else None
            if ref is None:
                for v in text_values(row):
                    if v in known:
                        ref = v
                        break
            if ref and ref in known and ref not in wanted and ref not in seen:
                seen.append(ref)
        return seen

    def known_refs(self) -> set[str]:
        return {ref for ladder in self.ladders for ref in ladder.rungs}

    def cart_total(self, observation: Any) -> Optional[float]:
        if not isinstance(observation, dict):
            return None
        for k, v in observation.items():
            if is_number(v) and re.search(r"total|sum|grand|amount|price|cost", k.lower()):
                return float(v)
        rows = self.lens.rows_of(observation)
        prices = [self.lens.price_of(r) for r in rows]
        if rows and all(p is not None for p in prices):
            return sum(p * (self.lens_quantity(r) or 1) for p, r in zip(prices, rows))
        return None

    @staticmethod
    def lens_quantity(row: dict) -> Optional[int]:
        for k, v in row.items():
            if is_number(v) and re.search(r"qty|quant|count|units", k.lower()):
                return int(v)
        return None

    def absorb_refusal(self, observation: Any, fits: dict[str, Rung]) -> None:
        """Turn a refusal into ladder changes: dead rungs, a new budget, dropped or added requirements."""
        assert self.allocation is not None
        if isinstance(observation, dict) and observation.get("abacus") == "audit":
            for reason in observation.get("reasons") or []:
                label, _sep, _why = reason.partition(":")
                if label in fits:
                    fits[label].dead = True
                    self.remove_line(label)
            if not any(r.dead for r in fits.values()):   # only the budget line: shed the heaviest
                observation = {"abacus": "over budget"}
            else:
                self.trace.append({"kind": "refusal", "sold": [r.listing.ref for r in fits.values() if r.dead], "changed": False})
                for ladder in self.ladders:
                    ladder.exhausted = False
                return
        if isinstance(observation, dict) and "abacus" in observation:
            # the cart failed the agent's own budget check: the line that weighs most is the one to replace
            heaviest, heaviest_cost = None, -1.0
            for l, rung in fits.items():
                cost = self.allocation.quantities[l] * (rung.price if rung.price else 0)
                if cost > heaviest_cost:
                    heaviest, heaviest_cost = l, cost
            fits[heaviest].dead = True
            self.remove_line(heaviest)
            self.trace.append({"kind": "refusal", "sold": [fits[heaviest].listing.ref], "changed": False})
            return
        spoken = isinstance(observation, dict) and "abacus_spoken" in observation
        lines = string_lists(observation)
        text = " ".join(lines) if lines else flat_text(observation)
        low = text.lower()
        if price_moved(low):
            # the price moved: the held lines are taken out, their CURRENT prices read again, and the ladders
            # re-judged on them, so a line that now breaks the budget is replaced instead of re-ordered
            moved = []
            for label, ref in list(self.cart_refs.items()):
                self.discard_ref(ref, self.allocation.quantities[label])
                rung = fits.get(label)
                if rung is not None:
                    before = rung.price
                    self.reread_price(rung)
                    moved.append([ref, before, rung.price])
            self.refused_carts.discard(self.cart_signature(fits))
            self.price_refresh += 1
            for ladder in self.ladders:
                ladder.exhausted = False
            self.trace.append({"kind": "refusal", "sold": [], "changed": True, "price_moved": True, "prices": moved})
            return
        raw = None if self.oracle.lean else self.oracle.ask("refusal", REFUSAL_SYSTEM,
                              f"Products: {[l.want.label for l in self.ladders]}\nCart: {self.cart_refs}\n"
                              f"Requirements: { {l.want.label: [n.describe() for n in l.want.needs] for l in self.ladders} }\n"
                              f"Refusal: {text[:3000]}", PLANNER_MODELS, tokens=1500)
        changed = False
        loosened = tightened = False
        sold = set()
        raw = unwrapped(raw, "sold_out_ids")
        if isinstance(raw, dict):
            sold = {str(s).strip() for s in as_list(raw.get("sold_out_ids"))}
            budget = number_in(raw.get("budget"))
            if budget:
                self.allocation.budget = budget * self.scale
                changed = tightened = True
            for drop in as_list(raw.get("dropped")):
                if isinstance(drop, str):
                    drop = {"property": drop}
                if isinstance(drop, dict):
                    for ladder in self.ladders:
                        if ladder.drop_need(str(drop.get("property") or "")):
                            changed = loosened = True
            for add in as_list(raw.get("added")):
                need = Need.from_raw(add) if isinstance(add, dict) else None
                target = next((l for l in self.ladders if isinstance(add, dict)
                               and label_overlap(l.want.label, str(add.get("product") or ""))), None)
                if need and target:
                    target.want.needs.append(need)
                    changed = tightened = True
        held = sorted({r.listing.ref for r in fits.values()})
        named = [ref for ref in held if ref in sold or ref.lower() in low]
        sold.update(named)
        if sold_out_words(low) and not named:
            # 'shows zero left' beside 'never mind the colour': a pick is gone either way.  One line is that line;
            # with several, each is looked at again and the ones now shown out of stock are the gone ones
            gone = held
            if len(held) > 1:
                for rung in fits.values():
                    self.reread_price(rung)
                gone = [r.listing.ref for r in fits.values() if r.listing.stock is False] or held
            sold.update(gone)
        if not sold and not changed:
            money = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:us\s+)?(?:cents?|cts|\u00a2|dollars?|usd|euros?|rand|pesos?)", low)
            if money:   # no model reading: a sum with a currency word is the new budget
                self.allocation.budget = float(money.group(1).replace(",", "")) * self.scale
                changed = tightened = True
        if not sold and not changed:
            # no model reading: a need whose property the shopper names in the refusal is the one being waived;
            # a need read without a property ('any = gray') is matched through the listing keys that carry its value
            for ladder in self.ladders:
                for need in list(ladder.want.needs):
                    names = [need.prop] + ladder.keys_carrying(need.value) if need.prop == "any" else [need.prop]
                    if any(label_overlap(name, text) > 0 for name in names) and ladder.drop_value_need(need):
                        changed = loosened = True
        if changed and not sold and self.resent < 2:
            # the shopper changed the budget or a requirement, not the listing: the same cart may go once more,
            # and the audit before the order decides whether it still fits what they now ask
            self.refused_carts.discard(self.cart_signature(fits))
            self.resent += 1
        if not sold and not changed and LOOSENING.search(low):
            # 'X is no longer a condition', 'leave out X', 'scrap X': a waiver of something the lines need not meet
            changed = loosened = True
        if loosened and not tightened and not sold and fits and set(self.cart_refs) == set(fits):
            # a condition was waived and nothing new was asked: the lines in the cart still fit, so they go again
            self.kept = dict(fits)
        if spoken and not observation.get("confirmation") and not sold and not changed:
            self.trace.append({"kind": "spoken", "words": text[:300], "changed": False})
            self.kept = dict(fits) if fits else None
            return   # words that change nothing leave the cart as it is
        if not sold and not changed and fits:
            sold = {r.listing.ref for r in fits.values()}   # unknown reason: never re-send the same cart
        for ladder in self.ladders:
            for ref in sold:
                if ref in ladder.rungs:
                    ladder.rungs[ref].dead = True
            ladder.exhausted = False
            ladder.rejudge()
        for label in list(self.cart_refs):
            if self.cart_refs[label] in sold:
                self.remove_line(label)
        self.trace.append({"kind": "refusal", "sold": sorted(sold), "changed": changed})

    def reread_price(self, rung: Rung) -> None:
        """Read one listing's current price and stock again (offer tool, else details, else the search row)."""
        desk = self.desk("offer") or self.desk("details")
        if desk is None or not self.counter.room(self.reserve()):
            return
        observation = self.call(desk, desk.build(id=rung.listing.ref))
        if not isinstance(observation, dict) or self.errored(observation):
            return
        seen = self.lens.listing(observation)
        if seen is None:
            seen = Listing(rung.listing.ref, self.lens.price_of(observation), self.lens.stock_of(observation),
                           "", "", {}, observation)
        rung.listing.merge(seen)

    # -- finishing ----------------------------------------------------------------------------------------------------

    def decline(self, reason: str) -> Any:
        """Decline through the decline tool only (messages are for hidden-need questions, never status notes).
        The per-need verdicts of the cheapest rungs go to the trace first.  Returns the world's answer when the
        decline did not finish the episode: the shopper is refusing the decline and usually waiving a need."""
        desk, plan = self.desk("decline"), self.plan
        self.trace.append({"kind": "ladders", "reason": reason,
                           "report": {l.want.label: l.report() for l in self.ladders}})
        if desk is None or self.finished:
            return None
        text = (plan.decline_rule if plan and plan.decline_rule else reason) or "No fitting option found."
        observation = self.call(desk, desk.build(text=text))
        return None if self.finished else observation

    def salvage(self, fits: dict[str, Rung]) -> dict[str, Rung]:
        """After a refused decline nothing may be declined again: every product gets its best literal candidate,
        the cheapest read rung failing the fewest needs."""
        for ladder in self.ladders:
            if ladder.want.label in fits:
                continue
            ceiling = self.allocation.slack(ladder.want.label) if self.allocation else None
            candidates = []
            for r in ladder.live():
                if r.price is None or (ceiling is not None and r.price > ceiling + 1e-9) or r.kind_ok is False:
                    continue
                unproven = [i for i in range(len(ladder.want.needs)) if not ladder.waived(i) and r.verdicts.get(i) is not True]
                if unproven:
                    continue   # never order a listing that fails, or does not state, a stated requirement
                candidates.append(r)
            if candidates:
                def shortfall(r: Rung) -> tuple:
                    failed = list(r.verdicts.values()).count(False)
                    return failed, not r.read, r.kind_ok is False, -ladder.word_score(r), r.price

                fits[ladder.want.label] = min(candidates, key=shortfall)
        return fits

    def run(self) -> None:
        self.make_plan()
        if not self.desk("search") or self.finished:
            self.decline("the world offers no way to search")
            return
        self.observe_prices()
        fits = self.run_ladders()
        if len(fits) < len(self.ladders) and self.counter.late():
            fits = self.salvage(fits)   # an order on the best code-ranked listing beats no order
            self.salvaged = True
        asked = False
        cycles = 0
        while not self.finished and cycles < 4:
            cycles += 1
            if len(fits) == len(self.ladders):
                if not asked and self.counter.room(self.comfort()) and not self.counter.late():
                    asked = True
                    if self.ask_hidden_needs(fits):
                        fits = self.run_ladders()
                        continue
                if self.price_refresh > 2:
                    for rung in fits.values():
                        rung.dead = True   # a price that keeps moving is not a cart to resend forever
                    self.price_refresh = 0
                    fits = self.run_ladders()
                    continue
                if self.cart_signature(fits) in self.refused_carts:
                    # the world already said no to exactly this cart; whatever the reason, these picks are over
                    for rung in fits.values():
                        rung.dead = True
                    fits = self.run_ladders()
                    continue
                refusal = self.checkout(fits)
                if self.finished:
                    return
                if isinstance(refusal, dict) and refusal.get("abacus") == "order failed":
                    if self.order_failures <= 2 and self.counter.steps_left() >= self.checkout_steps():
                        cycles -= 1   # the same picks go again with the lines set afresh
                        continue
                    break
                if isinstance(refusal, dict) and refusal.get("abacus") in ("no order tool", "cart incomplete"):
                    self.trace.append({"kind": "halt", "reason": refusal["abacus"]})
                    self.finish_somehow(f"the cart could not be filled ({refusal['abacus']})")
                    return
                if refusal is not None:
                    self.absorb_refusal(refusal, fits)
                    if self.kept:
                        # the lines are still in the cart: re-sending them needs no rebuild, only the order
                        fits, self.kept = self.kept, None
                        continue
                    if self.counter.steps_left() < self.checkout_steps() + 1:
                        self.finish_somehow("no steps left to rebuild the cart after a refusal")
                        return
                    fits = self.run_ladders()
                    if len(fits) < len(self.ladders) and self.counter.late():
                        fits = self.salvage(fits)
                        self.salvaged = True
                    continue
            missing = [l.want.label for l in self.ladders if l.want.label not in fits]
            if missing and any(not l.exhausted for l in self.ladders if l.want.label in missing) \
                    and self.counter.room(self.reserve()):
                fits = self.run_ladders()
                if len(fits) == len(self.ladders):
                    continue
            break
        if self.finished:
            return
        missing = [l.want.label for l in self.ladders if l.want.label not in fits]
        reason = (f"no affordable option met every requirement for {', '.join(missing)}" if missing
                  else "the only fitting options were refused")
        if not missing and fits and not self.refused_carts:
            if self.checkout(fits) is None:
                return
        elif missing and fits and len(self.ladders) > 1 and not self.refused_carts and not self.priced_out(missing):
            # one product nobody could find never cancels the others: their verified lines are ordered
            self.trace.append({"kind": "partial", "missing": missing})
            if self.checkout(fits) is None:
                return
        refused = self.decline(reason)
        if refused is None or self.finished:
            return
        # the shopper refused the decline: read what they waived, re-run the ladders and order; a refusal of that
        # order is read and answered the same way, a bounded number of times
        self.absorb_refusal(refused, fits)
        for _attempt in range(3):
            fits = self.run_ladders()
            if len(fits) < len(self.ladders):
                for ladder in self.ladders:
                    ladder.soft.update(range(len(ladder.want.needs)))
                fits = self.salvage(self.run_ladders())
                self.salvaged = True
            if self.finished or not fits or self.cart_signature(fits) in self.refused_carts:
                break
            answer = self.checkout(fits)
            if self.finished or answer is None:
                return
            if isinstance(answer, dict) and answer.get("abacus") in ("no order tool", "cart incomplete"):
                break
            if isinstance(answer, dict) and answer.get("abacus") == "order failed":
                continue   # the lines are set again on the next pass; the picks are not to blame
            self.absorb_refusal(answer, fits)
        self.finish_somehow("no order could be completed after the refusals")

    def priced_out(self, labels: list[str]) -> bool:
        """A missing product with a listing that meets every need but the budget: the basket as asked costs too
        much, which is a real reason to decline rather than order a part of it."""
        for ladder in self.ladders:
            if ladder.want.label not in labels:
                continue
            slack = self.allocation.slack(ladder.want.label) if self.allocation else None
            for rung in ladder.live():
                if rung.kind_ok is False or any(v is False for v in rung.verdicts.values()):
                    continue
                if rung.read and all(ladder.waived(i) or rung.verdicts.get(i) is True for i in range(len(ladder.want.needs))):
                    return True
                if slack is not None and rung.price is not None and rung.price > slack + 1e-9:
                    return True   # a candidate the budget alone put out of reach
        return False

    def finish_somehow(self, reason: str) -> None:
        """An episode never ends open: with a step left and nothing ordered, the decline tool closes it."""
        if self.finished or self.counter.steps_left() < 1:
            return
        desk = self.desk("decline")
        if desk is not None:
            self.trace.append({"kind": "ladders", "reason": reason, "report": {l.want.label: l.report() for l in self.ladders}})
            self.call(desk, desk.build(text=(self.plan.decline_rule if self.plan and self.plan.decline_rule else reason)))


def agent_main(problem_data: dict) -> list[dict]:
    """Entry point: run the broker, never let an exception escape, always return the trace."""
    broker = Broker(problem_data)
    try:
        broker.run()
    except WireDown as down:
        broker.trace.append({"kind": "halt", "reason": f"environment unreachable at {down}"})
    except Exception as error:  # a crash must still return a trace; the platform scores the finish, not the log
        broker.trace.append({"kind": "crash", "error": f"{type(error).__name__}: {error}"})
    closing = {"kind": "end", "steps": broker.step, "finished": broker.finished}
    closing["seconds"] = round(broker.counter.elapsed(), 2)
    broker.trace.append(closing)
    return broker.trace


__all__ = ["agent_main"]
