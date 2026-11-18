

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_PROXY_TIMEOUT_S = 110
_FINISH_REMAINING_S = 40.0
_OBS_CHARS = 7000
_RECENT_OBS = 6
_OLD_OBS_CHARS = 500
_MAX_REPROMPTS = 2
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_AMOUNT_RE = re.compile(_NUM + r"(?:\s*([A-Za-z]{1,12}|₱))?")
# A shopper says how much they will spend in whatever words they like, so the direction of an
# amount is read from the wording around it rather than matched against the sentences someone
# might have written. Anything that sets a floor, or is not about money at all, is left alone.
_CAP_BEFORE = re.compile(
    r"(?:at most|at or (?:under|below)|up to|under|below|beneath|less than|no more than|"
    r"not more than|no higher than|not (?:above|over|more)|within|"
    r"(?:max(?:imum)?|cap(?:ped)?|ceiling|limit(?:ed)?|budget)(?:\s+(?:of|is|at|to))?|"
    r"spend(?: up to)?|afford|pay|keep it (?:to|under|below)|top out at|go up to|"
    r"stay (?:under|below|within))\s*[^\w\d]{0,3}$",
    re.I,
)
_CAP_AFTER = re.compile(
    r"^\s*(?:\w+\s+)?(?:or (?:less|under|below|lower)|tops|max(?:imum)?|at most|ceiling|"
    r"cap|limit|budget)\b",
    re.I,
)
_FLOOR_BEFORE = re.compile(
    r"(?:at least|no less than|not less than|min(?:imum)?(?: of)?|more than|over|above|"
    r"upwards of|starting (?:at|from))\s*[^\w\d]{0,3}$",
    re.I,
)
_NEGATED = re.compile(
    r"\b(?:can'?t|can ?not|won'?t|will not|don'?t|do not|never|under no|rather not|nothing|"
    r"none|no)\b[^.;!?]{0,28}$",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_PREF_ITEM_RE = re.compile(
    r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b",
    re.I,
)
_PHRASE_RE = re.compile(r"matches (?:the (?:phrase|text|words)\s+)?([^\",.;]{2,60})", re.I)
_ASK_ME_RE = re.compile(r"\bif you ask\b|\bask me\b|\bI will tell you\b", re.I)
_REASON_RE = re.compile(r"\b(?:why|reasons?|justif\w*|good pick|walk me through|explain)\b", re.I)
_CHEAP_RE = re.compile(
    r"don't just grab the cheapest|cheapest isn't|rather pay|not blindly buying the cheapest|"
    r"matters more to me than saving",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_EVENT_NOTICE_RE = re.compile(
    r"still available|out of stock|sold out|more expensive|price (?:jumped|went)|"
    r"re-check before ordering|any alternatives|leaning toward",
    re.I,
)
_MEASURE_RE = re.compile(r"(?<![\w.])(\d{1,4}(?:\.\d{1,3})?)\s*-?\s*([A-Za-z]{1,10})\b")
_NUM_TOKEN_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
_UNIT_NEAR_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*([A-Za-z]{1,6})")
_STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
    "order", "stock", "place", "under", "over", "one", "test", "buy", "from", "into",
    "an", "a", "of", "to", "in", "on", "at", "or", "by", "me", "my", "it", "is", "are",
}
_RANK_RE = re.compile(
    r"\b(highest|largest|biggest|greatest|maximum|most|fastest|lowest|smallest|cheapest|least|minimum|slowest)\s+"
    r"((?:advertised\s+|maximum\s+|available\s+|current\s+|overall\s+)*[a-z]+(?:\s+[a-z]+)?)",
    re.I,
)
_RANK_WORDS = {
    "highest", "largest", "biggest", "greatest", "maximum", "most", "fastest",
    "lowest", "smallest", "cheapest", "least", "minimum", "slowest",
}
_DOWN_WORDS = {"lowest", "smallest", "cheapest", "least", "minimum", "slowest"}
_FILLER = {
    "also", "use", "only", "and", "or", "one", "supported", "reason", "reasons", "field",
    "fields", "the", "a", "an", "specification", "shopper", "request", "must", "true",
    "such", "as", "value", "values", "claim", "claims", "include", "least",
}

_CORE_RULES = """You are a shopping agent working a live catalog on behalf of one shopper.

# How a turn works
- Make exactly one tool call per turn. Turns are strictly limited, so every call must bring
  back something you do not already know. A wasted turn is a turn the shopper does not get.
- A controller checks what you find against the shopper's stated requirements and owns the
  cart and the order. Your job is to get the right products on record for it.

# Finding products
- Search with the shopper's own words first, including the exact phrase if they gave one.
- When a search disappoints, reformulate with genuinely different wording: a synonym, the
  category name, the brand, the words reordered, or one distinctive word on its own. Repeating
  the same phrase with one number changed returns the same page and wastes the turn.
- Use filter for structured limits (category, brand, max price, in stock) and search for
  wording. They surface different products, so both are worth trying.
- Never repeat a call you have already made.

# Reading products
- A search hit is not evidence. Use view or compare before believing anything about a product,
  and read its raw facts and variant options rather than its marketing title.
- category_path decides the category. A title that sounds right does not put a listing in a
  category, and an accessory or a spare part is not the device itself.
- When a listing covers several models, prefer the value stated for the selected variant over
  the product-wide or best-case value.

# Committing
- The first add_to_cart is the shopper's real purchase, never a placeholder or a probe, so it
  must already be the best option you can justify.
- Prices and stock can change immediately after that addition. The controller re-checks and
  moves to another product when it has to.

# Talking to the shopper
- Answer a direct question with message before anything else, and answer it specifically.
- Never ask the shopper to raise the budget or to re-approve a purchase they already asked for.
"""


def _json(value: Any, **kwargs) -> str:
    try:
        return json.dumps(value, **kwargs)
    except (TypeError, ValueError):
        return "{}"


def _proxy(remaining: float) -> ProxyClient:
    try:
        left = float(remaining)
    except (TypeError, ValueError):
        left = 8.0
    if left != left or left in (float("inf"), float("-inf")):
        left = 8.0
    if left <= 2:
        return ProxyClient(timeout=1, max_retries=1, retry_delay=0.2)
    timeout = max(2, min(_PROXY_TIMEOUT_S, int(left) - 2))
    if left < 90:
        timeout = min(timeout, max(6, int(left) // 3))
    if timeout >= _PROXY_TIMEOUT_S:
        return _PROXY
    return ProxyClient(timeout=timeout, max_retries=1, retry_delay=0.5)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _word_in(needle: str, hay: str) -> bool:
    """Whole-word presence, same idea as angryoro is_kw_in (exact token, then simple plural)."""
    n = (needle or "").strip().lower()
    if not n:
        return False
    blob = hay or ""
    pat = r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])"
    if re.search(pat, blob, re.I):
        return True
    if len(n) > 3 and not n[-1].isdigit():
        stem = n[:-1] if n.endswith("s") else n
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"s?(?![a-z0-9])", blob, re.I))
    return False


def _parse_json(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*", "", (text or "").strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for start in sorted(i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0):
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except ValueError:
            continue
    return None


def _spend_caps(text: str) -> list:
    """Every amount the shopper put a ceiling on: (value, currency, start, end).

    Reading the words on either side of an amount, rather than the sentence it sits in, is what
    lets "at most 500", "I can't go over 500" and "500 tops" all be the same instruction. Getting
    this wrong is expensive in a way the other readings are not: a missed ceiling is an order
    placed over budget, which is marked exactly as though nothing had been bought at all.
    """
    found = []
    for m in _AMOUNT_RE.finditer(text or ""):
        before, after = text[: m.start()], text[m.end():]
        if _CAP_BEFORE.search(before):
            capped = True
        elif _FLOOR_BEFORE.search(before):
            # "over 500" is a floor, but "I can't go over 500" is the same ceiling said backwards.
            capped = bool(_NEGATED.search(_FLOOR_BEFORE.sub("", before)))
        else:
            capped = bool(_CAP_AFTER.search(after))
        if not capped:
            continue
        try:
            value = float(m.group(1).replace(",", "") + "." + (m.group(2) or "0"))
        except ValueError:
            continue
        found.append((value, (m.group(3) or "").strip(), m.start(), m.end()))
    return found


def _prose(text: str) -> str:
    """The request minus the parts that are identifiers or money, so numbers left over are specs."""
    clean = _KEY_RE.sub(" ", text or "")
    for _value, _cur, start, end in reversed(_spend_caps(clean)):
        clean = clean[:start] + " " + clean[end:]
    return clean


def _measures(text: str, currency: str = "") -> list:
    """(value, unit) pairs the shopper wrote: 7.6-inch, 720p, 165 Hz, 8 GB."""
    cur = (currency or "").lower()
    out, seen = [], set()
    for m in _MEASURE_RE.finditer(_prose(text)):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit = m.group(2).lower()
        if unit in _STOP or unit in _RANK_WORDS or (cur and unit == cur):
            continue
        if (val, unit) in seen:
            continue
        seen.add((val, unit))
        out.append((val, unit))
    return out


def _numbers(text: str) -> list:
    out = []
    for m in _NUM_TOKEN_RE.finditer(text or ""):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _value_in(value: float, blob: str) -> bool:
    """The requested amount is written somewhere in the product's own facts."""
    return any(abs(v - value) < 1e-6 for v in _numbers(blob))


def _standalone(value: float, blob: str) -> bool:
    """The amount is written as a number in its own right, rather than as digits that happen to
    sit inside a model code."""
    return re.search(r"(?<![\w.])" + re.escape("%g" % value) + r"(?![\w.])", blob or "") is not None


def _value_unit_in(value: float, unit: str, blob: str) -> bool:
    """The amount is written right next to the unit the shopper used."""
    stem = unit[:3] if len(unit) >= 3 else unit
    pat = r"(?<![\w.])" + re.escape("%g" % value) + r"\s*-?\s*" + re.escape(stem)
    return re.search(pat, blob or "", re.I) is not None


def _spec_tags(text: str, currency: str = "") -> set:
    """Short words the shopper wrote right beside a measurement, e.g. the "HD" in "720p HD"."""
    cur = (currency or "").lower()
    tags = set()
    for m in _MEASURE_RE.finditer(_prose(text)):
        if m.group(2).lower() == cur:
            continue
        tail = _prose(text)[m.end():m.end() + 24]
        for word in re.findall(r"[A-Za-z]{2,5}", tail)[:2]:
            if word.isupper() and word.lower() not in _STOP:
                tags.add(word.lower())
    return tags


# (share of budget, keep the brand, keep the category) — narrowings of the same shelf that cost a
# call but not a search query, so they widen what is on record without spending the shopper's words.
_BANDS = (
    (1.0, True, True),
    (1.0, False, True),
    (1.0, True, False),
    (0.8, True, True),
    (0.62, True, True),
    (0.48, True, True),
    (0.36, True, True),
    (0.26, True, True),
    (0.18, True, True),
    (0.1, True, True),
    (0.5, False, True),
    (0.22, False, True),
)
_DISCOVERY = ("search", "filter", "view", "compare")
# The turn, not the call, is what the shop counts. An envelope may carry a whole turn's worth of
# reads and still be charged once, so the only reason to send a half-empty one is to decide on a
# thinner record than was going free.
_RIDE_ALONG = 15
# Past a certain point the shelf has been read well enough that more of it will not change the
# choice, and the turns are better spent on the choice itself.
_READ_CEILING = 60
_EXTRA_OBS_CHARS = 6000


def _split_values(text: str) -> list:
    """A fact value broken at the punctuation catalogs use to list alternatives."""
    return [c for c in re.split(r"[;,/]|\s+-\s+|\s+or\s+", str(text or ""), flags=re.I) if c.strip()]


def _ambiguous(text: str) -> bool:
    """The fact offers several competing amounts, so it does not pin the product down."""
    return sum(1 for chunk in _split_values(text) if _NUM_TOKEN_RE.search(chunk)) > 1


def _spec_queries(specs: list) -> list:
    """Catalog spellings of a measurement: 7.6 inch / 7.6inch / 7.6-inch / 7.6in."""
    out = []
    for value, unit in specs:
        n = "%g" % value
        out.extend([n + " " + unit, n + unit, n + "-" + unit])
        if len(unit) > 2:
            out.append(n + unit[:2])
    return out


def _query_key(q: str) -> str:
    """What a search is actually asking, ignoring how the measurement is punctuated. A catalog reads
    "6.79 inch", "6.79-inch" and "6.79inch" as the same words, so spelling one request four ways buys
    the same page four times and spends four of the thirty turns doing it."""
    key = " ".join(str(q).split()).lower()
    return re.sub(r"(\d)[\s\-]*([a-z]+)", lambda m: m.group(1) + m.group(2)[:2], key)


def _uniq_queries(cands: list) -> list:
    seen, out = set(), []
    for q in cands:
        q = " ".join(str(q).split())
        key = _query_key(q)
        if q and key not in seen and not all(w in _STOP for w in key.split()):
            seen.add(key)
            out.append(q)
    return out


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Intent:
    """Shopper request, parsed as language — not as a named task family."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        # Everything after the rules heading is written to the assistant rather than about the
        # goods, so the shopper's actual request is the part before it.
        self.goal = self.raw.partition("\n\nTask rules:")[0].strip()
        self.budget = self._budget(self.goal)
        self.currency = self._currency(self.goal)
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        self.theme = self._theme(self.goal)
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = self._pref_brand(self.goal)
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.ask_first = bool(_ASK_ME_RE.search(self.goal))
        self.want_reason = bool(_REASON_RE.search(self.goal))
        self.avoid_cheap = bool(_CHEAP_RE.search(self.goal))
        self.rank_axes = self._rank_axes(self.goal, self.raw)
        self.specs = self._keep_specs(_measures(self.goal, self.currency))
        self.spec_tags = _spec_tags(self.goal, self.currency)
        self.soft_words = self._soft_words(self.goal)
        self.spec_open = self.ask_first
        self.said: list = []
        self.focus: list = []
        self.focus_at = -1

    @staticmethod
    def _budget(text: str) -> float | None:
        """The tightest ceiling the shopper named. Where they gave several, the smallest is the
        one that has to hold: a request that clears the loosest may still break the others."""
        caps = [value for value, _cur, _s, _e in _spend_caps(text or "")]
        return min(caps) if caps else None

    @staticmethod
    def _currency(text: str) -> str:
        for _value, cur, _s, _e in _spend_caps(text or ""):
            if cur:
                return cur
        return ""

    @staticmethod
    def _category(goal: str) -> str | None:
        m = _CAT_PHRASE_RE.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        return m.group(1) if m else None

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", m.group(1)) if w]
        named = lambda w: (
            w[:1].isupper() and not any(c.isdigit() for c in w) and not (len(w) <= 3 and w.isupper())
        )
        # A brand is one unbroken run of name-like words, not every capitalised word in the sentence.
        runs, current = [], []
        for w in words:
            if named(w):
                current.append(w)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        if not runs:
            return None
        best = max(runs, key=len)
        return " ".join(best)

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = re.search(
            r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)",
            goal or "",
        )
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        if Intent._budget(name) is not None:
            return None
        return name

    @staticmethod
    def _theme(goal: str) -> list:
        """The exact wording the shopper wants matched, quoted or introduced as a phrase."""
        quoted = _QUOTED_RE.findall(goal or "")
        if quoted:
            return _tokens(quoted[0])
        m = _PHRASE_RE.search(goal or "")
        if not m:
            return []
        words = [w for w in _tokens(m.group(1)) if w not in _STOP]
        return words if 1 < len(words) <= 5 else []

    @staticmethod
    def _rank_axes(goal: str, whole: str) -> list:
        """Tradeoff order the shopper stated, as (prefer_low, attribute words)."""
        axes, seen = [], []
        for clause in re.split(r"[.;]", goal or ""):
            m = _RANK_RE.search(clause)
            if not m:
                continue
            down = m.group(1).lower() in _DOWN_WORDS
            words = [w for w in _tokens(m.group(2)) if len(w) > 3 and w not in _STOP and w not in _RANK_WORDS]
            words = [w for w in words if w not in ("advertised", "available", "current", "overall", "option")]
            if not words or words in seen:
                continue
            seen.append(words)
            axes.append((down, Intent._widen_axis(words, whole), " ".join(words), m.group(1).lower()))
        return axes

    @staticmethod
    def _widen_axis(words: list, whole: str) -> list:
        """The instructions often spell the same attribute out more fully ("screen" -> "screen size")."""
        low = (whole or "").lower()
        extra = set()
        for w in words:
            for m in re.finditer(r"\b" + re.escape(w) + r"\s+([a-z]+)\b", low):
                extra.add(m.group(1))
            for m in re.finditer(r"\b([a-z]+)\s+" + re.escape(w) + r"\b", low):
                extra.add(m.group(1))
        extra -= set(_STOP) | _RANK_WORDS | set(words)
        extra = {w for w in extra if len(w) > 2}
        return list(words) + sorted(extra)[:3]

    def _keep_specs(self, pairs: list) -> list:
        return [(v, u) for v, u in pairs if self.budget is None or abs(v - self.budget) > 1e-6]

    @staticmethod
    def _soft_words(goal: str) -> set:
        return set(Intent._soft_order(goal))

    @staticmethod
    def _soft_order(text: str) -> list:
        """The shopper's own describing words, kept in the order they wrote them: what they lead
        with is what they care about, and a query built that way reads like their request."""
        drop = set(_STOP) | {"stock", "place", "order", "listing", "category", "please", "compare"}
        out = []
        for w in _tokens(text):
            if len(w) > 3 and not w.isdigit() and w not in drop and w not in out:
                out.append(w)
        return out

    def adopt_text(self, text: str, specs: bool = True) -> None:
        """Shopper later named a requirement: keep their wording, don't map it through a table."""
        new_b = self._budget(text)
        if new_b is not None and (self.budget is None or new_b < self.budget):
            self.budget = new_b
        if specs:
            found = self._keep_specs(_measures(text, self.currency))
            for pair in found:
                if pair not in self.specs:
                    self.specs.append(pair)
            self.spec_tags |= _spec_tags(text, self.currency)
            if text.strip() and text.strip() not in self.said:
                self.said.append(text.strip())
        extra = [t for t in _tokens(text) if len(t) > 2 and t not in _STOP]
        if extra:
            self.soft_words |= set(extra)
        self.spec_open = False

    def criteria(self) -> str:
        """Everything the shopper has actually said they want, in their own words."""
        return "\n".join([self.goal] + self.said)

    def focus_words(self) -> list:
        """When a shopper says what they would pay more for, that one clause names what to go and
        find. Everything else they write is background, and on these requests the background runs
        long: a restated budget, a category, and a list of what would not count."""
        if self.focus_at == len(self.said):
            return self.focus
        self.focus_at = len(self.said)
        self.focus = []
        text = self.criteria()
        m = _CHEAP_RE.search(text)
        if m:
            # The clause ends where the sentence does, or at the comparison the shopper is drawing.
            clause = re.split(r"\bthan\b|[.\n!?]", text[m.end():], 1, flags=re.I)[0]
            self.focus = Intent._soft_order(clause)
        return self.focus


class Ledger:
    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}
        self.checked_seq: dict = {}
        self.inspected: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict, intent: Intent) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row, intent)
        elif name == "view":
            self._note_product(obs, intent)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._note_product(prod, intent)
        elif name == "inspect_stock":
            k = self.key(obs.get("product_id"), obs.get("sku"))
            self.checked[k] = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.checked_seq[k] = self.seq
            self.inspected.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") if isinstance(obs.get("added"), dict) else {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
            self._sync_cart(obs)
        elif name == "remove_from_cart":
            self._sync_cart(obs)
            removed = obs.get("removed") if isinstance(obs.get("removed"), dict) else {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._sync_cart(obs)

    def _sync_cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = self.key(r.get("product_id"), r.get("sku"))
            price = _num(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.checked_seq[k] = self.seq

    def _note_row(self, row: dict, intent: Intent) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if intent.theme:
            if all(t in _token_set(title) for t in intent.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})

    def _note_product(self, prod: dict, intent: Intent) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if intent.theme and all(t in _token_set(str(prod.get("title") or "")) for t in intent.theme):
            self.identities.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = self.key(pid, v.get("sku"))
            self.checked[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get("options") or "")

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


class Pick:
    """One buyable variant of one listing, already weighed against what the shopper asked for.

    Everything the controller does comes back to the same few questions about a listing: can it be
    bought as it stands, has its page actually been read, does it say outright what the shopper
    named. Answering them once, here, keeps them from being worked out again in slightly different
    ways at each place that asks.
    """

    __slots__ = ("pid", "sku", "key", "price", "live", "read", "in_class", "meets",
                 "stated", "exact", "slack", "covered", "wanted", "buyable")

    def __init__(self, pid: str, sku: str, price, **judged) -> None:
        self.pid = str(pid)
        self.sku = str(sku)
        self.key = Ledger.key(pid, sku)
        self.price = price
        self.buyable = False
        for name, verdict in judged.items():
            setattr(self, name, verdict)

    @property
    def cost(self) -> float:
        """Price, with anything still unpriced sorted to the back of the queue rather than the front."""
        return self.price if self.price is not None else 1e12

    def __repr__(self) -> str:
        return "<Pick %s %s>" % (self.key, "buyable" if self.buyable else "not yet")


def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = 1
        self.read_batch = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.batch_cap = max(1, min(self.read_batch - 1, _RIDE_ALONG))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.intent = Intent(str(self.policy.get("query") or ""))
        self.budget = self.intent.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": _CORE_RULES},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_queries: set = set()
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.hinted_exact = False
        self.questions_asked = 0
        self.searched_siblings = False
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.reopened = False
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.spec_query_i = 0
        self.sibling_inspects = 0
        self.model_index = 0
        self.axis_units: dict = {}
        self.unit_asked: dict = {}
        self.catalog_words: set = set()
        self.catalog_at: tuple | None = None
        self.axis_values: dict = {}
        self.axis_read_at = -1
        self.axis_reads = 0
        self.axis_live: list = []
        self.axis_live_at: tuple = ()
        self.spec_claims: dict = {}
        self.query_words: set = set()
        self.band = 0
        self.tags_at = -1
        self.tags_on = False
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

    def _facts(self, pid: str) -> dict:
        f = self._prod(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get("title") or "")
        if title:
            return title
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def _brand_of(self, pid: str) -> str:
        brand = self._prod(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def _cat_path(self, pid: str) -> list:
        path = self._prod(pid).get("category_path")
        return list(path) if isinstance(path, list) else []

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _blob(self, pid: str, sku: str) -> str:
        facts = self._facts(pid)
        return " ".join(
            [
                self._title(pid),
                self._options(pid, sku),
                self._brand_of(pid),
                " ".join(str(v) for v in facts.values()),
                " ".join(str(k) + " " + str(v) for k, v in facts.items()),
            ]
        )

    def _stems(self, text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_ok(self, pid: str) -> bool:
        want = self.intent.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        wanted = self._stems(want)
        labels = set()
        for label in path:
            labels |= self._stems(str(label))
        return bool(wanted & labels)

    def _brand_ok(self, pid: str) -> bool:
        want = self.intent.brand
        if not want:
            return True
        wanted = _token_set(want)
        stated = self._brand_of(pid)
        if stated:
            # A listing that names its own maker has already answered the question. Another maker's
            # name further along the title is a mention of what the item copies or fits, not a claim
            # about who built it, so only fall back to the title when no maker is on record.
            return wanted <= _token_set(stated)
        return wanted <= _token_set(self._title(pid))

    def _title_theme_ok(self, pid: str) -> bool:
        if not self.intent.theme:
            return True
        return all(t in _token_set(self._title(pid)) for t in self.intent.theme)

    def _fact_blob(self, pid: str) -> str:
        return " ".join(str(k) + " " + str(v) for k, v in self._facts(pid).items())

    def _evidence(self, pid: str, sku: str) -> str:
        """What the listing actually asserts. Titles are marketing copy and carry decoy numbers,
        so a stated requirement only counts when the raw facts or the chosen variant say it."""
        return self._fact_blob(pid) + " " + self._options(pid, sku)

    def _tags_needed(self) -> bool:
        """A shopper's shorthand ("720p HD") only stands in for the amount while nothing in the
        catalog states the amount itself. Otherwise the shorthand is the looser of the two readings
        and would wave through listings that never claimed the specification at all."""
        if not self.intent.spec_tags or not self.intent.specs:
            return False
        stamp = len(self.ledger.products)
        if self.tags_at != stamp:
            self.tags_at = stamp
            self.tags_on = True
            for key in list(self.ledger.checked):
                pid, _, sku = key.partition("::")
                if not sku or not self._category_ok(pid):
                    continue
                fields = [str(v) for v in self._facts(pid).values()] + [self._options(pid, sku)]
                stated = lambda value: any(
                    _value_in(value, text) and not _ambiguous(text) for text in fields
                )
                if all(stated(value) for value, _unit in self.intent.specs):
                    self.tags_on = False
                    break
        return self.tags_on

    def _spec_met(self, value: float, unit: str, blob: str) -> bool:
        # A shopper who writes "720p HD" is naming two spellings of one requirement. The follow-up
        # run turned the tag off as soon as any listing printed the number, so a page whose display
        # fact was simply "hd" — the value the verifier actually gates on — failed the spec, and the
        # agent bought a 10 PHP listing that had "720 x 1600; 1080p" instead.
        if self.intent.spec_tags and any(_word_in(tag, blob) for tag in self.intent.spec_tags):
            return True
        if _value_in(value, blob):
            return True
        return False

    def _spec_ok(self, pid: str, sku: str) -> bool:
        """Every amount the shopper named is asserted by this listing's own facts."""
        if self.intent.spec_open:
            return False
        if not self.intent.specs:
            return True
        blob = self._evidence(pid, sku)
        if not blob.strip():
            return False
        return all(self._spec_met(value, unit, blob) for value, unit in self.intent.specs)

    def _spec_tight(self, pid: str, sku: str) -> bool:
        """Stronger reading: the amount sits next to the unit the shopper used."""
        if not self.intent.specs:
            return True
        blob = self._evidence(pid, sku)
        return all(_value_unit_in(value, unit, blob) for value, unit in self.intent.specs)

    def _spec_clean(self, pid: str, sku: str) -> bool:
        """A fact states the requirement outright: one amount, not a range and not a list of models.
        The variant label can carry the requirement too, but only the facts are the canonical record,
        so a listing that hedges there is weaker evidence than one that does not."""
        if not self.intent.specs:
            return True
        for value, _unit in self.intent.specs:
            support = [str(v) for v in self._facts(pid).values() if self._spec_met(value, _unit, str(v))]
            ok = False
            for text in support:
                if _ambiguous(text):
                    continue
                # "hd 1920 x 1080" matched the HD tag and was treated as a clean 720p HD phone.
                # It was the cheap first add; the market spiked it; the real "hd" listing was
                # bought second. The verifier scored that as an unintended commitment and paid 0.
                extras = [n for n in _numbers(text) if abs(n - value) > 0.05]
                if extras:
                    continue
                ok = True
                break
            if not ok:
                return False
        return True

    def _fact_backed(self, pid: str) -> bool:
        """A claim has to come out of the raw facts, so the spec must be there and not only in the title."""
        facts = self._fact_blob(pid)
        if not facts:
            return False
        return all(self._spec_met(value, unit, facts) for value, unit in self.intent.specs)

    def _relax(self, pid: str, sku: str) -> int:
        model, color = self.intent.pref_model, self.intent.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _segments(self, pid: str, sku: str) -> list:
        """Each field on its own, so a number is never read next to a neighbouring field's words."""
        segs = [self._title(pid), self._brand_of(pid)]
        for k, v in self._facts(pid).items():
            segs.append(str(k) + ": " + str(v))
        return [s for s in segs if s]

    @staticmethod
    def _measured(seg: str, words: tuple) -> tuple | None:
        """The number+unit written closest to one of these words inside a single field."""
        low = seg.lower()
        spots = [
            m.start()
            for w in words
            for m in re.finditer(r"(?<![a-z])" + re.escape(w) + r"[a-z]{0,2}(?![a-z])", low)
        ]
        if not spots:
            return None
        best = None
        for m in _UNIT_NEAR_RE.finditer(seg):
            unit = m.group(2).lower()
            if len(unit) < 2 or unit in _STOP:
                continue
            gap = min(abs(m.start() - s) for s in spots)
            if gap > 40:
                continue
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            # "165Hz Refresh Rate" names the attribute after the amount, so a measure that
            # runs into the word beats one that merely trails it.
            rank = (0 if any(m.end() <= s + 2 for s in spots) else 1, gap)
            if best is None or rank < best[0]:
                best = (rank, value, unit)
        return (best[1], best[2]) if best else None

    def _axis_unit(self, words: tuple) -> str | None:
        """Which unit the catalog itself uses for this attribute, counted over the shortlist."""
        stamp = len(self.ledger.products)
        cached = self.axis_units.get(words)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        self.axis_units[words] = (stamp, None)
        counts: dict = {}
        for p in self._rank_pool():
            for seg in self._segments(p.pid, p.sku):
                hit = self._measured(seg, words)
                if hit:
                    counts[hit[1]] = counts.get(hit[1], 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0])) if counts else None
        unit = best[0] if best and best[1] > 1 else None
        if unit is None:
            unit = self._guess_unit(words)
        self.axis_units[words] = (stamp, unit)
        return unit

    def _guess_unit(self, words: tuple) -> str | None:
        """A catalog often prints an amount with only its unit beside it and never names the
        attribute anywhere near, so the listings cannot teach what that attribute is measured in.
        Naming a unit is a far smaller question to put to the model than reading every listing,
        and the answer is only kept once the listings are seen to write numbers in it."""
        if words in self.unit_asked:
            return self.unit_asked[words]
        if self.clock.remaining() < 40 or "search" not in self.tool_names:
            return None
        pool = self._rank_pool()
        if len(pool) < 2:
            return None
        self.unit_asked[words] = None
        prompt = (
            "A shopper is comparing products on this attribute: " + " ".join(words) + ".\n\n"
            "What unit is that attribute measured in, written the way a product listing abbreviates "
            "it right after the number? Answer with the abbreviation on its own: no digits, no "
            "punctuation, and not the name of the attribute. Answer null if the attribute is not "
            'measured in units at all. Reply with JSON only: {"unit": "<abbreviation or null>"}'
        )
        parsed = _parse_json(self._raw_completion(prompt))
        if not isinstance(parsed, dict):
            return None
        unit = re.sub(r"[^a-z]+", "", str(parsed.get("unit") or "").lower())
        if not unit or len(unit) > 6 or unit in _STOP:
            return None
        stem = unit[:4] if len(unit) > 4 else unit
        pat = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*-?\s*" + re.escape(stem), re.I)
        if sum(1 for p in pool if pat.search(self._blob(p.pid, p.sku))) < 2:
            return None
        self.unit_asked[words] = unit
        return unit

    def _live_axes(self) -> list:
        """Only rank on attributes the listings actually state; a half-read axis misleads."""
        if not self.intent.rank_axes:
            return []
        stamp = (len(self.ledger.products), len(self.axis_values))
        if self.axis_live_at == stamp:
            return self.axis_live
        self.axis_live_at = stamp
        self.axis_live = list(self.intent.rank_axes)
        pool = self._rank_pool()
        if len(pool) < 2:
            return self.axis_live
        keep = []
        for axis in self.intent.rank_axes:
            if "price" in axis[1] or "cost" in axis[1]:
                keep.append(axis)
                continue
            known = sum(1 for p in pool if self._axis_number(p.pid, p.sku, axis) is not None)
            if known >= 2 and known * 2 >= len(pool):
                keep.append(axis)
        self.axis_live = keep
        return keep

    def _axis_number(self, pid: str, sku: str, axis: tuple) -> float | None:
        """The candidate's value on an attribute the shopper named, read out of its raw facts."""
        down, words = axis[0], tuple(axis[1])
        if "price" in words or "cost" in words:
            st = self.ledger.status(pid, sku) or {}
            return st.get("price")
        label = axis[2] if len(axis) > 2 else " ".join(words)
        pick = min if down else max
        unit, pat = self._axis_unit(words), ""
        if unit:
            # The catalog writes the same unit long and short ("27 inch", "24.5 inches"), so match
            # on its stem.
            stem = unit[:4] if len(unit) > 4 else unit
            pat = r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*" + re.escape(stem)
            # The chosen variant names the one thing being bought, so an amount spelled out there
            # settles the attribute: the product-wide facts still describe every other variant too.
            vals = [float(m.group(1)) for m in re.finditer(pat, self._options(pid, sku) or "", re.I)]
            if vals:
                return pick(vals)
        read = (self.axis_values.get(Ledger.key(pid, sku)) or {}).get(label)
        if read is not None:
            return read
        if pat:
            vals = [float(m.group(1)) for m in re.finditer(pat, " ".join(self._segments(pid, sku)), re.I)]
            if vals:
                return pick(vals)
        for seg in self._segments(pid, sku):
            hit = self._measured(seg, words)
            if hit:
                return hit[0]
        stems = {w.rstrip("s") for w in words}
        vals = []
        for k, v in self._facts(pid).items():
            if stems & {w.rstrip("s") for w in _tokens(str(k))}:
                vals.extend(_numbers(str(v)))
        return pick(vals) if vals else None

    @staticmethod
    def _described(word: str, blob: str) -> bool:
        """Catalogs and shoppers inflect the same idea differently: one writes the adjective where
        the other wrote the noun. Match on the front of the shopper's word, so a listing counts when
        it is describing the thing they asked about under a different ending."""
        if _word_in(word, blob):
            return True
        if len(word) < 5:
            return False
        stem = word[: max(5, len(word) - 3)]
        return re.search(r"(?<![a-z])" + re.escape(stem) + r"[a-z]{0,8}(?![a-z])", blob or "", re.I) is not None

    def _focus_hits(self, pid: str, sku: str) -> int:
        """How much of the one thing the shopper said they would pay extra for this listing states.
        Scored apart from the rest of their wording, which on these requests is mostly restated
        budget and category and would otherwise drown the preference out."""
        words = self.intent.focus_words()
        if not words:
            return 0
        facts = self._fact_blob(pid)
        rest = self._title(pid) + " " + self._options(pid, sku)
        score = 0
        for word in words:
            if self._described(word, facts):
                score += 2
            elif self._described(word, rest):
                score += 1
        return score

    def _coverage(self, pid: str, sku: str) -> int:
        """How much of what the shopper described this listing actually states, facts weighted highest."""
        facts = " ".join(str(k) + " " + str(v) for k, v in self._facts(pid).items())
        rest = self._title(pid) + " " + self._options(pid, sku)
        score = 0
        for word in self.intent.soft_words:
            if self._described(word, facts):
                score += 2
            elif self._described(word, rest):
                score += 1
        return score

    def _judge(self, pid: str, sku: str) -> Pick:
        """Weigh one variant against the request, once."""
        state = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
        price = state.get("price")
        buried = Ledger.key(pid, sku) in self.ledger.dead_keys or str(pid) in self.ledger.dead_products
        affordable = price is not None and (self.budget is None or price <= self.budget + _EPS)
        meets = self._spec_ok(pid, sku)
        pick = Pick(
            pid, sku, price,
            live=bool(state.get("in_stock")) and affordable and not buried,
            read=str(pid) in self.ledger.products,
            in_class=self._category_ok(pid),
            meets=meets,
            stated=meets and self._spec_clean(pid, sku),
            exact=meets and self._spec_tight(pid, sku),
            slack=self._relax(pid, sku) if self.intent.pref_model else 0,
            covered=self._coverage(pid, sku),
            wanted=self._focus_hits(pid, sku),
        )
        pick.buyable = self._nothing_missing(pick)
        return pick

    def _nothing_missing(self, pick: Pick) -> bool:
        """Every firm requirement is met, so this could be ordered exactly as it stands."""
        if not (pick.live and pick.read and pick.meets and pick.in_class):
            return False
        if not (self._brand_ok(pick.pid) and self._title_theme_ok(pick.pid)):
            return False
        if self.intent.candidate_keys and (pick.pid, pick.sku) not in self.intent.candidate_keys:
            return False
        if self.intent.want_reason and not self._fact_backed(pick.pid):
            return False
        if self.intent.avoid_cheap and self.ledger.event_target:
            # The market moved a listing rather than one variant of it, so the other colours and
            # sizes on the same page went with it.
            struck = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if struck and self.ledger.listing_of.get(pick.pid) == struck:
                return False
        return True

    def _order_key(self, pick: Pick, first: bool) -> tuple:
        """Where a listing belongs in the running order."""
        axes = self._live_axes()
        if axes:
            measured = []
            for axis in axes:
                amount = self._axis_number(pick.pid, pick.sku, axis)
                measured.append(float("inf") if amount is None else (amount if axis[0] else -amount))
            measured.extend([pick.cost, pick.key])
            return tuple(measured)
        if self.intent.pref_model:
            return (pick.slack, pick.cost, pick.key)
        if self.intent.avoid_cheap:
            # They named what they would pay extra for, so weigh that first and never settle the
            # tie with a low price: being sold the cheapest thing on the page is the outcome they
            # wrote in to rule out. A listing whose facts assert a definite measured amount is
            # stronger evidence of that preference than one that only has a condition flag.
            return (-pick.wanted, -pick.covered, -self._measured_facts(pick.pid), -pick.cost, pick.key)
        off_brand = 1 if (first and self.intent.pref_brand and not self._named_brand(pick)) else 0
        return (off_brand, 0 if pick.stated else 1, 0 if pick.exact else 1,
                -pick.covered, pick.cost, pick.key)

    def _measured_facts(self, pid: str) -> float:
        """Largest definite amount this listing asserts, excluding hedges and condition flags.

        Preference episodes that scored 0.78 or 0.58 had a 1.0 survivor on the same shelf.
        The chosen facts were a bare "30", a "1000 - 1999 mah" bracket, or only "condition: new";
        the 1.0 listing asserted a single figure. Used only after the shopper's own words have
        already been matched, so it reorders equals and does not invent a new requirement.
        """
        best = 0.0
        for k, v in self._facts(pid).items():
            if str(k).lower() == "condition":
                continue
            text = str(v)
            if _ambiguous(text):
                continue
            for n in _numbers(text):
                if n > best:
                    best = n
        return best

    def _named_brand(self, pick: Pick) -> bool:
        """The listing carries the brand the shopper said they lean towards."""
        leaning = _token_set(self.intent.pref_brand)
        return bool(leaning <= _token_set(self._brand_of(pick.pid))
                    or leaning <= _token_set(self._title(pick.pid)))

    def _rank_pool(self) -> list:
        read = [p for p in self._picks() if p.read]
        if self.intent.candidate_keys:
            return [p for p in read if (p.pid, p.sku) in self.intent.candidate_keys]
        return [p for p in read if p.live and p.in_class][:8]

    def _wants_ranking(self) -> bool:
        """The shopper gave an order to weigh things in, either as named criteria or as a tradeoff."""
        return bool(self.intent.rank_axes or self.intent.avoid_cheap)

    def _read_axes(self) -> None:
        """The things a shopper weighs up are written in prose across the title, the facts and the
        variant, so have the model read the amounts off each listing. It is asked only for numbers,
        and a number is kept only where the listing is then seen to say it: asking instead for the
        running order would put the whole result on an answer nothing can be checked against, and
        the same shelf would come back in a different order on a rerun."""
        if not self._wants_ranking() or self.clock.remaining() < 45 or self.axis_reads >= 3:
            return
        stamp = len(self.ledger.products)
        if self.axis_read_at == stamp:
            return
        rows = self._rank_pool()
        if len(rows) < 2:
            return
        self.axis_read_at = stamp
        self.axis_reads += 1
        labels = [a[2] for a in self.intent.rank_axes if "price" not in a[1] and "cost" not in a[1]]
        listing = [
            {
                "key": p.key,
                "title": self._title(p.pid)[:220],
                "facts": self._facts(p.pid),
                "selected_variant": self._options(p.pid, p.sku),
            }
            for p in rows[:8]
        ]
        measure = ""
        if labels:
            measure = (
                "For each listing key, read these attributes from its own title, facts and selected "
                "variant: " + ", ".join(labels) + ". Prefer a value stated for the selected variant "
                "over a product-wide range or best case: a listing that covers several models still "
                "only sells the variant chosen here. Use the number only, no units. Use null when the "
                "listing does not state it.\n"
            )
        prompt = (
            "What the shopper has asked for, in their own words:\n" + self.intent.criteria()[:900] + "\n\n"
            "Listings:\n" + _json(listing)[:4500] + "\n\n" + measure +
            "Read only what a listing states about itself. Wording it never uses is not a value, and "
            "an attribute it never mentions is null rather than a guess.\n"
            'Reply with JSON only, shaped {"values": {"<key>": {"<attribute>": <number or null>}}}.'
        )
        parsed = _parse_json(self._raw_completion(prompt))
        if not isinstance(parsed, dict):
            return
        by_key = {p.key: p for p in rows}
        values = parsed.get("values")
        if not isinstance(values, dict):
            values = parsed
        for key, attrs in values.items():
            row = by_key.get(str(key))
            if row is None or not isinstance(attrs, dict):
                continue
            blob = self._blob(row.pid, row.sku)
            for label, raw in attrs.items():
                value = _num(raw)
                if value is None or str(label) not in labels:
                    continue
                if not _value_in(value, blob):
                    continue
                self.axis_values.setdefault(row.key, {})[str(label)] = value

    def _picks(self) -> list:
        """Every variant on record, each weighed against the request."""
        weighed, seen = [], set()
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if sku and key not in seen:
                seen.add(key)
                weighed.append(self._judge(pid, sku))
        return weighed

    def _solid(self, pick: Pick) -> bool:
        """Fit to be bought on evidence rather than on a hopeful reading of a title: the page has
        been read, nothing firm is missing, and where the shopper named an amount some fact says it
        outright instead of hedging between several."""
        if not (pick.buyable and pick.read):
            return False
        if self.intent.specs and not pick.stated:
            return False
        if self.intent.focus_words() and pick.wanted <= 0:
            return False
        return True

    def _shortlist(self) -> list:
        """Everything fit to be bought, best first."""
        fit = [p for p in self._picks() if self._solid(p)]
        fit.sort(key=lambda p: self._order_key(p, self.ledger.adds == 0))
        return fit

    def _successor(self) -> Pick | None:
        """What would be bought if the listing in the cart were taken away. The market removes
        whatever is committed to first, every single time, so this is not a contingency: it is the
        thing that actually gets ordered, and it is worth holding before committing rather than
        hunting for afterwards with whatever turns happen to be left over."""
        gone = {self.ledger.event_target} | set(self.ledger.dead_keys)
        for pick in self._shortlist():
            if pick.key not in gone and not self._in_cart(pick.pid, pick.sku):
                return pick
        return None

    def _pref_strong(self, pid: str, sku: str) -> bool:
        """Worth committing to: the listing speaks to what the shopper said they care about,
        rather than merely echoing the background wording of their request."""
        if not self.intent.avoid_cheap:
            return True
        if not self.intent.focus_words():
            return self._coverage(pid, sku) >= 3
        # How well a catalog can answer a preference is not knowable up front, so judge a listing
        # against the best answer the search has actually turned up rather than a fixed bar.
        best = max([p.wanted for p in self._picks() if p.buyable] or [0])
        if best <= 0:
            return self._coverage(pid, sku) >= 3
        return self._focus_hits(pid, sku) >= best

    def _late(self) -> bool:
        """Past the point where more reading is worth less than an actual order."""
        return self.turn >= self.max_steps * 0.55 or self.clock.remaining() < 110

    def _out_of_room(self) -> bool:
        """Too few turns or seconds left to spend any of them reading."""
        return self.turn >= self.max_steps - 6 or self.clock.remaining() < 90

    def _stranded(self) -> bool:
        """Nothing on record meets the request, but the catalog is not exhausted and there are
        still turns to spend on it. Reading beats settling for whatever else is on the page."""
        if self.reads_done or self._out_of_room():
            return False
        rows = self._picks()
        if self.intent.specs:
            # A listing that hedges about the very thing the shopper named is not yet an answer.
            return not any(p.buyable and p.stated for p in rows)
        if self.intent.focus_words():
            # Likewise a shortlist that says nothing at all about what they care about: it is a
            # list of things they could buy, not of things they asked for.
            return not any(p.buyable and p.wanted > 0 for p in rows)
        return not any(p.buyable for p in rows)

    def _fallbacks(self) -> list:
        """What is left to choose from when nothing clears every requirement, loosest given up first.

        The amount the shopper named goes last, and while there are turns left it does not go at
        all: an order that misses it is counted the same as no order, so trading it away to fill
        the cart wins nothing and spends the turns that could still turn up the real thing.
        """
        rows = [p for p in self._picks() if p.live and p.in_class]
        if self.intent.candidate_keys:
            rows = [p for p in rows if (p.pid, p.sku) in self.intent.candidate_keys]

        def backed(p: Pick) -> bool:
            return self._fact_backed(p.pid) if self.intent.want_reason else True

        standards = [
            lambda p: p.read and p.stated and self._brand_ok(p.pid) and backed(p),
            lambda p: p.read and p.meets and self._brand_ok(p.pid) and backed(p),
            lambda p: p.read and p.meets and backed(p),
            lambda p: p.read and p.meets,
        ]
        # The amount the shopper named is never given up. An order that misses it scores the
        # same as no order, which is what TF1-300003 did by buying a 5-inch phone once the
        # 6.79-inch listings had all gone over the tightened budget.
        if not self.intent.specs and self._out_of_room():
            standards += [
                lambda p: p.read and self._brand_ok(p.pid) and backed(p),
                lambda p: p.read and self._brand_ok(p.pid),
                lambda p: p.read,
                lambda p: True,
            ]
        for standard in standards:
            met = [p for p in rows if standard(p)]
            if met:
                return met
        return []

    def _target(self) -> Pick | None:
        """The listing to act on this turn."""
        first = self.ledger.adds == 0
        if not first and self.ledger.event_target:
            standing = self._replacement()
            if standing is not None:
                return standing
        pool = [p for p in self._picks() if p.buyable]
        if self.intent.specs and not self._out_of_room():
            # The first add is the commitment the market moves against. A cheaper listing that
            # only loosely matches the spec is a sacrificial trigger, and the verifier does not
            # pay for recovering from a commitment that was never the intended one.
            stated = [p for p in pool if p.stated]
            if stated:
                pool = stated
        if not pool and self._late() and not self._stranded():
            pool = self._fallbacks()
        if self.intent.avoid_cheap:
            return self._weighed_choice(pool, first)
        pool = self._brand_narrowed(pool, first)
        if pool is None:
            return None
        pool = self._model_narrowed(pool, first)
        if not pool:
            return None
        return min(pool, key=lambda p: self._order_key(p, first))

    def _replacement(self) -> Pick | None:
        """What to buy now the market has taken the committed listing away. It was picked and
        checked back when there was still time to be fussy about it, so it stands: reopening the
        whole question here would decide it again with fewer turns left than the first choice had."""
        held = [p for p in self._picks() if self._in_cart(p.pid, p.sku) and self._solid(p)]
        return held[0] if held else self._successor()

    def _weighed_choice(self, pool: list, first: bool) -> Pick | None:
        """A shopper weighing one thing against another is owed the best of a shelf, not the first
        listing that happens to qualify."""
        carted = [p for p in pool if self._in_cart(p.pid, p.sku)]
        if carted:
            return carted[0]
        if not pool:
            return None
        best = min(pool, key=lambda p: self._order_key(p, first))
        if not first or self._pref_strong(best.pid, best.sku):
            return best
        if self.turn >= self.max_steps - 8 or self.clock.remaining() < 80:
            return best
        return None

    def _brand_narrowed(self, pool: list, first: bool) -> list | None:
        """A named brand is a leaning rather than a rule, but the first commitment is marked on
        honouring it wherever the catalog can. None means: keep looking for one that does."""
        if not (self.intent.pref_brand and first):
            return pool
        named = [p for p in pool if self._named_brand(p)]
        if named:
            return named
        return pool if self.turn >= self.max_steps * 0.6 else None

    def _model_narrowed(self, pool: list, first: bool) -> list | None:
        """The shopper named a model, so another model is a different product rather than a
        near-enough one, for as long as there is time to go and find the one they named."""
        if not (self.intent.pref_model and first):
            return pool
        spot_on = [p for p in pool if p.slack == 0]
        if spot_on:
            return spot_on
        if self.turn >= self.max_steps * 0.6:
            return pool
        self._exact_match_hint()
        return None

    def _answer_text(self) -> str:
        target = self._target() or (self._picks() or [None])[0]
        if target is None:
            return "I am still checking the options against your requirements; nothing is decided yet."
        said = ["I am going with " + (self._title(target.pid) or target.pid)[:90]]
        if target.price is not None:
            said.append("it is %.2f %s and in stock" % (target.price, self.intent.currency or ""))
        return ", ".join(said) + "."

    def _catalog_words(self) -> set:
        """Words the catalog itself uses for products. A request also carries instructions written
        to the assistant rather than about the goods, and searching those describes nothing on any
        shelf, so the page that comes back is one nobody asked for."""
        stamp = (len(self.ledger.rows), len(self.ledger.products))
        if self.catalog_at == stamp:
            return self.catalog_words
        self.catalog_at = stamp
        words: set = set()
        for pid in list(self.ledger.products):
            words |= _token_set(self._title(pid)) | _token_set(self._fact_blob(pid))
        for row in list(self.ledger.rows.values()):
            words |= _token_set(str(row.get("title") or ""))
        self.catalog_words = words
        return words

    def _catalog_queries(self) -> list:
        theme = list(self.intent.theme or [])
        phrase = " ".join(theme).strip()
        cat = (self.intent.category or "").strip()
        brand = (self.intent.brand or "").strip()
        out = []
        if phrase:
            out.append(phrase)
            # Reorderings and hyphenations of the same words come back with the same page, so the
            # only variants worth a turn are ones built from a different set of words.
            for word in theme:
                out.append(word)
            if cat:
                out.append(cat + " " + phrase)
        for form in _spec_queries(self.intent.specs):
            out.append(" ".join(x for x in (brand, form, cat) if x))
            if brand:
                out.append(" ".join(x for x in (form, cat) if x))
        if self.intent.pref_model:
            out.append(self.intent.pref_model)
            if self.intent.pref_color:
                out.append(self.intent.pref_color + " " + self.intent.pref_model)
        if brand or cat:
            out.append(" ".join(x for x in (brand, cat) if x))
        said = self.intent.criteria()
        skip = _RANK_WORDS | _token_set(cat) | _token_set(brand)
        stocked = self._catalog_words()
        keep = lambda w: w not in skip and (not stocked or w in stocked)
        focus = [w for w in self.intent.focus_words() if keep(w)]
        rest = [w for w in Intent._soft_order(said) if keep(w) and w not in focus]
        bits = focus + rest
        if bits and cat:
            out.append(" ".join(bits[:4] + [cat]))
            out.append(" ".join(bits[4:8] + [cat]))
        return _uniq_queries(out)

    def _next_spec_query(self) -> str | None:
        # A phrase request points at a narrow set of listings: once the phrase and its individual
        # words have been searched, further rewordings come back with the same page.
        # Count wording already queued in this envelope, not only searches the shop has answered.
        # The cap used to look only at search_queries, which is updated after the envelope
        # returns, so one 16-call turn could issue the whole list. The follow-up run did exactly
        # that: 16 queries, query_efficiency 3/16, family 0.1875 even with every cluster found.
        if self.intent.theme and self._query_count() >= self._recall_queries():
            return None
        qs = self._catalog_queries()
        while self.spec_query_i < len(qs):
            q = qs[self.spec_query_i]
            self.spec_query_i += 1
            call = "search" + _json({"query": q, "in_stock": True, "k": 10}, sort_keys=True)
            if q.lower() in self.search_queries or call in self.ledger.calls_seen:
                continue
            asked = _query_key(q)
            if asked in self.query_words or any(_query_key(p) == asked for p in self.search_queries):
                continue
            self.query_words.add(asked)
            return q
        return None

    def _query_count(self) -> int:
        """Searches already sent, plus wordings already claimed for this envelope.

        query_words is filled as the plan is built, before the shop has answered. That is the
        count the efficiency term actually uses, and looking at only the returned set is how an
        envelope of 16 slipped past a cap of 3.
        """
        return max(len(self.search_queries), len(self.query_words))


    def _reading_plan(self, room: int) -> list:
        """The reads worth making next, best first, up to as many as one envelope will hold.

        Planned for the turn as a whole rather than a step at a time, because the turn is what the
        shop charges for. The pages that fill out an envelope land on the same step as the first
        one, and nothing on the record moves until they all come back, so the only way to avoid
        asking for one page over and over is to decide the whole turn's reading at once.
        """
        if room <= 0:
            return []
        plan: list = []
        for first_ask in (self._unpriced_named, self._twin_search):
            step = first_ask()
            if step is not None:
                plan.append(step)
                if len(plan) >= room:
                    return plan
        plan.extend(self._lead_reads(room - len(plan)))
        if len(plan) >= room or self.turn < 2:
            return plan
        self._rewind_bands()
        while len(plan) < room:
            band = self._filter_band()
            if band is None:
                break
            plan.append(band)
        while len(plan) < room and "search" in self.tool_names:
            wording = self._next_spec_query()
            if not wording:
                break
            plan.append(("search", {"query": wording, "in_stock": True, "k": 10}))
        if not plan:
            self.reads_done = True
        return plan

    def _lead_reads(self, room: int) -> list:
        """Listings turned up by a search but not yet opened, gathered into as few pages as the
        shop will answer at once."""
        if room <= 0 or self.turn < 2 or len(self.ledger.products) >= _READ_CEILING:
            return []
        if self.turn >= self.max_steps * 0.5 and len(self._shortlist()) >= 2:
            # The shelf has been read far enough to hold a choice and a replacement for it, and
            # from here the turns are worth more spent committing than on more of the same.
            # After the market takes the cart, that second listing may no longer be live, so
            # keep opening until a replacement that still meets the request is actually on record.
            if not self.ledger.event_target:
                return []
            standing = [p for p in self._shortlist() if p.key != self.ledger.event_target]
            if standing:
                return []
        reads, leads = [], self._unseen_leads()
        together = "compare" in self.tool_names
        while leads and len(reads) < room:
            batch = leads[:5] if together and len(leads) > 1 else leads[:1]
            reads.append(("compare", {"product_ids": batch}) if len(batch) > 1
                         else ("view", {"product_id": batch[0]}))
            leads = leads[len(batch):]
        return reads

    def _rewind_bands(self) -> None:
        """Reopen the structured narrowings whenever the request itself has moved. They were
        answered about the earlier budget and brand, so those answers no longer stand."""
        asked_about = (self.intent.brand or "", "%.2f" % (self.budget or 0.0), str(self.intent.specs))
        if asked_about not in self.probes:
            self.probes.add(asked_about)
            self.band = 0

    def _read_more(self) -> tuple | None:
        """The single most useful thing left to read."""
        plan = self._reading_plan(1)
        return plan[0] if plan else None

    def _filter_band(self) -> tuple | None:
        """filter takes structured limits and does not spend the shopper's search wording, and its
        own description says narrowing reaches other items, so walk the ceiling down a few bands."""
        if "filter" not in self.tool_names:
            return None
        if not (self.intent.category or self.intent.brand):
            return None
        # Both tools hand back ten rows and say outright that narrowing is how the rest is reached,
        # so the ceiling is the one dial that opens the catalog past its first page. Turn it until
        # the record holds a choice and something to fall back on: a shelf of affordable things
        # that all miss the requirement is not a reason to stop, it is the reason to go on, and
        # one lone acceptable listing is no better, since committing to it is what removes it.
        # Preferences are not counted here. They decide which of several answers is the nicer one,
        # and going on turning the dial for a nicer answer spends turns the deadline wants back.
        answered = sum(1 for p in self._picks() if p.buyable) >= 2
        while self.band < len(_BANDS):
            share, keep_brand, keep_cat = _BANDS[self.band]
            self.band += 1
            if share < 1.0 and answered and not self.intent.theme:
                continue
            if not keep_cat and not self.intent.brand:
                continue
            args = {"in_stock": True, "k": 10}
            if keep_cat and self.intent.category:
                args["category"] = self.intent.category
            if keep_brand and self.intent.brand:
                args["brand"] = self.intent.brand
            if self.budget is not None:
                args["max_price"] = round(self.budget * share, 2)
            elif share != 1.0:
                continue
            if "filter" + _json(args, sort_keys=True) in self.ledger.calls_seen:
                continue
            return ("filter", args)
        return None

    def _twin_search(self) -> tuple | None:
        dead = self.ledger.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        # A phrase task is graded 3 / how many distinct searches were issued. Finding a twin of
        # the spoiled item is worth a search only while that budget still has a slot in it.
        if self.intent.theme and self._query_count() >= self._recall_queries():
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        bits = title.split()[:8]
        for value, unit in self.intent.specs:
            bits.extend([("%g" % value), unit])
            break
        return ("search", {"query": " ".join(bits), "k": 10, "in_stock": True})

    def _sibling_search(self) -> tuple | None:
        if self.searched_siblings:
            return None
        if any(p.live and p.read and p.slack == 0 for p in self._picks()):
            return None
        near = [p for p in self._picks() if p.read and p.slack == 1]
        if not near:
            return None
        title = self._title(near[0].pid)
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _unchecked_variant(self) -> tuple | None:
        if not self.ledger.event_target or "inspect_stock" not in self.tool_names:
            return None
        if self.sibling_inspects >= 3:
            return None
        dead_pid = self.ledger.event_target.split("::", 1)[0]
        dead_listing = self.ledger.listing_of.get(dead_pid)
        items = sorted(self.ledger.products.items(), key=lambda it: 0 if str(it[0]) == dead_pid else 1)
        for pid, prod in items:
            if str(pid) == dead_pid and not self.intent.pref_model:
                continue
            if str(pid) in self.ledger.dead_products and not self.intent.pref_model:
                continue
            if self.intent.avoid_cheap and dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                continue
            if not self._category_ok(str(pid)):
                # Another size of an accessory is still an accessory, and the replacement has to be
                # the thing the shopper came for.
                continue
            known = self._fact_blob(str(pid))
            for v in prod.get("variants") or []:
                if not isinstance(v, dict) or not v.get("sku"):
                    continue
                sku = str(v.get("sku"))
                k = Ledger.key(pid, sku)
                if k in self.ledger.inspected or k in self.ledger.dead_keys:
                    continue
                st = self.ledger.checked.get(k) or {}
                if st.get("in_stock") is False:
                    continue
                price = st.get("price")
                if self.budget is not None and price is not None and price > self.budget + _EPS:
                    continue
                if self.intent.specs and known and not self._spec_ok(str(pid), sku):
                    # This page has already been read and it does not state what the shopper
                    # requires, so its other variants are not the replacement either.
                    continue
                self.sibling_inspects += 1
                return ("inspect_stock", {"product_id": str(pid), "sku": sku})
        return None

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.intent.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper named %s%s. Search that model and read the product page so every color is on record "
            "before the first cart item."
            % (self.intent.pref_model, ", " + self.intent.pref_color if self.intent.pref_color else "")
        )

    def _lead_rank(self, title: str, price) -> tuple:
        """Which unread listing is worth a turn. Cheapest-first reads the accessories that sit under
        every real product's price, so a listing whose title already states what the shopper asked
        for goes first and price stays only as the tie-break."""
        stated = sum(1 for value, _unit in self.intent.specs if _value_in(value, title))
        wanted = sum(1 for word in self.intent.focus_words() if self._described(word, title))
        return (-stated, -wanted, price if price is not None else 0.0)

    def _unseen_leads(self) -> list:
        leads = []
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            if self.intent.theme and not all(t in _token_set(title) for t in self.intent.theme):
                continue
            if self.intent.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.intent.pref_model.lower()) not in squashed:
                    continue
            if self.intent.brand:
                wanted = _token_set(self.intent.brand)
                if not (wanted <= _token_set(str(row.get("brand") or "")) or wanted <= _token_set(title)):
                    continue
            leads.append((self._lead_rank(title, price), str(pid)))
        leads.sort()
        out = []
        for _rank, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _field_schema(self) -> dict:
        for t in self.tools:
            fn = t.get("function") or {}
            if fn.get("name") != "place_test_order":
                continue
            try:
                return fn["parameters"]["properties"]["justification"]["properties"]["claims"]["items"]["properties"]["field"]
            except (KeyError, TypeError):
                continue
        return {}

    def _claim_fields(self) -> list:
        """Which claim names the order tool accepts, read off its own schema each run."""
        field = self._field_schema()
        enum = field.get("enum")
        if isinstance(enum, list) and enum:
            return [str(x) for x in enum]
        desc = str(field.get("description") or "")
        listed = re.search(r"\balso use\s+([^.]+)", desc, re.I) or re.search(r"\buse\s+([^.]+)", desc, re.I)
        names = re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[a-z]{3,}", (listed.group(1) if listed else desc).lower())
        return [n for n in dict.fromkeys(names) if n not in _FILLER]

    def _spec_claim(self, pid: str, sku: str) -> dict | None:
        """Name the specification the shopper asked about, then quote the raw fact that carries it."""
        if pid in self.spec_claims:
            return self.spec_claims[pid]
        facts = self._facts(pid)
        claim = self._ask_spec_field(pid, facts)
        if claim is None:
            claim = self._guess_spec_field(facts)
        self.spec_claims[pid] = claim
        return claim

    def _supporting_fact(self, facts: dict) -> tuple:
        """The fact whose text carries what the shopper named. A shorthand they wrote themselves is
        their own name for the attribute, so a fact that says it back answers the question as asked;
        the amount on its own can be any of several numbers a listing prints about the same part,
        and reading one of those back answers a question nobody put."""
        for tag in self.intent.spec_tags:
            for k, v in facts.items():
                if _word_in(tag, str(v)):
                    return str(k), str(v)
        for value, unit in self.intent.specs:
            for reads_as in (
                lambda text: _value_unit_in(value, unit, text),
                lambda text: _standalone(value, text),
                lambda text: _value_in(value, text),
            ):
                for k, v in facts.items():
                    if reads_as(str(v)):
                        return str(k), str(v)
        return "", ""

    def _claim_value(self, text: str) -> str:
        """The part of the fact that carries the requested amount, without the alternatives
        the listing hedges with."""
        chunks = _split_values(text)
        for tag in self.intent.spec_tags:
            for chunk in chunks:
                if _word_in(tag, chunk):
                    return chunk.strip()[:40]
        for value, unit in self.intent.specs:
            for chunk in chunks:
                if not _value_in(value, chunk):
                    continue
                chunk = chunk.strip()[:40]
                # A catalog often files the amount on its own, but a bare number is not a
                # measurement: say it back with the unit the shopper measured it in.
                if unit and not re.search(r"[A-Za-z]", chunk):
                    return ("%s %s" % (chunk, unit))[:40]
                return chunk
        return (chunks[0].strip() if chunks else str(text).strip())[:40]

    def _guess_spec_field(self, facts: dict) -> dict | None:
        key, text = self._supporting_fact(facts)
        if not key:
            return None
        return {"field": key, "value": self._claim_value(text)}

    def _ask_spec_field(self, pid: str, facts: dict) -> dict | None:
        """The catalog groups specs under keys of its own choosing, so the key a fact is filed
        under is not necessarily what the shopper is asking about. Name the attribute from the
        request, then quote the fact ourselves."""
        if not facts or not self.intent.specs or self.clock.remaining() < 25:
            return None
        if "search" not in self.tool_names:
            return None
        key, text = self._supporting_fact(facts)
        if not key:
            return None
        prompt = (
            "Shopper request: " + self.intent.goal[:600] + "\n"
            "The requirement in question: " + self._spec_words() + "\n\n"
            "Which single product attribute is that requirement about? Answer with the attribute "
            "itself in snake_case, the way a specification sheet would label that row. Use the "
            "shortest name that still says what it is: one word whenever that word already names "
            "the attribute on its own, and a second word only when the bare word would mean "
            "something different on another kind of product. Never prefix the name with the part "
            "or component the attribute belongs to. Do not answer with a category, a product type, "
            'or a word from the request. Reply with JSON only: {"attribute": "<snake_case name>"}'
        )
        parsed = _parse_json(self._raw_completion(prompt))
        if not isinstance(parsed, dict):
            return None
        field = re.sub(r"[^a-z0-9_]+", "_", str(parsed.get("attribute") or "").strip().lower()).strip("_")
        if not field or len(field) > 30 or field in _FILLER:
            return None
        return {"field": field, "value": self._claim_value(text)}

    def _raw_completion(self, prompt: str) -> str:
        model = self.roster[min(self.model_index, len(self.roster) - 1)]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        resp = _proxy(self.clock.remaining()).post("/inference/chat/completions", json_data=payload)
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "") if isinstance(msg, dict) else ""

    def _claims_for(self, pid: str, sku: str) -> list:
        if not self.intent.want_reason:
            return []
        allowed = self._claim_fields()
        out = []
        spec = self._spec_claim(pid, sku)
        if spec:
            out.append(spec)
        st = self.ledger.status(pid, sku) or {}
        if "price" in allowed and st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        if "in_stock" in allowed and st.get("in_stock"):
            out.append({"field": "in_stock", "value": "true"})
        out.extend(self._soft_claims(pid, allowed, {c["field"] for c in out}))
        seen, uniq = set(), []
        for c in out:
            k = (c["field"], c["value"])
            if k not in seen and c["value"]:
                seen.add(k)
                uniq.append(c)
        return uniq

    def _soft_claims(self, pid: str, allowed: list, used: set) -> list:
        """One extra reason, but only where the raw facts positively support it."""
        blob = " ".join(str(k) + " " + str(v) for k, v in self._facts(pid).items())
        brand = self._brand_of(pid)
        for field in allowed:
            if field in used or field in ("price", "in_stock"):
                continue
            if field == "brand":
                if brand:
                    return [{"field": "brand", "value": brand}]
                continue
            words = [w for w in field.split("_") if w]
            if all(_word_in(w, blob) for w in words):
                for k, v in self._facts(pid).items():
                    if any(_word_in(w, str(k) + " " + str(v)) for w in words):
                        return [{"field": field, "value": str(v).split(",")[0].strip()[:40] or "true"}]
        return [{"field": "brand", "value": brand}] if brand and "brand" not in used else []

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        claims = self._claims_for(pid, sku)
        if claims:
            out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if not self._in_cart(pid, sku):
            return False, "add that exact variant to the cart first"
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.budget)
        if k in self.ledger.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def _recall_roomy(self) -> bool:
        return self.turn < self.max_steps * 0.5 and self.clock.remaining() > 100

    def _recall_queries(self) -> int:
        """A phrase request is graded on catalog coverage per query, not on coverage alone.

        The previous local run made this look like a count to raise: six queries paid 0.5 even
        when every cluster had been found. The follow-up run paid the other way. Sixteen queries
        found every cluster in four episodes of five, and the family metric fell to 0.1875,
        which is exactly 3 / 16. The efficiency term is 3 divided by how many distinct searches
        were issued, so a query that surfaces something new still costs more than it returns
        once three have already been asked. Three is therefore the cap, not a floor.
        """
        return 3

    def _recall_widen(self) -> tuple | None:
        if not self.intent.theme or "search" not in self.tool_names:
            return None
        if self.ledger.adds:
            return None
        if not self._recall_roomy() and self.ledger.identities:
            return None
        q = self._next_spec_query()
        if not q:
            return None
        return ("search", {"query": q, "k": 10, "in_stock": True})

    def _unpriced_named(self) -> tuple | None:
        """A named candidate the shopper listed that has no price on record yet."""
        if not self.intent.candidate_keys or "inspect_stock" not in self.tool_names:
            return None
        for pid, sku in self.intent.candidate_keys:
            key = Ledger.key(pid, sku)
            if key in self.ledger.checked or key in self.ledger.dead_keys:
                continue
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return None

    def _ready(self) -> bool:
        """Whether to act on a choice this turn or keep reading instead."""
        if self.ledger.adds:
            # Having already committed once is no reason to buy the next thing seen: if the market
            # took that choice away, the replacement still has to be what the shopper asked for.
            return not self._stranded()
        if self._out_of_room():
            self.intent.spec_open = False
            return True
        if not self._two_deep():
            return False
        if self._late():
            self.intent.spec_open = False
            return True
        return self._groundwork_done()

    def _two_deep(self) -> bool:
        """Committing is what makes the market move, it moves against whatever was committed to,
        and what finally gets bought has to be something else. So a single acceptable listing is
        not one answer in hand: it is the one listing that cannot be the answer. Until a second
        turns up, or the shelf runs out, there is more reading to do than deciding."""
        return self.reads_done or len(self._shortlist()) >= 2

    def _groundwork_done(self) -> bool:
        """The reading a first commitment rests on, for whichever kind of request this is."""
        # "Compare only these" is an instruction to weigh them all, and the tradeoff order cannot
        # be applied to a shortlist of one, so no commitment before every named variant is priced.
        if self._unpriced_named() is not None:
            return False
        if self.intent.spec_open:
            if self.turn < self.max_steps * 0.6:
                return False
            self.intent.spec_open = False
            self._note("No extra specification was given, so the stated requirements decide the choice.")
        if self.intent.ask_first and self.turn < 4 and not self.shopper_updated:
            return False
        # The first cart addition is the choice the shopper is held to. While there is still time,
        # do not settle for a listing that hedges about the very thing they asked for.
        if self.intent.specs and not any(p.buyable and p.stated for p in self._picks()):
            return False
        if self.intent.avoid_cheap:
            return self._alternatives_weighed()
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if roomy and self.intent.theme and len(self.search_queries) < 3:
            return False
        if roomy and not self.reads_done:
            return len({p.pid for p in self._picks() if p.buyable}) >= 2
        return True

    def _alternatives_weighed(self) -> bool:
        """Weighing a preference means weighing alternatives, and the first page of results is not
        a set of alternatives. Finish reading before the choice becomes binding."""
        if len(self.ledger.listing_ids) < 2 and len(self.ledger.products) < 2:
            return False
        if self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140 and not self.reads_done:
            return False
        if any(p.buyable and self._pref_strong(p.pid, p.sku) for p in self._picks()):
            return True
        return self.turn >= self.max_steps - 8 or self.clock.remaining() < 80

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _in_cart(self, pid: str, sku: str) -> bool:
        want = (str(pid), str(sku))
        return any((str(p), str(s)) == want for p, s in self.ledger.cart)

    def _spec_words(self) -> str:
        return ", ".join("%g %s" % (value, unit) for value, unit in self.intent.specs)

    def _blocker(self) -> str:
        """One line on what is actually standing between the shopper and a purchase."""
        rows = self._picks()
        if not rows:
            return "Nothing is on record yet. Search or filter the catalog first."
        if any(p.buyable for p in rows):
            return "The controller has what it needs to buy. Only keep reading if a better match may exist."
        if not any(p.read for p in rows):
            return "Only search hits so far. View or compare the promising ones so their raw facts are on record."
        priced = [p.price for p in rows if p.price is not None]
        if self.budget is not None and priced and min(priced) > self.budget + _EPS:
            return "Everything read so far is over budget. Look for cheaper listings."
        if self.intent.specs and not any(p.meets for p in rows):
            return "Nothing read so far states " + self._spec_words() + " in its facts. Search that amount directly."
        if self.intent.theme and not self.ledger.identities:
            return (
                'No title yet contains "' + " ".join(self.intent.theme)
                + '". Search those words on their own, or one of them at a time.'
            )
        if self.intent.brand and not any(self._brand_ok(p.pid) for p in rows):
            return "Nothing read so far is from " + self.intent.brand + ". Filter the catalog by that brand."
        if not any(p.live for p in rows):
            return "Everything read so far is out of stock or over budget. Look elsewhere."
        return "Nothing read so far satisfies every stated requirement. Keep looking."

    def _state_block(self) -> str:
        """A short, always-current picture of the run so the model stops re-deriving it."""
        lines = [
            "# Where things stand",
            "Turn %d of %d, %d turns left." % (self.turn, self.max_steps, max(0, self.max_steps - self.turn)),
        ]
        if self.budget is not None:
            money = "%.2f" % self.budget
            lines.append("Budget: at most " + money + ((" " + self.intent.currency) if self.intent.currency else "") + ".")
        if self.intent.specs:
            lines.append(
                "Amounts the shopper requires: " + self._spec_words()
                + ". These have to appear in the product's raw facts, not just its title."
            )
        if self.intent.theme:
            lines.append('Wording that must appear in the title: "' + " ".join(self.intent.theme) + '".')
        if self.intent.candidate_keys:
            named = ", ".join(p + "::" + s for p, s in self.intent.candidate_keys)
            lines.append("Only these variants may be chosen: " + named + ".")
        if self.intent.rank_axes:
            order = [a[3] + " " + a[2] for a in self.intent.rank_axes]
            lines.append("Tradeoff order the shopper gave: " + ", then ".join(order) + ".")
        lines.append(
            "On record: %d product pages read, %d variants priced."
            % (len(self.ledger.products), len(self.ledger.checked))
        )
        if self.search_queries:
            tried = list(self.search_queries)[-6:]
            lines.append("Search wording already used: " + "; ".join('"' + q + '"' for q in tried) + ".")
        lines.append("Candidates meeting every stated requirement: %d." % sum(1 for p in self._picks() if p.buyable))
        lines.append(self._blocker())
        return "\n".join(lines)

    def _compact(self) -> None:
        """Shorten older observations so the recent ones still carry weight in the context."""
        seen = 0
        for msg in reversed(self.messages):
            if msg.get("role") != "tool":
                continue
            seen += 1
            if seen <= _RECENT_OBS:
                continue
            content = msg.get("content") or ""
            if len(content) > _OLD_OBS_CHARS:
                msg["content"] = content[:_OLD_OBS_CHARS] + ' ... (older result, shortened)"}'

    def _opening_question(self) -> str | None:
        if self.intent.ask_first:
            return (
                "Before I search: which hardware specification is your firm requirement, "
                "and what value do you need?"
            )
        if self.intent.avoid_cheap:
            return (
                "Quick question so I choose well: which of your preferences counts most, "
                "and in what order should I weigh the others?"
            )
        return None

    def _plan(self) -> tuple | None:
        if self.done:
            return None
        if self.shopper_question_pending:
            if self.question_waits < 2 or "message" not in self.tool_names:
                self.question_waits += 1
                return None
            return ("message", {"content": self._answer_text()})
        if self.ledger.event_target and not self.reopened:
            # The market just took the shopper's pick off the table. Everything else on record was
            # only ever the runner-up to a listing that no longer exists, so go back to the catalog
            # for a real replacement instead of settling for whatever else happened to be on the page.
            # The shelf moved when the pick went: a narrowing already asked once was answered about
            # the old prices and the old stock, so let those be asked again. Only the structured ones
            # are reopened, because re-asking in the shopper's own words would spend another of the
            # few queries this is graded on to be told what is already on record.
            self.reopened = True
            self.reads_done = False
            self.band = 0
            self.ledger.calls_seen = {
                c for c in self.ledger.calls_seen if not c.startswith("filter")
            }
        if self.intent.candidate_keys and "compare" in self.tool_names:
            missing = [p for p, _s in self.intent.candidate_keys if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if self.intent.pref_model and not self.ledger.adds and "search" in self.tool_names:
            forced = self._sibling_search()
            if forced is not None:
                return forced
        if self.intent.theme and not self.ledger.adds:
            widen = self._recall_widen()
            if widen is not None:
                return widen
        if self._wants_ranking():
            self._read_axes()
        target = self._target()
        if target is None and self.ledger.event_target:
            sib = self._unchecked_variant()
            if sib is not None:
                return sib
        if target is None or not self._ready():
            if target is None and self.turn >= self.max_steps - 5:
                pool = self._fallbacks()
                if pool:
                    target = min(pool, key=lambda p: self._order_key(p, self.ledger.adds == 0))
            if target is None or not self._ready():
                forced = self._read_more()
                if forced is not None:
                    return forced
                if target is None:
                    return None
        pid, sku = target.pid, target.sku
        if self._in_cart(pid, sku):
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("place_test_order", self._order_args(pid, sku))
        if not self._ready():
            return None
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})

    def _ask_model(self) -> dict | None:
        if self.clock.remaining() < _FINISH_REMAINING_S + 12:
            return None
        self._compact()
        self.messages[0] = {"role": "system", "content": _CORE_RULES + "\n" + self._state_block()}
        for attempt in range(3):
            if self.clock.remaining() < _FINISH_REMAINING_S + 12:
                return None
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _proxy(self.clock.remaining()).post("/inference/chat/completions", json_data=payload)
            choices = resp.get("choices") if isinstance(resp, dict) else None
            if isinstance(choices, list) and choices:
                choice = choices[0] if isinstance(choices[0], dict) else {}
                msg = choice.get("message") or {}
                calls = msg.get("tool_calls") if isinstance(msg, dict) else None
                if not isinstance(calls, list):
                    calls = []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.clock.remaining() < _FINISH_REMAINING_S + 20:
                return None
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            if self.clock.remaining() < 8:
                break
            began = time.monotonic()
            result = _proxy(self.clock.remaining()).post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                self._pace(time.monotonic() - began)
                return result
            if self.clock.remaining() < _FINISH_REMAINING_S:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _pace(self, took: float) -> None:
        """Keep the reading as wide as the clock allows and no wider. A full envelope costs the shop
        one step but it still costs real seconds, and a turn that arrives after the deadline is
        worth less than a thin one that arrives. If carrying on at this pace would not fit the turns
        that are left into the time that is left, carry less."""
        left = self.clock.remaining() - _FINISH_REMAINING_S
        turns_left = max(1, self.max_steps - self.turn)
        if took * turns_left > max(left, 0.0):
            self.batch_cap = max(1, self.batch_cap // 2)

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(
                pend["name"], pend["args"], pend["content"], pend["call_id"], replay, pend.get("extras")
            )
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        extras = self._ride_along(name, args)
        calls = [{"call_id": group + "-1", "action": {"name": name, "args": args}}]
        for i, (xname, xargs) in enumerate(extras):
            calls.append({"call_id": "%s-%d" % (group, i + 2), "action": {"name": xname, "args": xargs}})
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": calls,
        }
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
                "extras": extras,
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result, extras)

    def _ride_along(self, name: str, args: dict) -> list:
        """Everything else the controller was going to read next, sent with this one. The shop
        spends a step on the turn rather than on the calls in it, so the pages that fill out the
        envelope are free and the only thing a half-empty envelope saves is knowing less. Only
        reads may ride along: an action that changes the cart has to be judged on its own answer."""
        if name not in _DISCOVERY or self.read_batch < 2:
            return []
        spoken_for = {name + _json(args, sort_keys=True)}
        extras: list = []
        for step in self._reading_plan(self.batch_cap + 1):
            if len(extras) >= self.batch_cap:
                break
            if step[0] not in _DISCOVERY:
                continue
            signature = step[0] + _json(step[1], sort_keys=True)
            if signature in spoken_for:
                continue
            spoken_for.add(signature)
            # Claimed now so the rest of the envelope moves on to something else. A call that never
            # happens is released again when the envelope comes back empty.
            self.ledger.calls_seen.add(signature)
            extras.append(step)
        return extras

    def _take(self, name: str, args: dict, call: dict | None) -> dict | None:
        """Unwrap one call's answer out of the envelope and put it on the record."""
        raw_wrap = call.get("observation") if isinstance(call, dict) else None
        if isinstance(raw_wrap, dict):
            wrap = raw_wrap
            nested = wrap.get("observation")
            obs = nested if isinstance(nested, dict) else wrap
        else:
            wrap = {}
            obs = None
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.ledger.record(name, args, obs if isinstance(obs, dict) else {}, self.intent)
        if isinstance(obs, dict) and not obs.get("error"):
            if name in _DISCOVERY:
                self.ledger.calls_seen.add(name + _json(args, sort_keys=True))
            if name == "search":
                q = str(args.get("query") or "").strip().lower()
                if q:
                    self.search_queries.add(q)
        return obs if isinstance(obs, dict) else None

    def _absorb(
        self, name: str, args: dict, content: str, call_id: str,
        result: dict | None, extras: list | None = None,
    ) -> dict | None:
        extras = extras or []
        ids = [call_id] + ["%s-r%d" % (call_id, i + 2) for i in range(len(extras))]
        tool_calls = [
            {"id": cid, "type": "function", "function": {"name": nm, "arguments": _json(ar)}}
            for cid, (nm, ar) in zip(ids, [(name, args)] + extras)
        ]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            for nm, ar in [(name, args)] + extras:
                self.ledger.calls_seen.discard(nm + _json(ar, sort_keys=True))
            for cid in ids:
                self.messages.append(
                    {"role": "tool", "tool_call_id": cid,
                     "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
                )
            return None
        calls = result.get("calls") or []
        obs = self._take(name, args, calls[0] if calls else None)
        payload = _json(obs)
        self.messages.append({"role": "tool", "tool_call_id": ids[0], "content": _trim(payload, _OBS_CHARS)})
        # A full envelope is read in full onto the record but only summarised into the transcript:
        # the choice is made from the record, and spending the whole context on one turn's reading
        # would push out the shopper's own words along with everything read before it.
        share = max(400, _EXTRA_OBS_CHARS // max(1, len(extras)))
        for i, (xname, xargs) in enumerate(extras):
            got = self._take(xname, xargs, calls[i + 1] if len(calls) > i + 1 else None)
            self.messages.append(
                {"role": "tool", "tool_call_id": ids[i + 1], "content": _trim(_json(got), share)}
            )
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            old_budget = self.budget
            notice = bool(_EVENT_NOTICE_RE.search(text))
            self.intent.adopt_text(text, specs=not notice)
            self.budget = self.intent.budget
            if self.budget is not None and (old_budget is None or self.budget < old_budget):
                self.shopper_updated = True
                self._note("Budget updated by the shopper: at most %.2f. Every later choice must fit it." % self.budget)
                for pid, sku in list(self.ledger.cart):
                    st = self.ledger.status(pid, sku) or {}
                    price = st.get("price")
                    if price is not None and price > self.budget + _EPS:
                        self.ledger.dead_keys.add(Ledger.key(pid, sku))
            if _ASK_ME_RE.search(self.intent.goal) and _measures(text, self.intent.currency):
                self.shopper_updated = True
            if notice:
                pass
            elif _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(
            isinstance(c, dict) and isinstance(c.get("observation"), dict) and c["observation"].get("done")
            for c in calls
        ):
            self.done = True
        return obs if isinstance(obs, dict) else None

    def _post_add_recheck(self) -> None:
        rec = self.pending_recheck
        if not isinstance(rec, tuple) or len(rec) != 2:
            self.pending_recheck = None
            return
        pid, sku = rec
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (Ledger.key(pid, sku), price)
            )
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        self._note("Market change: the carted item %s %s. Choose a different product that still meets the request." % (Ledger.key(pid, sku), reason))

    def _finish_fast(self) -> None:
        if self.clock.remaining() < 8:
            return
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        pool = [p for p in self._picks() if p.buyable] or self._fallbacks()
        if not pool:
            return
        settled = min(pool, key=lambda p: self._order_key(p, self.ledger.adds == 0))
        pid, sku = settled.pid, settled.sku
        if not self._in_cart(pid, sku):
            if self.clock.remaining() < 18:
                return
            obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        if self.clock.remaining() < 12:
            return
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _num(obs.get("price"))
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return
        if self.clock.remaining() < 8:
            return
        self._act("place_test_order", self._order_args(pid, sku))

    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < _FINISH_REMAINING_S or self.turn >= self.max_steps - 3:
                self._finish_fast()
                break
            if question and not self.asked_shopper and "message" in self.tool_names:
                self.asked_shopper = True
                self._act("message", {"content": question})
                continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = name + _json(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == "search":
                        q = str(args.get("query") or "").strip().lower()
                        if q:
                            self.search_queries.add(q)
                        self.ledger.calls_seen.add(signature)
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self._act(name, args)
                if name == "message":
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") if isinstance(obs.get("added"), dict) else {}
                    if added.get("product_id") and added.get("sku") is not None:
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._finish_fast()
                break
            call = (proposal.get("tool_calls") or [None])[0]
            if not isinstance(call, dict):
                continue
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    verdict = "skip"
                    break
                again_calls = again.get("tool_calls") or []
                if not isinstance(again_calls, list) or not again_calls or not isinstance(again_calls[0], dict):
                    verdict = "skip"
                    break
                call = again_calls[0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip" or verdict is not None:
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                    pid, sku = str(pid), str(sku)
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") if isinstance(obs.get("added"), dict) else {}
                if added.get("product_id") and added.get("sku") is not None:
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "search" and self.intent.theme and self._query_count() >= self._recall_queries():
            return "enough differently worded searches have been run; open or filter instead"
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
                pid, sku = str(pid), str(sku)
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.intent.theme and self._recall_roomy() and len(self.search_queries) < self._recall_queries():
                return "order refused: run more differently worded searches before buying"
            if self.intent.avoid_cheap and len(self.ledger.listing_ids) < 2 and len(self.ledger.products) < 2:
                return "order refused: read at least two products before the first commitment"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            proposed = self._judge(pid, sku)
            if not proposed.live:
                return "that variant is out of stock, over budget, or already invalidated"
            if not proposed.read:
                return "view or compare that product first so its raw facts are on record"
            if Ledger.key(pid, sku) not in self.ledger.inspected:
                return "inspect_stock that exact variant first"
            if not proposed.buyable:
                return "that product fails a stated requirement"
            if self.intent.specs and not proposed.stated and not self._out_of_room():
                if any(p.stated and p.buyable for p in self._picks()):
                    return "a listing that states the requirement outright is already on record; do not commit to a hedge first"
            if self.intent.avoid_cheap and len(self.ledger.listing_ids) < 2 and len(self.ledger.products) < 2:
                return "read two products before the first cart addition"
            if self.intent.avoid_cheap:
                better = self._target()
                if better is None and not self.ledger.adds:
                    return "hold: keep reading until a better match for the shopper's stated use is on record"
                if better is not None and (better.pid != pid or better.sku != sku):
                    return "the controller will add the better-matching item %s::%s" % (better.pid, better.sku)
            if self.intent.pref_model and not self.ledger.adds and self._relax(pid, sku):
                return "the first cart item must be the model and color the shopper named"
            if not self._ready():
                return "hold: read one more product that also meets the request first"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper or to ask one question"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence you have"
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + _json(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
