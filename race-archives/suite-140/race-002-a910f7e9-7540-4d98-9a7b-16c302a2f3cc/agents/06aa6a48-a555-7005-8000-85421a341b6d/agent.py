"""ORO-Bench shopping agent.

Design: a language model proposes one tool call per turn; a deterministic
controller owns every decision the runtime actually grades. The controller

  * reads product evidence the way the environment does (labelled raw facts
    first, then the selected variant's option text, then the title),
  * only ever commits a fully verified candidate to the cart, because the
    first cart addition is the public commitment the scorer checks,
  * re-inspects that item straight after the addition, since the market moves
    the moment something is carted,
  * recovers to a different, still-valid product when the carted one dies, and
  * orders as soon as a verified item is in the cart.

The model is left in charge of exploration (search wording, which products to
look at, how to answer the shopper); the controller takes over the endgame.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Callable, Iterable

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

# ---------------------------------------------------------------------------
# Grandfather observatory — structural facades (0prompt diversification layer).
# These objects describe and wrap the live workflow. They deliberately do not
# decide which shopping task family is active; Brief._family remains the sole
# owner of that classification.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _GrandfatherRuntimeProfile:
    """Read-only limits used by the observational layer."""

    journal_capacity: int = 72
    preview_characters: int = 420
    token_minimum: int = 2
    timing_capacity: int = 24
    tool_history_capacity: int = 48


_GF_PROFILE = _GrandfatherRuntimeProfile()


@dataclass(frozen=True, slots=True)
class _IngressReading:
    """A small immutable description of a problem entering the agent."""

    problem_id: str
    environment_present: bool
    query_characters: int
    query_tokens: int

    @property
    def has_query(self) -> bool:
        return self.query_characters > 0


@dataclass(frozen=True, slots=True)
class _PhaseReading:
    label: str
    elapsed_ms: float
    completed: bool

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "completed": self.completed,
        }


class _BoundedJournal:
    """Ordered, de-duplicated breadcrumbs with a fixed memory ceiling."""

    __slots__ = ("_entries", "_capacity")

    def __init__(self, capacity: int | None = None) -> None:
        self._entries: list[str] = []
        self._capacity = max(1, int(capacity or _GF_PROFILE.journal_capacity))

    def append(self, label: str) -> "_BoundedJournal":
        mark = str(label or "").strip()
        if mark and (not self._entries or self._entries[-1] != mark):
            self._entries.append(mark)
        overflow = len(self._entries) - self._capacity
        if overflow > 0:
            del self._entries[:overflow]
        return self

    def extend(self, labels: Iterable[str]) -> "_BoundedJournal":
        iterator = iter(labels)
        while True:
            try:
                self.append(next(iterator))
            except StopIteration:
                break
        return self

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def clear(self) -> None:
        self._entries.clear()


class _PhaseClock:
    """Context-manager timing span that never suppresses exceptions."""

    __slots__ = ("label", "_started", "_elapsed", "_completed")

    def __init__(self, label: str) -> None:
        self.label = str(label)
        self._started = 0.0
        self._elapsed = 0.0
        self._completed = False

    def __enter__(self) -> "_PhaseClock":
        self._started = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._elapsed = (time.monotonic() - self._started) * 1000.0
        self._completed = True
        return False

    def reading(self) -> _PhaseReading:
        return _PhaseReading(self.label, self._elapsed, self._completed)


class _MetricsShelf:
    """Bounded phase store, kept separate from decision state."""

    __slots__ = ("_rows",)

    def __init__(self) -> None:
        self._rows: list[_PhaseReading] = []

    def record(self, clock: _PhaseClock) -> "_MetricsShelf":
        reading = clock.reading()
        if reading.completed:
            self._rows.append(reading)
            del self._rows[:-_GF_PROFILE.timing_capacity]
        return self

    def snapshot(self) -> tuple[dict, ...]:
        return tuple(row.as_dict() for row in self._rows)

    def total_ms(self) -> float:
        return round(sum(row.elapsed_ms for row in self._rows), 3)


class _IngressSurveyor:
    """Reads public problem shape without changing or classifying it."""

    @staticmethod
    def query_from(problem: Any) -> str:
        if not isinstance(problem, dict):
            return ""
        environment = problem.get("environment")
        if not isinstance(environment, dict):
            return ""
        policy = environment.get("policy_view")
        return str(policy.get("query") or "") if isinstance(policy, dict) else ""

    @classmethod
    def inspect(cls, problem: Any) -> _IngressReading:
        shell = problem if isinstance(problem, dict) else {}
        query = cls.query_from(shell)
        words = re.findall(r"\b[a-z0-9]+\b", query.lower())
        useful = tuple(word for word in words if len(word) >= _GF_PROFILE.token_minimum)
        return _IngressReading(
            problem_id=str(shell.get("problem_id") or shell.get("id") or "problem"),
            environment_present=isinstance(shell.get("environment"), dict),
            query_characters=len(query),
            query_tokens=len(useful),
        )


class _WorkflowRelay:
    """Shared observational spine for entry, model, and tool boundaries."""

    __slots__ = ("journal", "metrics")

    def __init__(self) -> None:
        self.journal = _BoundedJournal()
        self.metrics = _MetricsShelf()

    def mark(self, stage: str) -> "_WorkflowRelay":
        self.journal.append(stage)
        return self

    def observe_problem(self, problem: Any) -> _IngressReading:
        reading = _IngressSurveyor.inspect(problem)
        self.mark("ingress:" + reading.problem_id)
        return reading

    def observe_clock(self, clock: _PhaseClock) -> None:
        self.metrics.record(clock)

    def health(self) -> dict:
        return {
            "stages": self.journal.snapshot(),
            "timings": self.metrics.snapshot(),
            "total_ms": self.metrics.total_ms(),
        }


_GF_RELAY = _WorkflowRelay()


class _TextWorkbench:
    """Functional text helpers shared by wrappers later in the module."""

    @staticmethod
    def compact(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def words(value: Any) -> frozenset[str]:
        return frozenset(re.findall(r"[a-z0-9]+", str(value or "").lower()))

    @staticmethod
    def clipped(value: Any, limit: int | None = None) -> str:
        text = str(value or "")
        ceiling = max(8, int(limit or _GF_PROFILE.preview_characters))
        return text if len(text) <= ceiling else text[: ceiling - 3] + "..."

    @classmethod
    def overlap(cls, left: Any, right: Any) -> float:
        a, b = cls.words(left), cls.words(right)
        return len(a & b) / max(len(a), len(b)) if a and b else 0.0


_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"\s*(PHP|php|₱|pesos?)?",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_PREF_ITEM_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RE = re.compile(r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)")
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
# wording that describes a specification rather than a maker, dropped before a brand is read
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(?:why|reason|explain|justif)", re.I
)
_CLAIM_FIELDS = ("resolution", "refresh_rate", "screen_size", "ram", "price", "in_stock", "brand")
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# ---------------------------------------------------------------- attributes
# The environment resolves a gated specification from public evidence only: the
# labelled raw facts a product page exposes, the selected variant's option text
# and the listing title. A number counts only when its unit or concept word sits
# next to it, and evidence that disagrees with itself resolves to nothing. These
# helpers mirror that reading so the controller can predict the verdict.
_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")

# gated concept -> the display label its evidence arrives under in product facts
_FACT_LABEL = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate", "ram": "storage"}


def _inch_vals(text: str) -> set:
    return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text or "")}


def _hz_vals(text: str) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text or "")}


def _gb_vals(text: str) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RE.finditer(text or "")
    }


def _pixel_class(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _res_vals(text: str) -> set:
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    text = text or ""
    if _QUARTER_RE.search(text):
        return set()  # quarter HD is a different, much smaller panel
    low = text.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _pixel_class(pair[0], pair[1])
        if cls:
            out.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        out.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        out.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        out.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        out.add("hd")
    return out


_VALUE_SETS = {"screen_size": _inch_vals, "refresh_rate": _hz_vals, "resolution": _res_vals, "ram": _gb_vals}


def _adjacent_vals(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word.

    Titles and variant option strings mix several capacities, so a memory figure
    only counts when the word for it follows the number; storage totals and a
    graphics chip's own memory must never be read as system memory.
    """
    text = text or ""
    if field != "ram":
        return _VALUE_SETS[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}


def _mentions(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if field == "screen_size":
        return bool(_INCH_RE.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(_WXH_RE.search(text) or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I))


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _wanted_value(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    if not field:
        return None
    return _one(_VALUE_SETS[field](text or ""))


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _same_word(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


# ------------------------------------------------------------ device identity
# Some categories are graded on what the listing IS, not what it is named after:
# a case, a spare panel or a keypad handset are not the device a shopper asked
# for. The runtime decides that from the same public evidence the agent sees.
_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_IS_PHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_PHONE_RE = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_PART_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NEGATED_RE = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_BUNDLE_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)


def _identity_text(text: str, title: bool = False) -> str:
    """Strip claims that describe what a listing is NOT, or what ships beside it."""
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def _phone_identity(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    if _PART_ONLY_RE.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _identity_text(raw_title, title=True)
    kind = " ".join(_identity_text(str(facts.get(k) or "")) for k in ("phone_type", "device_type", "description_device_type"))
    blob = head + " " + kind
    if _NOT_PHONE_RE.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _IS_PHONE_RE.search(blob):
        return "yes"
    os_blob = " ".join(
        _identity_text(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RE.search(os_blob):
        return "yes"
    if _OS_RE.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


# Which shape of task this is. Structure decides it wherever structure can: named
# candidate variants in the goal, a stated number of identities to surface, or a
# stated exact model and color. Only the rest fall back to a phrase in the rules.
_FAMILY_SIGNS = (
    ("preference", "listing_id"),
    ("constraint", "exact variant"),
    ("justification", "trustworthy reason"),
    ("intent", "nice-to-haves"),
)

_PLAYBOOK = {
    "intent": (
        "Firm requirements: the category, the named hardware specification, the budget and stock. A "
        "named brand is only a preference: choose it when a fully compliant item of that brand exists, "
        "otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view "
        "or compare to read the exact variant's specification from raw facts. Answer the shopper with "
        "message; a budget the shopper lowers later applies to every later choice."
    ),
    "recall": (
        "Run several differently worded searches (k=10, in_stock=true) built from the requested words and "
        "close synonyms until enough distinct matching products have appeared; every requested word must "
        "appear as a whole word of the title. Then view the strongest candidates so the controller can "
        "verify one and buy it."
    ),
    "constraint": (
        "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for "
        "the specific variant, reading the variant's own option text before any product-wide value. Use "
        "filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant "
        "sells out, move to a different product that still meets every requirement."
    ),
    "preference": (
        "You already asked which preference matters most; use the answer. Search or filter the category "
        "within budget, then compare candidates with different listing_id values, reading raw facts. The "
        "first cart item is graded, so make it the best genuine match. If it is repriced above budget, "
        "recover to a different listing_id."
    ),
    "ranking": (
        "Only the named candidate variants count, and the controller ranks them for you: compare all of "
        "them first so their facts are on the record. Highest advertised maximum refresh rate wins, then "
        "the largest screen, then the lowest price, reading the selected variant's own value before a "
        "product-wide one. After the market moves, the remaining candidates are re-ranked."
    ),
    "recovery": (
        "Find the preferred model and color first: search the model code, then view it so every variant "
        "and color is on the record. The first cart item must be exactly that model and color. If it "
        "disappears, relax as little as possible - the same model in another color before any other "
        "model - and never relax the category, budget or stock."
    ),
    "justification": (
        "Find a product whose own raw facts state the requested specification, in the requested category "
        "and brand, in stock and within budget. View or compare it so its facts are on the record, then "
        "the controller inspects it, carts it and orders it with claims taken only from that evidence."
    ),
    "generic": (
        "Understand the firm requirements, filter or search the catalog, view candidates, and let the "
        "controller verify and buy the best one."
    ),
}

_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn, at "
    "most 30 turns, so be efficient. A deterministic controller checks every proposal and takes over the "
    "cart and the order once a candidate is fully verified, so your job is to surface and read candidates: "
    "search or filter, then view or compare so their raw facts, variant options and prices are on record. "
    "Never add an item to the cart as a placeholder - the first cart addition is graded. Prices and stock "
    "change while you shop; the controller re-checks the carted item and tells you when to move to a "
    "different product. Answer every direct question from the shopper with one short message before "
    "anything else, and never send progress or status updates. Never ask the shopper to raise the budget. "
    "Do not repeat a tool call with identical arguments; one compare of up to five products beats several "
    "view calls."
)


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = m.group(1)
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)]
        phrase = _PHRASE_RE.search(self.goal)
        quoted = _QUOTED_RE.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RE.search(self.rules)
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec(self.goal)
        # A hidden requirement is stated only when the shopper is asked for it.
        self.spec_sealed = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.family = self._family()

    def _family(self) -> str:
        if self.candidate_keys:
            return "ranking"
        if self.required_identities:
            return "recall"
        if self.pref_model:
            return "recovery"
        for name, sign in _FAMILY_SIGNS:
            if sign in self.rules:
                return name
        # The rules text is the primary signal. If it ever reads differently, fall back
        # on what the shopper themself asked for rather than giving up on the task shape.
        if self.theme:
            return "recall"
        if _REASON_RE.search(self.goal):
            return "justification"
        if self.pref_brand:
            return "intent"
        if self.spec_sealed or self.spec_field or self.brand:
            return "constraint"
        return "preference"

    @staticmethod
    def _budget(text: str) -> float | None:
        best = None
        for m in _BUDGET_RE.finditer(text or ""):
            whole = m.group(1).replace(",", "")
            frac = m.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

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
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        if not category:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res_vals(w)]
        return " ".join(keep) or None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RE.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _qualifier(goal: str) -> str | None:
        """A product shape that no shelf name captures, so the title must show it."""
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            return "gaming"
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            return "over ear"
        return None

    @staticmethod
    def _specs(goal: str) -> dict:
        """Every hardware value the request states unambiguously.

        A request can name more than one ("a 27-inch 144Hz monitor") while only
        one of them is the gated requirement, so all of them are kept: the gate is
        honored whichever it turns out to be, and a reason can cite each of them.
        """
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = _wanted_value(field, goal)
            if value:
                out[field] = value
        return out

    def _spec(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        for field, value in self.specs.items():
            return field, value
        return None, None

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = _wanted_value(field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.specs[field] = value
                self.spec_sealed = False
                return field + " = " + value
        return None


# ---------------------------------------------------------------------------
# Requirement and evidence workshop — adapters around the live parsers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RequirementSnapshot:
    category: str | None
    brand: str | None
    preferred_brand: str | None
    budget: float | None
    currency: str
    specification: tuple[str | None, str | None]
    candidate_count: int

    @classmethod
    def capture(cls, brief: "Brief") -> "_RequirementSnapshot":
        return cls(
            category=brief.category,
            brand=brief.brand,
            preferred_brand=brief.pref_brand,
            budget=brief.budget,
            currency=brief.currency,
            specification=(brief.spec_field, brief.spec_value),
            candidate_count=len(brief.candidate_keys),
        )

    def constraints(self) -> dict:
        field_name, field_value = self.specification
        result = {
            "category": self.category,
            "brand": self.brand,
            "budget": self.budget,
            "currency": self.currency,
        }
        if field_name and field_value:
            result["specification"] = {field_name: field_value}
        return {key: value for key, value in result.items() if value is not None}


class _BriefFacade:
    """Wrapper surface over Brief without owning its classification logic."""

    @staticmethod
    def create(query: str) -> "Brief":
        return Brief(query)

    @staticmethod
    def snapshot(brief: "Brief") -> _RequirementSnapshot:
        return _RequirementSnapshot.capture(brief)

    @staticmethod
    def accept_revealed_spec(brief: "Brief", text: str) -> str | None:
        return brief.adopt_spec(text)

    @staticmethod
    def playbook_for(brief: "Brief") -> str:
        return _PLAYBOOK.get(brief.family, _PLAYBOOK["generic"])


class _EvidenceValueCodec:
    """A single dispatch surface over the existing public-evidence readers."""

    _READERS: dict[str, Callable[[str], set]] = {
        "screen_size": _inch_vals,
        "refresh_rate": _hz_vals,
        "resolution": _res_vals,
        "ram": _gb_vals,
    }

    @classmethod
    def read(cls, field_name: str, text: str) -> set:
        reader = cls._READERS.get(field_name)
        return reader(text or "") if reader is not None else set()

    @classmethod
    def read_adjacent(cls, field_name: str, text: str) -> set:
        return _adjacent_vals(field_name, text)

    @classmethod
    def unique(cls, field_name: str, text: str) -> str | None:
        return _one(cls.read(field_name, text))

    @classmethod
    def describes(cls, field_name: str, text: str) -> bool:
        return _mentions(field_name, text)

    @classmethod
    def known_fields(cls) -> tuple[str, ...]:
        return tuple(cls._READERS)


class _IdentityWorkbench:
    """Delegating wrapper for product identity and lexical comparisons."""

    @staticmethod
    def phone_state(facts: dict, title: str, options: str) -> str:
        return _phone_identity(facts, title, options)

    @staticmethod
    def normalized_words(text: str) -> set:
        return _tokens(text)

    @staticmethod
    def equivalent_noun(left: str, right: str) -> bool:
        return _same_word(left, right)

    @staticmethod
    def public_identity_text(text: str, is_title: bool = False) -> str:
        return _identity_text(text, title=is_title)


class _NumberWorkbench:
    """Numeric coercion expressed as a small conversion pipeline."""

    @staticmethod
    def normalize(value: Any) -> str | None:
        if value is None:
            return None
        compact = str(value).replace(",", "").strip()
        return compact or None

    @classmethod
    def parse(cls, value: Any) -> float | None:
        normalized = cls.normalize(value)
        if normalized is None:
            return None
        try:
            return float(normalized)
        except (TypeError, ValueError):
            return None

    @classmethod
    def finite_nonnegative(cls, value: Any) -> float | None:
        parsed = cls.parse(value)
        return parsed if parsed is not None and parsed >= 0 else None


class _PromptComposer:
    """Assembles existing generic instructions from their live components."""

    @staticmethod
    def system_text(brief: "Brief") -> str:
        playbook = _BriefFacade.playbook_for(brief)
        suffix = ""
        if brief.family == "recall" and brief.required_identities:
            suffix = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the controller reports the count after each search."
                % brief.required_identities
            )
        return _CORE_RULES + "\n\nFamily playbook: " + playbook + suffix

    @staticmethod
    def note(text: str) -> dict:
        return {"role": "user", "content": "[note] " + str(text)}


def _num(value: Any) -> float | None:
    return _NumberWorkbench.parse(value)


class Ledger:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        self.cart: list = []
        self.checked: dict = {}        # variant key -> latest price/stock
        self.checked_seq: dict = {}    # variant key -> when that was learned
        self.inspected: set = set()    # variants confirmed by a stock inspection
        self.products: dict = {}       # product id -> view/compare record
        self.rows: dict = {}           # variant key -> summary row from a listing page
        self.options: dict = {}        # variant key -> selected option text
        self.listing_ids: set = set()
        self.listing_of: dict = {}   # product id -> the listing it belongs to
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

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row, brief)
        elif name == "view":
            self._note_product(obs, brief)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief)
        elif name == "inspect_stock":
            k = self.key(obs.get("product_id"), obs.get("sku"))
            self.checked[k] = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.checked_seq[k] = self.seq
            self.inspected.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
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
            removed = obs.get("removed") or {}
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

    def _note_row(self, row: dict, brief: Brief) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if brief.theme:
            if all(t in _tokens(title) for t in brief.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get("sku"):
            return
        k = self.key(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})

    def _note_product(self, prod: dict, brief: Brief) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.products[str(pid)] = prod
        if prod.get("listing_id"):
            self.listing_ids.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if brief.theme and all(t in _tokens(str(prod.get("title") or "")) for t in brief.theme):
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


def _arguments(tool_call: dict) -> dict:
    return _ToolCallCodec.arguments(tool_call)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


# ---------------------------------------------------------------------------
# Tool exchange workshop — structured envelopes and ledger-facing wrappers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ToolProposal:
    call_id: str
    name: str
    arguments: dict = field(default_factory=dict)
    content: str = ""

    @property
    def actionable(self) -> bool:
        return bool(self.name)

    def signature(self) -> str:
        return self.name + ":" + json.dumps(self.arguments, sort_keys=True)


@dataclass(frozen=True, slots=True)
class _ToolObservation:
    name: str
    arguments: dict
    payload: dict | None

    @property
    def failed(self) -> bool:
        return not isinstance(self.payload, dict) or bool(self.payload.get("error"))

    def preview(self) -> str:
        return _TextWorkbench.clipped(
            json.dumps(self.payload, ensure_ascii=False, default=str)
            if self.payload is not None
            else "null"
        )


class _ToolCallCodec:
    """Converts model wire dictionaries to stable internal records."""

    @staticmethod
    def arguments(tool_call: dict) -> dict:
        raw = (tool_call.get("function") or {}).get("arguments", "{}")
        if isinstance(raw, dict):
            return dict(raw)
        try:
            decoded = json.loads(raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @classmethod
    def proposal(cls, response: dict | None) -> _ToolProposal | None:
        if not isinstance(response, dict):
            return None
        calls = response.get("tool_calls") or []
        if not calls or not isinstance(calls[0], dict):
            return None
        call = calls[0]
        function = call.get("function") or {}
        return _ToolProposal(
            call_id=str(call.get("id") or "x"),
            name=str(function.get("name") or ""),
            arguments=cls.arguments(call),
            content=str(response.get("content") or ""),
        )

    @staticmethod
    def assistant_message(proposal: _ToolProposal, original_call: dict) -> dict:
        return {
            "role": "assistant",
            "content": proposal.content,
            "tool_calls": [original_call],
        }

    @staticmethod
    def rejection_message(call_id: str, reason: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": call_id or "x",
            "content": json.dumps({"rejected": str(reason)}),
        }


class _ToolSchemaIndex:
    """Read-only index over the policy-provided tool schema."""

    __slots__ = ("_schemas", "_by_name")

    def __init__(self, schemas: Iterable[dict]) -> None:
        self._schemas = tuple(row for row in schemas if isinstance(row, dict))
        self._by_name: dict[str, dict] = {}
        cursor = 0
        while cursor < len(self._schemas):
            schema = self._schemas[cursor]
            name = str((schema.get("function") or {}).get("name") or "")
            if name:
                self._by_name[name] = schema
            cursor += 1

    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def schema(self, name: str) -> dict | None:
        return self._by_name.get(str(name))

    def supports(self, name: str) -> bool:
        return str(name) in self._by_name

    def snapshot(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))


class _LedgerFacade:
    """Compatibility wrapper around Ledger's mutable evidence store."""

    @staticmethod
    def variant_key(product_id: Any, sku: Any) -> str:
        return Ledger.key(product_id, sku)

    @staticmethod
    def record(
        ledger: "Ledger",
        name: str,
        arguments: dict,
        observation: dict,
        brief: "Brief",
    ) -> None:
        ledger.record(name, arguments, observation, brief)

    @staticmethod
    def status(ledger: "Ledger", product_id: str, sku: str) -> dict | None:
        return ledger.status(product_id, sku)

    @staticmethod
    def cart_pairs(ledger: "Ledger") -> tuple[tuple[Any, Any], ...]:
        return tuple(ledger.cart)

    @staticmethod
    def viewed_ids(ledger: "Ledger") -> frozenset[str]:
        return frozenset(ledger.products)

    @staticmethod
    def evidence_counts(ledger: "Ledger") -> dict[str, int]:
        return {
            "products": len(ledger.products),
            "variants": len(ledger.checked),
            "rows": len(ledger.rows),
            "identities": len(ledger.identities),
            "cart": len(ledger.cart),
        }


class _ActionHistory:
    """Bounded history of tool signatures, separate from validation state."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[str] = []

    def remember(self, proposal: _ToolProposal) -> None:
        self._items.append(proposal.signature())
        del self._items[:-_GF_PROFILE.tool_history_capacity]

    def contains(self, proposal: _ToolProposal) -> bool:
        return proposal.signature() in self._items

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._items)


class _AgentWorkbench:
    """Static wrapper collection over helpers used by the live Agent."""

    @staticmethod
    def parse_arguments(call: dict) -> dict:
        return _ToolCallCodec.arguments(call)

    @staticmethod
    def trim_observation(text: str, limit: int) -> str:
        return _trim(text, limit)

    @staticmethod
    def requirements(brief: "Brief") -> dict:
        return _BriefFacade.snapshot(brief).constraints()

    @staticmethod
    def candidate_status(ledger: "Ledger", pid: str, sku: str) -> dict | None:
        return _LedgerFacade.status(ledger, pid, sku)

    @staticmethod
    def identity(facts: dict, title: str, options: str) -> str:
        return _IdentityWorkbench.phone_state(facts, title, options)


class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_index = _ToolSchemaIndex(self.tools)
        self.tool_names = set(self.tool_index.names())
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = _BriefFacade.create(str(self.policy.get("query") or ""))
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.model_index = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        return _PromptComposer.system_text(self.brief)

    def _opening_question(self) -> str | None:
        if self.brief.family == "constraint" or self.brief.spec_sealed:
            return (
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        if self.brief.family == "preference":
            return (
                "Quick question so I choose well: which of your preferences counts most, and in what "
                "order should I weigh the others? Let me know your priority."
            )
        return None

    def _note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    # ------------------------------------------------------------------ evidence
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

    def _strip_models(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
        for token in re.split(r"[;,]", str(self._facts(pid).get("model") or "")):
            token = token.strip()
            stem = re.split(r"[-_]", token)[0] if token else ""
            if (
                token
                and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", token, re.I)
                and re.search(r"[a-z]", stem, re.I)
                and re.search(r"\d", stem)
                and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            ):
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
        return title

    def _fact_value(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept.

        A reason quoted back to the shopper reads these facts leniently: a fact
        printed as a bare number under a unit-bearing label carries that label's
        unit, so "27" under the size label is 27 inches.
        """
        raw = self._facts(pid).get(_FACT_LABEL[field])
        if not raw:
            return None
        text = str(raw).strip()
        found = _VALUE_SETS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return _one(found)

    def _product_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        vals: set = set()
        conflict = False
        raw = self._facts(pid).get(_FACT_LABEL[field])
        if raw:
            found = _VALUE_SETS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QUARTER_RE.search(str(raw)):
                conflict = True
            vals |= found
        title = self._title(pid)
        if field == "resolution":
            title = self._strip_models(pid, title)
            if _QUARTER_RE.search(title):
                conflict = True
        vals |= _adjacent_vals(field, title)
        return None if conflict else _one(vals)

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        opt = self._options(pid, sku)
        vals = _adjacent_vals(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if _mentions(field, opt):
            return None
        return self._product_value(pid, field)

    def _measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self._variant_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_ok(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for.

        Retailers spell a shelf many ways and nest it differently, so a shared
        word anywhere in the path counts. Where the request names a kind of
        product the catalog has no shelf for, the title has to carry it.
        """
        want = self.brief.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        wanted = self._stems(want)
        labels = set()
        for label in path:
            labels |= self._stems(str(label))
        placed = bool(wanted & labels) or any(
            self._stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        if not placed and "smartphone" in wanted:
            # "Cellphones" and "Mobile Phones" are the same shelf; "Feature Phones" is not.
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed:
            return False
        qualifier = self.brief.cat_qualifier
        if not qualifier:
            return True
        title = self._title(pid)
        words = self._stems(title) | _tokens(title)
        if qualifier == "gaming":
            fast = any(float(v[:-2]) >= 120 for v in _hz_vals(title))
            return "monitor" in words and ("gaming" in words or fast)
        return "over" in words and "ear" in words or "overear" in words or "circumaural" in words

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
        if not _same_word(self.brief.category or "", "smartphone"):
            return True
        verdict = _phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
        if self.brief.family in ("constraint", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._title(pid), re.I))

    def _relax(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.brief.pref_model, self.brief.pref_color
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

    def _assess(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        fam = self.brief.family
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != "recovery")
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        field, want = self.brief.spec_field, self.brief.spec_value
        spec = 2
        if field and want:
            got = self._variant_value(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if fam == "justification":
            ok = ok and not self._resale_title(pid) and bool(self._brand_of(pid))
            if field and want:
                # The reason has to quote the product's own facts, so the facts must state it.
                ok = ok and self._fact_value(pid, field) == want and self._product_value(pid, field) == want
        if fam == "preference" and self.ledger.event_target:
            # A replacement has to come from a different listing, not a twin of the dead one.
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fam == "recall" and self.brief.theme:
            ok = ok and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
        if fam == "ranking":
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
            ok = ok and self._measure(pid, sku, "refresh_rate") is not None and self._measure(pid, sku, "screen_size") is not None
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._relax(pid, sku) if fam == "recovery" else 0,
        }

    def _rank_of(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        fam = self.brief.family
        price = c["price"] if c["price"] is not None else 1e12
        if fam == "ranking":
            hz = self._measure(c["pid"], c["sku"], "refresh_rate") or 0.0
            inch = self._measure(c["pid"], c["sku"], "screen_size") or 0.0
            return (-hz, -inch, price, c["key"])
        if fam == "recovery":
            return (c["relax"], price, c["key"])
        extra = sum(
            1 for f, v in self.brief.specs.items()
            if f != self.brief.spec_field and self._variant_value(c["pid"], c["sku"], f) != v
        )
        proven = 0 if _phone_identity(self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (near, extra, proven, price, c["key"])
        return (extra, proven, price, c["key"])

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        first = self.ledger.adds == 0
        fam = self.brief.family
        good = [c for c in self._candidates() if c["ok"]]
        if fam == "preference":
            # What makes one of these the right answer is a judgement about the
            # shopper's own priorities, so the choice stays with the model; the
            # controller only carries its choice through to a valid order.
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            # Which of these best fits the shopper is a judgement call the model makes
            # better than any keyword rule, so the controller only steps in when the
            # episode would otherwise end with nothing bought.
            if good and self.turn >= self.max_steps - 8:
                good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                         c["price"] if c["price"] is not None else 1e12))
                return good[0]
            return None
        if fam == "intent" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            named = [c for c in good if wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))]
            if named:
                good = named
            elif self.turn < self.max_steps * 0.6:
                self._preferred_brand_hint()
                return None
        if fam == "recovery" and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        field, want = self.brief.spec_field, self.brief.spec_value
        if field and want and self._variant_value(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.brief.currency))
        return ", ".join(parts) + "."

    def _read_more(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
        fam = self.brief.family
        if fam == "recall" and len(self.ledger.identities) < self._recall_need():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if leads and self.forced_compares < 8 and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        # No unread rows left: ask the catalog again under the requirements as they stand now.
        probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True  # the catalog has been asked this already
            return None
        self.probes.add(probe)
        if "filter" in self.tool_names and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            if self.brief.brand:
                args["brand"] = self.brief.brand
            return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.ledger.event_target
        if self.searched_twins or not dead or "search" not in self.tool_names:
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._note(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same requirements."
        )
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
            "graded on that brand, so filter by brand=%s within the budget and read those products before any "
            "other." % (self.brief.pref_brand, self.brief.pref_brand)
        )

    def _sibling_search(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        if self.searched_siblings:
            return None
        rows = [c for c in self._candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self._title(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
        )

    def _unseen_leads(self) -> list:
        """Promising variants whose product page has not been read yet."""
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
            if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    # ------------------------------------------------------------------ guidance
    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._candidates()
        if self.brief.family == "preference" and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self._note(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self._target()
        if best:
            self._note(
                "Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.brief.spec_field, self.brief.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self._note(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _priority_hits(self, pid: str, sku: str) -> int:
        """How much of what the shopper actually asked for this product's own facts mention."""
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(" ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku))
        return len(wants & have)

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        if self._wishes is None:
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.brief.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _budget_from_message(self, text: str) -> None:
        new = Brief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        self._note("Budget updated by the shopper: at most %.2f %s. Every later choice must fit it." % (new, self.brief.currency))
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Ledger.key(pid, sku), price)
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        out = []
        field, want = self.brief.spec_field, self.brief.spec_value
        if not field and self.brief.family == "justification":
            field, want = "resolution", self._fact_value(pid, "resolution")
        if not field or not want:
            return []
        stated = dict(self.brief.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []  # an unprovable claim is worse than none at all
        st = self.ledger.status(pid, sku) or {}
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.family == "justification":
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.ledger.cart:
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

    # ------------------------------------------------------------------ planning
    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
        return self.brief.required_identities + (2 if early else 0)

    def _ready(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        fam = self.brief.family
        if self.ledger.adds:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * 0.6:
                return False
            self.brief.spec_sealed = False  # it was never stated; buy the best of the rest
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fam in ("intent", "constraint") and self.turn < 4 and not self.shopper_updated:
            return False  # a later message can still tighten the budget
        if fam == "preference" and len(self.ledger.listing_ids) < 2:
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if fam == "recall" and roomy:
            # The score counts how much of the market the search surfaced, so a second
            # differently worded search is worth its turn.
            if len(self.ledger.identities) < self._recall_need() or self.search_calls < 2:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self._candidates() if c["ok"]}) < 2:
            # The market moves on the first cart action, so line up a fallback before
            # committing: hunting for one afterwards costs many extra turns. Once no
            # unread candidate remains, waiting only burns turns.
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _plan(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        if self.done:
            return None
        if self.shopper_question_pending:
            # The shopper is owed an answer; the model writes it, but not forever.
            if self.question_waits < 2 or "message" not in self.tool_names:
                self.question_waits += 1
                return None
            return ("message", {"content": self._answer_text()})
        fam = self.brief.family
        if fam == "ranking" and "compare" in self.tool_names:
            missing = [p for p, _s in self.brief.candidate_keys
                       if p not in self.ledger.products and p not in self.unreadable]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        if fam == "recovery" and not self.ledger.adds and self.brief.pref_model and "search" in self.tool_names:
            forced = self._sibling_search()
            if forced is not None:
                return forced
        target = self._target()
        if target is None or not self._ready():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = target["pid"], target["sku"]
        if (pid, sku) in self.ledger.cart:
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

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": 0,
            }
            resp = _PROXY.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            # The proxy applies backpressure; back off rather than hammering it.
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(4):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        self.ledger.record(name, args, obs, self.brief)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            if self.brief.family in ("constraint", "intent") and not self.brief.spec_value:
                stated = self.brief.adopt_spec(text)
                if stated:
                    self.shopper_updated = True
                    self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    # ------------------------------------------------------------------ controller steps
    def _post_add_recheck(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        pid, sku = self.pending_recheck
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
        self.event_turn = self.turn
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0  # the replacement search may need to read more pages
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = {
            "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
            "ranking": "The remaining named candidates are re-ranked; the controller takes the new best.",
            "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
            "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
        }.get(self.brief.family, "Choose a different product that meets every firm requirement.")
        self._note("Market change: the carted item %s %s. " % (Ledger.key(pid, sku), reason) + guidance)
        self._shortlist_note()

    def _finish_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act("place_test_order", self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._candidates() if c["ok"]]
        if not good and self.brief.family != "justification":
            good = [
                c for c in self._candidates()
                if c["live"] and c["spec"] == 2 and self._brand_ok(c["pid"]) and c["cat_ok"]
                and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)
            ]
        if not good:
            return
        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in self.ledger.cart:
            obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get("error"):
                return
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get("in_stock"):
            return
        price = _num(obs.get("price"))
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return
        self._act("place_test_order", self._order_args(pid, sku))

    # ------------------------------------------------------------------ main loop
    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
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
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    # The step is not moving the state on; abandon that candidate.
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
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
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._finish_fast()
                break
            call = proposal["tool_calls"][0]
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
                        self._shortlist_note()
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
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                args = self._order_args(pid, sku)
            if name == "message":
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.search_calls += 1
                if self.brief.family == "recall" and self.brief.required_identities:
                    have, need = len(self.ledger.identities), self._recall_need()
                    self._note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name in ("view", "compare"):
                self._shortlist_note()
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if self.brief.family == "recall" and len(self.ledger.identities) < self._recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
                )
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.brief.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.brief.spec_field, self.brief.spec_value)
                )
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            if self.brief.family == "preference" and len(self.ledger.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            if self.brief.family == "recovery" and not self.ledger.adds and self.brief.pref_model and self._relax(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self._ready():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= 2:
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare"):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None


# ---------------------------------------------------------------------------
# Entry workshop — lifecycle pipeline and compatibility helper registry.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RunEnvelope:
    payload: dict
    reading: _IngressReading
    stages: _BoundedJournal = field(default_factory=_BoundedJournal)

    @classmethod
    def wrap(cls, problem_data: Any) -> "_RunEnvelope":
        payload = problem_data if isinstance(problem_data, dict) else {}
        return cls(payload=payload, reading=_IngressSurveyor.inspect(payload))

    def mark(self, label: str) -> "_RunEnvelope":
        self.stages.append(label)
        return self

    def problem(self) -> dict:
        return self.payload


class _LifecycleStage:
    """One composable pipeline stage."""

    label = "stage"

    def apply(self, envelope: _RunEnvelope) -> _RunEnvelope:
        envelope.mark(self.label)
        return envelope


class _IngressStage(_LifecycleStage):
    label = "ingress"

    def apply(self, envelope: _RunEnvelope) -> _RunEnvelope:
        _GF_RELAY.mark(self.label)
        return super().apply(envelope)


class _ShapeStage(_LifecycleStage):
    label = "shape"

    def apply(self, envelope: _RunEnvelope) -> _RunEnvelope:
        state = "environment-present" if envelope.reading.environment_present else "environment-missing"
        envelope.mark(self.label + ":" + state)
        _GF_RELAY.mark(self.label + ":" + state)
        return envelope


class _DispatchStage(_LifecycleStage):
    label = "dispatch"

    def apply(self, envelope: _RunEnvelope) -> _RunEnvelope:
        envelope.mark(self.label)
        return envelope


class _LifecyclePipeline:
    """Runs independent ingress stages before the unchanged agent core."""

    __slots__ = ("_stages",)

    def __init__(self, stages: Iterable[_LifecycleStage] | None = None) -> None:
        self._stages = tuple(stages or (_IngressStage(), _ShapeStage(), _DispatchStage()))

    def prepare(self, problem_data: Any) -> _RunEnvelope:
        envelope = _RunEnvelope.wrap(problem_data)
        cursor = 0
        while cursor < len(self._stages):
            envelope = self._stages[cursor].apply(envelope)
            cursor += 1
        return envelope

    def labels(self) -> tuple[str, ...]:
        return tuple(stage.label for stage in self._stages)


class _CoreRunner:
    """Delegates execution to the historical core and measures only duration."""

    __slots__ = ("_relay",)

    def __init__(self, relay: _WorkflowRelay | None = None) -> None:
        self._relay = relay or _GF_RELAY

    def execute(self, envelope: _RunEnvelope) -> list:
        self._relay.mark("core:begin")
        with _PhaseClock("agent-core") as clock:
            result = _run_agent_core(envelope.problem())
        self._relay.observe_clock(clock)
        self._relay.mark("core:end")
        return result


class _GrandfatherConductor:
    """Top-level orchestration facade; task classification remains inside Brief."""

    __slots__ = ("_pipeline", "_runner")

    def __init__(
        self,
        pipeline: _LifecyclePipeline | None = None,
        runner: _CoreRunner | None = None,
    ) -> None:
        self._pipeline = pipeline or _LifecyclePipeline()
        self._runner = runner or _CoreRunner()

    def run(self, problem_data: Any) -> list:
        envelope = self._pipeline.prepare(problem_data)
        _GF_RELAY.observe_problem(envelope.problem())
        return self._runner.execute(envelope)

    def describe(self) -> dict:
        return {
            "pipeline": self._pipeline.labels(),
            "relay": _GF_RELAY.health(),
        }


class _HelperRegistry:
    """Lazy symbolic wrapper table for diagnostics and compatibility."""

    _TABLE: dict[str, Callable] = {}

    @classmethod
    def table(cls) -> dict[str, Callable]:
        if not cls._TABLE:
            cls._TABLE = {
                "number": _NumberWorkbench.parse,
                "tokens": _IdentityWorkbench.normalized_words,
                "same_word": _IdentityWorkbench.equivalent_noun,
                "phone_identity": _IdentityWorkbench.phone_state,
                "tool_arguments": _ToolCallCodec.arguments,
                "text_overlap": _TextWorkbench.overlap,
                "brief": _BriefFacade.create,
                "brief_snapshot": _BriefFacade.snapshot,
                "variant_key": _LedgerFacade.variant_key,
                "evidence_value": _EvidenceValueCodec.read,
                "entry": _GrandfatherConductor().run,
            }
        return cls._TABLE

    @classmethod
    def invoke(cls, label: str, *args: Any, **kwargs: Any) -> Any:
        helper = cls.table().get(str(label))
        if helper is None:
            return {"error": "unknown helper", "label": str(label)}
        return helper(*args, **kwargs)

    @classmethod
    def labels(cls) -> tuple[str, ...]:
        return tuple(sorted(cls.table()))


class _CompatibilityBridge:
    """Named wrappers over original helper surfaces."""

    @staticmethod
    def parse_number(value: Any) -> float | None:
        return _num(value)

    @staticmethod
    def parse_arguments(call: dict) -> dict:
        return _arguments(call)

    @staticmethod
    def trim_payload(text: str, limit: int) -> str:
        return _trim(text, limit)

    @staticmethod
    def ledger_key(pid: Any, sku: Any) -> str:
        return Ledger.key(pid, sku)

    @staticmethod
    def run(problem_data: dict) -> list:
        return _GrandfatherConductor().run(problem_data)

    @staticmethod
    def health() -> dict:
        return {
            "module": __name__,
            "helpers": _HelperRegistry.labels(),
            "workflow": _GF_RELAY.health(),
        }


def _run_agent_core(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


def _grandfather_health_probe() -> dict:
    return _CompatibilityBridge.health()


def _grandfather_invoke(label: str, *args: Any, **kwargs: Any) -> Any:
    return _HelperRegistry.invoke(label, *args, **kwargs)


def agent_main(problem_data: dict) -> list:
    """Sandbox entry routed through the diversified lifecycle facade."""
    return _GrandfatherConductor().run(problem_data)


GrandfatherConductor = _GrandfatherConductor
GrandfatherHelperRegistry = _HelperRegistry
GrandfatherCompatibilityBridge = _CompatibilityBridge


__all__ = [
    "agent_main",
    "GrandfatherConductor",
    "GrandfatherHelperRegistry",
    "GrandfatherCompatibilityBridge",
]
