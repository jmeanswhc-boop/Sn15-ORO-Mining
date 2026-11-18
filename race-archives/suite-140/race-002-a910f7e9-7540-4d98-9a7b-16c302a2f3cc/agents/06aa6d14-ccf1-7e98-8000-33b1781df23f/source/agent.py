"""ORO Bench EnvPack agent — hioro controller + ELON product DNA.

Runtime is EnvPack only (POST /inference/chat/completions + /environment/call).
ELON ShoppingBench engines are omitted. Product habits live in the playbook
and per-family ladders. Ladders stay short so a miss cannot burn the episode.
"""
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


def _noop(*_a: Any, **_k: Any) -> None:
    return None


# ELON OpenRouter parse/rank slugs + matching Chutes TEE.
# Live roster is 3 deep: first model does the work; glm is last, never first
# on justification (remote TF7 agent_error / steps=0).
_OR_DEEPSEEK = "deepseek/deepseek-v3.2"
_OR_QWEN = "qwen/qwen3.5-397b-a17b"
_OR_KIMI = "moonshotai/kimi-k2.6"
_CH_DEEPSEEK = "deepseek-ai/DeepSeek-V3.2-TEE"
_CH_QWEN = "Qwen/Qwen3.5-397B-A17B-TEE"
_CH_KIMI = "moonshotai/Kimi-K2.6-TEE"

_FAMILY_MODELS = {
    "openrouter": {
        "intent": (_OR_DEEPSEEK, _OR_KIMI, _OR_QWEN),
        "retrieval": (_OR_DEEPSEEK, _OR_QWEN, _OR_KIMI),
        "recall": (_OR_DEEPSEEK, _OR_QWEN, _OR_KIMI),
        "constraint": (_OR_DEEPSEEK, _OR_KIMI, _OR_QWEN),
        "preference": (_OR_KIMI, _OR_DEEPSEEK, _OR_QWEN),
        "ranking": (_OR_DEEPSEEK, _OR_QWEN, _OR_KIMI),
        "recovery": (_OR_DEEPSEEK, _OR_KIMI, _OR_QWEN),
        "justification": (_OR_DEEPSEEK, _OR_KIMI, _OR_QWEN),
        "generic": (_OR_DEEPSEEK, _OR_KIMI, _OR_QWEN),
    },
    "chutes": {
        "intent": (_CH_DEEPSEEK, _CH_KIMI, _CH_QWEN),
        "retrieval": (_CH_DEEPSEEK, _CH_QWEN, _CH_KIMI),
        "recall": (_CH_DEEPSEEK, _CH_QWEN, _CH_KIMI),
        "constraint": (_CH_DEEPSEEK, _CH_KIMI, _CH_QWEN),
        "preference": (_CH_KIMI, _CH_DEEPSEEK, _CH_QWEN),
        "ranking": (_CH_DEEPSEEK, _CH_QWEN, _CH_KIMI),
        "recovery": (_CH_DEEPSEEK, _CH_KIMI, _CH_QWEN),
        "justification": (_CH_DEEPSEEK, _CH_KIMI, _CH_QWEN),
        "generic": (_CH_DEEPSEEK, _CH_KIMI, _CH_QWEN),
    },
}
_PROXY = ProxyClient(timeout=28, max_retries=1, retry_delay=0.4)
_PROBLEM_BUDGET_S = 240.0
_CTRL_FAMS = frozenset({"intent", "preference", "ranking", "recovery"})
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9
_WEAK_SEARCH_CAP = 4
_CTRL_SEARCH_CAP = 6
_CLAIM_CORE = frozenset({"price", "in_stock", "brand"})
_CLAIM_DENY = frozenset({"category", "category_path", "operating_system", "os"})
_ASK_PERMIT_RE = re.compile(
    r"\b(should i|shall i|may i|would you (?:like|prefer)|do you want|is it ok(?:ay)?|can i)\b",
    re.I,
)

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|under|below|no more than|not more than|max(?:imum)? of|"
    r"within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
    + _NUM
    + r"\s*([A-Za-z]{3}|[$€£¥₱])?",
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
    r"\d+(?:\.\d+)?\s*-?\s*[a-z]{1,12}\b",
    re.I,
)
_REASON_RE = re.compile(
    r"\b(?:why|reason|explain|justif)", re.I
)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)

# ---------------------------------------------------------------- attributes
# Read what the shopper wrote and what a listing printed. A number counts only
# when a unit word sits next to it. Do not map pixel pairs or class nicknames;
# do not keep a catalog of product types. Compare the request's own tokens.
_MEASURE_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-–]?\s*([a-z]{1,12}|[\"”″''])?",
    re.I,
)
_INCH_WORDS = frozenset({"in", "inch", "inches", "\"", "”", "″", "''"})


class _Axle:
    """Factory namespace: nested gauges, clocks, and getattr desks."""

    class _Gauge:
        """Generic readers: tokens, number+unit pairs, and the request's own wording."""

        @staticmethod
        def tokens(text: str) -> set:
            return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

        @staticmethod
        def one(values: set) -> str | None:
            return next(iter(values)) if len(values) == 1 else None

        @staticmethod
        def unit_fold(unit: str) -> str:
            raw = (unit or "").strip().lower()
            if raw in _INCH_WORDS:
                return "in"
            return raw

        @classmethod
        def measures(cls, text: str) -> set:
            """Number+unit pairs a string actually printed. No class-name mapping."""
            out = set()

            def take(match) -> None:
                unit = cls.unit_fold(match.group(2) or "")
                if not unit:
                    return
                try:
                    out.add("%g%s" % (float(match.group(1)), unit))
                except ValueError:
                    return

            for m in _MEASURE_RE.finditer(text or ""):
                take(m)
            return out

        @classmethod
        def unit_of(cls, field: str | None) -> str:
            name = (field or "").lower()

            def as_inch() -> str:
                return "in" if ("screen" in name or name.endswith("size")) else ""

            def as_hz() -> str:
                return "hz" if "refresh" in name else ""

            def as_gb() -> str:
                return "gb" if name in {"ram", "memory"} else ""

            for stage in (as_inch, as_hz, as_gb):
                hit = stage()
                if hit:
                    return hit
            return ""

        @classmethod
        def for_field(cls, field: str | None, text: str) -> set:
            found = cls.measures(text)
            unit = cls.unit_of(field)
            if not unit:
                return found
            return {v for v in found if v.endswith(unit)}

        @classmethod
        def adjacent(cls, field: str | None, text: str) -> set:
            """Values a free-form string asserts for the request's concept."""
            if (field or "").lower() == "resolution":
                return set()
            return cls.for_field(field, text)

        @classmethod
        def mentions(cls, field: str | None, text: str) -> bool:
            """Whether option text talks about the concept's own unit word."""
            text = text or ""
            unit = cls.unit_of(field)
            if unit == "in":
                return bool(re.search(r"inch|\bin\b|[\"”″]", text, re.I))
            if unit:
                return unit in text.lower()
            return False

        @classmethod
        def wanted(cls, field: str | None, text: str) -> str | None:
            """The single value the request states for a concept, or nothing when unclear."""
            if not field:
                return None
            if field == "resolution":
                m = re.search(r"\b(\d{3,4}p)\b", text or "", re.I)
                return m.group(1).lower() if m else None
            return cls.one(cls.for_field(field, text or ""))

        @classmethod
        def from_request(cls, text: str) -> dict:
            """Concepts the shopper themselves named, keyed by the words they used."""
            text = text or ""
            out = {}

            def pull(field: str, unit_hits: set, cue) -> None:
                if len(unit_hits) == 1 and re.search(cue, text, re.I):
                    out[field] = next(iter(unit_hits))

            pull("screen_size", cls.for_field("screen_size", text), r"screen|inch")
            pull("refresh_rate", cls.for_field("refresh_rate", text), r"refresh|\bhz\b")
            pull("ram", cls.for_field("ram", text), r"\bram\b|memory")
            m = re.search(r"\b(\d{3,4}p)\b", text, re.I)
            if m and re.search(r"resolution|display|\d{3,4}p", text, re.I):
                out["resolution"] = m.group(1).lower()
            return out

    @staticmethod
    def clock(budget: float):
        class _Tick:
            def __init__(self) -> None:
                self.start = time.monotonic()
                self.budget = budget

            def remaining(self) -> float:
                return self.budget - (time.monotonic() - self.start)

        return _Tick()


_Lens = _Axle._Gauge


_adjacent_vals = _Lens.adjacent
_mentions = _Lens.mentions
_one = _Lens.one
_tokens = _Lens.tokens


# Which shape of task this is. Structure decides it wherever structure can: named
# candidate variants in the goal, a stated number of identities to surface, or a
# stated exact model and color. Only the rest fall back to a phrase in the rules.

_FAMILY_SIGNS = (
    ("preference", "listing_id"),
    ("preference", "soft preferences matter"),
    ("constraint", "exact variant"),
    ("constraint", "every requirement must hold for the exact variant"),
    ("intent", "nice-to-haves"),
    ("intent", "mix of firm requirements and nice-to-haves"),
    ("justification", "trustworthy reason"),
    ("justification", "walk me through why"),
    ("retrieval", "survey the realistic options"),
    ("ranking", "highest advertised maximum refresh rate"),
    ("recovery", "relaxable only after the exact item becomes unavailable"),
)
_CAT_SHOPPER_RE = re.compile(
    r"(?:catalog\s+)?category is\s+([A-Za-z][\w&'\- ]{1,40}?)(?:\.|$|,|;)",
    re.I,
)
_RESALE_RE = re.compile(
    r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
    re.I,
)
# Process / filler words — not product types. Used so preference search reads the
# shopper's use-case, not the task-rules boilerplate or the "place a test order" frame.
_WISH_DROP = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
        "order", "stock", "place", "under", "over", "one", "test", "buy", "rather", "than",
        "pay", "bit", "more", "settle", "something", "its", "would", "like", "have", "from",
        "into", "just", "grab", "cheap", "cheapest", "isn", "goal", "here", "actually",
        "care", "about", "lots", "lot", "few", "dollars", "saving", "dont", "only",
        "instock", "listing", "every", "kind", "then", "they", "them", "what", "when",
        "will", "does", "did", "can", "should", "could", "please", "also", "very",
        "much", "some", "any", "all", "not", "but", "because", "survives", "flimsy",
        "daily", "hard", "heavy", "year", "round", "sessions",
        "required", "catalog", "category", "rules", "task", "firm",
        "soft", "preference", "preferences", "priority", "priorities", "classification",
        "justification", "specification", "structured", "optional", "claims", "field",
        "value", "note", "shopper", "agent", "matters", "saving",
    }
)

_CORE_RULES = (
    "You are a careful shopping agent working a catalog through tools. Exactly one tool call per turn. "
    "A deterministic controller checks every proposal and takes over cart and order once a candidate is "
    "fully verified. Your job is to surface and read candidates.\n"
    "Firm vs soft (ELON parse habit): budget, in-stock, category, and a named hardware value are firm. "
    "A preferred brand is soft unless the rules say otherwise. Numeric ranges next to SPF / wattage / "
    "age / capacity are attributes, not prices.\n"
    "Keywords: short 2–5 word queries; keep value-side tokens (type, brand, model, number, unit). Drop "
    "filler and schema-key words (size, color, brand as keys). in_stock=true when buying. After 3–5 "
    "weak searches, stop thrashing and finish.\n"
    "Read before you commit. view/compare so raw facts, variant options and prices are on record. After "
    "view, literal-match brand/model/size/number; do not reject a confirmed product_type. Title is an "
    "unordered token bag. Attributes beat title guesses.\n"
    "The first add_to_cart is graded — never a placeholder. After every cart change, inspect_stock that "
    "SKU. Recover by COMMIT (never ask permission), then remove → search model → inspect_stock → cart → "
    "order. Never order an out-of-stock SKU. Never ask the shopper to raise the budget.\n"
    "Justification only if asked: claims are {field, value} from the last view + latest inspect_stock of "
    "the final product. Allow the named spec, price, in_stock, and a real brand. Never category. Never "
    "invent a specification class the listing did not print.\n"
    "If the goal lists pid::sku pairs, only those pairs may be added or ordered. Answer every direct "
    "shopper question with one short message before anything else. Do not repeat identical tool calls."
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
        "the controller inspects it, carts it and orders it with claims taken only from that evidence. "
        "Quote the listing's own words; do not translate them into a class name it did not print."
    ),
    "generic": (
        "Understand the firm requirements, filter or search the catalog, view candidates, and let the "
        "controller verify and buy the best one."
    ),
}

def _make_clock(budget: float):
    return _Axle.clock(budget)


class Brief:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = ""
        cm = re.search(
            r"(?:at or under|at or below|up to|under|below|at most|within)\s+"
            + _NUM
            + r"\s*([A-Za-z]{3})\b",
            self.goal,
            re.I,
        )
        if cm:
            self.currency = cm.group(3).upper()
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
        self.cat_qualifier = self._qualifier(self.goal, self.category)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec(self.goal)
        # A hidden requirement is stated only when the shopper is asked for it.
        self.spec_sealed = self.spec_field is None and "firm requirement" in self.goal.lower()
        self.family = self._family()

    def _family(self) -> str:
        class _Kind:
            __slots__ = ("brief",)

            def __init__(self, brief: "Brief") -> None:
                self.brief = brief

            def structured(self) -> str:
                host = self.brief
                if host.candidate_keys:
                    return "ranking"
                if host.required_identities:
                    return "recall"
                if host.pref_model:
                    return "recovery"
                if host.pref_brand and "nice-to-have" in host.rules:
                    return "intent"
                return ""

            def signed(self) -> str:
                for name, sign in _FAMILY_SIGNS:
                    if sign in self.brief.rules:
                        return name
                return ""

            def inferred(self) -> str:
                # The rules text is the primary signal. If it ever reads differently, fall back
                # on what the shopper themself asked for rather than giving up on the task shape.
                host = self.brief
                if host.theme:
                    return "recall"
                if _REASON_RE.search(host.goal):
                    return "justification"
                if host.pref_brand:
                    return "intent"
                if host.spec_sealed or host.spec_field or host.brand:
                    return "constraint"
                return "preference"

            def resolve(self) -> str:
                for name in ("structured", "signed", "inferred"):
                    hit = getattr(self, name)()
                    if hit:
                        return hit
                return "preference"

        return _Kind(self).resolve()

    class _Scan:
        @staticmethod
        def budget(text: str) -> float | None:
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
        def category(goal: str) -> str | None:
            m = _CAT_PHRASE_RE.search(goal or "")
            if m:
                return m.group(1).strip()
            m = re.search(
                r"\ban?\s+(?:in-stock\s+)?([A-Za-z][A-Za-z\-]{2,20})\s+(?:with|at or|under|for)\b",
                goal or "",
                re.I,
            )
            if m:
                return m.group(1)
            m = re.search(
                r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
                goal or "",
                re.I,
            )
            word = m.group(1) if m else None
            if word and word.casefold() in {"screen", "inch", "inches", "stock", "order", "test"}:
                return None
            return word

        @staticmethod
        def brand(goal: str, category: str | None) -> str | None:
            """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
            if not category:
                return None
            m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
            if not m:
                return None
            words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
            keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _Lens.measures(w)]
            return " ".join(keep) or None

        @staticmethod
        def pref_brand(goal: str) -> str | None:
            m = _PREF_BRAND_RE.search(goal or "")
            if not m:
                return None
            name = m.group(1).strip(" .,")
            words = name.split()
            if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
                return None
            return name

        @staticmethod
        def qualifier(goal: str, category: str | None) -> str | None:
            """A modifier the shopper put in front of the named category, if any."""
            if not category:
                return None
            m = re.search(
                r"\b([A-Za-z][A-Za-z\-]+)\s+" + re.escape(category) + r"s?\b",
                goal or "",
                re.I,
            )
            if not m:
                return None
            word = m.group(1).strip().casefold()
            if word in {
                "a", "an", "the", "one", "in", "stock", "in-stock", "instock", "need", "want",
                "find", "get", "buy", "order", "cheap", "cheapest", "available",
            }:
                return None
            return word

        @staticmethod
        def specs(goal: str) -> dict:
            """Values the request itself states, keyed by the shopper's own concept words."""
            return _Lens.from_request(goal or "")

    _budget = staticmethod(_Scan.budget)
    _category = staticmethod(_Scan.category)
    _brand = staticmethod(_Scan.brand)
    _pref_brand = staticmethod(_Scan.pref_brand)
    _qualifier = staticmethod(_Scan.qualifier)
    _specs = staticmethod(_Scan.specs)

    def _spec(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        for field, value in self.specs.items():
            return field, value
        return None, None

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        found = _Lens.from_request(text or "")
        for field, value in found.items():
            self.spec_field, self.spec_value = field, value
            self.specs[field] = value
            self.spec_sealed = False
            return field + " = " + value
        return None

    def adopt_category(self, text: str) -> str | None:
        """Record the catalog shelf the shopper names later; returns that name."""
        m = _CAT_SHOPPER_RE.search(text or "")
        if not m:
            return None
        name = m.group(1).strip()
        if not name or name.casefold() in {"the", "a", "an", "required"}:
            return None
        self.category = name
        return name


class _Wire:
    @staticmethod
    def num(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def provider() -> str:
        explicit = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        if explicit in {"openrouter", "chutes"}:
            return explicit
        if getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        return "chutes"

    @staticmethod
    def roster(family: str) -> tuple[str, ...]:
        def table_of() -> dict:
            return _FAMILY_MODELS.get(_Wire.provider()) or _FAMILY_MODELS["chutes"]

        def pin(names: tuple[str, ...]) -> tuple[str, ...]:
            pinned = (getenv("SANDBOX_MODEL") or "").strip()
            if not pinned:
                return names
            return (pinned,) + tuple(name for name in names if name != pinned)

        table = table_of()
        return pin(tuple(table.get(family) or table["generic"]))

    @staticmethod
    def commit(text: str, title: str = "") -> str:
        if not _ASK_PERMIT_RE.search(text or ""):
            return text
        who = (title or "the nearest same-model option").strip()
        return (
            "The exact item is unavailable. I am switching to "
            + who
            + " and completing the order now."
        )

    @staticmethod
    def args(tool_call: dict) -> dict:
        raw = (tool_call.get("function") or {}).get("arguments", "{}")
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError):
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def trim(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 40] + ' ... (truncated by agent)"}'


_Axle.Link = _Wire


_num = _Wire.num
_provider = _Wire.provider
_family_roster = _Wire.roster
_rewrite_commit = _Wire.commit
_arguments = _Wire.args
_trim = _Wire.trim


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

        class _Ingest:
            __slots__ = ("host", "obs", "brief")

            def __init__(self, host: "Ledger", payload: dict, parsed: Brief) -> None:
                self.host = host
                self.obs = payload
                self.brief = parsed

            def rows(self) -> None:
                for row in self.obs.get("results") or []:
                    if isinstance(row, dict):
                        self.host._note_row(row, self.brief)

            def view(self) -> None:
                self.host._note_product(self.obs, self.brief)

            def compare(self) -> None:
                for prod in self.obs.get("compared") or []:
                    if isinstance(prod, dict):
                        self.host._note_product(prod, self.brief)

            def stock(self) -> None:
                k = self.host.key(self.obs.get("product_id"), self.obs.get("sku"))
                self.host.checked[k] = {
                    "in_stock": bool(self.obs.get("in_stock")),
                    "price": _num(self.obs.get("price")),
                }
                self.host.checked_seq[k] = self.host.seq
                self.host.inspected.add(k)

            def add(self) -> None:
                added = self.obs.get("added") or {}
                pid, sku = added.get("product_id"), added.get("sku")
                if pid and sku:
                    self.host.adds += 1
                    if (pid, sku) not in self.host.cart:
                        self.host.cart.append((pid, sku))
                    if self.host.first_add_seq is None:
                        self.host.first_add_seq = self.host.seq
                    self.host.last_add_seq = self.host.seq
                self.host._sync_cart(self.obs)

            def remove(self) -> None:
                self.host._sync_cart(self.obs)
                removed = self.obs.get("removed") or {}
                pair = (removed.get("product_id"), removed.get("sku"))
                if pair in self.host.cart:
                    self.host.cart.remove(pair)

            def cart(self) -> None:
                self.host._sync_cart(self.obs)

            def dispatch(self, tool: str) -> None:
                alias = {
                    "search": "rows",
                    "filter": "rows",
                    "view": "view",
                    "compare": "compare",
                    "inspect_stock": "stock",
                    "add_to_cart": "add",
                    "remove_from_cart": "remove",
                    "inspect_cart": "cart",
                }.get(tool)
                if not alias:
                    return
                getattr(self, alias)()

        _Ingest(self, obs, brief).dispatch(name)

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



class Agent:
    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = Brief(str(self.policy.get("query") or ""))
        self.budget = self.brief.budget
        self.ledger = Ledger()
        self.clock = _make_clock(_PROBLEM_BUDGET_S)
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
        self.roster = _family_roster(self.brief.family)

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        play = _PLAYBOOK.get(self.brief.family, _PLAYBOOK["generic"])
        extra = ""
        if self.brief.family == "recall" and self.brief.required_identities:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the controller reports the count after each search."
                % self.brief.required_identities
            )
        return _CORE_RULES + "\n\nFamily playbook: " + play + extra

    def _opening_question(self) -> str | None:
        class _Ask:
            @staticmethod
            def constraint() -> str:
                return (
                    "Before I search: which hardware specification is your firm requirement, "
                    "and what value do you need?"
                )

            @staticmethod
            def preference() -> str:
                return (
                    "Quick question so I choose well: which of your preferences counts most, and in what "
                    "order should I weigh the others? Let me know your priority."
                )

            def pick(self, host: "Agent") -> str | None:
                if host.brief.family == "constraint" or host.brief.spec_sealed:
                    return self.constraint()
                if host.brief.family == "preference":
                    return self.preference()
                return None

        return _Ask().pick(self)

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

    def _fact_blob(self, pid: str, field: str | None = None) -> str:
        """All labelled facts, preferring keys that share a word with the concept."""
        facts = self._facts(pid)
        if not facts:
            return ""
        want = _tokens(field or "")
        picked = []
        for key, raw in facts.items():
            if want and (_tokens(str(key)) & want or str(key).lower().split("_")[-1] in want):
                picked.append(str(raw))
        if not picked:
            picked = [str(v) for v in facts.values()]
        return " ".join(picked)

    def _literal_spec(self, text: str, want: str | None) -> bool:
        if not want:
            return False
        low = (text or "").lower()
        if want.lower() in low:
            return True
        return bool(_tokens(want)) and _tokens(want) <= _tokens(low)

    def _fact_value(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept."""
        blob = self._fact_blob(pid, field)
        if not blob:
            return None
        if field == "resolution":
            want = self.brief.spec_value if self.brief.spec_field == field else self.brief.specs.get(field)
            return want if self._literal_spec(blob, want) else None
        found = _adjacent_vals(field, blob)
        unit = _Lens.unit_of(field)
        if not found and unit:
            tail = field.split("_")[-1]
            for key, raw in self._facts(pid).items():
                text = str(raw).strip()
                if re.fullmatch(r"\d+(?:\.\d+)?", text) and (_tokens(str(key)) & (_tokens(field) | {tail})):
                    found.add("%g%s" % (float(text), unit))
        return _one(found)

    def _product_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: facts plus the title, and nothing when they disagree."""
        if field == "resolution":
            want = self.brief.spec_value if self.brief.spec_field == field else self.brief.specs.get(field)
            blob = self._fact_blob(pid, field) + " " + self._title(pid)
            return want if self._literal_spec(blob, want) else None
        vals = _adjacent_vals(field, self._fact_blob(pid, field))
        vals |= _adjacent_vals(field, self._title(pid))
        return _one(vals)

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        if field == "resolution":
            want = self.brief.spec_value if self.brief.spec_field == field else self.brief.specs.get(field)
            blob = self._options(pid, sku) + " " + self._fact_blob(pid, field) + " " + self._title(pid)
            return want if self._literal_spec(blob, want) else None
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
        if not placed:
            placed = any(
                any(min(len(w), len(lab)) >= 4 and (w in lab or lab in w) for w in wanted)
                for lab in labels
            )
        if not placed:
            return False
        qualifier = self.brief.cat_qualifier
        if not qualifier:
            return True
        title = self._title(pid)
        words = self._stems(title) | _tokens(title)
        q = qualifier.replace("-", " ")
        qtoks = set(q.split())
        compact = q.replace(" ", "")
        blob = re.sub(r"[^a-z0-9]+", "", title.lower())
        return qtoks <= words or compact in blob

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _accessory_listing(self, pid: str) -> bool:
        """A listing that is an add-on for the requested shelf, not the item itself."""
        title = self._title(pid)
        cat = (self.brief.category or "").strip()
        if not cat:
            return False
        stem = re.escape(cat.rstrip("s"))
        return bool(
            re.search(
                rf"{stem}s?\s+(?:case|cover|sleeve|pouch|protector|replacement)s?\b|"
                rf"\b(?:case|cover|sleeve|pouch|protector)s?\s+(?:for\s+)?(?:\w+\s+){{0,3}}{stem}",
                title,
                re.I,
            )
        )

    def _device_ok(self, pid: str, sku: str) -> bool:
        """Reject resale pages and add-ons of the named category; otherwise the listing may count."""
        if self._resale_title(pid) or self._accessory_listing(pid):
            return False
        return True

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        return bool(_RESALE_RE.search(self._title(pid)))

    def _relax(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        model, color = self.brief.pref_model, self.brief.pref_color
        if not model:
            return 0

        def squash(t: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

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

        class _Extra:
            __slots__ = ("host", "pid", "sku", "field", "want")

            def __init__(self, host: "Agent", product: str, variant: str, spec_field, spec_want) -> None:
                self.host = host
                self.pid = product
                self.sku = variant
                self.field = spec_field
                self.want = spec_want

            def justification(self, flag: list) -> None:
                flag[0] = flag[0] and not self.host._resale_title(self.pid) and bool(self.host._brand_of(self.pid))
                if self.field and self.want:
                    # The reason has to quote the product's own facts, so the facts must state it.
                    flag[0] = (
                        flag[0]
                        and self.host._fact_value(self.pid, self.field) == self.want
                        and self.host._product_value(self.pid, self.field) == self.want
                    )

            def preference(self, flag: list) -> None:
                if self.host._resale_title(self.pid) or self.host._accessory_listing(self.pid):
                    flag[0] = False
                if not self.host.ledger.event_target:
                    return
                # A replacement has to come from a different listing, not a twin of the dead one.
                dead_listing = self.host.ledger.listing_of.get(
                    self.host.ledger.event_target.split("::", 1)[0]
                )
                if dead_listing and self.host.ledger.listing_of.get(str(self.pid)) == dead_listing:
                    flag[0] = False

            def recall(self, flag: list) -> None:
                if self.host.brief.theme:
                    flag[0] = flag[0] and all(
                        t in _tokens(self.host._title(self.pid)) for t in self.host.brief.theme
                    )

            def ranking(self, flag: list) -> None:
                flag[0] = flag[0] and (str(self.pid), str(self.sku)) in self.host.brief.candidate_keys

        box = [ok]
        getattr(_Extra(self, pid, sku, field, want), fam, _noop)(box)
        ok = box[0]
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._relax(pid, sku) if fam == "recovery" else 0,
        }

    def _rank_axes(self) -> list:
        """Sort keys named in the live request, not a fixed product shape."""
        blob = (self.brief.goal + " " + self.brief.rules).lower()
        axes = []
        if "refresh" in blob:
            axes.append("refresh_rate")
        if "screen" in blob or re.search(r"\binch", blob):
            axes.append("screen_size")
        for field in self.brief.specs:
            if field not in axes:
                axes.append(field)
        return axes

    def _rank_of(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        fam = self.brief.family
        price = c["price"] if c["price"] is not None else 1e12

        class _Order:
            def ranking(self) -> tuple:
                keys = []
                for axis in host._rank_axes():
                    keys.append(-(host._measure(c["pid"], c["sku"], axis) or 0.0))
                keys.extend([price, c["key"]])
                return tuple(keys)

            def recovery(self) -> tuple:
                return (c["relax"], price, c["key"])

            def fallback(self) -> tuple:
                extra = sum(
                    1 for f, v in host.brief.specs.items()
                    if f != host.brief.spec_field and host._variant_value(c["pid"], c["sku"], f) != v
                )
                if fam == "intent" and first and host.brief.pref_brand:
                    wanted = _tokens(host.brief.pref_brand)
                    near = 0 if (
                        wanted <= _tokens(host._brand_of(c["pid"]))
                        or wanted <= _tokens(host._title(c["pid"]))
                    ) else 1
                    return (near, extra, price, c["key"])
                return (extra, price, c["key"])

        host = self
        desk = _Order()
        return getattr(desk, fam, desk.fallback)()

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
        class _Pick:
            __slots__ = ("host", "first", "fam", "good")

            def __init__(self, host: "Agent") -> None:
                self.host = host
                self.first = host.ledger.adds == 0
                self.fam = host.brief.family
                self.good = [c for c in host._candidates() if c["ok"]]

            def pref_pick(self) -> dict | None:
                carted = [c for c in self.good if (c["pid"], c["sku"]) in self.host.ledger.cart]
                if carted:
                    return carted[0]
                two = len(self.host.ledger.listing_ids) >= 2 or len(self.host.ledger.products) >= 2
                if not (self.good and (two or self.host.reads_done or self.host.forced_compares >= 1)):
                    return None

                def pref_key(c: dict) -> tuple:
                    score = self.host._pref_score(c["pid"], c["sku"])
                    price = c["price"] if c["price"] is not None else 0.0
                    return (-score, -price, c["key"])

                scored = [c for c in self.good if self.host._pref_score(c["pid"], c["sku"]) >= 2]
                if not scored:
                    return None
                scored.sort(key=pref_key)
                return scored[0]

            def intent_narrow(self) -> bool:
                if not (self.fam == "intent" and self.first and self.host.brief.pref_brand):
                    return True
                wanted = _tokens(self.host.brief.pref_brand)
                named = [
                    c for c in self.good
                    if wanted <= _tokens(self.host._brand_of(c["pid"]))
                    or wanted <= _tokens(self.host._title(c["pid"]))
                ]
                if named:
                    self.good = named
                    return True
                if not self.host._intent_brand_done():
                    self.host._preferred_brand_hint()
                    return False
                return True

            def recovery_narrow(self) -> bool:
                if not (self.fam == "recovery" and self.first):
                    return True
                exact = [c for c in self.good if c["relax"] == 0]
                if exact:
                    self.good = exact
                    return True
                self.host._exact_match_hint()
                return False

            def choose(self) -> dict | None:
                if self.fam == "preference":
                    return self.pref_pick()
                if not self.intent_narrow() or not self.recovery_narrow():
                    return None
                if not self.good and self.fam == "ranking":
                    self.good = [
                        c for c in self.host._candidates()
                        if c["viewed"] and c["live"] and (c["pid"], c["sku"]) in self.host.brief.candidate_keys
                    ]
                if not self.good:
                    return None

                def rank_key(c: dict) -> tuple:
                    return self.host._rank_of(c, self.first)

                self.good.sort(key=rank_key)
                return self.good[0]

        return _Pick(self).choose()

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
        if fam == "ranking":
            return None
        if fam == "recall" and len(self.ledger.identities) < self._recall_need():
            return None
        if fam == "recovery" and self.brief.pref_model and self.search_calls < _WEAK_SEARCH_CAP:
            keyed = ("model", self.brief.pref_model)
            if keyed not in self.probes and "search" in self.tool_names:
                self.probes.add(keyed)
                return ("search", {"query": self.brief.pref_model, "k": 10, "in_stock": True})
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        if fam == "intent" and self.brief.spec_value:
            spec_leads = [pid for pid in leads if self._row_mentions_spec(pid)]
            if spec_leads:
                leads = spec_leads
        if fam == "preference":
            hinted = [pid for pid in leads if self._row_pref_hint(pid)]
            if hinted:
                leads = hinted
        if leads and self.forced_compares < 8:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            return ("view", {"product_id": leads[0]})
        if fam == "recovery" and self.brief.pref_model and self.ledger.adds == 0:
            self.reads_done = True
            return None
        if fam == "preference":
            return self._pref_boot()
        if fam == "intent":
            boot = self._intent_boot()
            if boot is not None:
                return boot
        # No unread rows left: ask the catalog again under the requirements as they stand now.
        brand = self.brief.brand or (self.brief.pref_brand if fam == "intent" else None)
        probe = (brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.search_calls >= _WEAK_SEARCH_CAP:
            self.reads_done = True
            return None
        if probe in self.probes:
            self.reads_done = True  # the catalog has been asked this already
            return None
        self.probes.add(probe)
        if "filter" in self.tool_names and (self.brief.category or brand):
            args = {"in_stock": True, "k": 10}
            if self.brief.category:
                args["category"] = self.brief.category
            if self.budget is not None:
                args["max_price"] = self.budget
            if brand:
                args["brand"] = brand
            return ("filter", args)
        if "search" in self.tool_names:
            words = [w for w in (brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return ("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, its own twins are the nearest replacements.

        A listing's other colors and sizes are separate products carrying the same
        title, so searching that title surfaces exactly the items that still meet
        every requirement the dead one met.
        """
        dead = self.ledger.event_target
        if self.brief.family == "intent":
            return None
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

    def _intent_brand_done(self) -> bool:
        """True once the preferred brand has been asked for the gated spec."""
        if not self.brief.pref_brand:
            return True
        if not self._brand_probed():
            return False
        if self._branded_spec_viewed():
            return True
        if any(pid for pid in self._unseen_leads() if self._row_is_pref_brand(pid) and self._row_mentions_spec(pid)):
            return False
        if self._spec_brand_probed():
            return True
        return not any(pid for pid in self._unseen_leads() if self._row_is_pref_brand(pid))

    def _brand_probed(self) -> bool:
        brand = self.brief.pref_brand
        if not brand:
            return True
        return any(isinstance(p, tuple) and p and p[0] in {brand, "spec-brand"} for p in self.probes)

    def _spec_brand_probed(self) -> bool:
        return any(isinstance(p, tuple) and p and p[0] == "spec-brand" for p in self.probes)

    def _row_is_pref_brand(self, pid: str) -> bool:
        if not self.brief.pref_brand:
            return False
        wanted = _tokens(self.brief.pref_brand)
        row = next((r for k, r in self.ledger.rows.items() if k.startswith(str(pid) + "::")), {})
        blob = _tokens(str((row or {}).get("brand") or "") + " " + str((row or {}).get("title") or self._title(pid)))
        return bool(wanted) and wanted <= blob

    def _row_mentions_spec(self, pid: str) -> bool:
        want = self.brief.spec_value
        if not want:
            return False
        row = next((r for k, r in self.ledger.rows.items() if k.startswith(str(pid) + "::")), {})
        title = str((row or {}).get("title") or self._title(pid) or "")
        num = re.sub(r"[^0-9.]+", "", str(want))
        if not num:
            return str(want).lower() in title.lower()
        return bool(re.search(r"(?<![\d.])" + re.escape(num) + r"(?![\d])", title))

    def _branded_spec_viewed(self) -> bool:
        if not self.brief.pref_brand:
            return False
        wanted = _tokens(self.brief.pref_brand)
        for c in self._candidates():
            if c["viewed"] and c["spec"] == 2 and (
                wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))
            ):
                return True
        return False

    def _spec_query(self) -> str:
        val = str(self.brief.spec_value or "")
        m = re.fullmatch(r"([\d.]+)in", val)
        if m:
            return m.group(1) + " inch"
        m = re.fullmatch(r"([\d.]+)hz", val)
        if m:
            return m.group(1) + " hz"
        m = re.fullmatch(r"([\d.]+)gb", val)
        if m:
            return m.group(1) + " gb"
        return val

    def _intent_boot(self) -> tuple | None:
        """TF1: retrieve the preferred brand with tools, never the 90s model."""
        if self.brief.family != "intent":
            return None
        leads = self._unseen_leads()

        def prefer_branded() -> list:
            if not self.brief.pref_brand:
                return leads
            branded = [pid for pid in leads if self._row_is_pref_brand(pid)]
            if branded:
                return branded + [p for p in leads if p not in branded]
            return leads

        def view_leads(rows: list) -> tuple | None:
            if not rows or self.forced_compares >= 8:
                return None
            self.forced_compares += 1
            if len(rows) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": rows[:5]})
            if "view" in self.tool_names:
                return ("view", {"product_id": rows[0]})
            return None

        def search_bits(bits: list) -> tuple | None:
            if "search" not in self.tool_names or self.search_calls >= _CTRL_SEARCH_CAP:
                return None
            query = " ".join(str(b) for b in bits if b)
            if not query:
                return None
            return ("search", {"query": query, "in_stock": True, "k": 10})

        def probe_brand() -> tuple | None:
            brand = self.brief.pref_brand or self.brief.brand
            if not (brand and self.search_calls < _CTRL_SEARCH_CAP):
                return None
            probe = (brand, "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
            if probe in self.probes:
                return None
            self.probes.add(probe)
            if "filter" in self.tool_names:
                args = {"brand": brand, "in_stock": True, "k": 10}
                if self.brief.category:
                    args["category"] = self.brief.category
                if self.budget is not None:
                    args["max_price"] = self.budget
                return ("filter", args)
            return search_bits([self._spec_query() or self.brief.spec_value, self.brief.category, brand])

        def probe_spec_brand() -> tuple | None:
            brand = self.brief.pref_brand or self.brief.brand
            if not (brand and self.brief.spec_value and self.search_calls < _CTRL_SEARCH_CAP):
                return None
            probe = ("spec-brand", brand, str(self.brief.spec_value), "%.2f" % (self.budget or 0.0))
            if probe in self.probes:
                return None
            self.probes.add(probe)
            return search_bits([self._spec_query(), self.brief.category, brand])

        def probe_wide() -> tuple | None:
            wide = ("intent-wide", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
            if wide in self.probes or self.search_calls >= _CTRL_SEARCH_CAP:
                return None
            self.probes.add(wide)
            hit = search_bits([self._spec_query() or self.brief.spec_value, self.brief.category])
            if hit is not None:
                return hit
            if "filter" in self.tool_names and self.brief.category:
                args = {"category": self.brief.category, "in_stock": True, "k": 10}
                if self.budget is not None:
                    args["max_price"] = self.budget
                return ("filter", args)
            return None

        def probe_event() -> tuple | None:
            if not self.ledger.event_target or self.search_calls >= _CTRL_SEARCH_CAP:
                return None
            post = ("post-event", str(self.brief.spec_value), "%.2f" % (self.budget or 0.0))
            if post in self.probes:
                return None
            self.probes.add(post)
            return search_bits([self._spec_query() or self.brief.spec_value, self.brief.category])

        branded = prefer_branded()
        spec_first = [pid for pid in branded if self._row_mentions_spec(pid)] or [
            pid for pid in leads if self._row_mentions_spec(pid)
        ]

        def view_spec() -> tuple | None:
            return view_leads(spec_first)

        def view_branded() -> tuple | None:
            if spec_first:
                return None
            if self._spec_brand_probed() or not self.brief.pref_brand:
                return view_leads(branded if branded != leads else [])
            return None

        for stage in (probe_event, view_spec, probe_brand, probe_spec_brand, view_branded, probe_wide):
            hit = stage()
            if hit is not None:
                return hit
        return None

    def _row_pref_hint(self, pid: str) -> bool:
        row = next((r for k, r in self.ledger.rows.items() if k.startswith(str(pid) + "::")), {})
        title = str((row or {}).get("title") or self._title(pid) or "")
        return self._blob_pref_hits(title) > 0

    def _pref_queries(self) -> list:
        """Search strings built from the shopper's use-case words and the named shelf."""
        cat = self.brief.category or ""
        ordered = self._wish_order()
        queries = []
        if ordered:
            queries.append(" ".join(ordered[:4] + ([cat] if cat else [])))
            if len(ordered) >= 2:
                queries.append(" ".join(ordered[:2] + ([cat] if cat else [])))
            for a, b in zip(ordered, ordered[1:]):
                phrase = a + " " + b
                queries.append((phrase + " " + cat).strip())
            if cat:
                queries.append(ordered[0] + " " + cat)
        if cat:
            queries.append(cat)
        seen = []
        for q in queries:
            q = " ".join(q.split())
            if q and q not in seen:
                seen.append(q)
        return seen

    def _pref_boot(self) -> tuple | None:
        """TF4: search the shopper's own priority words plus the named category."""
        if self.brief.family != "preference":
            return None
        leads = self._unseen_leads()
        hinted = [pid for pid in leads if self._row_pref_hint(pid)]
        if hinted and self.forced_compares < 8:
            self.forced_compares += 1
            if len(hinted) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": hinted[:5]})
            if "view" in self.tool_names:
                return ("view", {"product_id": hinted[0]})
        if self.search_calls >= _CTRL_SEARCH_CAP:
            if leads and self.forced_compares < 8:
                self.forced_compares += 1
                if len(leads) > 1 and "compare" in self.tool_names:
                    return ("compare", {"product_ids": leads[:5]})
                if "view" in self.tool_names:
                    return ("view", {"product_id": leads[0]})
            return None
        for query in self._pref_queries():
            key = ("pref", query)
            if key in self.probes:
                continue
            self.probes.add(key)
            if "search" in self.tool_names:
                return ("search", {"query": query, "k": 10, "in_stock": True})
        cat = self.brief.category
        filt = ("pref-filter", cat or "", "%.2f" % (self.budget or 0.0))
        if filt not in self.probes and "filter" in self.tool_names and cat:
            self.probes.add(filt)
            args = {"category": cat, "in_stock": True, "k": 10}
            if self.budget is not None:
                args["max_price"] = self.budget
            return ("filter", args)
        if leads and self.forced_compares < 8:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            if "view" in self.tool_names:
                return ("view", {"product_id": leads[0]})
        return None

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
                model = re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower())
                if model and model not in squashed and model not in str(pid):
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
        if (
            not out
            and self.brief.family == "recovery"
            and self.brief.pref_model
            and self.ledger.adds == 0
        ):
            for key, row in self.ledger.rows.items():
                pid, _, sku = key.partition("::")
                if not sku or str(pid) in self.ledger.products or str(pid) in self.unreadable:
                    continue
                if pid not in out:
                    out.append(str(pid))
        return out

    # ------------------------------------------------------------------ guidance
    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        cands = self._candidates()
        if self.brief.family == "preference" and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            def pref_note_key(c: dict) -> tuple:
                return (-self._pref_score(c["pid"], c["sku"]), -(c["price"] or 0.0))
            good.sort(key=pref_note_key)
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
        blob = " ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku)
        return self._blob_pref_hits(blob)

    def _pref_score(self, pid: str, sku: str) -> int:
        """Soft-preference score from the shopper's own use-case words against this listing."""
        return self._priority_hits(pid, sku)

    def _shopper_priority_text(self) -> str:
        """The shopper's use-case, without task-rules or the 'place a test order' frame."""
        chunks = []
        goal = self.brief.goal or ""
        for part in re.split(r"(?<=[.!?])\s+", goal):
            part = part.strip()
            if not part:
                continue
            if re.match(r"(?:find and place|find and|place one test order|i need|i want)\b", part, re.I):
                tail = re.split(r"\b(?:php|usd|eur|[$€£¥₱])\b", part, flags=re.I)
                extra = tail[-1] if len(tail) > 1 else ""
                if extra and re.search(r"[a-z]{4,}", extra, re.I):
                    chunks.append(extra)
                continue
            chunks.append(part)
        query = str(self.policy.get("query") or "")
        for msg in self.messages:
            if msg.get("role") != "user":
                continue
            text = str(msg.get("content") or "").strip()
            if not text or text.startswith("[note]") or text == query:
                continue
            if _CAT_SHOPPER_RE.search(text) and len(text) < 90:
                continue
            chunks.append(text)
        return " ".join(chunks) if chunks else goal

    def _emphasis_text(self) -> str:
        """The clause the shopper marked as what they will pay for / care about."""
        text = self._shopper_priority_text()
        spans = []
        for pat in (
            r"rather(?:\s+\w+){0,8}\s+for\s+(.+?)(?:than|\.|$)",
            r"([^.|;]{8,90}?)\s+is what i actually care about",
            r"care about[:\s]+(.+?)(?:\.|$)",
            r"([^.|;]{8,90}?)\s+matters more",
        ):
            for m in re.finditer(pat, text, re.I | re.S):
                span = (m.group(1) or "").strip()
                if span:
                    spans.append(span)
        return " ".join(spans)

    def _wish_order(self) -> list:
        """Use-case tokens, with the shopper's 'care about' clause first."""
        drop = _WISH_DROP | _tokens(self.brief.category or "") | _tokens(self.brief.currency or "")

        def take(text: str, into: list) -> None:
            for word in re.findall(r"[a-z]{4,}", (text or "").lower()):
                if word in drop or word.isdigit() or word in into:
                    continue
                into.append(word)

        ordered = []
        take(self._emphasis_text(), ordered)
        take(self._shopper_priority_text(), ordered)
        return ordered

    def _wish_words(self) -> set:
        """Content words from the shopper's use-case, gathered once."""
        if self._wishes is None:
            self._wishes = set(self._wish_order())
        return self._wishes

    def _blob_pref_hits(self, blob: str) -> int:
        """How many of the shopper's use-case tokens this text supports, including stems."""
        wants = self._wish_words()
        if not wants:
            return 0
        toks = _tokens(blob)
        compact = re.sub(r"[^a-z0-9]+", "", (blob or "").lower())
        score = 0
        for word in wants:
            if word in toks or word in compact:
                score += 2
                continue
            stem = word.rstrip("s")
            if len(stem) >= 5 and (stem in toks or stem in compact):
                score += 2
                continue
            if len(word) >= 4 and any(t.startswith(word) or (len(t) >= 4 and word.startswith(t)) for t in toks):
                score += 1
        return score

    def _budget_from_message(self, text: str) -> None:
        new = Brief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        self.reads_done = False
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

    def _claimable(self) -> set:
        names = set(_CLAIM_CORE)
        if self.brief.spec_field:
            names.add(self.brief.spec_field)
        names.update(self.brief.specs)
        return names - _CLAIM_DENY

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by the last view + latest inspect_stock of this exact item."""
        allow = self._claimable()
        out = []
        field, want = self.brief.spec_field, self.brief.spec_value
        if field and want and field in allow and field not in _CLAIM_DENY:
            if self._fact_value(pid, field) == want and self._product_value(pid, field) == want:
                out.append({"field": field, "value": want})
        for name, value in self.brief.specs.items():
            if name == field or name not in allow or name in _CLAIM_DENY:
                continue
            if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        return self._sanitize_claims(out, pid, sku)

    def _sanitize_claims(self, claims: list, pid: str, sku: str) -> list:
        cleaned = []
        seen = set()
        allow = self._claimable()
        st = self.ledger.status(pid, sku) or {}
        live_price = st.get("price")
        live_stock = bool(st.get("in_stock"))
        real_brand = str(self._brand_of(pid) or "").strip()
        for row in claims:
            if not isinstance(row, dict):
                continue
            name = str(row.get("field") or "").strip()
            value = str(row.get("value") or "").strip()
            if not name or name in seen or name in _CLAIM_DENY or name not in allow:
                continue
            if name == "brand":
                if not real_brand or real_brand.casefold() in {"none", "null", "n/a"}:
                    continue
                value = real_brand
            elif name == "price":
                if live_price is None:
                    continue
                value = "%.2f" % live_price if isinstance(live_price, (int, float)) else str(live_price)
            elif name == "in_stock":
                if not live_stock:
                    continue
                value = "true"
            elif self.brief.spec_field and name == self.brief.spec_field:
                got = self._fact_value(pid, name)
                if not got or got != value:
                    continue
            seen.add(name)
            cleaned.append({"field": name, "value": value})
        if live_price is not None and "price" not in seen:
            cleaned.append(
                {"field": "price", "value": "%.2f" % live_price if isinstance(live_price, (int, float)) else str(live_price)}
            )
        if live_stock and "in_stock" not in seen:
            cleaned.append({"field": "in_stock", "value": "true"})
        if real_brand and real_brand.casefold() not in {"none", "null", "n/a"} and "brand" not in seen:
            cleaned.append({"field": "brand", "value": real_brand})
        return [c for c in cleaned if c["field"] in allow]

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.brief.family == "justification":
            claims = self._sanitize_claims(self._claims_for(pid, sku), pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _closed_ok(self, pid: str, sku: str) -> bool:
        closed = self.brief.candidate_keys
        if not closed:
            return True
        return (str(pid), str(sku)) in closed

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
        if not self._closed_ok(pid, sku):
            return False, "product_id::sku is outside the closed candidate set"
        return True, ""

    # ------------------------------------------------------------------ planning
    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        early = self.search_calls < _WEAK_SEARCH_CAP and self.turn < self.max_steps * 0.45
        return self.brief.required_identities + (1 if early else 0)

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
        if fam == "constraint" and self.turn < 4 and not self.shopper_updated:
            return False  # a later message can still tighten the budget
        if fam in ("intent", "ranking", "recovery"):
            return True
        if fam == "preference":
            if len(self.ledger.listing_ids) >= 2 or len(self.ledger.products) >= 2:
                return True
            return bool(self.reads_done or self.forced_compares >= 2)
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if fam in ("recall", "retrieval") and roomy and self.search_calls < _WEAK_SEARCH_CAP:
            if len(self.ledger.identities) < self._recall_need() or self.search_calls < 2:
                return False
        if fam in _CTRL_FAMS:
            return True
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

        class _Stages:
            __slots__ = ("host", "fam")

            def __init__(self, host: "Agent") -> None:
                self.host = host
                self.fam = host.brief.family

            def shopper_q(self) -> tuple:
                host = self.host
                fam = self.fam
                if not (host.shopper_question_pending and "message" in host.tool_names):
                    return False, None
                if fam == "intent" and host._target() is None:
                    boot = host._intent_boot()
                    if boot is not None:
                        return True, boot
                if fam == "preference" and host._target() is None:
                    boot = host._pref_boot()
                    if boot is not None:
                        return True, boot
                if fam in ("intent", "preference") or host.question_waits >= 2:
                    return True, ("message", {"content": host._answer_text()})
                if fam in _CTRL_FAMS and (host._target() is not None or host.ledger.event_target):
                    return False, None
                if fam in _CTRL_FAMS:
                    return True, ("message", {"content": host._answer_text()})
                host.question_waits += 1
                return True, None

            def intent_boot(self) -> tuple | None:
                if self.fam == "intent" and self.host._target() is None:
                    return self.host._intent_boot()
                return None

            def pref_boot(self) -> tuple | None:
                if self.fam == "preference" and self.host._target() is None:
                    return self.host._pref_boot()
                return None

            def rank_compare(self) -> tuple | None:
                if self.fam == "ranking" and "compare" in self.host.tool_names:
                    missing = [
                        p for p, _s in self.host.brief.candidate_keys
                        if p not in self.host.ledger.products and p not in self.host.unreadable
                    ]
                    if missing:
                        return ("compare", {"product_ids": missing[:5]})
                return None

            def recovery_sib(self) -> tuple | None:
                host = self.host
                if self.fam == "recovery" and not host.ledger.adds and host.brief.pref_model and "search" in host.tool_names:
                    return host._sibling_search()
                return None

            def buy_or_read(self) -> tuple | None:
                host = self.host
                target = host._target()
                if target is None or not host._ready():
                    forced = host._read_more()
                    if forced is not None:
                        return forced
                    if target is None:
                        return None
                pid, sku = target["pid"], target["sku"]
                if (pid, sku) in host.ledger.cart:
                    if not host._fresh(pid, sku):
                        return ("inspect_stock", {"product_id": pid, "sku": sku})
                    return ("place_test_order", host._order_args(pid, sku))
                if not host._ready():
                    return None
                for cpid, csku in host.ledger.cart:
                    if Ledger.key(cpid, csku) in host.ledger.dead_keys and "remove_from_cart" in host.tool_names:
                        return ("remove_from_cart", {"product_id": cpid, "sku": csku})
                if not host._fresh(pid, sku):
                    return ("inspect_stock", {"product_id": pid, "sku": sku})
                return ("add_to_cart", {"product_id": pid, "sku": sku})

            def pick(self) -> tuple | None:
                stop, hit = self.shopper_q()
                if stop:
                    return hit
                for name in ("intent_boot", "pref_boot", "rank_compare", "recovery_sib"):
                    hit = getattr(self, name)()
                    if hit is not None:
                        return hit
                return self.buy_or_read()

        return _Stages(self).pick()

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        if self.clock.remaining() < 50:
            return None
        if self.brief.family in _CTRL_FAMS:
            return None
        for attempt in range(2):
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
            if self.clock.remaining() > 30:
                time.sleep(min(0.8 * (attempt + 1), 1.5))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    # ------------------------------------------------------------------ environment
    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; an identical envelope replays idempotently."""
        for attempt in range(2):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 12:
                break
            time.sleep(min(0.5 * (attempt + 1), 1.0))
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

        def ingest_user(payload: dict) -> None:
            text = str(payload["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            if self.brief.family in ("constraint", "intent") and not self.brief.spec_value:
                stated = self.brief.adopt_spec(text)
                if stated:
                    self.shopper_updated = True
                    self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            named_cat = self.brief.adopt_category(text)
            if named_cat:
                self.shopper_updated = True
                self._wishes = None
                self._note("Catalog category recorded: " + named_cat + ". Search that shelf for a real device that fits the use-case.")
            self._wishes = None
            if _QUESTION_RE.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")

        if isinstance(user, dict) and user.get("content"):
            ingest_user(user)
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
        self.reads_done = False
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
        class _Closer:
            __slots__ = ("host",)

            def __init__(self, host: "Agent") -> None:
                self.host = host

            def try_carted(self) -> bool:
                for pid, sku in reversed(self.host.ledger.cart):
                    ok, _why = self.host._valid_now(pid, sku)
                    if ok and not self.host.done:
                        self.host._act("place_test_order", self.host._order_args(pid, sku))
                        return True
                return False

            def shortlist(self) -> list:
                host = self.host
                good = [c for c in host._candidates() if c["ok"]]
                if not good and host.brief.family != "justification":
                    good = [
                        c for c in host._candidates()
                        if c["live"] and c["spec"] == 2 and host._brand_ok(c["pid"]) and c["cat_ok"]
                        and (not host.brief.candidate_keys or (c["pid"], c["sku"]) in host.brief.candidate_keys)
                    ]
                fam = host.brief.family
                if fam == "intent":
                    good = [c for c in good if c.get("spec") == 2 and c.get("cat_ok")]
                if fam == "preference":
                    good = [c for c in good if host._pref_score(c["pid"], c["sku"]) >= 2]
                if not good:
                    if fam in ("intent", "preference"):
                        return []
                    good = [
                        c for c in host._candidates()
                        if c["viewed"] and c["live"]
                        and (not host.brief.candidate_keys or (c["pid"], c["sku"]) in host.brief.candidate_keys)
                    ]
                if fam == "recovery" and host.ledger.adds == 0:
                    good = [c for c in good if c.get("relax", 0) == 0]
                return good

            def rush(self) -> None:
                host = self.host
                if self.try_carted():
                    return
                if host.done or host.turn > host.max_steps - 3:
                    return
                good = self.shortlist()
                if not good:
                    return
                fam = host.brief.family

                def finish_key(c: dict) -> tuple:
                    if fam == "preference":
                        return (-host._pref_score(c["pid"], c["sku"]), -(c["price"] or 0.0), c["key"])
                    return host._rank_of(c, host.ledger.adds == 0)

                good.sort(key=finish_key)
                pid, sku = good[0]["pid"], good[0]["sku"]
                if (pid, sku) not in host.ledger.cart:
                    obs = host._act("add_to_cart", {"product_id": pid, "sku": sku})
                    if host.done or not isinstance(obs, dict) or obs.get("error"):
                        return
                obs = host._act("inspect_stock", {"product_id": pid, "sku": sku})
                if host.done or not isinstance(obs, dict) or not obs.get("in_stock"):
                    return
                price = _num(obs.get("price"))
                if host.budget is not None and price is not None and price > host.budget + _EPS:
                    return
                host._act("place_test_order", host._order_args(pid, sku))

        _Closer(self).rush()

    # ------------------------------------------------------------------ main loop
    class _Episode:
        __slots__ = ("host", "question", "last_forced", "repeats")

        def __init__(self, host: "Agent") -> None:
            self.host = host
            self.question = host._opening_question()
            self.last_forced = None
            self.repeats = 0

        def time_gate(self) -> str:
            host = self.host
            if host.clock.remaining() < 30 or host.turn >= host.max_steps - 3:
                host._finish_fast()
                return "break"
            return ""

        def ask_opening(self) -> str:
            host = self.host
            if self.question and not host.asked_shopper and "message" in host.tool_names:
                host.asked_shopper = True
                host._act("message", {"content": self.question})
                return "continue"
            return ""

        def do_recheck(self) -> str:
            host = self.host
            if host.pending_recheck and "inspect_stock" in host.tool_names:
                host._post_add_recheck()
                return "continue"
            return ""

        def do_forced(self) -> str:
            host = self.host
            plan = host._plan()
            if plan is None:
                return ""
            name, args = plan
            signature = name + json.dumps(args, sort_keys=True)
            self.repeats = self.repeats + 1 if signature == self.last_forced else 1
            self.last_forced = signature
            if self.repeats > 2:
                # The step is not moving the state on; abandon that candidate.
                if name in ("inspect_stock", "add_to_cart"):
                    host.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                if name == "compare":
                    host.unreadable.update(str(p) for p in args.get("product_ids") or [])
                if name == "view":
                    host.unreadable.add(str(args.get("product_id") or ""))
                if name == "message":
                    host.shopper_question_pending = False
                self.last_forced, self.repeats = None, 0
                host._note("That step changed nothing, so the controller moved on to other candidates.")
                return "continue"
            obs = host._act(name, args)
            if name in ("search", "filter"):
                host.search_calls += 1
            if name == "message":
                host.shopper_question_pending = False
                host.question_waits = 0
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    host.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            return "continue"

        def do_ctrl(self) -> str:
            host = self.host
            if host.brief.family not in _CTRL_FAMS:
                return ""
            boot = host._intent_boot() or host._pref_boot()
            if boot is not None:
                host._act(boot[0], boot[1])
                if boot[0] in ("search", "filter"):
                    host.search_calls += 1
                return "continue"
            before = host.turn
            host._finish_fast()
            if host.done or host.turn == before:
                return "break"
            return "continue"

        def apply_model(self) -> str:
            host = self.host
            proposal = host._ask_model()
            if proposal is None:
                host._finish_fast()
                return "break"
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            verdict = None
            while True:
                verdict = host._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= _MAX_REPROMPTS:
                    host.rejected_streak += 1
                    host._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if host.rejected_streak >= 3:
                        host._shortlist_note()
                        host.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                host.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                host.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                host.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = host._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip":
                return "continue"
            host.rejected_streak = 0
            if name not in host.tool_names:
                return "continue"
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and host.ledger.cart:
                    pid, sku = host.ledger.cart[-1]
                args = host._order_args(pid, sku)
            if name == "message":
                if host.brief.family == "recovery":
                    args["content"] = _rewrite_commit(
                        str(args.get("content") or ""),
                        host.brief.pref_model or "",
                    )
                if not host.shopper_question_pending:
                    host.questions_asked += 1
                host.shopper_question_pending = False
                host.question_waits = 0
            obs = host._act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    host.pending_recheck = (str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                host.search_calls += 1
                if host.brief.family == "recall" and host.brief.required_identities:
                    have, need = len(host.ledger.identities), host._recall_need()
                    host._note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == "inspect_cart":
                host.cart_inspections += 1
            if name in ("view", "compare"):
                host._shortlist_note()
            return ""

        def drive(self) -> list:
            host = self.host
            while not host.done and host.turn < host.max_steps:
                verdict = ""
                for name in ("time_gate", "ask_opening", "do_recheck", "do_forced"):
                    verdict = getattr(self, name)()
                    if verdict:
                        break
                if verdict == "break":
                    break
                if verdict == "continue":
                    continue
                self.last_forced, self.repeats = None, 0
                verdict = self.do_ctrl()
                if verdict == "break":
                    break
                if verdict == "continue":
                    continue
                verdict = self.apply_model()
                if verdict == "break":
                    break
            return host.dialogue

    def run(self) -> list:
        return self._Episode(self).drive()

    def _validate(self, name: str, args: dict) -> str | None:
        class _Gate:
            __slots__ = ("host", "args")

            def __init__(self, host: "Agent", payload: dict) -> None:
                self.host = host
                self.args = payload

            def order(self) -> str | None:
                host = self.host
                args = self.args
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and host.ledger.cart:
                    pid, sku = host.ledger.cart[-1]
                if not host._closed_ok(pid, sku):
                    return "order refused: product_id::sku is outside the closed candidate set"
                ok, why = host._valid_now(pid, sku)
                if not ok:
                    return "order refused: " + why
                if host.brief.family == "recall" and len(host.ledger.identities) < host._recall_need():
                    return (
                        "order refused: only %d distinct matching products have been surfaced; search with different "
                        "wording until at least %d appear" % (len(host.ledger.identities), host._recall_need())
                    )
                if host.brief.family == "preference" and len(host.ledger.listing_ids) < 2:
                    return "order refused: compare at least two products with different listing_id values first"
                return None

            def add(self) -> str | None:
                host = self.host
                args = self.args
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid or not sku:
                    return "add_to_cart needs both product_id and sku"
                if not host._closed_ok(pid, sku):
                    return "add_to_cart refused: product_id::sku is outside the closed candidate set"
                c = host._assess(pid, sku)
                key = Ledger.key(pid, sku)
                if not c["live"]:
                    return "that variant is out of stock, over budget, or already invalidated"
                if not c["viewed"]:
                    return "view or compare that product first so its raw facts are on record"
                if key not in host.ledger.inspected:
                    return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
                if c["spec"] != 2 and host.brief.spec_field:
                    return (
                        "that variant does not resolve to %s = %s from its own option text or raw facts"
                        % (host.brief.spec_field, host.brief.spec_value)
                    )
                if not c["ok"]:
                    return "that product fails a firm requirement (category, brand, or the kind of device requested)"
                if host.brief.family == "preference" and len(host.ledger.listing_ids) < 2:
                    return "compare two products with different listing_id values before the first cart addition"
                if host.brief.family == "preference" and host._pref_score(pid, sku) < 2:
                    return "that listing does not mention the shopper's stated priorities; search those words first"
                if host.brief.family == "recovery" and not host.ledger.adds and host.brief.pref_model and host._relax(pid, sku):
                    return (
                        "the first cart item must be the exact model and color the shopper named; the other colors are "
                        "separate listings sharing the same title, so search that title and read them first"
                    )
                if not host._ready():
                    return (
                        "hold: read one more product that also meets every requirement first, so there is a "
                        "fallback when the market moves right after the cart addition"
                    )
                return None

            def msg(self) -> str | None:
                host = self.host
                args = self.args
                text = str(args.get("content") or "")
                if not text.strip():
                    return "message content is required"
                if host.brief.family == "recovery" and _ASK_PERMIT_RE.search(text):
                    args["content"] = _rewrite_commit(text, host.brief.pref_model or "")
                    text = args["content"]
                if not host.shopper_question_pending and "?" not in text:
                    return "do not send status updates; use message only to answer the shopper's question or to ask one"
                if not host.shopper_question_pending and host.questions_asked >= 2:
                    return (
                        "the shopper has already answered; decide from the evidence you have rather than asking again"
                    )
                return None

            def cart(self) -> str | None:
                if self.host.cart_inspections >= 1:
                    return "the cart contents are already known from earlier observations"
                return None

            def check(self, tool: str) -> str | None:
                host = self.host
                if tool not in host.tool_names:
                    return "unknown tool; choose one of " + str(sorted(n for n in host.tool_names if n))
                alias = {
                    "place_test_order": "order",
                    "add_to_cart": "add",
                    "message": "msg",
                    "inspect_cart": "cart",
                }.get(tool)
                hit = getattr(self, alias, _noop)() if alias else None
                if hit:
                    return hit
                key = tool + json.dumps(self.args, sort_keys=True)
                if key in host.ledger.calls_seen and tool in ("search", "filter", "view", "compare"):
                    return "identical call already made; change the arguments"
                host.ledger.calls_seen.add(key)
                return None

        return _Gate(self, args).check(name)


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
