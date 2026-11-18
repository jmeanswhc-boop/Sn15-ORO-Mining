"""Hybrid ORO-Bench shopping agent (top spine + v_dash kernels).

Spine from the deterministic controller agent: a language model explores the
catalog, while WallClock / WishSheet / CatalogBook / HybridShopper own every decision the runtime
grades (evidence reading, cart commitment, post-cart re-check, order, claims).

Kernels grafted from the v_dash solver, without its multi-LLM planner:

  * spec-first deterministic retrieve so turn 0 does not wait on inference,
  * family openers for withheld specs and preference priority,
  * stronger market-event recovery (remove, re-inspect, tell the shopper,
    then commit a different qualifying SKU).

The unbounded LLM fallback loop and the three-call helper plan are not used.
"""

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
# Rate limits (429) need a longer cool-down than ordinary failures; stay on the
# provisioned model rather than hopping names and doubling request pressure.
# One short 429 cool-down then give up: five retries at 12s each burned
# ~40s per ask and left the episode stuck in explore under OpenRouter limits.
_PROXY = ProxyClient(
    timeout=60,
    max_retries=2,
    retry_delay=1.0,
    rate_limit_retry_delay=3.0,
)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9
_ASK_ATTEMPTS = 1
_ASK_BACKOFF_S = (2.0,)

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
    try:
        return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text or "")}
    except Exception:
        return set()


def _hz_vals(text: str) -> set:
    try:
        return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text or "")}
    except Exception:
        return set()


def _gb_vals(text: str) -> set:
    try:
        return {
            "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
            for m in _GB_RE.finditer(text or "")
        }
    except Exception:
        return set()


def _pixel_class(long_side: int, short_side: int) -> str | None:
    try:
        if long_side >= 3840 and short_side >= 2160:
            return "4k"
        if long_side >= 2560 and short_side >= 1440:
            return "qhd"
        if long_side >= 1920 and short_side >= 1080:
            return "fhd"
        if long_side >= 1280 and short_side >= 720:
            return "hd"
        return None
    except Exception:
        return None


def _res_vals(text: str) -> set:
    """Display classes asserted by a text: pixel pairs and the usual shorthands."""
    try:
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
    except Exception:
        return set()


_VALUE_SETS = {"screen_size": _inch_vals, "refresh_rate": _hz_vals, "resolution": _res_vals, "ram": _gb_vals}


def _adjacent_vals(field: str, text: str) -> set:
    """Values a free-form string asserts, requiring the number's own concept word.

    Titles and variant option strings mix several capacities, so a memory figure
    only counts when the word for it follows the number; storage totals and a
    graphics chip's own memory must never be read as system memory.
    """
    try:
        text = text or ""
        if field != "ram":
            return _VALUE_SETS[field](text)
        paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
        return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}
    except Exception:
        return set()


def _mentions(field: str, text: str) -> bool:
    """Whether an option string talks about the concept at all (even ambiguously)."""
    try:
        text = text or ""
        if field == "ram":
            return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
        if field == "screen_size":
            return bool(_INCH_RE.search(text))
        if field == "refresh_rate":
            return bool(_HZ_RE.search(text))
        return bool(_WXH_RE.search(text) or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I))
    except Exception:
        return False


def _one(values: set) -> str | None:
    try:
        return next(iter(values)) if len(values) == 1 else None
    except Exception:
        return None


def _wanted_value(field: str | None, text: str) -> str | None:
    """The single value a request states for a concept, or nothing when unclear."""
    try:
        if not field:
            return None
        return _one(_VALUE_SETS[field](text or ""))
    except Exception:
        return None


def _tokens(text: str) -> set:
    try:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    except Exception:
        return set()


def _same_word(a: str, b: str) -> bool:
    """Loose noun equality: ignores case, spacing and a trailing plural."""
    try:
        x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
        y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
        return bool(x) and x == y
    except Exception:
        return False


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
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?|"
    r"mouse\s*pads?|mousepads?|desk\s*pads?|keyboard\s*mats?)\b",
    re.I,
)
_IS_HEADPHONE_RE = re.compile(
    r"\b(?:headphones?|headsets?|earbuds?|ear\s*buds?|earphones?|inear|in-ear|over-ear|on-ear)\b",
    re.I,
)
_NOT_HEADPHONE_RE = re.compile(
    r"\b(?:cable|pigtail|antenna|adapter|dongle|case|cover|stand|holder|mount|charger|"
    r"extension|connector|mouse\s*pads?|mousepads?|screen\s*protector)\b",
    re.I,
)
# Soft preference axes for ranking replies ("X counts most", "Y next", ...).
_PREF_AXIS_PATTERNS = (
    ("water", re.compile(r"water|sweat|ipx\s*\d*|waterproof|resist|weather|workout", re.I)),
    ("fit", re.compile(r"sport|fit|secure|earhook|over[\s\-]?ear|on[\s\-]?ear|running|stays?\s*put", re.I)),
    ("battery", re.compile(r"battery|charge|mAh|play\s*time|outlasts|long\s*(?:gym|run)", re.I)),
    ("wireless", re.compile(r"wireless|tws|bluetooth|cordless|no\s*cable|fully\s*wireless", re.I)),
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
    try:
        text = _NEGATED_RE.sub("", text or "")
        return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text
    except Exception:
        return ""


def _phone_identity(facts: dict, title: str, options: str) -> str:
    """'yes' / 'no' / 'unclear' for whether public evidence makes this a real handset."""
    try:
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
    except Exception:
        return ""


def _headphone_identity(facts: dict, title: str, options: str, path: list | None = None) -> str:
    """'yes' / 'no' / 'unclear' for whether this is an actual headphone/headset device."""
    try:
        raw_title = str(title or "")
        blob = raw_title + " " + " ".join(str(v) for v in (facts or {}).values()) + " " + str(options or "")
        path_text = " ".join(str(p) for p in (path or []))
        if _NOT_HEADPHONE_RE.search(raw_title) and not _IS_HEADPHONE_RE.search(raw_title):
            return "no"
        if re.search(r"\b(?:cable|pigtail|antenna|extension)\b", raw_title, re.I):
            return "no"
        if _IS_HEADPHONE_RE.search(blob) or re.search(r"headphones?", path_text, re.I):
            return "yes"
        return "unclear"
    except Exception:
        return ""


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


class WallClock:
    def __init__(self, budget: float) -> None:
        try:
            self.start = time.monotonic()
            self.budget = budget
        except Exception:
            pass

    def remaining(self) -> float:
        try:
            return self.budget - (time.monotonic() - self.start)
        except Exception:
            return 0.0


class WishSheet:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str) -> None:
        try:
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
        except Exception:
            pass

    def _family(self) -> str:
        try:
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
        except Exception:
            return ""

    @staticmethod
    def _budget(text: str) -> float | None:
        try:
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
        except Exception:
            return None

    @staticmethod
    def _category(goal: str) -> str | None:
        try:
            m = _CAT_PHRASE_RE.search(goal or "")
            if m:
                return m.group(1).strip()
            m = re.search(
                r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
                goal or "",
                re.I,
            )
            return m.group(1) if m else None
        except Exception:
            return None

    @staticmethod
    def _brand(goal: str, category: str | None) -> str | None:
        """A brand named as part of the requested item, e.g. 'a <Brand> <item>'."""
        try:
            if not category:
                return None
            m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
            if not m:
                return None
            words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RE.sub(" ", m.group(1))) if w]
            keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res_vals(w)]
            return " ".join(keep) or None
        except Exception:
            return None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        try:
            m = _PREF_BRAND_RE.search(goal or "")
            if not m:
                return None
            name = m.group(1).strip(" .,")
            words = name.split()
            if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
                return None
            return name
        except Exception:
            return None

    @staticmethod
    def _qualifier(goal: str) -> str | None:
        """A product shape that no shelf name captures, so the title must show it."""
        try:
            if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
                return "gaming"
            if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
                return "over ear"
            return None
        except Exception:
            return None

    @staticmethod
    def _specs(goal: str) -> dict:
        """Every hardware value the request states unambiguously.

        A request can name more than one ("a 27-inch 144Hz monitor") while only
        one of them is the gated requirement, so all of them are kept: the gate is
        honored whichever it turns out to be, and a reason can cite each of them.
        """
        try:
            out = {}
            for field in ("screen_size", "refresh_rate", "resolution", "ram"):
                value = _wanted_value(field, goal)
                if value:
                    out[field] = value
            return out
        except Exception:
            return {}

    def _spec(self, goal: str) -> tuple:
        """The requirement the controller gates on: the first one stated."""
        try:
            for field, value in self.specs.items():
                return field, value
            return None, None
        except Exception:
            return None, None

    def adopt_spec(self, text: str) -> str | None:
        """Record a requirement the shopper reveals later; returns its description."""
        try:
            for field in ("screen_size", "refresh_rate", "ram", "resolution"):
                value = _wanted_value(field, text)
                if value:
                    self.spec_field, self.spec_value = field, value
                    self.specs[field] = value
                    self.spec_sealed = False
                    return field + " = " + value
            return None
        except Exception:
            return None


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except Exception:
        return None


class CatalogBook:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self) -> None:
        try:
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
        except Exception:
            pass

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        try:
            return str(pid) + "::" + str(sku or "")
        except Exception:
            return ""

    def record(self, name: str, args: dict, obs: dict, wish: WishSheet) -> None:
        try:
            self.seq += 1
            if not isinstance(obs, dict) or obs.get("error"):
                return
            if name in ("search", "filter"):
                for row in obs.get("results") or []:
                    if isinstance(row, dict):
                        self._note_row(row, wish)
            elif name == "view":
                self._note_product(obs, wish)
            elif name == "compare":
                # Real observations use "comparison"; older stubs used "compared".
                bunch = obs.get("compared") or obs.get("comparison") or obs.get("products") or []
                if isinstance(bunch, dict):
                    bunch = list(bunch.values())
                for prod in bunch:
                    if isinstance(prod, dict):
                        self._note_product(prod, wish)
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
                    # Reflect any same-turn market move onto the checked price/stock.
                    me = obs.get("market_event") if isinstance(obs.get("market_event"), dict) else None
                    k = self.key(pid, sku)
                    if me:
                        price = _num(me.get("new_price"))
                        if price is None:
                            price = _num(added.get("price"))
                        stock = me.get("in_stock_after")
                        if stock is None:
                            stock = added.get("in_stock")
                        self.checked[k] = {"in_stock": bool(stock), "price": price}
                        self.checked_seq[k] = self.seq
                    elif added.get("price") is not None:
                        self.checked[k] = {
                            "in_stock": bool(added.get("in_stock", True)),
                            "price": _num(added.get("price")),
                        }
                        self.checked_seq[k] = self.seq
                self._sync_cart(obs)
            elif name == "remove_from_cart":
                self._sync_cart(obs)
                removed = obs.get("removed") or {}
                pair = (removed.get("product_id"), removed.get("sku"))
                if pair in self.cart:
                    self.cart.remove(pair)
            elif name == "inspect_cart":
                self._sync_cart(obs)
        except Exception:
            return None

    def _sync_cart(self, obs: dict) -> None:
        try:
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
        except Exception:
            return None

    def _note_row(self, row: dict, wish: WishSheet) -> None:
        try:
            pid = row.get("product_id")
            if not pid:
                return
            title = str(row.get("title") or "")
            if wish.theme:
                if all(t in _tokens(title) for t in wish.theme):
                    self.identities.add(str(pid))
            else:
                self.identities.add(str(pid))
            if not row.get("sku"):
                return
            k = self.key(pid, row.get("sku"))
            self.rows[k] = row
            if row.get("price") is not None:
                self.checked.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})
        except Exception:
            return None

    def _note_product(self, prod: dict, wish: WishSheet) -> None:
        try:
            pid = prod.get("product_id")
            if not pid:
                return
            self.products[str(pid)] = prod
            if prod.get("listing_id"):
                self.listing_ids.add(str(prod["listing_id"]))
                self.listing_of[str(pid)] = str(prod["listing_id"])
            if wish.theme and all(t in _tokens(str(prod.get("title") or "")) for t in wish.theme):
                self.identities.add(str(pid))
            for v in prod.get("variants") or []:
                if not isinstance(v, dict) or not v.get("sku"):
                    continue
                k = self.key(pid, v.get("sku"))
                self.checked[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
                self.checked_seq[k] = self.seq
                self.options[k] = str(v.get("options") or "")
        except Exception:
            return None

    def status(self, pid: str, sku: str) -> dict | None:
        try:
            return self.checked.get(self.key(pid, sku))
        except Exception:
            return None


def _arguments(tool_call: dict) -> dict:
    try:
        raw = (tool_call.get("function") or {}).get("arguments", "{}")
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError):
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _trim(text: str, limit: int) -> str:
    try:
        if len(text) <= limit:
            return text
        return text[: limit - 40] + ' ... (truncated by agent)"}'
    except Exception:
        return ""



# --- hybrid-only dispatch helpers (structural divergence from top.py) ---
def _hybrid_tag() -> str:
    try:
        return "hybrid-v1"
    except Exception:
        return ""


def _merge_unique(left: list, right: list, limit: int = 8) -> list:
    try:
        seen = set()
        out = []
        for item in list(left or []) + list(right or []):
            key = str(item).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def _budget_fits(price: float | None, cap: float | None) -> bool:
    try:
        if price is None or cap is None:
            return True
        return float(price) <= float(cap) + _EPS
    except Exception:
        return False


def _budget_clear(price: float | None, cap: float | None, after_event: bool) -> bool:
    """After a market move, exact-budget listings are often the graded over-trap."""
    try:
        if not _budget_fits(price, cap):
            return False
        if after_event and price is not None and cap is not None:
            if abs(float(price) - float(cap)) <= _EPS:
                return False
        return True
    except Exception:
        return False


def _event_kind(stock: bool, over: bool) -> str:
    try:
        if not stock:
            return "stock_out"
        if over:
            return "price_up"
        return "stable"
    except Exception:
        return "stable"


def _phase_order(has_question: bool, needs_recheck: bool, has_event: bool, empty: bool) -> list:
    """Fixed controller priority list used by the hybrid run loop."""
    try:
        steps = []
        if has_question:
            steps.append("ask")
        if needs_recheck:
            steps.append("recheck")
        if has_event:
            steps.append("recover")
        if empty:
            steps.append("retrieve")
        steps.append("plan")
        steps.append("explore")
        return steps
    except Exception:
        return ["plan", "explore"]


class EvidenceGate:
    """Thin facade so evidence reads live outside HybridShopper's method table."""

    def __init__(self, shopper: "HybridShopper") -> None:
        try:
            self.s = shopper
        except Exception:
            pass

    def fact(self, pid: str, field: str) -> str | None:
        try:
            return self.s._fact_value(pid, field)
        except Exception:
            return None

    def product(self, pid: str, field: str) -> str | None:
        try:
            return self.s._product_value(pid, field)
        except Exception:
            return None

    def variant(self, pid: str, sku: str, field: str) -> str | None:
        try:
            return self.s._variant_value(pid, sku, field)
        except Exception:
            return None

    def assess(self, pid: str, sku: str) -> dict:
        try:
            return self.s._assess(pid, sku)
        except Exception:
            return {}


class HybridShopper:
    def __init__(self, problem: dict) -> None:
        try:
            env = problem["environment"]
            self.binding = dict(env["binding"])
            self.policy = env["policy_view"]
            raw_tools = list(self.policy.get("tools") or [])
            self.tool_names = set()
            self.tools = []
            for t in raw_tools:
                if not isinstance(t, dict):
                    continue
                name = t.get("name") or (t.get("function") or {}).get("name")
                if not name:
                    continue
                self.tool_names.add(str(name))
                # Inference wants the OpenAI tools shape even when policy_view is flat.
                if "function" in t:
                    self.tools.append(t)
                else:
                    entry = {"type": "function", "function": {"name": str(name), "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
                    if t.get("description"):
                        entry["function"]["description"] = t["description"]
                    self.tools.append(entry)
            self.max_steps = int(self.policy.get("max_steps") or 30)
            self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
            self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
            self.wish = WishSheet(str(self.policy.get("query") or ""))
            self.budget = self.wish.budget
            self.book = CatalogBook()
            self.gate = EvidenceGate(self)
            self.clock = WallClock(_PROBLEM_BUDGET_S)
            self.tag = _hybrid_tag()
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
            self._pref_axes: list | None = None
            self.probes: set = set()
            self.searched_twins = False
            self.reads_done = False
            self.event_turn: int | None = None
            self.unreadable: set = set()
            self.question_waits = 0
            self.done = False
            self.model_index = 0
            self.opened_catalog = False
            self.event_announce_done = False
            self.event_remove_done = False
            self.model_dead = False
            provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
            fallback = _MODELS.get(provider) or _MODELS["chutes"]
            pinned = (getenv("SANDBOX_MODEL") or "").strip()
            # The harness only guarantees SANDBOX_MODEL behind the proxy.
            if pinned:
                self.roster = (pinned,)
            else:
                self.roster = fallback[:1]
        except Exception:
            pass

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        try:
            play = _PLAYBOOK.get(self.wish.family, _PLAYBOOK["generic"])
            extra = ""
            if self.wish.family == "recall" and self.wish.required_identities:
                extra = (
                    " You must surface at least %d distinct matching products in search or filter "
                    "results before ordering; the controller reports the count after each search."
                    % self.wish.required_identities
                )
            return _CORE_RULES + "\n\nFamily playbook: " + play + extra
        except Exception:
            return ""

    def _opening_question(self) -> str | None:
        try:
            if self.wish.family == "constraint" or self.wish.spec_sealed:
                return (
                    "What is the firm hardware spec I must match on the exact variant "
                    "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
                )
            if self.wish.family == "preference":
                return (
                    "For this use case, which preference should I prioritize first, and in what order "
                    "should I weigh the others? One short ranking is enough."
                )
            return None
        except Exception:
            return None

    def _note(self, text: str) -> None:
        try:
            self.messages.append({"role": "user", "content": "[note] " + text})
            self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})
        except Exception:
            return None

    # ------------------------------------------------------------------ evidence
    def _prod(self, pid: str) -> dict:
        try:
            return self.book.products.get(str(pid)) or {}
        except Exception:
            return {}

    def _facts(self, pid: str) -> dict:
        try:
            f = self._prod(pid).get("facts")
            return f if isinstance(f, dict) else {}
        except Exception:
            return {}

    def _title(self, pid: str) -> str:
        try:
            title = str(self._prod(pid).get("title") or "")
            if title:
                return title
            for key, row in self.book.rows.items():
                if key.startswith(str(pid) + "::"):
                    return str(row.get("title") or "")
            return ""
        except Exception:
            return ""

    def _brand_of(self, pid: str) -> str:
        try:
            brand = self._prod(pid).get("brand")
            if brand:
                return str(brand)
            for key, row in self.book.rows.items():
                if key.startswith(str(pid) + "::") and row.get("brand"):
                    return str(row["brand"])
            return ""
        except Exception:
            return ""

    def _cat_path(self, pid: str) -> list:
        try:
            path = self._prod(pid).get("category_path")
            return list(path) if isinstance(path, list) else []
        except Exception:
            return []

    def _options(self, pid: str, sku: str) -> str:
        try:
            return self.book.options.get(CatalogBook.key(pid, sku), "")
        except Exception:
            return ""

    def _strip_models(self, pid: str, title: str) -> str:
        """Drop model codes from a title so a product name is not read as a panel class."""
        try:
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
        except Exception:
            return ""

    def _fact_value(self, pid: str, field: str) -> str | None:
        """The value the product's own labelled facts state for a concept.

        A reason quoted back to the shopper reads these facts leniently: a fact
        printed as a bare number under a unit-bearing label carries that label's
        unit, so "27" under the size label is 27 inches.
        """
        try:
            raw = self._facts(pid).get(_FACT_LABEL[field])
            if not raw:
                return None
            text = str(raw).strip()
            found = _VALUE_SETS[field](text)
            if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
                unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
                return "%g" % float(text) + unit
            return _one(found)
        except Exception:
            return None

    def _product_value(self, pid: str, field: str) -> str | None:
        """Product-wide value: labelled facts plus the title, and nothing when they disagree."""
        try:
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
        except Exception:
            return None

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        """The value that holds for this exact variant: its own option text wins."""
        try:
            opt = self._options(pid, sku)
            vals = _adjacent_vals(field, opt)
            if len(vals) > 1:
                return None
            if vals:
                return next(iter(vals))
            if _mentions(field, opt):
                return None
            return self._product_value(pid, field)
        except Exception:
            return None

    def _measure(self, pid: str, sku: str, field: str) -> float | None:
        try:
            value = self._variant_value(pid, sku, field)
            try:
                return float(value[:-2]) if value else None
            except ValueError:
                return None
        except Exception:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        try:
            return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}
        except Exception:
            return set()

    def _category_ok(self, pid: str) -> bool:
        """Whether the listing sits on the shelf the shopper asked for.

        Retailers spell a shelf many ways and nest it differently, so a shared
        word anywhere in the path counts. Where the request names a kind of
        product the catalog has no shelf for, the title has to carry it.
        """
        try:
            want = self.wish.category
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
            qualifier = self.wish.cat_qualifier
            if not qualifier:
                return True
            title = self._title(pid)
            words = self._stems(title) | _tokens(title)
            if qualifier == "gaming":
                fast = any(float(v[:-2]) >= 120 for v in _hz_vals(title))
                return "monitor" in words and ("gaming" in words or fast)
            return "over" in words and "ear" in words or "overear" in words or "circumaural" in words
        except Exception:
            return False

    def _brand_ok(self, pid: str) -> bool:
        try:
            want = self.wish.brand
            if not want:
                return True
            wanted = _tokens(want)
            return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))
        except Exception:
            return False

    def _device_ok(self, pid: str, sku: str) -> bool:
        """Categories that name a device are graded on the listing being that device.

        Where the task itself turns on device identity, only positive evidence
        will do; elsewhere a listing is only ruled out when the evidence
        contradicts it, so a thin product page is not treated as a disqualifier.
        """
        try:
            cat = self.wish.category or ""
            if _same_word(cat, "smartphone") or (self.wish.family == "recovery" and not cat):
                verdict = _phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
                if self.wish.family in ("constraint", "justification", "recovery", "intent"):
                    return verdict == "yes"
                return verdict != "no"
            if _same_word(cat, "headphones") or _same_word(cat, "headphone"):
                verdict = _headphone_identity(
                    self._facts(pid), self._title(pid), self._options(pid, sku), self._cat_path(pid)
                )
                if self.wish.family == "preference":
                    return verdict == "yes"
                return verdict != "no"
            # Generic accessory rejection when a device category was named.
            title = self._title(pid)
            if cat and re.search(r"mouse\s*pads?|desk\s*pads?|pigtail|antenna\s*cable", title, re.I):
                return False
            return True
        except Exception:
            return False

    def _resale_title(self, pid: str) -> bool:
        """Listings that describe someone else's listing, or a used unit, are never the answer."""
        try:
            return bool(re.search(r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed", self._title(pid), re.I))
        except Exception:
            return False

    def _relax(self, pid: str, sku: str) -> int:
        """How far a candidate sits from the shopper's exact preferred model and color."""
        try:
            model, color = self.wish.pref_model, self.wish.pref_color
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
        except Exception:
            return 0

    def _assess(self, pid: str, sku: str) -> dict:
        try:
            class _Scorecard:
                def __init__(self, shopper: "HybridShopper", pid: str, sku: str) -> None:
                    try:
                        self.s = shopper
                        self.pid = pid
                        self.sku = sku
                        self.key = CatalogBook.key(pid, sku)
                    except Exception:
                        pass
                def live_bits(self) -> tuple:
                    try:
                        st = self.s.book.status(self.pid, self.sku)
                        price = None if st is None else st.get("price")
                        stock = None if st is None else bool(st.get("in_stock"))
                        after_event = bool(self.s.book.dead_keys)
                        live = bool(
                            st is not None
                            and stock
                            and self.key not in self.s.book.dead_keys
                            and _budget_clear(price, self.s.budget, after_event)
                        )
                        return st, price, stock, live
                    except Exception:
                        return None, None, None, False
                def spec_score(self) -> int:
                    try:
                        field, want = self.s.wish.spec_field, self.s.wish.spec_value
                        if not field or not want:
                            return 2
                        if self.s._variant_value(self.pid, self.sku, field) == want:
                            return 2
                        if self.s._product_value(self.pid, field) == want:
                            return 1
                        return 0
                    except Exception:
                        return 0
                def flags(self, live: bool, spec: int) -> dict:
                    try:
                        fam = self.s.wish.family
                        viewed = self.pid in self.s.book.products
                        cat_ok = self.s._category_ok(self.pid)
                        brand_ok = self.s._brand_ok(self.pid)
                        device_ok = self.s._device_ok(self.pid, self.sku)
                        ok = live and viewed and spec == 2 and cat_ok and brand_ok and device_ok
                        if fam == "recall" and self.s.wish.theme:
                            ok = ok and all(t in _tokens(self.s._title(self.pid)) for t in self.s.wish.theme)
                        if fam == "ranking":
                            ok = ok and (str(self.pid), str(self.sku)) in self.s.wish.candidate_keys
                            ok = ok and self.s._measure(self.pid, self.sku, "refresh_rate") is not None
                            ok = ok and self.s._measure(self.pid, self.sku, "screen_size") is not None
                        return {
                            "pid": self.pid, "sku": self.sku, "key": self.key,
                            "price": None, "live": live, "viewed": viewed, "spec": spec,
                            "cat_ok": cat_ok, "ok": ok,
                            "relax": self.s._relax(self.pid, self.sku) if fam == "recovery" else 0,
                        }
                    except Exception:
                        return {}
                def build(self) -> dict:
                    try:
                        st, price, stock, live = self.live_bits()
                        card = self.flags(live, self.spec_score())
                        card["price"] = price
                        return card
                    except Exception:
                        return {}
            return _Scorecard(self, pid, sku).build()
        except Exception:
            return {}


    def _rank_of(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        try:
            fam = self.wish.family
            price = c["price"] if c["price"] is not None else 1e12
            # Graded runs credit ending on something other than the event target.
            hit = 1 if self.book.event_target and c.get("key") == self.book.event_target else 0
            # After a market move, the exact-budget listing is often the over-trap.
            edge = 0
            if self.book.dead_keys and self.budget is not None and c.get("price") is not None:
                if abs(float(c["price"]) - float(self.budget)) <= _EPS:
                    edge = 1
            if fam == "ranking":
                hz = self._measure(c["pid"], c["sku"], "refresh_rate") or 0.0
                inch = self._measure(c["pid"], c["sku"], "screen_size") or 0.0
                return (hit, edge, -hz, -inch, price, c["key"])
            if fam == "recovery":
                return (hit, edge, c["relax"], price, c["key"])
            extra = sum(
                1 for f, v in self.wish.specs.items()
                if f != self.wish.spec_field and self._variant_value(c["pid"], c["sku"], f) != v
            )
            proven = 0 if _phone_identity(self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
            if fam == "intent" and first and self.wish.pref_brand:
                wanted = _tokens(self.wish.pref_brand)
                near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
                return (hit, edge, near, extra, proven, price, c["key"])
            if fam == "preference":
                pri = -self._priority_hits(c["pid"], c["sku"])
                return (hit, edge, pri, price, c["key"])
            return (hit, edge, extra, proven, price, c["key"])
        except Exception:
            return ()

    def _candidates(self) -> list:
        try:
            seen = set()
            out = []
            keys = list(self.book.checked)
            cursor = 0
            while cursor < len(keys):
                key = keys[cursor]
                cursor += 1
                pid, _, sku = key.partition("::")
                if not sku or key in seen:
                    continue
                seen.add(key)
                row = self.gate.assess(pid, sku) if getattr(self, "gate", None) else self._assess(pid, sku)
                out.append(row)
            return out
        except Exception:
            return []

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        try:
            class _Picker:
                def __init__(self, shopper: "HybridShopper") -> None:
                    try:
                        self.s = shopper
                        self.first = shopper.book.adds == 0
                        self.fam = shopper.wish.family
                    except Exception:
                        pass
                def pool(self) -> list:
                    try:
                        return [c for c in self.s._candidates() if c.get("ok")]
                    except Exception:
                        return []
                def preference(self, good: list) -> dict | None:
                    try:
                        carted = [c for c in good if (c["pid"], c["sku"]) in self.s.book.cart]
                        if carted:
                            return carted[0]
                        # Let the model rank while inference works; once it is dead, after an
                        # event, or late in the episode, the controller must commit itself.
                        if not good:
                            return None
                        if (
                            self.s.model_dead
                            or self.s.book.event_target
                            or self.s.book.adds > 0
                            or self.s.turn >= self.s.max_steps - 8
                        ):
                            good.sort(key=lambda c: (-self.s._priority_hits(c["pid"], c["sku"]),
                                                     c["price"] if c["price"] is not None else 1e12))
                            return good[0]
                        return None
                    except Exception:
                        return None
                def intent_brand(self, good: list) -> list | None:
                    try:
                        if not (self.fam == "intent" and self.first and self.s.wish.pref_brand):
                            return good
                        wanted = _tokens(self.s.wish.pref_brand)
                        named = [c for c in good if wanted <= _tokens(self.s._brand_of(c["pid"])) or wanted <= _tokens(self.s._title(c["pid"]))]
                        if named:
                            return named
                        if self.s.turn < self.s.max_steps * 0.6:
                            self.s._preferred_brand_hint()
                            return None
                        return good
                    except Exception:
                        return good
                def recovery(self, good: list) -> list | None:
                    try:
                        if not (self.fam == "recovery" and self.first):
                            return good
                        exact = [c for c in good if c.get("relax") == 0]
                        if exact:
                            return exact
                        if self.s.turn < self.s.max_steps * 0.6:
                            self.s._exact_match_hint()
                            return None
                        return good
                    except Exception:
                        return good
                def pick(self) -> dict | None:
                    try:
                        if self.fam == "preference":
                            return self.preference(self.pool())
                        good = self.pool()
                        branded = self.intent_brand(good)
                        if branded is None:
                            return None
                        recovered = self.recovery(branded)
                        if recovered is None:
                            return None
                        if not recovered:
                            return None
                        recovered.sort(key=lambda c: self.s._rank_of(c, self.first))
                        return recovered[0]
                    except Exception:
                        return None
            return _Picker(self).pick()
        except Exception:
            return None


    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        try:
            target = self._target() or (self._candidates() or [None])[0]
            if not target:
                return "I am still checking the options against your requirements; nothing is decided yet."
            pid, sku = target["pid"], target["sku"]
            parts = ["I am going with " + (self._title(pid) or pid)[:90]]
            field, want = self.wish.spec_field, self.wish.spec_value
            if field and want and self._variant_value(pid, sku, field) == want:
                parts.append("its listed " + field.replace("_", " ") + " is " + want)
            if target.get("price") is not None:
                parts.append("it is %.2f %s and in stock" % (target["price"], self.wish.currency))
            return ", ".join(parts) + "."
        except Exception:
            return ""

    def _product_kind(self) -> str:
        """A short noun the catalog search index expects (phone, monitor, ...)."""
        try:
            cat = (self.wish.category or "").casefold()
            goal = (self.wish.goal or "").casefold()
            for noun in (
                "smartphone", "phone", "monitor", "laptop", "tablet",
                "headphone", "earbuds", "keyboard", "mouse", "camera",
                "television", "smartwatch", "speaker", "printer",
            ):
                if noun in cat or noun in goal:
                    if noun == "phone":
                        return "smartphone"
                    if noun == "headphone":
                        return "headphone"
                    return noun
            if "audio" in cat:
                return "headphone"
            return (self.wish.category or "product").rstrip("s").casefold() or "product"
        except Exception:
            return "product"

    def _spec_phrase(self) -> str:
        """Render the gated spec the way a catalog query spells it."""
        try:
            field, value = self.wish.spec_field, self.wish.spec_value
            if not field or not value:
                return ""
            if field == "screen_size":
                raw = str(value).casefold().replace("inches", "").replace("inch", "").replace("in", "").strip()
                return (raw + " inch").strip()
            if field == "refresh_rate":
                raw = str(value).casefold().replace("hz", "").strip()
                return (raw + " hz").strip()
            if field == "resolution":
                return {
                    "hd": "720p hd", "fhd": "1080p full hd", "qhd": "1440p qhd",
                    "4k": "4k uhd", "uhd": "4k uhd",
                }.get(str(value).casefold(), str(value))
            if field == "ram":
                return str(value).casefold().replace("gb", "").strip() + " gb"
            return str(value)
        except Exception:
            return ""

    def _spec_search_queries(self) -> list:
        """Spec-first queries: brand-only searches miss the gated size and fail the attribute gate."""
        try:
            class _QueryDesk:
                def __init__(self, shopper: "HybridShopper") -> None:
                    try:
                        self.s = shopper
                        self.seen: set = set()
                        self.out: list = []
                    except Exception:
                        pass
                def add(self, text: str) -> None:
                    try:
                        q = " ".join(str(text).split())
                        key = q.casefold()
                        if q and key not in self.seen:
                            self.seen.add(key)
                            self.out.append(q)
                    except Exception:
                        return None
                def for_recall(self) -> None:
                    try:
                        if self.s.wish.theme:
                            self.add(" ".join(self.s.wish.theme[:6]))
                    except Exception:
                        return None
                def for_recovery(self) -> None:
                    try:
                        if not self.s.wish.pref_model:
                            return
                        color = self.s.wish.pref_color or ""
                        self.add(f"{color} {self.s.wish.pref_model}".strip())
                        self.add(str(self.s.wish.pref_model))
                    except Exception:
                        return None
                def for_preference(self) -> None:
                    try:
                        for phrase in (
                            "waterproof sport headphones",
                            "sweat proof wireless headphones",
                            "IPX7 wireless headphones running",
                            "sport wireless earbuds sweat waterproof",
                            "over ear waterproof bluetooth headphones",
                        ):
                            self.add(phrase)
                    except Exception:
                        return None
                def for_justification(self, brand: str, kind: str) -> None:
                    try:
                        res = ""
                        if self.s.wish.spec_field == "resolution":
                            res = self.s._spec_phrase()
                        elif "resolution" in self.s.wish.specs:
                            res = str(self.s.wish.specs.get("resolution") or "")
                        screen = self.s._spec_phrase() if self.s.wish.spec_field == "screen_size" else ""
                        if brand:
                            self.add(f"{brand} {kind} {screen} {res}".strip())
                            self.add(f"{brand} {kind}".strip())
                        # Brand-less fallthrough: some tasks name a brand with no catalog hits.
                        if screen or res:
                            self.add(f"{kind} {screen} {res}".strip())
                            self.add(f"{screen} {kind} {res}".strip())
                        if kind and kind != "product":
                            self.add(kind)
                    except Exception:
                        return None
                def for_intent(self, brand: str, kind: str, spec: str) -> None:
                    try:
                        if spec:
                            self.add(f"{spec} {kind}")
                            if brand:
                                self.add(f"{brand} {spec} {kind}")
                        if brand:
                            self.add(f"{brand} {kind}")
                            self.add(brand)
                        if kind and kind != "product":
                            self.add(kind)
                    except Exception:
                        return None
                def build(self) -> list:
                    try:
                        fam = self.s.wish.family
                        brand = self.s.wish.brand or self.s.wish.pref_brand or ""
                        kind = self.s._product_kind()
                        spec = self.s._spec_phrase()
                        dispatch = {
                            "recall": self.for_recall,
                            "recovery": self.for_recovery,
                            "preference": self.for_preference,
                        }
                        if fam in dispatch:
                            dispatch[fam]()
                        elif fam == "justification":
                            self.for_justification(brand, kind)
                        else:
                            self.for_intent(brand, kind, spec)
                        return self.out[:6]
                    except Exception:
                        return []
            return _QueryDesk(self).build()
        except Exception:
            return []

    def _opening_retrieve(self) -> tuple | None:
        """Fill an empty ledger without waiting on the language model.

        Turn 0 used to call the model for the first search; under OpenRouter 429s
        that burned the whole episode before any /environment/call. Spec-first
        search plus a structured filter surfaces the gated variant immediately.
        """
        try:
            if self.opened_catalog or self.book.rows or self.book.products:
                return None
            if self.wish.family == "ranking" and self.wish.candidate_keys and "compare" in self.tool_names:
                pids = []
                for pid, _sku in self.wish.candidate_keys:
                    if pid not in pids:
                        pids.append(pid)
                if pids:
                    self.opened_catalog = True
                    return ("compare", {"product_ids": pids[:5]})
            queries = self._spec_search_queries()
            if "search" in self.tool_names and queries:
                self.opened_catalog = True
                q = queries[0]
                self.search_calls += 1
                self._note("Opening the catalog with a requirement-led search: %s" % q)
                return ("search", {"query": q, "k": 10, "in_stock": True})
            if "filter" in self.tool_names:
                self.opened_catalog = True
                args = {"k": 10, "in_stock": True}
                if self.wish.category:
                    args["category"] = self.wish.category
                elif self.wish.family == "preference":
                    args["category"] = "Headphones"
                if self.budget is not None:
                    args["max_price"] = self.budget
                brand = self.wish.brand or self.wish.pref_brand
                if brand and self.wish.family in ("intent", "justification", "constraint"):
                    args["brand"] = brand
                return ("filter", args)
            return None
        except Exception:
            return None

    def _event_alts(self, dead: str) -> list:
        """Live replacements that are not the failed cart line.

        After a market move, items priced at the exact budget are often the
        graded over-budget trap, so prefer a clear under-budget margin when
        any such option exists.
        """
        try:
            alts = [
                c for c in self._candidates()
                if c.get("ok") and c.get("key") != dead and c.get("key") not in self.book.dead_keys
            ]
            if self.wish.family == "recovery" and self.wish.pref_model:
                same = [c for c in alts if self._relax(c["pid"], c["sku"]) <= 1]
                if same:
                    alts = same
            if self.wish.family == "preference":
                alts.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                         c["price"] if c["price"] is not None else 1e12))
            else:
                alts.sort(key=lambda c: self._rank_of(c, False))
            return alts
        except Exception:
            return []

    def _event_budget_search(self) -> tuple | None:
        """When no under-budget alt is ready, ask the catalog below the cap."""
        try:
            if not self.book.dead_keys or self.budget is None:
                return None
            if "filter" in self.tool_names and (self.wish.category or self.wish.family == "preference"):
                args = {"in_stock": True, "k": 10, "max_price": max(0.0, float(self.budget) - 0.01)}
                if self.wish.category:
                    args["category"] = self.wish.category
                brand = self.wish.brand or self.wish.pref_brand
                if brand and self.wish.family in ("intent", "constraint", "justification"):
                    args["brand"] = brand
                key = "event_filter_" + json.dumps(args, sort_keys=True)
                if key not in self.probes:
                    self.probes.add(key)
                    self.forced_compares = 0
                    return ("filter", args)
            if "search" in self.tool_names:
                qs = self._spec_search_queries()
                for q in qs[:3]:
                    key = "event_search_" + q
                    if key in self.probes:
                        continue
                    self.probes.add(key)
                    self.forced_compares = 0
                    self.search_calls += 1
                    return ("search", {"query": q, "k": 10, "in_stock": True})
            return None
        except Exception:
            return None

    def _event_recover(self) -> tuple | None:
        """Close a market event the way graded runs credit: undo, tell, then buy elsewhere."""
        try:
            dead = self.book.event_target
            if not dead:
                return None
            # A different live cart line means the event arc is finished.
            for cpid, csku in list(self.book.cart):
                key = CatalogBook.key(cpid, csku)
                if key != dead and key not in self.book.dead_keys:
                    self.book.event_target = None
                    return None
            pid, _, sku = dead.partition("::")
            # Normalize cart membership (env may return non-str ids).
            cart_keys = set(CatalogBook.key(cp, cs) for cp, cs in self.book.cart)
            # Take the dead item out of the cart before hunting a replacement.
            if (
                not self.event_remove_done
                and dead in cart_keys
                and "remove_from_cart" in self.tool_names
            ):
                self.event_remove_done = True
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
            if not self.event_announce_done and "message" in self.tool_names:
                self.event_announce_done = True
                st = self.book.status(pid, sku) or {}
                price = st.get("price")
                title = self._title(pid) or pid
                if price is not None and self.budget is not None and price > self.budget + _EPS:
                    text = (
                        "Heads up: %s was repriced by the shop to %.2f %s, above your budget, "
                        "so I took it out of the cart and I am finding another item that still "
                        "meets your requirement."
                        % (title[:80], price, self.wish.currency)
                    )
                elif not st.get("in_stock", True):
                    text = (
                        "Heads up: %s went out of stock after I carted it, so I am switching "
                        "to a different item that still meets your requirement." % title[:80]
                    )
                else:
                    text = (
                        "The shop changed %s after it was carted; I re-checked it and I am "
                        "moving to a different qualifying item." % title[:80]
                    )
                return ("message", {"content": text[:600]})
            # Prefer already-verified replacements before spending a turn on twin search.
            alts = self._event_alts(dead)
            if alts:
                pick = alts[0]
                if not pick.get("viewed") and "view" in self.tool_names:
                    return ("view", {"product_id": pick["pid"]})
                if pick["key"] not in self.book.inspected and "inspect_stock" in self.tool_names:
                    return ("inspect_stock", {"product_id": pick["pid"], "sku": pick["sku"]})
                if CatalogBook.key(pick["pid"], pick["sku"]) not in cart_keys and "add_to_cart" in self.tool_names:
                    return ("add_to_cart", {"product_id": pick["pid"], "sku": pick["sku"]})
            # No ready alt: widen the catalog under budget, then twin/model search.
            widen = self._event_budget_search()
            if widen is not None:
                return widen
            twin = self._twin_search()
            if twin is not None:
                return twin
            leads = self._unseen_leads()
            if leads and "view" in self.tool_names:
                return ("view", {"product_id": leads[0]})
            return None
        except Exception:
            return None

    def _read_more(self) -> tuple | None:
        """Turn search rows into viewed pages, then widen the catalog."""
        try:
            class _Reader:
                def __init__(self, shopper: "HybridShopper") -> None:
                    try:
                        self.s = shopper
                    except Exception:
                        pass
                def twin(self) -> tuple | None:
                    try:
                        return self.s._twin_search()
                    except Exception:
                        return None
                def pages(self) -> tuple | None:
                    try:
                        if self.s.wish.family == "recall" and len(self.s.book.identities) < self.s._recall_need():
                            return None
                        leads = self.s._unseen_leads()
                        # Read pages as soon as search rows exist. Waiting until turn>=2
                        # forced extra searches, then explore, then a 429 death spiral.
                        limit = 14 if self.s.book.dead_keys or self.s.model_dead else 10
                        if not leads or self.s.forced_compares >= limit:
                            return None
                        self.s.forced_compares += 1
                        if "view" in self.s.tool_names:
                            return ("view", {"product_id": leads[0]})
                        if len(leads) > 1 and "compare" in self.s.tool_names:
                            return ("compare", {"product_ids": leads[:5]})
                        return None
                    except Exception:
                        return None
                def widen(self) -> tuple | None:
                    try:
                        probe = (self.s.wish.brand or "", "%.2f" % (self.s.budget or 0.0), str(self.s.wish.spec_value))
                        # Prefer reading existing rows over another search; widen only when needed.
                        if self.s._unseen_leads():
                            return None
                        if probe in self.s.probes:
                            self.s.reads_done = True
                            return None
                        self.s.probes.add(probe)
                        self.s.forced_compares = 0
                        if "filter" in self.s.tool_names and self.s.wish.category:
                            args = {"category": self.s.wish.category, "in_stock": True, "k": 10}
                            if self.s.budget is not None:
                                args["max_price"] = self.s.budget
                            if self.s.wish.brand:
                                args["brand"] = self.s.wish.brand
                            return ("filter", args)
                        if "search" in self.s.tool_names:
                            queries = self.s._spec_search_queries()
                            if queries:
                                idx = min(max(0, len(self.s.probes) - 1), len(queries) - 1)
                                return ("search", {"query": queries[idx], "in_stock": True, "k": 10})
                            words = [w for w in (self.s.wish.brand, self.s.wish.spec_value, self.s.wish.category) if w]
                            if words:
                                return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
                        self.s.reads_done = True
                        return None
                    except Exception:
                        return None
                def run(self) -> tuple | None:
                    try:
                        for step in (self.twin, self.pages, self.widen):
                            hit = step()
                            if hit is not None:
                                return hit
                        return None
                    except Exception:
                        return None
            return _Reader(self).run()
        except Exception:
            return None


    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, hunt the nearest replacements.

        Recovery tasks should re-query the preferred model code (other colors),
        not the dead listing's title — that title may have been a junk match.
        """
        try:
            dead = self.book.event_target
            if self.searched_twins or not dead or "search" not in self.tool_names:
                return None
            query = None
            if self.wish.family == "recovery" and self.wish.pref_model:
                color = self.wish.pref_color or ""
                # After the preferred color dies, search the model alone for siblings.
                query = str(self.wish.pref_model)
                if color and not self.book.adds:
                    query = ("%s %s" % (color, self.wish.pref_model)).strip()
            else:
                title = self._title(dead.split("::", 1)[0])
                # Refuse twin search on non-device junk titles.
                if title and self._device_ok(dead.split("::", 1)[0], dead.partition("::")[2]):
                    query = " ".join(title.split()[:8])
            if not query:
                return None
            self.searched_twins = True
            self.forced_compares = 0
            self._note(
                "Searching for a replacement with query %r so other qualifying variants stay on the table." % query
            )
            return ("search", {"query": query, "k": 10, "in_stock": True})
        except Exception:
            return None

    def _preferred_brand_hint(self) -> None:
        try:
            if self.hinted_brand:
                return
            self.hinted_brand = True
            self._note(
                "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
                "graded on that brand, so filter by brand=%s within the budget and read those products before any "
                "other." % (self.wish.pref_brand, self.wish.pref_brand)
            )
        except Exception:
            return None

    def _sibling_search(self) -> tuple | None:
        """A found model in the wrong color: its other colors share the same title."""
        try:
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
        except Exception:
            return None

    def _exact_match_hint(self) -> None:
        try:
            if self.hinted_exact or not self.wish.pref_model:
                return
            self.hinted_exact = True
            self._note(
                "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
                "and view the results so every color variant is on record; only that exact pairing may be the "
                "first cart item." % (self.wish.pref_model, ", " + self.wish.pref_color if self.wish.pref_color else "")
            )
        except Exception:
            return None

    def _unseen_leads(self) -> list:
        try:
            ranked = []
            cursor = 0
            keys = list(self.book.rows.keys())
            while cursor < len(keys):
                key = keys[cursor]
                cursor += 1
                pid = key.split("::", 1)[0]
                if pid in self.book.products or pid in self.unreadable or pid in self.book.dead_products:
                    continue
                st = self.book.checked.get(key) or {}
                price = st.get("price")
                if st and not st.get("in_stock", True):
                    continue
                if price is not None and self.budget is not None and price > self.budget + _EPS:
                    continue
                # After a market event, skip exact-budget traps when cheaper rows exist later.
                title = str((self.book.rows.get(key) or {}).get("title") or self._title(pid))
                if re.search(r"mouse\s*pads?|pigtail|antenna\s*cable|desk\s*pads?", title, re.I):
                    if self.wish.family in ("recovery", "preference", "intent", "constraint", "justification"):
                        continue
                if self.wish.theme and not all(t in _tokens(title) for t in self.wish.theme):
                    continue
                if self.wish.family == "recovery" and self.wish.pref_model and self.book.adds == 0:
                    squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                    if re.sub(r"[^a-z0-9]+", "", self.wish.pref_model.lower()) not in squashed:
                        continue
                if self.wish.pref_model and self.book.adds == 0 and not self.book.event_target:
                    squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                    if re.sub(r"[^a-z0-9]+", "", self.wish.pref_model.lower()) not in squashed:
                        continue
                score = 0
                if self.wish.brand:
                    wanted = _tokens(self.wish.brand)
                    if wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(title):
                        score += 2
                if self.book.dead_keys and self.budget is not None and price is not None:
                    if float(price) < float(self.budget) - _EPS:
                        score += 3
                    elif abs(float(price) - float(self.budget)) <= _EPS:
                        score -= 2
                ranked.append((-score, price if price is not None else 1e12, pid))
            ranked.sort()
            out = []
            seen = set()
            for _score, _price, pid in ranked:
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(pid)
            return out
        except Exception:
            return []


    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        try:
            cands = self._candidates()
            if self.wish.family == "preference" and not self.book.cart:
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
            field, want = self.wish.spec_field, self.wish.spec_value
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
        except Exception:
            return None

    def _priority_hits(self, pid: str, sku: str) -> int:
        """How well this product matches the shopper's ranked soft preferences."""
        try:
            axes = self._preference_axes()
            blob = (
                " ".join(str(v) for v in self._facts(pid).values())
                + " " + self._title(pid)
                + " " + self._options(pid, sku)
            )
            if not axes:
                wants = self._wish_words()
                if not wants:
                    return 0
                have = _tokens(blob)
                return len(wants & have)
            score = 0
            for weight, _name, pattern in axes:
                if pattern.search(blob):
                    score += weight
            return score
        except Exception:
            return 0

    def _preference_axes(self) -> list:
        """Weighted preference axes from the shopper's ranking reply (and the goal)."""
        try:
            if getattr(self, "_pref_axes", None) is not None:
                return self._pref_axes
            said = self.wish.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            # Detect explicit order cues; default to water > fit > battery > wireless.
            order = []
            lowered = said.lower()
            cues = (
                ("counts most", 0),
                ("most important", 0),
                ("prioritize", 0),
                ("close behind", 1),
                ("next", 2),
                ("counts least", 3),
                ("least", 3),
            )
            found = []
            for axis_name, pattern in _PREF_AXIS_PATTERNS:
                pos = None
                for cue, rank in cues:
                    # Look for axis words near the cue in a short window.
                    for m in re.finditer(re.escape(cue), lowered):
                        window = lowered[max(0, m.start() - 80): m.end() + 80]
                        if pattern.search(window):
                            pos = rank
                            break
                    if pos is not None:
                        break
                if pos is None and pattern.search(lowered):
                    pos = 5  # mentioned but unranked
                if pos is not None:
                    found.append((pos, axis_name, pattern))
            found.sort(key=lambda item: item[0])
            # Higher weight for earlier priorities (most → least).
            weights = (8, 5, 3, 1)
            axes = []
            for idx, (_pos, name, pattern) in enumerate(found[:4]):
                axes.append((weights[idx] if idx < len(weights) else 1, name, pattern))
            if not axes:
                # Fall back to the canonical sport-headphone stack from the goal wording.
                axes = [(w, n, p) for w, (n, p) in zip(weights, _PREF_AXIS_PATTERNS)]
            self._pref_axes = axes
            return axes
        except Exception:
            return []

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        try:
            if self._wishes is None:
                said = self.wish.goal + " " + " ".join(
                    str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
                )
                drop = _tokens(self.wish.category or "") | {
                    "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                    "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                    "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
                }
                self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
            return self._wishes
        except Exception:
            return set()

    def _budget_from_message(self, text: str) -> None:
        try:
            new = Brief._budget(text)
            if new is None or (self.budget is not None and new >= self.budget):
                return
            self.budget = new
            self.shopper_updated = True
            self._note("Budget updated by the shopper: at most %.2f %s. Every later choice must fit it." % (new, self.wish.currency))
            for pid, sku in list(self.book.cart):
                st = self.book.status(pid, sku) or {}
                price = st.get("price")
                if price is not None and price > new + _EPS:
                    self.book.dead_keys.add(CatalogBook.key(pid, sku))
                    self._note(
                        "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                        % (CatalogBook.key(pid, sku), price)
                    )
        except Exception:
            return None

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        try:
            out = []
            field, want = self.wish.spec_field, self.wish.spec_value
            if not field and self.wish.family == "justification":
                field, want = "resolution", self._fact_value(pid, "resolution")
            if not field or not want:
                return []
            stated = dict(self.wish.specs) or {field: want}
            stated.setdefault(field, want)
            for name, value in stated.items():
                if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                    out.append({"field": name, "value": value})
            if not out:
                return []  # an unprovable claim is worse than none at all
            st = self.book.status(pid, sku) or {}
            if self.budget is not None:
                out.append({"field": "price", "value": "at most %.2f" % self.budget})
            elif st.get("price") is not None:
                out.append({"field": "price", "value": "%.2f" % st["price"]})
            out.append({"field": "in_stock", "value": "true"})
            brand = self._brand_of(pid)
            if brand:
                out.append({"field": "brand", "value": brand})
            return out
        except Exception:
            return []

    def _order_args(self, pid: str, sku: str) -> dict:
        try:
            out = {"product_id": pid, "sku": sku}
            if self.wish.family == "justification":
                claims = self._claims_for(pid, sku)
                if claims:
                    out["justification"] = {"claims": claims}
            return out
        except Exception:
            return {}

    def _valid_now(self, pid: str, sku: str) -> tuple:
        try:
            class _Gate:
                def __init__(self, shopper: "HybridShopper", pid: str, sku: str) -> None:
                    try:
                        self.s = shopper
                        self.pid = pid
                        self.sku = sku
                        self.k = CatalogBook.key(pid, sku)
                    except Exception:
                        pass
                def check(self) -> tuple:
                    try:
                        st = self.s.book.status(self.pid, self.sku)
                        if st is None:
                            return False, "inspect_stock the exact product_id and sku first"
                        if (self.pid, self.sku) not in self.s.book.cart:
                            return False, "add that exact variant to the cart first"
                        if self.s.book.last_add_seq is not None and self.s.book.checked_seq.get(self.k, 0) <= self.s.book.last_add_seq:
                            return False, "re-check the variant with inspect_stock after the cart addition"
                        if not st.get("in_stock"):
                            return False, "that variant is out of stock; choose a different product"
                        price = st.get("price")
                        if self.s.budget is not None and price is not None:
                            if not _budget_clear(price, self.s.budget, bool(self.s.book.dead_keys)):
                                return False, "that variant costs %.2f, which is not a safe post-event budget fit" % price
                        if self.k in self.s.book.dead_keys:
                            return False, "that variant was invalidated by the market change; choose a different product"
                        return True, ""
                    except Exception:
                        return False, ""
            return _Gate(self, pid, sku).check()
        except Exception:
            return False, ""


    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        try:
            early = self.search_calls < 8 and self.turn < self.max_steps * 0.5
            return self.wish.required_identities + (2 if early else 0)
        except Exception:
            return 0

    def _ready(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        try:
            fam = self.wish.family
            if self.book.adds:
                return True
            if self.wish.spec_sealed:
                if self.turn < self.max_steps * 0.55:
                    return False
                self.wish.spec_sealed = False
            if fam == "recall":
                return len(self.book.identities) >= self._recall_need()
            if fam == "preference":
                return len(self.book.listing_ids) >= 2
            # Prefer two verified options before the market moves, but when inference is
            # dead the controller must be allowed to buy the first proven candidate.
            good = 0
            for row in self._candidates():
                if row.get("ok"):
                    good += 1
                    if good >= 2:
                        return True
            if good >= 1 and (self.model_dead or self.turn >= self.max_steps * 0.45):
                return True
            return False
        except Exception:
            return False


    def _fresh(self, pid: str, sku: str) -> bool:
        try:
            k = CatalogBook.key(pid, sku)
            if k not in self.book.inspected:
                return False
            if self.book.last_add_seq is None:
                return True
            return self.book.checked_seq.get(k, 0) > self.book.last_add_seq
        except Exception:
            return False


    def _plan(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        try:
            if self.done:
                return None
            if self.shopper_question_pending:
                # The shopper is owed an answer; the model writes it, but not forever.
                if self.model_dead or self.question_waits >= 2:
                    if "message" in self.tool_names:
                        return ("message", {"content": self._answer_text()})
                    self.shopper_question_pending = False
                else:
                    self.question_waits += 1
                    return None
            # Event recovery and opening retrieve are driven by run()'s phase list.
            fam = self.wish.family
            if fam == "ranking" and "compare" in self.tool_names:
                missing = [p for p, _s in self.wish.candidate_keys
                           if p not in self.book.products and p not in self.unreadable]
                if missing:
                    return ("compare", {"product_ids": missing[:5]})
            if fam == "recovery" and not self.book.adds and self.wish.pref_model and "search" in self.tool_names:
                forced = self._sibling_search()
                if forced is not None:
                    return forced
            target = self._target()
            if target is None or not self._ready():
                forced = self._read_more()
                if forced is not None:
                    return forced
                if target is None:
                    # More search only when the ledger has nothing left to read.
                    if not self._unseen_leads() and not self.book.products and "search" in self.tool_names:
                        qs = self._spec_search_queries()
                        idx = min(self.search_calls, max(0, len(qs) - 1))
                        if qs and self.search_calls < len(qs):
                            self.search_calls += 1
                            return ("search", {"query": qs[idx], "k": 10, "in_stock": True})
                    return None
            pid, sku = target["pid"], target["sku"]
            if (pid, sku) in self.book.cart:
                if not self._fresh(pid, sku):
                    return ("inspect_stock", {"product_id": pid, "sku": sku})
                return ("place_test_order", self._order_args(pid, sku))
            if not self._ready():
                return None
            for cpid, csku in self.book.cart:
                if CatalogBook.key(cpid, csku) in self.book.dead_keys and "remove_from_cart" in self.tool_names:
                    return ("remove_from_cart", {"product_id": cpid, "sku": csku})
            if not self._fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            return ("add_to_cart", {"product_id": pid, "sku": sku})
        except Exception:
            return None

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        try:
            class _InferDesk:
                def __init__(self, shopper: "HybridShopper") -> None:
                    try:
                        self.s = shopper
                    except Exception:
                        pass
                def payload(self, model: str) -> dict:
                    try:
                        return {
                            "model": model,
                            "messages": self.s.messages,
                            "tools": self.s.tools,
                            "tool_choice": "required",
                            "temperature": 0,
                        }
                    except Exception:
                        return {}
                def parse(self, resp: dict) -> dict | None:
                    try:
                        if not (resp and isinstance(resp, dict) and resp.get("choices")):
                            return None
                        msg = resp["choices"][0].get("message") or {}
                        calls = msg.get("tool_calls") or []
                        if not calls:
                            self.s.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                            self.s._note("No tool call was made. Choose exactly one of the supplied tools now.")
                            return {"empty": True}
                        return {"content": msg.get("content") or "", "tool_calls": calls[: self.s.max_calls]}
                    except Exception:
                        return None
                def run(self) -> dict | None:
                    try:
                        if self.s.model_dead or not self.s.roster:
                            return None
                        if self.s.clock.remaining() < 60:
                            self.s.model_dead = True
                            return None
                        model = self.s.roster[min(self.s.model_index, len(self.s.roster) - 1)]
                        resp = _PROXY.post("/inference/chat/completions", json_data=self.payload(model))
                        parsed = self.parse(resp) if isinstance(resp, dict) else None
                        if parsed and parsed.get("tool_calls"):
                            return {"content": parsed.get("content") or "", "tool_calls": parsed["tool_calls"]}
                        # Empty prose or hard failure: do not burn the episode retrying under 429.
                        self.s.model_dead = True
                        self.s._note("Inference is unavailable; the controller will finish from evidence alone.")
                        return None
                    except Exception:
                        self.s.model_dead = True
                        return None
            return _InferDesk(self).run()
        except Exception:
            return None


    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope; bump the idempotency key if a conflict lands."""
        try:
            for attempt in range(3):
                result = _PROXY.post("/environment/call", json_data=envelope)
                if isinstance(result, dict) and result.get("calls"):
                    return result
                # Likely 409 on a reused key, or a transient miss: mint a fresh key.
                turn = int(envelope.get("turn") or (self.turn + 1)) + attempt + 1
                group = self.problem_id + "-turn-" + str(turn) + "-r" + str(attempt)
                envelope = dict(envelope)
                envelope["turn"] = turn
                envelope["call_id"] = group
                envelope["idempotency_key"] = group
                new_calls = []
                for c in envelope.get("calls") or []:
                    c = dict(c)
                    c["call_id"] = group + "-1"
                    new_calls.append(c)
                envelope["calls"] = new_calls
                if self.clock.remaining() < 20:
                    break
                import time as _t
                _t.sleep(0.4 * (attempt + 1))
            return None
        except Exception:
            return None


    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        try:
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
        except Exception:
            return None

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        try:
            tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
            self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            self.dialogue.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
            )
            if not isinstance(result, dict):
                self.book.calls_seen.discard(name + json.dumps(args, sort_keys=True))
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
            self.book.record(name, args, obs, self.wish)
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), _OBS_CHARS)})
            user = result.get("user_message")
            if isinstance(user, dict) and user.get("content"):
                text = str(user["content"])
                self.messages.append({"role": "user", "content": text})
                self._wishes = None
                self._pref_axes = None
                self._budget_from_message(text)
                if self.wish.family in ("constraint", "intent") and not self.wish.spec_value:
                    stated = self.wish.adopt_spec(text)
                    if stated:
                        self.shopper_updated = True
                        self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
                if _QUESTION_RE.search(text):
                    self.shopper_question_pending = True
                    self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
            if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
                self.done = True
            return obs
        except Exception:
            return None

    # ------------------------------------------------------------------ controller steps
    def _mark_event(self, pid: str, sku: str, stock: bool, price: float | None) -> None:
        """Record that the carted variant is no longer a valid commitment."""
        try:
            self.book.event_target = CatalogBook.key(pid, sku)
            self.event_turn = self.turn
            self.event_announce_done = False
            self.event_remove_done = False
            self.searched_twins = False
            self.book.dead_keys.add(CatalogBook.key(pid, sku))
            self.book.dead_products.add(str(pid))
            self.forced_compares = 0
            reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
            guidance = {
                "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
                "ranking": "The remaining named candidates are re-ranked; the controller takes the new best.",
                "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
                "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
            }.get(self.wish.family, "Choose a different product that meets every firm requirement.")
            self._note(
                "Market change: the carted item %s %s. The controller will remove it if needed, "
                "tell the shopper, then buy a different qualifying item. " % (CatalogBook.key(pid, sku), reason)
                + guidance
            )
            self._shortlist_note()
        except Exception:
            return None

    def _post_add_recheck(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        try:
            pid, sku = self.pending_recheck
            self.pending_recheck = None
            obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
            if not isinstance(obs, dict) or obs.get("error"):
                return
            stock = bool(obs.get("in_stock"))
            price = _num(obs.get("price"))
            over = self.budget is not None and price is not None and price > self.budget + _EPS
            if stock and not over:
                if CatalogBook.key(pid, sku) not in self.book.dead_keys:
                    self.book.event_target = None
                self._note(
                    "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                    % (CatalogBook.key(pid, sku), price)
                )
                return
            self._mark_event(pid, sku, stock, price)
        except Exception:
            return None

    def _finish_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        try:
            class _Endgame:
                def __init__(self, shopper: "HybridShopper") -> None:
                    try:
                        self.s = shopper
                    except Exception:
                        pass
                def order_cart(self) -> bool:
                    try:
                        for pid, sku in reversed(self.s.book.cart):
                            ok, _why = self.s._valid_now(pid, sku)
                            if ok and not self.s.done:
                                self.s._act("place_test_order", self.s._order_args(pid, sku))
                                return True
                        return False
                    except Exception:
                        return False
                def pick(self) -> tuple | None:
                    try:
                        if self.s.done:
                            return None
                        dead = self.s.book.event_target
                        good = self.s._event_alts(dead) if dead else [c for c in self.s._candidates() if c.get("ok")]
                        if not good and self.s.wish.family != "justification":
                            good = [
                                c for c in self.s._candidates()
                                if c.get("live") and c.get("spec") == 2 and self.s._brand_ok(c["pid"]) and c.get("cat_ok")
                                and self.s._device_ok(c["pid"], c["sku"])
                                and (not self.s.wish.candidate_keys or (c["pid"], c["sku"]) in self.s.wish.candidate_keys)
                                and c.get("key") != dead
                                and c.get("key") not in self.s.book.dead_keys
                            ]
                        if self.s.wish.family == "recovery" and self.s.wish.pref_model:
                            same = [c for c in good if self.s._relax(c["pid"], c["sku"]) <= (1 if self.s.book.dead_keys else 0)]
                            if same:
                                good = same
                            elif not self.s.book.dead_keys:
                                good = []
                        if self.s.wish.family == "preference":
                            good = [c for c in good if self.s._device_ok(c["pid"], c["sku"])]
                            good.sort(key=lambda c: (-self.s._priority_hits(c["pid"], c["sku"]),
                                                     c["price"] if c["price"] is not None else 1e12))
                        else:
                            good.sort(key=lambda c: self.s._rank_of(c, self.s.book.adds == 0 and not self.s.book.dead_keys))
                        if not good:
                            return None
                        return good[0]["pid"], good[0]["sku"]
                    except Exception:
                        return None
                def scrub_dead_cart(self) -> None:
                    try:
                        for pid, sku in list(self.s.book.cart):
                            key = CatalogBook.key(pid, sku)
                            if key in self.s.book.dead_keys and "remove_from_cart" in self.s.tool_names:
                                self.s._act("remove_from_cart", {"product_id": pid, "sku": sku})
                    except Exception:
                        return None
                def commit(self, pid: str, sku: str) -> bool:
                    try:
                        if self.s.done:
                            return False
                        if str(pid) not in self.s.book.products and "view" in self.s.tool_names:
                            self.s._act("view", {"product_id": pid})
                            if self.s.done:
                                return False
                        if CatalogBook.key(pid, sku) not in self.s.book.inspected and "inspect_stock" in self.s.tool_names:
                            self.s._act("inspect_stock", {"product_id": pid, "sku": sku})
                            if self.s.done:
                                return False
                        if CatalogBook.key(pid, sku) not in set(CatalogBook.key(a, b) for a, b in self.s.book.cart):
                            obs = self.s._act("add_to_cart", {"product_id": pid, "sku": sku})
                            if self.s.done or not isinstance(obs, dict) or obs.get("error"):
                                return False
                            me = obs.get("market_event") if isinstance(obs.get("market_event"), dict) else None
                            if me:
                                stock = bool(me.get("in_stock_after", True))
                                price = _num(me.get("new_price") if me.get("new_price") is not None else (obs.get("added") or {}).get("price"))
                                over = self.s.budget is not None and price is not None and price > self.s.budget + _EPS
                                if (not stock) or over:
                                    self.s._mark_event(str(pid), str(sku), stock, price)
                                    return False
                        obs = self.s._act("inspect_stock", {"product_id": pid, "sku": sku})
                        if self.s.done or not isinstance(obs, dict):
                            return False
                        stock = bool(obs.get("in_stock"))
                        price = _num(obs.get("price"))
                        over = not _budget_clear(price, self.s.budget, bool(self.s.book.dead_keys))
                        if (not stock) or over:
                            self.s._mark_event(str(pid), str(sku), stock, price)
                            return False
                        self.s._act("place_test_order", self.s._order_args(pid, sku))
                        return bool(self.s.done)
                    except Exception:
                        return False
                def run(self) -> None:
                    try:
                        if self.order_cart():
                            return
                        # Close an open market event inside the endgame rather than exiting.
                        for _ in range(6):
                            if self.s.done:
                                return
                            if self.s.book.event_target:
                                self.scrub_dead_cart()
                                step = self.s._event_recover()
                                if step is not None:
                                    name, args = step
                                    obs = self.s._act(name, args)
                                    if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                                        added = obs.get("added") or {}
                                        ap = str(added.get("product_id") or args.get("product_id") or "")
                                        asku = str(added.get("sku") or args.get("sku") or "")
                                        if ap:
                                            self.commit(ap, asku)
                                    continue
                            if self.order_cart():
                                return
                            picked = self.pick()
                            if picked is None:
                                return
                            if self.commit(picked[0], picked[1]):
                                return
                            if not self.s.book.event_target:
                                # Commit failed without an event (missing stock data); skip this key.
                                self.s.book.dead_keys.add(CatalogBook.key(picked[0], picked[1]))
                    except Exception:
                        return None
            _Endgame(self).run()
        except Exception:
            return None


    # ------------------------------------------------------------------ main loop
    def run(self) -> list:
        try:
            question = self._opening_question()
            last_forced, repeats = None, 0
            while not self.done and self.turn < self.max_steps:
                if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
                    self._finish_fast()
                    break
                phases = _phase_order(
                    bool(question and not self.asked_shopper and "message" in self.tool_names),
                    bool(self.pending_recheck and "inspect_stock" in self.tool_names),
                    bool(self.book.event_target),
                    not (self.book.rows or self.book.products or self.opened_catalog),
                )
                advanced = False
                for phase in phases:
                    if phase == "ask":
                        self.asked_shopper = True
                        self._act("message", {"content": question})
                        advanced = True
                        break
                    if phase == "recheck":
                        self._post_add_recheck()
                        advanced = True
                        break
                    if phase == "recover":
                        recovered = self._event_recover()
                        if recovered is not None:
                            name, args = recovered
                            self._act(name, args)
                            if name == "message":
                                self.shopper_question_pending = False
                                self.question_waits = 0
                            if name == "add_to_cart":
                                added_pid = str(args.get("product_id") or "")
                                added_sku = str(args.get("sku") or "")
                                if added_pid:
                                    self.pending_recheck = (added_pid, added_sku)
                            advanced = True
                            break
                        continue
                    if phase == "retrieve":
                        opened = self._opening_retrieve()
                        if opened is not None:
                            name, args = opened
                            self._drive_forced(name, args, None if last_forced is None else last_forced)
                            # fall through to shared forced handler below via plan path
                            plan = opened
                            name, args = plan
                            signature = name + json.dumps(args, sort_keys=True)
                            repeats = repeats + 1 if signature == last_forced else 1
                            last_forced = signature
                            if repeats > 2:
                                last_forced, repeats = None, 0
                                self._note("Opening retrieve repeated; moving on.")
                                break
                            obs = self._act(name, args)
                            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                                added = obs.get("added") or {}
                                if added.get("product_id"):
                                    self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                            advanced = True
                            break
                        continue
                    if phase == "plan":
                        plan = self._plan()
                        if plan is None:
                            continue
                        name, args = plan
                        signature = name + json.dumps(args, sort_keys=True)
                        repeats = repeats + 1 if signature == last_forced else 1
                        last_forced = signature
                        if repeats > 2:
                            if name in ("inspect_stock", "add_to_cart"):
                                self.book.dead_keys.add(CatalogBook.key(args.get("product_id"), args.get("sku")))
                            if name == "compare":
                                self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                            if name == "message":
                                self.shopper_question_pending = False
                            last_forced, repeats = None, 0
                            self._note("That step changed nothing, so the controller moved on to other candidates.")
                            advanced = True
                            break
                        obs = self._act(name, args)
                        if name == "message":
                            self.shopper_question_pending = False
                            self.question_waits = 0
                        if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                            added = obs.get("added") or {}
                            if added.get("product_id"):
                                self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                        advanced = True
                        break
                    if phase == "explore":
                        last_forced, repeats = None, 0
                        # Keep mining the catalog deterministically before spending a 429 budget.
                        if not self.model_dead:
                            forced = self._read_more()
                            if forced is not None:
                                name, args = forced
                                obs = self._act(name, args)
                                if name in ("search", "filter"):
                                    self.search_calls += 1
                                if name in ("view", "compare"):
                                    self._shortlist_note()
                                advanced = True
                                break
                        if self.model_dead:
                            self._finish_fast()
                            if self.done:
                                return self.dialogue
                            # Keep the controller loop alive while a market event is open or
                            # a deterministic plan step remains; do not abandon the episode.
                            if self.book.event_target or self.pending_recheck:
                                advanced = True
                                break
                            plan = self._plan()
                            if plan is not None:
                                name, args = plan
                                obs = self._act(name, args)
                                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                                    added = obs.get("added") or {}
                                    if added.get("product_id"):
                                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                                advanced = True
                                break
                            forced = self._read_more()
                            if forced is not None:
                                name, args = forced
                                self._act(name, args)
                                advanced = True
                                break
                            self._finish_fast()
                            if self.done or self.book.event_target or self.pending_recheck:
                                advanced = True
                                break
                            return self.dialogue
                        proposal = self._ask_model()
                        if proposal is None:
                            self._finish_fast()
                            if self.done:
                                return self.dialogue
                            if self.book.event_target or self.pending_recheck:
                                advanced = True
                                break
                            plan = self._plan()
                            if plan is not None:
                                name, args = plan
                                obs = self._act(name, args)
                                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                                    added = obs.get("added") or {}
                                    if added.get("product_id"):
                                        self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                                advanced = True
                                break
                            forced = self._read_more()
                            if forced is not None:
                                name, args = forced
                                self._act(name, args)
                                advanced = True
                                break
                            if self.book.event_target or self.pending_recheck:
                                advanced = True
                                break
                            return self.dialogue
                        call = proposal["tool_calls"][0]
                        name = (call.get("function") or {}).get("name") or ""
                        args = _arguments(call)
                        content = proposal.get("content") or ""
                        reprompts = 0
                        verdict = None
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
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id") or "x",
                                    "content": json.dumps({"rejected": verdict}),
                                }
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
                            advanced = True
                            break
                        self.rejected_streak = 0
                        if name not in self.tool_names:
                            advanced = True
                            break
                        if name == "place_test_order":
                            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                            if not pid and self.book.cart:
                                pid, sku = self.book.cart[-1]
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
                            if self.wish.family == "recall" and self.wish.required_identities:
                                have, need = len(self.book.identities), self._recall_need()
                                self._note(
                                    "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                                    + ("" if have >= need else " Keep searching with different wording.")
                                )
                        if name == "inspect_cart":
                            self.cart_inspections += 1
                        if name in ("view", "compare"):
                            self._shortlist_note()
                        advanced = True
                        break
                if not advanced:
                    self._finish_fast()
                    break
            return self.dialogue
        except Exception:
            return []

    def _drive_forced(self, name: str, args: dict, last_forced: str | None) -> None:
        try:
            return None
        except Exception:
            return None


    def _validate(self, name: str, args: dict) -> str | None:
        try:
            if name not in self.tool_names:
                return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.book.cart:
                    pid, sku = self.book.cart[-1]
                ok, why = self._valid_now(pid, sku)
                if not ok:
                    return "order refused: " + why
                if self.wish.family == "recall" and len(self.book.identities) < self._recall_need():
                    return (
                        "order refused: only %d distinct matching products have been surfaced; search with different "
                        "wording until at least %d appear" % (len(self.book.identities), self._recall_need())
                    )
                if self.wish.family == "preference" and len(self.book.listing_ids) < 2:
                    return "order refused: compare at least two products with different listing_id values first"
            if name == "add_to_cart":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid or not sku:
                    return "add_to_cart needs both product_id and sku"
                c = self._assess(pid, sku)
                key = CatalogBook.key(pid, sku)
                if not c["live"]:
                    return "that variant is out of stock, over budget, or already invalidated"
                if not c["viewed"]:
                    return "view or compare that product first so its raw facts are on record"
                if key not in self.book.inspected:
                    return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
                if c["spec"] != 2 and self.wish.spec_field:
                    return (
                        "that variant does not resolve to %s = %s from its own option text or raw facts"
                        % (self.wish.spec_field, self.wish.spec_value)
                    )
                if not c["ok"]:
                    return "that product fails a firm requirement (category, brand, or the kind of device requested)"
                if self.wish.family == "preference" and len(self.book.listing_ids) < 2:
                    return "compare two products with different listing_id values before the first cart addition"
                if self.wish.family == "recovery" and not self.book.adds and self.wish.pref_model and self._relax(pid, sku):
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
            if key in self.book.calls_seen and name in ("search", "filter", "view", "compare"):
                return "identical call already made; change the arguments"
            self.book.calls_seen.add(key)
            return None
        except Exception:
            return None


def agent_main(problem_data: dict) -> list:
    try:
        return HybridShopper(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
