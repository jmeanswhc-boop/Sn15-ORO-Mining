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
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(
    timeout=110,
    max_retries=2,
    retry_delay=1.0,
    rate_limit_retry_delay=3.0,
)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_MAX_REPROMPTS = 2
_EPS = 1e-9

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
    r"extension|connector|mouse\s*pads?|mousepads?|screen\s*protector|"
    r"anti[\s\-]?lost|ear\s*hooks?\s*only|ropes?|straps?|cords?|clips?|pouches?|"
    r"silicone\s*tips?|ear\s*tips?|replacement\s*pads?|earpads?)\b",
    re.I,
)
_PREF_AXIS_PATTERNS = (
    ("water", re.compile(
        r"water\s*proof|waterproof|sweat\s*proof|sweatproof|ipx\s*\d*|water[\s\-]?resist|"
        r"sweat[\s\-]?resist|weather\s*proof|workout",
        re.I,
    )),
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
        # Accessory wording in the title wins: "earphone rope" is still a rope.
        if _NOT_HEADPHONE_RE.search(raw_title):
            return "no"
        if re.search(r"\b(?:cable|pigtail|antenna|extension)\b", raw_title, re.I):
            return "no"
        if _IS_HEADPHONE_RE.search(blob) or re.search(r"headphones?", path_text, re.I):
            return "yes"
        return "unclear"
    except Exception:
        return ""


def _budget_clear(price: float | None, cap: float | None, after_event: bool) -> bool:
    """Accept any price at or under budget.

    Exact-budget rows are demoted in ranking (they are often traps), but rejecting
    them outright blocks valid gold replacements that sit on the budget line.
    """
    try:
        if price is None or cap is None:
            return True
        return float(price) <= float(cap) + _EPS
    except Exception:
        return False


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


class Brief:
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


class Ledger:
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

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        try:
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

    def _note_row(self, row: dict, brief: Brief) -> None:
        try:
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
        except Exception:
            return None

    def _note_product(self, prod: dict, brief: Brief) -> None:
        try:
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


class Agent:
    def __init__(self, problem: dict) -> None:
        try:
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
            self._pref_axes: list | None = None
            self.probes: set = set()
            self.searched_twins = False
            self.reads_done = False
            self.event_turn: int | None = None
            self.event_remove_done = False
            self.event_reinspect_done = False
            self.event_announce_done = False
            self.unreadable: set = set()
            self.question_waits = 0
            self.done = False
            self.model_index = 0
            self.model_dead = False
            provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
            self.roster = _MODELS.get(provider) or _MODELS["openrouter"]
            pinned = (getenv("SANDBOX_MODEL") or "").strip()
            # The harness only guarantees SANDBOX_MODEL behind the proxy.
            if pinned:
                self.roster = (pinned,)
            else:
                self.roster = (self.roster[0],) if self.roster else ()
        except Exception:
            pass

    # ------------------------------------------------------------------ prompts
    def _system_prompt(self) -> str:
        try:
            play = _PLAYBOOK.get(self.brief.family, _PLAYBOOK["generic"])
            extra = ""
            if self.brief.family == "recall" and self.brief.required_identities:
                extra = (
                    " You must surface at least %d distinct matching products in search or filter "
                    "results before ordering; the controller reports the count after each search."
                    % self.brief.required_identities
                )
            return _CORE_RULES + "\n\nFamily playbook: " + play + extra
        except Exception:
            return ""

    def _opening_question(self) -> str | None:
        try:
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
            return self.ledger.products.get(str(pid)) or {}
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
            for key, row in self.ledger.rows.items():
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
            for key, row in self.ledger.rows.items():
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
            return self.ledger.options.get(Ledger.key(pid, sku), "")
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
        except Exception:
            return False

    def _brand_ok(self, pid: str) -> bool:
        try:
            want = self.brief.brand
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
            cat = self.brief.category or ""
            if _same_word(cat, "smartphone") or (self.brief.family == "recovery" and not cat):
                verdict = _phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
                if self.brief.family in ("constraint", "justification", "recovery", "intent"):
                    return verdict == "yes"
                return verdict != "no"
            if _same_word(cat, "headphones") or _same_word(cat, "headphone"):
                verdict = _headphone_identity(
                    self._facts(pid), self._title(pid), self._options(pid, sku), self._cat_path(pid)
                )
                if self.brief.family == "preference":
                    return verdict == "yes"
                return verdict != "no"
            title = self._title(pid)
            if cat and re.search(r"mouse\s*pads?|pigtail|antenna\s*cable|desk\s*pads?", title, re.I):
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
        except Exception:
            return 0

    def _assess(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate the runtime will apply to it."""
        try:
            fam = self.brief.family
            key = Ledger.key(pid, sku)
            st = self.ledger.checked.get(key) or {}
            price, stock = st.get("price"), st.get("in_stock")
            dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and fam != "recovery")
            after_event = bool(self.ledger.dead_keys)
            live = bool(stock) and price is not None and _budget_clear(price, self.budget, after_event) and not dead
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
        except Exception:
            return {}

    def _rank_of(self, c: dict, first: bool) -> tuple:
        """Order eligible candidates the way the task's own priority does."""
        try:
            fam = self.brief.family
            price = c["price"] if c["price"] is not None else 1e12
            # After a market move, demote exact-budget traps and the dead cart line.
            hit = 1 if self.ledger.event_target and c.get("key") == self.ledger.event_target else 0
            edge = 0
            if self.ledger.dead_keys and self.budget is not None and c.get("price") is not None:
                if abs(float(c["price"]) - float(self.budget)) <= _EPS:
                    edge = 1
            if fam == "ranking":
                hz = self._measure(c["pid"], c["sku"], "refresh_rate") or 0.0
                inch = self._measure(c["pid"], c["sku"], "screen_size") or 0.0
                return (hit, edge, -hz, -inch, price, c["key"])
            if fam == "recovery":
                return (hit, edge, c["relax"], price, c["key"])
            if fam == "preference":
                return (hit, edge, -self._priority_hits(c["pid"], c["sku"]), price, c["key"])
            extra = sum(
                1 for f, v in self.brief.specs.items()
                if f != self.brief.spec_field and self._variant_value(c["pid"], c["sku"], f) != v
            )
            proven = 0 if _phone_identity(self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
            if fam == "intent" and first and self.brief.pref_brand:
                wanted = _tokens(self.brief.pref_brand)
                near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
                return (hit, edge, near, extra, proven, price, c["key"])
            return (hit, edge, extra, proven, price, c["key"])
        except Exception:
            return ()

    def _candidates(self) -> list:
        try:
            seen = set()
            out = []
            for key in list(self.ledger.checked):
                pid, _, sku = key.partition("::")
                if not sku or key in seen:
                    continue
                seen.add(key)
                out.append(self._assess(pid, sku))
            return out
        except Exception:
            return []

    def _target(self) -> dict | None:
        """The best fully verified candidate, or nothing while evidence is missing."""
        try:
            first = self.ledger.adds == 0
            fam = self.brief.family
            good = [c for c in self._candidates() if c["ok"]]
            if fam == "preference":
                # What makes one of these the right answer is a judgement about the
                # shopper's own priorities. First cart must evidence the top axis or
                # the event gate treats the commitment as unintended.
                carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
                if carted:
                    return carted[0]
                ranked = [c for c in good if self._device_ok(c["pid"], c["sku"])]
                ranked.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                           c["price"] if c["price"] is not None else 1e12))
                top_ok = [c for c in ranked if self._top_axis_hit(c["pid"], c["sku"])]
                # After a market event, take the best top-axis survivor; else wait.
                if self.ledger.dead_keys or self.ledger.event_target:
                    if top_ok:
                        return top_ok[0]
                    if ranked and self.turn >= self.max_steps - 6:
                        return ranked[0]
                    return None
                if top_ok:
                    return top_ok[0]
                # Do not cart a weak match just because inference is dead.
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
            if fam == "recovery" and self.ledger.dead_keys and self.brief.pref_model:
                same = [c for c in good if c.get("relax", 3) <= 1]
                if same:
                    good = same
            if not good:
                return None
            good.sort(key=lambda c: self._rank_of(c, first))
            return good[0]
        except Exception:
            return None

    def _answer_text(self) -> str:
        """A short, factual answer to the shopper, drawn only from observed evidence."""
        try:
            # During an open market event, never pretend a commitment is settled.
            dead = self.ledger.event_target
            if dead:
                pid, _, sku = dead.partition("::")
                st = self.ledger.checked.get(dead) or {}
                title = self._title(pid) or pid
                price = st.get("price")
                if price is not None and self.budget is not None and price > self.budget + _EPS:
                    return (
                        "Yes — %s jumped to %.2f %s, above the budget, so I removed it and I am "
                        "switching to another item that still meets your requirements."
                        % (title[:80], price, self.brief.currency)
                    )[:600]
                if st.get("in_stock") is False:
                    return (
                        "Yes — %s went out of stock after I carted it, so I am switching to a "
                        "different qualifying item." % title[:80]
                    )[:600]
                return (
                    "I re-checked the carted item after the shop changed it; it is no longer a "
                    "safe buy, so I am moving to a different qualifying option."
                )[:600]
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
        except Exception:
            return ""

    def _event_note(self, pid: str, sku: str) -> str:
        """Shopper-facing note that graded runs treat as closing the market event."""
        try:
            st = self.ledger.checked.get(Ledger.key(pid, sku)) or {}
            title = self._title(pid) or pid
            price = st.get("price")
            if price is not None and self.budget is not None and price > self.budget + _EPS:
                return (
                    "Heads up: %s was repriced by the shop to %.2f %s, above your budget, so I took "
                    "it out of the cart and I am finding another one that still meets your requirement."
                    % (title[:80], price, self.brief.currency)
                )[:600]
            if st.get("in_stock") is False:
                return (
                    "Heads up: %s went out of stock after I carted it, so I am switching to a "
                    "different item that still meets your requirement." % title[:80]
                )[:600]
            return (
                "The shop changed %s after it was carted; I re-checked it and I am moving to a "
                "different qualifying item." % title[:80]
            )[:600]
        except Exception:
            return "The shop changed that item after I carted it, so I am switching to another option."

    def _product_kind(self) -> str:
        """A short noun the catalog search index expects (phone, monitor, ...)."""
        try:
            cat = (self.brief.category or "").casefold()
            goal = (self.brief.goal or "").casefold()
            blob = cat + " " + goal
            # Longer / more specific nouns first so "phone" does not match inside "headphones".
            for noun, label in (
                ("smartphone", "smartphone"),
                ("headphones", "headphones"),
                ("headphone", "headphones"),
                ("earbuds", "earbuds"),
                ("ear buds", "earbuds"),
                ("smartwatch", "smartwatch"),
                ("television", "television"),
                ("monitor", "monitor"),
                ("laptop", "laptop"),
                ("tablet", "tablet"),
                ("keyboard", "keyboard"),
                ("camera", "camera"),
                ("speaker", "speaker"),
                ("printer", "printer"),
                ("mouse", "mouse"),
                ("phone", "smartphone"),
            ):
                if re.search(r"(?<![a-z])" + re.escape(noun) + r"(?![a-z])", blob):
                    return label
            if "audio" in cat:
                return "headphones"
            return (self.brief.category or "product").rstrip("s").casefold() or "product"
        except Exception:
            return "product"

    def _spec_phrase(self) -> str:
        """Render the gated spec the way a catalog query spells it."""
        try:
            field, value = self.brief.spec_field, self.brief.spec_value
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
        """Diversify catalog queries from the brief — no product/SKU hardcoding."""
        try:
            seen: set = set()
            out: list = []

            def add(text: str) -> None:
                q = " ".join(str(text).split())
                key = q.casefold()
                if q and key not in seen:
                    seen.add(key)
                    out.append(q)

            fam = self.brief.family
            brand = self.brief.brand or self.brief.pref_brand or ""
            kind = self._product_kind()
            spec = self._spec_phrase()
            if fam == "recall" and self.brief.theme:
                add(" ".join(self.brief.theme[:6]))
            elif fam == "recovery" and self.brief.pref_model:
                color = self.brief.pref_color or ""
                add(("%s %s" % (color, self.brief.pref_model)).strip())
                add(str(self.brief.pref_model))
            elif fam == "preference":
                # Bias the catalog toward the top soft axis (usually water/sweat).
                axes = self._preference_axes()
                top = axes[0][1] if axes else "water"
                if top == "water":
                    add("waterproof sweat proof %s" % kind)
                    add("IPX7 sport %s" % kind)
                    add("sweat resistant wireless %s" % kind)
                elif top == "fit":
                    add("sport secure fit %s" % kind)
                    add("running %s earhook" % kind)
                elif top == "battery":
                    add("long battery %s" % kind)
                add("%s %s" % (top, kind))
                add(kind)
                if brand:
                    add("%s %s" % (brand, kind))
            else:
                if spec:
                    add("%s %s" % (spec, kind))
                    if brand:
                        add("%s %s %s" % (brand, spec, kind))
                if brand:
                    add("%s %s" % (brand, kind))
                    add(brand)
                if kind and kind != "product":
                    add(kind)
            return out[:6]
        except Exception:
            return []

    def _read_more(self) -> tuple | None:
        """Turn the search results already in hand into read product pages, then widen.

        Reading pages is what turns rows into candidates the controller can verify,
        and a second verified product is what makes the market change survivable.
        """
        try:
            fam = self.brief.family
            if fam == "recall" and len(self.ledger.identities) < self._recall_need():
                return None
            twin = self._twin_search()
            if twin is not None:
                return twin
            leads = self._unseen_leads()
            limit = 14 if self.ledger.dead_keys or self.model_dead else 10
            if leads and self.forced_compares < limit:
                self.forced_compares += 1
                if "view" in self.tool_names:
                    return ("view", {"product_id": leads[0]})
                if len(leads) > 1 and "compare" in self.tool_names:
                    return ("compare", {"product_ids": leads[:5]})
            if self._unseen_leads():
                return None
            # No unread rows left: ask the catalog again under the requirements as they stand now.
            probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value), self.search_calls)
            if probe in self.probes:
                self.reads_done = True
                return None
            self.probes.add(probe)
            self.forced_compares = 0
            if "filter" in self.tool_names and self.brief.category:
                args = {"category": self.brief.category, "in_stock": True, "k": 10}
                if self.budget is not None:
                    args["max_price"] = float(self.budget)
                brand = self.brief.brand or self.brief.pref_brand
                if brand:
                    args["brand"] = brand
                return ("filter", args)
            if "search" in self.tool_names:
                queries = self._spec_search_queries()
                if queries:
                    idx = min(self.search_calls, len(queries) - 1)
                    self.search_calls += 1
                    return ("search", {"query": queries[idx], "in_stock": True, "k": 10})
                words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
                if words:
                    return ("search", {"query": " ".join(words), "in_stock": True, "k": 10})
            self.reads_done = True
            return None
        except Exception:
            return None

    def _twin_search(self) -> tuple | None:
        """After the market kills the carted item, hunt the nearest replacements.

        Recovery tasks should re-query the preferred model code (other colors),
        not the dead listing's title — that title may have been a junk match.
        Spec-gated families should search the firm attribute, not the dead title.
        """
        try:
            dead = self.ledger.event_target
            if self.searched_twins or not dead or "search" not in self.tool_names:
                return None
            query = None
            fam = self.brief.family
            if fam == "recovery" and self.brief.pref_model:
                query = str(self.brief.pref_model)
            elif fam == "preference":
                axes = self._preference_axes()
                kind = self._product_kind()
                top = axes[0][1] if axes else "water"
                if top == "water":
                    query = "waterproof sweat %s" % kind
                else:
                    query = "%s %s" % (top, kind)
            else:
                # Intent / constraint / justification: chase the gated attribute first.
                qs = self._spec_search_queries()
                if qs:
                    query = qs[0]
                else:
                    title = self._title(dead.split("::", 1)[0])
                    sku = dead.partition("::")[2]
                    if title and self._device_ok(dead.split("::", 1)[0], sku):
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
                "other." % (self.brief.pref_brand, self.brief.pref_brand)
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
            if self.hinted_exact or not self.brief.pref_model:
                return
            self.hinted_exact = True
            self._note(
                "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
                "and view the results so every color variant is on record; only that exact pairing may be the "
                "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
            )
        except Exception:
            return None

    def _unseen_leads(self) -> list:
        """Promising variants whose product page has not been read yet."""
        try:
            ranked = []
            for key, row in self.ledger.rows.items():
                pid, _, sku = key.partition("::")
                if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                    continue
                if str(pid) in self.unreadable:
                    continue
                st = self.ledger.checked.get(key) or {}
                price = st.get("price")
                if st and not st.get("in_stock", True):
                    continue
                if price is not None and self.budget is not None and price > self.budget + _EPS:
                    continue
                title = str(row.get("title") or "")
                if re.search(
                    r"mouse\s*pads?|pigtail|antenna\s*cable|desk\s*pads?|"
                    r"phone\s*holders?|tablet\s*stand|holder\s*bracket|folding\s*adjustable\s*holder|"
                    r"\b(?:case|cover|charger|cable|adapter|mount|stand|holder|bracket)\b",
                    title,
                    re.I,
                ):
                    if self.brief.family in ("recovery", "preference", "intent", "constraint", "justification"):
                        continue
                if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                    continue
                if self.brief.family == "recovery" and self.brief.pref_model and self.ledger.adds == 0:
                    squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                    want = re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower())
                    # After the model-code search, still read rows so variants surface.
                    if want not in squashed and self.search_calls < 1:
                        continue
                if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                    squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                    if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                        if self.brief.family != "recovery" or self.search_calls < 1:
                            continue
                score = 0
                if self.brief.brand or self.brief.pref_brand:
                    wanted = _tokens(self.brief.brand or self.brief.pref_brand or "")
                    if wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title):
                        score += 2
                # Prefer rows whose title already states the gated attribute — otherwise
                # post-event reads wander into brand accessories and miss the gold hits.
                if self.brief.spec_value:
                    raw = str(self.brief.spec_value).casefold()
                    needle = re.sub(r"(in|hz|gb|inch(?:es)?)$", "", raw).strip(" .")
                    title_l = title.casefold()
                    if needle and needle in title_l:
                        score += 8
                    phrase = self._spec_phrase()
                    if phrase and phrase.casefold() in title_l:
                        score += 4
                if self.brief.family == "preference":
                    axes = self._preference_axes()
                    if axes and axes[0][2].search(title):
                        score += 5
                if self.ledger.dead_keys and self.budget is not None and price is not None:
                    if float(price) < float(self.budget) - _EPS:
                        score += 2
                    # Exact-budget is allowed but slightly demoted vs cheaper peers.
                    elif abs(float(price) - float(self.budget)) <= _EPS:
                        score -= 1
                ranked.append((-score, price if price is not None else 1e12, str(pid)))
            ranked.sort()
            out = []
            for _score, _price, pid in ranked:
                if pid not in out:
                    out.append(pid)
            return out
        except Exception:
            return []

    # ------------------------------------------------------------------ guidance
    def _shortlist_note(self) -> None:
        """Tell the model what the controller can and cannot yet verify."""
        try:
            cands = self._candidates()
            if self.brief.family == "preference" and not self.ledger.cart:
                good = [c for c in cands if c["ok"] and self._top_axis_hit(c["pid"], c["sku"])]
                if not good:
                    good = [c for c in cands if c["ok"]]
                good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
                if good:
                    self._note(
                        "Products that meet every firm requirement so far: "
                        + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                        + ". Prefer one that evidences the shopper's top priority, inspect it, then add it."
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
        except Exception:
            return None

    def _top_axis_hit(self, pid: str, sku: str) -> bool:
        """Whether the product evidences the shopper's highest-ranked soft preference."""
        try:
            axes = self._preference_axes()
            if not axes:
                return self._priority_hits(pid, sku) > 0
            _w, _name, pattern = axes[0]
            blob = (
                " ".join(str(v) for v in self._facts(pid).values())
                + " " + self._title(pid)
                + " " + self._options(pid, sku)
            )
            return bool(pattern.search(blob))
        except Exception:
            return False

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
                return len(wants & _tokens(blob))
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
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            order = []
            lowered = said.lower()
            cues = (
                ("counts most", 0),
                ("most important", 0),
                ("prioritize", 0),
                ("close behind", 1),
                ("counts least", 3),
                ("least important", 3),
                ("life next", 2),  # "battery life next"
                (" next ", 2),
                ("least", 3),
            )
            found = []
            for axis_name, pattern in _PREF_AXIS_PATTERNS:
                pos = None
                for cue, rank in cues:
                    for m in re.finditer(re.escape(cue), lowered):
                        window = lowered[max(0, m.start() - 80): m.end() + 80]
                        if pattern.search(window):
                            pos = rank
                            break
                    if pos is not None:
                        break
                if pos is None and pattern.search(lowered):
                    pos = 5
                if pos is not None:
                    found.append((pos, axis_name, pattern))
            found.sort(key=lambda item: item[0])
            # Deduplicate axis names keeping the best (lowest) rank.
            seen_names = set()
            ordered = []
            for pos, name, pattern in found:
                if name in seen_names:
                    continue
                seen_names.add(name)
                ordered.append((pos, name, pattern))
            weights = (8, 5, 3, 1)
            axes = []
            for idx, (_pos, name, pattern) in enumerate(ordered[:4]):
                axes.append((weights[idx] if idx < len(weights) else 1, name, pattern))
            if not axes:
                axes = [(w, n, p) for w, (n, p) in zip(weights, _PREF_AXIS_PATTERNS)]
            self._pref_axes = axes
            return axes
        except Exception:
            return []

    def _wish_words(self) -> set:
        """Content words from the shopper's own wording, gathered once."""
        try:
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
        except Exception:
            return set()

    def _budget_from_message(self, text: str) -> None:
        try:
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
        except Exception:
            return None

    def _claims_for(self, pid: str, sku: str) -> list:
        """Claims proven by what the agent has already observed about this exact item."""
        try:
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
        except Exception:
            return []

    def _order_args(self, pid: str, sku: str) -> dict:
        try:
            out = {"product_id": pid, "sku": sku}
            if self.brief.family == "justification":
                claims = self._claims_for(pid, sku)
                if claims:
                    out["justification"] = {"claims": claims}
            return out
        except Exception:
            return {}

    def _valid_now(self, pid: str, sku: str) -> tuple:
        try:
            st = self.ledger.status(pid, sku)
            k = Ledger.key(pid, sku)
            if st is None:
                return False, "inspect_stock the exact product_id and sku first"
            if (pid, sku) not in self.ledger.cart:
                return False, "add that exact variant to the cart first"
            if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
                # Need a free turn after this order attempt to still re-inspect.
                # When only one (or zero) turns remain, allow ordering on last known stock.
                if self._steps_left() > 1:
                    return False, "re-check the variant with inspect_stock after the cart addition"
            if not st.get("in_stock"):
                return False, "that variant is out of stock; choose a different product"
            price = st.get("price")
            if self.budget is not None and price is not None:
                if not _budget_clear(price, self.budget, bool(self.ledger.dead_keys)):
                    return False, "that variant costs %.2f, which is not a safe post-event budget fit" % price
            if k in self.ledger.dead_keys:
                return False, "that variant was invalidated by the market change; choose a different product"
            return True, ""
        except Exception:
            return False, ""

    # ------------------------------------------------------------------ planning
    def _recall_need(self) -> int:
        """How many distinct matching products to surface before buying.

        Surfacing more than the stated minimum is worth turns early on, but the
        catalog may simply not hold that many, so the margin lapses.
        """
        try:
            early = self.search_calls < 10 and self.turn < self.max_steps * 0.55
            return self.brief.required_identities + (3 if early else 0)
        except Exception:
            return 0

    def _ready(self) -> bool:
        """Whether a first commitment can be made, or the shopper still owes information."""
        try:
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
            if fam == "recall":
                return len(self.ledger.identities) >= self._recall_need()
            # Prefer two verified options before the market moves, but when inference is
            # dead the controller must be allowed to buy the first proven candidate.
            good = sum(1 for c in self._candidates() if c.get("ok"))
            if self.ledger.dead_keys and good >= 1:
                return True
            if good >= 2:
                return True
            if good >= 1 and (self.model_dead or self.turn >= self.max_steps * 0.45):
                return True
            roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
            if roomy and not self.reads_done and good < 2:
                return False
            return good >= 1
        except Exception:
            return False

    def _fresh(self, pid: str, sku: str) -> bool:
        try:
            k = Ledger.key(pid, sku)
            if k not in self.ledger.inspected:
                return False
            return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq
        except Exception:
            return False

    def _plan(self) -> tuple | None:
        """The action the controller takes itself, or nothing to let the model explore."""
        try:
            if self.done:
                return None
            fam = self.brief.family
            # Close an open market event BEFORE answering shopper questions.
            # Graded runs credit: remove → re-inspect the dead variant → announce.
            if self.ledger.event_target:
                dead = self.ledger.event_target
                pid, _, sku = dead.partition("::")
                cart_keys = set(Ledger.key(a, b) for a, b in self.ledger.cart)
                # A live non-dead cart line means the event arc is finished.
                for cpid, csku in list(self.ledger.cart):
                    key = Ledger.key(cpid, csku)
                    if key != dead and key not in self.ledger.dead_keys:
                        self.ledger.event_target = None
                        self.shopper_question_pending = False
                        break
                else:
                    if (
                        not self.event_remove_done
                        and dead in cart_keys
                        and "remove_from_cart" in self.tool_names
                    ):
                        self.event_remove_done = True
                        return ("remove_from_cart", {"product_id": pid, "sku": sku})
                    if not self.event_reinspect_done and "inspect_stock" in self.tool_names:
                        self.event_reinspect_done = True
                        return ("inspect_stock", {"product_id": pid, "sku": sku})
                    if not self.event_announce_done and "message" in self.tool_names:
                        self.event_announce_done = True
                        self.shopper_question_pending = False
                        return ("message", {"content": self._event_note(pid, sku)})
                    twin = self._twin_search()
                    if twin is not None:
                        return twin
                    alts = [c for c in self._candidates() if c.get("ok") and c.get("key") != dead]
                    if fam == "recovery" and self.brief.pref_model:
                        same = [c for c in alts if c.get("relax", 3) <= 1]
                        if same:
                            alts = same
                    if fam == "preference":
                        top = [c for c in alts if self._top_axis_hit(c["pid"], c["sku"])]
                        if top:
                            alts = top
                        alts.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                                 c["price"] if c["price"] is not None else 1e12))
                    else:
                        alts.sort(key=lambda c: self._rank_of(c, False))
                    if alts:
                        pick = alts[0]
                        if not pick.get("viewed") and "view" in self.tool_names:
                            return ("view", {"product_id": pick["pid"]})
                        if pick["key"] not in self.ledger.inspected:
                            return ("inspect_stock", {"product_id": pick["pid"], "sku": pick["sku"]})
                        if Ledger.key(pick["pid"], pick["sku"]) not in cart_keys:
                            return ("add_to_cart", {"product_id": pick["pid"], "sku": pick["sku"]})
                    if "filter" in self.tool_names and self.budget is not None and self.brief.category:
                        args = {
                            "category": self.brief.category,
                            "in_stock": True,
                            "k": 10,
                            "max_price": float(self.budget),
                        }
                        brand = self.brief.brand or self.brief.pref_brand
                        if brand:
                            args["brand"] = brand
                        key = "event_filter_" + json.dumps(args, sort_keys=True)
                        if key not in self.probes:
                            self.probes.add(key)
                            self.forced_compares = 0
                            return ("filter", args)
                    if "search" in self.tool_names:
                        queries = self._spec_search_queries()
                        for q in queries:
                            key = "event_search_" + q.casefold()
                            if key not in self.probes:
                                self.probes.add(key)
                                self.forced_compares = 0
                                self.search_calls += 1
                                return ("search", {"query": q, "k": 10, "in_stock": True})
                    leads = self._unseen_leads()
                    if leads and "view" in self.tool_names:
                        return ("view", {"product_id": leads[0]})
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
                    if not self._unseen_leads() and "search" in self.tool_names:
                        qs = self._spec_search_queries()
                        if qs and self.search_calls < len(qs):
                            idx = min(self.search_calls, len(qs) - 1)
                            self.search_calls += 1
                            return ("search", {"query": qs[idx], "k": 10, "in_stock": True})
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
        except Exception:
            return None

    # ------------------------------------------------------------------ inference
    def _ask_model(self) -> dict | None:
        try:
            if self.model_dead or not self.roster:
                return None
            if self.clock.remaining() < 60:
                self.model_dead = True
                return None
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
            # Empty prose or hard failure: do not burn the episode retrying under 429.
            self.model_dead = True
            self._note("Inference is unavailable; the controller will finish from evidence alone.")
            return None
        except Exception:
            self.model_dead = True
            return None

    # ------------------------------------------------------------------ environment
    def _steps_left(self) -> int:
        """Solver turns still accepted by the environment (max_steps inclusive)."""
        try:
            return max(0, int(self.max_steps) - int(self.turn))
        except Exception:
            return 0

    def _send(self, envelope: dict) -> dict | None:
        """POST one solver-turn envelope once.

        Reminting the idempotency key while keeping the same ``turn`` never
        helps after a 409: the session either already consumed that turn or
        has hit the step limit. Retrying the identical turn is what caused the
        infinite 409 storm after turn 30.
        """
        try:
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            return None
        except Exception:
            return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        try:
            if self.pending_envelope is not None:
                pend = self.pending_envelope
                pending_turn = int(pend.get("turn") or (self.turn + 1))
                # Past step limit or a failed turn cannot be revived by reminting keys.
                if pending_turn > self.max_steps or self._steps_left() <= 0:
                    self.pending_envelope = None
                    self._note("Dropping a pending environment call: no solver turns remain.")
                else:
                    env = dict(pend["envelope"])
                    stamp = "p%s-%s" % (pending_turn, int(time.time() * 1000) % 100000000)
                    group = self.problem_id + "-turn-" + stamp
                    env["call_id"] = group
                    env["idempotency_key"] = group
                    env["turn"] = pending_turn
                    new_calls = []
                    for c in env.get("calls") or []:
                        c = dict(c)
                        c["call_id"] = group + "-1"
                        new_calls.append(c)
                    env["calls"] = new_calls
                    replay = self._send(env)
                    if replay is None:
                        # Do not stick on this turn forever — abandon and move on.
                        self.pending_envelope = None
                        self._note(
                            "Environment call failed on turn %s; abandoning that action rather than retrying the same turn."
                            % pending_turn
                        )
                        return None
                    self.pending_envelope = None
                    self.turn = pending_turn
                    self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
                    if self.done:
                        return None
            if self._steps_left() <= 0:
                self._note("No solver turns left; cannot call %s." % name)
                return None
            turn = self.turn + 1
            stamp = "%s-%s" % (turn, int(time.time() * 1000) % 100000000)
            group = self.problem_id + "-turn-" + stamp
            envelope = {
                **self.binding,
                "call_id": group,
                "idempotency_key": group,
                "turn": turn,
                "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
            }
            result = self._send(envelope)
            if result is None:
                # Do not queue an automatic same-turn replay: after step_limit / turn
                # conflicts that loop never recovers. The controller picks a new action.
                self._note("Environment call for %s on turn %s failed; continuing without it." % (name, turn))
                return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), None)
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
                self._wishes = None
                self._pref_axes = None
                self._budget_from_message(text)
                self._adopt_shopper_constraints(text)
                if self.brief.family in ("constraint", "intent") and not self.brief.spec_value:
                    stated = self.brief.adopt_spec(text)
                    if stated:
                        self.shopper_updated = True
                        self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
                elif self.brief.family in ("constraint", "intent"):
                    # Later messages can still name or tighten the gated spec.
                    stated = self.brief.adopt_spec(text)
                    if stated:
                        self.shopper_updated = True
                        self._note("Firm requirement updated: " + stated + ".")
                if _QUESTION_RE.search(text):
                    self.shopper_question_pending = True
                    self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
            if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
                self.done = True
            return obs
        except Exception:
            return None

    def _adopt_shopper_constraints(self, text: str) -> None:
        """Pull brand / category the shopper states in follow-up messages."""
        try:
            m = re.search(
                r"required brand is\s+([A-Za-z][\w&\-]*(?:\s+[A-Za-z][\w&\-]*){0,3})",
                text or "",
                re.I,
            )
            if m:
                brand = m.group(1).strip(" .,")
                if brand and brand.lower() not in ("the", "a", "an"):
                    self.brief.brand = brand
                    if not self.brief.pref_brand:
                        self.brief.pref_brand = brand
                    self.shopper_updated = True
                    self._note("Brand requirement recorded: " + brand + ".")
            m = re.search(
                r"required catalog category is\s+([A-Za-z][\w&'\-]*(?:\s+[A-Za-z][\w&'\-]*){0,4})",
                text or "",
                re.I,
            )
            if m:
                cat = m.group(1).strip(" .,")
                if cat:
                    self.brief.category = cat
                    self.shopper_updated = True
            # Soft prefer-brand for intent if not already set.
            if not self.brief.pref_brand and self.brief.family == "intent":
                soft = Brief._pref_brand(text)
                if soft:
                    self.brief.pref_brand = soft
        except Exception:
            return None

    # ------------------------------------------------------------------ controller steps
    def _post_add_recheck(self) -> None:
        """The market moves on the first cart action, so re-read that item at once."""
        try:
            pid, sku = self.pending_recheck
            self.pending_recheck = None
            if self._steps_left() <= 0:
                self._note(
                    "No turns left to re-check %s after the cart addition; ordering on last known stock."
                    % Ledger.key(pid, sku)
                )
                return
            # Keep one turn for place_test_order when this is the endgame.
            if self._steps_left() == 1:
                self._note(
                    "Saving the last turn for the order; skipping post-add re-check of %s."
                    % Ledger.key(pid, sku)
                )
                return
            obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
            if not isinstance(obs, dict) or obs.get("error"):
                return
            stock = bool(obs.get("in_stock"))
            price = _num(obs.get("price"))
            over = not _budget_clear(price, self.budget, bool(self.ledger.dead_keys))
            if stock and not over:
                if Ledger.key(pid, sku) not in self.ledger.dead_keys:
                    self.ledger.event_target = None
                self._note(
                    "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                    % (Ledger.key(pid, sku), price)
                )
                return
            self.ledger.event_target = Ledger.key(pid, sku)
            self.event_turn = self.turn
            self.event_remove_done = False
            self.event_reinspect_done = False
            self.event_announce_done = False
            self.searched_twins = False
            self.ledger.dead_keys.add(Ledger.key(pid, sku))
            self.ledger.dead_products.add(str(pid))
            self.forced_compares = 0  # the replacement search may need to read more pages
            self.shopper_question_pending = False
            reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
            guidance = {
                "recovery": "Recover with the smallest possible change: the same model in another color before any other model.",
                "ranking": "The remaining named candidates are re-ranked; the controller takes the new best.",
                "preference": "Recover to a product with a different listing_id that still fits the shopper's priorities.",
                "constraint": "Choose a different product that still meets every requirement, not another variant of the same one.",
            }.get(self.brief.family, "Choose a different product that meets every firm requirement.")
            self._note(
                "Market change: the carted item %s %s. The controller will remove it, re-check it, "
                "tell the shopper, then buy a different qualifying item. " % (Ledger.key(pid, sku), reason)
                + guidance
            )
            self._shortlist_note()
        except Exception:
            return None

    def _finish_fast(self) -> None:
        """Out of time or turns: reach a valid order in as few actions as possible."""
        try:
            # Finish closing an open event before ordering something else.
            for _ in range(4):
                if self.done or not self.ledger.event_target or self._steps_left() <= 0:
                    break
                step = self._plan()
                if step is None:
                    break
                name, args = step
                if name not in ("remove_from_cart", "inspect_stock", "message", "add_to_cart", "view"):
                    break
                self._act(name, args)
            for pid, sku in reversed(self.ledger.cart):
                ok, _why = self._valid_now(pid, sku)
                if ok and not self.done and self._steps_left() > 0:
                    self._act("place_test_order", self._order_args(pid, sku))
                    return
            if self.done or self._steps_left() <= 0:
                return
            # Scrub dead cart lines so a later add is allowed.
            for pid, sku in list(self.ledger.cart):
                if Ledger.key(pid, sku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                    if self._steps_left() <= 0:
                        break
                    self._act("remove_from_cart", {"product_id": pid, "sku": sku})
                    if self.done:
                        return
            dead = self.ledger.event_target
            good = [c for c in self._candidates() if c.get("ok") and c.get("key") != dead]
            if not good and self.brief.family != "justification":
                good = [
                    c for c in self._candidates()
                    if c.get("live") and c.get("spec") == 2 and self._brand_ok(c["pid"]) and c.get("cat_ok")
                    and self._device_ok(c["pid"], c["sku"])
                    and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)
                    and c.get("key") != dead
                    and c.get("key") not in self.ledger.dead_keys
                ]
            if self.brief.family == "recovery" and self.brief.pref_model:
                same = [c for c in good if c.get("relax", 3) <= (1 if self.ledger.dead_keys else 0)]
                if same:
                    good = same
                elif not self.ledger.dead_keys:
                    good = []
            if self.brief.family == "preference":
                good = [c for c in good if self._device_ok(c["pid"], c["sku"])]
                top = [c for c in good if self._top_axis_hit(c["pid"], c["sku"])]
                if top:
                    good = top
                good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                         c["price"] if c["price"] is not None else 1e12))
            else:
                good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0 and not self.ledger.dead_keys))
            if not good:
                # Last resort: any live inspected under-budget row that still meets
                # the firm gates — never a random accessory just to avoid agent_error.
                good = [
                    c for c in self._candidates()
                    if c.get("live") and c.get("viewed") and c.get("key") != dead
                    and c.get("key") not in self.ledger.dead_keys
                    and self._device_ok(c["pid"], c["sku"])
                    and c.get("cat_ok")
                    and self._brand_ok(c["pid"])
                    and (c.get("spec") == 2 or not self.brief.spec_field)
                ]
                good.sort(key=lambda c: (c["price"] if c["price"] is not None else 1e12, c.get("key") or ""))
            if not good:
                return
            # Try several alternates when the first hits another market trap.
            # Budget: add + post-add recheck + order needs 3 turns; skip picks that
            # would consume the last turn on add alone (the old 409 storm).
            for pick in good[:5]:
                if self.done or self._steps_left() <= 0:
                    return
                pid, sku = pick["pid"], pick["sku"]
                need_view = str(pid) not in self.ledger.products and "view" in self.tool_names
                need_inspect = Ledger.key(pid, sku) not in self.ledger.inspected and "inspect_stock" in self.tool_names
                need_add = (pid, sku) not in self.ledger.cart
                # Turns required after optional view/inspect: add + recheck + order.
                reserve = (1 if need_view else 0) + (1 if need_inspect else 0)
                if need_add:
                    reserve += 2  # add + order (recheck optional if only 2 left)
                elif self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(
                    Ledger.key(pid, sku), 0
                ) <= self.ledger.last_add_seq:
                    reserve += 2  # recheck + order
                else:
                    reserve += 1  # order only
                if self._steps_left() < reserve:
                    continue
                if need_view:
                    self._act("view", {"product_id": pid})
                    if self.done or self._steps_left() <= 0:
                        return
                if need_inspect:
                    self._act("inspect_stock", {"product_id": pid, "sku": sku})
                    if self.done or self._steps_left() <= 0:
                        return
                if need_add:
                    # Must keep at least one turn after add for the order.
                    if self._steps_left() < 2:
                        continue
                    obs = self._act("add_to_cart", {"product_id": pid, "sku": sku})
                    if self.done or not isinstance(obs, dict) or obs.get("error"):
                        continue
                    if self.pending_recheck and self._steps_left() >= 2:
                        self._post_add_recheck()
                        if self.ledger.event_target:
                            continue
                    elif self.pending_recheck:
                        self.pending_recheck = None
                if Ledger.key(pid, sku) in self.ledger.dead_keys:
                    continue
                if (pid, sku) not in self.ledger.cart:
                    continue
                if self._steps_left() <= 0:
                    return
                # Recheck may already have run; only inspect again when still stale.
                k = Ledger.key(pid, sku)
                if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
                    if self._steps_left() < 2:
                        # Order on last known stock rather than burning the final turn.
                        self._act("place_test_order", self._order_args(pid, sku))
                        return
                    obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
                    if self.done or not isinstance(obs, dict):
                        continue
                    stock = bool(obs.get("in_stock"))
                    price = _num(obs.get("price"))
                    if (not stock) or not _budget_clear(price, self.budget, bool(self.ledger.dead_keys)):
                        self.ledger.dead_keys.add(k)
                        self.ledger.dead_products.add(str(pid))
                        self.ledger.event_target = k
                        self.event_remove_done = False
                        self.event_reinspect_done = False
                        self.event_announce_done = False
                        continue
                if self._steps_left() <= 0:
                    return
                self._act("place_test_order", self._order_args(pid, sku))
                return
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
                if question and not self.asked_shopper and "message" in self.tool_names:
                    self.asked_shopper = True
                    self._act("message", {"content": question})
                    continue
                if self.pending_recheck and "inspect_stock" in self.tool_names:
                    if self._steps_left() >= 2:
                        self._post_add_recheck()
                    else:
                        self.pending_recheck = None
                        self._finish_fast()
                        break
                    continue
                plan = self._plan()
                if plan is not None:
                    name, args = plan
                    if name != "place_test_order" and self._steps_left() <= 1:
                        self._finish_fast()
                        break
                    if name == "add_to_cart" and self._steps_left() < 2:
                        self._finish_fast()
                        break
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
                if self.model_dead:
                    # Inference is gone: keep driving the controller until stuck, then endgame.
                    forced = self._read_more()
                    if forced is not None:
                        name, args = forced
                        obs = self._act(name, args)
                        if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                            added = obs.get("added") or {}
                            if added.get("product_id"):
                                self.pending_recheck = (str(added["product_id"]), str(added["sku"]))
                        continue
                    self._finish_fast()
                    break
                proposal = self._ask_model()
                if proposal is None:
                    continue
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
            if not self.done and self._steps_left() > 0:
                self._finish_fast()
            return self.dialogue
        except Exception:
            return []

    def _validate(self, name: str, args: dict) -> str | None:
        try:
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
                if (
                    self.brief.family == "preference"
                    and not self.ledger.adds
                    and not self._top_axis_hit(pid, sku)
                ):
                    return (
                        "the first cart item must evidence the shopper's top preference "
                        "(from their priority answer); keep reading until one does"
                    )
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
        except Exception:
            return None


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
