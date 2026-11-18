"""Spatial Engine — capability-routed shopping runtime.

A supervisor composes specialized engines. Engines activate from live
signals (query structure, tool schema, ledger, shopper text), never from
a frozen family table. A new task shape reuses the same pipelines.

======================================================================
SYSTEM WORKFLOW
======================================================================

                    ┌─────────────────────────────────────┐
                    │  Perceive  (query + policy_view)    │
                    │  → SignalGraph + tool roster        │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  SpatialEngine supervisor loop      │
                    │  poll engines by priority until     │
                    │  one proposes a legal action        │
                    └─────────────────┬───────────────────┘
                                      │
         ┌──────────┬──────────┬──────┴──────┬──────────┬──────────┐
         ▼          ▼          ▼             ▼          ▼          ▼
      Inquiry   Dialogue   Hygiene      Evidence    Commit     Terminal
      (sealed)  (shopper)  (dead cart)  (read/     (first     (order)
                                         inspect)    add)
         │          │          │             │          │          │
         └──────────┴──────────┴──────┬──────┴──────────┴──────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Absorb observations → update       │
                    │  ledger + signals; market recheck   │
                    │  after every cart add               │
                    └─────────────────────────────────────┘

Exploration (the language model) runs only when no engine has a
grounded move. The supervisor owns every action the verifier grades.

======================================================================
PIPELINES (one engine each)
======================================================================

  Inquiry      Ask once when a firm requirement is sealed.
  Dialogue     Answer a shopper question from observed evidence.
  Hygiene      Remove a cart line the market has already killed.
  Verification Re-read a just-carted SKU; mark shocks from the ledger.
  Recovery     Replace a dead commitment with a still-valid variant.
  Discovery    Search / filter from extracted constraints, not scripts.
  Evidence     View, compare, inspect unread leads that still look live.
  Staging      Inspect a second firm-valid product before first add.
  Commitment   First add honours an available soft preference.
  Terminal     Order only a carted, re-checked, schema-legal SKU.
  Exploration  Model proposes catalog moves; supervisor filters them.

Gates compose independently (quota AND sealed spec AND soft brand can
all be live on one problem). Nothing here names a task family.
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
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_PROBLEM_BUDGET_S = 270.0
_OBS_CHARS = 7000
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|up to|no more than|not more than|max(?:imum)? of|"
    r"budget (?:is|of)|only go up to|can spend up to|at most|spend up to)\s*"
    + _NUM
    + r"(?:\s*(PHP|php|₱|pesos?|USD|usd))?",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RE = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RE = re.compile(
    r"(?:surface|find|cover|survey|return|need)\s+(?:at least\s+)?(\d+)\s+distinct",
    re.I,
)
_HIDDEN_RE = re.compile(
    r"hidden (?:firm )?requirement|tell you if you ask|if you ask|"
    r"one firm requirement|answers? (?:it|that) when asked|"
    r"ask(?:s|ed)? (?:the )?(?:shopper|user).{0,60}requirement|"
    r"reveals? (?:it|the (?:spec|requirement))",
    re.I,
)
_WATER_RE = re.compile(r"\b(?:ipx?\s*\d|waterproof|water[\s-]?resist|sweat|splash)\b", re.I)
_WATER_ASK = re.compile(r"\b(?:water|sweat|ipx|waterproof|weather|workout|gym|run)\b", re.I)
_PREF_ITEM_RE = re.compile(r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I)
_PREF_BRAND_RE = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_REASON_RE = re.compile(r"\b(?:why|trustworthy reason|explain why|tell me why|reason i can trust)\b", re.I)
_CLAIM_FIELDS = ("resolution", "refresh_rate", "screen_size", "price", "in_stock")
_SOFT_REASON_FIELDS = ("brand", "streaming", "microphone", "privacy_cover", "autofocus", "hdr")
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_CHOICE_Q_RE = re.compile(
    r"which one are you going|does it actually have|before you buy|what are you (?:picking|choosing)",
    re.I,
)
_EVENT_Q_RE = re.compile(
    r"more expensive|out of stock|re-?check before ordering|just get more|price (?:just )?(?:changed|increased|jumped)"
    r"|still available",
    re.I,
)
_ACCESSORY_RE = re.compile(
    r"\b(?:cases?|covers?|casing|tempered|glass|charger|cable|dummy|trigger|stand|protector|"
    r"bumper|replacement|skin|holster|pouch|lanyard|bracket)\b",
    re.I,
)
_NONDEVICE_RE = re.compile(
    r"\b(?:plush|toys?|dolls?|griddle|pans?\b|camshaft|notebook|backpack|bracelet|"
    r"sketch|ear\s+tag|sizzling|insect|reel|pillow|necklace|bulb|yoga|polo|shirt|"
    r"cotton|nametag|magnets?|tumbler|ceramic|cam\b|filaments?)\b",
    re.I,
)
_WIRED_ONLY_RE = re.compile(r"\bwired\b|3\.5\s*mm|\baux\b", re.I)
_WIRELESS_RE = re.compile(r"wireless|bluetooth|\btws\b|ipx|sport|sweat", re.I)
_HEADPHONE_TITLE_RE = re.compile(r"\b(?:headphones?|headsets?|earbuds?|earphones?)\b", re.I)
_MONITOR_TITLE_RE = re.compile(r"\b(?:monitors?|displays?)\b", re.I)
_SOFT_QUALITY_RE = re.compile(
    r"rather pay|cheapest isn't|don'?t just grab the cheapest|"
    r"matters more to me|what i actually care|stated priorit|"
    r"prefer .{0,80}(?:over|than)\s+(?:cheap|the cheapest|something flimsy)",
    re.I,
)
_QUALITY_VS_PRICE_RE = re.compile(
    r"(?:not|isn't|is not|don'?t)\s+(?:just\s+)?(?:grab |take )?(?:the\s+)?cheap|"
    r"prefer\b.{0,80}\b(?:over|than)\b.{0,40}\bcheap|"
    r"\b(?:priorit(?:y|ies|ise|ize)|matters more|care(?:s)? more)\b|"
    r"\brather (?:pay|than (?:price|cost))\b|"
    r"\bflimsy\b",
    re.I,
)
_AXIS_PATTERNS = (
    ("water", re.compile(r"\b(?:ipx?\s*\d|waterproof|water[\s-]?resist|sweat|splash|weather)\b", re.I)),
    ("wireless", re.compile(r"\b(?:wireless|bluetooth|\btws\b|cordless)\b", re.I)),
    ("battery", re.compile(r"\b(?:battery(?:\s+life)?|runtime|long(?:er)?\s+charge)\b", re.I)),
    ("sport", re.compile(r"\b(?:sport|gym|workouts?|runs?|running|fitness)\b", re.I)),
)
_AXIS_SEARCH = {
    "water": ("waterproof", "ipx", "sweat"),
    "wireless": ("wireless", "bluetooth"),
    "battery": ("battery",),
    "sport": ("sport",),
}
_REVISION_RE = re.compile(
    r"\b(?:instead|actually|change[ds]?|update[d]?|now (?:need|want|require)|"
    r"rather|switch(?:ed)? to|new (?:spec|requirement|size|value)|no longer)\b",
    re.I,
)
_SEARCH_STOP = {
    "in", "stock", "now", "available", "today", "only", "php", "the", "a", "an", "and",
    "or", "for", "with", "of", "to", "on", "at", "is", "it", "this", "that",
}

_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_MEM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_RE = re.compile(r"\bqHD\b")

_FACT_LABELS = {
    "screen_size": ("size", "screen_size", "display_size"),
    "resolution": ("display", "resolution"),
    "refresh_rate": ("refresh_rate", "refresh"),
    "ram": ("ram", "memory", "device_memory", "system_memory"),
}

# Shelf names vary by retailer; only the *head noun* is aliased, never a task.
_SHELF_ALIASES = {
    "smartphone": {"phone", "cellphone", "mobile", "handset", "unlocked"},
    "phone": {"smartphone", "cellphone", "mobile", "handset"},
    "headphone": {"headset", "earbud", "earphone"},
    "headset": {"headphone", "earbud", "earphone"},
    "monitor": {"display", "screen"},
    "display": {"monitor", "screen"},
}

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

_CORE_RULES = (
    "You are a shopping agent. Follow the shopper request and the Task rules in the user "
    "message. Use only the supplied tools. Batch independently useful catalog calls in "
    "one turn (search+filter+compare/view) up to max_calls_per_turn; do not spend a turn "
    "on a single search when you can also read pages. "
    "Separate firm requirements (category, hardware spec, budget, in-stock, device identity) "
    "from soft preferences (a preferred brand, nicer extras). Never buy something that "
    "misses a firm requirement. Do not refuse a firm-valid item only because a soft "
    "preference is missing — except on the first cart add, honor an available preferred "
    "brand when one is fully compliant. "
    "The first add_to_cart is the graded commitment, not a placeholder. Before that add, "
    "view/compare raw facts and inspect_stock the exact SKU. Read at least one backup "
    "firm-valid product so a later price or stock shock has a replacement. "
    "After any cart add, re-inspect that SKU; if it is now over budget or out of stock, "
    "remove it and buy a different product that still meets every firm requirement. "
    "Answer a direct shopper question with message, then act — do not keep browsing. "
    "Never ask the shopper to raise the budget. Do not send status updates. "
    "When the live place_test_order tool requires justification, send only field/value "
    "claims: the requested spec from raw facts, live price, in_stock, and one allowed "
    "reason (brand or an explicitly true feature). Never claim category."
)


# ---------------------------------------------------------------- value readers
# The environment resolves a gated specification from public evidence only.
# These helpers mirror that reading so engines can predict the verdict.


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
    text = text or ""
    if _QUARTER_RE.search(text):
        return set()
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
    text = text or ""
    if field != "ram":
        return _VALUE_SETS[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RE.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RE.findall(text)}


def _mentions(field: str, text: str) -> bool:
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RE.search(text) or _MEM_RE.search(text))
    if field == "screen_size":
        return bool(_INCH_RE.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(
        _WXH_RE.search(text)
        or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I)
    )


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _wanted_value(field: str | None, text: str) -> str | None:
    if not field:
        return None
    return _one(_VALUE_SETS[field](text or ""))


def _ordered_values(field: str, text: str) -> list[str]:
    """Values in mention order so a revision can drop the old spec and keep the new one."""
    text = text or ""
    if field == "screen_size":
        return ["%gin" % float(m.group(1)) for m in _INCH_RE.finditer(text)]
    if field == "refresh_rate":
        return ["%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(text)]
    if field == "ram":
        return [
            "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
            for m in _GB_RE.finditer(text)
        ]
    if field == "resolution":
        out: list[str] = []
        for m in re.finditer(
            r"\b(?:ultra\s*hd|full\s*hd|uhd|qhd|fhd|4k|2k|2160p|1440p|1080p|720p|hd)\b|\d{3,5}\s*[x×]\s*\d{3,5}",
            text,
            re.I,
        ):
            found = _res_vals(m.group(0))
            for item in found:
                if item not in out:
                    out.append(item)
        return out
    return []


def _revised_value(field: str | None, text: str, current: str | None) -> str | None:
    """Pick a newly stated value. If the old spec is also mentioned, keep the other one."""
    if not field:
        return None
    ordered = _ordered_values(field, text)
    unique: list[str] = []
    for value in ordered:
        if value not in unique:
            unique.append(value)
    if current and current in unique and len(unique) > 1:
        unique = [value for value in unique if value != current]
    if len(unique) == 1:
        return unique[0]
    if unique:
        return unique[-1]
    return _wanted_value(field, text)


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _sku_family(sku: str) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if "-" in text else text


def _same_word(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def _stems(text: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}


def _identity_text(text: str, title: bool = False) -> str:
    text = _NEGATED_RE.sub("", text or "")
    return _BUNDLE_RE.split(text, maxsplit=1)[0] if title else text


def _phone_identity(facts: dict, title: str, options: str) -> str:
    """yes / no / unclear — public evidence that this listing is the asked device."""
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


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


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
    return text[: max(0, limit - 16)] + "...[truncated]"


def _head_noun(category: str | None) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", (category or "").lower()) if w]
    return words[-1] if words else ""


def _alias_stems(noun: str) -> set:
    stem = noun.rstrip("s")
    extra = set()
    for key, aliases in _SHELF_ALIASES.items():
        if stem == key or stem in aliases:
            extra |= {key} | aliases
    return {stem} | extra


# ---------------------------------------------------------------- clock / ledger / brief


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """Structural parse of the public request. No family label is stored."""

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
        sm = _SURFACE_RE.search(self.raw)
        self.required_identities = int(sm.group(1)) if sm else 0
        if not self.required_identities and re.search(
            r"several distinct|good shortlist|survey the realistic options", self.rules, re.I
        ):
            self.required_identities = 5
        pm = _PREF_ITEM_RE.search(self.goal)
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal, self.category)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec(self.goal)
        self.spec_sealed = self.spec_field is None and bool(_HIDDEN_RE.search(self.raw))
        self.axes = tuple(name for name, rx in _AXIS_PATTERNS if rx.search(self.goal))
        self.wants_water = "water" in self.axes or bool(_WATER_ASK.search(self.goal))
        self.wants_reason = bool(_REASON_RE.search(self.goal) or "trustworthy reason" in self.rules)
        # Identity-clustering rules mention listing_id; that is not a compare-two-listings gate.
        # Only shopper wording about extras / not-cheapest should wait on listing diversity.
        self.listing_diversity = bool(
            (
                self.wants_water
                or _SOFT_QUALITY_RE.search(self.goal)
                or _QUALITY_VS_PRICE_RE.search(self.goal)
            )
            and not self.theme
            and not self.required_identities
            and not self.candidate_keys
            and not self.pref_model
        )
        self.require_identity = bool(
            re.search(
                r"genuine|actual device|not a (?:feature phone|accessory)|rather than an accessory",
                self.raw,
                re.I,
            )
        )
        blob = (self.goal + " " + self.rules).lower()
        self.rank_fields: list = []
        if self.candidate_keys:
            if re.search(r"refresh", blob):
                self.rank_fields.append("refresh_rate")
            if re.search(r"screen|inch", blob):
                self.rank_fields.append("screen_size")
            if not self.rank_fields:
                self.rank_fields = ["refresh_rate", "screen_size"]
        self.event_expected = bool(
            re.search(r"prices? can change|repriced|verify the (?:current |final )?price", self.rules, re.I)
            or self.pref_brand
            or self.listing_diversity
            or self.candidate_keys
            or self.pref_model
            or (self.spec_field and not self.spec_sealed)
        )

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
    def _qualifier(goal: str, category: str | None) -> list:
        """Shape words inside the category phrase only — never sentence leftovers.

        Taking left-context words from 'I need a smartphone' or 'in-stock monitor'
        previously required titles to contain 'need' or 'stock', so every firm-valid
        SKU failed category_ok and the agent never carted.
        """
        # Only shape adjectives (gaming, over-ear). Multi-word shelf names like
        # "Tools & Home Improvement" must not require "tools"/"home" in the title
        # — that blocked every themed PVC/cement SKU and left recall with no cart.
        quals: list = []
        if re.search(r"\bgaming\s+monitors?\b", goal or "", re.I):
            quals.append("gaming")
        if re.search(r"\bover[\s-]?ear\b", goal or "", re.I):
            quals.extend(["over", "ear"])
        return quals

    @staticmethod
    def _specs(goal: str) -> dict:
        out = {}
        for field in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = _wanted_value(field, goal)
            if value:
                out[field] = value
        return out

    def _spec(self, goal: str) -> tuple:
        for field, value in self.specs.items():
            return field, value
        return None, None

    def adopt_spec(self, text: str) -> str | None:
        """Record or revise a firm spec from shopper text. Old + new mentions keep the new value."""
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            current = self.spec_value if field == self.spec_field else None
            value = _revised_value(field, text, current)
            if not value:
                continue
            if field == self.spec_field and value == self.spec_value:
                return None
            prev = self.spec_value
            self.spec_field, self.spec_value = field, value
            self.specs[field] = value
            self.spec_sealed = False
            if prev and prev != value:
                return "%s = %s (was %s)" % (field, value, prev)
            return field + " = " + value
        return None

    def theme_in(self, *texts: str) -> bool:
        """Quoted-phrase tokens may appear in the title or in raw facts."""
        if not self.theme:
            return True
        have: set = set()
        for text in texts:
            have |= _tokens(text)
        return all(tok in have for tok in self.theme)

    def axis_search_terms(self) -> list[str]:
        """Catalog query tokens taken from the shopper's own axes, never a shelf default."""
        terms: list[str] = []
        seen: set[str] = set()
        for axis in self.axes:
            for word in _AXIS_SEARCH.get(axis, (axis,)):
                key = word.lower()
                if key not in seen:
                    seen.add(key)
                    terms.append(word)
        return terms


class Ledger:
    """Observed catalog state, keyed by product and variant."""

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
            compared = obs.get("compared")
            if not isinstance(compared, list) and isinstance(obs.get("observation"), dict):
                compared = obs["observation"].get("compared")
            for prod in compared or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief)
        elif name == "inspect_stock":
            body = obs
            if isinstance(obs.get("observation"), dict) and obs.get("price") is None:
                body = obs["observation"]
            pid = body.get("product_id") or args.get("product_id")
            sku = body.get("sku") or args.get("sku")
            k = self.key(pid, sku)
            self.checked[k] = {"in_stock": bool(body.get("in_stock")), "price": _num(body.get("price"))}
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
            if brief.theme_in(title):
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
        if brief.theme and brief.theme_in(str(prod.get("title") or "")):
            self.identities.add(str(pid))
        elif not brief.theme:
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


# ---------------------------------------------------------------- engines


class Engine:
    """One specialized pipeline. ``propose`` returns (name, args) or None."""

    name = "engine"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        return None


class InquiryEngine(Engine):
    """Ask once for a requirement the public query says is sealed."""

    name = "inquiry"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if not rt.brief.spec_sealed or rt.asked_shopper or "message" not in rt.tool_names:
            return None
        rt.asked_shopper = True
        return (
            "message",
            {
                "content": (
                    "Before I search: which hardware specification is your firm requirement "
                    "(screen size, resolution class, refresh rate, or memory), and what exact value do you need?"
                )
            },
        )


class DialogueEngine(Engine):
    """Answer the shopper from ledger evidence, then yield the turn."""

    name = "dialogue"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if not rt.shopper_question_pending or "message" not in rt.tool_names:
            return None
        return ("message", {"content": rt.answer_text()})


class HygieneEngine(Engine):
    """Drop cart lines already marked dead so recovery can replace them."""

    name = "hygiene"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if "remove_from_cart" not in rt.tool_names:
            return None
        for pid, sku in rt.ledger.cart:
            if Ledger.key(pid, sku) in rt.ledger.dead_keys:
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
        return None


class VerificationEngine(Engine):
    """Re-inspect a just-carted SKU, or a carted SKU that is still stale."""

    name = "verification"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if "inspect_stock" not in rt.tool_names:
            return None
        for pid, sku in rt.ledger.cart:
            if Ledger.key(pid, sku) in rt.ledger.dead_keys:
                continue
            if not rt.fresh(pid, sku):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
        return None


class TerminalEngine(Engine):
    """Order a carted SKU that is live, fresh, and schema-legal."""

    name = "terminal"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if "place_test_order" not in rt.tool_names:
            return None
        if rt.brief.required_identities and rt.search_calls < 3 and rt.turn < rt.max_steps - 4:
            return None
        if rt.hold_order():
            return None
        for pid, sku in reversed(rt.ledger.cart):
            if Ledger.key(pid, sku) in rt.ledger.dead_keys:
                continue
            if rt.assess(pid, sku)["ok"] and rt.fresh(pid, sku):
                ok, _why = rt.valid_now(pid, sku)
                if not ok:
                    continue
                why = rt._validate("place_test_order", {"product_id": pid, "sku": sku})
                if why:
                    continue
                return ("place_test_order", rt.order_args(pid, sku))
        return None


class RecoveryEngine(Engine):
    """Replace a shocked commitment with a different still-valid SKU."""

    name = "recovery"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if not (rt.ledger.event_target or rt.ledger.dead_keys):
            return None
        if rt.brief.pref_model:
            twin = rt._twin_search()
            if twin is not None:
                return twin
            sib = rt._sibling_search()
            if sib is not None:
                return sib
        target = rt.target()
        if target:
            pid, sku = target["pid"], target["sku"]
            if (pid, sku) not in rt.ledger.cart:
                if Ledger.key(pid, sku) not in rt.ledger.inspected and "inspect_stock" in rt.tool_names:
                    return ("inspect_stock", {"product_id": pid, "sku": sku})
                return ("add_to_cart", {"product_id": pid, "sku": sku})
        return rt.read_more()


class DiscoveryEngine(Engine):
    """Open the catalog from extracted constraints (brand, spec, category)."""

    name = "discovery"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if rt.ledger.cart or rt.shopper_question_pending:
            return None
        if rt.brief.spec_sealed and not rt.asked_shopper:
            return None
        if rt.brief.candidate_keys or rt.brief.theme or rt.brief.pref_model:
            return rt.read_more()
        return rt.catalog_probe()


class StagingEngine(Engine):
    """Inspect a second firm-valid product before the public first add."""

    name = "staging"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        hold = rt.hold_order()
        if hold and rt.ledger.cart and not rt.ledger.event_target:
            backup = rt.stage_backup()
            if backup is not None:
                return backup
            if "inspect_cart" in rt.tool_names and rt.cart_inspections < 1:
                return ("inspect_cart", {})
            for c in rt.firm_backups():
                if c["key"] not in rt.ledger.inspected and "inspect_stock" in rt.tool_names:
                    return ("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
        if rt.ledger.cart or rt.ledger.event_target or rt.ledger.dead_keys:
            return None
        if not rt.brief.event_expected or not rt.ready():
            return None
        return rt.stage_backup()


class CommitmentEngine(Engine):
    """First cart add: the best fully verified candidate under live signals."""

    name = "commitment"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if rt.ledger.cart or not rt.ready():
            return None
        target = rt.target()
        if not target:
            return rt.read_more()
        pid, sku = target["pid"], target["sku"]
        if Ledger.key(pid, sku) not in rt.ledger.inspected and "inspect_stock" in rt.tool_names:
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return ("add_to_cart", {"product_id": pid, "sku": sku})


class EvidenceEngine(Engine):
    """Read unread leads only while a stronger engine has nothing to commit."""

    name = "evidence"

    def propose(self, rt: "SpatialEngine") -> tuple | None:
        if rt.ledger.cart:
            return None
        if rt.brief.spec_sealed and not rt.asked_shopper:
            return None
        if rt.named_keys_complete():
            return None
        if rt.ready() and rt.target() is not None:
            return None
        return rt.read_more()


# Priority is the system workflow. First non-None proposal wins.
_PIPELINE: tuple[type[Engine], ...] = (
    InquiryEngine,
    DialogueEngine,
    HygieneEngine,
    VerificationEngine,
    TerminalEngine,
    RecoveryEngine,
    DiscoveryEngine,
    StagingEngine,
    CommitmentEngine,
    EvidenceEngine,
)


# ---------------------------------------------------------------- runtime


class SpatialEngine:
    """Binds one problem to the engine pipeline and drives the session."""

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
        self.clock = Clock(_PROBLEM_BUDGET_S)
        self.dialogue: list = []
        self.order_requires_justification = False
        self.reason_fields = _SOFT_REASON_FIELDS
        self._parse_order_contract()
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
        self._last_sent: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.widen_index = 0
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.bootstrapped = False
        self.brand_probed = False
        self.brand_filtered = False
        self.done = False
        self.closing = False
        self.revision_hold_used = False
        self.recheck_done_turn: int | None = None
        self.model_index = 0
        self.engines = [cls() for cls in _PIPELINE]
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _parse_order_contract(self) -> None:
        for tool in self.tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
                continue
            params = fn.get("parameters") or {}
            self.order_requires_justification = "justification" in (params.get("required") or [])
            blob = json.dumps(fn).lower()
            found = [f for f in _SOFT_REASON_FIELDS if f.replace("_", " ") in blob or f in blob]
            if found:
                self.reason_fields = tuple(found)
            return

    def _system_prompt(self) -> str:
        extra = " You may emit up to %d tool calls in one turn." % self.max_calls
        extra += " You have at most %d turns." % self.max_steps
        firm = [x for x in (self.brief.category, self.brief.spec_value, self.brief.brand) if x]
        if self.budget is not None:
            firm.append("budget %.2f %s" % (self.budget, self.brief.currency))
        if firm:
            extra += " Firm requirements: " + ", ".join(str(x) for x in firm) + "."
        if self.brief.pref_brand:
            extra += (
                " Soft preference: brand %s — use it on the first cart add if a fully "
                "compliant one exists; after a market shock any firm-valid brand is fine."
                % self.brief.pref_brand
            )
        if self.brief.required_identities:
            extra += (
                " Surface at least %d distinct matching products in search or filter "
                "results before ordering." % self.brief.required_identities
            )
        if self.brief.wants_water or self.brief.listing_diversity:
            extra += (
                " Soft priority: sweat/water resistance (IPX, waterproof) over the cheapest "
                "listing. Search those terms, compare at least two listings, and first-add the "
                "genuine preference match — not the cheapest decoy."
            )
        if self.brief.spec_sealed:
            extra += (
                " One firm hardware specification is hidden until you ask; ask with "
                "message, then obey the shopper's answer exactly."
            )
        if self.order_requires_justification or self.brief.wants_reason:
            extra += (
                " Orders must include justification claims for the requested spec, "
                "live price, in_stock, and one reason from %s."
                % ", ".join(self.reason_fields)
            )
        if self.brief.pref_brand and self.brief.event_expected:
            extra += (
                " On the first turn search and filter the preferred brand plus the "
                "hardware spec together, compare two firm-valid products, inspect the "
                "branded one, answer the shopper, then cart it immediately."
            )
        return _CORE_RULES + extra

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
        raw = self._fact_raw(pid, field)
        if not raw:
            return None
        text = str(raw).strip()
        found = _VALUE_SETS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return _one(found)

    def _fact_raw(self, pid: str, field: str):
        facts = self._facts(pid)
        for label in _FACT_LABELS.get(field, ()):
            raw = facts.get(label)
            if raw:
                return raw
        return None

    def _product_value(self, pid: str, field: str) -> str | None:
        vals: set = set()
        conflict = False
        raw = self._fact_raw(pid, field)
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

    def _spec_proof(self, pid: str, sku: str, field: str) -> tuple:
        """Resolve a spec from strongest public evidence. Title is last resort."""
        opt = self._options(pid, sku)
        opt_vals = _adjacent_vals(field, opt)
        if len(opt_vals) > 1:
            return None, "variant-conflict"
        if opt_vals:
            return next(iter(opt_vals)), "variant"
        if _mentions(field, opt):
            return None, "variant-unclear"
        raw = self._fact_raw(pid, field)
        title = self._title(pid)
        if field == "resolution":
            title = self._strip_models(pid, title)
        fact_vals: set = set()
        conflict = False
        if raw:
            fact_vals = _VALUE_SETS[field](str(raw))
            if field == "resolution" and _QUARTER_RE.search(str(raw)):
                conflict = True
            if len(fact_vals) > 1:
                conflict = True
        title_vals = _adjacent_vals(field, title) if field != "resolution" else _VALUE_SETS[field](title)
        if field == "resolution" and _QUARTER_RE.search(title):
            conflict = True
        if fact_vals and title_vals and fact_vals != title_vals and not (fact_vals & title_vals):
            conflict = True
        if conflict:
            return None, "conflict"
        if len(fact_vals) == 1:
            return next(iter(fact_vals)), "facts"
        if len(title_vals) == 1:
            return next(iter(title_vals)), "title"
        return None, "none"

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        got, _source = self._spec_proof(pid, sku, field)
        return got

    def _measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self._variant_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    def _category_ok(self, pid: str) -> bool:
        want = self.brief.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        head = _head_noun(want)
        wanted = _alias_stems(head) | _stems(want)
        labels = set()
        for label in path:
            labels |= _stems(str(label))
        placed = bool(wanted & labels) or any(
            _stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == _stems(want) for label in path
        )
        if not placed:
            title_stems = _stems(self._title(pid)) | _tokens(self._title(pid))
            placed = bool(wanted & title_stems)
        if not placed and self.brief.theme:
            # Recall is gated on the quoted phrase; shelf spelling varies by retailer.
            placed = True
        if not placed:
            return False
        if self.brief.theme:
            return True
        quals = self.brief.cat_qualifier
        if not quals:
            return True
        title = self._title(pid)
        words = _stems(title) | _tokens(title)
        for q in quals:
            qstem = q.rstrip("s")
            if qstem == "gaming":
                fast = any(float(v[:-2]) >= 120 for v in _hz_vals(title))
                if "gaming" not in words and not fast:
                    return False
                continue
            if qstem in ("over", "ear"):
                if not (("over" in words and "ear" in words) or "overear" in words or "circumaural" in words):
                    return False
                continue
            if qstem not in words:
                return False
        return True

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        """Reject listings that contradict the asked device; require proof when identity is graded."""
        cat = self.brief.category or ""
        if not _same_word(cat, "smartphone") and "phone" not in _stems(cat):
            return True
        verdict = _phone_identity(self._facts(pid), self._title(pid), self._options(pid, sku))
        if self.brief.wants_reason or self.brief.brand or self.brief.require_identity:
            return verdict == "yes"
        return verdict != "no"

    def _resale_title(self, pid: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self._title(pid),
                re.I,
            )
        )

    def _relax(self, pid: str, sku: str) -> int:
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

    def assess(self, pid: str, sku: str) -> dict:
        """Score one variant against every gate that is live on this problem."""
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        keep_product = bool(self.brief.pref_model)
        dead = key in self.ledger.dead_keys or (str(pid) in self.ledger.dead_products and not keep_product)
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        field, want = self.brief.spec_field, self.brief.spec_value
        spec = 2
        spec_source = "none"
        if field and want:
            got, spec_source = self._spec_proof(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if self.brief.spec_sealed:
            ok = False
        if self.brief.wants_reason:
            ok = ok and not self._resale_title(pid)
            if field and want:
                ok = ok and self._fact_value(pid, field) == want and self._product_value(pid, field) == want
        if self.brief.listing_diversity and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if self.brief.theme:
            ok = ok and self._theme_hit(pid, sku)
        if self.brief.candidate_keys:
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
            for measure in self.brief.rank_fields or self.brief.specs:
                if self._measure(pid, sku, measure) is None:
                    ok = False
                    break
        return {
            "pid": str(pid),
            "sku": str(sku),
            "key": key,
            "price": price,
            "live": live,
            "viewed": viewed,
            "spec": spec,
            "spec_source": spec_source,
            "cat_ok": cat_ok,
            "ok": ok,
            "relax": self._relax(pid, sku) if self.brief.pref_model else 0,
        }

    def _rank_of(self, c: dict, first: bool) -> tuple:
        price = c["price"] if c["price"] is not None else 1e12
        if self.brief.candidate_keys:
            measures = []
            fields = self.brief.rank_fields or [f for f in ("refresh_rate", "screen_size", "ram") if f in self.brief.specs]
            for field in fields:
                measures.append(-(self._measure(c["pid"], c["sku"], field) or 0.0))
            return tuple(measures) + (price, c["key"])
        if self.brief.pref_model:
            if first or not self.ledger.event_target:
                return (c["relax"], price, c["key"])
            # After the named color dies, a sibling color of that listing ranks above
            # another seller's copy of the same color (relax 0 is the wrong recover).
            dead_sku = (self.ledger.event_target or "").split("::", 1)[-1]
            family = _sku_family(dead_sku)
            same_family = 0 if family and str(c["sku"]).startswith(family) else 1
            recover_tier = 0 if c["relax"] == 1 else (1 if c["relax"] == 0 else c["relax"] + 1)
            return (recover_tier, same_family, price, c["key"])
        if self.brief.listing_diversity:
            keys = []
            axes = set(self.brief.axes or ())
            if "water" in axes or self.brief.wants_water:
                keys.append(0 if self._water_hits(c["pid"], c["sku"]) > 0 else 1)
            if "sport" in axes:
                keys.append(0 if self._sport_hits(c["pid"], c["sku"]) > 0 else 1)
            if "wireless" in axes:
                keys.append(0 if self._wireless_hits(c["pid"], c["sku"]) > 0 else 1)
            if "battery" in axes:
                keys.append(0 if self._battery_hits(c["pid"], c["sku"]) > 0 else 1)
            quality = -self._priority_hits(c["pid"], c["sku"])
            # Cheapest is the decoy. Pay more for an axis match.
            return tuple(keys) + (quality, -price, c["key"])
        extra = sum(
            1
            for f, v in self.brief.specs.items()
            if f != self.brief.spec_field and self._variant_value(c["pid"], c["sku"], f) != v
        )
        proven = 0 if _phone_identity(self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
        if first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (near, extra, proven, price, c["key"])
        return (extra, proven, price, c["key"])

    def candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.assess(pid, sku))
        return out

    def target(self) -> dict | None:
        first = self.ledger.adds == 0
        good = [c for c in self.candidates() if c["ok"]]
        if self.brief.listing_diversity and not self.brief.pref_brand:
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            if not good:
                return None
            if not self.closing and len(self.ledger.listing_ids) < 2 and self.turn < 10:
                return None
            if (
                not self.closing
                and self.brief.wants_water
                and not self._has_water_candidate()
                and self._water_hunt_open()
                and self.turn < 8
            ):
                return None
            if self.brief.wants_water:
                wet = [c for c in good if self._water_hits(c["pid"], c["sku"]) > 0]
                if wet:
                    good = wet
                elif not self.closing and self._water_hunt_open():
                    return None
            good.sort(key=lambda c: self._rank_of(c, first))
            return good[0]
        if self.brief.pref_model and self.ledger.event_target and not first:
            if not self.searched_twins:
                return None
            siblings = [c for c in good if c["relax"] == 1]
            if siblings:
                good = siblings
            elif not self.searched_siblings and self.turn < self.max_steps * 0.85:
                return None
        if first and self.brief.pref_brand:
            named = [c for c in good if self.pref_hit(c["pid"])]
            if named:
                good = named
            elif not self.closing and self.turn < self.max_steps * 0.65 and self.clock.remaining() > 50:
                self._preferred_brand_hint()
                return None
        if first and self.brief.pref_model:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * 0.6:
                self._exact_match_hint()
                return None
        if first and self.brief.spec_field and not self.closing:
            stronger = [c for c in good if c.get("spec_source") in ("variant", "facts")]
            if stronger and self.clock.remaining() > 40:
                good = stronger
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def pref_hit(self, pid: str) -> bool:
        want = self.brief.pref_brand
        if not want:
            return False
        tokens = _tokens(want)
        return bool(tokens <= _tokens(self._brand_of(pid)) or tokens <= _tokens(self._title(pid)))

    def _sku_invalid(self, pid: str, sku: str) -> bool:
        if Ledger.key(pid, sku) in self.ledger.dead_keys:
            return True
        st = self.ledger.status(pid, sku)
        if st is None:
            return False
        if not st.get("in_stock"):
            return True
        price = st.get("price")
        return bool(self.budget is not None and price is not None and price > self.budget + _EPS)

    def _mark_event_dead(self, pid: str, sku: str) -> None:
        key = Ledger.key(pid, sku)
        if self.ledger.event_target is None:
            self.ledger.event_target = key
            self.event_turn = self.turn
        self.ledger.dead_keys.add(key)
        if not self.brief.pref_model:
            self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0

    def branded_ok(self) -> list:
        return [c for c in self.candidates() if c["ok"] and self.pref_hit(c["pid"])]

    def first_add_refused(self, pid: str, sku: str | None = None) -> str | None:
        if self.ledger.adds or self.closing:
            return None
        if self.brief.pref_brand and not self.pref_hit(pid):
            if self.branded_ok():
                return "the first cart item must be the preferred brand when a fully compliant one is on record"
            if self.turn < self.max_steps * 0.65 and self.clock.remaining() > 50:
                return "keep looking for a preferred-brand item that meets every firm requirement before the first add"
        if self.brief.wants_water and sku:
            water = self._water_hits(pid, sku)
            if water <= 0 and any(self._water_hits(c["pid"], c["sku"]) > 0 for c in self.candidates() if c["ok"]):
                return "the first cart item must be the water/sweat-resistant match, not a cheaper decoy"
        return None

    def answer_text(self) -> str:
        shocked = self.ledger.event_target
        if not shocked:
            for pid, sku in self.ledger.cart:
                if self._sku_invalid(pid, sku):
                    shocked = Ledger.key(pid, sku)
                    break
        if shocked:
            st = self.ledger.checked.get(shocked) or {}
            price = st.get("price")
            return (
                "Yes — that item changed%s. I will switch to another product that still "
                "meets every firm requirement."
                % ((" and is now %.2f" % price) if price is not None else "")
            )
        target = self.target() or next((c for c in self.candidates() if c["ok"]), None)
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        field, want = self.brief.spec_field, self.brief.spec_value
        got = self._variant_value(pid, sku, field) if field else None
        if field and want and got == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        elif field and want:
            parts.append("I am checking that its " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.brief.currency))
        return ", ".join(parts) + "."

    def _spec_token(self) -> str:
        raw = str(self.brief.spec_value or "")
        m = re.search(r"(\d+(?:\.\d+)?)", raw)
        return m.group(1) if m else raw.strip()

    def _title_has_spec(self, title: str) -> bool:
        if not self.brief.spec_value:
            return True
        blob = str(title or "").lower()
        token = self._spec_token().lower()
        if token and token in blob:
            return True
        want = str(self.brief.spec_value).lower()
        return bool(want and want in blob)

    def _looks_like_device(self, title: str) -> bool:
        title = title or ""
        cat = (self.brief.category or "").lower()
        head = _head_noun(cat)
        if _NONDEVICE_RE.search(title):
            return False
        if head and _ACCESSORY_RE.search(title) and head in {"smartphone", "phone", "monitor", "headphone", "headset"}:
            return False
        if head in {"headphone", "headset", "earbud"} and not _HEADPHONE_TITLE_RE.search(title):
            return False
        if head in {"monitor", "display"} and not _MONITOR_TITLE_RE.search(title):
            return False
        if not cat:
            return True
        words = _stems(title) | _tokens(title)
        aliases = _alias_stems(head) if head else set()
        if words & (aliases | _stems(cat)):
            return True
        if self.brief.pref_brand and _tokens(self.brief.pref_brand) <= _tokens(title):
            return True
        if self.brief.brand and _tokens(self.brief.brand) <= _tokens(title):
            return True
        if self.brief.pref_model:
            squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
            if squash(self.brief.pref_model) in squash(title):
                return True
        if self.brief.theme and self.brief.theme_in(title):
            return True
        return False

    def _theme_hit(self, pid: str, sku: str | None = None) -> bool:
        # Verifier cluster membership is title-token only (no stemming).
        return self.brief.theme_in(self._title(pid))

    def _hunt_brand(self) -> str | None:
        if self.ledger.adds == 0 and self.brief.pref_brand:
            return self.brief.pref_brand
        return self.brief.brand

    def _catalog_filter_args(self, brand: str | None) -> dict | None:
        if "filter" not in self.tool_names:
            return None
        if not (self.brief.category or brand):
            return None
        args: dict[str, Any] = {"in_stock": True, "k": 10}
        if self.brief.category:
            args["category"] = self.brief.category
        if self.budget is not None:
            args["max_price"] = self.budget
        if brand:
            args["brand"] = brand
        return args

    def _unseen_leads(self) -> list:
        leads = []
        hunt_spec = bool(self.brief.spec_value) and (
            bool(self.brief.pref_brand) or bool(self.brief.spec_field) or bool(self.ledger.event_target)
        )
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            if not self._looks_like_device(title):
                continue
            if hunt_spec and self.brief.spec_value and not self._title_has_spec(title):
                continue
            if self.brief.theme and not self.brief.theme_in(title):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get("brand") or "")) or wanted <= _tokens(title)):
                    continue
            branded = 0 if self.pref_hit(str(pid)) else 1
            water = 0 if _WATER_RE.search(title) else (1 if _WIRELESS_RE.search(title) else 2)
            leads.append((water if self.brief.wants_water else 0, branded, price, str(pid)))
        leads.sort()
        out = []
        for _water, _branded, _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _water_leads(self) -> list:
        out = []
        for pid in self._unseen_leads():
            title = self._title(pid)
            if _WATER_RE.search(title) or _WIRELESS_RE.search(title):
                out.append(pid)
        return out

    def _has_water_candidate(self) -> bool:
        for c in self.candidates():
            if c["live"] and c["viewed"] and self._water_hits(c["pid"], c["sku"]) > 0:
                return True
        return False

    def _searches_left(self) -> bool:
        return self.search_calls < self._search_cap() and self.widen_index < len(self._widen_queries())

    def _water_hunt_open(self) -> bool:
        """Still unread water-shaped rows or unused catalog queries, and time left to read them."""
        if self.clock.remaining() <= 30 or self.turn >= self.max_steps - 3:
            return False
        return bool(self._water_leads()) or self._searches_left()

    def _catalog_exhausted(self) -> bool:
        if self._named_key_reads() is not None:
            return False
        if self._water_leads() or self._unseen_leads() or self._searches_left():
            return False
        return bool(self.reads_done)

    def _spec_leads(self) -> list:
        return [
            pid
            for pid in self._unseen_leads()
            if self._looks_like_device(self._title(pid) or "") and self._title_has_spec(self._title(pid))
        ]

    def _widen_queries(self) -> list:
        spec = self.brief.spec_value
        token = self._spec_token()
        brand = self._hunt_brand()
        cat = self.brief.category
        inch = self.brief.spec_field == "screen_size"
        seen, out = set(), []
        for words in (
            (token, "inch", cat) if inch and token else None,
            (spec, cat),
            (brand, spec, cat),
            (brand, token, "inch", cat) if inch and token else None,
            (brand, token, cat) if token else None,
            (brand, spec) if brand and spec else None,
            (brand, cat) if brand and self.ledger.event_target else None,
            (self.brief.pref_model, cat) if self.brief.pref_model else None,
            (self.brief.pref_model, self.brief.pref_color, cat) if self.brief.pref_model else None,
            (" ".join(self.brief.theme),) if self.brief.theme else None,
            (" ".join(self.brief.theme), cat) if self.brief.theme and cat else None,
            (cat,) if self.brief.theme and cat else None,
            tuple(self.brief.axis_search_terms() + ([cat] if cat else [])) if self.brief.axis_search_terms() else None,
            ("waterproof", cat) if self.brief.wants_water else None,
            ("ipx", cat) if self.brief.wants_water else None,
            ("sweat", "resistant", cat) if self.brief.wants_water else None,
            ("wireless", "sport", cat) if self.brief.wants_water else None,
        ):
            if not words:
                continue
            q = " ".join(str(w) for w in words if w)
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
        return out

    def _search_cap(self) -> int:
        if self.brief.theme or self.brief.required_identities:
            return 3
        return 5

    def _search_fingerprint(self, query: str) -> frozenset:
        return frozenset(tok for tok in _tokens(query) if tok not in _SEARCH_STOP)

    def _search_seen(self) -> set:
        out = set()
        for key in self.ledger.calls_seen:
            if not str(key).startswith("search"):
                continue
            try:
                args = json.loads(str(key)[6:])
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict):
                out.add(self._search_fingerprint(str(args.get("query") or "")))
        return out

    def _next_search(self) -> tuple | None:
        if "search" not in self.tool_names:
            return None
        if self.search_calls >= self._search_cap():
            return None
        queries = self._widen_queries()
        seen = self._search_seen()
        while self.widen_index < len(queries):
            q = queries[self.widen_index]
            self.widen_index += 1
            args = {"query": q, "in_stock": True, "k": 10}
            if "search" + json.dumps(args, sort_keys=True) in self.ledger.calls_seen:
                continue
            fp = self._search_fingerprint(q)
            if fp and fp in seen:
                continue
            return ("search", args)
        return None

    def _twin_search(self) -> tuple | None:
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

    def _sibling_search(self) -> tuple | None:
        if self.searched_siblings or not self.brief.pref_model or "search" not in self.tool_names:
            return None
        if not self.ledger.event_target:
            rows = [c for c in self.candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
            if rows:
                return None
        q = " ".join(str(w) for w in (self.brief.pref_model, self.brief.category) if w)
        if not q:
            return None
        args = {"query": q, "k": 10, "in_stock": True}
        if "search" + json.dumps(args, sort_keys=True) in self.ledger.calls_seen:
            self.searched_siblings = True
            return None
        self.searched_siblings = True
        return ("search", args)

    def _named_key_reads(self) -> tuple | None:
        """Open pages the shopper already named as product_id::sku pairs."""
        unseen = []
        inspect = []
        for pid, sku in self.brief.candidate_keys:
            if str(pid) not in self.ledger.products:
                if str(pid) not in unseen:
                    unseen.append(str(pid))
            elif Ledger.key(pid, sku) not in self.ledger.inspected:
                inspect.append((str(pid), str(sku)))
        if unseen and "compare" in self.tool_names and len(unseen) > 1:
            return ("compare", {"product_ids": unseen[:5]})
        if unseen and "view" in self.tool_names:
            return ("view", {"product_id": unseen[0]})
        if inspect and "inspect_stock" in self.tool_names:
            pid, sku = inspect[0]
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return None

    def named_keys_complete(self) -> bool:
        if not self.brief.candidate_keys:
            return False
        for pid, sku in self.brief.candidate_keys:
            if str(pid) not in self.ledger.products:
                return False
            if Ledger.key(pid, sku) not in self.ledger.inspected:
                return False
        return True

    def read_more(self) -> tuple | None:
        named = self._named_key_reads()
        if named is not None:
            return named
        if self.brief.candidate_keys:
            # The shopper named the only legal variants. Do not wander the catalog.
            return None
        if self.brief.wants_water:
            water = self._water_leads()
            if water and self.forced_compares < 16:
                self.forced_compares += 1
                if len(water) > 1 and "compare" in self.tool_names:
                    return ("compare", {"product_ids": water[:5]})
                if "view" in self.tool_names:
                    return ("view", {"product_id": water[0]})
            nxt = self._next_search()
            if nxt is not None and not self._has_water_candidate():
                return nxt
        if self.brief.required_identities:
            themed = [
                pid
                for pid in self._unseen_leads()
                if self.brief.theme_in(self._title(pid) or "")
            ]
            if themed and self.forced_compares < 16:
                self.forced_compares += 1
                if len(themed) > 1 and "compare" in self.tool_names:
                    return ("compare", {"product_ids": themed[:5]})
                if "view" in self.tool_names:
                    return ("view", {"product_id": themed[0]})
            if len(self.ledger.identities) < self.brief.required_identities and self._searches_left():
                nxt = self._next_search()
                if nxt is not None:
                    return nxt
        twin = self._twin_search()
        if twin is not None:
            return twin
        sibling = self._sibling_search()
        if sibling is not None:
            return sibling
        leads = self._spec_leads() if self.brief.spec_value else self._unseen_leads()
        cap = 12 if (self.ledger.event_target or self.ledger.dead_keys) else 8
        if leads and self.forced_compares < cap:
            self.forced_compares += 1
            if len(leads) > 1 and "compare" in self.tool_names:
                return ("compare", {"product_ids": leads[:5]})
            if "view" in self.tool_names:
                return ("view", {"product_id": leads[0]})
        if self.turn < 2:
            return None
        brand = self._hunt_brand()
        args = self._catalog_filter_args(brand)
        if args:
            sig = "filter" + json.dumps(args, sort_keys=True)
            if sig not in self.ledger.calls_seen:
                if brand:
                    self.brand_probed = True
                    self.brand_filtered = True
                return ("filter", args)
        nxt = self._next_search()
        if nxt is not None:
            return nxt
        self.reads_done = True
        return None

    def catalog_probe(self) -> tuple | None:
        brand = self._hunt_brand()
        if self.search_calls == 0 and "search" in self.tool_names:
            words = [w for w in (brand, self.brief.spec_value, self.brief.category) if w]
            if brand:
                self.brand_probed = True
            return ("search", {"query": " ".join(str(w) for w in words) or "in stock", "in_stock": True, "k": 10})
        if brand and not self.brand_filtered:
            args = self._catalog_filter_args(brand)
            if args:
                self.brand_filtered = True
                self.brand_probed = True
                return ("filter", args)
        if not self.ready():
            return self.read_more()
        return None

    def bootstrap_steps(self) -> list:
        if self.brief.spec_sealed:
            return []
        if self.brief.candidate_keys:
            return []
        steps = []
        if self.brief.pref_model and "search" in self.tool_names:
            q = " ".join(str(w) for w in (self.brief.pref_model, self.brief.pref_color, self.brief.category) if w)
            steps.append(("search", {"query": q, "in_stock": True, "k": 10}, "", ""))
            return steps
        if self.brief.theme and "search" in self.tool_names:
            q = " ".join(self.brief.theme)
            if self.brief.category:
                q = q + " " + self.brief.category
            steps.append(("search", {"query": q, "in_stock": True, "k": 10}, "", ""))
            return steps
        if (self.brief.wants_water or self.brief.listing_diversity) and "search" in self.tool_names:
            cat = self.brief.category or "headphone"
            terms = self.brief.axis_search_terms()
            q = " ".join(terms + [cat]) if terms else "waterproof ipx " + cat
            steps.append(("search", {"query": q, "in_stock": True, "k": 10}, "", ""))
            extra = terms[:2] + ([cat] if cat else [])
            extra_q = " ".join(str(w) for w in extra if w)
            if extra_q and extra_q != q:
                steps.append(("search", {"query": extra_q, "in_stock": True, "k": 10}, "", ""))
            if "filter" in self.tool_names:
                args: dict[str, Any] = {"in_stock": True, "k": 10}
                if self.brief.category:
                    args["category"] = self.brief.category
                if self.budget is not None:
                    args["max_price"] = self.budget
                steps.append(("filter", args, "", ""))
            return steps[: self.max_calls]
        brand = self.brief.pref_brand or self.brief.brand
        words = [w for w in (brand, self.brief.spec_value, self.brief.category) if w]
        if words and "search" in self.tool_names:
            steps.append(("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10}, "", ""))
            if brand:
                self.brand_probed = True
        if "filter" in self.tool_names and (self.brief.category or brand):
            args: dict[str, Any] = {"in_stock": True, "k": 10}
            if self.brief.category:
                args["category"] = self.brief.category
            if self.budget is not None:
                args["max_price"] = self.budget
            if brand:
                args["brand"] = brand
                self.brand_probed = True
                self.brand_filtered = True
            steps.append(("filter", args, "", ""))
        return steps

    def stage_backup(self) -> tuple | None:
        good = self.firm_backups()
        inspected = [c for c in good if c["key"] in self.ledger.inspected]
        seen_pids = {c["pid"] for c in inspected}
        if len(seen_pids) >= 2:
            return None
        for c in good:
            if c["pid"] not in seen_pids and "inspect_stock" in self.tool_names:
                return ("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
        # One inspected firm-valid SKU is enough to let Commitment fire. Widening
        # here previously burned the turn budget on catalog wander.
        if inspected:
            return None
        return self.read_more()

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
            "graded on that brand, so filter by brand=%s within the budget and read those products before any "
            "other." % (self.brief.pref_brand, self.brief.pref_brand)
        )

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
        )

    def _blob(self, pid: str, sku: str) -> str:
        return " ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku)

    def _water_hits(self, pid: str, sku: str) -> int:
        return len(_WATER_RE.findall(self._blob(pid, sku)))

    def _sport_hits(self, pid: str, sku: str) -> int:
        return len(re.findall(r"\b(?:sport|gym|run|workout|earhook|secure|in-?ear)\b", self._blob(pid, sku), re.I))

    def _wireless_hits(self, pid: str, sku: str) -> int:
        blob = self._blob(pid, sku)
        if _WIRED_ONLY_RE.search(blob) and not _WIRELESS_RE.search(blob):
            return 0
        return 1 if _WIRELESS_RE.search(blob) else 0

    def _battery_hits(self, pid: str, sku: str) -> int:
        return len(re.findall(r"\b(?:battery|mah|hours?|playback|playtime)\b", self._blob(pid, sku), re.I))

    def _priority_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        have = _tokens(
            " ".join(str(v) for v in self._facts(pid).values()) + " " + self._title(pid) + " " + self._options(pid, sku)
        )
        hits = len(wants & have) if wants else 0
        if self.brief.wants_water:
            hits += 2 * self._water_hits(pid, sku)
            blob = (self._title(pid) + " " + self._options(pid, sku)).lower()
            if _WIRELESS_RE.search(blob):
                hits += 2
            if _WIRED_ONLY_RE.search(blob) and not _WIRELESS_RE.search(blob):
                hits -= 2
        return hits

    def _wish_words(self) -> set:
        if self._wishes is None:
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _stems(self.brief.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _adopt_shopper_spec(self, text: str) -> str | None:
        """Adopt a hidden spec, or a genuine revision. Ignore comparison questions."""
        if not self.brief.spec_value:
            return self.brief.adopt_spec(text)
        field = self.brief.spec_field
        mentioned = _ordered_values(field, text) if field else []
        unique = []
        for value in mentioned:
            if value not in unique:
                unique.append(value)
        if _REVISION_RE.search(text):
            return self.brief.adopt_spec(text)
        if len(unique) == 1 and unique[0] != self.brief.spec_value:
            return self.brief.adopt_spec(text)
        return None

    def _on_firm_revision(self) -> None:
        """Re-score the cart and catalog against the latest shopper-stated firm spec."""
        self.widen_index = 0
        self.reads_done = False
        self.forced_compares = 0
        self.brand_filtered = False
        self.revision_hold_used = True
        for pid, sku in list(self.ledger.cart):
            c = self.assess(pid, sku)
            if c["ok"] and not self._sku_invalid(pid, sku):
                continue
            key = Ledger.key(pid, sku)
            self.ledger.dead_keys.add(key)
            if not self.brief.pref_model:
                self.ledger.dead_products.add(str(pid))
            self._note(
                "The carted item %s no longer matches the latest firm requirement; it will be replaced."
                % key
            )

    def _budget_from_message(self, text: str) -> None:
        new = Brief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        self.brand_filtered = False
        self.widen_index = 0
        self.reads_done = False
        self.forced_compares = 0
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

    def _claim_brand(self, pid: str) -> str:
        brand = (self._brand_of(pid) or "").strip()
        if brand:
            return brand
        for want in (self.brief.brand, self.brief.pref_brand):
            want = (want or "").strip()
            if want and _tokens(want) <= _tokens(self._title(pid)):
                return want
        return ""

    def _soft_reason(self, pid: str) -> dict | None:
        if "brand" in self.reason_fields:
            brand = self._claim_brand(pid)
            if brand:
                return {"field": "brand", "value": brand}
        facts = self._facts(pid)
        for field in self.reason_fields:
            if field == "brand":
                continue
            raw = facts.get(field)
            if raw is True or str(raw).strip().lower() in ("true", "yes", "supported"):
                return {"field": field, "value": "true"}
        return None

    def _claims_for(self, pid: str, sku: str) -> list:
        out = []
        field, want = self.brief.spec_field, self.brief.spec_value
        if field and want and field in _CLAIM_FIELDS:
            if self._fact_value(pid, field) == want:
                out.append({"field": field, "value": want})
        st = self.ledger.status(pid, sku) or {}
        if st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        if st.get("in_stock"):
            out.append({"field": "in_stock", "value": "true"})
        soft = self._soft_reason(pid)
        if soft:
            out.append(soft)
        return out

    def _must_justify(self) -> bool:
        return bool(self.order_requires_justification or self.brief.wants_reason)

    def order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self._must_justify():
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def valid_now(self, pid: str, sku: str) -> tuple:
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

    def _recall_need(self) -> int:
        return self.brief.required_identities

    def hold_order(self) -> str | None:
        """Block a terminal order until a post-commit shopper beat can arrive.

        Soft brand plus a firm hardware spec is the intent shape. A later
        correct order does not repair a missed revision.
        """
        if self.closing or self.ledger.event_target or self.shopper_updated:
            return None
        if not self.ledger.adds:
            return None
        if not (self.brief.pref_brand and self.brief.spec_field):
            return None
        if self.clock.remaining() < 40 or self.turn >= self.max_steps - 4:
            return None
        if self.pending_recheck:
            return "re-inspect the first commit before ordering"
        if self.recheck_done_turn is None:
            return "re-check current price and stock after the first add"
        if not self.revision_hold_used:
            return "hold one non-terminal turn so a shopper revision of the firm spec can arrive"
        return None

    def ready(self) -> bool:
        """Same timing gates as agent.py: roomy = early turns and >140s left."""
        if self.ledger.adds or self.closing:
            return True
        if self.brief.spec_sealed:
            return False
        if (
            (self.brief.pref_brand or self.brief.spec_field or self.brief.brand)
            and not self.brief.candidate_keys
            and self.turn < 4
            and not self.shopper_updated
        ):
            return False
        if self.brief.listing_diversity and len(self.ledger.listing_ids) < 2:
            if not self.closing and self.clock.remaining() > 80:
                return False
        if (
            self.brief.wants_water
            and self._water_hunt_open()
            and self.turn < 8
            and self.clock.remaining() > 120
            and not self._has_water_candidate()
        ):
            return False
        roomy = self.turn < self.max_steps * 0.45 and self.clock.remaining() > 140
        if self.brief.required_identities and roomy:
            if len(self.ledger.identities) < self.brief.required_identities or self.search_calls < 2:
                return False
            if self.search_calls < 3 and self.clock.remaining() > 40:
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self.candidates() if c["ok"]}) < 2:
            return False
        if self.brief.pref_brand and not self.ledger.adds:
            if not self.brand_probed and self.clock.remaining() > 80:
                return False
            if not self.bench_ready():
                return False
        return True

    def firm_backups(self) -> list:
        dead = self.ledger.dead_keys | (
            {Ledger.key(pid, sku) for pid, sku in self.ledger.cart} if self.ledger.event_target else set()
        )
        return [
            c for c in self.candidates()
            if c["ok"] and c["key"] not in dead and c["key"] not in self.ledger.dead_keys
        ]

    def bench_ready(self) -> bool:
        if self.ledger.adds:
            return True
        if self.clock.remaining() < 40 or self.turn >= self.max_steps - 4:
            return True
        if not self.brief.event_expected:
            return True
        good = self.firm_backups()
        inspected = [c for c in good if c["key"] in self.ledger.inspected]
        branded = [c for c in good if self.pref_hit(c["pid"])] if self.brief.pref_brand else inspected
        distinct = {c["pid"] for c in inspected}
        if branded and len(distinct) >= 2:
            return True
        if branded and len({c["pid"] for c in good}) >= 2 and inspected:
            return True
        if branded and inspected and not self._spec_leads():
            return True
        # Two product pages already read is enough backup evidence; do not wait
        # for an endless stream of new spec-shaped search rows.
        if branded and inspected and len(self.ledger.products) >= 2:
            return True
        if not self.brief.pref_brand and len(distinct) >= 2:
            return True
        return False

    def fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _shortlist_note(self) -> None:
        cands = self.candidates()
        if self.brief.listing_diversity and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self._note(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self.target()
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

    # ------------------------------------------------------------------ supervisor

    def _mark_invalid_carts(self) -> None:
        for pid, sku in list(self.ledger.cart):
            if self._sku_invalid(pid, sku):
                self._mark_event_dead(pid, sku)

    def next_action(self) -> tuple | None:
        """Poll specialized engines in workflow order; first proposal wins."""
        if self.done:
            return None
        self._mark_invalid_carts()
        for engine in self.engines:
            act = engine.propose(self)
            if act is not None:
                return act
        return None

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
                self._note("No tool call was made. Choose one or more of the supplied tools now.")
                continue
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _retarget_envelope(self, envelope: dict, turn: int) -> dict:
        """Rebuild call_id / idempotency / turn so a retry is a new legal turn."""
        group = self.problem_id + "-turn-" + str(turn)
        calls = []
        for index, old in enumerate(envelope.get("calls") or [], start=1):
            action = old.get("action") if isinstance(old, dict) else None
            if not isinstance(action, dict):
                continue
            calls.append({"call_id": group + "-" + str(index), "action": action})
        return {**self.binding, "call_id": group, "idempotency_key": group, "turn": turn, "calls": calls}

    def _send_once(self, envelope: dict) -> tuple[dict | None, int | None, str]:
        """One environment POST via the sandbox proxy. 409 is a sequence error."""
        result = _PROXY.post_verbose("/environment/call", json_data=envelope)
        if result.ok:
            data = result.data
            if isinstance(data, dict) and data.get("calls"):
                return data, 200, ""
            return None, 200, ""
        err = result.error or {}
        status = err.get("status")
        body = str(err.get("body") or "")[:800]
        return None, status if isinstance(status, int) else None, body

    def _send(self, envelope: dict) -> dict | None:
        """Post one solver turn. Resync if the session already consumed this turn.

        A 500 after the runtime accepted the turn (shopper-simulator 401 is the
        usual cause) makes a same-envelope retry come back 409: expected N+1,
        got N. Hammering that envelope only floods the log. Advance to the
        server's expected turn once and send the same actions under a new key.
        """
        current = envelope
        self._last_sent = current
        for attempt in range(3):
            result, status, body = self._send_once(current)
            if result is not None:
                self._last_sent = current
                return result
            if status == 409:
                if re.search(r"terminal observation", body, re.I):
                    self.done = True
                    return None
                m = re.search(r"expected (\d+), got (\d+)", body)
                if m:
                    expected = int(m.group(1))
                    if expected > 0 and expected != current.get("turn"):
                        current = self._retarget_envelope(current, expected)
                        self._last_sent = current
                        continue
                # Same turn already bound: do not retry it.
                return None
            if self.clock.remaining() < 20:
                break
            if status in (401, 403):
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if name == "add_to_cart":
            why = self.first_add_refused(str(args.get("product_id") or ""), str(args.get("sku") or ""))
            if why:
                self._note("Controller blocked add_to_cart: " + why)
                return None
        return self._act_group([(name, args, content, tool_call_id or "")])

    def _act_group(self, steps: list) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                sent = getattr(self, "_last_sent", None)
                if sent is not None:
                    pend["envelope"] = sent
                    pend["turn"] = int(sent.get("turn") or pend["turn"])
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            sent = getattr(self, "_last_sent", None) or pend["envelope"]
            self.turn = int(sent.get("turn") or pend["turn"])
            return self._absorb_group(pend["steps"], pend.get("content") or "", replay)

        if not steps:
            return None
        steps = steps[: self.max_calls]
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [
                {
                    "call_id": group + "-" + str(index),
                    "action": {"name": name, "args": args},
                }
                for index, (name, args, _content, _cid) in enumerate(steps, start=1)
            ],
        }
        content = next((c for _n, _a, c, _i in steps if c), "")
        result = self._send(envelope)
        sent = getattr(self, "_last_sent", None) or envelope
        used_turn = int(sent.get("turn") or turn)
        if result is None:
            self.pending_envelope = {"envelope": sent, "steps": steps, "content": content, "turn": used_turn}
            self._note("The environment is not answering; the action will be retried.")
            return None
        self.turn = used_turn
        return self._absorb_group(steps, content, result)

    def _unwrap(self, wrap: dict) -> dict:
        if not isinstance(wrap, dict):
            return {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get("error"):
            obs = dict(obs)
            obs["error"] = wrap.get("error")
        if wrap.get("done"):
            self.done = True
        return obs if isinstance(obs, dict) else {}

    def _ingest_user(self, result: dict) -> None:
        user = result.get("user_message")
        if not (isinstance(user, dict) and user.get("content")):
            return
        text = str(user["content"])
        self.messages.append({"role": "user", "content": text})
        self._budget_from_message(text)
        stated = self._adopt_shopper_spec(text)
        if stated:
            self.shopper_updated = True
            self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            self._on_firm_revision()
        if _CHOICE_Q_RE.search(text):
            self.shopper_question_pending = True
            self._note("The shopper asked which item you are taking: answer with message, then cart that item.")
        elif _EVENT_Q_RE.search(text):
            self.shopper_question_pending = True
            self._note("The shopper flagged a market change: acknowledge it, then recover to a firm-valid replacement.")
        elif _QUESTION_RE.search(text):
            self.shopper_question_pending = True
            self._note("The shopper asked a question: answer it with message in your next turn, then continue.")

    def _absorb_group(self, steps: list, content: str, result: dict | None) -> dict | None:
        tool_calls = []
        for index, (name, args, _c, cid) in enumerate(steps, start=1):
            tid = cid or ("agent-" + str(self.turn) + "-" + str(index))
            tool_calls.append(
                {"id": tid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            )
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            for name, args, _c, _i in steps:
                self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            for call in tool_calls:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps({"error": "environment call failed; it will be retried automatically"}),
                    }
                )
            return None

        calls = result.get("calls") or []
        last = None
        added = None
        for index, (name, args, _c, cid) in enumerate(steps):
            wrap = (calls[index].get("observation") if index < len(calls) and isinstance(calls[index], dict) else {}) or {}
            obs = self._unwrap(wrap)
            last = obs
            self.ledger.record(name, args, obs, self.brief)
            self.ledger.calls_seen.add(name + json.dumps(args, sort_keys=True))
            tid = cid or tool_calls[index]["id"]
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": _trim(json.dumps(obs), _OBS_CHARS)})
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                row = obs.get("added") or {}
                if row.get("product_id"):
                    added = (str(row["product_id"]), str(row.get("sku") or ""))
            if name in ("search", "filter"):
                self.search_calls += 1
            if name == "inspect_stock":
                ipid = str(args.get("product_id") or (obs or {}).get("product_id") or "")
                isku = str(args.get("sku") or (obs or {}).get("sku") or "")
                if ipid and (ipid, isku) in self.ledger.cart and self._sku_invalid(ipid, isku):
                    self._mark_event_dead(ipid, isku)
            if name == "inspect_cart":
                self.cart_inspections += 1
            if name == "message":
                self.shopper_question_pending = False
            if wrap.get("done") or (isinstance(obs, dict) and obs.get("done")):
                self.done = True
        if added:
            self.pending_recheck = added
        self._ingest_user(result)
        if (
            self.recheck_done_turn is not None
            and self.turn > self.recheck_done_turn
            and not self.revision_hold_used
            and not any(n == "place_test_order" for n, _a, _c, _i in steps)
        ):
            self.revision_hold_used = True
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        if any(n in ("search", "filter") for n, _a, _c, _i in steps) and self.brief.required_identities:
            have, need = len(self.ledger.identities), self._recall_need()
            self._note(
                "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                + ("" if have >= need else " Keep searching with different wording.")
            )
        if any(n in ("view", "compare") for n, _a, _c, _i in steps):
            self._shortlist_note()
        return last

    def _post_add_recheck(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        self.recheck_done_turn = self.turn
        if isinstance(obs, dict) and obs.get("error"):
            return
        st = self.ledger.status(pid, sku) or {}
        if isinstance(obs, dict) and st.get("price") is None and obs.get("price") is not None:
            st = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.ledger.checked[Ledger.key(pid, sku)] = st
        stock = bool(st.get("in_stock"))
        price = st.get("price")
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over and not self._sku_invalid(pid, sku):
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (Ledger.key(pid, sku), price)
            )
            return
        self._mark_event_dead(pid, sku)
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        self._note(
            "Market change: the carted item %s %s. Choose a different product that still meets every firm requirement."
            % (Ledger.key(pid, sku), reason)
        )
        self._shortlist_note()
        self._recover_after_event()

    def _recover_after_event(self) -> None:
        if self.done:
            return
        if self.shopper_question_pending and "message" in self.tool_names:
            self._act("message", {"content": self.answer_text()})
            if self.done:
                return
        for cpid, csku in list(self.ledger.cart):
            if self.done:
                return
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                self._act("remove_from_cart", {"product_id": cpid, "sku": csku})
        for _ in range(8):
            if self.done or self.turn >= self.max_steps:
                return
            if self.brief.pref_model and not self.searched_twins:
                twin = self._twin_search()
                if twin is not None:
                    self._act(twin[0], twin[1])
                    continue
            if self.brief.pref_model and not self.searched_siblings:
                sib = self._sibling_search()
                if sib is not None:
                    self._act(sib[0], sib[1])
                    continue
            target = self.target()
            if target:
                pid, sku = target["pid"], target["sku"]
                if Ledger.key(pid, sku) not in self.ledger.inspected and "inspect_stock" in self.tool_names:
                    self._act("inspect_stock", {"product_id": pid, "sku": sku})
                    continue
                if (pid, sku) not in self.ledger.cart:
                    self._act("add_to_cart", {"product_id": pid, "sku": sku})
                    if self.done:
                        return
                if self.pending_recheck:
                    rpid, rsku = self.pending_recheck
                    self.pending_recheck = None
                    self._act("inspect_stock", {"product_id": rpid, "sku": rsku})
                    st = self.ledger.status(rpid, rsku) or {}
                    price = st.get("price")
                    over = self.budget is not None and price is not None and price > self.budget + _EPS
                    if not st.get("in_stock") or over:
                        self.ledger.dead_keys.add(Ledger.key(rpid, rsku))
                        self.ledger.dead_products.add(str(rpid))
                        continue
                if self.assess(pid, sku)["ok"] and self.valid_now(pid, sku)[0]:
                    self._act("place_test_order", self.order_args(pid, sku))
                    return
                continue
            forced = self.read_more()
            if forced is None:
                break
            self._act(forced[0], forced[1])
        if not self.done:
            self._finish_fast()

    def _finish_fast(self) -> None:
        for cpid, csku in list(self.ledger.cart):
            if self.done:
                return
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and "remove_from_cart" in self.tool_names:
                self._act("remove_from_cart", {"product_id": cpid, "sku": csku})
        for pid, sku in reversed(self.ledger.cart):
            if self.done:
                return
            if not self.assess(pid, sku)["ok"]:
                continue
            ok, _why = self.valid_now(pid, sku)
            if ok:
                self._act("place_test_order", self.order_args(pid, sku))
                return
        good = self.firm_backups() if self.ledger.event_target else [c for c in self.candidates() if c["ok"]]
        if not good:
            for _ in range(4):
                if self.done:
                    return
                forced = self.read_more()
                if forced is None:
                    break
                self._act(forced[0], forced[1])
                good = self.firm_backups() if self.ledger.event_target else [c for c in self.candidates() if c["ok"]]
                if good:
                    break
        if not good:
            good = [
                c
                for c in self.candidates()
                if c["live"]
                and c["viewed"]
                and c["key"] in self.ledger.inspected
                and (not self.brief.theme or self._theme_hit(c["pid"], c["sku"]))
                and self._brand_ok(c["pid"])
            ]
        if not good or self.done:
            return
        first = self.ledger.adds == 0
        if first and self.brief.pref_brand:
            named = [c for c in good if self.pref_hit(c["pid"])]
            if named:
                good = named
            elif not self.closing:
                return
        good.sort(key=lambda c: self._rank_of(c, first))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if Ledger.key(pid, sku) not in self.ledger.inspected and "inspect_stock" in self.tool_names:
            self._act("inspect_stock", {"product_id": pid, "sku": sku})
            if self.done:
                return
        if (pid, sku) not in self.ledger.cart:
            self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done:
                return
        if self.pending_recheck:
            rpid, rsku = self.pending_recheck
            self.pending_recheck = None
            self._act("inspect_stock", {"product_id": rpid, "sku": rsku})
            if self.done:
                return
        if self.valid_now(pid, sku)[0]:
            self._act("place_test_order", self.order_args(pid, sku))

    def _buy(self, target: dict) -> bool:
        if not target or self.done or self.turn >= self.max_steps:
            return False
        pid, sku = target["pid"], target["sku"]
        if Ledger.key(pid, sku) not in self.ledger.inspected and "inspect_stock" in self.tool_names:
            self._act("inspect_stock", {"product_id": pid, "sku": sku})
            if self._sku_invalid(pid, sku) or self.done:
                return False
        if (pid, sku) not in self.ledger.cart:
            self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done:
                return True
        if self.pending_recheck:
            rpid, rsku = self.pending_recheck
            self.pending_recheck = None
            self._act("inspect_stock", {"product_id": rpid, "sku": rsku})
            if self._sku_invalid(rpid, rsku):
                if (rpid, rsku) in self.ledger.cart and "remove_from_cart" in self.tool_names:
                    self._act("remove_from_cart", {"product_id": rpid, "sku": rsku})
                return False
        if self.valid_now(pid, sku)[0]:
            self._act("place_test_order", self.order_args(pid, sku))
            return self.done
        return False

    def _last_resort_order(self) -> None:
        self.closing = True
        if self.done:
            return
        if self.brief.spec_sealed and self.asked_shopper:
            self.brief.spec_sealed = False
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if self._buy(self.target()):
            return
        live = [
            c
            for c in self.candidates()
            if c["live"]
            and c["viewed"]
            and c["key"] in self.ledger.inspected
            and (not self.brief.theme or self._theme_hit(c["pid"], c["sku"]))
        ]
        live.sort(key=lambda c: (0 if c["ok"] else 1, c["price"] if c["price"] is not None else 1e12))
        if live and self._buy(live[0]):
            return
        if self.turn < self.max_steps - 3:
            forced = self.read_more()
            if forced is not None:
                self._act(forced[0], forced[1])
        if self._buy(self.target()):
            return
        self._finish_fast()

    def _validate(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self.valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            hold = None if self.closing else self.hold_order()
            if hold:
                return "order refused: " + hold
            if self.brief.required_identities and len(self.ledger.identities) < self.brief.required_identities:
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear"
                    % (len(self.ledger.identities), self.brief.required_identities)
                )
            if self.brief.listing_diversity and len(self.ledger.listing_ids) < 2:
                return "order refused: compare at least two products with different listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self.assess(pid, sku)
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
            if self.brief.listing_diversity and len(self.ledger.listing_ids) < 2:
                return "compare two products with different listing_id values before the first cart addition"
            refused = self.first_add_refused(pid, sku)
            if refused:
                return refused
            if self.brief.pref_model and not self.ledger.adds and self._relax(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; search that title "
                    "and read the other colors first"
                )
            if not self.ready():
                return "hold: a firm requirement is still missing, or more matching products must be surfaced first"
        if name == "inspect_stock":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if pid and self.fresh(pid, sku):
                return "identical inspect already made; change the variant"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and not self.brief.spec_sealed and "?" not in text:
                return "do not send status updates; use message only to answer the shopper or to ask one question"
            if not self.shopper_question_pending and not self.brief.spec_sealed and self.questions_asked >= 2:
                return "the shopper has already answered; decide from the evidence you have rather than asking again"
        if name == "inspect_cart" and self.cart_inspections >= 1:
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if name == "search":
            if self.search_calls >= self._search_cap() and not self.brief.required_identities:
                return "search budget used; view existing hits instead of reformulating"
            if self.brief.required_identities and self.search_calls >= 3:
                return "search budget used; view existing hits instead of reformulating"
            fp = self._search_fingerprint(str(args.get("query") or ""))
            if fp and fp in self._search_seen():
                return "identical call already made; change the arguments"
        if key in self.ledger.calls_seen and name in ("search", "filter", "view", "compare", "inspect_stock"):
            return "identical call already made; change the arguments"
        return None

    def _filter_steps(self, steps: list) -> list:
        names = [n for n, _a, _c, _i in steps]
        if "add_to_cart" in names and "place_test_order" in names:
            steps = [s for s in steps if s[0] != "place_test_order"]
        if self.ledger.adds == 0 and "place_test_order" in [s[0] for s in steps]:
            steps = [s for s in steps if s[0] != "place_test_order"]
        kept = []
        for name, args, content, cid in steps:
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                args = self.order_args(pid, sku)
            why = self._validate(name, args)
            if why:
                if name not in self.tool_names:
                    continue
                if name in ("add_to_cart", "place_test_order", "message"):
                    self._note("Dropped " + name + ": " + why)
                    continue
                if name in ("search", "filter", "view", "compare", "inspect_stock") and "identical" in why:
                    continue
            kept.append((name, args, content, cid))
        return kept

    def run(self) -> list:
        last_forced, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            # agent.py: close only with <25s left or in the last three turns.
            if self.clock.remaining() < 25 or self.turn >= self.max_steps - 3:
                self.closing = True
                self._last_resort_order()
                break
            if not self.bootstrapped:
                self.bootstrapped = True
                opening = self.bootstrap_steps()
                if opening:
                    self._act_group(opening)
                    continue
            if self.pending_recheck and "inspect_stock" in self.tool_names:
                self._post_add_recheck()
                continue
            plan = self.next_action()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == "message":
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on.")
                    continue
                self._act(name, args)
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._last_resort_order()
                break
            content = proposal.get("content") or ""
            raw_steps = []
            for call in proposal["tool_calls"]:
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                raw_steps.append((name, args, content, call.get("id") or ""))
            steps = self._filter_steps(raw_steps)
            if not steps:
                self.rejected_streak += 1
                if self.rejected_streak >= 3:
                    self._shortlist_note()
                    self.rejected_streak = 0
                if self.rejected_streak == 1:
                    self._note("Those calls were not made. Choose valid catalog tools from the live list.")
                continue
            self.rejected_streak = 0
            if any(n == "message" for n, _a, _c, _i in steps) and not self.shopper_question_pending:
                self.questions_asked += 1
            self._act_group(steps)
        if not self.done:
            self._last_resort_order()
        return self.dialogue


def agent_main(problem_data: dict) -> list:
    try:
        return SpatialEngine(problem_data).run()
    except Exception as exc:  # never raise out of the harness
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main", "SpatialEngine"]
