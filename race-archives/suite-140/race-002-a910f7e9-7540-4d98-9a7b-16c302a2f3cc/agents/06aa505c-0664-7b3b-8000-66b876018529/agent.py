"""ORO-Bench supervisor: one file, seven score contracts.

The kernel owns session I/O, the catalog index, and every graded action.
A structural router picks a family policy from the public query. The model
writes shopper text and may explore unread pages; it never emits the first
add or the terminal order.
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
_CHAT = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_ENV = ProxyClient(timeout=110, max_retries=1, retry_delay=1.0)
_WALL_S = 270.0
_CLIP = 7000
_EPS = 1e-9

_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RX = re.compile(
    r"(?:at or under|at or below|up to|no more than|not more than|max(?:imum)? of|"
    r"budget (?:is|of)|only go up to|can spend up to|at most|spend up to)\s*"
    + _NUM
    + r"(?:\s*(PHP|php|₱|pesos?|USD|usd))?",
    re.I,
)
_KEY_RX = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTE_RX = re.compile(r"\"([^\"]{2,80})\"")
_PHRASE_RX = re.compile(r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,", re.I)
_SURFACE_RX = re.compile(
    r"(?:surface|find|cover|survey|return|need)\s+(?:at least\s+)?(\d+)\s+distinct",
    re.I,
)
_HIDDEN_RX = re.compile(
    r"hidden (?:firm )?requirement|tell you if you ask|if you ask|"
    r"one firm requirement|answers? (?:it|that) when asked|"
    r"ask(?:s|ed)? (?:the )?(?:shopper|user).{0,60}requirement|"
    r"reveals? (?:it|the (?:spec|requirement))",
    re.I,
)
_WATER_RX = re.compile(r"\b(?:ipx?\s*\d|waterproof|water[\s-]?resist|sweat|splash)\b", re.I)
_WATER_CUE = re.compile(r"\b(?:water|sweat|ipx|waterproof|weather|workout|gym|run)\b", re.I)
_PREF_ITEM_RX = re.compile(
    r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I
)
_PREF_BRAND_RX = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_CAT_RX = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SPEC_WORD_RX = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_REASON_RX = re.compile(
    r"\b(?:why|explain why|tell me why|reason i can trust|trustworthy reason)\b", re.I
)
_SOFT_RX = re.compile(r"nice[\s-]?to[\s-]?have|soft preference|preferred brand", re.I)
_INTENT_RX = re.compile(
    r"ideally\b.{0,48}\bprefer|"
    r"\bprefer(?:ably)?\b.{0,80}\b(?:but|however)\b|"
    r"firm requirements? matter|"
    r"requirements? matter more",
    re.I,
)
_QUALITY_RX = re.compile(
    r"(?:not|isn't|is not|don'?t)\s+(?:just\s+)?(?:grab |take )?(?:the\s+)?cheap|"
    r"prefer\b.{0,80}\b(?:over|than)\b.{0,40}\bcheap|"
    r"\b(?:priorit(?:y|ies|ise|ize)|matters more|care(?:s)? more)\b|"
    r"\brather (?:pay|than (?:price|cost))\b|"
    r"\bflimsy\b",
    re.I,
)
_REVISE_RX = re.compile(
    r"\b(?:instead|actually|change[ds]?|update[d]?|now (?:need|want|require)|"
    r"rather|switch(?:ed)? to|new (?:spec|requirement|size|value)|no longer)\b",
    re.I,
)
_EVENT_RULE_RX = re.compile(
    r"price(?:s)?\s+(?:can|may|might|will)\s+change|re-?check|verify\s+(?:the\s+)?(?:current\s+|final\s+)?price|repric",
    re.I,
)
_IDENTITY_RX = re.compile(
    r"\b(?:genuine|actual)\s+(?:device|item|product)\b|"
    r"\bnot (?:a[n]? )?(?:accessor|case|part|replacement|feature phone)\b",
    re.I,
)
_ASK_RX = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_CHOICE_RX = re.compile(
    r"which one are you going|does it actually have|before you buy|what are you (?:picking|choosing)",
    re.I,
)
_EVENT_ASK_RX = re.compile(
    r"more expensive|out of stock|re-?check before ordering|just get more|price (?:just )?(?:changed|increased|jumped)"
    r"|still available",
    re.I,
)
_ACCESSORY_RX = re.compile(
    r"\b(?:cases?|covers?|casing|tempered|glass|charger|cable|dummy|trigger|stand|protector|"
    r"bumper|replacement|skin|holster|pouch|lanyard|bracket|armbands?|arm\s*bags?|"
    r"(?:phone|cellphone|mobile)\s+holders?|holders?|cleaning\s+kit|lens|controller|"
    r"clamp|wooden|menu|nametag|paper\s+stand|dust\s+removal|macro)\b",
    re.I,
)
_RANGE_INCH_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-~–to]+\s*(\d+(?:\.\d+)?)\s*-?\s*(?:inch(?:es)?|in)\b", re.I
)
_JUNK_RX = re.compile(
    r"\b(?:plush|toys?|dolls?|griddle|pans?\b|camshaft|notebook|backpack|bracelet|"
    r"sketch|ear\s+tag|sizzling|insect|reel|pillow|necklace|bulb|yoga|polo|shirt|"
    r"cotton|nametag|magnets?|tumbler|ceramic|cam\b|filaments?|"
    r"chairs?|desks?|sofas?|stools?|furniture|aeron)\b",
    re.I,
)
_TABLET_RX = re.compile(r"\b(?:tablets?|ipads?)\b", re.I)
_HEADPHONE_RX = re.compile(r"\b(?:headphones?|headsets?|earbuds?|earphones?)\b", re.I)
_MONITOR_RX = re.compile(r"\b(?:monitors?|displays?)\b", re.I)
_WIRED_RX = re.compile(r"\bwired\b|3\.5\s*mm|\baux\b", re.I)
_WIRELESS_RX = re.compile(r"wireless|bluetooth|\btws\b|ipx|sport|sweat", re.I)
_INCH_RX = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RX = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_WXH_RX = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_PAIR_RX = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I
)
_PAIR_HINT_RX = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I
)
_MEM_RX = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QHD_FALSE = re.compile(r"\bqHD\b")
_OS_RX = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|"
    r"hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_PHONE_RX = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_PHONE_RX = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?|"
    r"armbands?|arm\s*bags?|(?:phone|cellphone|mobile)\s+holders?|"
    r"dummy\s+phone|display\s+model|lcd\s+display)\b",
    re.I,
)
_PART_RX = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\blcd\s+display\b|\bdisplay\s+to\b|\bdummy\s+phone\b",
    re.I,
)
_NEGATE_RX = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_BUNDLE_RX = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)

_CLAIM_FIELDS = ("resolution", "refresh_rate", "screen_size", "price", "in_stock")
_SOFT_FIELDS = ("brand", "streaming", "microphone", "privacy_cover", "autofocus", "hdr")
_AXES = (
    ("water", re.compile(r"\b(?:ipx?\s*\d|waterproof|water[\s-]?resist|sweat|splash|weather)\b", re.I)),
    ("wireless", re.compile(r"\b(?:wireless|bluetooth|\btws\b|cordless)\b", re.I)),
    ("battery", re.compile(r"\b(?:battery(?:\s+life)?|runtime|long(?:er)?\s+charge)\b", re.I)),
    ("sport", re.compile(r"\b(?:sport|gym|workouts?|runs?|running|fitness)\b", re.I)),
)
_AXIS_Q = {
    "water": ("waterproof", "ipx", "sweat"),
    "wireless": ("wireless", "bluetooth"),
    "battery": ("battery",),
    "sport": ("sport",),
}
_FACT_LABELS = {
    "screen_size": ("size", "screen_size", "display_size"),
    "resolution": ("display", "resolution"),
    "refresh_rate": ("refresh_rate", "refresh"),
    "ram": ("ram", "memory", "device_memory", "system_memory"),
}
_SHELF = {
    "smartphone": {"phone", "cellphone", "mobile", "handset", "unlocked"},
    "phone": {"smartphone", "cellphone", "mobile", "handset"},
    "headphone": {"headset", "earbud", "earphone"},
    "headset": {"headphone", "earbud", "earphone"},
    "monitor": {"display", "screen"},
    "display": {"monitor", "screen"},
}
_STOP = {
    "in", "stock", "now", "available", "today", "only", "php", "the", "a", "an", "and",
    "or", "for", "with", "of", "to", "on", "at", "is", "it", "this", "that",
}

_GUIDE = (
    "Shop for the user with the live tools only. Batch independent catalog reads "
    "in one turn. Firm gates are category, hardware spec, budget, stock, and a "
    "real device. Soft gates (preferred brand, nicer extras) never override a "
    "firm miss. The first add_to_cart is the scored commitment: view facts and "
    "inspect_stock that exact SKU first. After any add, re-inspect; if price or "
    "stock dies, recover to a different still-valid product. Answer a shopper "
    "question with message before ordering. When place_test_order requires "
    "justification, send field/value claims only."
)


def _inches(text: str) -> set:
    text = text or ""
    vals = {"%gin" % float(m.group(1)) for m in _INCH_RX.finditer(text)}
    for m in _RANGE_INCH_RX.finditer(text):
        vals.discard("%gin" % float(m.group(1)))
        vals.discard("%gin" % float(m.group(2)))
    return vals


def _hz(text: str) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RX.finditer(text or "")}


def _gb(text: str) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RX.finditer(text or "")
    }


def _pixel(long_side: int, short_side: int) -> str | None:
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def _res(text: str) -> set:
    text = text or ""
    if _QHD_FALSE.search(text):
        return set()
    low = text.lower()
    out = set()
    for m in _WXH_RX.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _pixel(pair[0], pair[1])
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


_READERS = {"screen_size": _inches, "refresh_rate": _hz, "resolution": _res, "ram": _gb}


def _adjacent(field: str, text: str) -> set:
    text = text or ""
    if field != "ram":
        return _READERS[field](text)
    paired = {"%ggb" % float(a) for a, _b in _PAIR_RX.findall(text)}
    return paired or {"%ggb" % float(n) for n in _MEM_RX.findall(text)}


def _mentions(field: str, text: str) -> bool:
    text = text or ""
    if field == "ram":
        return bool(_PAIR_HINT_RX.search(text) or _MEM_RX.search(text))
    if field == "screen_size":
        return bool(_INCH_RX.search(text))
    if field == "refresh_rate":
        return bool(_HZ_RX.search(text))
    return bool(
        _WXH_RX.search(text)
        or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I)
    )


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _wanted(field: str | None, text: str) -> str | None:
    if not field:
        return None
    return _one(_READERS[field](text or ""))


def _ordered(field: str, text: str) -> list[str]:
    text = text or ""
    if field == "screen_size":
        return ["%gin" % float(m.group(1)) for m in _INCH_RX.finditer(text)]
    if field == "refresh_rate":
        return ["%ghz" % float(m.group(1)) for m in _HZ_RX.finditer(text)]
    if field == "ram":
        return [
            "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
            for m in _GB_RX.finditer(text)
        ]
    if field == "resolution":
        out: list[str] = []
        for m in re.finditer(
            r"\b(?:ultra\s*hd|full\s*hd|uhd|qhd|fhd|4k|2k|2160p|1440p|1080p|720p|hd)\b|\d{3,5}\s*[x×]\s*\d{3,5}",
            text,
            re.I,
        ):
            for item in _res(m.group(0)):
                if item not in out:
                    out.append(item)
        return out
    return []


def _revised(field: str | None, text: str, current: str | None) -> str | None:
    if not field:
        return None
    unique: list[str] = []
    for value in _ordered(field, text):
        if value not in unique:
            unique.append(value)
    if current and current in unique and len(unique) > 1:
        unique = [value for value in unique if value != current]
    if len(unique) == 1:
        return unique[0]
    if unique:
        return unique[-1]
    return _wanted(field, text)


def _tok(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _sku_stem(sku: str) -> str:
    text = str(sku or "")
    return text.rsplit("-", 1)[0] if "-" in text else text


def _same(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def _stems(text: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}


def _id_text(text: str, title: bool = False) -> str:
    text = _NEGATE_RX.sub("", text or "")
    return _BUNDLE_RX.split(text, maxsplit=1)[0] if title else text


def _phone_ok(facts: dict, title: str, options: str) -> str:
    if _PART_RX.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _id_text(raw_title, title=True)
    kind = " ".join(
        _id_text(str(facts.get(k) or ""))
        for k in ("phone_type", "device_type", "description_device_type")
    )
    blob = head + " " + kind
    if _NOT_PHONE_RX.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _PHONE_RX.search(blob):
        return "yes"
    os_blob = " ".join(
        _id_text(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RX.search(os_blob):
        return "yes"
    if _OS_RX.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return "yes"
    return "unclear"


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _args(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + "...[truncated]"


def _noun(category: str | None) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", (category or "").lower()) if w]
    return words[-1] if words else ""


def _aliases(noun: str) -> set:
    stem = noun.rstrip("s")
    extra = set()
    for key, aliases in _SHELF.items():
        if stem == key or stem in aliases:
            extra |= {key} | aliases
    return {stem} | extra


def _vid(pid: Any, sku: Any) -> str:
    return str(pid) + "::" + str(sku or "")


# ---------------------------------------------------------------- query + router


class Query:
    """Deterministic parse of policy_view.query. No pack IDs."""

    def __init__(self, raw: str) -> None:
        self.raw = raw or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        self.currency = "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        if m and m.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            self.currency = m.group(1)
        self.keys = [(pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RX.findall(self.goal)]
        phrase = _PHRASE_RX.search(self.goal)
        quoted = _QUOTE_RX.findall(self.goal)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE_RX.search(self.raw)
        self.need_ids = int(sm.group(1)) if sm else 0
        pm = _PREF_ITEM_RX.search(self.goal)
        self.color = pm.group(1).lower() if pm else None
        self.model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.quals = self._quals(self.goal)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = (None, None)
        for field, value in self.specs.items():
            self.spec_field, self.spec_value = field, value
            break
        self.sealed = self.spec_field is None and bool(_HIDDEN_RX.search(self.raw))
        self.axes = tuple(name for name, rx in _AXES if rx.search(self.goal))
        if "battery" not in self.axes and re.search(r"\bbattery\b", self.goal + " " + self.rules, re.I):
            self.axes = self.axes + ("battery",)
        self.wants_water = "water" in self.axes or bool(_WATER_CUE.search(self.goal))
        self.wants_reason = bool(_REASON_RX.search(self.goal))
        self.soft = bool(_SOFT_RX.search(self.rules) or _SOFT_RX.search(self.goal))
        self.intent_cue = bool(_INTENT_RX.search(self.goal))
        self.listing = bool(
            (self.wants_water or _QUALITY_RX.search(self.goal) or "listing_id" in self.rules)
            and not self.theme
            and not self.need_ids
            and not self.keys
            and not self.model
        )
        self.require_id = bool(_IDENTITY_RX.search(self.raw))
        blob = (self.goal + " " + self.rules).lower()
        self.rank_fields: list = []
        if self.keys:
            if re.search(r"refresh", blob):
                self.rank_fields.append("refresh_rate")
            if re.search(r"screen|inch", blob):
                self.rank_fields.append("screen_size")
            if re.search(r"\bram\b|memory", blob):
                self.rank_fields.append("ram")
        self.expect_event = bool(
            bool(_EVENT_RULE_RX.search(self.rules) or _EVENT_RULE_RX.search(self.goal))
            or self.pref_brand
            or self.listing
            or self.keys
            or self.model
            or (self.spec_field and not self.sealed)
        )
        self.family, self.confidence = route_family(self)
        if self.family == "preference":
            self.listing = True
            self.expect_event = True
        if self.family == "intent":
            self.expect_event = True
        if self.family == "constraint":
            self.expect_event = True
        if self.family == "recall" and not self.need_ids:
            self.need_ids = 5

    @staticmethod
    def _budget(text: str) -> float | None:
        best = None
        for m in _BUDGET_RX.finditer(text or ""):
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
        m = _CAT_RX.search(goal or "")
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
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPEC_WORD_RX.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res(w)]
        return " ".join(keep) or None

    @staticmethod
    def _pref_brand(goal: str) -> str | None:
        m = _PREF_BRAND_RX.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    @staticmethod
    def _quals(goal: str) -> list:
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
            value = _wanted(field, goal)
            if value:
                out[field] = value
        return out

    def adopt(self, text: str) -> str | None:
        for field in ("screen_size", "refresh_rate", "ram", "resolution"):
            current = self.spec_value if field == self.spec_field else None
            value = _revised(field, text, current)
            if not value:
                continue
            if field == self.spec_field and value == self.spec_value:
                return None
            prev = self.spec_value
            self.spec_field, self.spec_value = field, value
            self.specs[field] = value
            self.sealed = False
            self.expect_event = True
            if prev and prev != value:
                return "%s = %s (was %s)" % (field, value, prev)
            return field + " = " + value
        return None

    def theme_hit(self, *texts: str) -> bool:
        if not self.theme:
            return True
        have: set = set()
        for text in texts:
            have |= _tok(text)
        return all(tok in have for tok in self.theme)

    def axis_terms(self) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for axis in self.axes:
            for word in _AXIS_Q.get(axis, (axis,)):
                key = word.lower()
                if key not in seen:
                    seen.add(key)
                    terms.append(word)
        return terms


def route_family(q: Query) -> tuple[str, float]:
    if q.keys:
        return "ranking", 1.0
    if q.model:
        return "recovery", 1.0
    if q.theme or q.need_ids:
        return "recall", 0.95
    if q.sealed:
        return "constraint", 0.95
    if q.listing:
        return "preference", 0.9
    if q.pref_brand or q.soft or q.intent_cue:
        return "intent", 0.9
    if q.wants_reason:
        return "justification", 0.95
    if q.spec_field or q.brand:
        return "constraint", 0.7
    return "generic", 0.4


# ---------------------------------------------------------------- index + inbox


class CatalogIndex:
    """Every product, SKU, listing, fact, and live quote the tools returned."""

    def __init__(self) -> None:
        self.cart: list = []
        self.quotes: dict = {}
        self.quote_seq: dict = {}
        self.inspected: set = set()
        self.pages: dict = {}
        self.hits: dict = {}
        self.options: dict = {}
        self.listings: set = set()
        self.listing_of: dict = {}
        self.clusters: set = set()
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_key: str | None = None
        self.dead: set = set()
        self.dead_pids: set = set()
        self.seen: set = set()
        self.spec_hits: set = set()

    def ingest(self, name: str, args: dict, obs: dict, query: Query) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            qtext = str(args.get("query") or "").lower()
            token = ""
            if query.spec_value:
                m = re.search(r"(\d+(?:\.\d+)?)", str(query.spec_value))
                token = (m.group(1) if m else str(query.spec_value)).lower()
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._hit(row, query)
                    pid = row.get("product_id")
                    title = str(row.get("title") or "")
                    if (
                        pid
                        and token
                        and token in qtext
                        and not _JUNK_RX.search(title)
                        and not _ACCESSORY_RX.search(title)
                        and not _PART_RX.search(title)
                        and not _NOT_PHONE_RX.search(title)
                    ):
                        self.spec_hits.add(str(pid))
        elif name == "view":
            self._page(obs, query)
        elif name == "compare":
            compared = obs.get("compared")
            if not isinstance(compared, list) and isinstance(obs.get("observation"), dict):
                compared = obs["observation"].get("compared")
            for prod in compared or []:
                if isinstance(prod, dict):
                    self._page(prod, query)
        elif name == "inspect_stock":
            body = obs
            if isinstance(obs.get("observation"), dict) and obs.get("price") is None:
                body = obs["observation"]
            pid = body.get("product_id") or args.get("product_id")
            sku = body.get("sku") or args.get("sku")
            k = _vid(pid, sku)
            self.quotes[k] = {"in_stock": bool(body.get("in_stock")), "price": _num(body.get("price"))}
            self.quote_seq[k] = self.seq
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
            self._cart(obs)
        elif name == "remove_from_cart":
            self._cart(obs)
            removed = obs.get("removed") or {}
            pair = (removed.get("product_id"), removed.get("sku"))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._cart(obs)

    def _cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [(r.get("product_id"), r.get("sku")) for r in rows if isinstance(r, dict) and r.get("product_id")]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = _vid(r.get("product_id"), r.get("sku"))
            price = _num(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.quotes.get(k, {})
            self.quotes[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.quote_seq[k] = self.seq

    def _hit(self, row: dict, query: Query) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if query.theme:
            if query.theme_hit(title, str(row.get("brand") or "")):
                self.clusters.add(str(pid))
        else:
            self.clusters.add(str(pid))
        if not row.get("sku"):
            return
        k = _vid(pid, row.get("sku"))
        self.hits[k] = row
        if row.get("price") is not None:
            self.quotes.setdefault(k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))})

    def _page(self, prod: dict, query: Query) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listings.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        facts = prod.get("facts") or {}
        fact_blob = " ".join(str(v) for v in facts.values()) if isinstance(facts, dict) else ""
        if query.theme and query.theme_hit(str(prod.get("title") or ""), fact_blob, str(prod.get("brand") or "")):
            self.clusters.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = _vid(pid, v.get("sku"))
            self.quotes[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
            self.quote_seq[k] = self.seq
            self.options[k] = str(v.get("options") or "")

    def quote(self, pid: str, sku: str) -> dict | None:
        return self.quotes.get(_vid(pid, sku))


class ShopperInbox:
    """Shopper text, interventions, and market events."""

    def __init__(self) -> None:
        self.pending_q = False
        self.updated = False
        self.asked = False
        self.questions = 0
        self.lines: list[str] = []

    def take(self, result: dict, kernel: "Kernel") -> None:
        event = None
        for wrap in result.get("calls") or []:
            if not isinstance(wrap, dict):
                continue
            obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
            if isinstance(obs, dict) and obs.get("market_event"):
                event = obs.get("market_event")
        if event:
            kernel.note("Market event observed on the last commerce call.")
        user = result.get("user_message")
        if not (isinstance(user, dict) and user.get("content")):
            return
        text = str(user["content"])
        self.lines.append(text)
        kernel.messages.append({"role": "user", "content": text})
        kernel._tighten(text)
        stated = kernel._adopt(text)
        if stated:
            self.updated = True
            kernel.note("Firm requirement recorded: " + stated + ".")
            kernel._on_revision()
        if _CHOICE_RX.search(text):
            self.pending_q = True
            kernel.note("Shopper asked which item: reply with message, then commit.")
        elif _EVENT_ASK_RX.search(text):
            self.pending_q = True
            kernel.note("Shopper flagged a market change: acknowledge, then recover.")
        elif _ASK_RX.search(text):
            self.pending_q = True
            kernel.note("Shopper asked a question: answer with message before ordering.")


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def left(self) -> float:
        return self.budget - (time.monotonic() - self.start)


# ---------------------------------------------------------------- family policies


class Policy:
    name = "generic"

    def inquire(self, k: "Kernel") -> tuple | None:
        if not k.query.sealed or k.inbox.asked or "message" not in k.tools:
            return None
        k.inbox.asked = True
        return (
            "message",
            {
                "content": (
                    "Before I search: which hardware specification is your firm requirement "
                    "(screen size, resolution class, refresh rate, or memory), and what exact value do you need?"
                )
            },
        )

    def ready(self, k: "Kernel") -> bool:
        return True

    def rank(self, k: "Kernel", c: dict, first: bool) -> tuple:
        extra = sum(
            1
            for f, v in k.query.specs.items()
            if f != k.query.spec_field and k.variant_value(c["pid"], c["sku"], f) != v
        )
        proven = 0 if _phone_ok(k.facts(c["pid"]), k.title(c["pid"]), k.options(c["pid"], c["sku"])) == "yes" else 1
        price = c["price"] if c["price"] is not None else 1e12
        grounded = 0
        field, want = k.query.spec_field, k.query.spec_value
        if field and want:
            grounded = 0 if k.fact_value(c["pid"], field) == want or k.variant_value(c["pid"], c["sku"], field) == want else 1
        if _ACCESSORY_RX.search(k.title(c["pid"])):
            proven = 2
        if first and k.query.pref_brand:
            near = 0 if k.pref_hit(c["pid"]) else 1
            return (near, grounded, extra, proven, price, c["key"])
        return (grounded, extra, proven, price, c["key"])

    def first_ok(self, k: "Kernel", pid: str, sku: str) -> str | None:
        if k.index.adds or k.closing:
            return None
        if k.query.pref_brand and not k.pref_hit(pid):
            if k.branded():
                return "first add must be the preferred brand when a compliant one is on record"
            if k.turn < k.max_steps * 0.65 and k.clock.left() > 50:
                return "keep looking for a preferred-brand item that meets every firm gate"
        if k.query.wants_water:
            water = k.water_hits(pid, sku)
            wet = [c for c in k.candidates() if c["ok"] and k.water_hits(c["pid"], c["sku"]) > 0]
            if water <= 0 and wet:
                return "first add must be the sweat/water match, not a cheaper decoy"
            if wet:
                best = max(k.water_hits(c["pid"], c["sku"]) for c in wet)
                if water < best and k.clock.left() > 40:
                    return "a stronger sweat/water-resistant listing is already on record"
        if k.query.spec_field:
            c = k.assess(pid, sku)
            if c.get("source") == "title":
                stronger = [
                    x
                    for x in k.candidates()
                    if x["ok"]
                    and x.get("source") in ("variant", "facts")
                    and (not k.query.pref_brand or not k.pref_hit(pid) or k.pref_hit(x["pid"]))
                ]
                if stronger and k.clock.left() > 40:
                    return "title-only spec; prefer a SKU whose options or facts confirm it"
        return None

    def open(self, k: "Kernel") -> list:
        return []

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.query.sealed or k.inbox.pending_q:
            return None
        return k.probe()

    def pick(self, k: "Kernel") -> dict | None:
        first = k.index.adds == 0
        good = [c for c in k.candidates() if c["ok"]]
        if first and k.query.pref_brand:
            named = [c for c in good if k.pref_hit(c["pid"])]
            if named:
                good = named
            elif k.turn < k.max_steps * 0.65 and k.clock.left() > 50:
                k._brand_nudge()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, first))
        return good[0]

    def hunt(self, k: "Kernel") -> tuple | None:
        return None

    def hold(self, k: "Kernel") -> str | None:
        if k.closing or k.index.event_key or k.inbox.updated:
            return None
        if not k.index.adds:
            return None
        if not (k.query.pref_brand and k.query.spec_field):
            return None
        if k.clock.left() < 40 or k.turn >= k.max_steps - 4:
            return None
        if k.awaiting:
            return "re-inspect the first commit before ordering"
        if k.recheck_turn is None:
            return "re-check current price and stock after the first add"
        if not k.held_revision:
            return "hold one non-terminal turn so a shopper revision of the firm spec can arrive"
        return None


class IntentPolicy(Policy):
    name = "intent"

    def ready(self, k: "Kernel") -> bool:
        if k.index.adds or k.closing:
            return True
        if not k.brand_seen and k.clock.left() > 60:
            return False
        proven = [c for c in k.candidates() if c["ok"]]
        if not proven and k.clock.left() > 40 and k.turn < k.max_steps - 4:
            return False
        if k.query.pref_brand:
            branded = [c for c in proven if k.pref_hit(c["pid"])]
            if not branded and k.clock.left() > 50 and k.turn < k.max_steps * 0.65:
                if k.unread() or k.searches_left() or not k.brand_filtered:
                    return False
        return k.backup_ready()

    def hunt(self, k: "Kernel") -> tuple | None:
        return ConstraintPolicy.hunt(self, k)

    def rank(self, k: "Kernel", c: dict, first: bool) -> tuple:
        extra = sum(
            1
            for f, v in k.query.specs.items()
            if f != k.query.spec_field and k.variant_value(c["pid"], c["sku"], f) != v
        )
        proven = 0 if _phone_ok(k.facts(c["pid"]), k.title(c["pid"]), k.options(c["pid"], c["sku"])) == "yes" else 1
        price = c["price"] if c["price"] is not None else 1e12
        source = c.get("source") or "none"
        evidence = 0 if source in ("variant", "facts") else (1 if source == "title" else 2)
        if first and k.query.pref_brand:
            wanted = _tok(k.query.pref_brand)
            near = 0 if (wanted <= _tok(k.brand_of(c["pid"])) or wanted <= _tok(k.title(c["pid"]))) else 1
            return (near, evidence, extra, proven, price, c["key"])
        return (evidence, extra, proven, price, c["key"])

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.inbox.pending_q:
            return None
        nxt = k.next_read()
        if nxt is not None:
            return nxt
        return k.probe()

    def open(self, k: "Kernel") -> list:
        steps = []
        brand = k.query.pref_brand or k.query.brand
        words = [w for w in (brand, k.query.spec_value, k.query.category) if w]
        if words and "search" in k.tools:
            steps.append(("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10}, "", ""))
            if brand:
                k.brand_seen = True
        if "filter" in k.tools and (k.query.category or brand):
            args: dict[str, Any] = {"in_stock": True, "k": 10}
            if k.query.category:
                args["category"] = k.query.category
            if k.budget is not None:
                args["max_price"] = k.budget
            if brand:
                args["brand"] = brand
                k.brand_seen = True
                k.brand_filtered = True
            steps.append(("filter", args, "", ""))
        return steps

    def pick(self, k: "Kernel") -> dict | None:
        first = k.index.adds == 0
        good = [c for c in k.candidates() if c["ok"]]
        if first and k.query.pref_brand:
            named = [c for c in good if k.pref_hit(c["pid"])]
            if named:
                good = named
            elif k.turn < k.max_steps * 0.6:
                k._brand_nudge()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, first))
        return good[0]


class RecallPolicy(Policy):
    name = "recall"

    def ready(self, k: "Kernel") -> bool:
        if k.index.adds:
            return True
        if k.searches < 3 and k.clock.left() > 40 and k.turn < k.max_steps - 4:
            return False
        themed = [pid for pid in k.unread() if k.query.theme_hit(k.title(pid) or "")]
        if themed and k.turn < k.max_steps - 5 and k.clock.left() > 40:
            return False
        return True

    def open(self, k: "Kernel") -> list:
        if not k.query.theme or "search" not in k.tools:
            return []
        q = " ".join(k.query.theme)
        if k.query.category:
            q = q + " " + k.query.category
        return [("search", {"query": q, "in_stock": True, "k": 10}, "", "")]

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.inbox.pending_q:
            return None
        if k.searches < 3:
            nxt = k.next_search()
            if nxt is not None:
                return nxt
        return k.read_more()

    def pick(self, k: "Kernel") -> dict | None:
        good = [c for c in k.candidates() if c["ok"]]
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, k.index.adds == 0))
        return good[0]


class ConstraintPolicy(Policy):
    name = "constraint"

    def ready(self, k: "Kernel") -> bool:
        if k.index.adds or k.closing:
            return True
        if k.query.sealed:
            if k.turn < k.max_steps * 0.35:
                return False
            k.query.sealed = False
            k.note("Hidden spec was never stated; remaining firm gates decide.")
        if k.turn < 3 and not k.inbox.updated and k.query.spec_field:
            return False
        return k.backup_ready()

    def hunt(self, k: "Kernel") -> tuple | None:
        spec_i = k.spec_inspect()
        if spec_i:
            return spec_i
        deep = k.deepen()
        if deep:
            return deep
        if "inspect_stock" in k.tools:
            for c in k.candidates():
                if not c["ok"] or c["key"] in k.index.inspected or c["key"] in k.index.dead:
                    continue
                if _ACCESSORY_RX.search(k.title(c["pid"])):
                    continue
                return ("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
        leads = k.sibling_leads() or k.spec_leads() or k.unread()
        if leads:
            if len(leads) > 1 and "compare" in k.tools:
                return ("compare", {"product_ids": leads[:5]})
            if "view" in k.tools:
                return ("view", {"product_id": leads[0]})
        model_q = k.model_search()
        if model_q:
            return model_q
        return k.next_search()

    def open(self, k: "Kernel") -> list:
        return []

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.query.sealed or k.inbox.pending_q:
            return None
        return k.probe()

    def pick(self, k: "Kernel") -> dict | None:
        good = [c for c in k.candidates() if c["ok"] and not k.decoy(c)]
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, k.index.adds == 0))
        return good[0]


class PreferencePolicy(Policy):
    name = "preference"

    def ready(self, k: "Kernel") -> bool:
        if k.index.adds or k.closing:
            return True
        if len(k.index.listings) < 2 and k.turn < k.max_steps * 0.45 and k.clock.left() > 80:
            return False
        if (
            k.query.wants_water
            and k.water_open()
            and k.turn < 8
            and k.clock.left() > 120
            and not k.has_water()
        ):
            return False
        return True

    def rank(self, k: "Kernel", c: dict, first: bool) -> tuple:
        axes = set(k.query.axes)
        keys = []
        if "water" in axes or k.query.wants_water:
            keys.append(0 if k.water_hits(c["pid"], c["sku"]) > 0 else 1)
            keys.append(-k.water_hits(c["pid"], c["sku"]))
        if "sport" in axes:
            keys.append(0 if k.sport_hits(c["pid"], c["sku"]) > 0 else 1)
        if "wireless" in axes:
            keys.append(0 if k.wireless_hits(c["pid"], c["sku"]) > 0 else 1)
        if "battery" in axes:
            keys.append(0 if k.battery_hits(c["pid"], c["sku"]) > 0 else 1)
        price = c["price"] if c["price"] is not None else 1e12
        quality = -k.priority(c["pid"], c["sku"])
        return tuple(keys) + (quality, price, c["key"])

    def open(self, k: "Kernel") -> list:
        steps = []
        cat = k.query.category or ""
        terms = k.query.axis_terms()
        if "search" in k.tools:
            if terms:
                steps.append(
                    ("search", {"query": " ".join(terms + ([cat] if cat else [])), "in_stock": True, "k": 10}, "", "")
                )
                if len(terms) > 2:
                    steps.append(
                        (
                            "search",
                            {"query": " ".join(terms[:2] + ([cat] if cat else [])), "in_stock": True, "k": 10},
                            "",
                            "",
                        )
                    )
            elif cat:
                steps.append(("search", {"query": cat, "in_stock": True, "k": 10}, "", ""))
        if "filter" in k.tools:
            args: dict[str, Any] = {"in_stock": True, "k": 10}
            if k.query.category:
                args["category"] = k.query.category
            if k.budget is not None:
                args["max_price"] = k.budget
            steps.append(("filter", args, "", ""))
        return steps[: k.max_calls]

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.inbox.pending_q:
            return None
        return k.read_more()

    def pick(self, k: "Kernel") -> dict | None:
        first = k.index.adds == 0
        good = [c for c in k.candidates() if c["ok"]]
        if not first:
            carted = [c for c in good if (c["pid"], c["sku"]) in k.index.cart]
            if carted:
                return carted[0]
        if not good:
            return None
        if first and not k.closing and len(k.index.listings) < 2 and k.turn < 10:
            return None
        if first and not k.closing and k.query.wants_water and not k.has_water() and k.water_open() and k.turn < 8:
            return None
        if k.query.wants_water:
            wet = [c for c in good if k.water_hits(c["pid"], c["sku"]) > 0]
            if wet:
                good = wet
            elif not k.closing and k.water_open():
                return None
            elif not first and not k.closing:
                # After a shock, keep hunting a water match rather than the cheapest dry decoy.
                more = k.read_more()
                if more is not None and k.turn < k.max_steps - 4:
                    return None
        good.sort(key=lambda c: self.rank(k, c, first))
        for pick in good:
            if not first or not self.first_ok(k, pick["pid"], pick["sku"]):
                return pick
        return good[0] if k.closing else None


class RankingPolicy(Policy):
    name = "ranking"

    def ready(self, k: "Kernel") -> bool:
        return k.keys_ready() or k.turn >= k.max_steps - 6

    def rank(self, k: "Kernel", c: dict, first: bool) -> tuple:
        measures = []
        fields = k.query.rank_fields or [f for f in ("refresh_rate", "screen_size", "ram") if f in k.query.specs]
        for field in fields:
            measures.append(-(k.measure(c["pid"], c["sku"], field) or 0.0))
        price = c["price"] if c["price"] is not None else 1e12
        return tuple(measures) + (price, c["key"])

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart:
            return None
        return k.key_reads()

    def pick(self, k: "Kernel") -> dict | None:
        good = [c for c in k.candidates() if c["ok"]]
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, k.index.adds == 0))
        return good[0]


class RecoveryPolicy(Policy):
    name = "recovery"

    def rank(self, k: "Kernel", c: dict, first: bool) -> tuple:
        price = c["price"] if c["price"] is not None else 1e12
        if first or not k.index.event_key:
            return (c["relax"], price, c["key"])
        dead_sku = (k.index.event_key or "").split("::", 1)[-1]
        family = _sku_stem(dead_sku)
        same = 0 if family and str(c["sku"]).startswith(family) else 1
        tier = 0 if c["relax"] == 1 else (1 if c["relax"] == 0 else c["relax"] + 1)
        return (tier, same, price, c["key"])

    def first_ok(self, k: "Kernel", pid: str, sku: str) -> str | None:
        if k.index.adds:
            return None
        if k.relax(pid, sku):
            return "first add must be the exact named model and color"
        return None

    def open(self, k: "Kernel") -> list:
        if not k.query.model or "search" not in k.tools:
            return []
        q = " ".join(str(w) for w in (k.query.model, k.query.color, k.query.category) if w)
        return [("search", {"query": q, "in_stock": True, "k": 10}, "", "")]

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.inbox.pending_q:
            return None
        return k.read_more()

    def hunt(self, k: "Kernel") -> tuple | None:
        twin = k.twin_search()
        if twin is not None:
            return twin
        return k.sibling_search()

    def pick(self, k: "Kernel") -> dict | None:
        first = k.index.adds == 0
        good = [c for c in k.candidates() if c["ok"]]
        if k.query.model and k.index.event_key and not first:
            if not k.twins_done:
                return None
            siblings = [c for c in good if c["relax"] == 1]
            if siblings:
                good = siblings
            elif not k.sibs_done and k.turn < k.max_steps * 0.85:
                return None
        if first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif k.turn < k.max_steps * 0.6:
                k._exact_nudge()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, first))
        return good[0]


class JustificationPolicy(Policy):
    name = "justification"

    def ready(self, k: "Kernel") -> bool:
        if k.index.adds or k.closing:
            return True
        if k.turn < 3 and not k.inbox.updated:
            return False
        return True

    def discover(self, k: "Kernel") -> tuple | None:
        if k.index.cart or k.inbox.pending_q:
            return None
        if any(c["ok"] for c in k.candidates()) or k.turn >= 6:
            return None
        pick = self.pick(k)
        if pick and pick["key"] in k.index.inspected:
            return None
        return k.probe()

    def pick(self, k: "Kernel") -> dict | None:
        good = [c for c in k.candidates() if c["ok"] and not k.decoy(c)]
        if not good:
            field, want = k.query.spec_field, k.query.spec_value
            for c in k.candidates():
                if not c["live"] or c["key"] in k.index.dead or k.decoy(c):
                    continue
                if k.query.brand and not k.brand_ok(c["pid"]):
                    continue
                if field and want:
                    if k.fact_value(c["pid"], field) != want and k.product_value(c["pid"], field) != want:
                        continue
                if not k.device_ok(c["pid"], c["sku"]):
                    continue
                good.append(c)
        if not good:
            return None
        good.sort(key=lambda c: self.rank(k, c, k.index.adds == 0))
        return good[0]

    def open(self, k: "Kernel") -> list:
        brand = k.query.pref_brand or k.query.brand
        words = [w for w in (brand, k.query.spec_value, k.query.category) if w]
        steps = []
        if words and "search" in k.tools:
            steps.append(("search", {"query": " ".join(str(w) for w in words), "in_stock": True, "k": 10}, "", ""))
        if "filter" in k.tools and (k.query.category or brand):
            args: dict[str, Any] = {"in_stock": True, "k": 10}
            if k.query.category:
                args["category"] = k.query.category
            if k.budget is not None:
                args["max_price"] = k.budget
            if brand:
                args["brand"] = brand
            steps.append(("filter", args, "", ""))
        return steps


class GenericPolicy(Policy):
    name = "generic"

    def ready(self, k: "Kernel") -> bool:
        if k.index.adds or k.closing:
            return True
        if k.query.sealed and k.turn < k.max_steps * 0.35:
            return False
        if k.query.listing and len(k.index.listings) < 2:
            if k.turn < k.max_steps * 0.45 and k.clock.left() > 80:
                return False
        if k.query.pref_brand and not k.brand_seen and k.clock.left() > 80:
            return False
        return True


_POLICIES = {
    "intent": IntentPolicy,
    "recall": RecallPolicy,
    "constraint": ConstraintPolicy,
    "preference": PreferencePolicy,
    "ranking": RankingPolicy,
    "recovery": RecoveryPolicy,
    "justification": JustificationPolicy,
    "generic": GenericPolicy,
}


# ---------------------------------------------------------------- kernel


class Kernel:
    """Shared commerce runtime. Family policy owns graded decisions."""

    def __init__(self, problem: dict) -> None:
        env = problem["environment"]
        self.binding = dict(env["binding"])
        view = env["policy_view"]
        self.view = view
        self.schema = list(view.get("tools") or [])
        self.tools = {(t.get("function") or {}).get("name") for t in self.schema if isinstance(t, dict)}
        self.max_steps = int(view.get("max_steps") or 30)
        self.max_calls = max(1, int(view.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.query = Query(str(view.get("query") or ""))
        self.budget = self.query.budget
        self.index = CatalogIndex()
        self.inbox = ShopperInbox()
        self.clock = Clock(_WALL_S)
        self.dialogue: list = []
        self.need_claims = False
        self.reason_fields = _SOFT_FIELDS
        self._read_order_schema()
        self.messages: list = [
            {"role": "system", "content": self._prompt()},
            {"role": "user", "content": view.get("query") or ""},
        ]
        self.turn = 0
        self.pending: dict | None = None
        self._last: dict | None = None
        self._consumed = False
        self.closing = False
        self.searches = 0
        self.cart_reads = 0
        self.awaiting: tuple | None = None
        self.compares = 0
        self.nudge_exact = False
        self.nudge_brand = False
        self.twins_done = False
        self.sibs_done = False
        self.model_q = False
        self.widen = 0
        self.reads_done = False
        self.opened = False
        self.brand_seen = False
        self.brand_filtered = False
        self.first_add_turn: int | None = None
        self.recheck_turn: int | None = None
        self.held_revision = False
        self.done = False
        self.model_i = 0
        self.rejected = 0
        self.wishes: set | None = None
        self.policy = _POLICIES.get(self.query.family, GenericPolicy)()
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _read_order_schema(self) -> None:
        for tool in self.schema:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
                continue
            params = fn.get("parameters") or {}
            self.need_claims = "justification" in (params.get("required") or [])
            blob = json.dumps(fn).lower()
            found = [f for f in _SOFT_FIELDS if f.replace("_", " ") in blob or f in blob]
            if found:
                self.reason_fields = tuple(found)
            return

    def _prompt(self) -> str:
        extra = " Up to %d tool calls per turn, %d turns." % (self.max_calls, self.max_steps)
        firm = [x for x in (self.query.category, self.query.spec_value, self.query.brand) if x]
        if self.budget is not None:
            firm.append("budget %.2f %s" % (self.budget, self.query.currency))
        if firm:
            extra += " Firm: " + ", ".join(str(x) for x in firm) + "."
        if self.query.pref_brand:
            extra += " Soft brand %s on the first add if a fully compliant SKU exists." % self.query.pref_brand
        if self.query.need_ids:
            extra += " Surface at least %d distinct matching products before ordering." % self.query.need_ids
        if self.query.wants_water:
            extra += " Prefer sweat/water resistance over the cheapest listing."
        if self.query.sealed:
            extra += " Ask for the hidden hardware spec with message before committing."
        if self.need_claims or self.query.wants_reason:
            extra += " Orders need grounded claims for the spec, price, in_stock, and one of %s." % ", ".join(
                self.reason_fields
            )
        extra += " " + self._analysis()
        return _GUIDE + extra

    def _analysis(self) -> str:
        q = self.query
        firm = []
        if q.category:
            firm.append("category " + q.category)
        if q.budget is not None:
            firm.append("budget %.2f %s" % (q.budget, q.currency))
        if q.spec_field and q.spec_value:
            firm.append("%s = %s" % (q.spec_field, q.spec_value))
        if q.brand:
            firm.append("brand " + q.brand)
        if q.keys:
            firm.append("%d named keys" % len(q.keys))
        if q.model:
            firm.append("exact %s %s" % ((q.color or "").strip(), q.model))
        soft = []
        if q.pref_brand:
            soft.append("preferred brand " + q.pref_brand)
        for axis in q.axes:
            soft.append(axis + " over cheapest")
        if q.theme:
            soft.append("theme " + " ".join(q.theme))
        return "Firm: %s. Soft: %s." % ("; ".join(firm) or "stock and stated budget", "; ".join(soft) or "none")

    def note(self, text: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:300]})

    def page(self, pid: str) -> dict:
        return self.index.pages.get(str(pid)) or {}

    def facts(self, pid: str) -> dict:
        f = self.page(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def title(self, pid: str) -> str:
        title = str(self.page(pid).get("title") or "")
        if title:
            return title
        for key, row in self.index.hits.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def brand_of(self, pid: str) -> str:
        brand = self.page(pid).get("brand")
        if brand:
            return str(brand)
        for key, row in self.index.hits.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def path(self, pid: str) -> list:
        p = self.page(pid).get("category_path")
        return list(p) if isinstance(p, list) else []

    def options(self, pid: str, sku: str) -> str:
        return self.index.options.get(_vid(pid, sku), "")

    def _strip_models(self, pid: str, title: str) -> str:
        for token in re.split(r"[;,]", str(self.facts(pid).get("model") or "")):
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

    def fact_raw(self, pid: str, field: str):
        facts = self.facts(pid)
        for label in _FACT_LABELS.get(field, ()):
            raw = facts.get(label)
            if raw:
                return raw
        return None

    def fact_value(self, pid: str, field: str) -> str | None:
        raw = self.fact_raw(pid, field)
        if not raw:
            return None
        text = str(raw).strip()
        found = _READERS[field](text)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(text) + unit
        return _one(found)

    def product_value(self, pid: str, field: str) -> str | None:
        vals: set = set()
        conflict = False
        raw = self.fact_raw(pid, field)
        if raw:
            found = _READERS[field](str(raw))
            conflict = conflict or len(found) > 1
            if field == "resolution" and _QHD_FALSE.search(str(raw)):
                conflict = True
            vals |= found
        title = self.title(pid)
        if field == "resolution":
            title = self._strip_models(pid, title)
            if _QHD_FALSE.search(title):
                conflict = True
        vals |= _adjacent(field, title)
        return None if conflict else _one(vals)

    def proof(self, pid: str, sku: str, field: str) -> tuple[str | None, str]:
        opt = self.options(pid, sku)
        opt_vals = _adjacent(field, opt)
        if len(opt_vals) > 1:
            return None, "variant-conflict"
        if opt_vals:
            return next(iter(opt_vals)), "variant"
        if _mentions(field, opt):
            return None, "variant-unclear"
        raw = self.fact_raw(pid, field)
        title = self.title(pid)
        if field == "resolution":
            title = self._strip_models(pid, title)
        fact_vals: set = set()
        conflict = False
        if raw:
            fact_vals = _READERS[field](str(raw))
            if field == "resolution" and _QHD_FALSE.search(str(raw)):
                conflict = True
            if len(fact_vals) > 1:
                conflict = True
        title_vals = _adjacent(field, title) if field != "resolution" else _READERS[field](title)
        if field == "resolution" and _QHD_FALSE.search(title):
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

    def variant_value(self, pid: str, sku: str, field: str) -> str | None:
        got, _src = self.proof(pid, sku, field)
        return got

    def measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self.variant_value(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    def category_ok(self, pid: str) -> bool:
        want = self.query.category
        path = self.path(pid)
        if not want or not path:
            return True
        head = _noun(want)
        wanted = _aliases(head) | _stems(want)
        labels = set()
        for label in path:
            labels |= _stems(str(label))
        placed = bool(wanted & labels) or any(
            _stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == _stems(want) for label in path
        )
        if not placed:
            title_stems = _stems(self.title(pid)) | _tok(self.title(pid))
            placed = bool(wanted & title_stems)
        if not placed and self.query.theme:
            placed = True
        if not placed:
            return False
        if self.query.theme:
            return True
        quals = self.query.quals
        if not quals:
            return True
        title = self.title(pid)
        words = _stems(title) | _tok(title)
        for q in quals:
            qstem = q.rstrip("s")
            if qstem == "gaming":
                fast = any(float(v[:-2]) >= 120 for v in _hz(title))
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

    def brand_ok(self, pid: str) -> bool:
        want = self.query.brand
        if not want:
            return True
        wanted = _tok(want)
        if wanted <= _tok(self.brand_of(pid)):
            return True
        title = self.title(pid)
        if not re.search(r"(?i)(?<![a-z])" + re.escape(want) + r"(?![a-z])", title):
            return False
        if _ACCESSORY_RX.search(title) or _PART_RX.search(title):
            return False
        if re.search(r"(?i)\b(?:for|compatible(?:\s+with)?|case(?:s)?|charger|keyboard|dummy)\b", title):
            if not re.search(r"(?i)^\W*" + re.escape(want), title):
                return False
        return True

    def brand_title(self, title: str, row: dict | None = None) -> bool:
        want = self.query.brand
        if not want:
            return False
        wanted = _tok(want)
        brand = str((row or {}).get("brand") or "")
        return bool(wanted <= _tok(brand) or wanted <= _tok(title))

    def device_ok(self, pid: str, sku: str) -> bool:
        cat = self.query.category or ""
        title = self.title(pid)
        head = _noun(cat)
        if head in {"tablet", "ipad"}:
            return bool(_TABLET_RX.search(title))
        if head in {"monitor", "display"}:
            return bool(_MONITOR_RX.search(title))
        if not _same(cat, "smartphone") and "phone" not in _stems(cat):
            return True
        verdict = _phone_ok(self.facts(pid), title, self.options(pid, sku))
        if self.query.wants_reason or self.query.brand or self.query.require_id:
            return verdict == "yes"
        return verdict != "no"

    def resale(self, pid: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self.title(pid),
                re.I,
            )
        )

    def relax(self, pid: str, sku: str) -> int:
        model, color = self.query.model, self.query.color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self.title(pid) + " " + " ".join(str(v) for v in self.facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self.options(pid, sku) + " " + str(self.facts(pid).get("color") or "") + " " + self.title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def assess(self, pid: str, sku: str) -> dict:
        key = _vid(pid, sku)
        st = self.index.quotes.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        keep = bool(self.query.model)
        dead = key in self.index.dead or (str(pid) in self.index.dead_pids and not keep)
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.index.pages
        field, want = self.query.spec_field, self.query.spec_value
        spec = 2
        source = "none"
        if field and want:
            got, source = self.proof(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
            if spec != 2 and self.query.wants_reason:
                if self.fact_value(pid, field) == want or self.product_value(pid, field) == want:
                    spec = 2
                    source = "title-or-fact"
        ok = bool(live and viewed and spec == 2 and self.brand_ok(pid) and self.device_ok(pid, sku))
        ok = ok and self.category_ok(pid)
        if price is not None and price <= 50:
            ok = False
        if _ACCESSORY_RX.search(self.title(pid)) and _same(self.query.category or "", "smartphone"):
            ok = False
        if _PART_RX.search(self.title(pid)) or _NOT_PHONE_RX.search(self.title(pid)):
            if _same(self.query.category or "", "smartphone"):
                ok = False
        if self.query.sealed:
            ok = False
        if self.query.wants_reason:
            ok = ok and not self.resale(pid)
        if self.query.listing and self.index.event_key:
            dead_listing = self.index.listing_of.get(self.index.event_key.split("::", 1)[0])
            if dead_listing and self.index.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if self.query.theme:
            ok = ok and self.theme_ok(pid, sku)
        if self.query.keys:
            ok = ok and (str(pid), str(sku)) in self.query.keys
            for measure in self.query.rank_fields or self.query.specs:
                if self.measure(pid, sku, measure) is None:
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
            "source": source,
            "ok": ok,
            "relax": self.relax(pid, sku) if self.query.model else 0,
        }

    def decoy(self, c: dict) -> bool:
        """Catalog bait: 10 PHP 'phones' that pass brand/title filters."""
        price = c.get("price") if isinstance(c, dict) else None
        if price is None:
            return False
        if price <= 50:
            return True
        if self.budget is not None and self.budget >= 500 and price <= min(99.0, self.budget * 0.015):
            return True
        return False

    def candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.index.quotes):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.assess(pid, sku))
        return out

    def target(self) -> dict | None:
        return self.policy.pick(self)

    def pref_hit(self, pid: str) -> bool:
        want = self.query.pref_brand
        if not want:
            return False
        tokens = _tok(want)
        return bool(tokens <= _tok(self.brand_of(pid)) or tokens <= _tok(self.title(pid)))

    def branded(self) -> list:
        return [c for c in self.candidates() if c["ok"] and self.pref_hit(c["pid"])]

    def invalid(self, pid: str, sku: str) -> bool:
        if _vid(pid, sku) in self.index.dead:
            return True
        st = self.index.quote(pid, sku)
        if st is None:
            return False
        if not st.get("in_stock"):
            return True
        price = st.get("price")
        return bool(self.budget is not None and price is not None and price > self.budget + _EPS)

    def kill(self, pid: str, sku: str) -> None:
        key = _vid(pid, sku)
        if self.index.event_key is None:
            self.index.event_key = key
        self.index.dead.add(key)
        if not self.query.model:
            self.index.dead_pids.add(str(pid))
        self.compares = 0
        self.widen = 0
        self.reads_done = False
        self.brand_filtered = False

    def reply(self) -> str:
        shocked = self.index.event_key
        if not shocked:
            for pid, sku in self.index.cart:
                if self.invalid(pid, sku):
                    shocked = _vid(pid, sku)
                    break
        if shocked:
            st = self.index.quotes.get(shocked) or {}
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
        parts = ["I am going with " + (self.title(pid) or pid)[:90]]
        field, want = self.query.spec_field, self.query.spec_value
        got = self.variant_value(pid, sku, field) if field else None
        if field and want and got == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        elif field and want:
            parts.append("I am checking that its " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.query.currency))
        return ", ".join(parts) + "."

    def spec_token(self) -> str:
        raw = str(self.query.spec_value or "")
        m = re.search(r"(\d+(?:\.\d+)?)", raw)
        return m.group(1) if m else raw.strip()

    def title_has_spec(self, title: str) -> bool:
        if not self.query.spec_value:
            return True
        blob = str(title or "").lower()
        token = self.spec_token().lower()
        if token and token in blob:
            return True
        want = str(self.query.spec_value).lower()
        return bool(want and want in blob)

    def looks_device(self, title: str) -> bool:
        title = title or ""
        cat = (self.query.category or "").lower()
        head = _noun(cat)
        if _JUNK_RX.search(title):
            return False
        if re.search(r"\biphone\b", title, re.I):
            return True
        if head in {"smartphone", "phone"} and _ACCESSORY_RX.search(title):
            return False
        if head in {"headphone", "headset", "earbud"} and not _HEADPHONE_RX.search(title):
            return False
        if head in {"monitor", "display"} and not _MONITOR_RX.search(title):
            return False
        if head in {"tablet", "ipad"} and not _TABLET_RX.search(title):
            return False
        if not cat:
            return True
        words = _stems(title) | _tok(title)
        aliases = _aliases(head) if head else set()
        if words & (aliases | _stems(cat)):
            return True
        if self.query.model:
            squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
            if squash(self.query.model) in squash(title):
                return True
        if self.query.theme and self.query.theme_hit(title):
            return True
        return False

    def theme_ok(self, pid: str, sku: str | None = None) -> bool:
        facts = " ".join(str(v) for v in self.facts(pid).values())
        opt = self.options(pid, sku) if sku else ""
        return self.query.theme_hit(self.title(pid), facts, opt, self.brand_of(pid))

    def hunt_brand(self) -> str | None:
        if self.index.adds == 0 and self.query.pref_brand:
            return self.query.pref_brand
        return self.query.brand

    def filter_args(self, brand: str | None) -> dict | None:
        if "filter" not in self.tools:
            return None
        if not (self.query.category or brand):
            return None
        args: dict[str, Any] = {"in_stock": True, "k": 10}
        if self.query.category:
            args["category"] = self.query.category
        if self.budget is not None:
            args["max_price"] = self.budget
        if brand:
            args["brand"] = brand
        return args

    def unread(self) -> list:
        leads = []
        hunt_spec = bool(self.query.spec_value) and (
            bool(self.query.pref_brand) or bool(self.query.spec_field) or bool(self.index.event_key)
        )
        for key, row in self.index.hits.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.index.pages or str(pid) in self.index.dead_pids:
                continue
            st = self.index.quotes.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get("title") or "")
            spec_hit = str(pid) in self.index.spec_hits
            if not self.looks_device(title):
                if not spec_hit or _JUNK_RX.search(title) or _ACCESSORY_RX.search(title):
                    continue
            if hunt_spec and self.query.spec_value and not self.title_has_spec(title) and not spec_hit:
                brand_hit = bool(
                    (self.query.brand and self.brand_title(title, row))
                    or (
                        self.query.pref_brand
                        and _tok(self.query.pref_brand) <= (_tok(title) | _tok(str(row.get("brand") or "")))
                    )
                )
                opened = self.searches >= 2 or bool(self.index.event_key) or bool(self.index.pages)
                if not (brand_hit or opened):
                    continue
            if self.query.theme and not self.query.theme_hit(title, str(row.get("brand") or "")):
                continue
            if self.query.model and self.index.adds == 0 and not self.index.event_key:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.query.model.lower()) not in squashed:
                    continue
            if self.query.brand:
                wanted = _tok(self.query.brand)
                if not (wanted <= _tok(str(row.get("brand") or "")) or wanted <= _tok(title)):
                    continue
            branded = 0 if self.pref_hit(str(pid)) else 1
            water = 0 if _WATER_RX.search(title) else (1 if _WIRELESS_RX.search(title) else 2)
            leads.append((water if self.query.wants_water else 0, branded, price, str(pid)))
        leads.sort()
        out = []
        for _w, _b, _p, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def water_leads(self) -> list:
        out = []
        for pid in self.unread():
            title = self.title(pid)
            if _WATER_RX.search(title) or _WIRELESS_RX.search(title):
                out.append(pid)
        return out

    def has_water(self) -> bool:
        for c in self.candidates():
            if c["live"] and c["viewed"] and self.water_hits(c["pid"], c["sku"]) > 0:
                return True
        return False

    def searches_left(self) -> bool:
        return self.widen < len(self.widen_qs())

    def water_open(self) -> bool:
        if self.clock.left() <= 30 or self.turn >= self.max_steps - 3:
            return False
        return bool(self.water_leads()) or self.searches_left()

    def sibling_leads(self) -> list:
        keys = self._dead_keys()
        if len(keys) < 2:
            return []
        out = []
        for pid in self.unread():
            title = self.title(pid)
            if _ACCESSORY_RX.search(title) or _JUNK_RX.search(title) or _PART_RX.search(title):
                continue
            if len(_tok(title) & keys) >= 2:
                out.append(pid)
        return out

    def spec_leads(self) -> list:
        out = []
        for pid in self.unread():
            title = self.title(pid) or ""
            if _ACCESSORY_RX.search(title) or _JUNK_RX.search(title):
                continue
            if not self.looks_device(title) or not self.title_has_spec(title):
                continue
            out.append(pid)
        return out

    def widen_qs(self) -> list:
        spec = self.query.spec_value
        token = self.spec_token()
        brand = self.hunt_brand()
        cat = self.query.category
        inch = self.query.spec_field == "screen_size"
        seen, out = set(), []
        if self.query.theme:
            theme = " ".join(self.query.theme)
            parts = list(self.query.theme)
            for words in (
                (theme, cat) if cat else (theme,),
                (theme,) if cat else None,
                (parts[-1], cat) if len(parts) > 1 and cat else None,
                (parts[0],) if parts else None,
                (parts[0], cat) if parts and cat else None,
            ):
                if not words:
                    continue
                q = " ".join(str(w) for w in words if w)
                key = q.lower()
                if q and key not in seen:
                    seen.add(key)
                    out.append(q)
            return out
        shocked = bool(self.index.event_key or self.index.dead)
        for words in (
            (brand, cat) if brand and cat and shocked else None,
            (token, "inch", "phone") if inch and token else None,
            (token, "inch", cat) if inch and token else None,
            (spec, cat),
            (brand, spec, cat),
            (brand, token, "inch", cat) if inch and token else None,
            (brand, token, cat) if token else None,
            (brand, spec) if brand and spec else None,
            (brand, cat) if brand else None,
            (self.query.model, cat) if self.query.model else None,
            (self.query.model, self.query.color, cat) if self.query.model else None,
            (token,) if self.index.event_key and token else None,
            (spec,) if self.index.event_key and spec else None,
            *[(term, cat) for term in self.query.axis_terms()],
        ):
            if not words:
                continue
            q = " ".join(str(w) for w in words if w)
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
        return out

    def search_cap(self) -> int:
        if self.query.theme or self.query.need_ids:
            return 3
        if self.index.event_key or self.index.dead:
            return 8
        return 5

    def search_fp(self, query: str) -> frozenset:
        return frozenset(tok for tok in _tok(query) if tok not in _STOP)

    def search_seen(self) -> set:
        out = set()
        for key in self.index.seen:
            if not str(key).startswith("search"):
                continue
            try:
                args = json.loads(str(key)[6:])
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict):
                out.add(self.search_fp(str(args.get("query") or "")))
        return out

    def next_search(self) -> tuple | None:
        if "search" not in self.tools:
            return None
        if self.searches >= self.search_cap():
            return None
        queries = self.widen_qs()
        seen = self.search_seen()
        while self.widen < len(queries):
            q = queries[self.widen]
            self.widen += 1
            args = {"query": q, "in_stock": True, "k": 10}
            if "search" + json.dumps(args, sort_keys=True) in self.index.seen:
                continue
            fp = self.search_fp(q)
            if fp and fp in seen:
                continue
            return ("search", args)
        return None

    def twin_search(self) -> tuple | None:
        dead = self.index.event_key
        if self.twins_done or not dead or "search" not in self.tools:
            return None
        # Same-title siblings help color/model recovery. On spec families they
        # pull the invalidated listing back and hide the next firm-valid SKU.
        if self.query.spec_field or self.query.wants_reason:
            self.twins_done = True
            return None
        if not self.query.model and not self.query.listing:
            self.twins_done = True
            return None
        title = self.title(dead.split("::", 1)[0])
        if not title:
            return None
        self.twins_done = True
        self.note("Searching the dead listing's title to surface sibling colors and sizes.")
        return ("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def model_search(self) -> tuple | None:
        """After a shock, search the dead listing's brand/model tokens."""
        if self.model_q or not self.index.event_key or "search" not in self.tools:
            return None
        pid = self.index.event_key.split("::", 1)[0]
        title = self.title(pid)
        words = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+", title or "")
            if w.lower() not in _STOP and not _ACCESSORY_RX.fullmatch(w)
        ]
        skip = {"new", "sale", "original", "promo", "official", "free", "cod"}
        keep = [
            w
            for w in words
            if (w[:1].isupper() or re.search(r"\d", w)) and w.lower() not in skip
        ]
        q = " ".join((keep or words)[:4])
        if not q or self.search_fp(q) in self.search_seen():
            self.model_q = True
            return None
        self.model_q = True
        return ("search", {"query": q, "in_stock": True, "k": 10})

    def sibling_search(self) -> tuple | None:
        if self.sibs_done or not self.query.model or "search" not in self.tools:
            return None
        if not self.index.event_key:
            rows = [c for c in self.candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
            if rows:
                return None
        q = " ".join(str(w) for w in (self.query.model, self.query.category) if w)
        if not q:
            return None
        args = {"query": q, "k": 10, "in_stock": True}
        if "search" + json.dumps(args, sort_keys=True) in self.index.seen:
            self.sibs_done = True
            return None
        self.sibs_done = True
        return ("search", args)

    def key_reads(self) -> tuple | None:
        unseen = []
        inspect = []
        for pid, sku in self.query.keys:
            if str(pid) not in self.index.pages:
                if str(pid) not in unseen:
                    unseen.append(str(pid))
            elif _vid(pid, sku) not in self.index.inspected:
                inspect.append((str(pid), str(sku)))
        if unseen and "compare" in self.tools and len(unseen) > 1:
            return ("compare", {"product_ids": unseen[:5]})
        if unseen and "view" in self.tools:
            return ("view", {"product_id": unseen[0]})
        if inspect and "inspect_stock" in self.tools:
            pid, sku = inspect[0]
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        return None

    def keys_ready(self) -> bool:
        if not self.query.keys:
            return False
        for pid, sku in self.query.keys:
            if str(pid) not in self.index.pages:
                return False
            if _vid(pid, sku) not in self.index.inspected:
                return False
        return True

    def read_more(self) -> tuple | None:
        named = self.key_reads()
        if named is not None:
            return named
        if self.query.keys:
            return None
        if self.query.wants_water:
            water = self.water_leads()
            if water and self.compares < 16:
                self.compares += 1
                if len(water) > 1 and "compare" in self.tools:
                    return ("compare", {"product_ids": water[:5]})
                if "view" in self.tools:
                    return ("view", {"product_id": water[0]})
            nxt = self.next_search()
            if nxt is not None and not self.has_water():
                return nxt
        if self.query.need_ids:
            themed = [pid for pid in self.unread() if self.query.theme_hit(self.title(pid) or "")]
            if themed and self.compares < 16:
                self.compares += 1
                if len(themed) > 1 and "compare" in self.tools:
                    return ("compare", {"product_ids": themed[:5]})
                if "view" in self.tools:
                    return ("view", {"product_id": themed[0]})
            if self.searches < 3 and len(self.index.clusters) < self.query.need_ids and self.searches_left():
                nxt = self.next_search()
                if nxt is not None:
                    return nxt
        twin = self.twin_search()
        if twin is not None:
            return twin
        sibling = self.sibling_search()
        if sibling is not None:
            return sibling
        model_q = self.model_search()
        if model_q is not None:
            return model_q
        spec_i = self.spec_inspect()
        if spec_i:
            return spec_i
        leads = self.spec_leads() if self.query.spec_value else self.unread()
        if (self.index.event_key or self.index.dead) and self.query.spec_value:
            extra = [pid for pid in self.unread() if pid not in leads]
            leads = leads + extra
        cap = 12 if (self.index.event_key or self.index.dead) else 8
        if leads and self.compares < cap:
            self.compares += 1
            if len(leads) > 1 and "compare" in self.tools:
                return ("compare", {"product_ids": leads[:5]})
            if "view" in self.tools:
                return ("view", {"product_id": leads[0]})
        if self.turn < 2:
            return None
        if self.query.theme:
            nxt = self.next_search()
            if nxt is not None:
                return nxt
            self.reads_done = True
            return None
        brand = self.hunt_brand()
        args = self.filter_args(brand)
        if args:
            sig = "filter" + json.dumps(args, sort_keys=True)
            if sig not in self.index.seen:
                if brand:
                    self.brand_seen = True
                    self.brand_filtered = True
                return ("filter", args)
        nxt = self.next_search()
        if nxt is not None:
            return nxt
        self.reads_done = True
        return None

    def probe(self) -> tuple | None:
        spec_i = self.spec_inspect()
        if spec_i:
            return spec_i
        brand = self.hunt_brand()
        if self.searches == 0 and "search" in self.tools:
            words = [w for w in (brand, self.query.spec_value, self.query.category) if w]
            if brand:
                self.brand_seen = True
            return ("search", {"query": " ".join(str(w) for w in words) or "in stock", "in_stock": True, "k": 10})
        if brand and not self.brand_filtered:
            args = self.filter_args(brand)
            if args:
                self.brand_filtered = True
                self.brand_seen = True
                return ("filter", args)
        if not self.ready():
            return self.read_more()
        return None

    def next_read(self) -> tuple | None:
        spec_i = self.spec_inspect()
        if spec_i:
            return spec_i
        leads = self.spec_leads() or self.unread()
        if not leads:
            return self.read_more()
        if len(leads) > 1 and "compare" in self.tools and self.compares < 12:
            self.compares += 1
            return ("compare", {"product_ids": leads[:5]})
        if "view" in self.tools and self.compares < 12:
            self.compares += 1
            return ("view", {"product_id": leads[0]})
        return self.read_more()

    def _adopt(self, text: str) -> str | None:
        if not self.query.spec_value:
            return self.query.adopt(text)
        field = self.query.spec_field
        mentioned = _ordered(field, text) if field else []
        unique = []
        for value in mentioned:
            if value not in unique:
                unique.append(value)
        if _REVISE_RX.search(text):
            return self.query.adopt(text)
        if len(unique) == 1 and unique[0] != self.query.spec_value:
            return self.query.adopt(text)
        return None

    def _on_revision(self) -> None:
        self.widen = 0
        self.reads_done = False
        self.compares = 0
        self.brand_filtered = False
        self.held_revision = True
        self.note(self._analysis())
        for pid, sku in list(self.index.cart):
            c = self.assess(pid, sku)
            if c["ok"] and not self.invalid(pid, sku):
                continue
            key = _vid(pid, sku)
            self.index.dead.add(key)
            if not self.query.model:
                self.index.dead_pids.add(str(pid))
            self.note("Carted item %s no longer matches the latest firm spec." % key)

    def _tighten(self, text: str) -> None:
        new = Query._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.inbox.updated = True
        self.brand_filtered = False
        self.widen = 0
        self.reads_done = False
        self.compares = 0
        self.note("Budget tightened to %.2f %s." % (new, self.query.currency))
        for pid, sku in list(self.index.cart):
            st = self.index.quote(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.index.dead.add(_vid(pid, sku))
                self.note("Carted item %s is now over budget." % _vid(pid, sku))

    def opening(self) -> list:
        if self.query.sealed:
            return []
        return self.policy.open(self)

    def stage(self) -> tuple | None:
        spec_i = self.spec_inspect()
        if spec_i:
            return spec_i
        good = self.backups()
        inspected = [c for c in good if c["key"] in self.index.inspected]
        seen_pids = {c["pid"] for c in inspected}
        if len(seen_pids) >= 2:
            return None
        for c in good:
            if c["pid"] not in seen_pids and "inspect_stock" in self.tools:
                return ("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
        if inspected:
            return None
        return self.read_more()

    def _brand_nudge(self) -> None:
        if self.nudge_brand:
            return
        self.nudge_brand = True
        self.note(
            "No %s product that meets every firm gate has been read yet. Filter that brand first."
            % self.query.pref_brand
        )

    def _exact_nudge(self) -> None:
        if self.nudge_exact or not self.query.model:
            return
        self.nudge_exact = True
        self.note(
            "Exact preference (%s%s) is not on record yet. Search the model and view every color."
            % (self.query.model, ", " + self.query.color if self.query.color else "")
        )

    def _blob(self, pid: str, sku: str) -> str:
        return " ".join(str(v) for v in self.facts(pid).values()) + " " + self.title(pid) + " " + self.options(pid, sku)

    def water_hits(self, pid: str, sku: str) -> int:
        blob = self._blob(pid, sku)
        hits = len(_WATER_RX.findall(blob))
        if re.search(r"\b(?:ipx|waterproof|water[\s-]?resist|sweat[\s-]?proof|splash)\b", blob, re.I):
            hits += 2
        return hits

    def sport_hits(self, pid: str, sku: str) -> int:
        return len(re.findall(r"\b(?:sport|gym|run|workout|earhook|secure|in-?ear)\b", self._blob(pid, sku), re.I))

    def wireless_hits(self, pid: str, sku: str) -> int:
        blob = self._blob(pid, sku)
        if _WIRED_RX.search(blob) and not _WIRELESS_RX.search(blob):
            return 0
        return 1 if _WIRELESS_RX.search(blob) else 0

    def battery_hits(self, pid: str, sku: str) -> int:
        return len(re.findall(r"\b(?:battery|mah|hours?|playback|playtime)\b", self._blob(pid, sku), re.I))

    def priority(self, pid: str, sku: str) -> int:
        wants = self._wishes()
        have = _tok(self._blob(pid, sku))
        hits = len(wants & have) if wants else 0
        if self.query.wants_water:
            hits += 2 * self.water_hits(pid, sku)
            blob = (self.title(pid) + " " + self.options(pid, sku)).lower()
            if _WIRELESS_RX.search(blob):
                hits += 2
            if _WIRED_RX.search(blob) and not _WIRELESS_RX.search(blob):
                hits -= 2
        return hits

    def _wishes(self) -> set:
        if self.wishes is None:
            said = self.query.goal + " " + " ".join(self.inbox.lines)
            drop = _stems(self.query.category or "") | {
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            }
            self.wishes = {w for w in _tok(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self.wishes

    def _claim_brand(self, pid: str) -> str:
        brand = (self.brand_of(pid) or "").strip()
        if brand:
            return brand
        for want in (self.query.brand, self.query.pref_brand):
            want = (want or "").strip()
            if want and _tok(want) <= _tok(self.title(pid)):
                return want
        return ""

    def _soft_reason(self, pid: str) -> dict | None:
        if "brand" in self.reason_fields:
            brand = self._claim_brand(pid)
            if brand:
                return {"field": "brand", "value": brand}
        facts = self.facts(pid)
        for field in self.reason_fields:
            if field == "brand":
                continue
            raw = facts.get(field)
            if raw is True or str(raw).strip().lower() in ("true", "yes", "supported"):
                return {"field": field, "value": "true"}
        return None

    def _claims(self, pid: str, sku: str) -> list:
        out = []
        field, want = self.query.spec_field, self.query.spec_value
        if field and want and field in _CLAIM_FIELDS:
            if self.fact_value(pid, field) == want:
                out.append({"field": field, "value": want})
        st = self.index.quote(pid, sku) or {}
        if st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        if st.get("in_stock"):
            out.append({"field": "in_stock", "value": "true"})
        soft = self._soft_reason(pid)
        if soft:
            out.append(soft)
        return out

    def must_claim(self) -> bool:
        return bool(self.need_claims or self.query.wants_reason)

    def order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.must_claim():
            claims = self._claims(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def legal(self, pid: str, sku: str) -> tuple:
        st = self.index.quote(pid, sku)
        k = _vid(pid, sku)
        if st is None:
            return False, "inspect_stock first"
        if (pid, sku) not in self.index.cart:
            return False, "add that variant first"
        if self.index.last_add_seq is not None and self.index.quote_seq.get(k, 0) <= self.index.last_add_seq:
            return False, "re-inspect after the cart addition"
        if not st.get("in_stock"):
            return False, "out of stock"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "over budget"
        if k in self.index.dead:
            return False, "invalidated"
        return True, ""

    def ready(self) -> bool:
        if self.index.adds:
            return True
        return self.policy.ready(self)

    def backups(self) -> list:
        dead = self.index.dead | (
            {_vid(pid, sku) for pid, sku in self.index.cart} if self.index.event_key else set()
        )
        return [c for c in self.candidates() if c["ok"] and c["key"] not in dead]

    def backup_ready(self) -> bool:
        if self.index.adds:
            return True
        if self.clock.left() < 40 or self.turn >= self.max_steps - 4:
            return True
        if not self.query.expect_event:
            return True
        good = self.backups()
        inspected = [c for c in good if c["key"] in self.index.inspected]
        distinct = {c["pid"] for c in inspected}
        if len(distinct) >= 2:
            return True
        if inspected:
            pending = self.spec_leads() or [
                pid
                for pid in self.unread()
                if self.title_has_spec(self.title(pid)) or str(pid) in self.index.spec_hits
            ]
            if not pending:
                return True
        return False

    def skus_of(self, pid: str) -> list[str]:
        pid = str(pid)
        out, seen = [], set()
        prod = self.index.pages.get(pid) or {}
        for v in prod.get("variants") or []:
            sku = v.get("sku") if isinstance(v, dict) else None
            if sku and str(sku) not in seen:
                seen.add(str(sku))
                out.append(str(sku))
        for key in list(self.index.quotes) + list(self.index.hits):
            p, _, sku = str(key).partition("::")
            if p == pid and sku and sku not in seen:
                seen.add(sku)
                out.append(sku)
        return out

    def _dead_keys(self) -> set:
        dead = self.index.event_key
        if not dead:
            return set()
        generic = _STOP | {
            "smartphone", "phone", "cellphone", "cellphones", "mobile", "sale",
            "original", "new", "5g", "4g", "lte", "android", "gb", "ram", "rom",
        }
        return _tok(self.title(dead.split("::", 1)[0])) - generic

    def deepen(self) -> tuple | None:
        """View a compared sibling that still lacks labelled facts."""
        if not self.index.event_key or "view" not in self.tools:
            return None
        keys = self._dead_keys()
        if len(keys) < 2:
            return None
        ranked = []
        for pid, prod in self.index.pages.items():
            if str(pid) in self.index.dead_pids:
                continue
            title = self.title(pid)
            if _ACCESSORY_RX.search(title) or _JUNK_RX.search(title) or _PART_RX.search(title):
                continue
            overlap = len(_tok(title) & keys)
            if overlap < 2:
                continue
            facts = prod.get("facts") if isinstance(prod, dict) else None
            if isinstance(facts, dict) and facts:
                continue
            ranked.append((-overlap, str(pid)))
        ranked.sort()
        if ranked:
            return ("view", {"product_id": ranked[0][1]})
        unread = []
        for pid in self.unread():
            title = self.title(pid)
            if _ACCESSORY_RX.search(title) or _JUNK_RX.search(title) or _PART_RX.search(title):
                continue
            overlap = len(_tok(title) & keys)
            if overlap >= 2:
                unread.append((-overlap, pid))
        unread.sort()
        if unread and "view" in self.tools:
            return ("view", {"product_id": unread[0][1]})
        return None

    def spec_inspect(self) -> tuple | None:
        """Inspect a viewed SKU whose facts already match the firm spec."""
        if not self.query.spec_field or "inspect_stock" not in self.tools:
            return None
        for pid in list(self.index.pages):
            if str(pid) in self.index.dead_pids:
                continue
            if _ACCESSORY_RX.search(self.title(pid)) or _JUNK_RX.search(self.title(pid)):
                continue
            if not self.brand_ok(pid) or not self.device_ok(pid, ""):
                continue
            got, _src = self.proof(pid, "", self.query.spec_field)
            if got != self.query.spec_value:
                continue
            for sku in self.skus_of(pid):
                key = _vid(pid, sku)
                if key in self.index.inspected or key in self.index.dead:
                    continue
                return ("inspect_stock", {"product_id": str(pid), "sku": sku})
        return None

    def fresh(self, pid: str, sku: str) -> bool:
        k = _vid(pid, sku)
        if k not in self.index.inspected:
            return False
        return self.index.last_add_seq is None or self.index.quote_seq.get(k, 0) > self.index.last_add_seq

    def _shortlist(self) -> None:
        cands = self.candidates()
        if self.query.listing and not self.index.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self.priority(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                self.note(
                    "Firm-valid listings: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:6])
                )
                return
        best = self.target()
        if best:
            self.note("Verified candidate: %s::%s at %.2f." % (best["pid"], best["sku"], best["price"] or 0.0))
            return
        field, want = self.query.spec_field, self.query.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            self.note(
                "No verified match yet. %s must resolve to %s: %s."
                % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in blocked[:3]))
            )
        elif blocked:
            self.note("Read products fail a firm gate. Search different listings.")

    def choose(self) -> tuple | None:
        if self.done:
            return None
        for pid, sku in list(self.index.cart):
            if self.invalid(pid, sku):
                self.kill(pid, sku)
        ask = self.policy.inquire(self)
        if ask:
            return ask
        if self.inbox.pending_q and "message" in self.tools:
            return ("message", {"content": self.reply()})
        if "remove_from_cart" in self.tools:
            for pid, sku in self.index.cart:
                if _vid(pid, sku) in self.index.dead:
                    return ("remove_from_cart", {"product_id": pid, "sku": sku})
        if "inspect_stock" in self.tools:
            for pid, sku in self.index.cart:
                if _vid(pid, sku) in self.index.dead:
                    continue
                if not self.fresh(pid, sku):
                    return ("inspect_stock", {"product_id": pid, "sku": sku})
        if "place_test_order" in self.tools:
            if not (self.query.family == "recall" and self.searches < 3 and self.turn < self.max_steps - 4):
                if not self.policy.hold(self):
                    for pid, sku in reversed(self.index.cart):
                        if _vid(pid, sku) in self.index.dead:
                            continue
                        if self.assess(pid, sku)["ok"] and self.fresh(pid, sku):
                            ok, _why = self.legal(pid, sku)
                            if ok:
                                return ("place_test_order", self.order_args(pid, sku))
        if self.index.event_key or self.index.dead:
            spec_i = self.spec_inspect()
            if spec_i:
                return spec_i
            hunt = self.policy.hunt(self)
            if hunt:
                return hunt
            target = self.target() or (self.closing_pick() if self.panic() or self.closing else None)
            if target:
                pid, sku = target["pid"], target["sku"]
                if str(pid) not in self.index.pages and "view" in self.tools:
                    return ("view", {"product_id": pid})
                if (pid, sku) not in self.index.cart:
                    if _vid(pid, sku) not in self.index.inspected and "inspect_stock" in self.tools:
                        return ("inspect_stock", {"product_id": pid, "sku": sku})
                    return ("add_to_cart", {"product_id": pid, "sku": sku})
            more = self.read_more()
            if more:
                return more
        found = self.policy.discover(self)
        if found:
            return found
        hold = self.policy.hold(self)
        if hold and self.index.cart and not self.index.event_key:
            backup = self.stage()
            if backup:
                return backup
            if "inspect_cart" in self.tools and self.cart_reads < 1:
                return ("inspect_cart", {})
        if not self.index.cart and not self.index.event_key and not self.index.dead:
            if self.query.expect_event and self.ready() and self.query.family in ("intent", "constraint"):
                staged = self.stage()
                if staged:
                    return staged
            if self.ready():
                target = self.target()
                if target:
                    why = self.policy.first_ok(self, target["pid"], target["sku"])
                    if why:
                        self.note("Hold first add: " + why)
                        return self.read_more()
                    if _vid(target["pid"], target["sku"]) not in self.index.inspected and "inspect_stock" in self.tools:
                        return ("inspect_stock", {"product_id": target["pid"], "sku": target["sku"]})
                    return ("add_to_cart", {"product_id": target["pid"], "sku": target["sku"]})
                return self.read_more()
            if not self.query.sealed:
                return self.read_more()
        return None

    def _ask(self) -> dict | None:
        for attempt in range(3):
            model = self.roster[min(self.model_i, len(self.roster) - 1)]
            resp = _CHAT.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": self.messages,
                    "tools": self.schema,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self.note("No tool call. Choose a supplied catalog tool.")
                continue
            if self.clock.left() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_i = min(self.model_i + 1, len(self.roster) - 1)
        return None

    def _retarget(self, envelope: dict, turn: int) -> dict:
        group = self.problem_id + "-turn-" + str(turn)
        calls = []
        for index, old in enumerate(envelope.get("calls") or [], start=1):
            action = old.get("action") if isinstance(old, dict) else None
            if not isinstance(action, dict):
                continue
            calls.append({"call_id": group + "-" + str(index), "action": action})
        return {**self.binding, "call_id": group, "idempotency_key": group, "turn": turn, "calls": calls}

    def _once(self, envelope: dict) -> tuple[dict | None, int | None, str]:
        result = _ENV.post_verbose("/environment/call", json_data=envelope)
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
        current = envelope
        self._last = current
        self._consumed = False
        for attempt in range(3):
            result, status, body = self._once(current)
            if result is not None:
                self._last = current
                return result
            if status == 409:
                if re.search(r"terminal observation", body, re.I):
                    self.done = True
                    self.pending = None
                    return None
                m = re.search(r"expected (\d+), got (\d+)", body)
                if m:
                    expected = int(m.group(1))
                    if expected > 0 and expected != current.get("turn"):
                        current = self._retarget(current, expected)
                        self._last = current
                        continue
                used = int(current.get("turn") or self.turn)
                if used > self.turn:
                    self.turn = used
                self._consumed = True
                self.pending = None
                return None
            if self.clock.left() < 20 or status in (401, 403):
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def act(self, name: str, args: dict, content: str = "", cid: str = "") -> dict | None:
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not self.closing:
                why = self.policy.first_ok(self, pid, sku)
                if why:
                    self.note("Blocked add: " + why)
                    return None
            c = self.assess(pid, sku)
            if not c["ok"]:
                stocked = bool((self.index.quotes.get(_vid(pid, sku)) or {}).get("in_stock"))
                if not (self.closing and c["viewed"] and stocked and c["key"] not in self.index.dead):
                    self.note("Blocked add: candidate is not firm-valid.")
                    return None
        try:
            return self.act_group([(name, args, content, cid)])
        except Exception as exc:
            self.note("Action %s failed (%s)." % (name, type(exc).__name__))
            return None

    def act_group(self, steps: list) -> dict | None:
        if self.pending is not None:
            pend = self.pending
            replay = self._send(pend["envelope"])
            if replay is None:
                if self._consumed or self.done:
                    self.pending = None
                    return None
                sent = self._last
                if sent is not None:
                    pend["envelope"] = sent
                    pend["turn"] = int(sent.get("turn") or pend["turn"])
                self.note("Environment pending; retrying the same turn.")
                return None
            self.pending = None
            sent = self._last or pend["envelope"]
            self.turn = int(sent.get("turn") or pend["turn"])
            return self._absorb(pend["steps"], pend.get("content") or "", replay)
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
                {"call_id": group + "-" + str(index), "action": {"name": name, "args": args}}
                for index, (name, args, _c, _i) in enumerate(steps, start=1)
            ],
        }
        content = next((c for _n, _a, c, _i in steps if c), "")
        result = self._send(envelope)
        sent = self._last or envelope
        used = int(sent.get("turn") or turn)
        if result is None:
            if self._consumed or self.done:
                return None
            self.pending = {"envelope": sent, "steps": steps, "content": content, "turn": used}
            self.note("Environment did not answer; the action will be retried.")
            return None
        self.turn = used
        return self._absorb(steps, content, result)

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

    def _absorb(self, steps: list, content: str, result: dict | None) -> dict | None:
        tool_calls = []
        for index, (name, args, _c, cid) in enumerate(steps, start=1):
            tid = cid or ("agent-" + str(self.turn) + "-" + str(index))
            tool_calls.append({"id": tid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            for name, args, _c, _i in steps:
                self.index.seen.discard(name + json.dumps(args, sort_keys=True))
            for call in tool_calls:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps({"error": "environment call failed"}),
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
            self.index.ingest(name, args, obs, self.query)
            self.index.seen.add(name + json.dumps(args, sort_keys=True))
            tid = cid or tool_calls[index]["id"]
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": _clip(json.dumps(obs), _CLIP)})
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                row = obs.get("added") or {}
                if row.get("product_id"):
                    added = (str(row["product_id"]), str(row.get("sku") or ""))
            if name in ("search", "filter"):
                self.searches += 1
            if name == "inspect_stock":
                ipid = str(args.get("product_id") or (obs or {}).get("product_id") or "")
                isku = str(args.get("sku") or (obs or {}).get("sku") or "")
                if ipid and (ipid, isku) in self.index.cart and self.invalid(ipid, isku):
                    self.kill(ipid, isku)
            if name == "inspect_cart":
                self.cart_reads += 1
            if name == "message":
                self.inbox.pending_q = False
            if wrap.get("done") or (isinstance(obs, dict) and obs.get("done")):
                self.done = True
        if added:
            self.awaiting = added
            if self.first_add_turn is None:
                self.first_add_turn = self.turn
        self.inbox.take(result, self)
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        if (
            self.recheck_turn is not None
            and self.turn > self.recheck_turn
            and not self.held_revision
            and not any(n == "place_test_order" for n, _a, _c, _i in steps)
        ):
            self.held_revision = True
        if any(n in ("search", "filter") for n, _a, _c, _i in steps) and self.query.need_ids:
            self.note(
                "Distinct matching products: %d of %d."
                % (len(self.index.clusters), self.query.need_ids)
            )
        if any(n in ("view", "compare") for n, _a, _c, _i in steps):
            self._shortlist()
        return last

    def recheck(self) -> None:
        pid, sku = self.awaiting
        self.awaiting = None
        obs = self.act("inspect_stock", {"product_id": pid, "sku": sku})
        if isinstance(obs, dict) and obs.get("error"):
            return
        st = self.index.quote(pid, sku) or {}
        if isinstance(obs, dict) and st.get("price") is None and obs.get("price") is not None:
            st = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.index.quotes[_vid(pid, sku)] = st
        stock = bool(st.get("in_stock"))
        price = st.get("price")
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over and not self.invalid(pid, sku):
            self.recheck_turn = self.turn
            hold = self.policy.hold(self)
            if hold:
                self.note("Post-add recheck: %s still live at %s. %s" % (_vid(pid, sku), price, hold))
            else:
                self.note("Post-add recheck: %s still live at %s." % (_vid(pid, sku), price))
            return
        self.kill(pid, sku)
        reason = "went out of stock" if not stock else "was repriced to %.2f" % (price or 0.0)
        self.note("Market change: %s %s." % (_vid(pid, sku), reason))
        self._shortlist()
        self._recover()

    def _recover(self) -> None:
        if self.done:
            return
        if self.inbox.pending_q and "message" in self.tools:
            self.act("message", {"content": self.reply()})
            if self.done:
                return
        for cpid, csku in list(self.index.cart):
            if self.done:
                return
            if _vid(cpid, csku) in self.index.dead and "remove_from_cart" in self.tools:
                self.act("remove_from_cart", {"product_id": cpid, "sku": csku})
        for _ in range(8):
            if self.done or self.turn >= self.max_steps:
                return
            if self.query.model and not self.twins_done:
                twin = self.twin_search()
                if twin is not None:
                    self.act(twin[0], twin[1])
                    continue
            if self.query.model and not self.sibs_done:
                sib = self.sibling_search()
                if sib is not None:
                    self.act(sib[0], sib[1])
                    continue
            target = self.target()
            if target:
                pid, sku = target["pid"], target["sku"]
                if _vid(pid, sku) not in self.index.inspected and "inspect_stock" in self.tools:
                    self.act("inspect_stock", {"product_id": pid, "sku": sku})
                    continue
                if (pid, sku) not in self.index.cart:
                    self.act("add_to_cart", {"product_id": pid, "sku": sku})
                    if self.done:
                        return
                if self.awaiting:
                    rpid, rsku = self.awaiting
                    self.awaiting = None
                    self.act("inspect_stock", {"product_id": rpid, "sku": rsku})
                    st = self.index.quote(rpid, rsku) or {}
                    price = st.get("price")
                    over = self.budget is not None and price is not None and price > self.budget + _EPS
                    if not st.get("in_stock") or over:
                        self.index.dead.add(_vid(rpid, rsku))
                        self.index.dead_pids.add(str(rpid))
                        continue
                if self.assess(pid, sku)["ok"] and self.legal(pid, sku)[0]:
                    self.act("place_test_order", self.order_args(pid, sku))
                    return
                continue
            forced = self.read_more()
            if forced is None:
                break
            self.act(forced[0], forced[1])
        if not self.done:
            self._close()

    def _buy(self, target: dict | None) -> bool:
        if not target or self.done:
            return False
        if self.turn >= self.max_steps and not self.closing:
            return False
        pid, sku = target["pid"], target["sku"]
        if str(pid) not in self.index.pages and "view" in self.tools:
            self.act("view", {"product_id": pid})
            if self.done:
                return True
        if _vid(pid, sku) not in self.index.inspected and "inspect_stock" in self.tools:
            self.act("inspect_stock", {"product_id": pid, "sku": sku})
            if self.invalid(pid, sku) or self.done:
                return False
        if (pid, sku) not in self.index.cart:
            self.act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done:
                return True
        if self.awaiting:
            rpid, rsku = self.awaiting
            self.awaiting = None
            self.act("inspect_stock", {"product_id": rpid, "sku": rsku})
            if self.invalid(rpid, rsku):
                if (rpid, rsku) in self.index.cart and "remove_from_cart" in self.tools:
                    self.act("remove_from_cart", {"product_id": rpid, "sku": rsku})
                return False
        if self.legal(pid, sku)[0] or self._closing_order_ok(pid, sku):
            self.act("place_test_order", self.order_args(pid, sku))
            return self.done
        return False

    def _closing_order_ok(self, pid: str, sku: str) -> bool:
        """When finishing the episode, order a live inspected cart item even if a firm gate failed."""
        if not self.closing:
            return False
        if (pid, sku) not in self.index.cart:
            return False
        st = self.index.quote(pid, sku) or {}
        if not st.get("in_stock"):
            return False
        if _vid(pid, sku) not in self.index.inspected:
            return False
        if self.index.last_add_seq is not None and self.index.quote_seq.get(_vid(pid, sku), 0) <= self.index.last_add_seq:
            return False
        return True

    def closing_pick(self) -> dict | None:
        """Best live quote to end the episode. Never drop a known firm spec."""
        rows = []
        for c in self.candidates():
            if c["key"] in self.index.dead:
                continue
            if c["live"]:
                rows.append(c)
        if not rows:
            for c in self.candidates():
                if c["key"] in self.index.dead:
                    continue
                st = self.index.quotes.get(c["key"]) or {}
                if st.get("in_stock") and c["viewed"]:
                    rows.append(c)
        honest = [c for c in rows if not self.decoy(c)]
        if honest:
            rows = honest
        firm = [c for c in rows if c["ok"]]
        if firm:
            rows = firm
        else:
            spec_ok = [c for c in rows if c["spec"] == 2 and c["viewed"] and self.looks_device(self.title(c["pid"]))]
            if spec_ok:
                rows = spec_ok
            else:
                branded = [c for c in rows if self.brand_ok(c["pid"]) and self.looks_device(self.title(c["pid"]))]
                if branded:
                    rows = branded
        if not rows:
            return None
        rows.sort(
            key=lambda c: (
                0 if c["ok"] else 1,
                0 if c["spec"] == 2 else 1,
                0 if c["viewed"] else 1,
                0 if c["key"] in self.index.inspected else 1,
                self.policy.rank(self, c, self.index.adds == 0),
            )
        )
        return rows[0]

    def _close(self) -> None:
        for cpid, csku in list(self.index.cart):
            if self.done:
                return
            if _vid(cpid, csku) in self.index.dead and "remove_from_cart" in self.tools:
                self.act("remove_from_cart", {"product_id": cpid, "sku": csku})
        for pid, sku in reversed(self.index.cart):
            if self.done:
                return
            if not self.assess(pid, sku)["ok"] and not self.closing:
                continue
            if self.legal(pid, sku)[0] or self._closing_order_ok(pid, sku):
                self.act("place_test_order", self.order_args(pid, sku))
                return
        good = self.backups() if self.index.event_key else [c for c in self.candidates() if c["ok"]]
        if not good:
            for _ in range(4):
                if self.done:
                    return
                forced = self.spec_inspect() or self.read_more()
                if forced is None:
                    break
                self.act(forced[0], forced[1])
                good = self.backups() if self.index.event_key else [c for c in self.candidates() if c["ok"]]
                if good:
                    break
        if not good:
            unread = [
                c
                for c in self.candidates()
                if c["live"]
                and c["viewed"]
                and c["key"] not in self.index.inspected
                and c["spec"] == 2
                and not _ACCESSORY_RX.search(self.title(c["pid"]))
            ]
            unread.sort(key=lambda c: (0 if c["ok"] else 1, c["price"] if c["price"] is not None else 1e12))
            for c in unread[:3]:
                if self.done:
                    return
                if "inspect_stock" in self.tools:
                    self.act("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
            good = self.backups() if self.index.event_key else [c for c in self.candidates() if c["ok"]]
        if not good:
            good = [
                c
                for c in self.candidates()
                if c["ok"] and c["key"] in self.index.inspected and (not self.query.theme or self.theme_ok(c["pid"], c["sku"]))
            ]
        if not good:
            pick = self.closing_pick()
            if pick and not self.done:
                self._buy(pick)
            return
        if self.done:
            return
        first = self.index.adds == 0
        if first and self.query.pref_brand:
            named = [c for c in good if self.pref_hit(c["pid"])]
            if named:
                good = named
            else:
                pick = self.closing_pick()
                if pick:
                    self._buy(pick)
                return
        good.sort(key=lambda c: self.policy.rank(self, c, first))
        self._buy(good[0])

    def last_resort(self) -> None:
        if self.done:
            return
        self.closing = True
        if self.inbox.pending_q and "message" in self.tools:
            self.act("message", {"content": self.reply()})
            if self.done:
                return
        for cpid, csku in list(self.index.cart):
            if self.done:
                return
            if _vid(cpid, csku) in self.index.dead and "remove_from_cart" in self.tools:
                self.act("remove_from_cart", {"product_id": cpid, "sku": csku})
        spec_i = self.spec_inspect()
        if spec_i:
            self.act(spec_i[0], spec_i[1])
            if self.done:
                return
        deep = self.deepen()
        if deep:
            self.act(deep[0], deep[1])
            if self.done:
                return
        if self._buy(self.target() or self.closing_pick()):
            return
        for _ in range(4):
            if self.done:
                return
            pick = self.target() or self.closing_pick()
            if pick and self._buy(pick):
                return
            spec_i = self.spec_inspect()
            if spec_i:
                self.act(spec_i[0], spec_i[1])
                continue
            unread = [
                c
                for c in self.candidates()
                if c["live"] and c["key"] not in self.index.dead and self.looks_device(self.title(c["pid"]))
            ]
            unread.sort(
                key=lambda c: (
                    0 if c["ok"] else 1,
                    0 if c["spec"] == 2 else 1,
                    0 if c["viewed"] else 1,
                    0 if c["key"] in self.index.inspected else 1,
                    c["price"] if c["price"] is not None else 1e12,
                )
            )
            if unread:
                c = unread[0]
                if str(c["pid"]) not in self.index.pages and "view" in self.tools:
                    self.act("view", {"product_id": c["pid"]})
                    continue
                if c["key"] not in self.index.inspected and "inspect_stock" in self.tools:
                    self.act("inspect_stock", {"product_id": c["pid"], "sku": c["sku"]})
                    continue
                if self._buy(c):
                    return
            break
        if self._buy(self.target() or self.closing_pick()):
            return
        self._close()
        if not self.done:
            self._force_order()

    def _force_order(self) -> None:
        """Terminate with whatever inspected in-stock SKU we have. Prefer budget-legal."""
        if self.done:
            return
        rows = [c for c in self.candidates() if c["key"] in self.index.inspected and c["key"] not in self.index.dead]
        in_stock = []
        for c in rows:
            st = self.index.quotes.get(c["key"]) or {}
            if st.get("in_stock"):
                in_stock.append(c)
        in_stock.sort(
            key=lambda c: (
                0 if c["live"] else 1,
                0 if c["ok"] else 1,
                0 if c["spec"] == 2 else 1,
                c["price"] if c["price"] is not None else 1e12,
            )
        )
        for c in in_stock:
            if self.done:
                return
            if self._buy(c):
                return
        for pid, sku in reversed(self.index.cart):
            if self.done:
                return
            if self.legal(pid, sku)[0] or self._closing_order_ok(pid, sku):
                self.act("place_test_order", self.order_args(pid, sku))
                return
            st = self.index.quote(pid, sku) or {}
            if st.get("in_stock") and _vid(pid, sku) in self.index.inspected:
                self.act("place_test_order", self.order_args(pid, sku))
                return

    def _refuse(self, name: str, args: dict) -> str | None:
        if name not in self.tools:
            return "unknown tool"
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.index.cart:
                pid, sku = self.index.cart[-1]
            ok, why = self.legal(pid, sku)
            if not ok and not self._closing_order_ok(pid, sku):
                return "order refused: " + why
            if not self.closing and self.query.need_ids and len(self.index.clusters) < self.query.need_ids:
                return "order refused: only %d of %d identities surfaced" % (len(self.index.clusters), self.query.need_ids)
            if not self.closing and self.query.listing and len(self.index.listings) < 2:
                return "order refused: need two listing_id values first"
            hold = None if self.closing else self.policy.hold(self)
            if hold:
                return "order refused: " + hold
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs product_id and sku"
            c = self.assess(pid, sku)
            key = _vid(pid, sku)
            st = self.index.quotes.get(key) or {}
            if not c["live"]:
                if not (self.closing and c["viewed"] and st.get("in_stock") and key not in self.index.dead):
                    return "variant is dead, over budget, or out of stock"
            if not c["viewed"] and not self.closing:
                return "view or compare first"
            if key not in self.index.inspected:
                return "inspect_stock that SKU first"
            if not self.closing:
                if c["spec"] != 2 and self.query.spec_field:
                    return "SKU does not resolve to %s = %s" % (self.query.spec_field, self.query.spec_value)
                if not c["ok"]:
                    return "fails a firm gate"
            if self.query.listing and not self.closing and len(self.index.listings) < 2:
                return "compare two listing_id values first"
            refused = None if self.closing else self.policy.first_ok(self, pid, sku)
            if refused:
                return refused
            if self.query.model and not self.index.adds and not self.closing and self.relax(pid, sku):
                return "first add must be the exact model and color"
            if not self.closing and not self.ready():
                return "not ready to commit"
        if name == "inspect_stock":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if pid and self.fresh(pid, sku):
                return "already inspected"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "empty message"
            if not self.inbox.pending_q and not self.query.sealed and "?" not in text:
                return "message is only for questions"
            if not self.inbox.pending_q and not self.query.sealed and self.inbox.questions >= 2:
                return "already asked twice"
        if name == "inspect_cart" and self.cart_reads >= 1:
            return "cart already known"
        key = name + json.dumps(args, sort_keys=True)
        if name == "search":
            if self.searches >= self.search_cap() and not self.query.need_ids:
                return "search budget used"
            if self.query.need_ids and self.searches >= 3:
                return "search budget used"
            fp = self.search_fp(str(args.get("query") or ""))
            if fp and fp in self.search_seen():
                return "duplicate search"
        if key in self.index.seen and name in ("search", "filter", "view", "compare", "inspect_stock"):
            return "duplicate call"
        return None

    def _filter(self, steps: list) -> list:
        names = [n for n, _a, _c, _i in steps]
        if "add_to_cart" in names and "place_test_order" in names:
            steps = [s for s in steps if s[0] != "place_test_order"]
        kept = []
        for name, args, content, cid in steps:
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.index.cart:
                    pid, sku = self.index.cart[-1]
                args = self.order_args(pid, sku)
            why = self._refuse(name, args)
            if why:
                if name in ("add_to_cart", "place_test_order", "message"):
                    self.note("Dropped " + name + ": " + why)
                continue
            kept.append((name, args, content, cid))
        return kept

    def panic(self) -> bool:
        # Hard tasks have 20 turns. Recovery after the first add needs view,
        # inspect, add, recheck, order — leave enough runway to finish.
        extra = 8 if self.max_steps <= 20 or self.query.family in ("constraint", "preference", "justification") else 5
        return self.clock.left() < 50 or self.turn >= self.max_steps - extra

    def run(self) -> list:
        self.note(self._analysis())
        last, repeats = None, 0
        while not self.done and self.turn < self.max_steps:
            try:
                if self.panic():
                    self.closing = True
                    self.last_resort()
                    break
                if not self.opened:
                    self.opened = True
                    opening = self.opening()
                    if opening:
                        self.act_group(opening)
                        continue
                if self.awaiting and "inspect_stock" in self.tools:
                    self.recheck()
                    continue
                plan = self.choose()
                if plan is None:
                    more = self.read_more() or self.policy.hunt(self) or self.next_search()
                    if more is not None:
                        plan = more
                if plan is not None:
                    name, args = plan
                    sig = name + json.dumps(args, sort_keys=True)
                    repeats = repeats + 1 if sig == last else 1
                    last = sig
                    if repeats > 2:
                        if name in ("inspect_stock", "add_to_cart"):
                            self.index.dead.add(_vid(args.get("product_id"), args.get("sku")))
                        if name == "message":
                            self.inbox.pending_q = False
                        last, repeats = None, 0
                        self.note("Repeated step changed nothing; moving on.")
                        continue
                    self.act(name, args)
                    continue
                last, repeats = None, 0
                if not self.panic():
                    proposal = self._ask()
                    if proposal is not None:
                        content = proposal.get("content") or ""
                        raw = []
                        for call in proposal["tool_calls"]:
                            name = (call.get("function") or {}).get("name") or ""
                            raw.append((name, _args(call), content, call.get("id") or ""))
                        steps = self._filter(raw)
                        if steps:
                            self.rejected = 0
                            if any(n == "message" for n, _a, _c, _i in steps) and not self.inbox.pending_q:
                                self.inbox.questions += 1
                            self.act_group(steps)
                            continue
                        self.rejected += 1
                        if self.rejected >= 3:
                            self._shortlist()
                            self.rejected = 0
                        if self.rejected == 1:
                            self.note("Those calls were dropped. Use a live catalog tool.")
                        continue
                    nxt = self.next_search() or self.read_more() or self.deepen() or self.spec_inspect()
                    if nxt:
                        self.act(nxt[0], nxt[1])
                        continue
                self.closing = True
                self.last_resort()
                break
            except Exception as exc:
                self.note("Turn failed (%s: %s)." % (type(exc).__name__, str(exc)[:120]))
                self.closing = True
                try:
                    self.last_resort()
                except Exception:
                    pass
                break
        if not self.done:
            self.closing = True
            try:
                self.last_resort()
            except Exception as exc:
                self.note("Close failed (%s)." % type(exc).__name__)
        return self.dialogue


def agent_main(problem_data: dict) -> list:
    kernel = None
    try:
        kernel = Kernel(problem_data)
        return kernel.run()
    except Exception as exc:
        if kernel is not None:
            try:
                kernel.closing = True
                kernel.last_resort()
                if kernel.dialogue:
                    return kernel.dialogue
            except Exception:
                pass
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main"]
