
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.agent.proxy_client import ProxyClient


_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)

_EPISODE_SECONDS = 268.0

_TAIL_STEPS = 4
_TAIL_SECONDS = 42.0

_OBS_CHARS = 6500
_WIRE_LIMIT = 56 * 1024
_EPS = 1e-9


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _tokens(text: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _title_tokens(title: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(title or "").lower()))


def _squash(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _same_word(a: Any, b: Any) -> bool:
    x, y = _squash(a).rstrip("s"), _squash(b).rstrip("s")
    return bool(x) and x == y


def _stems(text: Any) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", str(text or "").lower())}


def _one(values: Iterable) -> Optional[str]:
    vals = list(values)
    return vals[0] if len(vals) == 1 else None


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


_INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz\b", re.I)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_RAM_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_RAM_PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
_RAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_QUARTER_HD_RE = re.compile(r"\bqHD\b")

_PUBLISHED_PIXEL_STEPS: List[Tuple[str, int, int]] = [
    ("4k", 3840, 2160),
    ("qhd", 2560, 1440),
    ("fhd", 1920, 1080),
    ("hd", 1280, 720),
]
_PIXEL_STEPS: List[Tuple[str, int, int]] = list(_PUBLISHED_PIXEL_STEPS)

_CLASS_ALIASES = {
    "4k": r"\b(?:4k|uhd|ultra\s*hd|2160p?)\b",
    "qhd": r"\b(?:qhd|1440p?|2k)\b",
    "fhd": r"\b(?:fhd|1080p?|full\s?hd)\b",
    "hd": r"\b(?:hd|720p?)\b",
}

_CLASS_NAMES = {
    "4k": ("4k", "uhd", "ultra hd"),
    "qhd": ("qhd", "2k", "1440p"),
    "fhd": ("fhd", "full hd", "1080p"),
    "hd": ("hd", "720p"),
}


def load_pixel_classes(rules: str) -> None:
    global _PIXEL_STEPS
    _PIXEL_STEPS = list(_PUBLISHED_PIXEL_STEPS)
    found: Dict[str, Tuple[int, int]] = {}
    for m in re.finditer(
        r"\b(4k\s*/\s*uhd|4k|uhd|qhd|full\s*hd|fhd|hd)\b[^.;]{0,24}?\bat\s+(\d{3,5})\s*(?:by|x|×)\s*(\d{3,5})",
        rules or "",
        re.I,
    ):
        raw = re.sub(r"\s+", " ", m.group(1).strip().lower())
        key = {"4k/uhd": "4k", "uhd": "4k", "full hd": "fhd"}.get(raw, raw)
        if key in _CLASS_ALIASES:
            found[key] = (int(m.group(2)), int(m.group(3)))
    if len(found) >= 2:
        steps = [(k, max(w, h), min(w, h)) for k, (w, h) in found.items()]
        for key, long_side, short_side in _PIXEL_STEPS:
            if key not in found:
                steps.append((key, long_side, short_side))
        _PIXEL_STEPS = sorted(steps, key=lambda s: (-s[1], -s[2]))


def _pixel_class(long_side: int, short_side: int) -> Optional[str]:
    for name, need_long, need_short in _PIXEL_STEPS:
        if long_side >= need_long and short_side >= need_short:
            return name
    return None


def _inch_vals(text: Any) -> set:
    return {"%gin" % float(m.group(1)) for m in _INCH_RE.finditer(str(text or ""))}


def _hz_vals(text: Any) -> set:
    return {"%ghz" % float(m.group(1)) for m in _HZ_RE.finditer(str(text or ""))}


def _gb_vals(text: Any) -> set:
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in _GB_RE.finditer(str(text or ""))
    }


def _res_vals(text: Any) -> set:
    raw = str(text or "")
    if _QUARTER_HD_RE.search(raw):
        return set()
    low = raw.lower()
    out = set()
    for m in _WXH_RE.finditer(low):
        pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        found = _pixel_class(pair[0], pair[1])
        if found:
            out.add(found)
    stripped = re.sub(r"\b(?:full|ultra)\s?hd\b", " ", low)
    for name, pattern in _CLASS_ALIASES.items():
        probe = stripped if name == "hd" else low
        if re.search(pattern, probe, re.I):
            out.add(name)
    return out


_VALUE_READERS: Dict[str, Callable[[Any], set]] = {
    "screen_size": _inch_vals,
    "refresh_rate": _hz_vals,
    "resolution": _res_vals,
    "ram": _gb_vals,
}

_FACT_LABELS: Dict[str, Tuple[str, ...]] = {
    "screen_size": ("size", "screen_size", "display_size", "screen"),
    "resolution": ("display", "resolution", "screen_resolution"),
    "refresh_rate": ("refresh_rate", "display", "navigation"),
    "ram": ("storage", "memory", "ram"),
}

_BARE_UNITS = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}


_MEASURE_UNITS = (
    "gb", "tb", "mb", "mp", "mah", "wh", "mm", "cm", "kg", "lb", "ml", "l",
    "w", "hz", "khz", "ghz", "mhz", "rpm", "nits", "dpi", "ppi", "bit", "core",
)
_UNIT_ALIASES = {"inches": "in", "inch": "in", "grams": "g", "gram": "g", "litre": "l", "liter": "l"}


def _unit_vals(unit: str, text: Any) -> set:
    unit = _UNIT_ALIASES.get(str(unit or "").lower(), str(unit or "").lower())
    if not unit:
        return set()
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*" + re.escape(unit) + r"\b", re.I)
    return {"%g%s" % (float(m.group(1)), unit) for m in pattern.finditer(str(text or ""))}


def _spec_vals(field_name: str, text: Any, unit: str = "") -> set:
    raw = str(text or "")
    if field_name != "ram":
        reader = _VALUE_READERS.get(field_name)
        if reader:
            return reader(raw)
        return _unit_vals(unit, raw) if unit else set()
    paired = {"%ggb" % float(a) for a, _b in _RAM_PAIR_RE.findall(raw)}
    return paired or {"%ggb" % float(n) for n in _RAM_RE.findall(raw)}


def _mentions_spec(field_name: str, text: Any, unit: str = "") -> bool:
    raw = str(text or "")
    if field_name == "ram":
        return bool(_RAM_PAIR_HINT_RE.search(raw) or _RAM_RE.search(raw))
    if field_name == "screen_size":
        return bool(_INCH_RE.search(raw))
    if field_name == "refresh_rate":
        return bool(_HZ_RE.search(raw))
    if field_name == "resolution":
        if _WXH_RE.search(raw):
            return True
        return any(re.search(p, raw, re.I) for p in _CLASS_ALIASES.values())
    return bool(_unit_vals(unit, raw)) if unit else False


def _measure(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m else None


def _class_rank(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    names = [name for name, _l, _s in _PIXEL_STEPS]
    return (len(names) - names.index(value)) if value in names else None


_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|"
    r"harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
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
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|"
    r"charger|accessor(?:y|ies))\b))",
    re.I,
)
_ACCESSORY_RE = re.compile(
    r"\b(?:case|cover|protector|holder|mount|bracket|stand|strap|lanyard|cable|charger|adapter|"
    r"dock|clip|pouch|sleeve|skin|sticker|film|lens\s*protector|replacement\s*part|spare\s*part)\b",
    re.I,
)
_HEAD_TOKENS = 5
_COMPAT_WINDOW = 60
_REVIEW_TITLE_RE = re.compile(
    r"customer\s+(?:rating|review)|\breviews?\s+page\b|\bcompare\s+prices\b|refurb|pre-owned|restored|renewed",
    re.I,
)
_TRUNCATED_RE = re.compile(r"\[truncated\]", re.I)

SUPPORTED, CONTRADICTED, UNRESOLVED = "supported", "contradicted", "unresolved"


def _strip_negations(text: Any, is_title: bool = False) -> str:
    cleaned = _NEGATED_RE.sub(" ", str(text or ""))
    return _BUNDLE_RE.split(cleaned, maxsplit=1)[0] if is_title else cleaned


def phone_identity(facts: Dict[str, Any], title: str, options: str) -> str:
    if _PART_ONLY_RE.search(str(options or "")):
        return CONTRADICTED
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _strip_negations(raw_title, is_title=True)
    kind = " ".join(
        _strip_negations(facts.get(k) or "")
        for k in ("phone_type", "device_type", "description_device_type")
    )
    if _NOT_PHONE_RE.search(head + " " + kind):
        return CONTRADICTED
    if any(_TRUNCATED_RE.search(str(facts.get(k) or "")) for k in
           ("phone_type", "device_type", "device_operating_system")):
        return UNRESOLVED
    if _IS_PHONE_RE.search(head + " " + kind):
        return SUPPORTED
    os_blob = " ".join(
        _strip_negations(facts.get(k) or "")
        for k in ("device_operating_system", "operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS_RE.search(os_blob):
        return SUPPORTED
    if _OS_RE.search(head) and not re.search(r"\bcompatib", raw_title, re.I):
        return SUPPORTED
    return UNRESOLVED


HARD, SOFT = "hard", "soft"
VARIANT_SCOPE, PRODUCT_SCOPE = "selected_variant", "product"


@dataclass
class Requirement:

    rid: str
    field: str
    op: str
    value: Any
    unit: str = ""
    strength: str = HARD
    scope: str = PRODUCT_SCOPE
    source_text: str = ""
    origin: str = "request"
    priority: int = 0
    status: str = "active"
    revision: int = 0

    @property
    def concept(self) -> str:
        return self.field.split(":", 1)[1] if self.field.startswith("spec:") else self.field

    def describe(self) -> str:
        if self.field == "budget":
            return "at most %.2f" % float(self.value)
        if self.field == "in_stock":
            return "in stock"
        return "%s = %s" % (self.concept.replace("_", " "), self.value)


@dataclass
class PreferenceAxis:

    key: str
    label: str
    priority: int
    pattern: Optional[Any] = None
    kind: str = "feature"
    source_text: str = ""


@dataclass
class Objective:

    direction: str
    field: str
    source_text: str = ""


_NUM_PAT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|at most|up to|no more than|not more than|max(?:imum)? of|"
    r"under|below|within|budget (?:is|of)|can spend|can only go up to|go up to|spend up to)"
    r"[^\d]{0,16}" + _NUM_PAT + r"\s*(PHP|php|₱|pesos?|usd|USD)?",
    re.I,
)
_CURRENCY_RE = re.compile(r"\b(PHP|USD|EUR|SGD|MYR|IDR|THB|VND)\b")
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_LOOSE_KEY_RE = re.compile(r"([A-Za-z0-9_\-]{1,48})::([A-Za-z0-9._\-]+)")
_QUOTED_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")
_PHRASE_RE = re.compile(
    r"matches (?:the phrase\s+)?[\"“]?([A-Za-z0-9][^\"”,.]{1,80}?)[\"”]?\s*(?:,|\.|$)", re.I
)
_IDENTITIES_RE = re.compile(r"surface at least (\d+) distinct", re.I)
_LISTING_RE = re.compile(r"at least (two|three|\d+) distinct listing_id", re.I)
_CAT_PHRASE_RE = re.compile(r"in the ([A-Za-z][\w&'\-]*(?:\s+[&A-Za-z][\w&'\-]*){0,4}?) category", re.I)
_CAT_STATED_RE = re.compile(r"required catalog category is ([A-Za-z][\w&'\- ]{1,48}?)\s*(?:\.|,|$)", re.I)
_BRAND_STATED_RE = re.compile(r"required brand is ([A-Za-z0-9][\w&.'\- ]{1,38}?)\s*(?:\.|,|$)", re.I)
_PREF_BRAND_RE = re.compile(
    r"\b(?:ideally,?\s*)?i\s+(?:prefer|would prefer|like)\s+(?:the\s+)?([A-Za-z][\w&.'\-]*(?:\s+[A-Za-z][\w&.'\-]*){0,3}?)"
    r"\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)",
    re.I,
)
_PREF_ITEM_RE = re.compile(
    r"prefer\s+(?:the\s+)?((?:[A-Za-z]+\s+){1,3}?)([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I
)
_HIDDEN_SPEC_RE = re.compile(r"tell you if you ask|will tell you (?:the|which)|if you ask", re.I)
_QUESTION_RE = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)|"
    r"tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_SPEC_WORD_RE = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb|tb)\b",
    re.I,
)
_STOP_ITEM_WORDS = {
    "in", "stock", "an", "a", "the", "one", "test", "order", "place", "find", "get", "me",
    "want", "need", "buy", "instock", "new", "genuine", "actual", "device", "listing",
}
_ITEM_CLAUSE_RE = re.compile(
    r"\b(?:an?|one)\s+((?:[A-Za-z0-9(][\w&.'()/\"\-]*\s+){0,6}?[A-Za-z][\w&.'\-]*?)\s*"
    r"(?=,|\.|;|:|\bat or\b|\bat most\b|\bunder\b|\bbelow\b|\bwith\b|\bfor\b|\bthat\b|\bwhich\b|"
    r"\band\b|\bin stock\b|\?|$)",
    re.I,
)

_QUALIFIER_WORDS = {"gaming", "portable", "wireless", "over", "ear", "in", "on", "true"}


def _read_budget(text: str) -> Optional[float]:
    best = None
    for m in _BUDGET_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2) or "0"
        try:
            value = float(whole + "." + frac)
        except ValueError:
            continue
        if best is None or value < best:
            best = value
    return best


_OBJECTIVE_NOISE = frozenset({
    "valid", "available", "option", "options", "item", "items", "choice", "match",
    "matches", "one", "value", "deal", "possible", "suited", "suitable", "fit",
    "fitting", "product", "products", "listing", "listings", "seller", "way",
})


def _read_objectives(text: str, allow_unknown: bool = False) -> List[Objective]:
    directions = {
        "highest": "max", "largest": "max", "biggest": "max", "greatest": "max",
        "maximum": "max", "most": "max", "best": "max", "fastest": "max",
        "longest": "max", "heaviest": "max", "widest": "max", "thickest": "max",
        "loudest": "max", "newest": "max", "strongest": "max", "deepest": "max",
        "tallest": "max", "brightest": "max", "richest": "max", "finest": "max",
        "lowest": "min", "smallest": "min", "cheapest": "min", "least": "min",
        "minimum": "min", "shortest": "min", "lightest": "min", "slowest": "min",
        "narrowest": "min", "thinnest": "min", "quietest": "min", "oldest": "min",
    }
    fields = [
        ("refresh_rate", r"refresh\s*rate|refresh"),
        ("screen_size", r"screen(?:\s*size)?|display\s*size|panel\s*size"),
        ("resolution", r"resolution"),
        ("ram", r"\bram\b|memory"),
        ("price", r"price|cost"),
    ]
    out: List[Objective] = []
    seen = set()
    for m in re.finditer(
        r"(\w+\s+)?\b(%s)\b([^.;,]{0,40})" % "|".join(sorted(directions, key=len, reverse=True)),
        text or "",
        re.I,
    ):
        if (m.group(1) or "").strip().lower() == "at":
            continue
        word = m.group(2).lower()
        tail = m.group(3)
        for name, pattern in fields:
            if re.search(pattern, tail, re.I):
                if name in seen:
                    break
                seen.add(name)
                out.append(Objective(directions[word], name, m.group(0).strip()))
                break
        else:
            if not allow_unknown:
                continue
            noun = re.search(r"([a-z][a-z\-]{2,}(?:\s+[a-z][a-z\-]{2,})?)", tail, re.I)
            if noun:
                name = re.sub(r"\s+", "_", noun.group(1).strip().lower())
                if name not in seen and name not in _OBJECTIVE_NOISE:
                    seen.add(name)
                    out.append(Objective(directions[word], name, m.group(0).strip()))
    return out if len(out) >= 2 else []


_AXIS_LEXICON: List[Tuple[str, str, str, str]] = [
    ("water_resistance", "sweat and water resistance",
     r"waterproof|water[\s\-]?resist\w*|sweat[\s\-]?(?:proof|resist\w*)|splash[\s\-]?proof|"
     r"\bip[x]?\d{1,2}\b|rain[\s\-]?proof|weather[\s\-]?proof", "feature"),
    ("secure_fit", "a secure sport fit",
     r"\bsport\w*\b|\brunning\b|\bworkout\b|\bgym\b|ear[\s\-]?hook|ear[\s\-]?fin|ear[\s\-]?wing|"
     r"secure[\s\-]?fit|stay[\s\-]?put|non[\s\-]?slip|anti[\s\-]?slip|neckband", "feature"),
    ("battery_life", "battery that lasts",
     r"battery|playtime|play\s*time|playback|standby|runtime|hours?\b|hrs?\b", "duration"),
    ("wireless", "wireless freedom",
     r"\bwireless\b|\bbluetooth\b|\btws\b|\bcordless\b|no\s+cable|\bbt\s*5", "feature"),
    ("noise_cancelling", "noise cancelling",
     r"noise[\s\-]?cancel\w*|noise[\s\-]?isolat\w*|\banc\b|\benc\b", "feature"),
    ("comfort", "comfort",
     r"comfort\w*|ergonomic|soft[\s\-]?ear|memory[\s\-]?foam|lightweight", "feature"),
    ("sound_quality", "sound quality",
     r"hi[\s\-]?fi|stereo|bass|audio\s*quality|sound\s*quality|surround", "feature"),
    ("microphone", "microphone",
     r"\bmic\b|microphone|\bhd\s*call|call\s*quality", "feature"),
    ("fast_charge", "fast charging",
     r"fast[\s\-]?charg\w*|quick[\s\-]?charg\w*|\btype[\s\-]?c\b", "feature"),
]

_AXIS_CUES: List[Tuple[str, str]] = [
    ("water_resistance", r"\bsweat\w*\b|\bwater\b|water[\s\-]?resist|\bweather\b|\brain\b|\bsplash\w*\b|waterproof"),
    ("secure_fit", r"\bsecure\w*\b|\bfits?\b|stays? put|\bslip\w*\b|\bsports?\b|\bsnug\b"),
    ("battery_life", r"\bbattery\b|long[\s\-]?last\w*|\boutlast\w*\b|\bendurance\b|\bplaytime\b|\bcharge\b"),
    ("wireless", r"\bwireless\b|\bcables?\b|\bcords?\b|\bbluetooth\b|\bconnectiv\w*\b"),
    ("noise_cancelling", r"\bnoise\b|\bcancel\w*\b|\bisolat\w*\b"),
    ("comfort", r"\bcomfort\w*\b|\bergonom\w*\b|\blightweight\b"),
    ("sound_quality", r"\bsound\b|\baudio\b|\bbass\b"),
    ("microphone", r"\bmic\b|\bmicrophones?\b"),
    ("fast_charge", r"fast[\s\-]?charg\w*|quick[\s\-]?charg\w*"),
]

_AXIS_STOPWORDS = frozenset({
    "matters", "matter", "most", "more", "than", "less", "counts", "count", "then",
    "next", "least", "behind", "close", "prefer", "preference", "preferences",
    "priority", "priorities", "care", "about", "want", "need", "really", "just",
    "with", "that", "this", "what", "which", "stock", "budget", "price", "cheap",
    "cheapest", "under", "over", "have", "having", "would", "like", "good", "best",
    "also", "something", "anything", "myself", "actually", "looking", "ideal",
    "ideally", "important", "there", "their", "they", "from", "your", "into",
})

_NEGATION_RE = re.compile(
    r"\b(?:no|not|non|without|lacks?|missing|excluding)\b[\s\-]{0,3}$", re.I
)
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(hours?|hrs?|h\b|days?)", re.I
)


class RequestRecord:

    _NOT_A_PRODUCT = {
        "order", "orders", "item", "items", "listing", "listings", "thing", "things",
        "product", "products", "option", "options", "pick", "choice", "reason", "price",
        "budget", "stock", "one", "look", "way", "bit", "lot", "test",
    }

    def __init__(self, query: str) -> None:
        self.raw = str(query or "")
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.strip()
        self.rules_low = self.rules.lower()
        self.all_text = self.goal + "\n" + self.rules
        load_pixel_classes(self.rules)

        self.requirements: List[Requirement] = []
        self.axes: List[PreferenceAxis] = []
        self.objectives: List[Objective] = []
        self.candidate_keys: List[Tuple[str, str]] = []
        self.title_terms: List[str] = []
        self.category_qualifier: Optional[str] = None
        self.category_aliases: List[str] = []
        self.currency = "PHP"
        self.notes: List[str] = []
        self.revision = 0
        self._seq = 0

        self._read_currency()
        self._read_obligations()
        self._read_candidates()
        self._read_budget_stock()
        self._read_category_and_brand()
        self._read_specs()
        self._read_soft_brand()
        self._read_model_and_colour()
        self._read_title_phrase()
        self._read_measured_specs()
        self.objectives = (
            _read_objectives(self.goal, allow_unknown=True) or _read_objectives(self.rules)
        )
        self._read_axes(self.goal, base_priority=0)
        self._read_axes(self._preference_sentences(), base_priority=10)
        self._read_freeform_axes(self.goal)

    def _preference_sentences(self) -> str:
        keep = []
        for sentence in re.split(r"(?<=[.;])\s+", self.rules):
            if re.search(r"preferenc|priorit|matters more|counts most|soft preference", sentence, re.I):
                keep.append(sentence)
        return " ".join(keep)

    def _add(
        self,
        field_name: str,
        op: str,
        value: Any,
        *,
        unit: str = "",
        strength: str = HARD,
        scope: str = PRODUCT_SCOPE,
        source_text: str = "",
        origin: str = "request",
        priority: int = 0,
    ) -> Requirement:
        for old in self.requirements:
            if old.field == field_name and old.status == "active":
                old.status = "superseded"
        self._seq += 1
        req = Requirement(
            rid="r%d" % self._seq,
            field=field_name,
            op=op,
            value=value,
            unit=unit,
            strength=strength,
            scope=scope,
            source_text=_clip(source_text, 160),
            origin=origin,
            priority=priority,
            revision=self.revision,
        )
        self.requirements.append(req)
        return req

    def active(self, strength: Optional[str] = None) -> List[Requirement]:
        return [
            r for r in self.requirements
            if r.status == "active" and (strength is None or r.strength == strength)
        ]

    def get(self, field_name: str) -> Optional[Requirement]:
        for r in reversed(self.requirements):
            if r.field == field_name and r.status == "active":
                return r
        return None

    def value_of(self, field_name: str, default: Any = None) -> Any:
        req = self.get(field_name)
        return default if req is None else req.value

    @property
    def budget(self) -> Optional[float]:
        value = self.value_of("budget")
        return float(value) if value is not None else None

    @property
    def category(self) -> Optional[str]:
        return self.value_of("category")

    @property
    def brand(self) -> Optional[str]:
        return self.value_of("brand")

    @property
    def preferred_brand(self) -> Optional[str]:
        return self.value_of("preferred_brand")

    @property
    def preferred_model(self) -> Optional[str]:
        return self.value_of("preferred_model")

    @property
    def preferred_colour(self) -> Optional[str]:
        return self.value_of("preferred_colour")

    def specs(self) -> Dict[str, str]:
        return {
            r.concept: r.value for r in self.active()
            if r.field.startswith("spec:")
        }

    def _read_currency(self) -> None:
        m = _CURRENCY_RE.search(self.goal)
        if m:
            self.currency = m.group(1)

    def _read_obligations(self) -> None:
        rules = self.rules_low
        goal = self.goal.lower()
        m = _IDENTITIES_RE.search(rules)
        self.required_identities = int(m.group(1)) if m else 0
        m = _LISTING_RE.search(rules)
        words = {"two": 2, "three": 3}
        self.required_listings = (
            int(words.get(m.group(1).lower(), m.group(1))) if m and (m.group(1).isdigit() or m.group(1).lower() in words) else 0
        )
        self.title_phrase_rule = "against the listing title" in rules
        self.identity_cluster_rule = "publicly stated model identifier" in rules
        self.spec_hidden = bool(_HIDDEN_SPEC_RE.search(self.goal)) and not self.specs()
        self.clarify_priorities = (
            "clarify the user's priorities" in rules or "clarify the users priorities" in rules
        )
        self.exact_variant_rule = (
            "exact variant" in rules or "selected sku" in rules or "specific sku" in rules
        )
        self.device_identity_rule = bool(
            re.search(r"must explicitly say smartphone|raw phone/device type", self.all_text, re.I)
        )
        self.reject_accessories = "reject accessories" in rules or "not an accessory" in goal
        self.relaxation_rule = bool(
            re.search(r"another colou?r of the same model before|same model in another colou?r", self.all_text, re.I)
        )
        self.reason_requested = bool(
            re.search(r"\b(?:give me a reason|tell me why|walk me through why|why it'?s a good pick|"
                      r"reason i can trust|trustworthy reason)\b", self.goal, re.I)
        )
        self.first_add_is_binding = bool(
            re.search(r"first cart (?:addition|choice)|initial cart choice|initial purchase choice|"
                      r"genuine intended purchase|not a placeholder", rules)
        )
        self.rerank_after_event = "re-rank" in rules or "rerank" in rules

    def _read_candidates(self) -> None:
        self.candidate_keys = [
            (pid, sku.rstrip(".,;:")) for pid, sku in _KEY_RE.findall(self.goal)
        ]
        if not self.candidate_keys and re.search(r"candidate variants?\s*:", self.goal, re.I):
            tail = self.goal[re.search(r"candidate variants?\s*:", self.goal, re.I).end():]
            self.candidate_keys = [
                (pid, sku.rstrip(".,;:"))
                for pid, sku in _LOOSE_KEY_RE.findall(tail.split(".")[0])
            ]
        if self.candidate_keys:
            self._add(
                "candidate_set", "one_of", list(self.candidate_keys),
                scope=VARIANT_SCOPE, source_text="compare only these candidate variants",
            )

    def _read_budget_stock(self) -> None:
        budget = _read_budget(self.goal)
        if budget is not None:
            m = _BUDGET_RE.search(self.goal)
            self._add("budget", "le", budget, unit=self.currency,
                      scope=VARIANT_SCOPE, source_text=m.group(0) if m else "")
        if re.search(r"\bin[\s\-]?stock\b|\bavailable\b", self.goal, re.I) or "in-stock availability" in self.rules_low:
            self._add("in_stock", "is_true", True, scope=VARIANT_SCOPE,
                      source_text="in stock")

    def _item_clause(self) -> Optional[str]:
        for m in _ITEM_CLAUSE_RE.finditer(self.goal):
            phrase = _SPEC_WORD_RE.sub(" ", m.group(1))
            words = [w for w in re.split(r"[\s/()]+", phrase) if w]
            words = [w for w in words if _squash(w) and _squash(w) not in _STOP_ITEM_WORDS]
            if not words:
                continue
            if _squash(words[-1]) in self._NOT_A_PRODUCT:
                continue
            return " ".join(words)
        return None

    def _read_category_and_brand(self) -> None:
        stated = _CAT_PHRASE_RE.search(self.goal)
        clause = self._item_clause()
        words = clause.split() if clause else []

        if stated:
            self._add("category", "eq", stated.group(1).strip(),
                      source_text=stated.group(0))
            head = words[-1] if words else ""
            if not head or not (_stems(head) & _stems(stated.group(1))):
                return
        elif words:
            head = words[-1]
            if len(words) >= 2 and _squash(words[-2]) in _QUALIFIER_WORDS:
                self.category_qualifier = words[-2].lower()
            self._add("category", "eq", head, source_text=clause or "")

        if words:
            tail = words[:-1]
            if self.category_qualifier and tail and _squash(tail[-1]) == _squash(self.category_qualifier):
                tail = tail[:-1]
            maker = [
                w for w in tail
                if w[:1].isupper() and _squash(w) not in _QUALIFIER_WORDS and not w.isdigit()
            ]
            if maker:
                self._add("brand", "eq", " ".join(maker), source_text=clause or "")

    def _read_specs(self) -> None:
        for name in ("screen_size", "resolution", "refresh_rate", "ram"):
            found = _spec_vals(name, self.goal)
            value = _one(found)
            if not value:
                continue
            span = ""
            reader = _VALUE_READERS.get(name)
            for chunk in re.split(r"[,.;]", self.goal):
                if reader and value in _spec_vals(name, chunk):
                    span = chunk.strip()
                    break
            self._add("spec:" + name, "eq", value, unit=_BARE_UNITS.get(name, ""),
                      scope=VARIANT_SCOPE, source_text=span or self.goal[:120])

    def _read_measured_specs(self) -> None:
        known = {"ram", "memory", "screen", "display", "resolution", "refresh", "rate",
                 "size", "inch", "inches", "budget", "price", "php", "stock"}
        taken = {r.concept for r in self.active() if r.field.startswith("spec:")}
        pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*-?\s*(%s)\b[\s,]*(?:of\s+)?([A-Za-z][A-Za-z\-]{2,20})?"
            % "|".join(_MEASURE_UNITS),
            re.I,
        )
        for m in pattern.finditer(self.goal):
            unit = m.group(2).lower()
            noun = (m.group(3) or "").lower().strip("-")
            if unit in ("hz",) or not noun or noun in known or noun in taken:
                continue
            if noun in self.title_terms or m.group(1).lower() in self.title_terms:
                continue
            concept = noun.rstrip("s") if len(noun) > 3 else noun
            if concept in taken or "spec:" + concept in {r.field for r in self.active()}:
                continue
            value = "%g%s" % (float(m.group(1)), unit)
            taken.add(concept)
            self._add("spec:" + concept, "eq", value, unit=unit, scope=VARIANT_SCOPE,
                      source_text=m.group(0).strip())

    def _read_soft_brand(self) -> None:
        if _PREF_ITEM_RE.search(self.goal):
            return
        m = _PREF_BRAND_RE.search(self.goal)
        if not m:
            return
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return
        if self.brand and _squash(self.brand) == _squash(name):
            return
        self._add("preferred_brand", "eq", name, strength=SOFT, priority=0,
                  source_text=m.group(0))

    def _read_model_and_colour(self) -> None:
        m = _PREF_ITEM_RE.search(self.goal)
        if not m:
            return
        colour = " ".join(w for w in m.group(1).split() if w.lower() not in ("the", "a", "an"))
        model = m.group(2)
        self._add("preferred_model", "eq", model, strength=SOFT, priority=0,
                  scope=PRODUCT_SCOPE, source_text=m.group(0))
        if colour:
            self._add("preferred_colour", "eq", colour.lower(), strength=SOFT, priority=1,
                      scope=VARIANT_SCOPE, source_text=m.group(0))

    def _read_title_phrase(self) -> None:
        m = _PHRASE_RE.search(self.goal)
        source = m.group(1) if m else ""
        if not source:
            quoted = _QUOTED_RE.findall(self.goal)
            source = quoted[0] if quoted else ""
        terms = [t for t in re.findall(r"[a-z0-9]+", source.lower())]
        if terms and (self.title_phrase_rule or self.required_identities):
            self.title_terms = terms
            self._add("title_terms", "contains", terms, source_text=source)

    def _read_axes(self, text: str, base_priority: int) -> None:
        lexicon = {key: (label, pattern, kind) for key, label, pattern, kind in _AXIS_LEXICON}
        found: List[Tuple[int, str, str]] = []
        for key, cue in _AXIS_CUES:
            m = re.search(cue, text or "", re.I)
            if m and key in lexicon:
                found.append((m.start(), key, m.group(0)))
        found.sort()
        known = {a.key for a in self.axes}
        for order, (_pos, key, span) in enumerate(found):
            if key in known:
                continue
            label, pattern, kind = lexicon[key]
            self.axes.append(
                PreferenceAxis(
                    key=key,
                    label=label,
                    priority=base_priority + order,
                    pattern=re.compile(pattern, re.I),
                    kind=kind,
                    source_text=span,
                )
            )
            known.add(key)

    def _read_freeform_axes(self, text: str) -> None:
        if self.axes or self.preferred_brand or self.preferred_model:
            return
        marker = re.compile(r"\bmatters?\b|\bcounts?\b|care about|priorit|important", re.I)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if marker.search(s)]
        if not sentences:
            return
        drop = _AXIS_STOPWORDS | _tokens(self.category or "")
        for clause in re.split(r"[,;]| then |\bthen\b", " ".join(sentences)):
            words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", clause.lower()) if w not in drop]
            if not words:
                continue
            key = "_".join(words[:2])
            if any(a.key == key for a in self.axes):
                continue
            pattern = "|".join(r"\b%s\w*\b" % re.escape(w) for w in words[:3])
            self.axes.append(
                PreferenceAxis(
                    key=key,
                    label=clause.strip()[:60] or key,
                    priority=len(self.axes),
                    pattern=re.compile(pattern, re.I),
                    kind="feature",
                    source_text=clause.strip()[:80],
                )
            )
            if len(self.axes) >= 4:
                return

    @staticmethod
    def _axis_order(text: str) -> List[str]:
        hits = []
        for key, cue in _AXIS_CUES:
            m = re.search(cue, text or "", re.I)
            if m:
                hits.append((m.start(), key))
        return [key for _pos, key in sorted(hits)]

    def apply_shopper_message(self, text: str) -> List[str]:
        said = str(text or "")
        if not said.strip():
            return []
        self.revision += 1
        changes: List[str] = []

        budget = _read_budget(said)
        if budget is not None and (self.budget is None or abs(budget - self.budget) > _EPS):
            self._add("budget", "le", budget, unit=self.currency, scope=VARIANT_SCOPE,
                      source_text=said[:120], origin="shopper")
            changes.append("budget is now at most %.2f %s" % (budget, self.currency))

        m = _CAT_STATED_RE.search(said)
        if m and not _same_word(m.group(1), self.category or ""):
            self._add("category", "eq", m.group(1).strip(), source_text=said[:120], origin="shopper")
            changes.append("required category is %s" % m.group(1).strip())

        m = _BRAND_STATED_RE.search(said)
        if m and not _same_word(m.group(1), self.brand or ""):
            self._add("brand", "eq", m.group(1).strip(), source_text=said[:120], origin="shopper")
            changes.append("required brand is %s" % m.group(1).strip())

        for name in ("screen_size", "resolution", "refresh_rate", "ram"):
            value = _one(_spec_vals(name, said))
            if not value:
                continue
            current = self.value_of("spec:" + name)
            if current == value:
                continue
            self._add("spec:" + name, "eq", value, unit=_BARE_UNITS.get(name, ""),
                      scope=VARIANT_SCOPE, source_text=said[:120], origin="shopper")
            self.spec_hidden = False
            changes.append("%s must be %s" % (name.replace("_", " "), value))

        before = {a.key for a in self.axes}
        stated = self._axis_order(said)
        self._read_axes(said, base_priority=0)
        self._read_freeform_axes(said)
        gained = [a.label for a in self.axes if a.key not in before]
        if len(stated) >= 3:
            for axis in self.axes:
                axis.priority = stated.index(axis.key) if axis.key in stated else len(stated)
        self.axes.sort(key=lambda a: a.priority)
        for index, axis in enumerate(self.axes):
            axis.priority = index
        if gained:
            changes.append("preferences: " + ", ".join(gained))

        if changes:
            self.notes.extend(changes)
        return changes

    def summary(self) -> str:
        parts = [r.describe() for r in self.active(HARD)]
        soft = [r.describe() for r in self.active(SOFT)]
        text = "firm: " + ("; ".join(parts) if parts else "none stated")
        if soft:
            text += " | preferred: " + "; ".join(soft)
        if self.axes:
            text += " | priorities: " + ", ".join(a.label for a in sorted(self.axes, key=lambda a: a.priority))
        if self.objectives:
            text += " | order by: " + ", ".join("%s %s" % (o.direction, o.field) for o in self.objectives)
        return text


class Budget:

    def __init__(self, seconds: float, max_steps: int) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.max_steps = max(1, int(max_steps))
        self.steps_used = 0

    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def allowance(self, reserve_after: float = 0.0, tail: float = 8.0, share: float = 0.5) -> float:
        usable = self.time_left() - tail
        if usable <= 0:
            return 0.0
        if reserve_after:
            return max(0.0, min(usable, usable - reserve_after))
        return max(0.0, min(usable, max(usable * share, 20.0)))

    def can_afford(self, need: float, reserve_after: float = 0.0) -> bool:
        return self.allowance(reserve_after) >= need

    def steps_left(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    def in_endgame(self) -> bool:
        return self.steps_left() <= _TAIL_STEPS or self.time_left() <= _TAIL_SECONDS

    def exploring(self, fraction: float = 0.55) -> bool:
        return (
            self.steps_used < self.max_steps * fraction
            and self.time_left() > _TAIL_SECONDS + 60.0
        )


class ToolCatalog:

    def __init__(self, tools: Sequence[dict]) -> None:
        self.schemas: Dict[str, dict] = {}
        for tool in tools or []:
            fn = (tool or {}).get("function") or {}
            name = fn.get("name")
            if name:
                self.schemas[str(name)] = fn
        self.names = set(self.schemas)

    def has(self, name: str) -> bool:
        return name in self.schemas

    def params(self, name: str) -> dict:
        return (self.schemas.get(name) or {}).get("parameters") or {}

    def properties(self, name: str) -> dict:
        props = self.params(name).get("properties")
        return props if isinstance(props, dict) else {}

    def enum_for(self, name: str, *path: str) -> List[str]:
        node: Any = self.params(name)
        for step in path:
            props = (node or {}).get("properties") or {}
            node = props.get(step)
            if node is None:
                return []
            if node.get("type") == "array":
                node = node.get("items") or {}
        values = (node or {}).get("enum")
        return [str(v) for v in values] if isinstance(values, list) else []

    def requires(self, name: str, *path: str) -> bool:
        node: Any = self.params(name)
        for step in path:
            required = node.get("required") if isinstance(node, dict) else None
            if not isinstance(required, list) or step not in required:
                return False
            props = (node or {}).get("properties") or {}
            node = props.get(step) or {}
        return True

    def coerce(self, name: str, args: dict) -> dict:
        props = self.properties(name)
        params = self.params(name)
        out: Dict[str, Any] = {}
        for key, value in (args or {}).items():
            spec = props.get(key)
            if spec is None:
                if params.get("additionalProperties") is False:
                    continue
                out[key] = value
                continue
            out[key] = self._coerce_value(spec, value)
        return out

    def _coerce_value(self, spec: dict, value: Any) -> Any:
        kind = spec.get("type")
        if kind == "integer":
            number = _num(value)
            if number is not None:
                value = int(number)
        elif kind == "number":
            number = _num(value)
            if number is not None:
                value = number
        elif kind == "boolean" and isinstance(value, str):
            value = value.strip().lower() in ("true", "yes", "1")
        elif kind == "string" and value is not None and not isinstance(value, str):
            value = str(value)
        elif kind == "array" and not isinstance(value, list):
            value = [value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.get("maximum") is not None:
                value = min(value, spec["maximum"])
            if spec.get("minimum") is not None:
                value = max(value, spec["minimum"])
        return value

    def validate(self, name: str, args: dict) -> Optional[str]:
        if name not in self.schemas:
            return "unknown tool %r; available: %s" % (name, ", ".join(sorted(self.names)))
        params = self.params(name)
        props = self.properties(name)
        for key in params.get("required") or []:
            if args.get(key) in (None, ""):
                return "%s requires %s" % (name, key)
        if params.get("additionalProperties") is False:
            extra = [k for k in (args or {}) if k not in props]
            if extra:
                return "%s does not accept %s" % (name, ", ".join(sorted(extra)))
        for key, value in (args or {}).items():
            spec = props.get(key)
            if not isinstance(spec, dict) or value is None:
                continue
            problem = self._check_value(spec, value)
            if problem:
                return "%s.%s %s" % (name, key, problem)
        return None

    def _check_value(self, spec: dict, value: Any) -> Optional[str]:
        kind = spec.get("type")
        checks = {
            "string": str,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        if kind in checks and not isinstance(value, checks[kind]):
            return "must be a %s" % kind
        if kind in ("number", "integer") and (isinstance(value, bool) or _num(value) is None):
            return "must be a %s" % kind
        if isinstance(spec.get("enum"), list) and value not in spec["enum"]:
            return "must be one of %s" % ", ".join(str(v) for v in spec["enum"])
        number = _num(value) if kind in ("number", "integer") else None
        if number is not None:
            if spec.get("maximum") is not None and number > spec["maximum"] + _EPS:
                return "must be at most %s" % spec["maximum"]
            if spec.get("minimum") is not None and number < spec["minimum"] - _EPS:
                return "must be at least %s" % spec["minimum"]
        if kind == "array":
            if spec.get("minItems") and len(value) < spec["minItems"]:
                return "needs at least %d entries" % spec["minItems"]
            item_spec = spec.get("items")
            if isinstance(item_spec, dict):
                for item in value:
                    problem = self._check_value(item_spec, item)
                    if problem:
                        return "entries " + problem
        if kind == "object" and isinstance(spec.get("properties"), dict):
            for key in spec.get("required") or []:
                if key not in value:
                    return "needs %s" % key
            if spec.get("additionalProperties") is False:
                extra = [k for k in value if k not in spec["properties"]]
                if extra:
                    return "does not accept %s" % ", ".join(sorted(extra))
            for key, inner in value.items():
                inner_spec = spec["properties"].get(key)
                if isinstance(inner_spec, dict) and inner is not None:
                    problem = self._check_value(inner_spec, inner)
                    if problem:
                        return "%s %s" % (key, problem)
        return None


class EnvironmentClient:

    def __init__(self, binding: dict, problem_id: str, budget: Budget) -> None:
        self.binding = dict(binding or {})
        self.problem_id = str(problem_id)
        self.budget = budget
        self.turn = 0
        self.pending: Optional[dict] = None
        self.transport_failures = 0

    def envelope(self, actions: Sequence[Tuple[str, dict]]) -> dict:
        turn = self.turn + 1
        group = "%s-turn-%d" % (self.problem_id, turn)
        return {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [
                {
                    "call_id": "%s-%d" % (group, index),
                    "action": {"name": name, "args": args},
                }
                for index, (name, args) in enumerate(actions, start=1)
            ],
        }

    def send(self, envelope: dict) -> Optional[dict]:
        wanted = len(envelope.get("calls") or [])
        for attempt in range(4):
            result = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and isinstance(result.get("calls"), list):
                if len(result["calls"]) == wanted:
                    return result
            self.transport_failures += 1
            if self.budget.time_left() < 20:
                break
            time.sleep(min(1.5 * (attempt + 1), 5.0))
        return None

    def call(self, actions: Sequence[Tuple[str, dict]]) -> Tuple[Optional[dict], int]:
        envelope = self.envelope(actions)
        result = self.send(envelope)
        if result is not None:
            self.turn = envelope["turn"]
            self.budget.steps_used = self.turn
        return result, envelope["turn"]


def _extract_json(text: str) -> Optional[dict]:
    raw = str(text or "")
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError):
        pass
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            pass
    start = raw.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start:index + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except (TypeError, ValueError):
                        break
        start = raw.find("{", start + 1)
    return None


class Inference:

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        self.roster = ((pinned,) + tuple(m for m in roster if m != pinned)) if pinned else roster
        self.index = 0
        self.calls = 0
        self.failures = 0

    def ask(
        self,
        prompt: str,
        *,
        system: str = "",
        expect_json: bool = True,
        tries: int = 2,
        reserve_after: float = _TAIL_SECONDS,
        temperature: float = 0.0,
    ) -> Optional[Any]:
        if not self.budget.can_afford(8.0, reserve_after):
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for attempt in range(max(1, tries)):
            if not self.budget.can_afford(8.0, reserve_after):
                break
            model = self.roster[min(self.index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature + 0.2 * attempt,
            }
            self.calls += 1
            response = _PROXY.post("/inference/chat/completions", json_data=payload)
            content = ""
            if isinstance(response, dict) and response.get("choices"):
                content = (response["choices"][0].get("message") or {}).get("content") or ""
            if content:
                if not expect_json:
                    return content
                parsed = _extract_json(content)
                if parsed is not None:
                    return parsed
            self.failures += 1
            self.index = min(self.index + 1, len(self.roster) - 1)
        return None


SOURCE_RANK = {"row": 0, "view": 1, "compare": 1, "cart": 2, "inspect": 3, "order": 3}


@dataclass
class VariantState:

    pid: str
    sku: str
    price: Optional[float] = None
    in_stock: Optional[bool] = None
    options: str = ""
    seq: int = 0
    source: str = "row"
    turn: int = 0

    @property
    def key(self) -> str:
        return self.pid + "::" + self.sku


@dataclass
class ProductRecord:

    pid: str
    title: str = ""
    brand: str = ""
    category_path: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    listing_id: str = ""
    variant_order: List[str] = field(default_factory=list)
    inspected: bool = False
    seq: int = 0
    turn: int = 0

    def fact_text(self) -> str:
        return " ".join("%s %s" % (k, v) for k, v in (self.facts or {}).items())


class EvidenceStore:
    def __init__(self) -> None:
        self.products: Dict[str, ProductRecord] = {}
        self.variants: Dict[str, VariantState] = {}
        self.rows: Dict[str, dict] = {}
        self.surfaced: set = set()
        self.search_surfaced: set = set()
        self.queries: List[str] = []
        self.cart: List[Tuple[str, str]] = []
        self.events: List[dict] = []
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.order: Optional[dict] = None
        self.seq = 0
        self.adds = 0
        self.first_add_seq: Optional[int] = None
        self.last_add_seq: Optional[int] = None
        self.event_target: Optional[str] = None

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return "%s::%s" % (pid, sku or "")

    def product(self, pid: Any) -> ProductRecord:
        return self.products.get(str(pid)) or ProductRecord(pid=str(pid))

    def title(self, pid: Any) -> str:
        record = self.products.get(str(pid))
        if record and record.title:
            return record.title
        for key, row in self.rows.items():
            if key.startswith(str(pid) + "::") and row.get("title"):
                return str(row["title"])
        return ""

    def brand(self, pid: Any) -> str:
        record = self.products.get(str(pid))
        if record and record.brand:
            return record.brand
        for key, row in self.rows.items():
            if key.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def facts(self, pid: Any) -> Dict[str, Any]:
        return self.product(pid).facts or {}

    def options(self, pid: Any, sku: Any) -> str:
        state = self.variants.get(self.key(pid, sku))
        return state.options if state else ""

    def state(self, pid: Any, sku: Any) -> Optional[VariantState]:
        return self.variants.get(self.key(pid, sku))

    def inspected(self, pid: Any) -> bool:
        return self.product(pid).inspected

    def verified(self, pid: Any, sku: Any) -> bool:
        state = self.state(pid, sku)
        if state is None or state.source not in ("inspect", "cart"):
            return False
        return self.last_add_seq is None or state.seq > self.last_add_seq

    def note_query(self, name: str, args: dict) -> None:
        if name != "search":
            return
        query = re.sub(r"\s+", " ", str((args or {}).get("query") or "")).strip().lower()
        if query and query not in self.queries:
            self.queries.append(query)

    def record(self, name: str, args: dict, obs: Any, turn: int) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        body = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
        if not isinstance(body, dict):
            return
        if name in ("search", "filter"):
            for row in body.get("results") or []:
                if isinstance(row, dict):
                    self._note_row(row, name)
        elif name == "view":
            self._note_product(body)
        elif name == "compare":
            for product in body.get("compared") or []:
                if isinstance(product, dict):
                    self._note_product(product)
        elif name == "inspect_stock":
            self._note_state(
                body.get("product_id"), body.get("sku"),
                _num(body.get("price")), body.get("in_stock"), "inspect", turn,
            )
        elif name == "add_to_cart":
            added = body.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                if (str(pid), str(sku)) not in self.cart:
                    self.cart.append((str(pid), str(sku)))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
                self._note_state(pid, sku, _num(added.get("price")), added.get("in_stock"),
                                 "cart", turn)
            self._sync_cart(body, turn)
            self._note_event(body.get("market_event"), turn)
        elif name == "remove_from_cart":
            self._sync_cart(body, turn)
            removed = body.get("removed") or {}
            pair = (str(removed.get("product_id")), str(removed.get("sku")))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._sync_cart(body, turn)
        elif name == "place_test_order":
            if body.get("order"):
                self.order = body["order"]
        if body.get("market_event"):
            self._note_event(body["market_event"], turn)

    def _note_row(self, row: dict, action: str) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        pid = str(pid)
        self.surfaced.add(pid)
        if action == "search":
            self.search_surfaced.add(pid)
        record = self.products.get(pid)
        if record is None:
            record = ProductRecord(pid=pid, title=str(row.get("title") or ""),
                                   brand=str(row.get("brand") or ""), seq=self.seq)
            self.products[pid] = record
        else:
            record.title = record.title or str(row.get("title") or "")
            record.brand = record.brand or str(row.get("brand") or "")
        sku = row.get("sku")
        if not sku:
            return
        self.rows[self.key(pid, sku)] = row
        self._note_state(pid, sku, _num(row.get("price")), row.get("in_stock"), "row", 0)

    def _note_product(self, product: dict) -> None:
        pid = product.get("product_id")
        if not pid:
            return
        pid = str(pid)
        record = self.products.get(pid) or ProductRecord(pid=pid)
        record.title = str(product.get("title") or record.title)
        record.brand = str(product.get("brand") or record.brand or "")
        path = product.get("category_path")
        if isinstance(path, list):
            record.category_path = [str(p) for p in path]
        facts = product.get("facts")
        if isinstance(facts, dict):
            record.facts = dict(facts)
        if product.get("listing_id"):
            record.listing_id = str(product["listing_id"])
        record.inspected = True
        record.seq = self.seq
        self.products[pid] = record
        for variant in product.get("variants") or []:
            if not isinstance(variant, dict) or not variant.get("sku"):
                continue
            sku = str(variant["sku"])
            if sku not in record.variant_order:
                record.variant_order.append(sku)
            self._note_state(pid, sku, _num(variant.get("price")), variant.get("in_stock"),
                             "view", 0, options=str(variant.get("options") or ""))

    def _note_state(
        self,
        pid: Any,
        sku: Any,
        price: Optional[float],
        in_stock: Any,
        source: str,
        turn: int,
        options: Optional[str] = None,
    ) -> None:
        if not pid or not sku:
            return
        key = self.key(pid, sku)
        state = self.variants.get(key)
        if state is None:
            state = VariantState(pid=str(pid), sku=str(sku))
            self.variants[key] = state
        fresher = self.seq > state.seq or (
            self.seq == state.seq and SOURCE_RANK.get(source, 0) >= SOURCE_RANK.get(state.source, 0)
        )
        if options is not None and (options or not state.options):
            state.options = options
        if fresher:
            if price is not None:
                state.price = price
            if in_stock is not None:
                state.in_stock = bool(in_stock)
            state.seq = self.seq
            state.source = source
            state.turn = turn or state.turn

    def _sync_cart(self, body: dict, turn: int) -> None:
        rows = body.get("cart")
        if not isinstance(rows, list):
            return
        self.cart = [
            (str(r.get("product_id")), str(r.get("sku")))
            for r in rows if isinstance(r, dict) and r.get("product_id")
        ]
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            price = _num(r.get("current_price"))
            if price is None:
                price = _num(r.get("price"))
            stock = r.get("current_in_stock")
            if stock is None:
                stock = r.get("in_stock")
            self._note_state(r.get("product_id"), r.get("sku"), price, stock, "cart", turn)

    def _note_event(self, event: Any, turn: int) -> None:
        if not isinstance(event, dict):
            return
        self.events.append(event)
        target = event.get("target") or {}
        pid, sku = target.get("product_id"), target.get("sku")
        if not pid or not sku:
            return
        key = self.key(pid, sku)
        self.event_target = key
        price = _num(event.get("new_price"))
        stock = event.get("in_stock_after")
        self._note_state(pid, sku, price, stock, "inspect", turn)
        if event.get("kind") == "stockout" or stock is False:
            self.dead_keys.add(key)
            self.dead_products.add(str(pid))

    def matching_products(self, terms: Sequence[str]) -> List[str]:
        if not terms:
            return sorted(self.surfaced | set(self.products))
        out = []
        for pid in set(self.surfaced) | set(self.products):
            tokens = _title_tokens(self.title(pid))
            if all(term in tokens for term in terms):
                out.append(pid)
        return sorted(out)

    def identity_clusters(self, terms: Sequence[str]) -> List[List[str]]:
        pids = self.matching_products(terms)
        parent: Dict[str, str] = {p: p for p in pids}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        by_model: Dict[str, str] = {}
        by_shape: Dict[Tuple[str, frozenset, int], str] = {}
        for pid in pids:
            record = self.products.get(pid)
            model = _squash((record.facts or {}).get("model")) if record else ""
            if 5 <= len(model) <= 24 and any(c.isdigit() for c in model):
                if model in by_model:
                    union(by_model[model], pid)
                else:
                    by_model[model] = pid
            price = None
            for key, state in self.variants.items():
                if key.startswith(pid + "::") and state.price is not None:
                    price = state.price if price is None else min(price, state.price)
            if price is None:
                continue
            shape = (
                _squash(self.brand(pid)),
                frozenset(_title_tokens(self.title(pid))),
                int(round(price * 100)),
            )
            if not shape[1]:
                continue
            if shape in by_shape:
                union(by_shape[shape], pid)
            else:
                by_shape[shape] = pid

        groups: Dict[str, List[str]] = {}
        for pid in pids:
            groups.setdefault(find(pid), []).append(pid)
        return sorted(groups.values(), key=lambda g: g[0])


_PRIMARY_LABELS = {
    "screen_size": ("size", "screen_size", "display_size", "screen"),
    "refresh_rate": ("refresh_rate",),
    "ram": ("ram", "memory"),
    "resolution": (),
}

_DEVICE_NOUNS = {
    "smartphone", "phone", "cellphone", "mobile", "monitor", "tablet", "laptop", "notebook",
    "headphone", "earphone", "headset", "earbud", "camera", "webcam", "speaker", "television",
    "watch", "keyboard", "mouse", "printer", "console", "router",
}
_ACCESSORY_NOUNS = {
    "case", "cases", "cover", "covers", "protector", "protectors", "lanyard", "rope", "strap",
    "holder", "mount", "bracket", "tripod", "pouch", "sleeve", "skin", "sticker", "film",
    "splitter", "spare", "replacement",
}
_ACCESSORY_SHELVES = {
    "case", "cover", "accessory", "accessorie", "protector", "part", "sparepart", "mount",
    "holder", "strap", "charger", "cable",
}

_COLOUR_WORDS = (
    "black", "white", "silver", "grey", "gray", "gold", "rose", "pink", "red", "blue", "navy",
    "green", "sage", "mint", "yellow", "orange", "purple", "violet", "brown", "beige", "cream",
    "ivory", "champagne", "titanium", "graphite", "midnight", "starlight", "space", "matte",
    "glossy", "transparent", "clear", "multicolor", "multicolour", "bronze", "copper", "teal",
    "cyan", "magenta", "lavender", "khaki", "maroon", "burgundy", "coral", "peach", "turquoise",
)
_COLOUR_RE = re.compile(
    r"\b((?:(?:%s)\s+){0,2}(?:%s))\b" % ("|".join(_COLOUR_WORDS), "|".join(_COLOUR_WORDS)), re.I
)
_COLOUR_FACT_KEYS = ("color", "colour", "color_family", "colour_family", "color_name")


@dataclass
class Verdict:
    field: str
    status: str
    observed: Any = None
    source: str = ""
    detail: str = ""


@dataclass
class Candidate:
    pid: str
    sku: str
    price: Optional[float] = None
    in_stock: Optional[bool] = None
    verdicts: Dict[str, Verdict] = field(default_factory=dict)
    contradicted: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    relax: int = 3
    relax_detail: str = ""
    axes: List[str] = field(default_factory=list)
    inspected: bool = False
    live: bool = False
    dead: bool = False

    @property
    def key(self) -> str:
        return self.pid + "::" + self.sku

    @property
    def eligible(self) -> bool:
        return self.live and self.inspected and not self.contradicted and not self.unresolved

    def why_not(self) -> str:
        if self.dead:
            return "invalidated by a market change"
        if not self.live:
            return "out of stock, over budget, or price unknown"
        if not self.inspected:
            return "its product page has not been read"
        if self.contradicted:
            return "fails " + ", ".join(self.contradicted)
        if self.unresolved:
            return "no evidence resolves " + ", ".join(self.unresolved)
        return ""


class ConstraintChecker:

    def __init__(self, record: RequestRecord, store: EvidenceStore) -> None:
        self.record = record
        self.store = store

    def unit_of(self, concept: str) -> str:
        if concept in _VALUE_READERS:
            return ""
        requirement = self.record.get("spec:" + concept)
        return requirement.unit if requirement else ""

    def fact_labels(self, pid: str, concept: str) -> Tuple[str, ...]:
        known = _FACT_LABELS.get(concept)
        if known:
            return known
        keys = [k for k in (self.store.facts(pid) or {}) if _stems(k) & _stems(concept)]
        return tuple(keys) or tuple(self.store.facts(pid) or {})

    def _strip_model_codes(self, pid: str, title: str) -> str:
        model_fact = str(self.store.facts(pid).get("model") or "")
        for token in re.split(r"[;,/]", model_fact):
            token = token.strip()
            stem = re.split(r"[-_\s]", token)[0] if token else ""
            if (
                token
                and re.search(r"[a-z]", stem, re.I)
                and re.search(r"\d", stem)
                and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            ):
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", " ", title, flags=re.I)
        return title

    def product_spec(self, pid: str, concept: str) -> Tuple[Optional[str], str]:
        facts = self.store.facts(pid)
        unit = self.unit_of(concept)
        values: set = set()
        conflict = False
        for label in self.fact_labels(pid, concept):
            raw = facts.get(label)
            if raw in (None, ""):
                continue
            text = str(raw)
            if _TRUNCATED_RE.search(text):
                conflict = True
                continue
            found = _spec_vals(concept, text, unit)
            if not found and label in _PRIMARY_LABELS.get(concept, ()) and re.fullmatch(
                r"\d+(?:\.\d+)?", text.strip()
            ):
                bare = _BARE_UNITS.get(concept)
                if bare:
                    found = {"%g%s" % (float(text.strip()), bare)}
            if concept == "resolution" and _QUARTER_HD_RE.search(text):
                conflict = True
            if len(found) > 1:
                conflict = True
            values |= found
        title = self.store.title(pid)
        if concept == "resolution":
            title = self._strip_model_codes(pid, title)
            if _QUARTER_HD_RE.search(title):
                conflict = True
        values |= _spec_vals(concept, title, unit)
        if conflict or len(values) > 1:
            return None, UNRESOLVED
        if not values:
            return None, UNRESOLVED
        return next(iter(values)), SUPPORTED

    def variant_spec(self, pid: str, sku: str, concept: str) -> Tuple[Optional[str], str]:
        options = self.store.options(pid, sku)
        unit = self.unit_of(concept)
        found = _spec_vals(concept, options, unit)
        if len(found) > 1:
            return None, UNRESOLVED
        if found:
            return next(iter(found)), SUPPORTED
        if _mentions_spec(concept, options, unit):
            return None, UNRESOLVED
        return self.product_spec(pid, concept)

    def fact_spec(self, pid: str, concept: str) -> Optional[str]:
        facts = self.store.facts(pid)
        unit = self.unit_of(concept)
        for label in self.fact_labels(pid, concept):
            raw = facts.get(label)
            if raw in (None, "") or _TRUNCATED_RE.search(str(raw)):
                continue
            text = str(raw).strip()
            found = _spec_vals(concept, text, unit)
            if len(found) == 1:
                return next(iter(found))
            if not found and label in _PRIMARY_LABELS.get(concept, ()) and re.fullmatch(
                r"\d+(?:\.\d+)?", text
            ):
                unit = _BARE_UNITS.get(concept)
                if unit:
                    return "%g%s" % (float(text), unit)
        return None

    def category_status(self, pid: str) -> Verdict:
        want = self.record.category
        if not want:
            return Verdict("category", SUPPORTED)
        record = self.store.products.get(str(pid))
        if record is None or not record.inspected:
            return Verdict("category", UNRESOLVED, detail="product page not read")
        path = record.category_path
        if not path:
            return Verdict("category", UNRESOLVED, detail="no category_path")
        wanted = _stems(want)
        for alias in self.record.category_aliases:
            wanted |= _stems(alias)
        labels: set = set()
        for label in path:
            labels |= _stems(label)
            labels |= {_squash(label)}
        placed = bool(wanted & labels)
        if not placed and {"smartphone", "phone", "cellphone", "mobile"} & wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(label).strip(), re.I)
                for label in path
            )
        if not placed and {"headphone", "earphone", "headset", "earbud"} & wanted:
            placed = any(
                re.search(r"headphone|headset|earphone|earbud", str(label), re.I) for label in path
            )
        if not placed:
            return Verdict("category", CONTRADICTED, observed=" > ".join(path),
                           source="category_path")
        qualifier = self.record.category_qualifier
        if qualifier:
            title_words = _stems(self.store.title(pid)) | _tokens(self.store.title(pid))
            if _squash(qualifier) not in {_squash(w) for w in title_words}:
                if qualifier == "gaming":
                    fast = any(_measure(v) or 0 >= 120 for v in _hz_vals(self.store.title(pid)))
                    if not fast:
                        return Verdict("category", UNRESOLVED,
                                       detail="title does not show it is a %s model" % qualifier)
                else:
                    return Verdict("category", UNRESOLVED,
                                   detail="title does not show it is %s" % qualifier)
        return Verdict("category", SUPPORTED, observed=" > ".join(path), source="category_path")

    def brand_status(self, pid: str, wanted: str) -> Verdict:
        if not wanted:
            return Verdict("brand", SUPPORTED)
        catalog = _tokens(self.store.brand(pid))
        title = _title_tokens(self.store.title(pid))
        parts = [w for w in _tokens(wanted)]
        if not parts:
            return Verdict("brand", SUPPORTED)
        head = parts[0]
        rest = set(parts[1:])
        if not catalog:
            return Verdict("brand", UNRESOLVED, detail="listing states no brand")
        if head not in catalog:
            return Verdict("brand", CONTRADICTED, observed=self.store.brand(pid), source="brand")
        if rest and not rest <= (catalog | title):
            return Verdict("brand", UNRESOLVED, observed=self.store.brand(pid),
                           detail="product line not shown")
        return Verdict("brand", SUPPORTED, observed=self.store.brand(pid), source="brand")

    def device_status(self, pid: str, sku: str) -> Verdict:
        category = self.record.category or ""
        if not ({"smartphone", "phone", "cellphone"} & _stems(category)):
            return Verdict("device", SUPPORTED)
        status = phone_identity(self.store.facts(pid), self.store.title(pid),
                                self.store.options(pid, sku))
        if status == SUPPORTED:
            return Verdict("device", SUPPORTED, source="device facts")
        if status == CONTRADICTED:
            return Verdict("device", CONTRADICTED, detail="evidence says it is not a smartphone")
        if self.record.device_identity_rule:
            return Verdict("device", UNRESOLVED, detail="no positive smartphone evidence")
        return Verdict("device", SUPPORTED, detail="nothing contradicts the device type")

    def accessory_status(self, pid: str) -> Verdict:
        category = self.record.category or ""
        if not (_stems(category) & _DEVICE_NOUNS) and _squash(category) not in _DEVICE_NOUNS:
            return Verdict("device_kind", SUPPORTED)
        record = self.store.products.get(str(pid))
        if record is not None and record.category_path:
            shelf = set()
            for label in record.category_path:
                shelf |= {w.rstrip("s") for w in _tokens(label)}
            if shelf & _ACCESSORY_SHELVES and not (_stems(category) & shelf):
                return Verdict("device_kind", CONTRADICTED,
                               observed=" > ".join(record.category_path), source="category_path")
        title = self.store.title(pid)
        if _REVIEW_TITLE_RE.search(title):
            return Verdict("device_kind", CONTRADICTED, observed=_clip(title, 80), source="title")
        named = self._accessory_head(title)
        if named:
            return Verdict("device_kind", CONTRADICTED, observed=named, source="title")
        return Verdict("device_kind", SUPPORTED)

    @staticmethod
    def _accessory_head(title: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(title or "").lower())
        for token in tokens[:_HEAD_TOKENS]:
            if token in _ACCESSORY_NOUNS:
                return token
        for m in re.finditer(r"\b(?:for|compatible with|applicable to|fits)\b", title or "", re.I):
            window = title[m.end(): m.end() + _COMPAT_WINDOW]
            hit = _title_tokens(window) & _ACCESSORY_NOUNS
            if hit:
                return sorted(hit)[0]
        return ""

    def colour_of(self, pid: str, sku: str) -> Tuple[Optional[str], str]:
        sources = [
            ("options", self.store.options(pid, sku)),
            ("facts", " ; ".join(
                str(self.store.facts(pid).get(k)) for k in _COLOUR_FACT_KEYS
                if self.store.facts(pid).get(k)
            )),
            ("title", self.store.title(pid)),
        ]
        for name, text in sources:
            if not text:
                continue
            shades = [m.group(1).strip().lower() for m in _COLOUR_RE.finditer(text)]
            shades = [s for s in shades if s]
            if not shades:
                continue
            distinct = {_squash(s) for s in shades}
            if len(distinct) > 1:
                heads = {s.split()[-1] for s in shades}
                if len(heads) > 1:
                    return None, UNRESOLVED
            longest = max(shades, key=lambda s: (len(s.split()), len(s)))
            return longest, SUPPORTED
        return None, UNRESOLVED

    def colour_matches(self, wanted: str, observed: Optional[str]) -> bool:
        if not observed:
            return False
        want_tokens = [w for w in _tokens(wanted)]
        have_tokens = [w for w in _tokens(observed)]
        if not want_tokens or not have_tokens:
            return False
        if set(want_tokens) <= set(have_tokens):
            return True
        return want_tokens[-1] == have_tokens[-1]

    def model_match(self, pid: str, wanted: str) -> int:
        target = _squash(wanted)
        if not target:
            return 0
        fact = str(self.store.facts(pid).get("model") or "")
        if fact:
            for piece in re.split(r"[;,/]", fact):
                squashed = _squash(piece)
                if not squashed:
                    continue
                if squashed == target or squashed.endswith(target):
                    if squashed == target or not squashed[: -len(target)][-1:].isdigit():
                        return 2
            return 0
        for token in re.findall(r"[a-z0-9]+", self.store.title(pid).lower()):
            if token == target or (token.endswith(target) and not token[: -len(target)][-1:].isdigit()):
                return 1
        return 0

    def relaxation(self, pid: str, sku: str) -> Tuple[int, str]:
        model = self.record.preferred_model
        colour = self.record.preferred_colour
        if not model:
            return 0, ""
        matched = self.model_match(pid, model)
        observed, _status = self.colour_of(pid, sku)
        same_colour = bool(colour) and self.colour_matches(colour, observed)
        if matched and (same_colour or not colour):
            return 0, "exact %s%s" % (colour + " " if colour else "", model)
        if matched:
            return 1, "%s in %s" % (model, observed or "another colour")
        if same_colour:
            return 2, "another model in %s" % colour
        return 3, "another model"

    def assess(self, pid: str, sku: str) -> Candidate:
        pid, sku = str(pid), str(sku)
        state = self.store.state(pid, sku)
        price = state.price if state else None
        in_stock = state.in_stock if state else None
        candidate = Candidate(pid=pid, sku=sku, price=price, in_stock=in_stock)
        candidate.inspected = self.store.inspected(pid)
        key = self.store.key(pid, sku)
        candidate.dead = key in self.store.dead_keys
        budget = self.record.budget

        def note(verdict: Verdict, label: str) -> None:
            candidate.verdicts[verdict.field] = verdict
            if verdict.status == CONTRADICTED:
                candidate.contradicted.append(label)
            elif verdict.status == UNRESOLVED:
                candidate.unresolved.append(label)

        if in_stock is False:
            note(Verdict("in_stock", CONTRADICTED, observed=False), "stock")
        elif in_stock is None:
            note(Verdict("in_stock", UNRESOLVED), "stock")
        else:
            candidate.verdicts["in_stock"] = Verdict("in_stock", SUPPORTED, observed=True)

        if price is None:
            note(Verdict("budget", UNRESOLVED), "price")
        elif budget is not None and price > budget + _EPS:
            note(Verdict("budget", CONTRADICTED, observed=price), "budget")
        else:
            candidate.verdicts["budget"] = Verdict("budget", SUPPORTED, observed=price)

        candidate.live = bool(
            in_stock and price is not None
            and (budget is None or price <= budget + _EPS)
            and not candidate.dead
        )

        note(self.category_status(pid), "category")
        brand = self.record.brand
        if brand:
            note(self.brand_status(pid, brand), "brand")
        note(self.device_status(pid, sku), "device type")
        if self.record.reject_accessories:
            note(self.accessory_status(pid), "device kind")

        for concept, wanted in self.record.specs().items():
            value, status = self.variant_spec(pid, sku, concept)
            label = concept.replace("_", " ")
            if status == SUPPORTED and value == wanted:
                candidate.verdicts["spec:" + concept] = Verdict(
                    "spec:" + concept, SUPPORTED, observed=value, source="variant evidence"
                )
            elif status == SUPPORTED:
                note(Verdict("spec:" + concept, CONTRADICTED, observed=value), label)
            else:
                note(Verdict("spec:" + concept, UNRESOLVED), label)

        if self.record.candidate_keys and (pid, sku) not in self.record.candidate_keys:
            note(Verdict("candidate_set", CONTRADICTED, observed=candidate.key), "candidate list")

        if self.record.title_terms:
            tokens = _title_tokens(self.store.title(pid))
            missing = [t for t in self.record.title_terms if t not in tokens]
            if missing:
                note(Verdict("title_terms", CONTRADICTED, observed=", ".join(missing)),
                     "title phrase")
            else:
                candidate.verdicts["title_terms"] = Verdict("title_terms", SUPPORTED)

        candidate.relax, candidate.relax_detail = self.relaxation(pid, sku)
        preferred = self.record.preferred_brand
        if preferred:
            verdict = self.brand_status(pid, preferred)
            candidate.verdicts["preferred_brand"] = verdict
        candidate.axes = self.axes_covered(pid, sku)
        return candidate

    def axes_covered(self, pid: str, sku: str) -> List[str]:
        if not self.record.axes:
            return []
        record = self.store.products.get(str(pid))
        blob = " ".join(
            [
                self.store.title(pid),
                record.fact_text() if record else "",
                self.store.options(pid, sku),
            ]
        )
        covered = []
        for axis in self.record.axes:
            if axis.pattern is None:
                continue
            match = axis.pattern.search(blob)
            if not match:
                continue
            before = blob[max(0, match.start() - 18):match.start()]
            if _NEGATION_RE.search(before):
                continue
            if axis.kind == "duration":
                window = blob[max(0, match.start() - 60):match.start() + 80]
                if not _DURATION_RE.search(window):
                    continue
            covered.append(axis.key)
        return covered


_QUERY_NOISE = frozenset({
    "for", "with", "and", "the", "new", "hot", "sale", "free", "original", "high",
    "quality", "best", "good", "super", "cod", "ready", "stock", "shop", "store",
    "pcs", "pack", "set", "piece", "items", "item", "product", "brand", "official",
})


def _surface_forms(term: str) -> List[str]:
    raw = str(term or "").strip()
    if not raw:
        return []
    forms = [raw]

    def add(form: str) -> None:
        form = form.strip()
        if form and form.lower() not in {f.lower() for f in forms}:
            forms.append(form)

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:-\s*)?(inch|in|\")", raw, re.I)
    if m:
        add("%s inch" % m.group(1))
        add("%sinch" % m.group(1))
        add('%s"' % m.group(1))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*hz", raw, re.I)
    if m:
        add("%shz" % m.group(1))
        add("%s hz" % m.group(1))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*gb", raw, re.I)
    if m:
        add("%sgb" % m.group(1))
        add("%sgb ram" % m.group(1))
    low = raw.lower()
    for key, names in _CLASS_NAMES.items():
        if low in names:
            for name in names:
                add(name)
            break
    fused = re.sub(r"(\w+)(?:\s+|-)res(?:is)[a-z]+", r"\1proof", raw, flags=re.I)
    if fused != raw:
        add(fused)
    compact = re.fullmatch(r"([a-z]{1,4})\s+(\d+(?:\.\d+)?)", raw, re.I)
    if compact:
        add("%s%s" % (compact.group(1), compact.group(2)))
    return forms


class Retriever:

    MAX_SEARCH_QUERIES = 3
    MAX_VARIANT_QUERIES = 6

    def __init__(self, record: RequestRecord, store: EvidenceStore, catalog: ToolCatalog) -> None:
        self.record = record
        self.store = store
        self.catalog = catalog
        self.made: set = set()
        self.searches = 0
        self.filters = 0
        self.twin_searched = False
        self.sibling_searched = False
        self.model_searched = False
        self.axis_recovered = False

    def keywords(self) -> List[str]:
        out: List[str] = []
        if self.record.title_terms:
            out.append(" ".join(self.record.title_terms))
        model = self.record.preferred_model
        if model:
            out.append(model)
        brand = self.record.brand or self.record.preferred_brand
        if brand:
            out.append(brand)
        category = self.record.category
        if category:
            out.append(category)
        for concept, value in self.record.specs().items():
            spoken = self._spoken(concept, value)
            if spoken:
                out.append(spoken)
        seen, uniq = set(), []
        for term in out:
            low = term.lower().strip()
            if low and low not in seen:
                seen.add(low)
                uniq.append(term.strip())
        return uniq

    @staticmethod
    def _spoken(concept: str, value: str) -> str:
        if concept == "resolution":
            return _CLASS_NAMES.get(value, (value,))[0]
        number = _measure(value)
        if number is None:
            return value
        unit = {"screen_size": "inch", "refresh_rate": "hz", "ram": "gb"}.get(concept, "")
        return ("%g %s" % (number, unit)).strip()

    def _variant_queries(self, terms: Sequence[str]) -> List[str]:
        forms = [_surface_forms(t) for t in terms]
        base = " ".join(f[0] for f in forms if f)
        queries = [base] if base else []
        for index, options in enumerate(forms):
            for alternative in options[1:]:
                words = [f[0] for f in forms if f]
                if index < len(words):
                    words[index] = alternative
                candidate = " ".join(words).strip()
                if candidate and candidate not in queries:
                    queries.append(candidate)
        return queries[: self.MAX_VARIANT_QUERIES]

    def _fresh(self, name: str, args: dict) -> bool:
        return (name + json.dumps(args, sort_keys=True)) not in self.made

    def _mark(self, name: str, args: dict) -> None:
        self.made.add(name + json.dumps(args, sort_keys=True))

    def _search(self, query: str, k: int = 10, in_stock: bool = True) -> Optional[Tuple[str, dict]]:
        if not self.catalog.has("search") or not query.strip():
            return None
        if self.record.required_identities and len(self.store.queries) >= self.MAX_SEARCH_QUERIES:
            return None
        normalized = re.sub(r"\s+", " ", query).strip().lower()
        if normalized in self.store.queries and len(self.store.queries) >= 1:
            return None
        args: Dict[str, Any] = {"query": query.strip(), "k": k}
        if in_stock and "in_stock" in self.catalog.properties("search"):
            args["in_stock"] = True
        if not self._fresh("search", args):
            return None
        return "search", args

    def _filter(self, **overrides: Any) -> Optional[Tuple[str, dict]]:
        if not self.catalog.has("filter"):
            return None
        props = self.catalog.properties("filter")
        args: Dict[str, Any] = {}
        if "k" in props:
            args["k"] = 10
        if "in_stock" in props:
            args["in_stock"] = True
        if "category" in props and self.record.category:
            args["category"] = self.record.category
        if "max_price" in props and self.record.budget is not None:
            args["max_price"] = self.record.budget
        if "brand" in props and self.record.brand:
            args["brand"] = self.record.brand
        args.update({k: v for k, v in overrides.items() if v is not None})
        for key in [k for k in args if k not in props]:
            args.pop(key)
        if not args or not self._fresh("filter", args):
            return None
        return "filter", args

    def budget_for_searches(self, desperate: bool = False) -> int:
        if self.record.required_identities:
            return self.MAX_SEARCH_QUERIES
        return 9 if desperate else self.MAX_SEARCH_QUERIES + 1

    def can_widen(self) -> bool:
        return self.next_action(need_more=True) is not None

    def axis_queries(self) -> List[str]:
        if not self.record.axes:
            return []
        category = self.record.category or ""
        wording = {
            "water_resistance": ("waterproof", "ipx"),
            "secure_fit": ("sports", "earhook"),
            "battery_life": ("hours battery", "playtime"),
            "wireless": ("wireless bluetooth", "tws"),
            "noise_cancelling": ("noise cancelling", "anc"),
            "comfort": ("comfortable", "lightweight"),
            "sound_quality": ("hifi stereo", "bass"),
            "microphone": ("microphone", "mic"),
            "fast_charge": ("fast charging", "type-c"),
        }
        ordered = sorted(self.record.axes, key=lambda a: a.priority)
        first = [wording.get(a.key, (a.key.replace("_", " "),))[0] for a in ordered]
        second = [wording.get(a.key, (a.key.replace("_", " "),))[-1] for a in ordered]
        queries = [
            " ".join([category] + first).strip(),
            " ".join([category] + second).strip(),
        ]
        top = first[:2]
        for shelf in self.shelf_names():
            queries.append(" ".join([shelf] + top).strip())
        return queries

    def shelf_names(self, limit: int = 2) -> List[str]:
        counts: Dict[str, int] = {}
        for record in self.store.products.values():
            if not record.inspected or not record.category_path:
                continue
            leaf = str(record.category_path[-1]).strip()
            if leaf and not _same_word(leaf, self.record.category or ""):
                counts[leaf] = counts.get(leaf, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [name for name, _n in ranked[:limit]]

    def axis_recovery(self, dead_pid: str) -> Optional[Tuple[str, dict]]:
        if not self.record.axes or self.axis_recovered:
            return None
        blob = " ".join([
            self.store.title(dead_pid),
            self.store.product(dead_pid).fact_text(),
        ])
        words: List[str] = []
        for axis in sorted(self.record.axes, key=lambda a: a.priority):
            if axis.pattern is None:
                continue
            found = axis.pattern.findall(blob)
            if found:
                word = str(found[0]).strip()
                if word and word.lower() not in [w.lower() for w in words]:
                    words.append(word)
        if not words:
            return None
        self.axis_recovered = True
        return self._search(" ".join([self.record.category or ""] + words).strip())

    def fallback_queries(self) -> List[str]:
        brand = self.record.brand or self.record.preferred_brand or ""
        category = self.record.category or ""
        spoken = [self._spoken(c, v) for c, v in self.record.specs().items()]
        out = [
            " ".join(x for x in (brand, category) if x),
            " ".join(x for x in (category,) + tuple(spoken) if x),
            brand,
            category,
            " ".join(x for x in (brand,) + tuple(spoken) if x),
        ]
        seen, uniq = set(), []
        for query in out:
            low = query.strip().lower()
            if low and low not in seen:
                seen.add(low)
                uniq.append(query.strip())
        return uniq

    def splitting_tokens(self, limit: int = 2) -> List[str]:
        terms = set(self.record.title_terms)
        if not terms:
            return []
        carriers: Dict[str, set] = {}
        titles = 0
        for pid in self.store.matching_products(self.record.title_terms):
            tokens = _title_tokens(self.store.title(pid))
            if not tokens:
                continue
            titles += 1
            for token in tokens:
                if token in terms or len(token) < 3 or token.isdigit():
                    continue
                if token in _QUERY_NOISE:
                    continue
                carriers.setdefault(token, set()).add(pid)
        if titles < 2:
            return []
        usable = [
            (token, holders) for token, holders in carriers.items()
            if 2 <= len(holders) < titles
        ]
        usable.sort(key=lambda item: (-len(item[1]), item[0]))
        chosen: List[str] = []
        taken: set = set()
        for token, holders in usable:
            if chosen and len(holders & taken) == len(holders):
                continue
            chosen.append(token)
            taken |= holders
            if len(chosen) >= limit:
                break
        return chosen

    def survey_queries(self) -> List[str]:
        terms = self.record.title_terms
        if not terms:
            return []
        phrase = " ".join(terms)
        queries = [phrase]
        for token in self.splitting_tokens():
            queries.append("%s %s" % (phrase, token))
        category = self.record.category
        if category:
            queries.append("%s %s" % (phrase, category))
        seen, uniq = set(), []
        for query in queries:
            low = query.strip().lower()
            if low and low not in seen:
                seen.add(low)
                uniq.append(query.strip())
        return uniq

    def next_action(self, *, need_more: bool) -> Optional[Tuple[str, dict]]:
        if self.record.candidate_keys:
            return None

        allowed = self.budget_for_searches(need_more)
        if self.record.title_terms:
            for query in self.survey_queries():
                if len(self.store.queries) >= allowed:
                    break
                action = self._search(query)
                if action:
                    return action
            action = self._filter()
            if action:
                return action
            return None

        action = self._filter()
        if action:
            return action
        for query in self.axis_queries():
            if len(self.store.queries) >= allowed:
                break
            action = self._search(query)
            if action:
                return action
        for alias in self.record.category_aliases:
            action = self._filter(category=alias)
            if action:
                return action
        preferred = self.record.preferred_brand
        if preferred and "brand" in self.catalog.properties("filter"):
            action = self._filter(brand=preferred)
            if action:
                return action
        for query in self._variant_queries(self.keywords()):
            if len(self.store.queries) >= allowed:
                break
            action = self._search(query)
            if action:
                return action
        if need_more:
            action = self._filter(brand=None)
            if action:
                return action
            for query in self.fallback_queries():
                if len(self.store.queries) >= allowed:
                    break
                action = self._search(query)
                if action:
                    return action
            action = self._filter(in_stock=None)
            if action:
                return action
        return None

    def model_search(self) -> Optional[Tuple[str, dict]]:
        model = self.record.preferred_model
        if not model or self.model_searched:
            return None
        self.model_searched = True
        colour = self.record.preferred_colour
        query = ("%s %s" % (colour, model)).strip() if colour else model
        return self._search(query) or self._search(model)

    def brand_hunt(self) -> Optional[Tuple[str, dict]]:
        brand = self.record.preferred_brand
        if not brand:
            return None
        if "brand" in self.catalog.properties("filter"):
            action = self._filter(brand=brand)
            if action:
                return action
        category = self.record.category or ""
        spoken = [self._spoken(c, v) for c, v in self.record.specs().items()]
        for query in (
            " ".join([brand, category] + spoken),
            " ".join([brand] + spoken),
            " ".join([brand, category]),
            brand,
        ):
            action = self._search(query.strip())
            if action:
                return action
        return None

    def twin_search(self, dead_pid: str) -> Optional[Tuple[str, dict]]:
        if self.twin_searched:
            return None
        title = self.store.title(dead_pid)
        if not title:
            return None
        self.twin_searched = True
        return self._search(" ".join(title.split()[:8]))

    def sibling_search(self, near_pid: str) -> Optional[Tuple[str, dict]]:
        if self.sibling_searched:
            return None
        title = self.store.title(near_pid)
        if not title:
            return None
        self.sibling_searched = True
        return self._search(" ".join(title.split()[:8]))


class Selector:

    def __init__(self, record: RequestRecord, store: EvidenceStore, checker: ConstraintChecker) -> None:
        self.record = record
        self.store = store
        self.checker = checker

    def objective_value(self, candidate: Candidate, field_name: str) -> Optional[float]:
        if field_name == "price":
            return candidate.price
        value, status = self.checker.variant_spec(candidate.pid, candidate.sku, field_name)
        if status == SUPPORTED and value is not None:
            if field_name == "resolution":
                rank = _class_rank(value)
                return float(rank) if rank is not None else None
            return _measure(value)
        if field_name in _VALUE_READERS:
            return None
        return self._stated_quantity(candidate, field_name)

    def _stated_quantity(self, candidate: Candidate, field_name: str) -> Optional[float]:
        concept = field_name.replace("_", " ").strip()
        words = [w for w in re.findall(r"[a-z]+", concept.lower()) if len(w) > 2]
        if not words:
            return None
        record = self.store.products.get(candidate.pid)
        blob = " ".join([
            self.store.title(candidate.pid),
            record.fact_text() if record else "",
            self.store.options(candidate.pid, candidate.sku),
        ])
        found: set = set()
        for word in words:
            before = r"(\d+(?:\.\d+)?)\s*[a-z%]{0,6}\s*" + re.escape(word)
            after = re.escape(word) + r"[^\d\n]{0,14}(\d+(?:\.\d+)?)"
            for m in re.finditer(before + "|" + after, blob, re.I):
                number = m.group(1) or m.group(2)
                if number:
                    found.add(float(number))
        return next(iter(found)) if len(found) == 1 else None

    def comparable(self, candidate: Candidate) -> bool:
        for objective in self.record.objectives:
            if self.objective_value(candidate, objective.field) is None:
                return False
        return True

    def _objective_key(self, candidate: Candidate) -> Tuple:
        key: List[float] = []
        for objective in self.record.objectives:
            value = self.objective_value(candidate, objective.field)
            value = 0.0 if value is None else float(value)
            key.append(-value if objective.direction == "max" else value)
        return tuple(key)

    def dominated(self, candidate: Candidate, others: Sequence[Candidate]) -> bool:
        if not self.record.objectives:
            return False
        mine = [self.objective_value(candidate, o.field) for o in self.record.objectives]
        if any(v is None for v in mine):
            return False
        for other in others:
            if other.key == candidate.key:
                continue
            theirs = [self.objective_value(other, o.field) for o in self.record.objectives]
            if any(v is None for v in theirs):
                continue
            better = False
            for objective, a, b in zip(self.record.objectives, mine, theirs):
                sign = 1.0 if objective.direction == "max" else -1.0
                if sign * (b - a) < -_EPS:
                    break
                if sign * (b - a) > _EPS:
                    better = True
            else:
                if better:
                    return True
        return False

    def axis_score(self, candidate: Candidate) -> Tuple[int, float]:
        if not self.record.axes:
            return 0, 0.0
        covered = set(candidate.axes)
        total = len(self.record.axes)
        weight = 0.0
        for axis in self.record.axes:
            if axis.key in covered:
                weight += float(total - min(axis.priority, total - 1))
        return len(covered), weight

    def rank_key(self, candidate: Candidate, first_commitment: bool) -> Tuple:
        price = candidate.price if candidate.price is not None else 1e12
        if self.record.objectives:
            return self._objective_key(candidate) + (price, candidate.key)
        if self.record.preferred_model:
            return (candidate.relax, price, candidate.key)
        if self.record.axes:
            count, weight = self.axis_score(candidate)
            return (-count, -weight, price, candidate.key)
        if self.record.preferred_brand and first_commitment:
            verdict = candidate.verdicts.get("preferred_brand")
            near = 0 if (verdict and verdict.status == SUPPORTED) else 1
            return (near, price, candidate.key)
        proven = 0 if candidate.verdicts.get("device", Verdict("device", SUPPORTED)).status == SUPPORTED else 1
        return (proven, price, candidate.key)

    def order(self, candidates: Sequence[Candidate], first_commitment: bool) -> List[Candidate]:
        pool = list(candidates)
        if self.record.objectives:
            pool = [c for c in pool if self.comparable(c)] or pool
        if self.record.preferred_brand and first_commitment:
            named = [
                c for c in pool
                if (c.verdicts.get("preferred_brand") or Verdict("preferred_brand", UNRESOLVED)).status == SUPPORTED
            ]
            if named:
                pool = named
        if self.record.preferred_model and first_commitment:
            exact = [c for c in pool if c.relax == 0]
            if exact:
                exact.sort(key=lambda c: (0 if self.checker.model_match(c.pid, self.record.preferred_model) == 2 else 1,
                                          c.price if c.price is not None else 1e12))
                pool = exact
        pool.sort(key=lambda c: self.rank_key(c, first_commitment))
        return pool


_REASON_FIELDS = (
    ("brand", lambda blob, brand: brand or None),
    ("streaming", lambda blob, brand: "true" if re.search(r"\bstream\w*\b", blob, re.I) else None),
    ("microphone", lambda blob, brand: "true" if re.search(r"\bmicrophone\b|\bmic\b", blob, re.I) else None),
    ("privacy_cover", lambda blob, brand: "true" if re.search(r"privacy\s*(?:cover|shutter)", blob, re.I) else None),
    ("autofocus", lambda blob, brand: "true" if re.search(r"auto[\s\-]?focus", blob, re.I) else None),
    ("hdr", lambda blob, brand: "true" if re.search(r"\bhdr\b", blob, re.I) else None),
)


class ClaimBuilder:

    def __init__(self, record: RequestRecord, store: EvidenceStore,
                 checker: ConstraintChecker, catalog: ToolCatalog) -> None:
        self.record = record
        self.store = store
        self.checker = checker
        self.catalog = catalog

    def allowed_fields(self) -> Optional[List[str]]:
        values = self.catalog.enum_for("place_test_order", "justification", "claims", "field")
        return values or None

    def required(self) -> bool:
        return self.catalog.requires("place_test_order", "justification")

    def wanted(self) -> bool:
        return self.required() or self.record.reason_requested

    def build(self, pid: str, sku: str) -> List[dict]:
        allowed = self.allowed_fields()
        state = self.store.state(pid, sku)
        if state is None or not state.in_stock:
            return []
        claims: List[dict] = []

        def add(field_name: str, value: Any) -> None:
            if value in (None, ""):
                return
            if allowed is not None and field_name not in allowed:
                return
            if any(c["field"] == field_name for c in claims):
                return
            claims.append({"field": field_name, "value": str(value)})

        for concept, wanted in self.record.specs().items():
            fact_value = self.checker.fact_spec(pid, concept)
            variant_value, status = self.checker.variant_spec(pid, sku, concept)
            if fact_value == wanted and status == SUPPORTED and variant_value == wanted:
                add(concept, wanted)
        if not claims:
            return []

        budget = self.record.budget
        if budget is not None and state.price is not None and state.price <= budget + _EPS:
            add("price", "at most %.2f" % budget)
        elif state.price is not None:
            add("price", "%.2f" % state.price)
        add("in_stock", "true")

        blob = " ".join(
            [self.store.title(pid), self.store.product(pid).fact_text(), self.store.options(pid, sku)]
        )
        brand = self.store.brand(pid)
        for name, reader in _REASON_FIELDS:
            if allowed is not None and name not in allowed:
                continue
            value = reader(blob, brand)
            if value:
                add(name, value)
                break
        return claims


_EXTRACT_SYSTEM = (
    "You convert a shopper's request into structured requirements. Answer with JSON only."
)

_EXTRACT_PROMPT = """Read the shopper's request and the task rules, then report what the request asks for.

# Request
{goal}

# Task rules
{rules}

# Response format
{{
  "category": <str|null>,          // the catalog shelf the item must sit on, as the shopper names it
  "brand": <str|null>,             // a brand that is REQUIRED, null when only preferred or unstated
  "preferred_brand": <str|null>,   // a brand named as a preference the firm requirements outrank
  "budget": <number|null>,         // the most the shopper will spend, as a number
  "specs": {{"screen_size"|"resolution"|"refresh_rate"|"ram": <str>}},  // values exactly as written
  "preferences": [<str>, ...],     // soft wishes, most important first, in the shopper's own words
  "objectives": [{{"direction": "max"|"min", "field": <str>}}, ...],    // a stated ranking order
  "clarify": <str|null>            // one question worth asking before choosing, else null
}}

Report only what the text states. Use null rather than guessing, and never invent a budget,
a brand or a specification the shopper did not write."""

_PREFER_PROMPT = """A shopper is buying one item. Choose which candidate best fits their stated priorities.

# What the shopper said
{goal}

# Their priorities, most important first
{axes}

# Candidates
{candidates}

Judge only from each candidate's own text. A feature nobody stated is not evidence.

# Response format
{{"best": <candidate number>, "why": "<one short clause naming the deciding evidence>"}}"""


def variant_rows(store: EvidenceStore, pid: str, limit: int = 8) -> str:
    record = store.products.get(str(pid))
    if record is None:
        return ""
    lines = []
    for index, sku in enumerate(record.variant_order[:limit], start=1):
        state = store.state(pid, sku)
        if state is None:
            continue
        lines.append(
            "[variant %d] sku: %s | options: %s | price: %s | in_stock: %s"
            % (index, sku, state.options or "-",
               "-" if state.price is None else "%.2f" % state.price, state.in_stock)
        )
    return "\n".join(lines)


def candidate_block(store: EvidenceStore, candidate: Candidate, index: int) -> str:
    record = store.products.get(candidate.pid)
    facts = record.fact_text() if record else ""
    lines = [
        "candidate %d  (%s)" % (index, candidate.key),
        "  title: " + _clip(store.title(candidate.pid), 180),
        "  brand: " + (store.brand(candidate.pid) or "-"),
        "  facts: " + (_clip(facts, 300) or "-"),
        "  selected options: " + (store.options(candidate.pid, candidate.sku) or "-"),
        "  price: %s   in_stock: %s" % (
            "-" if candidate.price is None else "%.2f" % candidate.price, candidate.in_stock),
    ]
    rows = variant_rows(store, candidate.pid)
    if rows:
        lines.append("  variants (each line is one purchasable unit):")
        lines.extend("    " + row for row in rows.split("\n"))
    return "\n".join(lines)


class EpisodeController:

    def __init__(self, problem: dict) -> None:
        environment = problem["environment"]
        self.policy = environment["policy_view"]
        self.catalog = ToolCatalog(self.policy.get("tools") or [])
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")

        self.budget = Budget(_EPISODE_SECONDS, self.max_steps)
        self.record = RequestRecord(str(self.policy.get("query") or ""))
        self.store = EvidenceStore()
        self.checker = ConstraintChecker(self.record, self.store)
        self.selector = Selector(self.record, self.store, self.checker)
        self.retriever = Retriever(self.record, self.store, self.catalog)
        self.claims = ClaimBuilder(self.record, self.store, self.checker, self.catalog)
        self.inference = Inference(self.budget)
        self.client = EnvironmentClient(environment["binding"], self.problem_id, self.budget)

        self.dialogue: List[dict] = []
        self.notes: List[str] = []
        self.done = False
        self.asked = 0
        self.question_pending = False
        self.question_text = ""
        self.pending_recheck: Optional[Tuple[str, str]] = None
        self.reads_exhausted = False
        self.hunt_over = False
        self.judged: set = set()

    def _note(self, text: str) -> None:
        if text and (not self.notes or self.notes[-1] != text):
            self.notes.append(_clip(text, 240))

    def candidates(self) -> List[Candidate]:
        out = []
        for key in list(self.store.variants):
            pid, _, sku = key.partition("::")
            if not sku:
                continue
            out.append(self.checker.assess(pid, sku))
        return out

    def eligible(self, pool: Optional[Sequence[Candidate]] = None) -> List[Candidate]:
        return [c for c in (pool if pool is not None else self.candidates()) if c.eligible]

    def best(self, pool: Optional[Sequence[Candidate]] = None) -> Optional[Candidate]:
        good = self.eligible(pool)
        if not good:
            return None
        first = self.store.adds == 0
        carted = [c for c in good if (c.pid, c.sku) in self.store.cart]
        if carted:
            carted.sort(key=lambda c: self.selector.rank_key(c, False))
            return carted[0]
        if self.record.objectives:
            good = [c for c in good if not self.selector.dominated(c, good)] or good
        ordered = self.selector.order(good, first)
        if not ordered:
            return None
        if first and self.record.axes and len(ordered) > 1:
            ordered = self._preference_tiebreak(ordered)
        return ordered[0]

    def _preference_tiebreak(self, ordered: List[Candidate]) -> List[Candidate]:
        top = self.selector.axis_score(ordered[0])
        tied = [c for c in ordered if self.selector.axis_score(c) == top][:5]
        if len(tied) < 2 or not self.budget.exploring(0.8):
            return ordered
        signature = "|".join(c.key for c in tied)
        if signature in self.judged:
            return ordered
        self.judged.add(signature)
        prompt = _PREFER_PROMPT.format(
            goal=_clip(self.record.goal, 800),
            axes="\n".join(
                "%d. %s" % (i + 1, a.label)
                for i, a in enumerate(sorted(self.record.axes, key=lambda a: a.priority))
            ),
            candidates="\n\n".join(
                candidate_block(self.store, c, i + 1) for i, c in enumerate(tied)
            ),
        )
        answer = self.inference.ask(prompt, tries=1, reserve_after=_TAIL_SECONDS + 30.0)
        if not isinstance(answer, dict):
            return ordered
        choice = _num(answer.get("best"))
        if choice is None or not (1 <= int(choice) <= len(tied)):
            return ordered
        picked = tied[int(choice) - 1]
        self._note("preference tiebreak chose %s: %s" % (picked.key, _clip(answer.get("why"), 90)))
        return [picked] + [c for c in ordered if c.key != picked.key]

    def leads(self, limit: int = 10) -> List[str]:
        budget = self.record.budget
        wanted = _tokens(" ".join(self.retriever.keywords()))
        scored: List[Tuple[float, float, str]] = []
        for pid, record in self.store.products.items():
            if record.inspected or pid in self.store.dead_products:
                continue
            prices = [
                s.price for k, s in self.store.variants.items()
                if k.startswith(pid + "::") and s.price is not None
            ]
            stocked = any(
                s.in_stock for k, s in self.store.variants.items() if k.startswith(pid + "::")
            )
            if not prices or not stocked:
                continue
            price = min(prices)
            if budget is not None and price > budget + _EPS:
                continue
            title = self.store.title(pid)
            tokens = _title_tokens(title)
            if self.record.title_terms and not all(t in tokens for t in self.record.title_terms):
                continue
            if self.record.brand:
                verdict = self.checker.brand_status(pid, self.record.brand)
                if verdict.status == CONTRADICTED:
                    continue
            if self.record.candidate_keys and pid not in {p for p, _s in self.record.candidate_keys}:
                continue
            preferred = 0
            if self.record.preferred_model:
                preferred = 0 if self.checker.model_match(pid, self.record.preferred_model) else 1
            overlap = len(wanted & tokens)
            scored.append((preferred, -overlap, price, pid))
        scored.sort()
        if self.record.preferred_model and self.store.adds == 0 and not self.store.event_target:
            exact = [row for row in scored if row[0] == 0]
            if exact or (self.budget.exploring(0.4) and not self.hunt_over):
                scored = exact
        return [pid for _pref, _o, _p, pid in scored[:limit]]

    def identity_count(self) -> int:
        return len(self.store.identity_clusters(self.record.title_terms))

    def obligations_met(self) -> Tuple[bool, str]:
        if self.record.required_identities:
            have = self.identity_count()
            if have < self.record.required_identities:
                return False, "only %d distinct matching identities surfaced of %d" % (
                    have, self.record.required_identities)
            if not self.store.search_surfaced:
                return False, "no matching product has appeared in a search result"
        if self.record.required_listings:
            inspected = {
                r.listing_id for r in self.store.products.values() if r.inspected and r.listing_id
            }
            if len(inspected) < self.record.required_listings:
                return False, "only %d distinct listings inspected of %d" % (
                    len(inspected), self.record.required_listings)
        if self.record.spec_hidden and self.budget.exploring(0.5):
            return False, "the firm specification has not been stated yet"
        if self.record.clarify_priorities and not self.asked and self.budget.exploring(0.4):
            return False, "the shopper's priorities have not been asked for"
        return True, ""

    def ready_to_commit(self, target: Candidate) -> Tuple[bool, str]:
        met, why = self.obligations_met()
        if not met:
            return False, why
        if self.store.adds == 0 and self.budget.exploring():
            distinct = {c.pid for c in self.eligible()}
            if len(distinct) < 2 and not self.reads_exhausted:
                return False, "no fallback product has been verified yet"
            if self.record.preferred_model and target.relax > 0 and not self.hunt_over:
                return False, "the exact preferred item has not been found yet"
            if self.record.preferred_brand:
                verdict = target.verdicts.get("preferred_brand")
                if (verdict is None or verdict.status != SUPPORTED) and (
                    not self.reads_exhausted or self.retriever.can_widen()
                ):
                    return False, "no fully compliant item of the preferred brand has been read yet"
        if self.record.axes and self.budget.exploring(0.7):
            missing = len(self.record.axes) - len(target.axes)
            if missing > 0 and (not self.reads_exhausted or self.retriever.can_widen()):
                return False, (
                    "the best candidate so far shows %d of the shopper's %d priorities"
                    % (len(target.axes), len(self.record.axes))
                )
        return True, ""

    def answer_text(self) -> str:
        target = self.best()
        if target is None:
            carted = [c for c in self.candidates() if (c.pid, c.sku) in self.store.cart]
            target = carted[0] if carted else None
        if target is None:
            return (
                "Still checking the options against your requirements - nothing meets all of them yet, "
                "so I have not committed to anything."
            )
        parts = ["I am going with " + _clip(self.store.title(target.pid) or target.pid, 90)]
        for concept, wanted in self.record.specs().items():
            value, status = self.checker.variant_spec(target.pid, target.sku, concept)
            if status == SUPPORTED and value == wanted:
                parts.append("its %s is %s" % (concept.replace("_", " "), value))
        if target.price is not None:
            parts.append("it is %.2f %s" % (target.price, self.record.currency))
        if target.in_stock and self.store.verified(target.pid, target.sku):
            parts.append("and I have just re-checked that it is in stock")
        elif target.in_stock:
            parts.append("it was in stock when I last read it, and I am re-checking now")
        else:
            parts.append("its stock is being re-checked")
        return ", ".join(parts) + "."

    def opening_question(self) -> Optional[str]:
        if self.record.spec_hidden:
            return (
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution, refresh rate, or memory), and what value do you need?"
            )
        if self.record.clarify_priorities or (self.record.axes and len(self.record.axes) < 3):
            return (
                "Quick question so I pick well: which of your priorities matters most, and in what "
                "order should I weigh the others?"
            )
        return None

    def act(self, actions: Sequence[Tuple[str, dict]], reason: str = "") -> List[Any]:
        prepared: List[Tuple[str, dict]] = []
        for name, args in actions[: self.max_calls]:
            if not self.catalog.has(name):
                continue
            args = self.catalog.coerce(name, dict(args or {}))
            problem = self.catalog.validate(name, args)
            if problem:
                self._note("skipped an invalid call: " + problem)
                continue
            prepared.append((name, args))
        if not prepared:
            return []
        for name, args in prepared:
            self.store.note_query(name, args)
        result, turn = self.client.call(prepared)
        tool_calls = [
            {
                "id": "%s-%d-%d" % (self.problem_id, turn, index),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for index, (name, args) in enumerate(prepared, start=1)
        ]
        self.dialogue.append(
            {
                "role": "assistant",
                "content": _clip(reason, 300),
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )
        if not isinstance(result, dict):
            self._note("the environment did not answer; that action group was abandoned")
            return []
        observations: List[Any] = []
        for (name, args), call in zip(prepared, result.get("calls") or []):
            wrapper = call.get("observation") if isinstance(call, dict) else None
            wrapper = wrapper if isinstance(wrapper, dict) else {}
            if wrapper.get("done"):
                self.done = True
            body = wrapper.get("observation")
            body = body if isinstance(body, dict) else wrapper
            if wrapper.get("error"):
                self._note("%s failed: %s" % (name, _clip(wrapper.get("error"), 90)))
                observations.append(None)
                continue
            self.store.record(name, args, body, turn)
            observations.append(body)
        message = result.get("user_message")
        if isinstance(message, dict) and str(message.get("content") or "").strip():
            self._absorb_message(str(message["content"]))
        return observations

    def _absorb_message(self, text: str) -> None:
        changes = self.record.apply_shopper_message(text)
        if changes:
            self._note("shopper update: " + "; ".join(changes))
            self._invalidate_on_revision()
        if _QUESTION_RE.search(text):
            self.question_pending = True
            self.question_text = text

    def _invalidate_on_revision(self) -> None:
        budget = self.record.budget
        if budget is None:
            return
        for pid, sku in list(self.store.cart):
            state = self.store.state(pid, sku)
            if state and state.price is not None and state.price > budget + _EPS:
                self.store.dead_keys.add(self.store.key(pid, sku))
                self._note("the carted item is above the new budget and was retired")

    def _read_batch(self) -> List[Tuple[str, dict]]:
        leads = self.leads(limit=10)
        if not leads:
            self.reads_exhausted = True
            return []
        actions: List[Tuple[str, dict]] = []
        if self.catalog.has("compare"):
            chunk = 5
            for start in range(0, min(len(leads), 2 * chunk), chunk):
                group = leads[start:start + chunk]
                if len(group) == 1 and self.catalog.has("view"):
                    actions.append(("view", {"product_id": group[0]}))
                elif group:
                    actions.append(("compare", {"product_ids": group}))
                if len(actions) >= 2:
                    break
        elif self.catalog.has("view"):
            actions.append(("view", {"product_id": leads[0]}))
        return actions

    def _retrieval_batch(self) -> List[Tuple[str, dict]]:
        actions: List[Tuple[str, dict]] = []
        seen: set = set()
        for _ in range(3):
            action = self.retriever.next_action(need_more=self.reads_exhausted)
            if action is None:
                break
            signature = action[0] + json.dumps(action[1], sort_keys=True)
            if signature in seen:
                break
            seen.add(signature)
            self.retriever._mark(action[0], action[1])
            if action[0] == "search":
                self.store.note_query(action[0], action[1])
            actions.append(action)
            if action[0] == "filter":
                break
            if action[0] == "search" and self.record.title_terms:
                break
        return actions

    def _named_candidate_reads(self) -> List[Tuple[str, dict]]:
        missing = [
            pid for pid, _sku in self.record.candidate_keys
            if not self.store.inspected(pid)
        ]
        unique: List[str] = []
        for pid in missing:
            if pid not in unique:
                unique.append(pid)
        if not unique:
            return []
        if self.catalog.has("compare"):
            return [("compare", {"product_ids": unique[:5]})]
        if self.catalog.has("view"):
            return [("view", {"product_id": unique[0]})]
        return []

    def plan(self) -> Tuple[List[Tuple[str, dict]], str]:
        if self.pending_recheck and self.catalog.has("inspect_stock"):
            pid, sku = self.pending_recheck
            return (
                [("inspect_stock", {"product_id": pid, "sku": sku})],
                "re-reading the carted item now the market has had a chance to move",
            )

        if self.question_pending and self.catalog.has("message"):
            return [("message", {"content": self.answer_text()})], "answering the shopper"

        if self.record.candidate_keys:
            reads = self._named_candidate_reads()
            if reads:
                return reads, "reading the named candidate listings"

        if self.record.preferred_model and self.store.adds == 0:
            exact = [c for c in self.eligible() if c.relax == 0]
            if not exact:
                near = [c for c in self.candidates() if c.relax == 1 and c.live]
                action = None
                if near:
                    action = self.retriever.sibling_search(near[0].pid)
                if action is None:
                    action = self.retriever.model_search()
                if action is not None:
                    self.retriever._mark(*action)
                    return [action], "looking for the exact model and colour the shopper named"

        if self.record.preferred_brand and self.store.adds == 0:
            wanted = self.record.preferred_brand
            named = [
                c for c in self.eligible()
                if (c.verdicts.get("preferred_brand") or Verdict("preferred_brand", UNRESOLVED)).status
                == SUPPORTED
            ]
            surfaced = any(
                self.checker.brand_status(pid, wanted).status == SUPPORTED
                for pid in self.leads(limit=10)
            )
            if not named and not surfaced:
                action = self.retriever.brand_hunt()
                if action is not None:
                    self.retriever._mark(*action)
                    return [action], "looking for the brand the shopper named"

        if self.store.event_target and self.store.adds:
            pid = self.store.event_target.split("::", 1)[0]
            action = self.retriever.axis_recovery(pid)
            if action is None:
                action = self.retriever.twin_search(pid)
            if action is not None:
                self.retriever._mark(*action)
                return [action], "the carted item died; looking for an equally good replacement"

        self.reads_exhausted = not self.leads(limit=1)
        target = self.best()
        ready, why = (False, "") if target is None else self.ready_to_commit(target)

        if target is None or not ready:
            reads = self._read_batch()
            if reads:
                return reads, "reading candidate pages so their facts are on the record"
            retrieval = self._retrieval_batch()
            if retrieval:
                return retrieval, "widening the search: " + (why or "no verified candidate yet")
            if target is None:
                return [], ""
            self._note("committing early: " + why if why else "")

        pid, sku = target.pid, target.sku
        if (pid, sku) not in self.store.cart:
            if not self.store.verified(pid, sku) and self.catalog.has("inspect_stock"):
                actions = [("inspect_stock", {"product_id": pid, "sku": sku})]
                runner_up = [c for c in self.eligible() if c.pid != pid][:1]
                if runner_up and self.max_calls > 1:
                    actions.append(
                        ("inspect_stock", {"product_id": runner_up[0].pid, "sku": runner_up[0].sku})
                    )
                return actions, "verifying current price and stock before committing"
            return (
                [("add_to_cart", {"product_id": pid, "sku": sku})],
                "committing to the item that meets every firm requirement",
            )
        if not self.store.verified(pid, sku) and self.catalog.has("inspect_stock"):
            return (
                [("inspect_stock", {"product_id": pid, "sku": sku})],
                "confirming the carted item is still valid",
            )
        return [("place_test_order", self._order_args(pid, sku))], "placing the order"

    def _order_args(self, pid: str, sku: str) -> dict:
        args: Dict[str, Any] = {}
        properties = self.catalog.properties("place_test_order")
        if "product_id" in properties:
            args["product_id"] = pid
        if "sku" in properties:
            args["sku"] = sku
        if "justification" in properties and self.claims.wanted():
            claims = self.claims.build(pid, sku)
            if claims:
                args["justification"] = {"claims": claims}
        return args

    def finish(self) -> None:
        for pid, sku in reversed(self.store.cart):
            candidate = self.checker.assess(pid, sku)
            if not candidate.eligible or self.done:
                continue
            if not self.store.verified(pid, sku) and self.catalog.has("inspect_stock") and self.budget.steps_left() > 1:
                self.act([("inspect_stock", {"product_id": pid, "sku": sku})], "final check")
                candidate = self.checker.assess(pid, sku)
                if not candidate.eligible:
                    continue
            if not self.done:
                self.act([("place_test_order", self._order_args(pid, sku))], "placing the order")
                return
        if self.done or self.budget.steps_left() < 2:
            return
        target = self.best()
        if target is None:
            self._note("no candidate met every firm requirement, so nothing was ordered")
            return
        pid, sku = target.pid, target.sku
        if (pid, sku) not in self.store.cart:
            self.act([("add_to_cart", {"product_id": pid, "sku": sku})], "committing before time runs out")
            if self.done:
                return
            if self.catalog.has("inspect_stock") and self.budget.steps_left() > 1:
                self.act([("inspect_stock", {"product_id": pid, "sku": sku})], "post-commit check")
        candidate = self.checker.assess(pid, sku)
        if candidate.eligible and not self.done:
            self.act([("place_test_order", self._order_args(pid, sku))], "placing the order")

    def run(self) -> List[dict]:
        self._bootstrap()
        last_signature, repeats = "", 0
        while not self.done and self.budget.steps_left() > 0:
            if self.budget.in_endgame():
                self.finish()
                break
            if self.client.transport_failures >= 8:
                self._note("the environment stopped answering; going straight for the order")
                self.finish()
                break
            actions, reason = self.plan()
            if not actions:
                self.reads_exhausted = True
                target = self.best()
                if target is None:
                    retrieval = self._retrieval_batch()
                    if retrieval:
                        actions, reason = retrieval, "last widening of the search"
                    elif not self.hunt_over:
                        self.hunt_over = True
                        self._note("the preferred item was never found; judging the rest")
                        continue
                    else:
                        self.finish()
                        break
                else:
                    self.finish()
                    break
            signature = json.dumps(actions, sort_keys=True)
            repeats = repeats + 1 if signature == last_signature else 0
            last_signature = signature
            if repeats >= 2:
                self._retire(actions)
                last_signature, repeats = "", 0
                continue
            before_adds = self.store.adds
            self.act(actions, reason)
            if self.store.adds > before_adds and self.store.cart:
                self.pending_recheck = self.store.cart[-1]
            elif actions and actions[0][0] == "inspect_stock" and self.pending_recheck:
                self._after_recheck()
            if actions and actions[0][0] == "message":
                self.question_pending = False
                self.asked += 1
        return self._wire_safe(self.dialogue)

    def _bootstrap(self) -> None:
        self._augment_record()
        question = self.opening_question()
        if question and self.catalog.has("message") and self.budget.steps_left() > 6:
            self.asked += 1
            self.act([("message", {"content": question})], "asking the one question that changes the choice")
            self.question_pending = False

    def _augment_record(self) -> None:
        if not self.budget.exploring(0.2):
            return
        needs = (
            self.record.category is None
            or self.record.budget is None
            or (not self.record.specs() and not self.record.title_terms
                and not self.record.candidate_keys and not self.record.axes)
        )
        if not needs:
            return
        answer = self.inference.ask(
            _EXTRACT_PROMPT.format(
                goal=_clip(self.record.goal, 1200), rules=_clip(self.record.rules, 2500)
            ),
            system=_EXTRACT_SYSTEM,
            tries=2,
            reserve_after=_TAIL_SECONDS + 40.0,
        )
        if not isinstance(answer, dict):
            return
        if isinstance(answer.get("category"), str):
            value = answer["category"].strip()
            if 1 < len(value) < 60:
                if self.record.category is None:
                    self.record._add("category", "eq", value, source_text="model reading",
                                     origin="rules")
                elif not (_stems(value) & _stems(self.record.category)):
                    self.record.category_aliases.append(value)
        if self.record.budget is None:
            budget = _num(answer.get("budget"))
            if budget is not None and budget > 0:
                self.record._add("budget", "le", budget, unit=self.record.currency,
                                 scope=VARIANT_SCOPE, source_text="model reading", origin="rules")
        if self.record.brand is None and isinstance(answer.get("brand"), str):
            value = answer["brand"].strip()
            if 1 < len(value) < 40:
                self.record._add("brand", "eq", value, source_text="model reading", origin="rules")
        specs = answer.get("specs")
        if isinstance(specs, dict) and not self.record.specs():
            for concept, raw in specs.items():
                if concept not in _VALUE_READERS:
                    continue
                value = _one(_spec_vals(concept, str(raw)))
                if value:
                    self.record._add("spec:" + concept, "eq", value,
                                     unit=_BARE_UNITS.get(concept, ""), scope=VARIANT_SCOPE,
                                     source_text=str(raw), origin="rules")
        if not self.record.axes:
            wishes = answer.get("preferences")
            if isinstance(wishes, list) and wishes:
                said = " . ".join(str(w) for w in wishes[:6])
                self.record._read_axes(said, base_priority=0)
                self.record._read_freeform_axes("priorities: " + said)
        if not self.record.objectives:
            stated = answer.get("objectives")
            if isinstance(stated, list):
                for entry in stated[:4]:
                    if not isinstance(entry, dict):
                        continue
                    direction = str(entry.get("direction") or "").lower()
                    field_name = str(entry.get("field") or "").lower().replace(" ", "_")
                    if direction in ("max", "min") and (
                        field_name == "price" or field_name in _VALUE_READERS
                    ):
                        self.record.objectives.append(
                            Objective(direction, field_name, "model reading")
                        )
        self._note("requirements: " + self.record.summary())

    def _after_recheck(self) -> None:
        if not self.pending_recheck:
            return
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        candidate = self.checker.assess(pid, sku)
        if candidate.eligible:
            self._note("the carted item survived the market: %s is still valid" % candidate.key)
            return
        self.store.dead_keys.add(self.store.key(pid, sku))
        self.store.event_target = self.store.key(pid, sku)
        self.store.dead_products.add(pid)
        self.retriever.twin_searched = False
        self.reads_exhausted = False
        self._note("the carted item is no longer valid (%s); recovering" % candidate.why_not())

    def _retire(self, actions: Sequence[Tuple[str, dict]]) -> None:
        for name, args in actions:
            if name in ("inspect_stock", "add_to_cart"):
                self.store.dead_keys.add(self.store.key(args.get("product_id"), args.get("sku")))
            if name == "message":
                self.question_pending = False
            if name in ("compare", "view"):
                for pid in (args.get("product_ids") or [args.get("product_id")]):
                    if pid:
                        self.store.dead_products.add(str(pid))
        self._note("that step stopped changing anything, so the controller moved on")

    @staticmethod
    def _wire_safe(dialogue: List[dict]) -> List[dict]:
        if len(json.dumps(dialogue).encode("utf-8")) <= _WIRE_LIMIT:
            return dialogue
        trimmed = [dict(entry) for entry in dialogue]
        for entry in trimmed:
            result = entry.get("environment_result")
            if not isinstance(result, dict):
                continue
            entry["environment_result"] = {
                "call_id": result.get("call_id"),
                "turn": result.get("turn"),
                "user_message": result.get("user_message"),
                "calls": [
                    {
                        "call_id": (call or {}).get("call_id"),
                        "observation": {
                            "done": ((call or {}).get("observation") or {}).get("done"),
                            "error": ((call or {}).get("observation") or {}).get("error"),
                            "observation": _clip(
                                json.dumps(((call or {}).get("observation") or {}).get("observation")),
                                _OBS_CHARS,
                            ),
                        },
                    }
                    for call in (result.get("calls") or [])
                ],
            }
            if len(json.dumps(trimmed).encode("utf-8")) <= _WIRE_LIMIT:
                return trimmed
        while len(trimmed) > 1 and len(json.dumps(trimmed).encode("utf-8")) > _WIRE_LIMIT:
            trimmed.pop(0)
        return trimmed


def agent_main(problem_data: dict) -> List[dict]:
    try:
        return EpisodeController(problem_data).run()
    except Exception as exc:
        return [
            {
                "role": "assistant",
                "content": "agent error: %s: %s" % (type(exc).__name__, str(exc)[:300]),
            }
        ]


__all__ = ["agent_main"]
