from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Dict, List, Optional, Tuple

from src.agent.proxy_client import ProxyClient

_MODELS = {"chutes": "deepseek-ai/DeepSeek-V3.2-TEE", "openrouter": "deepseek/deepseek-v3.2"}
_LLM = ProxyClient(timeout=10, max_retries=1)
_LLM_BUDGET_S = 45.0
_LLM_SPENT = [0.0]
_ENV = ProxyClient(timeout=10, max_retries=4, retry_delay=3.0)
_ENV_GAP_S = 0.12
_ENV_TRIES = 3
_ENV_CALL_CAP_S = 45.0
_HARD_STOP_S = 175.0
DEADLINE_S = 150.0
MAX_SEARCHES = 3
SHORTLIST = 5
DIALOG_LIMIT = 48_000
CLAIMABLE_FIELDS = ("resolution", "refresh_rate", "screen_size")

def _model() -> str:
    return getenv("SANDBOX_MODEL") or _MODELS.get(getenv("INFERENCE_PROVIDER", "openrouter"), _MODELS["openrouter"])

def _f(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        return float(str(v).replace(",", "").lstrip("$"))
    except (TypeError, ValueError):
        return None

def _num(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text, re.I)
    return _f(m.group(1)) if m else None

RES_CLASSES = [("4k", 3840, 2160), ("uhd", 3840, 2160), ("qhd", 2560, 1440), ("full hd", 1920, 1080),
               ("fhd", 1920, 1080), ("1080p", 1920, 1080), ("hd", 1280, 720), ("720p", 1280, 720)]
COLORS = ["black", "white", "blue", "red", "green", "pink", "purple", "gold", "silver", "gray", "grey",
          "yellow", "orange", "brown", "beige", "navy", "teal", "violet", "rose", "graphite", "midnight"]
CATEGORY_LABELS = {
    "smartphone": "Smartphones", "smartphones": "Smartphones", "headphone": "Headphones", "headphones": "Headphones",
    "earbud": "Headphones", "earbuds": "Headphones", "tablet": "Tablets", "tablets": "Tablets", "laptop": "Laptops",
    "laptops": "Laptops", "monitor": "Monitors", "monitors": "Monitors", "smartwatch": "Smartwatches",
    "smartwatches": "Smartwatches", "speaker": "Speakers", "speakers": "Speakers", "camera": "Cameras", "cameras": "Cameras",
    "keyboard": "Keyboards", "keyboards": "Keyboards", "mouse": "Mice", "mice": "Mice", "router": "Routers", "routers": "Routers",
    "tv": "TVs", "tvs": "TVs", "television": "TVs", "televisions": "TVs", "groceries": "Groceries", "snack": "Snacks",
    "snacks": "Snacks", "beverage": "Beverages", "beverages": "Beverages",
}
INCH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|\\u201[cd]|\\?[\"“”])", re.I)
HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
OS_RE = re.compile(r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b", re.I)
SMART_RE = re.compile(r"\bsmart\s*phones?\b", re.I)
CONFLICT_RES = [re.compile(r"\btablets?\b", re.I), re.compile(r"\b(?:feature|basic)\s*phones?\b", re.I),
                re.compile(r"\b(?:mystery|blind|lucky)\s*box(?:es)?\b", re.I),
                re.compile(r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b", re.I)]
ACCESSORY_RE = re.compile(r"\b(?:case|cases|cover|covers|sleeve|sleeves|skin|skins|pouch|protector|cable|cables|strap|straps|ear ?tips?|ear ?hooks?|adapter|holder|stand|replacement|carabiner)\b", re.I)
BUNDLE_RE = re.compile(r"(?:\b(?:with|free|includes?|including|bundle|bundled|comes|plus|and|w)\b|[+&/]|\bfreebies?\b)[^,;|]{0,25}$", re.I)
PREF_ACCESSORY_RE = re.compile(r"\b(ear ?pads?|ear ?tips?|cushions?|case|cover|cable|adapter|stand|holder|replacement|band for|strap for|charger for|screen protector|keyboard for|stylus|pen for|watch band|replacement band)\b", re.I)

def _accessory_title(title: str, rx: "re.Pattern") -> bool:
    text = title or ""
    for m in rx.finditer(text):
        if BUNDLE_RE.search(text[max(0, m.start() - 30):m.start()]):
            continue
        return True
    return False

FEATURE_PHONE_RE = re.compile(r"\b(?:feature|basic|dumb|button|keypad) ?phones?\b", re.I)
BAD_ITEM_WORDS = ("customer rating", "customer review", "compare ")
SHADES = ["midnight blue", "titanium black", "matte black", "triple black", "frost white", "white smoke", "cloud grey", "moonlight grey",
          "space gray", "street gray", "driftwood sand", "cypress green", "tidal blue", "light blue", "sky blue", "indigo blue", "canvas white",
          "platinum silver", "starlight", "graphite", "sandstone", "transparent", "silver", "beige", "black", "blue", "white", "gray", "grey",
          "green", "pink", "violet", "purple", "red", "sage", "gold", "khaki", "peach", "sage green"]
_SHADES_BASE = len(SHADES)
TITLE_MODEL_RE = re.compile(r"\b[a-z]*\d{2,5}[a-z0-9]*(?:-[a-z0-9]+)*\b", re.I)
NON_MODEL_RE = re.compile(r"^(?:android|bluetooth|wifi|ips|ddr|lpddr|mtk)|(?:gb|tb|mb|mah|hz|khz|mhz|ghz|mp|fps|inch|inches|p|g)$|^\d+x\d+$", re.I)
SPORT_RE = r"\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\b"
KID_RE = r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b"
NAV_RE = r"\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\b"
RUGGED_RE = r"\b(10\s*atm|5\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|fiber[- ]reinforced|titanium|rugged|tactical|tank)\b"
KIDWATCH_RE = r"\b(kid|kids|child|children|myfirst|watch phone)\b"
AXIS_WORDS: Tuple[Tuple[str, str], ...] = (
    ("water", r"\b(?:sweat\w*|water[- ]?(?:proof\w*|resistan\w+)?|weather|rain\w*|swim\w*|splash\w*)\b"),
    ("sport", r"\b(?:sports?|secure fit|stays? put|workouts?|gym|running|runs?|jogging|bootcamp|training|marathon)\b"),
    ("battery", r"\b(?:battery|all[- ]day|long[- ]lasting|charg\w+|multi[- ]day|playtime)\b"),
    ("wireless", r"\b(?:wireless|bluetooth|cable[- ]free|no cables?|cordless)\b"),
    ("kid", r"\b(?:kids?|child\w*|toddlers?|\d+[- ]year[- ]old|drop protection|drops? (?:it|everything|things)|durab\w+|rugged case)\b"),
    ("size", r"\b(?:size|small hands|(?:can|could) (?:actually )?hold|\d+\s*(?:to|-)\s*\d+\s*inch)\b"),
    ("wifi", r"\b(?:wi-?fi|downloads?|streaming)\b"),
    ("navigation", r"\b(?:gps|navigation|maps?|trails?|hiking|off[- ]grid|backpacking|compass)\b"),
    ("rugged", r"\b(?:rugged\w*|survives?|rough handling|tough|shock\w*)\b"),
)
AXIS_MERGE = {"smartwatch": {"water": "rugged"}, "tablet": {"rugged": "kid", "water": "kid"}, "headphone": {"rugged": "water"}}
DEFAULT_AXES = {"headphone": ["water", "sport", "battery", "wireless"], "tablet": ["kid", "battery", "size", "wifi"], "smartwatch": ["navigation", "rugged", "battery"]}
RANK_CONTEXT_RE = r"\bties\b|\btie[- ]?break\w*|\btrade[- ]?off\w*|\bpriorit\w+|\brank\w*|\bchoose\b|\bpick\b|\bcompare\b"
PRIORITY_WORD = r"(?:count(?:s)?|matter(?:s)?|rank(?:s)?|weigh(?:s)?|come(?:s)?|is|are)"
RANK_MARKERS_RE = re.compile(PRIORITY_WORD + r"\s+(?:the\s+)?(?:most|least|first|second|third|equally|next)\b|\bclose behind\b|\btop priority\b|\bleast important\b|\bmost important\b", re.I)
UNITS: Dict[str, Tuple[str, float]] = {
    "mah": ("mah", 1.0), "wh": ("wh", 1.0), "mp": ("mp", 1.0), "megapixel": ("mp", 1.0), "megapixels": ("mp", 1.0),
    "kg": ("g", 1000.0), "g": ("g", 1.0), "gram": ("g", 1.0), "grams": ("g", 1.0), "lb": ("g", 453.6), "lbs": ("g", 453.6), "oz": ("g", 28.35),
    "w": ("w", 1.0), "watt": ("w", 1.0), "watts": ("w", 1.0), "kw": ("w", 1000.0),
    "h": ("h", 1.0), "hr": ("h", 1.0), "hrs": ("h", 1.0), "hour": ("h", 1.0), "hours": ("h", 1.0),
    "tb": ("gb", 1000.0), "gb": ("gb", 1.0), "mb": ("gb", 0.001),
    "ghz": ("mhz", 1000.0), "mhz": ("mhz", 1.0), "khz": ("hz", 1000.0), "hz": ("hz", 1.0),
    "cm": ("mm", 10.0), "mm": ("mm", 1.0), "m": ("mm", 1000.0), "meters": ("mm", 1000.0), "meter": ("mm", 1000.0),
    "l": ("ml", 1000.0), "liter": ("ml", 1000.0), "liters": ("ml", 1000.0), "litre": ("ml", 1000.0), "litres": ("ml", 1000.0), "ml": ("ml", 1.0),
    "in": ("in", 1.0), "inch": ("in", 1.0), "inches": ("in", 1.0), "\"": ("in", 1.0),
    "db": ("db", 1.0), "lm": ("lm", 1.0), "lumens": ("lm", 1.0), "nits": ("nits", 1.0), "rpm": ("rpm", 1.0), "ppi": ("ppi", 1.0), "%": ("pct", 1.0),
    "core": ("core", 1.0), "cores": ("core", 1.0), "fps": ("fps", 1.0), "ms": ("ms", 1.0), "mbps": ("mbps", 1.0), "gbps": ("mbps", 1000.0),
}
_UNIT_ALT = "|".join(sorted((re.escape(u) for u in UNITS if u != "\""), key=len, reverse=True))
UNIT_VALUE_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(" + _UNIT_ALT + r"|\")(?![a-z])", re.I)
REQ_TAIL_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(" + _UNIT_ALT + r")\b(?:\s+(?:of\s+)?([a-z][a-z\- ]{2,24}?))?(?=\s*(?:[,.;)]|\b(?:and|at|under|below|budget|for|that|which|with|in)\b|$))", re.I)
REQ_HEAD_RE = re.compile(r"\b([a-z][a-z\- ]{2,24}?)\s+of\s+(?:at least\s+|at most\s+|about\s+|around\s+)?(\d+(?:\.\d+)?)\s*(" + _UNIT_ALT + r")\b", re.I)
REQ_STOP = {"at", "or", "under", "below", "in", "stock", "and", "with", "budget", "php", "the", "a", "an", "of", "up", "to", "least", "most", "than", "more", "less", "about", "around", "exactly", "for", "that", "which", "one", "please", "must", "have", "has", "i", "need", "want", "is"}

GB_PAIR_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:gb|g)?\s*[+/|]\s*(\d+(?:\.\d+)?)\s*(gb|tb|g)\b", re.I)

def _pair_aware_values(req: Dict[str, Any], text: str) -> List[float]:
    if req["base"] != "gb":
        return _unit_values(text, req["base"])
    name = req.get("name") or ""
    pairs = list(GB_PAIR_RE.finditer(text or ""))
    if pairs and re.search(r"storage|rom|ssd|disk|capacity|memory|ram", name):
        out: List[float] = []
        for m in pairs:
            a, b = float(m.group(1)), float(m.group(2)) * (1000.0 if m.group(3).lower() == "tb" else 1.0)
            out.append(min(a, b) if re.search(r"ram|memory", name) else max(a, b))
        return out
    return _unit_values(text, "gb")

def _unit_base(unit: str) -> Tuple[str, float]:
    return UNITS.get((unit or "").lower(), ("", 1.0))

def _unit_values(text: str, base: str) -> List[float]:
    out: List[float] = []
    for m in UNIT_VALUE_RE.finditer(text or ""):
        b, k = _unit_base(m.group(2))
        if b == base:
            try:
                out.append(float(m.group(1).replace(",", ".")) * k)
            except ValueError:
                continue
    return out

def _req_name(words: str) -> str:
    toks = [w for w in re.findall(r"[a-z]+", (words or "").lower()) if w not in REQ_STOP]
    return " ".join(toks[:3])

def _make_req(num: float, unit: str, name: str, raw: str, op: str = "=") -> Optional[Dict[str, Any]]:
    base, k = _unit_base(unit)
    if not base:
        return None
    return {"name": _req_name(name), "num": num * k, "base": base, "unit": unit.lower(), "value": f"{num:g}{unit.lower()}", "text": f"{num:g} {unit.lower()}", "raw": raw.strip(), "op": op}

def _req_matches(req: Dict[str, Any], values: List[float]) -> Optional[bool]:
    if not values:
        return None
    want = float(req["num"])
    tol = 1e-6 * max(1.0, abs(want))
    if req.get("op") == ">=":
        return any(v >= want - tol for v in values)
    if req.get("op") == "<=":
        return any(v <= want + tol for v in values)
    distinct = {round(v, 6) for v in values}
    if len(distinct) > 1:
        return None
    return abs(values[0] - want) <= tol
RANK_STEP_RE = re.compile(PRIORITY_WORD + r"\s+(?:the\s+)?(most|least|first|second|third|equally(?:\s+next)?|next)\b|\b(close behind)\b|\b(top priority)\b|\b(least important)\b|\b(most important)\b", re.I)
RANK_ORDER = {"most": 0, "first": 0, "top priority": 0, "most important": 0, "close behind": 1, "second": 1, "next": 2, "equally": 2, "equally next": 2, "third": 2, "least": 3, "least important": 3}

def _rank_step_key(m: "re.Match") -> int:
    for g in m.groups():
        if g:
            return RANK_ORDER.get(g.strip().lower(), 2)
    return 2

def _ranked_axes(low: str, category: str) -> List[str]:
    ranks: Dict[str, int] = {}
    pos = 0
    first = True
    for m in RANK_STEP_RE.finditer(low):
        seg = low[pos:m.start()]
        pos = m.end()
        found = _axes_in_text(seg, category)
        if not found:
            phrase = _axis_phrase(seg, tail=True)
            if 2 <= len(phrase) <= 40 and not re.search(r"\d{3,}|php|budget|category|stock", phrase):
                found = ["x:" + phrase]
            else:
                continue
        chosen = [found[-1]] if first else found
        first = False
        for a in chosen:
            ranks.setdefault(a, _rank_step_key(m))
    return [a for a, _ in sorted(ranks.items(), key=lambda x: x[1])]

def _lead_axis(low: str, category: str) -> str:
    for pat in (r"(?:rather|prefer(?:ably)?|willing to)\s+(?:pay|spend|go)[^.;]*?\bfor\s+([^.;]+?)\s+(?:than|over)\b",
                r"(?:^|[-;,.]|\bso\b)\s*([^.;,-]+?)\s+" + WEIGHT_TAIL):
        m = re.search(pat, low)
        if m:
            axes = _axes_in_text(m.group(1), category)
            if axes:
                return axes[0]
    return ""

AXIS_QUERY_WORDS = {"water": "waterproof sweatproof", "sport": "sport", "battery": "long battery", "wireless": "wireless bluetooth", "kid": "kids drop proof", "size": "", "wifi": "wifi", "navigation": "gps", "rugged": "rugged waterproof"}

def _cat_key(category: str) -> str:
    cat = (category or "").lower()
    for key in ("headphone", "tablet", "smartwatch"):
        if cat.startswith(key):
            return key
    return cat

def _axes_in_text(low: str, category: str) -> List[str]:
    key = _cat_key(category)
    merge = AXIS_MERGE.get(key, {})
    found: List[Tuple[int, str]] = []
    for axis, pat in AXIS_WORDS:
        m = re.search(pat, low)
        if m:
            found.append((m.start(), merge.get(axis, axis)))
    out: List[str] = []
    for _pos, axis in sorted(found):
        if axis not in out:
            out.append(axis)
    return out

def _pref_axes(cons: "Constraints") -> List[str]:
    known = {a for a, _ in AXIS_WORDS}
    axes = [a for a in cons.pref_words if a in known or a.startswith("x:")]
    return axes or list(DEFAULT_AXES.get(_cat_key(cons.category), []))

def _axis_phrase(seg: str, tail: bool = True) -> str:
    s = seg.lower()
    s = re.split(r"\d[\d.,]*\s*(?:php|pesos?|₱|\$|usd)\b", s)[-1]
    s = re.split(r"\b(?:that|which|so that|because|when|while)\b", s)[0]
    s = re.sub(r"\b(?:with|and|a|an|the|my|your|counts?|then|also|plus|close|behind|next|least|most|several|together|in stock|not|accessory|device)\b", " ", s)
    s = re.sub(r"[^a-z0-9/ \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")
    words = [w for w in s.split() if not re.fullmatch(r"\d+", w)]
    if not words:
        return ""
    return " ".join(words[-4:] if tail else words[:4])

WEIGHT_TAIL = r"(?:matters?(?:\s+\w+){0,2}|beats?(?:\s+\w+){0,3}\s+price|is\s+(?:\w+\s+){0,3}(?:care|want|need|priorit\w+|point)\w*|counts?(?:\s+\w+){0,2}|wins?)"
RECOVERY_HINT_RE = re.compile(r"colou?r\s*/\s*model|another colou?r|same model|exact (?:colou?r|model|one|item)|chang\\w+ models?|that exact", re.I)
PREF_CLAUSE_RE = re.compile(r"(?:rather|prefer(?:ably)?|willing to)\s+(?:pay|spend|go)[^.;]*?\bfor\s+([^.;]+?)\s+(?:than|over)\b|(?:^|[-;,.]|\bso\b)\s*([^.;,-]+?)\s+" + WEIGHT_TAIL)

def _axis_weights(n: int) -> List[float]:
    if n <= 0:
        return []
    total = n * (n + 1) / 2.0
    return [(n - i) / total for i in range(n)]

def _long_battery(text: str) -> float:
    days = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*days?", text)]
    hours = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hours?", text)]
    if days:
        best = max(days)
        return 1.0 if best >= 10 else 0.75 if best >= 5 else 0.5 if best >= 2 else 0.0
    if hours:
        best = max(hours)
        return 1.0 if best >= 120 else 0.75 if best >= 72 else 0.5 if best >= 36 else 0.0
    if "solar" in text or "battery life" in text:
        return 0.5
    return 0.0
WATER_RE = r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b"
WIRELESS_RE = r"\b(wireless|bluetooth|true wireless)\b"

def _only(vals: set) -> Optional[str]:
    return next(iter(vals)) if len(vals) == 1 else None

def _inch_vals(text: str) -> set:
    return {f"{float(m.group(1)):g}in" for m in INCH_RE.finditer(text or "")}

def _hz_vals(text: str) -> set:
    return {f"{float(m.group(1)):g}hz" for m in HZ_RE.finditer(text or "")}

def _res_class(a: int, b: int) -> Optional[str]:
    if a >= 3840 and b >= 2160:
        return "4k"
    if a >= 2560 and b >= 1440:
        return "qhd"
    if a >= 1920 and b >= 1080:
        return "fhd"
    if a >= 1280 and b >= 720:
        return "hd"
    return None

def _res_vals(text: str) -> set:
    if not text or re.search(r"\bqHD\b", text):
        return set()
    low = text.lower()
    vals = set()
    for m in WXH_RE.finditer(low):
        rc = _res_class(*sorted((int(m.group(1)), int(m.group(2))), reverse=True))
        if rc:
            vals.add(rc)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        vals.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        vals.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        vals.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        vals.add("hd")
    return vals

RAM_PAIR_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
RAM_TITLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
PAIR_HINT_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I)
RES_WORD_RE = re.compile(r"\b(?:qHD|qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", re.I)

def _ram_vals(text: str) -> set:
    paired = {f"{float(a):g}gb" for a, _b in RAM_PAIR_RE.findall(text or "")}
    if paired:
        return paired
    return {f"{float(n):g}gb" for n in RAM_TITLE_RE.findall(text or "")}

def _gb_vals(text: str) -> set:
    return {f"{float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1):g}gb" for m in GB_RE.finditer(text or "")}

VALUE_SETS = {"screen_size": _inch_vals, "refresh_rate": _hz_vals, "resolution": _res_vals, "ram": _ram_vals}
FACT_LABEL = {"screen_size": "size", "refresh_rate": "refresh_rate", "resolution": "display", "ram": "storage"}
INVALID_TITLE_FRAGMENTS = ("earbud", "in-ear", "in ear", "earphone", "open-ear", "true wireless")
OVER_ON_EAR_RE = re.compile(r"\b(?:over|on)[-\s]?ear\b", re.I)

def _norm_spec(field: str, text: str) -> Optional[str]:
    if field == "resolution":
        return _only(_res_vals(text))
    raw = (text or "").strip()
    vals = VALUE_SETS[field](raw)
    if not vals and re.fullmatch(r"\d+(?:\.\d+)?", raw):
        vals = {f"{float(raw):g}{'in' if field == 'screen_size' else 'hz'}"}
    return _only(vals)

def _short_battery(text: str) -> float:
    hours = [float(m.group(1)) for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b", text, re.I)]
    if re.search(r"\d+(?:\.\d+)?\s*days?", text, re.I):
        return 1.0
    if hours:
        best = max(hours)
        if best >= 15:
            return 1.0
        if best >= 10:
            return 0.75
        if best >= 6:
            return 0.5
    if "battery life" in text:
        return 0.25
    return 0.0

def _model_code(raw: str) -> Optional[str]:
    words = re.findall(r"[a-z0-9]+", (raw or "").lower())
    while words and words[-1] in SHADES:
        words.pop()
    code = "".join(words)
    return code.upper() if 3 <= len(code) <= 24 and any(ch.isdigit() for ch in code) else None

MODEL_SUFFIXES = ("ULTRA", "PRIME", "TURBO", "POWER", "PLUS", "LITE", "MINI", "NOTE", "EDGE", "PLAY", "PRO", "MAX", "NEO", "AIR", "5G", "4G", "GO", "FE", "GT", "SE")
MODEL_WORDS = ("SMARTPHONE", "CELLPHONE", "HEADPHONES", "HEADPHONE", "EARPHONES", "EARPHONE", "BLUETOOTH", "WIRELESS", "HEADSET", "GAMING", "MOBILE", "PHONE", "TABLET")
_PEEL = tuple(sorted(set(MODEL_SUFFIXES + MODEL_WORDS), key=len, reverse=True))

def _model_variants(code: str) -> List[str]:

    code = (code or "").strip().upper()
    if not code or " " in code or len(code) < 4:
        return []
    parts: List[str] = []
    rest = code
    while True:
        for s in _PEEL:
            if rest.endswith(s) and len(rest) > len(s):
                parts.insert(0, s)
                rest = rest[:-len(s)]
                break
        else:
            break
    pre: List[str] = []
    while True:
        for s in MODEL_WORDS:
            if rest.startswith(s) and len(rest) > len(s):
                pre.append(s)
                rest = rest[len(s):]
                break
        else:
            break
    m = re.match(r"^([A-Z]{3,})(\d+)$", rest)
    head = [m.group(1), m.group(2)] if m else [rest]
    out = " ".join(pre + head + parts)
    return [out] if out != code else []

def _color_label(options: str, color_fact: str, title: str) -> Optional[str]:
    for text in (options or "", color_fact or "", title or ""):
        matches = []
        for shade in SHADES:
            pat = r"(?<![a-z0-9])" + r"\W+".join(re.escape(w) for w in shade.split()) + r"(?![a-z0-9])"
            for m in re.finditer(pat, text.lower()):
                matches.append((m.start(), m.end(), "green" if shade == "sage green" else shade))
        if not matches:
            continue
        longest = [m for m in matches if not any(o[0] <= m[0] and o[1] >= m[1] and (o[1] - o[0]) > (m[1] - m[0]) for o in matches)]
        labels = {m[2] for m in longest}
        return next(iter(labels)) if len(labels) == 1 else None
    return None

PREF_AXES = {
    "water": r"\b(ipx?[4-8]|ip5\d|ip6\d|sweat ?(?:proof|resistant|resistance)|water ?(?:proof|resistant|resistance)|waterproof)\b",
    "battery": r"\b(\d{2,3}\s*(?:h|hr|hrs|hours)|\d{3,5}\s*mah|long battery|playtime)\b",
    "wireless": r"\b(wireless|bluetooth|bt\s*5|true wireless|tws)\b",
    "sport": r"\b(sport|running|gym|workout|earhook|ear hook|secure fit)\b",
    "noise": r"\b(anc|noise cancel\w*)\b",
    "rugged": r"\b(rugged|mil-std|shockproof|drop ?proof|sapphire)\b",
}

CATEGORY_STOP = {"with", "at", "for", "that", "which", "under", "below", "up", "in", "and", "or", "to", "from", "priced", "costing", "around", "about", "please", "now", "today"}
NOUN_SKIP = {"new", "good", "cheap", "decent", "nice", "great", "reliable", "genuine", "actual", "real", "budget", "affordable", "quality", "solid", "proper", "another", "some", "any", "one", "single", "test", "order"}

def _head_noun(low: str) -> str:
    m = re.search(r"\b(?:i need|i want|i am looking for|i'm looking for|looking for|need|buy me|find me|get me|order|for)\s+(?:an?\s+|some\s+|the\s+)?((?:[a-z][a-z\-]+\s+){0,3}[a-z][a-z\-]+)", low)
    if not m:
        return ""
    words = [w for w in m.group(1).split()]
    out: List[str] = []
    for w in words:
        if w in CATEGORY_STOP:
            break
        out.append(w)
    out = [w for w in out if w not in NOUN_SKIP and not re.search(r"\d", w)]
    if not out:
        return ""
    noun = out[-2:] if len(out) >= 2 and out[-2] not in ("in-stock", "in", "stock") else out[-1:]
    noun = [w for w in noun if w not in ("in-stock",)]
    if not noun or len(noun[-1]) < 2:
        return ""
    label = " ".join(noun)
    if not label.endswith("s"):
        label = label + ("es" if label.endswith(("ch", "sh", "x", "s")) else "s")
    return " ".join(w.capitalize() for w in label.split())

ASC_WORDS = {"lowest", "smallest", "cheapest", "lightest", "least", "shortest", "minimum", "min", "fewest", "thinnest", "slowest"}
DESC_WORDS = {"highest", "largest", "biggest", "longest", "most", "fastest", "greatest", "maximum", "max", "heaviest", "brightest", "widest", "tallest", "strongest"}
RANK_AXIS_RE = re.compile(r"\b(highest|largest|biggest|longest|most|fastest|greatest|maximum|max|heaviest|brightest|widest|tallest|strongest|lowest|smallest|cheapest|lightest|least|shortest|minimum|min|fewest|thinnest|slowest)\s+((?:[a-z][a-z\-]*\s?){1,3}?)(?=\s*(?:[,.;)]|\bamong\b|\bthen\b|\bif\b|\band\b|\bincluding\b|$))")
RES_RANK = {"hd": 1.0, "fhd": 2.0, "qhd": 3.0, "4k": 4.0}

REPLY_REQUEST_RE = re.compile(r"(?:^|[.!?;]\s*)(?:please\s+|kindly\s+)?(?:confirm|reply|respond|answer|tell me|let me know|which one|what are you)\b", re.I)
EVENT_NOTICE_RE = r"more expensive|cheaper|price (?:went|jumped|changed|rose|dropped)|from\s+[\d.,]+\s+to\s+[\d.,]+|still (?:available|in stock)|out of stock|sold out"
CURRENCY_RE = r"php|pesos?|peso|₱|\$|usd|dollars?|eur|euros?|£|gbp|thb|baht|idr|rupiah|myr|ringgit|sgd|vnd|dong|inr|rupees?|yen|jpy|krw|won|aud|cad|rmb|yuan|cny"

def _budget_num(t: str, pattern: str) -> Optional[float]:
    low = t.lower()
    for m in re.finditer(pattern, low):
        tail = low[m.end(): m.end() + 18]
        head = low[max(0, m.start() - 24): m.start()]
        if re.search(CURRENCY_RE, tail) or re.search(r"budget|spend|afford|pay|\bgo up to\b", head + " " + m.group(0)):
            v = _f(m.group(1))
            if v is not None:
                return v
    return None

class Constraints:
    def __init__(self) -> None:
        self.budget: Optional[float] = None
        self.category: str = ""
        self.brand: str = ""
        self.brand_hard: bool = False
        self.size_in: Optional[float] = None
        self.refresh_hz: Optional[float] = None
        self.res_class: str = ""
        self.ram_gb: Optional[float] = None
        self.model: str = ""
        self.color: str = ""
        self.pref_words: List[str] = []
        self.sealed: bool = False
        self.reqs: List[Dict[str, Any]] = []
        self.category_authoritative: bool = False
        self.llm_fields: set = set()
        self.device_hint: Optional[bool] = None
        self.noun_phrase: str = ""
        self.category_alts: List[str] = []
        self.wants_reason: bool = False
        self.ranking: bool = False
        self.rank_axes: List[Tuple[str, str]] = []
        self.phrase: str = ""
        self.raw_goal: str = ""
        self.candidate_keys: List[Tuple[str, str]] = []
        self.reported_empty: bool = False
        self.spec_want: str = ""
        self.res_phrase: str = ""

    def spec_value(self) -> str:
        field = self.spec_field()
        if field == "screen_size" and self.size_in is not None:
            return f"{self.size_in:g}in"
        if field == "refresh_rate" and self.refresh_hz is not None:
            return f"{self.refresh_hz:g}hz"
        if field == "resolution":
            return _only(_res_vals(self.res_class)) or ""
        if field == "ram" and self.ram_gb is not None:
            return f"{self.ram_gb:g}gb"
        return ""

    def has_spec(self) -> bool:
        return bool(self.spec_field() or self.reqs)

    def _absorb_reqs(self, t: str, low: str, dry: bool = False) -> List[Dict[str, Any]]:
        if re.search(EVENT_NOTICE_RE, low):
            return []
        found: List[Dict[str, Any]] = []
        for m in REQ_HEAD_RE.finditer(low):
            req = _make_req(float(m.group(2)), m.group(3), m.group(1), m.group(0))
            if req:
                found.append(req)
        for m in REQ_TAIL_RE.finditer(low):
            req = _make_req(float(m.group(1)), m.group(2), m.group(3) or "", m.group(0))
            if req:
                found.append(req)
        quoted = [(m.start(), m.end()) for m in re.finditer(r"[\"“‘'][^\"”’']{2,80}[\"”’']", low)]
        kept: List[Dict[str, Any]] = []
        for req in found:
            base = req["base"]
            pos = low.find(req["raw"])
            if any(a <= pos < b for a, b in quoted):
                continue
            if (base == "in" and self.size_in is not None) or (base == "hz" and self.refresh_hz is not None) or (base == "gb" and self.ram_gb is not None and re.search(r"\b(?:ram|memory)\b", low[pos: pos + len(req["raw"]) + 12] if pos >= 0 else req["raw"])):
                continue
            if base == "h" and self.pref_words:
                continue
            ctx = low[max(0, pos - 30) : pos] if pos >= 0 else ""
            head = req["raw"] + " " + ctx
            if re.search(r"\b(?:at least|minimum|min\.?|or more|no less than|over|above|more than|from)\b", head):
                req["op"] = ">="
            elif re.search(r"\b(?:at most|maximum|max\.?|no more than|up to|under|below|less than|within)\b", head):
                req["op"] = "<="
            if not req["name"]:
                prev = [w for w in re.findall(r"[a-z]+", ctx) if w not in REQ_STOP]
                if prev:
                    req["name"] = prev[-1]
            if dry:
                kept.append(req)
            elif not any(r["base"] == base and abs(r["num"] - req["num"]) < 1e-9 for r in self.reqs):
                self.reqs.append(req)
        return kept

    def spec_field(self) -> str:
        if self.res_class:
            return "resolution"
        if self.refresh_hz is not None:
            return "refresh_rate"
        if self.size_in is not None:
            return "screen_size"
        if self.ram_gb is not None:
            return "ram"
        return ""

    def absorb(self, text: str, from_shopper: bool = False) -> None:
        t = text or ""
        low = t.lower()
        b = _budget_num(t, r"(?:at or under|at or below|under|below|up to|at most|budget(?: is| of)?(?: at most)?|spend(?: up to)?|max(?:imum)?|go up to|no more than|cost(?:s|ing)? (?:at or )?(?:under|below)|<=)\s*(?:php|₱)?\s*([\d][\d,]*(?:\.\d+)?)")
        if b is None and not re.search(EVENT_NOTICE_RE, low):
            b = _num(t, r"([\d][\d,]*(?:\.\d+)?)\s*(?:php|pesos?)\b(?! now| more)")
        if b is not None and (from_shopper or self.budget is None or re.search(r"\bgo up to\b", low) or "now" in low):
            self.budget = b
        m = re.search(r"(?:required|exact|catalog)?\s*categor(?:y|ies)\s*(?:is|are|:|=)\s*([A-Za-z][\w&',\-/ ]{2,60}?)\s*(?:[.;]|$)", t) or re.search(r"\bin the\s+([A-Z][\w&',\-/ ]{2,60}?)\s+categor(?:y|ies)\b", t)
        if m and (from_shopper or not self.category):
            self.category = m.group(1).strip()
            self.category_authoritative = True
        elif not self.category:
            m2 = re.search(r"\b(smartphones?|headphones?|earbuds?|tablets?|laptops?|monitors?|smartwatch(?:es)?|speakers?|cameras?|keyboards?|mice|mouse|routers?|tvs?|televisions?|groceries|snacks?|beverages?)\b", low)
            if m2:
                self.category = CATEGORY_LABELS.get(m2.group(1), m2.group(1).capitalize())
                mod = re.search(r"(?:^|\s)([A-Za-z][A-Za-z\-]{2,})\s+$", t[max(0, m2.start() - 24): m2.start()])
                if mod and mod.group(1)[0].islower() and mod.group(1) not in CATEGORY_STOP and mod.group(1) not in NOUN_SKIP and mod.group(1) not in ("in-stock", "stock", "an", "a", "the", "of") and mod.group(1) != (self.brand or "").lower():
                    self.noun_phrase = f"{mod.group(1)} {m2.group(1)}"
                    alt = " ".join(w.capitalize() for w in self.noun_phrase.split())
                    if not alt.endswith("s"):
                        alt += "es" if alt.endswith(("ch", "sh", "x", "s")) else "s"
                    if alt.lower() != self.category.lower() and alt not in self.category_alts:
                        self.category_alts.append(alt)
            else:
                guess = _head_noun(low)
                if guess:
                    self.category = guess
        m = re.search(r"required brand is\s+([A-Za-z][\w\- ]{1,30}?)[.,]", t)
        if m:
            self.brand, self.brand_hard = m.group(1).strip(), True
        elif not self.brand:
            m = re.search(r"\b(?:i|we)\s+(?:would\s+)?prefer\s+([^,.;]{2,40}?)\s*[,;]?\s*(?:but|although|though|however)\b", t, re.I)
            if m:
                self.brand, self.brand_hard = m.group(1).strip(), False
            else:
                m = re.search(r"\b(?:prefer(?:ably)?|preferred brand(?: is)?|ideally)\s+(?:a\s+|an\s+)?([A-Z][\w\-]{1,20})", t)
                if m:
                    self.brand, self.brand_hard = m.group(1), False
        s = _num(t, r"(?<![\d.])(\d{1,3}(?:\.\d{1,3})?)\s*(?:-|\s)?(?:inch(?:es)?\b|\")")
        if s is not None and 1.0 <= s <= 100.0:
            self.size_in = s
        else:
            s = _num(t, r"(?<![\d.])(\d{1,3}(?:\.\d{1,3})?)\s*(?:-|\s)?in\b")
            if s is not None and 3.0 <= s <= 100.0:
                self.size_in = s
        hz = _num(t, r"(?<![\d.])(\d{2,3})\s*hz")
        if hz is not None:
            self.refresh_hz = hz
        found = [(m.start(), name) for name, _, _ in RES_CLASSES for m in [re.search(r"\b" + re.escape(name) + r"\b", low)] if m]
        if found:
            self.res_class = min(found)[1]
            m = re.search(r"((?:\d{3,4}p\s+)?(?:full\s+hd|ultra\s+hd|qhd|uhd|4k|fhd|hd)(?:\s+\(\d{3,4}p\))?|\d{3,4}p)", low)
            if m:
                self.res_phrase = m.group(1)
        ram = _num(t, r"(\d{1,3})\s*gb\s*(?:of\s*)?(?:ram|memory)")
        if ram is not None:
            self.ram_gb = ram
        if re.search(r"\b(why|reason|walk me through|explain|justif)", low):
            self.wants_reason = True
        if re.search(r"\b(?:" + "|".join(sorted(DESC_WORDS | ASC_WORDS)) + r")\s+[a-z]", low) and re.search(RANK_CONTEXT_RE, low):
            self.ranking = True
        if re.search(RANK_CONTEXT_RE, low):
            for m in RANK_AXIS_RE.finditer(low):
                direction = "asc" if m.group(1) in ASC_WORDS else "desc"
                phrase = re.sub(r"\s+", " ", m.group(2)).strip()
                if phrase and (direction, phrase) not in self.rank_axes:
                    self.rank_axes.append((direction, phrase))
            if self.rank_axes:
                self.ranking = True
        m = re.search(r'(?:match(?:es|ing)?|contain(?:s|ing)?|includ(?:es|ing)|titled|named|with)\s+(?:the\s+)?(?:phrase|words?|terms?|text)?\s*["“‘\']([^"”’\']+)["”’\']', t, re.I) or re.search(r'something like\s+["“‘\']([^"”’\']+)["”’\']', t) \
            or re.search(r'match(?:es|ing)\s+(?:the\s+phrase\s+)?([a-z0-9][a-z0-9 \-]{1,40}?)\s*(?:,|\s+at\b|\s+under\b|\s+below\b|\s+in\s+stock|\s+and\s+buy|$)', t, re.I)
        if m and not self.phrase:
            self.phrase = m.group(1).strip().strip('"')
        for pid, sku in re.findall(r"(\d{2,})::([\w\-]+)", t):
            if (pid, sku) not in self.candidate_keys:
                self.candidate_keys.append((pid, sku))
        if not self.brand:
            for m in re.finditer(r"(?<=[a-z,]\s)([A-Z][\w\-\.']+(?:\s+[A-Za-z][\w\-\.']+){0,2})\s+(?=\d+p\b|\d+(?:\.\d+)?-inch|\d+(?:\.\d+)?\s*inch|\d+\s*[Hh][Zz]|\d+\s*gb\b|(?i:qhd|uhd|fhd|4k|2k|8k|full hd|hd)\b|gaming\b|smartphone|phone\b|tablet|monitor|headphone|laptop|smartwatch)", t):
                words = m.group(1).split()
                while words and re.fullmatch(r"(?i:hd|fhd|qhd|uhd|led|lcd|ips|va|tn|oled|rgb|usb|hdmi|lte|nfc|gps|ram|rom|gb|tb|pc|tv|ai|4g|5g|full|ultra|new|in-stock|stock|gaming|curved|portable|smartphone|phone|tablet|monitor|laptop|headphone|smartwatch|with|and|for|the|a|an)|.*\d.*", words[-1]):
                    words.pop()
                if not words or any(re.fullmatch(r"(?i:hd|fhd|qhd|uhd|led|lcd|ips|va|tn|oled|rgb|usb|hdmi|lte|nfc|gps|ram|rom|gb|tb|pc|tv|ai|4g|5g|full|ultra|new|in-stock|stock)|.*\d.*", w) for w in words):
                    continue
                cand = " ".join(words)
                self.brand, self.brand_hard = cand, not bool(re.search(r"prefer|ideally", low))
                break
        if not self.brand and not from_shopper:
            m = re.search(r"\b(?:a|an|in-stock|me a|me an)\s+([A-Za-z][\w\-\+\.']{1,30}(?:\s+[A-Z][\w\-\+]+)?)\s+(?=\d+p\b|\d+(?:\.\d+)?-inch|\d+(?:\.\d+)?\s*inch|\d+\s*[Hh][Zz]|\d+\s*gb\b|(?i:qhd|uhd|fhd|4k|2k|8k|full hd)\b)", t)
            if m and not any(re.fullmatch(r"(?i:hd|fhd|qhd|uhd|led|lcd|ips|va|tn|oled|rgb|usb|hdmi|lte|nfc|gps|ram|rom|gb|tb|pc|tv|ai|4g|5g|full|ultra|new|in-stock|stock|genuine|cheap|good|budget|gaming|smartphone|phone|tablet|monitor|laptop|headphone|smartwatch|one|test|the|with|and|for)|.*\d.*", w) for w in m.group(1).split()):
                self.brand, self.brand_hard = m.group(1), not bool(re.search(r"prefer|ideally", low))
        for c in COLORS:
            if re.search(r"\b" + c + r"\b", low) and re.search(r"\b(color|colour|prefer|eye on|the " + c + r")\b", low):
                self.color = c
        m = re.search(r"\b(?:strongly\s+|really\s+|specifically\s+)?(?:prefer|want|after|eyeing|set on|looking for)\s+(?:the\s+)?([^.]{3,60}?)\.(?=\s|$)", t, re.I) if RECOVERY_HINT_RE.search(low) else None
        if m and not self.model:
            words = m.group(1).strip().split()
            if len(words) >= 2:
                lw = [w.lower() for w in words]
                cut = None
                for k in range(min(3, len(words) - 1), 0, -1):
                    if " ".join(lw[:k]) in SHADES:
                        cut = k
                        break
                if cut is None:
                    cut = len(words) - 1
                self.color, self.model = " ".join(lw[:cut]), " ".join(words[cut:])
                if self.color == "sage green":
                    self.color = "green"
                if self.color not in SHADES:
                    SHADES.append(self.color)
        shades = sorted(set(SHADES) | set(COLORS), key=len, reverse=True)
        m = re.search(r"\b(?:the\s+)?(" + "|".join(re.escape(x) for x in shades) + r")\s+([A-Za-z0-9][\w\-]{1,}(?:\s+(?!if\b|and\b|or\b|but\b|in\b|at\b|specifically\b)[A-Za-z0-9][\w\-]*){0,3})(?=[.,;:!?]|\s|$)", t, re.I)
        if m and not self.model and re.search(r"prefer|eye on|exact", low):
            self.color, self.model = m.group(1).lower(), m.group(2).strip()
            if self.color == "sage green":
                self.color = "green"
        if re.search(r"\btell you\b[^.]{0,14}\bask\b|\bwhen (?:you )?ask\b|\bwill share\b|\bon request\b|\bif you ask\b", low):
            self.sealed = True
        generic_spec = bool(self.reqs) or bool(self._absorb_reqs(t, low, dry=True)) if not (self.phrase or self.model or self.candidate_keys or self.ranking) else False
        if not (self.wants_reason or self.spec_field() or self.model or self.phrase or self.candidate_keys or generic_spec):
            ranked = _ranked_axes(low, self.category) if RANK_MARKERS_RE.search(low) else []
            if ranked:
                self.pref_words = ranked + [a for a in self.pref_words if a not in ranked]
            else:
                axes = _axes_in_text(low, self.category)
                gate = re.search(r"\b(rather|prefer|for|matters?|priorit\w*|use|want|care about|counts?|beats)\b", low)
                if axes and gate:
                    lead = _lead_axis(low, self.category)
                    for a in ([lead] if lead in axes else []) + [a for a in axes if a != lead]:
                        if a not in self.pref_words:
                            self.pref_words.append(a)
                elif not axes and not from_shopper:
                    mm = PREF_CLAUSE_RE.search(low)
                    clause = (mm.group(1) or mm.group(2)) if mm else ""
                    phrase = _axis_phrase(clause, tail=False) if clause else ""
                    if phrase and 2 <= len(phrase) <= 40 and not re.search(r"\d{3,}|php|budget|category|stock|requirement|firm|brand|price|cheap", phrase) and "x:" + phrase not in self.pref_words:
                        self.pref_words.append("x:" + phrase)
            if not self.pref_words and _cat_key(self.category) in ("smartwatch", "tablet") and re.search(r"\b(rather|prefer|matters?|priorit|survive|lasts?|counts?)\b", low):
                self.pref_words = list(DEFAULT_AXES[_cat_key(self.category)])
        if self.candidate_keys and not self.ranking:
            self.ranking = True
        if not (self.phrase or self.model or self.candidate_keys or self.pref_words or self.ranking):
            self._absorb_reqs(t, low)

    def summary(self) -> str:
        parts = []
        if self.budget is not None:
            parts.append(f"budget<= {self.budget:.2f}")
        if self.category:
            parts.append(f"category={self.category}")
        if self.brand:
            parts.append(f"brand={self.brand}({'required' if self.brand_hard else 'preferred'})")
        if self.size_in is not None:
            parts.append(f"screen={self.size_in} in")
        if self.refresh_hz is not None:
            parts.append(f"refresh>={self.refresh_hz:g} Hz")
        if self.res_class:
            parts.append(f"resolution={self.res_class}")
        if self.ram_gb is not None:
            parts.append(f"ram={self.ram_gb:g} GB")
        for r in self.reqs:
            parts.append(f"req:{r['name'] or r['base']}{r['op']}{r['value']}")
        if self.model or self.color:
            parts.append(f"preferred={self.model} {self.color}".strip())
        if self.pref_words:
            parts.append("prefs=" + ",".join(self.pref_words))
        return "; ".join(parts)

class Cand:
    def __init__(self, pid: str) -> None:
        self.pid = pid
        self.title = ""
        self.brand = ""
        self.category_path: List[str] = []
        self.facts: Dict[str, str] = {}
        self.variants: List[Dict[str, Any]] = []
        self.summary_sku = ""
        self.summary_price: Optional[float] = None
        self.summary_stock: Optional[bool] = None
        self.viewed = False
        self.rank = 999
        self.listing_id = ""

    def text(self) -> str:
        return (self.title + " " + " ".join(f"{k}: {v}" for k, v in self.facts.items())).lower()

    def spec(self, field: str, options: str = "") -> Tuple[Optional[str], bool]:

        fn = VALUE_SETS[field]
        opt = options or ""
        opt_vals = fn(opt)
        if len(opt_vals) > 1:
            return None, True
        if opt_vals:
            return next(iter(opt_vals)), True
        if field == "ram":
            wzmianka = bool(PAIR_HINT_RE.search(opt) or RAM_TITLE_RE.search(opt))
        elif field == "screen_size":
            wzmianka = bool(INCH_RE.search(opt))
        elif field == "refresh_rate":
            wzmianka = bool(HZ_RE.search(opt))
        else:
            wzmianka = bool(WXH_RE.search(opt) or RES_WORD_RE.search(opt))
        if wzmianka:
            return None, True
        vals = set()
        conflict = False
        for k, val in self.facts.items():
            if k in ("description_device_type", "description_operating_system", "phone_type", "device_type", "device_operating_system", "model", "color", "condition"):
                continue
            if field == "resolution" and k not in ("display", "resolution", "screen"):
                continue
            if field == "refresh_rate" and k not in ("refresh_rate", "display", "size", "screen"):
                continue
            if field == "screen_size" and k not in ("size", "display", "screen"):
                continue
            if field == "ram" and k not in ("storage",):
                continue
            if field == "resolution" and k == "display":
                classed = [(s.strip(), fn(s)) for s in str(val).split(";") if fn(s)]
                mocne = [(s, v) for s, v in classed
                         if WXH_RE.search(s) or re.search(r"\b(?:4k|uhd|ultra ?hd|qhd|fhd|full ?hd|hd)\b", s.lower())]
                classed = mocne if mocne else []
                got = set()
                for _s, v in classed:
                    if len(v) > 1:
                        conflict = True
                    got |= v
            else:
                got = _gb_vals(str(val)) if field == "ram" else fn(str(val))
            if len(got) > 1:
                conflict = True
            vals |= got
        vals |= fn(self.title)
        if conflict or len(vals) > 1:
            return None, True
        if len(vals) == 1:
            return next(iter(vals)), True
        return None, False

    def req_check(self, req: Dict[str, Any], options: str = "") -> Optional[bool]:
        base = req["base"]
        vals = _pair_aware_values(req, options or "")
        if vals:
            return _req_matches(req, vals)
        want = {w[:5] for w in re.findall(r"[a-z]+", req.get("name") or "") if len(w) >= 3}
        named = [str(v) for k, v in self.facts.items() if want and (want & {w[:5] for w in re.findall(r"[a-z]+", k.lower().replace("_", " "))})]
        for text in (named, [str(v) for v in self.facts.values()], [self.title]):
            vals = []
            for chunk in text:
                vals += _pair_aware_values(req, chunk)
            if vals:
                return _req_matches(req, vals)
        return None

    def axis_value(self, phrase: str, options: str = "", price: float = 0.0) -> Optional[float]:
        p = phrase.lower()
        if "price" in p or "cost" in p or "cheap" in p:
            return price
        if "refresh" in p or re.search(r"\bhz\b", p):
            return self.refresh_hz(options)
        if re.search(r"screen|display size|\bsize\b|inch|diagonal", p) and "resolution" not in p:
            return self.size_in(options)
        if re.search(r"\bram\b|memory", p):
            v, _ = self.spec("ram", options)
            return float(v[:-2]) if v else None
        if "resolution" in p or "pixel" in p:
            return RES_RANK.get(self.res_class(options), None)
        want = {w[:5] for w in re.findall(r"[a-z]+", p) if len(w) >= 3 and w not in ("with", "than", "rate", "life")}
        texts = [str(v) for k, v in self.facts.items() if want & {w[:5] for w in re.findall(r"[a-z]+", k.lower().replace("_", " "))}]
        if not texts:
            low = (options or "") + " " + self.title.lower()
            for w in want:
                m = re.search(w + r"[a-z]*\W{0,3}(?:of\s+)?(\d+(?:\.\d+)?)", low) or re.search(r"(\d+(?:\.\d+)?)\s*[a-z%\"]{0,4}\s+" + w, low)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        continue
            return None
        nums: List[float] = []
        for t in texts:
            for m in UNIT_VALUE_RE.finditer(t):
                try:
                    nums.append(float(m.group(1).replace(",", ".")))
                except ValueError:
                    continue
            if not nums:
                nums += [float(x) for x in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", t)]
        return max(nums) if nums else None

    def req_fact_value(self, req: Dict[str, Any], options: str = "") -> str:
        base = req["base"]
        for text in ([options or ""], [str(v) for v in self.facts.values()], [self.title]):
            for chunk in text:
                for m in UNIT_VALUE_RE.finditer(chunk):
                    b, k = _unit_base(m.group(2))
                    if b == base and _req_matches(req, [float(m.group(1).replace(",", ".")) * k]):
                        return m.group(0).strip()
        return ""

    def fact_spec(self, field: str) -> Optional[str]:
        return _norm_spec(field, self.facts.get(FACT_LABEL[field], ""))

    def size_in(self, options: str = "") -> Optional[float]:
        v, _ = self.spec("screen_size", options)
        return float(v[:-2]) if v else None

    def refresh_hz(self, options: str = "") -> Optional[float]:
        v, _ = self.spec("refresh_rate", options)
        return float(v[:-2]) if v else None

    def res_class(self, options: str = "") -> str:
        v, _ = self.spec("resolution", options)
        return v or ""

    def is_accessory(self) -> bool:
        return _accessory_title(self.title, ACCESSORY_RE) or any(w in self.title.lower() for w in BAD_ITEM_WORDS)

    def is_smartphone(self, options: str = "") -> bool:

        if re.search(r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b", options or "", re.I):
            return False

        def ident(text: str, title: bool = False) -> str:
            text = re.sub(r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b", "", text, flags=re.I)
            if title:
                text = re.split(r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))", text, maxsplit=1, flags=re.I)[0]
            return text

        title = ident(self.title, True)
        type_text = " ".join(ident(self.facts.get(k, "")) for k in ("phone_type", "device_type", "description_device_type"))
        text = f"{title} {type_text}"
        if any(p.search(text) for p in CONFLICT_RES):
            return False
        if any("[truncated]" in self.facts.get(k, "") for k in ("phone_type", "device_type", "device_operating_system")):
            return False
        if SMART_RE.search(text):
            return True
        os_text = " ".join(ident(self.facts.get(k, "")) for k in ("device_operating_system", "description_operating_system") if not re.search(r"\bcompatib", self.facts.get(k, ""), re.I))
        if OS_RE.search(os_text):
            return True
        return bool(OS_RE.search(title)) and not re.search(r"\bcompatib", self.title, re.I)

    def valid_recovery_title(self, category: str, options: str = "") -> bool:
        low = self.title.lower().strip()
        if not low or low.startswith(("customer ratings", "customer reviews")) or ACCESSORY_RE.search(low):
            return False
        if category.lower().startswith("smartphone"):
            return FEATURE_PHONE_RE.search(low) is None
        if any(f in low for f in INVALID_TITLE_FRAGMENTS):
            return False
        return OVER_ON_EAR_RE.search(self.title + " " + (options or "")) is not None

    def model_label(self) -> Optional[str]:
        raw = self.facts.get("model", "")
        if raw:
            if ";" in raw or "/" in raw:
                return None
            words = raw.lower().split()
            prefix = []
            while len(words) > 1 and words[0].isalpha() and any(ch.isdigit() for ch in " ".join(words[1:])):
                prefix.append(words.pop(0))
            ident = (self.title + " " + self.brand).lower()
            if prefix and " ".join(prefix) not in ident:
                return None
            if raw.isalpha() and 3 <= len(raw) <= 24:
                return raw.upper() if raw.lower() in ident else None
            return _model_code(" ".join(words))
        labels = set()
        for m in TITLE_MODEL_RE.finditer(self.title):
            tok = m.group()
            if NON_MODEL_RE.search(tok):
                continue
            suf = re.match(r"(?:\s+(?:pro|max|ultra|plus)\b)+", self.title[m.end():], re.I)
            code = _model_code(tok + (suf.group() if suf else ""))
            if code:
                labels.add(code)
        return next(iter(labels)) if len(labels) == 1 else None

    def color_label(self, options: str) -> Optional[str]:
        return _color_label(options, self.facts.get("color", ""), self.title)

    def axis_hits(self, cons: "Constraints") -> Dict[str, float]:
        key = _cat_key(cons.category)
        facts_text = " ".join(f"{k.replace('_', ' ')} {v}" for k, v in self.facts.items()).lower()
        title = self.title.lower()
        text = facts_text + " " + title
        watch_text = facts_text if (self.facts or self.viewed) else title
        hits: Dict[str, float] = {}
        for a in _pref_axes(cons):
            if a == "water":
                hits[a] = 1.0 if re.search(WATER_RE, text, re.I) else 0.0
            elif a == "sport":
                hits[a] = 1.0 if re.search(SPORT_RE, text, re.I) else 0.0
            elif a == "battery":
                hits[a] = _long_battery(watch_text) if key == "smartwatch" else _short_battery(text)
            elif a == "wireless":
                hits[a] = 1.0 if re.search(WIRELESS_RE, facts_text + " " + " ".join(self.category_path).lower() + " " + title, re.I) else 0.0
            elif a == "kid":
                hits[a] = 1.0 if re.search(KID_RE, text) else 0.0
            elif a == "size":
                hits[a] = 1.0 if re.search(r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")", text) else 0.0
            elif a == "wifi":
                hits[a] = 1.0 if re.search(r"\b(wi-?fi|wifi)\b", text) else 0.0
            elif a == "navigation":
                hits[a] = 1.0 if re.search(NAV_RE, watch_text) else 0.0
            elif a == "rugged":
                hits[a] = 1.0 if re.search(RUGGED_RE, watch_text) else (0.5 if re.search(r"\b(water|swim|steel|aluminum)\b", watch_text) else 0.0)
            elif a.startswith("x:"):
                hits[a] = float(getattr(self, "judged", {}).get(a, 0.0))
            else:
                hits[a] = 0.0
        return hits

    def kid_mismatch(self, cons: "Constraints") -> bool:
        if _cat_key(cons.category) != "smartwatch" or "kid" in _pref_axes(cons):
            return False
        text = " ".join(f"{k.replace('_', ' ')} {v}" for k, v in self.facts.items()).lower() or self.title.lower()
        return bool(re.search(KIDWATCH_RE, text))

    def pref_value(self, cons: "Constraints") -> float:
        axes = _pref_axes(cons)
        hits = self.axis_hits(cons)
        raw = sum(w * hits.get(a, 0.0) for a, w in zip(axes, _axis_weights(len(axes))))
        if self.kid_mismatch(cons):
            raw -= 0.2
        return round(max(0.0, min(1.0, raw)), 3)

    def pref_ok(self, cons: "Constraints") -> bool:
        axes = _pref_axes(cons)
        if not axes:
            return False
        hits = self.axis_hits(cons)
        sat = sum(1 for a in axes if hits.get(a, 0.0) >= 0.75)
        need = max(1, -(-3 * len(axes) // 4))
        return hits.get(axes[0], 0.0) >= 0.75 and sat >= need and not self.kid_mismatch(cons)

    def phrase_hits(self, phrase: str) -> float:
        toks = [w for w in re.findall(r"[a-z0-9]+", phrase.lower())]
        if not toks:
            return 1.0
        title_toks = set(re.findall(r"[a-z0-9]+", self.title.lower()))
        return sum(1.0 for w in toks if w in title_toks) / len(toks)

    def title_hint(self, field: str, want: str) -> bool:
        return bool(field and want and want in VALUE_SETS[field](self.title))

    def color_of(self, options: str) -> str:
        return self.color_label(options) or ""

class State:
    def __init__(self) -> None:
        self.cands: Dict[str, Cand] = {}
        self.order: List[str] = []
        self.searches = 0
        self.cart: List[Dict[str, Any]] = []
        self.asked = False
        self.retrieved = False
        self.answered = False
        self.pending_question = ""
        self.events: List[Dict[str, Any]] = []
        self.ordered = False
        self.done = False
        self.committed: Optional[Tuple[str, str]] = None
        self.rejected: set = set()
        self.log: List[str] = []
        self.used_queries: set = set()
        self.filtered = False
        self.search_price_ok = True
        self.shopper_texts: List[str] = []
        self.model_queries: List[str] = []
        self.judge_calls = 0
        self.env_fail = 0
        self.env_last = 0.0
        self.bands = 0
        self.enum = 0
        self.acts = 0
        self.env_lat = 1.0

    def cand(self, pid: str) -> Cand:
        pid = str(pid)
        if pid not in self.cands:
            self.cands[pid] = Cand(pid)
            self.order.append(pid)
        return self.cands[pid]

def _ingest(state: State, name: str, obs: Dict[str, Any]) -> None:
    if not isinstance(obs, dict):
        return
    if name in ("search", "filter"):
        for r in obs.get("results") or []:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            c = state.cand(r["product_id"])
            c.title = c.title or str(r.get("title") or "")
            c.brand = c.brand or str(r.get("brand") or "")
            c.summary_sku = str(r.get("sku") or c.summary_sku)
            c.summary_price = _f(r.get("price"))
            c.summary_stock = r.get("in_stock")
            c.rank = min(c.rank, len(state.order))
    elif name in ("view", "compare"):
        if name == "view":
            items = [obs]
        else:
            items = []
            for key in ("compared", "products", "items", "results"):
                if isinstance(obs.get(key), list):
                    items = obs[key]
                    break
            if not items:
                items = [v for v in obs.values() if isinstance(v, dict) and v.get("product_id")]
                for v in obs.values():
                    if isinstance(v, list):
                        items.extend(x for x in v if isinstance(x, dict) and x.get("product_id"))
        for it in items:
            if not isinstance(it, dict) or not it.get("product_id"):
                continue
            c = state.cand(it["product_id"])
            c.title = str(it.get("title") or c.title)
            c.brand = str(it.get("brand") or c.brand or "")
            c.category_path = list(it.get("category_path") or c.category_path)
            if it.get("listing_id"):
                c.listing_id = str(it["listing_id"])
            if isinstance(it.get("facts"), dict):
                c.facts = {str(k): str(v) for k, v in it["facts"].items()}
            if isinstance(it.get("variants"), list):
                c.variants = [v for v in it["variants"] if isinstance(v, dict)]
            c.viewed = True
    elif name == "inspect_stock":
        c = state.cand(obs.get("product_id", "")) if obs.get("product_id") else None
        if c is not None:
            for v in c.variants:
                if v.get("sku") == obs.get("sku"):
                    v["in_stock"] = obs.get("in_stock")
                    v["price"] = _f(obs.get("price"))
                    break
            else:
                c.variants.append({"sku": obs.get("sku"), "price": _f(obs.get("price")), "in_stock": obs.get("in_stock"), "options": ""})
    if isinstance(obs.get("cart"), list):
        state.cart = [x for x in obs["cart"] if isinstance(x, dict)]
    if isinstance(obs.get("market_event"), dict):
        state.events.append(obs["market_event"])
    if name == "place_test_order" and isinstance(obs.get("order"), dict):
        state.ordered = True

def _variant_ok(cons: Constraints, c: Cand, v: Dict[str, Any]) -> bool:
    price = _f(v.get("price"))
    if v.get("in_stock") is not True or price is None:
        return False
    if cons.budget is not None and price > cons.budget + 1e-6:
        return False
    return True

def _spec_ok(cons: Constraints, c: Cand, options: str) -> Optional[bool]:

    known = _known_spec_ok(cons, c, options)
    if known is False:
        return False
    unknown = known is None
    for req in cons.reqs:
        r = c.req_check(req, options)
        if r is False:
            return False
        if r is None:
            unknown = True
    return None if unknown else True

def _known_spec_ok(cons: Constraints, c: Cand, options: str) -> Optional[bool]:
    field = cons.spec_field()
    want = cons.spec_value()
    if not field or not want:
        return True
    got, known = c.spec(field, options)
    if got is None:
        return False if known else None
    if field == "resolution" and got == "qhd" and cons.category.lower().startswith("smartphone") and not re.search(r"1440|2560|\b2k\b", (c.facts.get("display", "") + " " + c.facts.get("resolution", "") + " " + c.title + " " + (options or "")).lower()):
        return None
    if field == "refresh_rate" and cons.ranking:
        return True
    ok = got == want
    if ok and cons.wants_reason and c.fact_spec(field) != want:

        return None
    return ok

def _brand_ok(cons: Constraints, c: Cand) -> bool:
    if not cons.brand:
        return True
    want = set(re.findall(r"[a-z0-9]+", cons.brand.lower()))
    if c.brand:
        have = set(re.findall(r"[a-z0-9]+", c.brand.lower()))
        return bool(want) and want <= have
    if cons.brand_hard:
        return False
    title_toks = set(re.findall(r"[a-z0-9]+", c.title.lower()))
    return bool(want) and want <= title_toks

CAT_ALIASES = {"mobile phones": "smartphone", "cell phones": "smartphone", "cellphones": "smartphone", "cellphone": "smartphone", "phones": "smartphone",
               "smartphones": "smartphone", "unlocked phones": "smartphone", "smart phones": "smartphone", "smartwatches": "smartwatch",
               "smart watches": "smartwatch", "headphones headsets": "headphones", "computer monitors": "monitors", "monitor": "monitors"}

def _cat_canon(label: str) -> set:
    low = " ".join(re.findall(r"[a-z0-9]+", str(label).lower()))
    return {low, CAT_ALIASES.get(low, low)} if low else set()

def _category_ok(cons: Constraints, c: Cand) -> bool:
    if not cons.category or getattr(cons, "category_soft", False):
        return True
    want = " ".join(re.findall(r"[a-z0-9]+", cons.category.lower()))
    wanted = _cat_canon(cons.category)
    for alt in cons.category_alts:
        wanted |= _cat_canon(alt)
    labels = set()
    for p in c.category_path:
        labels |= _cat_canon(p)
    if wanted & labels or {w.rstrip("s") for w in wanted} & {l.rstrip("s") for l in labels}:
        return True
    if want in ("gaming monitors", "gaming monitor"):
        toks = set(re.findall(r"[a-z0-9]+", c.title.lower()))
        return bool(labels & {"monitors", "computer monitors"}) and ("gaming" in toks or any(float(m.group(1)) >= 120 for m in HZ_RE.finditer(c.title)))
    if not c.category_path:
        return want.rstrip("s") in c.text()
    if not cons.category_authoritative and c.category_path:
        leaf = " ".join(re.findall(r"[a-z0-9]+", str(c.category_path[-1]).lower()))
        need = {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", want) if len(w) > 2}
        have = {w.rstrip("s") for w in leaf.split()}
        if need and need <= have and not re.search(r"\b(accessor|part|case|cover|charger|cable|adapter|protector|replacement|spare|mount|stand)", leaf):
            return True
    return False

def _score(cons: Constraints, c: Cand, v: Dict[str, Any]) -> Tuple:

    opts = str(v.get("options") or "")
    price = _f(v.get("price")) or 0.0
    spec = _spec_ok(cons, c, opts)
    spec_pts = 2 if spec is True else (1 if spec is None else 0)
    if spec is None and c.title_hint(cons.spec_field(), cons.spec_value()):
        spec_pts = 1.5
    brand_pts = 1 if _brand_ok(cons, c) else 0
    pref = c.pref_value(cons) if cons.pref_words else 0.0
    model_pts = 0
    if cons.model or cons.color:
        lvl = _relax_level(cons, c, opts)
        model_pts = 4 - lvl if lvl is not None else 0
        if not c.viewed and cons.model and re.search(r"(?<![a-z0-9])" + re.escape(cons.model.lower()) + r"(?![a-z0-9])", c.title.lower()):
            model_pts = max(model_pts, 3 if cons.color and cons.color in c.title.lower() else 2)
    if cons.ranking:
        axes = cons.rank_axes or [("desc", "refresh rate"), ("desc", "screen"), ("asc", "price")]
        vals = [c.axis_value(ph, opts, price) for _d, ph in axes]
        key = tuple((v if d == "desc" else -v) if v is not None else float("-inf") for (d, _p), v in zip(axes, vals))
        complete = sum(1 for (_d, ph), v in zip(axes, vals) if v is not None and "price" not in ph)
        return (complete, *key, -price)
    genuine = 1 if ("smartphone" not in cons.category.lower() or c.is_smartphone(opts)) else 0
    rich = min(3, len(c.facts)) + (1 if c.size_in(opts) is not None else 0)
    phr = round(c.phrase_hits(cons.phrase), 2) if cons.phrase else 0.0
    if cons.pref_words:
        return (genuine, spec_pts, brand_pts if cons.brand_hard else 0, round(pref, 3), rich, brand_pts, price)
    if cons.phrase:
        return (genuine, phr, spec_pts, brand_pts, rich, -price)
    if not c.viewed:
        return (genuine, spec_pts, brand_pts if cons.brand_hard else 0, model_pts, brand_pts, 0 if c.is_accessory() else 1, 0, 0)
    return (genuine, spec_pts, brand_pts if cons.brand_hard else 0, model_pts, brand_pts, rich, round(pref, 2), -price)

def _relax_level(cons: Constraints, c: Cand, options: str) -> Optional[int]:

    model = c.model_label()
    color = c.color_label(options)
    if not model or not color:
        return None
    want_model = _model_code(cons.model) if cons.model else None
    same_model = bool(cons.model) and (model.lower() == cons.model.lower() or (want_model is not None and _model_code(model) == want_model))
    same_color = bool(cons.color) and color.lower() == cons.color.lower()
    if same_model and same_color:
        return 0
    if same_model:
        return 1
    if same_color:
        return 2
    return 3

DEVICE_CATEGORIES = ("smartphone", "headphone", "tablet", "laptop", "monitor", "smartwatch", "speaker", "camera", "tv", "router", "keyboard", "mice", "mouse")

def _llm_actions(goal: str, rules: str, tools: Dict[str, Any], state: State, cons: Constraints) -> List[Tuple[str, Dict[str, Any]]]:

    spec: List[Dict[str, Any]] = []
    for name, t in list(tools.items())[:12]:
        fn = t.get("function") or {}
        props = ((fn.get("parameters") or {}).get("properties") or {})
        spec.append({"name": name, "args": list(props.keys())[:8]})
    seen = []
    for c in list(state.cands.values())[:8]:
        seen.append({"product_id": c.pid, "title": c.title[:80], "price": c.summary_price, "in_stock": c.summary_stock})
    d = _llm_json("You drive a shopping tool loop. Answer with JSON only.",
                  "Shopper request:\n" + goal[:900] + "\n\nRules (abridged):\n" + (rules or "")[:700] +
                  "\n\nAvailable tools and their argument names:\n" + json.dumps(spec, ensure_ascii=False) +
                  "\n\nAlready seen (may be empty):\n" + json.dumps(seen, ensure_ascii=False)[:900] +
                  "\n\nParsed so far: " + (cons.summary() or "nothing") +
                  "\n\nNothing valid was found yet. Propose the next one or two tool calls that would surface a matching, "
                  "in-stock product within budget. Prefer search with a different wording, or filter with the catalog category. "
                  "Reply with JSON only: {\"actions\": [{\"name\": \"search\", \"args\": {\"query\": \"...\"}}]}")
    out: List[Tuple[str, Dict[str, Any]]] = []
    for item in (d.get("actions") if isinstance(d, dict) else None) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        args = item.get("args")
        if name not in tools or not isinstance(args, dict):
            continue
        clean: Dict[str, Any] = {}
        for k, v in list(args.items())[:6]:
            k = str(k)
            if k in ("query", "category", "brand") and isinstance(v, str) and 0 < len(v) <= 60:
                clean[k] = v
            elif k in ("k", "limit") and isinstance(v, (int, float)) and 1 <= float(v) <= 10:
                clean[k] = int(v)
            elif k in ("in_stock",) and isinstance(v, bool):
                clean[k] = v
            elif k in ("max_price",) and isinstance(v, (int, float)) and float(v) > 0:
                clean[k] = float(v)
            elif k in ("product_id", "sku") and isinstance(v, (str, int)) and str(v) in state.cands:
                clean[k] = str(v)
        if name in ("search", "filter"):
            clean.setdefault("k", 10)
            clean.setdefault("in_stock", True)
            if cons.budget is not None and state.search_price_ok:
                clean.setdefault("max_price", cons.budget)
            if name == "search" and not clean.get("query"):
                continue
            if name == "filter" and not (clean.get("category") or clean.get("brand")):
                continue
        elif name in ("view", "inspect_stock", "add_to_cart", "remove_from_cart"):
            if not clean.get("product_id"):
                continue
        elif name == "compare":
            ids = [str(x) for x in (args.get("product_ids") or []) if str(x) in state.cands][:5]
            if len(ids) < 2:
                continue
            clean = {"product_ids": ids}
        else:
            continue
        q = json.dumps([name, clean], sort_keys=True)
        if q in state.used_queries:
            continue
        state.used_queries.add(q)
        out.append((name, clean))
        if len(out) >= 2:
            break
    return out

def _judge_unknown_axes(cons: Constraints, state: State) -> None:
    axes = [a for a in _pref_axes(cons) if a.startswith("x:")]
    if not axes or state.judge_calls >= 4:
        return
    pending = [c for c in state.cands.values() if c.viewed and not hasattr(c, "judged")]
    if not pending:
        return
    want = set()
    for a in axes:
        want |= {w for w in re.findall(r"[a-z]+", a[2:]) if len(w) > 3}

    def prior(c: Cand) -> Tuple[int, float]:
        text = (c.title + " " + " ".join(str(v) for v in c.facts.values())).lower()
        overlap = sum(1 for w in want if w in text)
        known = sum(v for k, v in c.axis_hits(cons).items() if not k.startswith("x:"))
        return (overlap, known)

    todo = sorted(pending, key=prior, reverse=True)[:8]
    state.judge_calls += 1
    rows = [{"product_id": c.pid, "title": c.title[:100], "facts": {k: str(v)[:60] for k, v in list(c.facts.items())[:10]}} for c in todo]
    d = _llm_json("You judge how well listed products fit a shopper's stated preferences using ONLY the listing title and the visible facts given. Answer with JSON only.",
                  "Preference axes in priority order (meaning in the shopper's words):\n" + "\n".join(f"- {a[2:]}" for a in axes) +
                  "\n\nCandidates:\n" + json.dumps(rows, ensure_ascii=False) +
                  "\n\nScore every candidate on every axis using exactly one of the values 1 (the title or facts clearly state it), 0.5 (partly or implicitly), "
                  "0 (nothing in the title or facts supports it). Use the axis wording as the key. "
                  "Reply with JSON only: {\"scores\": {\"<product_id>\": {\"<axis wording>\": 0|0.5|1}}}")
    scores = d.get("scores") if isinstance(d, dict) else None

    def norm_key(s: Any) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(s).lower().replace("x:", "", 1)))

    for c in todo:
        got = None
        if isinstance(scores, dict):
            for k, v in scores.items():
                if re.sub(r"\D", "", str(k)) == re.sub(r"\D", "", c.pid) and isinstance(v, dict):
                    got = {norm_key(ak): av for ak, av in v.items()}
                    break
        judged: Dict[str, float] = {}
        for a in axes:
            v = got.get(norm_key(a)) if isinstance(got, dict) else None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            fv = max(0.0, min(1.0, fv))
            judged[a] = 1.0 if fv >= 0.75 else (0.5 if fv >= 0.25 else 0.0)
        c.judged = judged

def _candidates(cons: Constraints, state: State, relaxed: int = 0) -> List[Tuple[Cand, Dict[str, Any], Tuple]]:

    out = []
    if cons.pref_words:
        _judge_unknown_axes(cons, state)
    keys = set(cons.candidate_keys)
    cat = cons.category.lower()
    is_device = any(cat.startswith(d) for d in DEVICE_CATEGORIES) or cons.device_hint is True
    for pid in state.order:
        c = state.cands[pid]
        variants = c.variants or ([{"sku": c.summary_sku, "price": c.summary_price, "in_stock": c.summary_stock, "options": ""}] if c.summary_sku else [])
        for v in variants:
            sku = str(v.get("sku"))
            if (c.pid, sku) in state.rejected:
                continue
            if keys and (c.pid, sku) not in keys:
                continue
            if not _variant_ok(cons, c, v):
                continue
            if c.viewed and c.category_path and not keys and not _category_ok(cons, c):
                continue
            if cons.brand_hard and not _brand_ok(cons, c):
                continue
            if is_device and not keys and c.is_accessory():
                continue
            spec = _spec_ok(cons, c, str(v.get("options") or "")) if c.viewed else None
            if spec is False and relaxed < 2:
                continue
            if spec is None and c.viewed and cons.wants_reason and cons.has_spec():
                continue
            if cons.wants_reason and c.viewed and not c.brand and relaxed < 2:
                continue
            if spec is None and cons.has_spec() and relaxed == 0 and c.viewed:
                continue
            if cat.startswith("smartphone") and c.viewed and not c.is_smartphone(str(v.get("options") or "")):
                continue
            if relaxed >= 2:
                out.append((c, v, _score(cons, c, v)))
                continue
            if (cons.model or cons.color) and c.viewed and _relax_level(cons, c, str(v.get("options") or "")) is None and relaxed == 0:
                continue
            if cons.model and cons.color and relaxed == 0 and state.committed is None and c.viewed and _relax_level(cons, c, str(v.get("options") or "")) != 0:
                continue
            if (cons.model or cons.color) and c.viewed and not c.valid_recovery_title(cons.category, str(v.get("options") or "")):
                continue
            if cons.pref_words and c.viewed and relaxed == 0 and not c.pref_ok(cons):
                continue
            if cons.pref_words and c.viewed and relaxed == 0 and _accessory_title(c.title, PREF_ACCESSORY_RE):
                continue
            if cons.phrase and relaxed == 0 and c.phrase_hits(cons.phrase) < 0.5:
                continue
            out.append((c, v, _score(cons, c, v)))
    out.sort(key=lambda x: x[2], reverse=True)
    return out

def _search_args(cons: "Constraints", state: "State", q: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {"query": q, "k": 10, "in_stock": True}
    if cons.budget is not None and state.search_price_ok:
        args["max_price"] = cons.budget
    return args

def _llm_json(system: str, prompt: str) -> Dict[str, Any]:
    if _LLM_SPENT[0] >= _LLM_BUDGET_S:
        return {}
    _t0 = time.monotonic()
    try:
        r = _LLM.post("/inference/chat/completions", json_data={"model": _model(), "temperature": 0,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
        content = (r or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}
    finally:
        _LLM_SPENT[0] += time.monotonic() - _t0

def _llm_model_names(cons: Constraints) -> List[str]:
    if not cons.model:
        return []
    cat = (cons.category or "smartphone").lower()
    dev = cat if cat.startswith("headphone") else cat.rstrip("s")
    try:
        return _model_name_queries(cons, dev)
    except Exception:
        return []

def _model_name_queries(cons: Constraints, dev: str) -> List[str]:
    d: Dict[str, Any] = {}
    for _attempt in range(2):
        d = _llm_json("You know consumer electronics catalogs and model numbering. Answer with JSON only.",
                  f"Category: {dev}. A shopper refers to the model only as \"{cons.model}\". This label is a model name or catalog model number "
                  "written with every space, hyphen, plus sign and slash removed and letters uppercased. Return (1) up to three likely original "
                  "spellings with the separators restored (for example a hyphen between a letter prefix and digits, spaces between words, a plus "
                  "between memory sizes) and (2) up to three full product names (brand, product line and this model). "
                  "Reply with JSON only: {\"spellings\": [\"...\"], \"names\": [\"...\"]}")
        if isinstance(d, dict) and (d.get("spellings") or d.get("names")):
            break
        time.sleep(1.0)

    def clean(items: Any, limit: int) -> List[str]:
        res: List[str] = []
        for x in (items if isinstance(items, list) else []):
            if not isinstance(x, str):
                continue
            q = re.sub(r"\s+", " ", x).strip()[:limit]
            if q and len(q.split()) <= 7 and not re.search(r"[\[\]{}\"<>\n]", q):
                res.append(q)
        return res[:3]

    if not isinstance(d, dict):
        return []
    spellings = clean(d.get("spellings"), 40)
    names = clean(d.get("names"), 60)
    out: List[str] = []
    for s in spellings:
        out.append(f"{s} {cons.color} {dev}".strip())
    for n in names:
        out.append(f"{n} {cons.color}".strip())
    out.extend(spellings)
    out.extend(names)
    seen: set = set()
    res = []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            res.append(q)
    return res[:12]

def _recovery_queries(cons: Constraints, state: State) -> List[str]:

    out: List[str] = []
    want_code = _model_code(cons.model) if cons.model else None
    for c in state.cands.values():
        label = c.model_label()
        if not label:
            continue
        if want_code and _model_code(label) == want_code:
            words = [w for w in re.findall(r"[A-Za-z0-9][\w\-\+\.]*", c.title) if w.lower() not in SHADES][:7]
            if words:
                out.append(" ".join(words))
                out.append(" ".join(words[:4]) + " " + cons.color)
    if "model-names-asked" not in state.log:
        state.log.append("model-names-asked")
        state.model_queries = _llm_model_names(cons)
    out.extend(state.model_queries)
    seen: set = set(); res = []
    for q in out:
        if q.lower() not in seen and q.lower() not in state.used_queries:
            seen.add(q.lower()); res.append(q)
    return res

def _llm_models(cons: Constraints, goal: str) -> List[str]:
    if not (cons.brand or cons.spec_field()) or not cons.category:
        return []
    d = _llm_json("You know consumer electronics catalogs. Answer with JSON only.",
                  "Shopper request:\n" + goal.split("\n\nTask rules:")[0] + "\n\nFirm requirements: " + cons.summary() +
                  "\n\nName up to three specific product models (brand and model name, no other words) that belong to this category, satisfy the firm spec "
                  "and are typically sold within the budget in the Philippines. Reply with JSON only: {\"models\": [\"...\", \"...\"]}")
    out = []
    for q in d.get("models") or []:
        q = re.sub(r"\s+", " ", str(q)).strip()[:60]
        if q and len(q.split()) <= 6:
            out.append(q)
    return out[:3]

def _llm_queries(cons: Constraints, goal: str) -> List[str]:
    prompt = ("Shopper request:\n" + goal + "\n\nKnown firm requirements: " + (cons.summary() or "none") +
              "\n\nWrite two short product-search queries (2-5 words each) for an e-commerce catalog search that matches product titles and "
              "attribute values. Include the product type and the decisive spec, brand or model when known. Reply with JSON only: {\"queries\": [\"...\", \"...\"]}")
    try:
        r = _LLM.post("/inference/chat/completions", json_data={"model": _model(), "temperature": 0,
                      "messages": [{"role": "system", "content": "You write catalog search queries. Answer with JSON only."},
                                   {"role": "user", "content": prompt}]})
        content = (r or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        m = re.search(r"\{.*\}", content, re.S)
        d = json.loads(m.group(0)) if m else {}
        out = []
        for q in d.get("queries") or []:
            q = re.sub(r"\s+", " ", str(q)).strip()[:80]
            if q and len(q.split()) <= 7:
                out.append(q)
        return out[:2]
    except Exception:
        return []

def _llm_parse(text: str) -> Dict[str, Any]:
    d = _llm_json("You extract shopping constraints from a shopper request. Answer with JSON only.",
                  "Request:\n" + text[:1800] + "\n\nReturn JSON with keys: category (catalog category noun in plural English such as Smartphones, Monitors, Tablets, "
                  "Laptops, Headphones, Smartwatches; null if unclear), brand (string or null), brand_required (true if the brand is a firm requirement, "
                  "false if merely preferred), budget (number, or null), spec_field (one of screen_size, resolution, refresh_rate, ram, or null), "
                  "spec_value (for example \"6.5in\", \"qhd\", \"fhd\", \"4k\", \"hd\", \"144hz\", \"8gb\"; null if none), "
                  "requirements (list of any OTHER firm numeric specifications as objects {name, value, unit}, for example "
                  "{\"name\": \"battery\", \"value\": 5000, \"unit\": \"mAh\"}; empty list if none), "
                  "is_device (true if the requested product is a consumer electronics device such as a phone, computer, audio or camera gear; false otherwise).")
    return d if isinstance(d, dict) else {}

def _apply_parse(cons: Constraints, d: Dict[str, Any], only_missing: bool = True, pool: Optional[Dict[str, "Cand"]] = None) -> List[str]:
    changed: List[str] = []
    if not d:
        return changed

    def s(v: Any) -> str:
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if not isinstance(v, (str, int, float)) or isinstance(v, bool):
            return ""
        txt = str(v).strip()
        if txt.lower() in ("null", "none", "n/a", "unknown") or len(txt) > 60 or re.search(r"[\[\]{}\"<>]", txt):
            return ""
        return txt

    def truthy(v: Any) -> bool:
        return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "yes", "1", "required"))

    cat = s(d.get("category"))
    if cat and not cons.category and re.fullmatch(r"[A-Za-z][A-Za-z0-9 &'\-/]{1,40}", cat):
        label = CATEGORY_LABELS.get(cat.lower()) or CATEGORY_LABELS.get(cat.lower().rstrip("s"))
        cons.category = label or (cat[:1].upper() + cat[1:])
        cons.category_soft = label is None
        changed.append("category")
    b = s(d.get("brand"))
    if b and re.fullmatch(r"[A-Za-z0-9][\w &'\-\+\.]{0,39}", b) and not re.search(r"\d{3,}", b) and (not cons.brand or (not only_missing and b.lower() != cons.brand.lower())):
        want = set(re.findall(r"[a-z0-9]+", b.lower()))
        supported = pool is None or any(want and want <= set(re.findall(r"[a-z0-9]+", (c.brand or c.title).lower())) for c in pool.values())
        if supported:
            cons.brand, cons.brand_hard = b, False
            cons.llm_fields.add("brand")
            changed.append("brand")
    if isinstance(d.get("is_device"), bool) and cons.device_hint is None:
        cons.device_hint = d.get("is_device")
    elif isinstance(d.get("is_device"), str) and cons.device_hint is None and d.get("is_device").strip().lower() in ("true", "false"):
        cons.device_hint = d.get("is_device").strip().lower() == "true"
    bud = _f(d.get("budget"))
    if bud is not None and bud > 0 and cons.budget is None:
        cons.budget = bud
        changed.append("budget")
    fld, val = s(d.get("spec_field")).lower(), s(d.get("spec_value")).lower()
    if fld and val and not cons.spec_field():
        num = re.search(r"(\d+(?:\.\d+)?)", val)
        if fld == "screen_size" and num and 3.0 <= float(num.group(1)) <= 100.0:
            cons.size_in = float(num.group(1))
            changed.append("size"); cons.llm_fields.add("size")
        elif fld == "refresh_rate" and num and 30 <= float(num.group(1)) <= 500:
            cons.refresh_hz = float(num.group(1))
            changed.append("hz"); cons.llm_fields.add("hz")
        elif fld == "resolution":
            rc = _only(_res_vals(val)) or (val if val in ("4k", "qhd", "fhd", "hd") else "")
            if rc:
                cons.res_class = rc
                cons.res_phrase = cons.res_phrase or val
                changed.append("res"); cons.llm_fields.add("res")
        elif fld == "ram" and num and 1 <= float(num.group(1)) <= 256:
            cons.ram_gb = float(num.group(1))
            changed.append("ram"); cons.llm_fields.add("ram")
    reqs = d.get("requirements")
    if isinstance(reqs, list) and not cons.has_spec():
        for item in reqs[:3]:
            if not isinstance(item, dict):
                continue
            name, unit = s(item.get("name")), s(item.get("unit"))
            val = _f(item.get("value"))
            if val is None or val <= 0 or val > 1e7 or not unit or not re.fullmatch(r"[A-Za-z%\"]{1,10}", unit) or not re.fullmatch(r"[A-Za-z][A-Za-z \-]{0,29}", name or "x"):
                continue
            req = _make_req(val, unit, name, f"{val:g} {unit} {name}".strip())
            if req and req["base"] not in ("in", "hz") and not any(r["base"] == req["base"] and abs(r["num"] - req["num"]) < 1e-9 for r in cons.reqs):
                cons.reqs.append(req)
                changed.append("req"); cons.llm_fields.add("req")
    return changed

def _should_ask(goal: str) -> bool:

    return "::" not in goal

def _known_shape(cons: Constraints) -> bool:
    return bool(cons.has_spec() or cons.model or cons.phrase or cons.pref_words or cons.candidate_keys or cons.wants_reason or cons.brand or cons.sealed)

def _question(cons: Constraints) -> str:
    return ("Before I pick anything, could you confirm your firm requirements and priorities? Specifically: the budget, the exact catalog "
            "category, whether a brand is required, any non-negotiable specification (for example screen size, resolution, refresh rate, memory, battery or storage), "
            "and which preference matters most to you?")

def _axis_evidence(cons: Constraints, c: Cand, opts: str) -> List[Tuple[str, str]]:

    out: List[Tuple[str, str]] = []
    s = c.size_in(opts)
    if s is not None and (cons.size_in is not None or not cons.reqs):
        out.append(("screen size", f"{s:g} inches"))
    hz = c.refresh_hz(opts)
    if hz is not None and cons.refresh_hz is not None:
        out.append(("refresh rate", f"{hz:g} Hz"))
    if cons.res_class:
        rc = c.res_class(opts)
        if rc:
            out.append(("resolution", rc.upper()))
    if cons.ram_gb is not None:
        rv, _ = c.spec("ram", opts)
        if rv:
            out.append(("memory", rv.upper()))
    for req in cons.reqs[:2]:
        val = c.req_fact_value(req, opts)
        if val:
            out.append((req["name"] or req["base"], val))
    if cons.pref_words:
        hits = c.axis_hits(cons)
        for a in _pref_axes(cons)[:2]:
            if hits.get(a, 0.0) < 0.75:
                continue
            label = a[2:] if a.startswith("x:") else a
            want = {w for w in re.findall(r"[a-z]+", label) if len(w) > 3}
            fact = next((str(x)[:40] for k, x in c.facts.items() if want & set(re.findall(r"[a-z]+", (str(k) + " " + str(x)).lower()))), "")
            if not fact and want:
                m = re.search(r"\b(?:" + "|".join(re.escape(w) for w in sorted(want)) + r")\w*(?:\s+[\w+.-]+){0,2}", c.title, re.I)
                fact = m.group(0).strip()[:50] if m else ""
            if fact:
                out.append((label, fact))
    return out

def _answer(cons: Constraints, state: State, chosen: Optional[Tuple[Cand, Dict[str, Any]]], text: str) -> str:
    if chosen is None:
        return "I am still comparing the in-stock options that meet every requirement; I will confirm the exact pick before ordering."
    c, v = chosen
    opts = str(v.get("options") or "")
    bits = [f"I am going with {c.title[:90]} ({c.brand or 'brand as listed'}) at {(_f(v.get('price')) or 0):.2f} PHP, in stock"]
    ev = _axis_evidence(cons, c, opts)
    for name, val in ev[:3]:
        bits.append(f"its {name} is {val}")
    if opts:
        bits.append(f"variant {opts}")
    alt = _contrast(cons, state, c, v)
    if alt:
        bits.append(alt)
    return "; ".join(bits) + "."

def _contrast(cons: Constraints, state: State, c: Cand, v: Dict[str, Any]) -> str:

    try:
        price = _f(v.get("price")) or 0.0
        seen_title = c.title[:40].lower()
        for cc, vv, _ in _candidates(cons, state, relaxed=1)[:8]:
            if cc.pid == c.pid or cc.title[:40].lower() == seen_title:
                continue
            if cc.listing_id and c.listing_id and cc.listing_id == c.listing_id:
                continue
            o2 = str(vv.get("options") or "")
            p2 = _f(vv.get("price")) or 0.0
            ev1 = dict(_axis_evidence(cons, c, str(v.get("options") or "")))
            ev2 = dict(_axis_evidence(cons, cc, o2))
            for name, val in ev1.items():
                if name in ev2 and ev2[name] != val:
                    return f"I compared it with {cc.title[:60]} ({name} {ev2[name]}) and kept this one because its {name} is {val}"
            if cons.pref_words and cc.viewed and c.viewed and cc.pref_value(cons) < c.pref_value(cons) - 1e-9:
                axis = (_pref_axes(cons) or ["fit"])[0]
                return f"I compared it with {cc.title[:60]}, which matches {axis[2:] if axis.startswith('x:') else axis} less closely"
            if abs(p2 - price) > 0.01 and ev1:
                name, val = next(iter(ev1.items()))
                return f"I compared it with {cc.title[:60]} at {p2:.2f} PHP and kept this one for its {name} {val}"
        return ""
    except Exception:
        return ""

def agent_main(problem_data: dict) -> list:
    t0 = time.monotonic()
    pd_ = problem_data if isinstance(problem_data, dict) else {}
    env = pd_.get("environment") if isinstance(pd_.get("environment"), dict) else {}
    binding = dict(env.get("binding") or {}) if isinstance(env.get("binding"), dict) else {}
    policy = env.get("policy_view") if isinstance(env.get("policy_view"), dict) else {}
    tools = {t.get("function", {}).get("name"): t for t in (policy.get("tools") or []) if isinstance(t, dict)}
    try:
        max_steps = int(policy.get("max_steps") or 20)
    except (TypeError, ValueError):
        max_steps = 20
    try:
        max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
    except (TypeError, ValueError):
        max_calls = 1
    full_query = str(policy.get("query") or pd_.get("query") or "")
    goal, _, rules = full_query.partition("\n\nTask rules:")
    goal = goal.strip() or full_query
    pid_ = str(pd_.get("problem_id") or "problem")
    _LLM_SPENT[0] = 0.0
    del SHADES[_SHADES_BASE:]
    cons = Constraints()
    cons.raw_goal = goal
    try:
        cons.absorb(goal)
        if (not cons.category or cons.budget is None or not _known_shape(cons)) and time.monotonic() - t0 < DEADLINE_S - 120:
            _apply_parse(cons, _llm_parse(goal), only_missing=True)
    except Exception:
        pass
    max_searches = MAX_SEARCHES if cons.phrase else (19 if cons.model else 12)
    state = State()
    dialogue: List[Dict[str, Any]] = []
    recoveries = 0
    pending_answer = ""

    last_sig = [""]
    repeats = [0]

    def call(turn: int, actions: List[Tuple[str, Dict[str, Any]]], note: str) -> Optional[Dict[str, Any]]:
        actions = [(n, a) for n, a in actions if n in tools][:max_calls]
        if not actions:
            return None
        state.acts += len(actions)
        sig = json.dumps(actions, sort_keys=True, default=str)
        if sig == last_sig[0]:
            repeats[0] += 1
            if repeats[0] >= 2:

                for n_, a_ in actions:
                    if n_ in ("compare", "view"):
                        for pid in (a_.get("product_ids") or [a_.get("product_id")]):
                            if pid and str(pid) in state.cands:
                                state.cands[str(pid)].viewed = True
                    if n_ == "add_to_cart" and a_.get("product_id"):
                        state.rejected.add((str(a_.get("product_id")), str(a_.get("sku"))))
                state.log.append("repeat-guard")
        else:
            repeats[0] = 0
        last_sig[0] = sig
        gid = f"{pid_}-turn-{turn}"
        envelope = {**binding, "call_id": gid, "idempotency_key": gid, "turn": turn,
                    "calls": [{"call_id": f"{gid}-{i}", "action": {"name": n, "args": a}} for i, (n, a) in enumerate(actions, 1)]}
        if time.monotonic() - t0 > _HARD_STOP_S:
            state.env_fail += 1
            return None
        _lat0 = time.monotonic()
        _gap = _lat0 - state.env_last
        if state.env_last > 0.0 and 0.0 <= _gap < _ENV_GAP_S:
            time.sleep(_ENV_GAP_S - _gap)
        result = None
        for _att in range(_ENV_TRIES):
            result = _ENV.post("/environment/call", json_data=envelope)
            if isinstance(result, dict):
                break
            if time.monotonic() - t0 > DEADLINE_S or time.monotonic() - _lat0 > _ENV_CALL_CAP_S:
                break
            time.sleep(min(0.4 * (2 ** _att), 3.0))
        state.env_last = time.monotonic()
        state.env_lat = 0.5 * state.env_lat + 0.5 * (state.env_last - _lat0)
        if not isinstance(result, dict):
            state.env_fail += 1
            return None
        state.env_fail = 0
        calls = result.get("calls") or []
        for (n, a), cr in zip(actions, calls):
            obs = (cr or {}).get("observation") or {}
            payload = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
            if isinstance(payload, dict):
                _ingest(state, n, payload)
            if obs.get("done") is True:
                state.done = True
            err = obs.get("error")
            if err:
                state.log.append(f"{n}: {str(err)[:80]}")
                if n == "add_to_cart" and a.get("product_id"):
                    state.rejected.add((str(a.get("product_id")), str(a.get("sku"))))
                    state.cart = [x for x in state.cart if not (str(x.get("product_id")) == str(a.get("product_id")) and str(x.get("sku")) == str(a.get("sku")))]
                if n == "place_test_order" and "cart" in str(err).lower():
                    state.cart = []
                if n == "search" and "max_price" in a and state.search_price_ok:
                    state.search_price_ok = False
                    state.used_queries.discard(str(a.get("query", "")).lower())
                    state.searches = max(0, state.searches - 1)
        um = result.get("user_message")
        if isinstance(um, dict) and um.get("content"):
            _shopper(um["content"], str(um.get("action") or ""))
        step = {"role": "assistant", "content": note[:400],
                "tool_calls": [{"id": f"{gid}-{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for i, (n, a) in enumerate(actions, 1)],
                "environment_result": result}
        if len(json.dumps(dialogue, ensure_ascii=False, default=str)) < DIALOG_LIMIT:
            dialogue.append(step)
        return result

    def _shopper(text: str, action: str = "") -> None:
        nonlocal pending_answer
        low = text.lower()
        state.shopper_texts.append(text)
        if re.search(r"\bbudget\b|\bcategor(?:y|ies)\b|\bbrand\b|\b(?:only|can) go up to\b|\bstill needs?\b|\bneeds? an? \d|\bmust (?:now )?have\b|\bchang\w+ (?:one )?requirement\b|\brestat\w+\b|\bnon-negotiable\b", low):
            old_budget = cons.budget
            cons.absorb(text, from_shopper=True)
            state.answered = True
            m = re.search(r"(?:\bgo up to\b|\bbudget\b[^.\d]{0,14}|spend[^.\d]{0,10}|cap(?:ped)?[^.\d]{0,6}|no more than)\s*[:=]?\s*([\d][\d,]*(?:\.\d+)?)", low)
            if m and _f(m.group(1)) is not None:
                cons.budget = _f(m.group(1))
            if old_budget is not None and cons.budget is not None and abs(cons.budget - old_budget) > 1e-6:
                state.filtered = False
                state.log.append("budget-changed")
        if "?" in text or "\uff1f" in text or REPLY_REQUEST_RE.search(text) or (action or "").startswith("clarify"):
            pending_answer = text
        if re.search(r"still (?:available|in stock)|more expensive|price (?:went|jumped|changed|rose)|went out of stock|sold out|unavailable|re-?check|double[- ]check", low):
            state.events.append({"kind": "shopper_notice", "text": text})

    def best() -> List[Tuple[Cand, Dict[str, Any], Tuple]]:
        return _candidates(cons, state)

    turn = 0
    try:
        while turn < max_steps and not state.done and state.env_fail < 6 and time.monotonic() - t0 < DEADLINE_S:
            turn += 1
            remaining = max_steps - turn + 1
            actions: List[Tuple[str, Dict[str, Any]]] = []
            note = ""
            ranked = best()
            cart_item = state.cart[-1] if state.cart else None

            if pending_answer and "message" in tools:
                chosen = None
                if cart_item:
                    c = state.cands.get(str(cart_item.get("product_id")))
                    if c:
                        chosen = (c, {"price": cart_item.get("current_price", cart_item.get("price_at_add")), "options": next((v.get("options") for v in c.variants if v.get("sku") == cart_item.get("sku")), "")})
                elif ranked:
                    chosen = (ranked[0][0], ranked[0][1])
                text = _answer(cons, state, chosen, pending_answer)
                pending_answer = ""
                actions = [("message", {"content": text})]
                if cart_item and "inspect_stock" in tools:
                    actions.append(("inspect_stock", {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")}))
                call(turn, actions, "Answering the shopper with observed facts and re-checking the cart item.")
                continue

            if not state.asked and "message" in tools and _should_ask(goal) and remaining > 6 and not state.retrieved:
                state.asked = True
                call(turn, [("message", {"content": _question(cons)})], "Asking the shopper for the firm requirements and priorities.")
                continue

            if cart_item and "place_test_order" in tools and (remaining <= 2 or time.monotonic() - t0 > DEADLINE_S - max(15.0, 2.0 * state.env_lat)):
                args = {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")}
                if cons.wants_reason:
                    args["justification"] = _justification(cons, state, cart_item)
                call(turn, [("place_test_order", args)], "Turn budget nearly spent: placing the order for the verified cart item.")
                continue

            if cons.candidate_keys and not state.retrieved:
                state.retrieved = True
                ids = []
                for pid, sku in cons.candidate_keys:
                    c = state.cand(pid)
                    c.summary_sku = c.summary_sku or sku
                    if pid not in ids:
                        ids.append(pid)
                if "compare" in tools and len(ids) >= 2:
                    actions.append(("compare", {"product_ids": ids[:5]}))
                    for pid in ids[5:]:
                        actions.append(("view", {"product_id": pid}))
                else:
                    for pid in ids:
                        actions.append(("view", {"product_id": pid}))
                if actions:
                    call(turn, actions, "Inspecting the shopper's named candidate variants.")
                    continue

            commit_after = DEADLINE_S - max(25.0, 3.5 * state.env_lat)
            if time.monotonic() - t0 > commit_after and ranked and not state.cart:
                state.log.append("watchdog-commit")
                cc0, vv0 = ranked[0][0], ranked[0][1]
                acts0: List[Tuple[str, Dict[str, Any]]] = []
                if not cc0.viewed and "view" in tools:
                    acts0.append(("view", {"product_id": cc0.pid}))
                acts0.append(("add_to_cart", {"product_id": cc0.pid, "sku": str(vv0.get("sku"))}))
                if "inspect_stock" in tools:
                    acts0.append(("inspect_stock", {"product_id": cc0.pid, "sku": str(vv0.get("sku"))}))
                call(turn, acts0, "Time budget is nearly spent: committing the best valid option.")
                continue
            potwierdzony = True
            if cons.has_spec():
                potwierdzony = any(_spec_ok(cons, cc, str(vv.get("options") or "")) is True for cc, vv, _ in ranked if cc.viewed)
            if (cons.has_spec() and not potwierdzony and not cons.phrase and not cons.wants_reason
                    and state.retrieved and not state.cart
                    and cons.category and cons.budget is not None and "filter" in tools
                    and state.enum < 1 and state.acts + 8 < max_steps):

                pasma = [0.75, 0.5, 0.3] if state.enum == 0 else [0.9, 0.6, 0.4]
                for frac in pasma:
                    cap = round(cons.budget * frac, 2)
                    if cap < 0.02 or state.acts + len(actions) >= max_steps - 6:
                        continue
                    arg = {"category": cons.category, "in_stock": True, "k": 10, "max_price": cap}
                    key = json.dumps(["filter", arg], sort_keys=True)
                    if key in state.used_queries:
                        continue
                    state.used_queries.add(key)
                    actions.append(("filter", arg))
                if actions:
                    state.enum += 1
                    call(turn, actions, "Enumerating the category across price bands before inspecting.")
                    ranked = best()
                    continue
            need_more = not state.retrieved or (len(ranked) < 2 and state.searches < max_searches) or (cons.phrase and state.searches == 2 and not state.cart)
            if need_more and state.searches < max_searches and not state.cart:
                state.retrieved = True
                queries = _queries(cons, goal, state)
                if cons.phrase:
                    per_turn = 2 if state.searches == 0 else 1
                elif cons.model:
                    per_turn = max_searches - state.searches
                else:
                    per_turn = 2
                    if cons.pref_words:
                        per_turn = 3
                    if "filter" in tools and cons.category and not state.filtered:
                        state.filtered = True
                        fa: Dict[str, Any] = {"category": cons.category, "in_stock": True, "k": 10}
                        if cons.budget is not None:
                            fa["max_price"] = cons.budget
                        if cons.brand:
                            actions.append(("filter", {**fa, "brand": cons.brand}))
                        if not (cons.brand and cons.brand_hard):
                            actions.append(("filter", dict(fa)))
                        for alt in cons.category_alts[:2]:
                            actions.append(("filter", {**fa, "category": alt}))
                for q in queries:
                    if len([a for a in actions if a[0] == "search"]) >= max(1, per_turn) or state.searches >= max_searches:
                        break
                    if q.lower() in state.used_queries:
                        continue
                    state.used_queries.add(q.lower())
                    actions.append(("search", _search_args(cons, state, q)))
                    state.searches += 1
                if actions:
                    call(turn, actions, "Retrieving candidates with structured filters and focused searches.")

                    ranked = best()
                    continue

            if (cons.phrase and not state.cart and "search" in tools and cons.budget is not None
                    and state.search_price_ok and state.searches >= max_searches and state.bands < 1):
                state.bands = 1
                full = [c for c in state.cands.values() if c.phrase_hits(cons.phrase) >= 0.999]
                lids = {c.listing_id or c.pid for c in full}
                if len(lids) < 10:
                    base_qs = [q for q in _queries(cons, goal, state)[:3] if q] or [cons.phrase]
                    prices = sorted({round(float(c.summary_price), 2) for c in full if c.summary_price})
                    caps: List[float] = []
                    if prices:
                        for frac in (0.6, 0.3):
                            v = prices[min(int(len(prices) * frac), len(prices) - 1)]
                            if v and v < cons.budget and v not in caps:
                                caps.append(v)
                        v0 = round(prices[0] - 0.01, 2)
                        if v0 >= 0.02 and v0 not in caps:
                            caps.append(v0)
                    for i, cap in enumerate(caps[:3]):
                        if state.acts + len(actions) >= max_steps - 8:
                            break
                        arg = {"query": base_qs[i % len(base_qs)], "k": 10, "in_stock": True, "max_price": cap}
                        key = json.dumps(["search", arg], sort_keys=True)
                        if key in state.used_queries:
                            continue
                        state.used_queries.add(key)
                        actions.append(("search", arg))
                    if actions:
                        call(turn, actions, "Widening coverage of the same queries across narrower price bands.")
                        ranked = best()
                        continue

            unviewed = [c for c, _, _ in ranked if not c.viewed]
            if cons.phrase:
                full = [c for c in unviewed if c.phrase_hits(cons.phrase) >= 0.999]
                viewed_full = [c for c, _, _ in ranked if c.viewed and c.phrase_hits(cons.phrase) >= 0.999]
                seen_lids = {c.listing_id or c.pid for c in viewed_full}
                if len(seen_lids) >= 2:
                    unviewed = []
                elif full:
                    unviewed = full[:3]
                elif not viewed_full and "low-hit" not in state.log:
                    state.log.append("low-hit")
                    unviewed = unviewed[:2]
                else:
                    unviewed = []
                if viewed_full and "brand-filters" not in state.log and "filter" in tools and not state.cart:
                    state.log.append("brand-filters")
                    brands = []
                    for c in viewed_full:
                        if c.brand and c.brand.lower() not in [b.lower() for b in brands]:
                            brands.append(c.brand)
                    for b in brands[:3]:
                        fa = {"category": cons.category, "brand": b, "in_stock": True, "k": 10}
                        if cons.budget is not None:
                            fa["max_price"] = cons.budget
                        actions.append(("filter", {k: v for k, v in fa.items() if v}))
                    if actions:
                        call(turn, actions, "Surfacing sibling listings of the matching brands.")
                        continue
            if cons.pref_words:
                viewed_scores = [(c.pref_value(cons), c.listing_id or c.pid) for c, _, _ in ranked if c.viewed]
                best_seen = max([sc for sc, _ in viewed_scores] or [0.0])
                n_best = len({lid for sc, lid in viewed_scores if sc >= best_seen - 1e-9})
                rounds = state.log.count("pref-round")
                if best_seen >= 0.999 and n_best >= 2:
                    unviewed = []
                elif rounds >= 4:
                    unviewed = []
                else:
                    unviewed.sort(key=lambda c: -c.pref_value(cons))
                    unviewed = [c for c in unviewed if c.pref_value(cons) >= 0.5]
                    if unviewed:
                        state.log.append("pref-round")
            if unviewed and not state.cart:
                wolne = max(0, (max_steps - 8) - state.acts)
                grupy = max(1, min(3, max(1, max_calls), wolne // max(1, SHORTLIST)))
                ids = []
                for c in unviewed:
                    if c.pid not in ids:
                        ids.append(c.pid)
                    if len(ids) >= SHORTLIST * grupy:
                        break
                if "compare" in tools and len(ids) >= 2:
                    for i in range(0, min(len(ids), SHORTLIST * grupy), SHORTLIST):
                        porcja = ids[i:i + SHORTLIST]
                        if len(porcja) >= 2:
                            actions.append(("compare", {"product_ids": porcja}))
                        elif porcja and "view" in tools:
                            actions.append(("view", {"product_id": porcja[0]}))
                elif "view" in tools:
                    for pid in ids[:6]:
                        actions.append(("view", {"product_id": pid}))
                if actions:
                    call(turn, actions, "Inspecting raw facts and variants of the shortlist before choosing.")
                    continue

            if not state.cart:
                ranked = best()
                if not ranked:
                    if state.searches < max_searches and remaining > 3:
                        pool_q = (_recovery_queries(cons, state) if cons.model and cons.color and time.monotonic() - t0 < DEADLINE_S - 90 else []) + _queries(cons, goal, state) + _queries(cons, goal, state, alt=True)
                        _ile = 6 if cons.model else (4 if cons.has_spec() else 1)
                        for q in [x for x in pool_q if x.lower() not in state.used_queries][:_ile]:
                            state.used_queries.add(q.lower())
                            actions.append(("search", _search_args(cons, state, q)))
                            state.searches += 1
                        if cons.category and "filter" in tools and cons.budget is not None and not state.filtered:
                            state.filtered = True
                            fa = {"category": cons.category, "in_stock": True, "k": 10, "max_price": cons.budget}
                            if cons.brand and cons.brand_hard:
                                fa["brand"] = cons.brand
                            actions.append(("filter", fa))
                        if actions:
                            call(turn, actions, "Widening retrieval: no in-stock candidate within budget yet.")
                            continue
                    ranked = _candidates(cons, state, relaxed=1) or _candidates(cons, state, relaxed=2)
                    if not ranked:
                        if not cons.reported_empty and "message" in tools:
                            cons.reported_empty = True
                            call(turn, [("message", {"content": "No available option within your budget meets every firm requirement. Should I keep the spec firm and look further, or can the budget or the spec change?"})],
                                 "Reporting the empty result to the shopper once.")
                            continue
                        if remaining > 3 and "nomore" not in state.log:
                            fresh_q = [x for x in _queries(cons, goal, state) + _queries(cons, goal, state, alt=True) if x.lower() not in state.used_queries]
                            if cons.model and cons.color and time.monotonic() - t0 < DEADLINE_S - 90:
                                fresh_q = _recovery_queries(cons, state) + fresh_q
                            if time.monotonic() - t0 < DEADLINE_S - 90:
                                if "models-asked" not in state.log:
                                    state.log.append("models-asked")
                                    fresh_q = [x for x in _llm_models(cons, goal) if x.lower() not in state.used_queries] + fresh_q
                                else:
                                    fresh_q = [x for x in _llm_queries(cons, goal) if x.lower() not in state.used_queries] + fresh_q
                            for q in fresh_q[:2]:
                                state.used_queries.add(q.lower())
                                actions.append(("search", _search_args(cons, state, q)))
                                state.searches += 1
                            if cons.category and "filter" in tools and not state.filtered:
                                state.filtered = True
                                fa = {"category": cons.category, "in_stock": True, "k": 10}
                                if cons.budget is not None:
                                    fa["max_price"] = cons.budget
                                if cons.brand and cons.brand_hard:
                                    fa["brand"] = cons.brand
                                actions.append(("filter", fa))
                            if actions:
                                call(turn, actions, "Nothing valid yet: widening the search under the current requirements.")
                                continue
                            if "llm-parse" not in state.log and time.monotonic() - t0 < DEADLINE_S - 90:
                                state.log.append("llm-parse")
                                changed = _apply_parse(cons, _llm_parse(goal + "\n" + "\n".join(state.shopper_texts[-4:])), only_missing=False, pool=state.cands)
                                if changed:
                                    state.log.append("reparsed:" + ",".join(changed))
                                    state.filtered = False
                                    state.searches = max(0, state.searches - 2)
                                    continue
                            if "general-mode" not in state.log and remaining > 3 and time.monotonic() - t0 < DEADLINE_S - max(25.0, 4.0 * state.env_lat):
                                state.log.append("general-mode")
                                gen = _llm_actions(goal, rules, tools, state, cons)
                                if gen:
                                    call(turn, gen, "No deterministic candidate: following a model-proposed tool plan.")
                                    continue
                            state.log.append("nomore")
                        break
                pick = None
                c, v = ranked[0][0], ranked[0][1]
                if pick:
                    for cc, vv, _ in ranked:
                        if cc.pid == pick[0] and str(vv.get("sku")) == pick[1]:
                            c, v = cc, vv
                            break
                if cons.pref_words:
                    top_lid = c.listing_id or c.pid
                    others = [(cc, vv) for cc, vv, _ in ranked if (cc.listing_id or cc.pid) != top_lid and cc.viewed and cc.pref_ok(cons)]
                    if others and others[0][0].pref_value(cons) < c.pref_value(cons) - 1e-9:
                        c, v = others[0]
                if not state.asked and _should_ask(goal) and "message" in tools and remaining > 3:
                    state.asked = True
                    call(turn, [("message", {"content": _question(cons)})], "Confirming requirements and priorities before the first cart action.")
                    continue
                if cons.has_spec() and _spec_ok(cons, c, str(v.get("options") or "")) is not True and remaining > 3:
                    nieobejrzane = [cc for cc, _, _ in ranked if not cc.viewed]
                    if nieobejrzane and state.acts + 4 < max_steps and "nomore" not in state.log:
                        ids_n = []
                        for cc in nieobejrzane:
                            if cc.pid not in ids_n:
                                ids_n.append(cc.pid)
                            if len(ids_n) >= SHORTLIST:
                                break
                        acts_n: List[Tuple[str, Dict[str, Any]]] = []
                        if "compare" in tools and len(ids_n) >= 2:
                            acts_n.append(("compare", {"product_ids": ids_n[:5]}))
                        elif "view" in tools and ids_n:
                            acts_n.append(("view", {"product_id": ids_n[0]}))
                        if acts_n:
                            call(turn, acts_n, "Confirming the stated specification on the remaining options before committing.")
                            continue
                if not c.viewed and remaining > 3 and "view" in tools:
                    ids = []
                    for cc, _, _ in ranked:
                        if not cc.viewed and cc.pid not in ids:
                            ids.append(cc.pid)
                        if len(ids) >= SHORTLIST:
                            break
                    acts = [("compare", {"product_ids": ids[:5]})] if "compare" in tools and len(ids) >= 2 else [("view", {"product_id": ids[0]})]
                    call(turn, acts, "Inspecting the remaining candidates before committing.")
                    continue
                state.committed = (c.pid, str(v.get("sku")))
                actions = [("add_to_cart", {"product_id": c.pid, "sku": str(v.get("sku"))})]
                if "inspect_stock" in tools:
                    actions.append(("inspect_stock", {"product_id": c.pid, "sku": str(v.get("sku"))}))
                call(turn, actions, f"Committing to {c.title[:60]} and verifying its live stock and price.")
                continue

            cart_item = state.cart[-1]
            c = state.cands.get(str(cart_item.get("product_id")))
            cur_price = _f(cart_item.get("current_price")) if cart_item.get("current_price") is not None else _f(cart_item.get("price_at_add"))
            in_stock = cart_item.get("in_stock")
            if c is not None:
                for v in c.variants:
                    if v.get("sku") == cart_item.get("sku"):
                        if v.get("in_stock") is not None:
                            in_stock = v.get("in_stock")
                        if _f(v.get("price")) is not None:
                            cur_price = _f(v.get("price"))
            over = cons.budget is not None and cur_price is not None and cur_price > cons.budget + 1e-6
            if c is not None and c.viewed:
                cart_opts = str(next((v.get("options") for v in c.variants if str(v.get("sku")) == str(cart_item.get("sku"))), "") or "")
                if _spec_ok(cons, c, cart_opts) is False or (cons.brand_hard and not _brand_ok(cons, c)):
                    over = True
            fresh = any(e.get("kind") == "shopper_notice" for e in state.events[-2:]) and not state.log[-1:] == ["checked"]
            hit_by_event = any(
                isinstance(e.get("target"), dict) and str(e["target"].get("product_id")) == str(cart_item.get("product_id")) and str(e["target"].get("sku")) == str(cart_item.get("sku"))
                for e in state.events
            ) and not any(e.get("kind") == "kept_after_event" for e in state.events)
            if not state.events and "rechecked" not in state.log and "inspect_stock" in tools and remaining > 2:

                state.log.append("rechecked")
                call(turn, [("inspect_stock", {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")})], "Re-reading live stock and price of the cart item before ordering.")
                continue
            if (in_stock is False or over or hit_by_event) and recoveries < 3:
                recoveries += 1
                state.rejected.add((str(cart_item.get("product_id")), str(cart_item.get("sku"))))
                actions = []
                if "remove_from_cart" in tools:
                    actions.append(("remove_from_cart", {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")}))
                ranked = best()
                if not ranked and (in_stock is False or over) and c is not None and c.title and "sibling-search" not in state.log and remaining > 4:
                    state.log.append("sibling-search")
                    recoveries -= 1
                    state.rejected.discard((str(cart_item.get("product_id")), str(cart_item.get("sku"))))
                    opts = str(next((v.get("options") for v in c.variants if str(v.get("sku")) == str(cart_item.get("sku"))), "") or "")
                    q = " ".join(re.findall(r"[A-Za-z0-9][\w\-\+\.]*", c.title)[:8] + re.findall(r"[a-z0-9]+", opts.lower())[:3])
                    sib = [("search", _search_args(cons, state, q))]
                    state.used_queries.add(q.lower()); state.searches += 1
                    model_fact = str(c.facts.get("model") or "").strip()
                    if model_fact and model_fact.lower() not in state.used_queries and len(model_fact) <= 40:
                        state.used_queries.add(model_fact.lower()); state.searches += 1
                        sib.append(("search", _search_args(cons, state, model_fact)))
                    for mq in (_llm_models(cons, goal) if "models-asked" not in state.log and time.monotonic() - t0 < DEADLINE_S - 90 else [])[:1]:
                        state.log.append("models-asked"); state.used_queries.add(mq.lower()); state.searches += 1
                        sib.append(("search", _search_args(cons, state, mq)))
                    call(turn, sib, "The committed item is gone; looking for sibling listings of the same product before relaxing.")
                    continue
                if not ranked and (in_stock is False or over):
                    ranked = _candidates(cons, state, relaxed=1)
                others = [r for r in ranked if r[0].pid != str(cart_item.get("product_id"))]
                same = [r for r in ranked if r[0].pid == str(cart_item.get("product_id"))]
                if cons.model or cons.color:
                    ranked = same + others
                elif hit_by_event and not (in_stock is False or over):

                    ranked = others if cons.pref_words else []
                if cons.pref_words and c is not None and c.listing_id:
                    ranked = [r for r in ranked if r[0].listing_id != c.listing_id]
                    if not ranked:
                        state.events.append({"kind": "kept_after_event"})
                        call(turn, [("inspect_stock", {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")})], "No alternative product; keeping the still-valid item after re-checking it.")
                        continue
                else:
                    ranked = others + same if not cons.ranking else ranked
                viewed_first = [r for r in ranked if r[0].viewed] + [r for r in ranked if not r[0].viewed]
                ranked = viewed_first
                if ranked and not ranked[0][0].viewed and remaining > 3:
                    ids = []
                    for r in ranked:
                        if not r[0].viewed and r[0].pid not in ids:
                            ids.append(r[0].pid)
                        if len(ids) >= SHORTLIST:
                            break
                    if "compare" in tools and len(ids) >= 2:
                        actions.append(("compare", {"product_ids": ids[:5]}))
                    else:
                        actions.append(("view", {"product_id": ids[0]}))
                    recoveries -= 1
                    call(turn, actions, "Inspecting replacement candidates before changing the cart.")
                    continue
                if ranked:
                    cc, vv = ranked[0][0], ranked[0][1]
                    actions.append(("add_to_cart", {"product_id": cc.pid, "sku": str(vv.get("sku"))}))
                    if "inspect_stock" in tools:
                        actions.append(("inspect_stock", {"product_id": cc.pid, "sku": str(vv.get("sku"))}))
                    call(turn, actions, "The committed item changed (stock or price); switching to the least-relaxed valid alternative.")
                else:
                    fresh_q = []
                    if c is not None and c.title and "sibling-search" not in state.log:
                        state.log.append("sibling-search")
                        fresh_q.append(" ".join(re.findall(r"[A-Za-z0-9][\w\-\+\.]*", c.title)[:8]))
                    if "models-asked" not in state.log and time.monotonic() - t0 < DEADLINE_S - 90:
                        state.log.append("models-asked")
                        fresh_q += _llm_models(cons, goal)
                    fresh_q += _queries(cons, goal, state) + _queries(cons, goal, state, alt=True)
                    for q in [x for x in fresh_q if x.lower() not in state.used_queries][:2]:
                        state.used_queries.add(q.lower())
                        actions.append(("search", _search_args(cons, state, q)))
                        state.searches += 1
                    if cons.category and "filter" in tools:
                        fa = {"category": cons.category, "in_stock": True, "k": 10}
                        if cons.budget is not None:
                            fa["max_price"] = cons.budget
                        if cons.brand and cons.brand_hard:
                            fa["brand"] = cons.brand
                        actions.append(("filter", fa))
                    call(turn, actions, "Committed item invalid; searching for a replacement.")
                continue
            if fresh and "inspect_stock" in tools:
                state.log.append("checked")
                call(turn, [("inspect_stock", {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")})], "Re-checking the cart item after the shopper's notice.")
                continue
            if state.log.count("order-attempt") >= 2:
                break
            state.log.append("order-attempt")
            args = {"product_id": cart_item.get("product_id"), "sku": cart_item.get("sku")}
            if cons.wants_reason:
                args["justification"] = _justification(cons, state, cart_item)
            call(turn, [("place_test_order", args)], "Verified in stock and within budget: placing the order.")
    except Exception as exc:
        state.log.append(f"error: {type(exc).__name__}: {str(exc)[:120]}")
    try:

        for _final in range(2):
            if state.ordered or state.done or "place_test_order" not in tools:
                break
            item = state.cart[-1] if state.cart else None
            if item is None:
                ranked = _candidates(cons, state, relaxed=0) or _candidates(cons, state, relaxed=1) or _candidates(cons, state, relaxed=2)
                if not ranked and cons.llm_fields:

                    if "brand" in cons.llm_fields:
                        cons.brand, cons.brand_hard = "", False
                    if "size" in cons.llm_fields:
                        cons.size_in = None
                    if "hz" in cons.llm_fields:
                        cons.refresh_hz = None
                    if "res" in cons.llm_fields:
                        cons.res_class = ""
                    if "ram" in cons.llm_fields:
                        cons.ram_gb = None
                    if "req" in cons.llm_fields:
                        cons.reqs = []
                    cons.llm_fields = set()
                    state.log.append("drop-llm-constraints")
                    ranked = _candidates(cons, state, relaxed=1) or _candidates(cons, state, relaxed=2)
                if not ranked:
                    break
                cc, vv = ranked[0][0], ranked[0][1]
                turn += 1
                acts: List[Tuple[str, Dict[str, Any]]] = [("add_to_cart", {"product_id": cc.pid, "sku": str(vv.get("sku"))})]
                if "inspect_stock" in tools:
                    acts.append(("inspect_stock", {"product_id": cc.pid, "sku": str(vv.get("sku"))}))
                call(turn, acts, "Committing the best remaining option before the episode ends.")
                item = state.cart[-1] if state.cart else None
                if item is None:
                    state.rejected.add((cc.pid, str(vv.get("sku"))))
                    continue
            args = {"product_id": item.get("product_id"), "sku": item.get("sku")}
            if cons.wants_reason:
                args["justification"] = _justification(cons, state, item)
            for _ord in range(3):
                turn += 1
                call(turn, [("place_test_order", args)], "Placing the order for the verified cart item.")
                if state.ordered or state.done:
                    break
            break
    except Exception as exc2:
        state.log.append(f"final: {type(exc2).__name__}: {str(exc2)[:80]}")
    if not dialogue:
        dialogue.append({"role": "assistant", "content": "No environment interaction was possible.", "tool_calls": []})
    try:
        dialogue.append({"role": "assistant", "content": "run: turns=%d ordered=%s cart=%d trail=%s" % (turn, state.ordered, len(state.cart), ",".join(state.log[-12:])[:300]), "tool_calls": []})
    except Exception:
        pass
    return dialogue

ATTR_SYNONYMS = {
    "gb": ("storage", "rom", "ssd", "capacity", "memory"),
    "mah": ("battery", "battery capacity"),
    "g": ("weight",),
    "mm": ("length", "size"),
    "w": ("power", "output"),
    "mp": ("camera", "megapixel"),
    "h": ("battery life", "playtime"),
    "ml": ("volume", "capacity"),
    "mhz": ("speed", "clock"),
    "hz": ("refresh rate",),
    "in": ("screen", "display"),
}

def _req_query_forms(cons: "Constraints") -> List[str]:

    cat = (cons.category or "").rstrip("s")
    dev = cat.lower()
    out: List[str] = []
    for r in cons.reqs[:2]:
        num = r["value"][: -len(r["unit"])] if r["value"].endswith(r["unit"]) else r["value"]
        unit = r["unit"]
        names = [r["name"]] if r["name"] else []
        names += [s for s in ATTR_SYNONYMS.get(r["base"], ()) if s not in names]
        vals = [f"{num}{unit}", f"{num} {unit}"]
        for v in vals[:2]:
            if dev:
                out.append(f"{dev} {v}")
            for nm in names[:3]:
                out.append(f"{dev} {v} {nm}".strip())
                out.append(f"{v} {nm} {dev}".strip())
            if cons.brand:
                out.append(f"{cons.brand} {dev} {v}".strip())
    seen: set = set()
    res: List[str] = []
    for q in out:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            res.append(q)
    return res[:8]

def _queries(cons: Constraints, goal: str, state: State, alt: bool = False) -> List[str]:
    cat = cons.category or ""
    out: List[str] = []
    if cons.phrase:
        out.append(cons.phrase)
        out.append(f"{cons.phrase} {cat}".strip())
        ptoks = set(re.findall(r"[a-z0-9]+", cons.phrase.lower()))
        freq: Dict[str, int] = {}
        for c in state.cands.values():
            if c.phrase_hits(cons.phrase) < 0.999:
                continue
            for w in set(re.findall(r"[a-z][a-z0-9]{2,}", c.title.lower())):
                if w not in ptoks:
                    freq[w] = freq.get(w, 0) + 1
        n_full = sum(1 for c in state.cands.values() if c.phrase_hits(cons.phrase) >= 0.999)
        discriminative = {w: f for w, f in freq.items() if n_full < 3 or f < 0.8 * n_full}
        extra = [w for w, _ in sorted((discriminative or freq).items(), key=lambda x: (-x[1], x[0]))[:2]]
        out.append(f"{cons.phrase} {' '.join(extra)}".strip() if extra else f"{cat.split()[0] if cat else 'in stock'} {cons.phrase}".strip())
    spec = ""
    if cons.size_in is not None:
        spec = f"{cons.size_in:g} inch"
        spec_alt = f"{cons.size_in:g}in"
    elif cons.refresh_hz is not None:
        spec = f"{cons.refresh_hz:g}hz"
    elif cons.res_class:
        spec = cons.res_phrase or cons.res_class
    elif cons.ram_gb is not None:
        spec = f"{cons.ram_gb:g}gb ram"
    elif cons.reqs:
        spec = " ".join(f"{r['text']} {r['name']}".strip() for r in cons.reqs[:2])
        spec_alt = " ".join(r["text"] for r in cons.reqs[:2])
        req_alts = _req_query_forms(cons)
    if cons.model:
        dev = cat.rstrip("s").lower()
        for v in _model_variants(cons.model) or [cons.model]:
            if dev.startswith("smartphone") and cons.color:
                out.append(f"{v} {cons.color} android")
            out.append(f"{v} {cons.color} {dev}".strip())
            out.append(f"{v} {dev}".strip())
        out.append(f"{cons.model} {cons.color} {dev}".strip())
        out.append(f"{cons.model} {dev}".strip())
        out.append(cons.model)
    if cons.brand and cat:
        branded = [f"{cons.brand} {cat.rstrip('s')} {spec}".strip()] + ([f"{cons.brand} {spec}".strip()] if spec else [])
        generic = [f"{cat.rstrip('s')} {spec}".strip(), f"{spec} {cat.rstrip('s').lower()}".strip()] if spec else [cat.rstrip("s")]
        if cons.brand_hard:
            out += branded
        elif "budget-changed" in state.log:
            out += generic + branded
        else:
            out += branded + generic
    if cons.pref_words and cat and any(a.startswith("x:") for a in cons.pref_words):
        dev = cat.rstrip("s").lower()
        words = [(a[2:] if a.startswith("x:") else AXIS_QUERY_WORDS.get(a, a)) for a in _pref_axes(cons)]
        if words:
            out.append(f"{words[0]} {dev}")
        if len(words) > 1:
            out.append(f"{words[0]} {words[1]} {dev}")
        if len(words) > 2:
            out.append(f"{dev} {' '.join(words[2:4])}")
    elif cons.pref_words and cat:
        dev = cat.rstrip("s").lower()
        if dev.startswith("smartwatch"):
            out += ["rugged gps smartwatch waterproof", "long battery gps smartwatch outdoor", "outdoor fitness gps watch 5atm days battery"]
        elif dev.startswith("tablet"):
            out += ["kids tablet case bundle drop proof", "kids tablet 10 inch wifi", "wifi tablet long battery hours kids"]
        else:
            out.append(f"sport wireless {dev} ipx7 waterproof 30 hours")
            out.append(f"sweatproof sports bluetooth {dev} long battery life hours")
            out.append(f"waterproof running {dev} bluetooth 5.3 40h playtime")
    if cat and not cons.phrase:
        out.append(f"{cat.rstrip('s')} {spec}".strip() if spec else cat.rstrip("s"))
        if cons.size_in is not None or cons.reqs:
            out.append(f"{cons.brand + ' ' if cons.brand else ''}{cat.rstrip('s').lower()} {spec_alt}")
    if alt:
        out = [f"{cat} {cons.brand}".strip()] + out
    if cons.reqs:
        out.extend(req_alts)
    if cons.noun_phrase:
        out.append(f"{cons.brand + ' ' if cons.brand else ''}{cons.noun_phrase} {spec}".strip())
        out.append(cons.noun_phrase)
    seen: set = set()
    res = []
    for q in out:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            res.append(q)
    return res

def _justification(cons: Constraints, state: State, cart_item: Dict[str, Any]) -> Dict[str, Any]:
    claims: List[Dict[str, str]] = []
    c = state.cands.get(str(cart_item.get("product_id")))
    field = cons.spec_field()
    if field not in CLAIMABLE_FIELDS:
        field = ""
    if c is not None and field:
        opts = next((str(v.get("options") or "") for v in c.variants if v.get("sku") == cart_item.get("sku")), "")
        got, _ = c.spec(field, opts)
        fact = c.fact_spec(field)
        val = fact or got
        if val:
            phrase = {"4k": "4K", "qhd": "QHD (1440p)", "fhd": "1080p Full HD", "hd": "720p HD"}.get(val, val) if field == "resolution" else (f"{val[:-2]} Hz" if field == "refresh_rate" else f"{val[:-2]} inch")
            claims.append({"field": field, "value": phrase})
    if c is not None and c.brand:
        claims.append({"field": "brand", "value": c.brand})
    if cons.budget is not None:
        claims.append({"field": "price", "value": f"at most {cons.budget:.2f}"})
    claims.append({"field": "in_stock", "value": "true"})
    return {"claims": claims}

__all__ = ["agent_main"]
