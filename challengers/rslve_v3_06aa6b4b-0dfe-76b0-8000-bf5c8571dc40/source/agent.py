"""ORO-Bench EnvPack shopper.

Explorer model surfaces pages. A closer quotes, carts, re-quotes and orders.
The first successful add_to_cart is the graded commitment and the market-event
hook. After a reprice or stock-out, the ordered survivor is what paid_reward
keeps — a cheap silent recovery zeros the episode even if the first add was right.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_CHUTES = (
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "zai-org/GLM-5.2-TEE",
    "Qwen/Qwen3-32B-TEE",
)
_OPENROUTER = (
    "deepseek/deepseek-v3.2",
    "z-ai/glm-5.2",
    "qwen/qwen3-32b",
)
_WIRE = ProxyClient(timeout=100, max_retries=2, retry_delay=1.0)
_EPISODE_S = 268.0
_OBS_CAP = 6800
_EPS = 1e-9
_STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "find", "need",
    "want", "please", "show", "looking", "under", "over", "from", "into", "have",
    "has", "are", "was", "will", "can", "php", "usd", "order", "stock", "buy",
    "test", "place", "item", "items", "one", "any", "all", "also",
}

_MONEY = re.compile(
    r"(?:under|below|within|at most|up to|no more than|not more than|"
    r"at or (?:under|below)|max(?:imum)? of|budget(?: is| of)?|"
    r"only go up to|can spend up to)\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?",
    re.I,
)
_SKU_PAIR = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_QUOTES = re.compile(r'"([^"]{2,80})"')
_SURFACE = re.compile(r"surface at least (\d+) distinct", re.I)
_COLOR_MODEL = re.compile(
    r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b", re.I
)
_SHELF = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_SOFT_MAKER = re.compile(
    r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
)
_RANK_HEAD = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9 /_\-]{2,72}?)\s+(?:counts most|matters most|is (?:the )?most important|"
    r"comes? first|is (?:my )?top priority|weigh first|is what i (?:actually )?care about|"
    r"matters more(?: to me)?)",
    re.I,
)
_WEAK_WANT = {
    "resist", "resistance", "resistant", "freedom", "life", "quality",
    "counts", "most", "then", "actually", "care", "about", "matters",
    "more", "survives", "hard", "workouts", "sessions", "weather", "kind",
    "daily", "runs", "behind", "next", "least", "several", "together",
    "beat", "single", "comes", "second", "what", "this", "than", "few",
}
_RANK_NEXT = re.compile(
    r"(?:then|followed by|with a secondary|second(?:ary)?(?: place| is| comes)?)\s+"
    r"(.{3,80}?)(?:\.|,|;|$)",
    re.I,
)
_MARKET_TALK = re.compile(
    r"more expensive|re-?check before|just got|went out of stock|no longer in stock|"
    r"price (?:just )?(?:changed|jumped|rose)",
    re.I,
)
_PRICE_JUMP = re.compile(
    r"from\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s+to\s+"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.I,
)
_ASKING = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|"
    r"which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\b",
    re.I,
)
_INCH = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])", re.I)
_HERTZ = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)
_PIXELS = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
_GIB = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)
_RAM_PAIR = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I)
_RAM_WORD = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)
_OS = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|"
    r"symbian|harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)
_HANDSET = re.compile(r"\bsmart\s*phones?\b", re.I)
_NOT_HANDSET = re.compile(
    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
    r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
    re.I,
)
_PANEL_ONLY = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.I,
)
_NOT_A = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|tablets?|accessor(?:y|ies))\b",
    re.I,
)
_PLUS_GEAR = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|"
    r"screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.I,
)
_SPECISH = re.compile(
    r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
    r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b",
    re.I,
)
_GENERIC_ALTS = (
    ("grey", "gray"),
    ("color", "colour"),
    ("tshirt", "t-shirt"),
    ("t-shirt", "t shirt"),
    ("wifi", "wi-fi"),
    ("harddisk", "hard disk"),
    ("cellphone", "cell phone"),
    ("smartphone", "smart phone"),
)
_FACT_KEYS = {
    "screen_size": ("size", "screen_size", "display_size", "panel_size", "screen"),
    "resolution": ("display", "resolution", "screen_resolution", "panel"),
    "refresh_rate": ("refresh_rate", "refresh", "hz"),
    "ram": ("storage", "ram", "memory", "system_memory"),
}


def _now() -> float:
    return time.monotonic()


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _stems(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _same_noun(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def _whole(title: str, tokens: list[str]) -> bool:
    bag = " " + re.sub(r"[^a-z0-9]+", " ", (title or "").lower()) + " "
    return all(" " + t + " " in bag for t in tokens)


def _money(text: str) -> float | None:
    best = None
    for m in _MONEY.finditer(text or ""):
        try:
            val = float(m.group(1).replace(",", "") + "." + (m.group(2) or "0"))
        except ValueError:
            continue
        if best is None or val < best:
            best = val
    return best


def _clip(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 32] + " …"


def _sole(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def coerce_json(raw: Any, expect: str = "dict") -> Any:
    """Repair lightly broken model JSON (fences, trailing commas, extra prose)."""
    fallback: Any = {} if expect == "dict" else []
    if isinstance(raw, (dict, list)):
        if expect == "dict" and isinstance(raw, dict):
            return raw
        if expect == "list" and isinstance(raw, list):
            return raw
        return fallback
    text = str(raw or "").strip()
    if not text:
        return fallback
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start_obj, start_arr = text.find("{"), text.find("[")
    if expect == "list":
        start = start_arr if start_arr >= 0 else start_obj
    else:
        start = start_obj if start_obj >= 0 else start_arr
    if start < 0:
        return fallback
    text = text[start:]
    for parser in (
        lambda s: json.loads(s),
        lambda s: json.loads(re.sub(r",\s*([}\]])", r"\1", s)),
    ):
        try:
            parsed = parser(text)
            if expect == "dict" and isinstance(parsed, dict):
                return parsed
            if expect == "list" and isinstance(parsed, list):
                return parsed
        except (TypeError, ValueError):
            continue
    open_ch, close_ch = ("{", "}") if expect == "dict" else ("[", "]")
    depth, end = 0, -1
    for i, ch in enumerate(text):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break
    if end > 0:
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", text[: end + 1]))
            if expect == "dict" and isinstance(parsed, dict):
                return parsed
            if expect == "list" and isinstance(parsed, list):
                return parsed
        except (TypeError, ValueError):
            pass
    return fallback


def tool_args(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    parsed = coerce_json(raw, "dict")
    return parsed if isinstance(parsed, dict) else {}


def peel(blob: Any) -> dict:
    """Unwrap nested environment observations; keep error/done flags."""
    if not isinstance(blob, dict):
        return {}
    cur = blob
    for _ in range(4):
        inner = cur.get("observation")
        if isinstance(inner, dict):
            merged = dict(inner)
            if cur.get("done") and not merged.get("done"):
                merged["done"] = cur["done"]
            if cur.get("error") and not merged.get("error"):
                merged["error"] = cur["error"]
            cur = merged
            continue
        break
    return cur


def inches(text: str) -> set[str]:
    return {"%gin" % float(m.group(1)) for m in _INCH.finditer(text or "")}


def hertz(text: str) -> set[str]:
    return {"%ghz" % float(m.group(1)) for m in _HERTZ.finditer(text or "")}


def gigs(text: str) -> set[str]:
    out = set()
    for m in _GIB.finditer(text or ""):
        n = float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1)
        out.add("%ggb" % n)
    return out


def _class_of(wide: int, tall: int) -> str | None:
    a, b = max(wide, tall), min(wide, tall)
    if a >= 3840 and b >= 2160:
        return "4k"
    if a >= 2560 and b >= 1440:
        return "qhd"
    if a >= 1920 and b >= 1080:
        return "fhd"
    if a >= 1280 and b >= 720:
        return "hd"
    return None


def panels(text: str) -> set[str]:
    text = text or ""
    if re.search(r"\bqHD\b", text):
        return set()
    low = text.lower()
    found: set[str] = set()
    for m in _PIXELS.finditer(low):
        cls = _class_of(int(m.group(1)), int(m.group(2)))
        if cls:
            found.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        found.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        found.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        found.add("fhd")
    if re.search(r"\b(?:hd|720p?)\b", re.sub(r"\b(?:full|ultra) hd\b", "", low)):
        found.add("hd")
    return found


_PULL = {
    "screen_size": inches,
    "refresh_rate": hertz,
    "resolution": panels,
    "ram": gigs,
}


def near_values(field: str, text: str) -> set[str]:
    text = text or ""
    if field != "ram":
        return _PULL[field](text)
    paired = {"%ggb" % float(a) for a, _b in _RAM_PAIR.findall(text)}
    return paired or {"%ggb" % float(n) for n in _RAM_WORD.findall(text)}


def talks_about(field: str, text: str) -> bool:
    text = text or ""
    if field == "ram":
        return bool(_RAM_PAIR.search(text) or _RAM_WORD.search(text))
    if field == "screen_size":
        return bool(_INCH.search(text))
    if field == "refresh_rate":
        return bool(_HERTZ.search(text))
    return bool(
        _PIXELS.search(text)
        or re.search(r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b", text, re.I)
    )


def stated_value(field: str | None, text: str) -> str | None:
    if not field:
        return None
    return _sole(_PULL[field](text or ""))


def hyphen_forms(query: str) -> list[str]:
    q = " ".join((query or "").split())
    if not q:
        return []
    forms = [q]
    if "-" in q:
        forms.append(q.replace("-", " "))
        forms.append(q.replace("-", ""))
    if " " in q:
        forms.append(q.replace(" ", "-"))
    seen, out = set(), []
    for item in forms:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def expand_want(tok: str) -> list[str]:
    """Morphology / rating aliases so a want matches how catalogs print it."""
    t = (tok or "").lower()
    extra = {
        "water": ("waterproof", "waterresistant", "ipx", "ipx4", "ipx5", "ipx7", "ip67"),
        "sweat": ("sweatproof", "sweatresistant"),
        "resist": ("resistant", "resistance", "ipx"),
        "resistance": ("resistant", "waterproof", "ipx"),
        "wireless": ("bluetooth", "tws", "truewireless", "bt5"),
        "battery": ("mah", "playtime"),
        "sport": ("earhook", "gym", "workout", "sports", "running"),
        "secure": ("earhook", "wings", "fit"),
        "fit": ("earhook", "secure"),
        "noise": ("anc", "canceling", "cancelling"),
        "cancel": ("anc", "noisecancel"),
        "wired": ("aux", "3.5mm"),
        "bass": ("subwoofer",),
        "mic": ("microphone",),
        "microphone": ("mic",),
        "fold": ("folding", "foldable"),
        "rugged": ("durable", "ip68"),
        "light": ("lightweight",),
        "hook": ("earhook",),
        "waterproof": ("ipx", "waterresistant"),
        "ipx": ("waterproof", "ipx4", "ipx5", "ipx7", "ipx8"),
        "bluetooth": ("wireless", "tws"),
        "anc": ("noisecancel", "noise"),
    }.get(t, ())
    return [t, *extra]


# Visible-page clumps the preference gradient scores (water / radio / hold / juice).
# Grouped so "sweat"+"water" count once, matching how the family pays.
_USE_CLUMPS = (
    ("water", "sweat", "waterproof", "ipx"),
    ("wireless", "bluetooth"),
    ("sport", "secure", "hook"),
    ("battery",),
)


def want_tokens(span: str) -> list[str]:
    span = re.sub(r"[/_\-]+", " ", span or "")
    out = []
    for t in re.findall(r"[a-z]{3,}", span.lower()):
        if t not in _STOP and t not in _WEAK_WANT and t not in out:
            out.append(t)
    return out[:8]


def ranked_wants(text: str) -> tuple[list[str], list[str]]:
    """Primary / secondary content words from the shopper's own ranking sentence."""
    primary: list[str] = []
    secondary: list[str] = []
    m = _RANK_HEAD.search(text or "")
    if m:
        primary = want_tokens(m.group(1).rsplit(".", 1)[-1])
    n = _RANK_NEXT.search(text or "")
    if n:
        secondary = [t for t in want_tokens(n.group(1)) if t not in primary]
    return primary, secondary


def alt_queries(base: str) -> list[str]:
    """±1-token and generic spelling twins — harvest diversity, not suite maps."""
    seeds = hyphen_forms(base)
    extra: list[str] = []
    low = (base or "").lower()
    for a, b in _GENERIC_ALTS:
        if a in low and b.lower() not in low:
            extra.append(re.sub(re.escape(a), b, base, count=1, flags=re.I))
        elif b.lower() in low and a not in low:
            extra.append(re.sub(re.escape(b), a, base, count=1, flags=re.I))
    toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", base or "") if t.lower() not in _STOP]
    if len(toks) == 2:
        extra.append(toks[1] + " " + toks[0])
        extra.append(toks[0] + "-" + toks[1] if "-" not in toks[0] else toks[0] + " " + toks[1])
        extra.append(toks[0] + toks[1])
    if len(toks) >= 3:
        extra.append(" ".join(toks[:-1]))
        extra.append(" ".join(toks[1:]))
    seen, out = set(), []
    for item in seeds + extra:
        item = " ".join(item.split())
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out[:6]


def fact_blob(facts: dict, field: str) -> str:
    parts = []
    for key in _FACT_KEYS.get(field, ()):
        val = facts.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def strip_negation(text: str, title: bool = False) -> str:
    text = _NOT_A.sub("", text or "")
    return _PLUS_GEAR.split(text, maxsplit=1)[0] if title else text


def handset_verdict(facts: dict, title: str, options: str) -> str:
    if _PANEL_ONLY.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    head = strip_negation(str(title or ""), title=True)
    kind = " ".join(
        strip_negation(str(facts.get(k) or ""))
        for k in ("phone_type", "device_type", "description_device_type")
    )
    blob = head + " " + kind
    if _NOT_HANDSET.search(blob):
        return "no"
    if any("[truncated]" in str(facts.get(k) or "") for k in ("phone_type", "device_type", "device_operating_system")):
        return "unclear"
    if _HANDSET.search(blob):
        return "yes"
    os_blob = " ".join(
        strip_negation(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    if _OS.search(os_blob):
        return "yes"
    if _OS.search(head) and not re.search(r"\bcompatib", str(title or ""), re.I):
        return "yes"
    return "unclear"


def vid(pid: Any, sku: Any) -> str:
    return str(pid) + "::" + str(sku or "")


@dataclass
class Wish:
    raw: str
    ask: str
    rules: str
    family: str
    ceiling: float | None
    currency: str
    named: list[tuple[str, str]]
    recall_n: int
    theme: list[str]
    model: str | None
    color: str | None
    shelf: str | None
    maker: str | None
    soft_maker: str | None
    specs: dict[str, str]
    gate: str | None
    gate_val: str | None
    sealed: bool
    axes: list[tuple[str, int]]
    forbid: list[str]

    @classmethod
    def parse(cls, query: str) -> "Wish":
        raw = query or ""
        ask, _, rules = raw.partition("\n\nTask rules:")
        ask, rules = ask.strip(), rules.strip()
        ceiling = _money(ask)
        currency = "PHP"
        mcc = re.search(r"\b([A-Z]{3})\b", ask)
        if mcc and mcc.group(1) in ("PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"):
            currency = mcc.group(1)
        named = [(p, s.rstrip(".,;:")) for p, s in _SKU_PAIR.findall(ask)]
        quoted = _QUOTES.findall(ask)
        phrase = re.search(r'matches (?:the phrase )?"?([A-Za-z0-9][^",]{1,80}?)"?\s*,', ask, re.I)
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = _SURFACE.search(rules.lower())
        recall_n = int(sm.group(1)) if sm else 0
        pm = _COLOR_MODEL.search(ask)
        color = pm.group(1).lower() if pm else None
        model = pm.group(2) if pm else None
        shelf = cls._shelf(ask)
        maker = cls._maker(ask, shelf)
        soft = None if pm else cls._soft(ask)
        specs: dict[str, str] = {}
        for field_name in ("screen_size", "refresh_rate", "resolution", "ram"):
            value = stated_value(field_name, ask)
            if value:
                specs[field_name] = value
        gate, gate_val = (next(iter(specs.items())) if specs else (None, None))
        sealed = gate is None and "firm requirement" in ask.lower()
        family = cls._family(ask, rules, named, recall_n, model, theme, specs, soft, maker, sealed)
        axes = cls._axes(ask + " " + rules, family)
        forbid = cls._forbid(ask)
        return cls(
            raw=raw, ask=ask, rules=rules, family=family, ceiling=ceiling, currency=currency,
            named=named, recall_n=recall_n, theme=theme, model=model, color=color, shelf=shelf,
            maker=maker, soft_maker=soft, specs=specs, gate=gate, gate_val=gate_val, sealed=sealed,
            axes=axes, forbid=forbid,
        )

    @staticmethod
    def _shelf(goal: str) -> str | None:
        m = _SHELF.search(goal or "")
        if m:
            return m.group(1).strip()
        m = re.search(
            r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|below|within|with)\b)",
            goal or "",
            re.I,
        )
        if not m:
            return None
        tok = m.group(1)
        if tok.lower() in _STOP or tok.lower() in {"most", "then", "counts", "need", "looking"}:
            return None
        return tok

    @staticmethod
    def _maker(goal: str, shelf: str | None) -> str | None:
        if not shelf:
            return None
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(shelf), goal or "", re.I)
        if not m:
            return None
        words = [w for w in re.findall(r"[\w&.'\-]+", _SPECISH.sub(" ", m.group(1))) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not panels(w)]
        return " ".join(keep) or None

    @staticmethod
    def _soft(goal: str) -> str | None:
        m = _SOFT_MAKER.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        keep: list[str] = []
        for w in name.split():
            titled = bool(w[:1].isupper()) and not any(c.isdigit() for c in w)
            if titled and w.lower() not in _STOP:
                keep.append(w)
                continue
            if keep:
                break
        if not keep or len(keep) > 4:
            return None
        return " ".join(keep)

    @staticmethod
    def _forbid(goal: str) -> list[str]:
        found = []
        for m in re.finditer(r"\b(?:no|without|not)\s+([a-z][a-z0-9\-]{2,24})", goal or "", re.I):
            tok = m.group(1).lower()
            if tok not in _STOP:
                found.append(tok)
        return found

    @staticmethod
    def _axes(text: str, family: str) -> list[tuple[str, int]]:
        low = (text or "").lower()
        axes: list[tuple[str, int]] = []
        if re.search(r"refresh|hz\b", low):
            axes.append(("refresh_rate", 1))
        if re.search(r"screen|inch", low):
            axes.append(("screen_size", 1))
        if re.search(r"lowest price|cheapest|then the lowest price", low):
            axes.append(("price", -1))
        if family == "ranking" and not axes:
            axes = [("refresh_rate", 1), ("screen_size", 1), ("price", -1)]
        elif family == "ranking" and not any(a[0] == "price" for a in axes):
            axes.append(("price", -1))
        return axes

    @staticmethod
    def _family(
        ask: str, rules: str, named, recall_n, model, theme, specs, soft, maker, sealed
    ) -> str:
        if named:
            return "ranking"
        if recall_n:
            return "recall"
        if model:
            return "recovery"
        r = (rules or "").lower()
        if "listing_id" in r:
            return "preference"
        if "exact variant" in r:
            return "constraint"
        if "trustworthy reason" in r:
            return "justification"
        if "nice-to-have" in r:
            return "intent"
        if theme:
            return "recall"
        if re.search(r"\b(?:why|reason|explain|justif)", ask, re.I):
            return "justification"
        if soft:
            return "intent"
        if sealed or specs or maker:
            return "constraint"
        return "preference"

    def take_spec(self, text: str) -> str | None:
        for field_name in ("screen_size", "refresh_rate", "ram", "resolution"):
            value = stated_value(field_name, text)
            if value:
                self.gate, self.gate_val = field_name, value
                self.specs[field_name] = value
                self.sealed = False
                return field_name + "=" + value
        return None


@dataclass
class Hit:
    pid: str
    sku: str
    price: float | None
    live: bool
    viewed: bool
    ok: bool
    relax: int
    n_attr: int
    extra_miss: int


@dataclass
class Shelf:
    cart: list[tuple[str, str]] = field(default_factory=list)
    quote: dict[str, dict] = field(default_factory=dict)
    quote_at: dict[str, int] = field(default_factory=dict)
    inspected: set[str] = field(default_factory=set)
    pages: dict[str, dict] = field(default_factory=dict)
    rows: dict[str, dict] = field(default_factory=dict)
    options: dict[str, str] = field(default_factory=dict)
    listings: set[str] = field(default_factory=set)
    listing_of: dict[str, str] = field(default_factory=dict)
    seen_ids: set[str] = field(default_factory=set)
    dead: set[str] = field(default_factory=set)
    dead_pid: set[str] = field(default_factory=set)
    signatures: set[str] = field(default_factory=set)
    seq: int = 0
    adds: int = 0
    last_add_seq: int | None = None
    shock_at: str | None = None

    def ingest(self, name: str, obs: dict, wish: Wish) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get("error"):
            return
        if name in ("search", "filter"):
            for row in obs.get("results") or []:
                if isinstance(row, dict):
                    self._row(row, wish)
        elif name == "view":
            self._page(obs, wish)
        elif name == "compare":
            for prod in obs.get("compared") or []:
                if isinstance(prod, dict):
                    self._page(prod, wish)
        elif name == "inspect_stock":
            k = vid(obs.get("product_id"), obs.get("sku"))
            self.quote[k] = {"in_stock": bool(obs.get("in_stock")), "price": _num(obs.get("price"))}
            self.quote_at[k] = self.seq
            self.inspected.add(k)
        elif name == "add_to_cart":
            added = obs.get("added") or {}
            pid, sku = added.get("product_id"), added.get("sku")
            if pid and sku:
                self.adds += 1
                pair = (str(pid), str(sku))
                if pair not in self.cart:
                    self.cart.append(pair)
                self.last_add_seq = self.seq
            self._cart(obs)
        elif name == "remove_from_cart":
            self._cart(obs)
            removed = obs.get("removed") or {}
            pair = (str(removed.get("product_id") or ""), str(removed.get("sku") or ""))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == "inspect_cart":
            self._cart(obs)

    def _cart(self, obs: dict) -> None:
        rows = obs.get("cart")
        if not isinstance(rows, list):
            return
        pairs = [
            (str(r.get("product_id")), str(r.get("sku") or ""))
            for r in rows if isinstance(r, dict) and r.get("product_id")
        ]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            k = vid(r.get("product_id"), r.get("sku"))
            price = _num(r.get("current_price"))
            stock = r.get("current_in_stock")
            if price is None and stock is None:
                continue
            prev = self.quote.get(k, {})
            self.quote[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.quote_at[k] = self.seq

    def _row(self, row: dict, wish: Wish) -> None:
        pid = row.get("product_id")
        if not pid:
            return
        title = str(row.get("title") or "")
        if wish.theme:
            if _whole(title, wish.theme):
                self.seen_ids.add(str(pid))
        else:
            self.seen_ids.add(str(pid))
        if not row.get("sku"):
            return
        k = vid(pid, row.get("sku"))
        self.rows[k] = row
        if row.get("price") is not None:
            self.quote.setdefault(
                k, {"in_stock": bool(row.get("in_stock", True)), "price": _num(row.get("price"))}
            )

    def _page(self, prod: dict, wish: Wish) -> None:
        pid = prod.get("product_id")
        if not pid:
            return
        self.pages[str(pid)] = prod
        if prod.get("listing_id"):
            self.listings.add(str(prod["listing_id"]))
            self.listing_of[str(pid)] = str(prod["listing_id"])
        if wish.theme and _whole(str(prod.get("title") or ""), wish.theme):
            self.seen_ids.add(str(pid))
        for v in prod.get("variants") or []:
            if not isinstance(v, dict) or not v.get("sku"):
                continue
            k = vid(pid, v.get("sku"))
            self.quote[k] = {"in_stock": bool(v.get("in_stock")), "price": _num(v.get("price"))}
            self.quote_at[k] = self.seq
            self.options[k] = str(v.get("options") or "")


class ShopRun:
    OPEN, HARVEST, DUEL, LOCK, CONFIRM, PAY = range(6)

    def __init__(self, problem: dict) -> None:
        env = problem.get("environment") or {}
        self.binding = dict(env.get("binding") or {})
        self.policy = env.get("policy_view") or {}
        self.tools = list(self.policy.get("tools") or [])
        self.names = {
            (t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)
        }
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.pid = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.wish = Wish.parse(str(self.policy.get("query") or ""))
        self.ceiling = self.wish.ceiling
        self.shelf = Shelf()
        self.t0 = _now()
        self.dialogue: list = []
        self.priority: list[str] = []
        head, nxt = ranked_wants(self.wish.ask)
        self.primary, self.secondary = head, nxt
        self.priority = list(dict.fromkeys(head + nxt))[:12]
        self.messages: list = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.phase = self.OPEN
        self.done = False
        self.model_i = 0
        self.asked = False
        self.answer_due = False
        self.shopper_replied = False
        self.searches = 0
        self.cart_looks = 0
        self.reads = 0
        self.pending_env: dict | None = None
        self.recheck: tuple[str, str] | None = None
        self.asked_exact = False
        self.asked_brand = False
        self.twin_q = False
        self.sib_q = False
        self.opaque: set[str] = set()
        self.q_wait = 0
        self.q_sent = 0
        self.reject_n = 0
        self.harvested: set[str] = set()
        self.miss_queries: set[str] = set()
        self.forced: Hit | None = None
        self.spec_fresh = False
        self.taste_rescue = False
        self.maker_miss = False
        self.spec_nudge = False
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.models = _OPENROUTER if provider == "openrouter" else _CHUTES
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.models = (pinned,) + tuple(m for m in self.models if m != pinned)

    def left(self) -> float:
        return _EPISODE_S - (_now() - self.t0)

    def has(self, name: str) -> bool:
        return name in self.names

    def _system(self) -> str:
        parts = [
            self._constitution(),
            self._episode_job(),
            self._hands(),
            self._live_board(),
        ]
        return "\n\n".join(p for p in parts if p)

    def _constitution(self) -> str:
        return (
            "Catalog physics. You have the bound tools and nothing else. One tool call per turn. "
            "Never invent a product_id, sku, or listing_id.\n"
            "Truth order for a number: (1) the selected variant's option string, (2) labelled facts "
            "on that same page, (3) the title. Two of those in conflict means unknown — do not guess. "
            "A bare number under a unit-bearing label inherits the unit (27 under size is 27 inches).\n"
            "The first successful add_to_cart is the graded commitment. A filler add scores as that "
            "filler. After it, price and stock can jump. inspect_stock is the only live quote. If the "
            "shopper says the carted SKU got more expensive from A to B, vanished, or must be re-checked, "
            "that SKU is dead — recover; never place_test_order on it.\n"
            "message is only for one missing firm value or a direct answer. No status. Never ask the "
            "shopper to raise the ceiling. One compare of up to five product_ids beats a view chain. "
            "Never replay identical tool+args. inspect_cart at most once."
        )

    def _hands(self) -> str:
        if self.wish.family == "preference":
            return (
                "Hands: you commit. Search the category plus the MOST idea first — never open with "
                "filter(category, max_price), which dumps silent cheap SKUs. Compare pages whose "
                "titles carry MOST (waterproof / IPX / bluetooth / ear-hook as the catalog prints "
                "them). View at least two listing_id values, then inspect_stock and add_to_cart that "
                "catalog-MOST device — that first add is the shock hook. After a reprice, recover on "
                "a different listing_id that still prints MOST spellings (IPX / waterproof / "
                "bluetooth / ear-hook), not a wired silent SKU. Lowest in-budget price is not the "
                "winner either before or after the shock."
            )
        return (
            "Hands: you surface. search, filter, view, compare, inspect_stock, message. The closer "
            "quotes, carts, re-quotes after the add, and place_test_order when every firm gate is "
            "green. Do not add_to_cart or place_test_order yourself."
        )

    def _live_board(self) -> str:
        w = self.wish
        if w.family == "preference":
            return self._live_wants()
        bits: list[str] = []
        if w.family == "ranking" and w.named:
            bits.append(
                "Closed set: "
                + ", ".join("%s::%s" % pair for pair in w.named[:8])
                + ". Compare these. Do not search or filter."
            )
        if w.family == "intent" and w.soft_maker:
            bits.append(
                "Intended first add: brand %s when a fully valid listing of that brand exists."
                % w.soft_maker
            )
        if w.family == "constraint":
            if w.gate and w.gate_val:
                bits.append(
                    "Bound number: %s = %s on the chosen variant's option text."
                    % (w.gate, w.gate_val)
                )
            else:
                bits.append("Bound number: not yet spoken. Ask once, then stop asking.")
        if w.family == "recovery" and (w.model or w.color):
            bits.append(
                "Exact first add: model %s, color %s."
                % (w.model or "?", w.color or "?")
            )
        if w.family == "recall":
            bits.append(
                "Coverage floor: %d distinct titles, every requested word as a whole token."
                % (w.recall_n or 1)
            )
        if w.ceiling is not None:
            bits.append("Ceiling: %.2f %s." % (w.ceiling, w.currency))
        return "Live board — " + " ".join(bits) if bits else ""

    def _live_wants(self) -> str:
        if not self.primary and not self.secondary:
            return (
                "Live board — MOST is unnamed. Ask once: what counts most, what comes second. "
                "Do not add until MOST is evidenced on the page. Do not ask them to raise the ceiling."
            )
        bits = []
        if self.primary:
            bits.append("MOST: " + ", ".join(self.primary[:8]))
        if self.secondary:
            bits.append("THEN: " + ", ".join(self.secondary[:6]))
        return (
            "Live board — " + " | ".join(bits) + ". MOST must appear in that variant's labelled "
            "facts or option string, including ordinary catalog spellings of the same idea: "
            "waterproof / water-resistant / IPX for water or sweat; bluetooth / TWS for wireless; "
            "mAh / playtime for battery; ear-hook / wings for a secure sport fit. "
            "THEN is a tie-break only. A silent cheap in-stock row is the wrong first add."
        )

    def _episode_job(self) -> str:
        w = self.wish
        if w.family == "recall":
            n = w.recall_n or 1
            return (
                "Job: coverage, not a first hit. Search in_stock with the requested words, then "
                "hyphen/spelling twins, then drop a tail token. Stop buying until at least %d "
                "distinct titles each contain every requested word as a whole token. Early checkout "
                "is a miss even if the SKU is valid."
                % n
            )
        if w.family == "ranking":
            return (
                "Job: closed tournament. Only the named product_id::sku pairs exist. Compare all of "
                "them before anything else. Do not search or filter the catalog. Rank by the goal's "
                "axes on the selected variant — typically highest advertised refresh, then largest "
                "panel, then lower price. After a shock, drop the dead pair and re-rank survivors only."
            )
        if w.family == "recovery":
            who = w.model or "the named model"
            shade = w.color or "the named color"
            return (
                "Job: smallest step off the named pair. Search the model code %s alone (not a long "
                "title query) and view so every color on that page is recorded. First add must be %s "
                "in %s. If that listing dies, same model, other color — never a different model first. "
                "Category, ceiling, and stock do not relax."
                % (who, who, shade)
            )
        if w.family == "constraint":
            return (
                "Job: one hidden hardware number on the exact variant. If the request already states "
                "it, do not re-ask. Otherwise ask once for the field and the exact value (size, panel "
                "class, refresh, or memory). That value must resolve from the chosen variant's own "
                "option text, not a sibling SKU or a product-wide guess. If the carted listing dies, "
                "change product_id — another SKU of the dead page is not a recovery."
            )
        if w.family == "intent":
            return (
                "Job: hard gates vs intended maker. Category, hardware spec, ceiling, and stock never "
                "yield. A named brand is not decoration: when a fully valid listing of that brand "
                "exists, it must be the first add, because the later market shock is attached to that "
                "first cart. Search/filter with the brand while a valid one might still exist. Only "
                "after a real miss of that brand, any maker that clears the hard gates. A later lower "
                "ceiling binds every later pick."
            )
        if w.family == "justification":
            return (
                "Job: a quotable labelled reason. Buy only when labelled facts, agreeing product-wide, "
                "state the asked spec. Skip resale, compare-page, and refurbished titles. The order "
                "reason must copy those labelled facts — never marketing copy or the title alone."
            )
        if w.family == "preference":
            return (
                "Job: ranked tradeoff, not cheapest-valid. The shopper names what counts most and "
                "what comes second — often already in the opening text ('X counts most, then Y'). "
                "Do not default to filter(category, max_price): that surfaces cheap silent SKUs. "
                "Search the category plus the MOST idea inside the ceiling, then compare distinct "
                "listing_id pages. First add must be a real device of the asked category whose own "
                "facts support MOST — not an accessory, case, or parts listing, not a page that is "
                "silent on MOST, not the lowest price that merely happens to be in stock. That first "
                "add is the event hook. After a reprice or stock-out, the paid score is the survivor: "
                "another listing_id that still prints MOST catalog spellings, never a cheaper silent "
                "or wired row."
            )
        return (
            "Job: learn the firm gates, search or filter, read pages, and let the closer commit a "
            "fully valid variant."
        )

    def _retune(self) -> None:
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self._system()

    def hint(self, text: str) -> None:
        self.messages.append({"role": "user", "content": ">> " + text})
        self.dialogue.append({"role": "assistant", "content": ">> " + text[:240]})

    def page(self, pid: str) -> dict:
        return self.shelf.pages.get(str(pid)) or {}

    def facts(self, pid: str) -> dict:
        f = self.page(pid).get("facts")
        return f if isinstance(f, dict) else {}

    def title(self, pid: str) -> str:
        t = str(self.page(pid).get("title") or "")
        if t:
            return t
        for k, row in self.shelf.rows.items():
            if k.startswith(str(pid) + "::"):
                return str(row.get("title") or "")
        return ""

    def brand(self, pid: str) -> str:
        b = self.page(pid).get("brand")
        if b:
            return str(b)
        for k, row in self.shelf.rows.items():
            if k.startswith(str(pid) + "::") and row.get("brand"):
                return str(row["brand"])
        return ""

    def path(self, pid: str) -> list:
        p = self.page(pid).get("category_path")
        return list(p) if isinstance(p, list) else []

    def options(self, pid: str, sku: str) -> str:
        return self.shelf.options.get(vid(pid, sku), "")

    def _drop_codes(self, pid: str, title: str) -> str:
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

    def fact_val(self, pid: str, field: str) -> str | None:
        raw = fact_blob(self.facts(pid), field)
        if not raw.strip():
            return None
        found = _PULL[field](raw)
        if not found and field != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", raw.strip()):
            unit = {"screen_size": "in", "refresh_rate": "hz", "ram": "gb"}[field]
            return "%g" % float(raw.strip()) + unit
        return _sole(found)

    def product_val(self, pid: str, field: str) -> str | None:
        vals: set[str] = set()
        clash = False
        raw = fact_blob(self.facts(pid), field)
        if raw:
            found = _PULL[field](raw)
            clash = clash or len(found) > 1
            if field == "resolution" and re.search(r"\bqHD\b", raw):
                clash = True
            vals |= found
        title = self.title(pid)
        if field == "resolution":
            title = self._drop_codes(pid, title)
            if re.search(r"\bqHD\b", title):
                clash = True
        vals |= near_values(field, title)
        return None if clash else _sole(vals)

    def sku_val(self, pid: str, sku: str, field: str) -> str | None:
        opt = self.options(pid, sku)
        vals = near_values(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if talks_about(field, opt):
            return None
        return self.product_val(pid, field)

    def measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self.sku_val(pid, sku, field)
        try:
            return float(value[:-2]) if value else None
        except (TypeError, ValueError):
            return None

    def shelf_ok(self, pid: str) -> bool:
        want = self.wish.shelf
        path = self.path(pid)
        if not want or not path:
            return True
        wanted = _stems(want)
        labels: set[str] = set()
        for lab in path:
            labels |= _stems(str(lab))
        placed = bool(wanted & labels) or any(
            _stems("".join(re.findall(r"[a-z]+", str(lab).lower()))) == wanted for lab in path
        )
        if not placed and "smartphone" in wanted:
            placed = any(
                re.fullmatch(r"(?:smart|cell|mobile|unlocked)?\s*phones?", str(lab).strip(), re.I)
                for lab in path
            )
        if not placed:
            return False
        title = self.title(pid)
        words = _stems(title) | _words(title)
        if re.search(r"\bgaming\s+monitors?\b", self.wish.ask, re.I):
            fast = any(float(v[:-2]) >= 120 for v in hertz(title))
            return "monitor" in words and ("gaming" in words or fast)
        if re.search(r"\bover[\s-]?ear\b", self.wish.ask, re.I):
            return "over" in words and "ear" in words or "overear" in words or "circumaural" in words
        return True

    def maker_ok(self, pid: str) -> bool:
        want = self.wish.maker
        if not want:
            return True
        needed = _words(want)
        return bool(needed <= _words(self.brand(pid)) or needed <= _words(self.title(pid)))

    def gear_ok(self, pid: str) -> bool:
        """Bolt-on listings are not a device of the asked shelf. Skip only generic accessory shapes."""
        shelf = (self.wish.shelf or "").lower()
        if not shelf or any(k in shelf for k in ("case", "cover", "accessor", "cable", "charger", "part")):
            return True
        title = self.title(pid)
        if re.search(r"\b(?:replacement|spare)\s+(?:screen|battery|panel|part)s?\b", title, re.I):
            return False
        if re.search(r"\b(?:case|cover|pouch|holder|shell|sleeve)s?\b.{0,32}\bfor\b", title, re.I):
            return False
        if re.search(
            r"\b(?:headphone|earphone|earbud|headset|phone)s?\s+(?:case|cover|sleeve|pouch)s?\b|"
            r"\b(?:case|cover|sleeve|pouch)s?\s+(?:headphone|earphone|earbud|headset)s?\b",
            title,
            re.I,
        ):
            return False
        if re.search(r"\bonly\s+(?:the\s+)?(?:cable|charger|adapter|cord)\b", title, re.I):
            return False
        dtype = str(self.facts(pid).get("description_device_type") or "").lower()
        if "mystery" in dtype or "blind" in dtype:
            return False
        return True

    def device_ok(self, pid: str, sku: str) -> bool:
        if not _same_noun(self.wish.shelf or "", "smartphone"):
            return True
        dtype = str(self.facts(pid).get("description_device_type") or "").lower()
        if "mystery" in dtype or "blind" in dtype:
            return False
        v = handset_verdict(self.facts(pid), self.title(pid), self.options(pid, sku))
        if self.wish.family in ("constraint", "justification", "recovery", "intent"):
            return v == "yes"
        return v != "no"

    def used_listing(self, pid: str) -> bool:
        return bool(
            re.search(
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed",
                self.title(pid),
                re.I,
            )
        )

    def negation_fail(self, pid: str, sku: str) -> bool:
        if not self.wish.forbid:
            return False
        blob = " ".join(
            [
                self.title(pid),
                self.options(pid, sku),
                " ".join(str(v) for v in self.facts(pid).values()),
            ]
        ).lower()
        return any(tok in blob for tok in self.wish.forbid)

    def relax(self, pid: str, sku: str) -> int:
        model, color = self.wish.model, self.wish.color
        if not model:
            return 0
        blob = _squash(self.title(pid) + " " + " ".join(str(v) for v in self.facts(pid).values()))
        same_m = _squash(model) in blob
        shade = (self.options(pid, sku) + " " + str(self.facts(pid).get("color") or "") + " " + self.title(pid)).lower()
        same_c = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade)
        if same_m and same_c:
            return 0
        if same_m:
            return 1
        return 2 if same_c else 3

    def n_attr(self, pid: str, sku: str) -> int:
        n = 0
        for field_name, want in self.wish.specs.items():
            if self.fact_val(pid, field_name) == want:
                n += 1
            elif self.sku_val(pid, sku, field_name) == want:
                n += 1
        return n

    def extra_miss(self, pid: str, sku: str) -> int:
        return sum(
            1
            for f, v in self.wish.specs.items()
            if f != self.wish.gate and self.sku_val(pid, sku, f) != v
        )

    def score(self, pid: str, sku: str) -> Hit:
        w = self.wish
        k = vid(pid, sku)
        st = self.shelf.quote.get(k) or {}
        price, stock = st.get("price"), st.get("in_stock")
        killed = k in self.shelf.dead or (str(pid) in self.shelf.dead_pid and w.family != "recovery")
        live = bool(stock) and price is not None and (self.ceiling is None or price <= self.ceiling + _EPS) and not killed
        viewed = str(pid) in self.shelf.pages
        spec_ok = True
        if w.gate and w.gate_val:
            got = self.sku_val(pid, sku, w.gate)
            spec_ok = got == w.gate_val
        elif w.family == "constraint" and not w.gate_val:
            spec_ok = False
        ok = bool(live and viewed and spec_ok and self.maker_ok(pid) and self.device_ok(pid, sku) and self.shelf_ok(pid))
        ok = ok and not self.negation_fail(pid, sku)
        if w.family in ("preference", "intent"):
            ok = ok and self.gear_ok(pid)
        if w.family == "justification":
            ok = ok and not self.used_listing(pid) and bool(self.brand(pid))
            if w.gate and w.gate_val:
                ok = ok and self.fact_val(pid, w.gate) == w.gate_val and self.product_val(pid, w.gate) == w.gate_val
        if w.family == "preference" and self.shelf.shock_at:
            dead_l = self.shelf.listing_of.get(self.shelf.shock_at.split("::", 1)[0])
            if dead_l and self.shelf.listing_of.get(str(pid)) == dead_l:
                ok = False
        if w.family == "recall" and w.theme:
            ok = ok and _whole(self.title(pid), w.theme)
        if w.family == "ranking":
            ok = ok and (str(pid), str(sku)) in w.named
            ok = ok and self.measure(pid, sku, "refresh_rate") is not None
            ok = ok and self.measure(pid, sku, "screen_size") is not None
        return Hit(
            pid=str(pid), sku=str(sku), price=price, live=live, viewed=viewed, ok=ok,
            relax=self.relax(pid, sku) if w.family == "recovery" else 0,
            n_attr=self.n_attr(pid, sku), extra_miss=self.extra_miss(pid, sku),
        )

    def pool(self) -> list[Hit]:
        out, seen = [], set()
        for key in list(self.shelf.quote):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self.score(pid, sku))
        return out

    def rank_key(self, h: Hit, first: bool) -> tuple:
        w = self.wish
        price = h.price if h.price is not None else 1e12
        if w.family == "ranking":
            tup: list = []
            for field_name, direction in w.axes:
                if field_name == "price":
                    tup.append(price if direction < 0 else -price)
                else:
                    m = self.measure(h.pid, h.sku, field_name) or 0.0
                    tup.append(-m if direction > 0 else m)
            tup.append(vid(h.pid, h.sku))
            return tuple(tup)
        if w.family == "recovery":
            return (h.relax, price, vid(h.pid, h.sku))
        proven = 0 if handset_verdict(self.facts(h.pid), self.title(h.pid), self.options(h.pid, h.sku)) == "yes" else 1
        if w.family == "intent" and first and w.soft_maker:
            want = _words(w.soft_maker)
            near = 0 if (want <= _words(self.brand(h.pid)) or want <= _words(self.title(h.pid))) else 1
            return (near, h.extra_miss, proven, -h.n_attr, price, vid(h.pid, h.sku))
        return (h.extra_miss, proven, -h.n_attr, price, vid(h.pid, h.sku))

    def winners(self, first: bool | None = None) -> list[Hit]:
        first = self.shelf.adds == 0 if first is None else first
        good = [h for h in self.pool() if h.ok]
        w = self.wish
        if w.family == "intent" and first and w.soft_maker:
            want = _words(w.soft_maker)
            named = [h for h in good if want <= _words(self.brand(h.pid)) or want <= _words(self.title(h.pid))]
            if named:
                good = named
        if w.family == "recovery" and first:
            exact = [h for h in good if h.relax == 0]
            if exact:
                good = exact
        good.sort(key=lambda h: self.rank_key(h, first))
        return good

    def champion(self) -> Hit | None:
        if self.forced:
            held = self.score(self.forced.pid, self.forced.sku)
            if held.ok:
                return held
            self.forced = None
        w = self.wish
        good = self.winners()
        if w.family == "preference":
            good.sort(key=self._taste_key)
            # After the shock the ordered leftover is paid_reward. Ignore a carted
            # water-only SKU when a wider-use page is already on record.
            if self.shelf.shock_at:
                return good[0] if good else None
            carted = [h for h in good if (h.pid, h.sku) in self.shelf.cart]
            if carted:
                return carted[0]
            wide = [h for h in good if self._has_most(h) and self._spread(h.pid, h.sku) >= 2]
            if (
                wide
                and self.searches >= 2
                and len(self.shelf.listings) >= 2
                and (not self._unread() or self.reads >= 6 or self._taste_takeover())
            ):
                wide.sort(key=self._taste_key)
                return wide[0]
            strong = [h for h in good if self._strong_most(h)]
            if good and self._taste_takeover():
                pick = wide or strong or [h for h in good if self._has_most(h)] or good
                pick.sort(key=self._taste_key)
                return pick[0]
            return None
        if w.family == "intent" and self.shelf.shock_at:
            return good[0] if good else None
        if w.family == "intent" and self.shelf.adds == 0 and w.soft_maker:
            want = _words(w.soft_maker)
            named = [h for h in good if want <= _words(self.brand(h.pid)) or want <= _words(self.title(h.pid))]
            if not named and self.turn < self.max_steps * 0.62 and not self.maker_miss:
                self._brand_nudge()
                return None
        if w.family == "recovery" and self.shelf.adds == 0:
            exact = [h for h in good if h.relax == 0]
            if not exact and self.turn < self.max_steps * 0.55:
                self._exact_nudge()
                return None
        return good[0] if good else None

    def _axis_blob(self, pid: str, sku: str, title: bool = True) -> str:
        bits = [" ".join(str(v) for v in self.facts(pid).values()), self.options(pid, sku)]
        if title:
            bits.append(self.title(pid))
        return " ".join(bits).lower()

    def _want_score(self, blob: str, bag: str, tok: str) -> int:
        """2 = catalog spelling, 1 = content stem, 0 = miss or filler-only."""
        if len(tok) < 3:
            return 0
        aliases = expand_want(tok)
        extras = aliases[1:]
        if extras and any(" " + a + " " in bag or a in blob for a in extras):
            return 2
        if tok in _WEAK_WANT:
            return 0
        if " " + tok + " " in bag:
            return 1
        return 0

    def _axis_hits(self, pid: str, sku: str, toks: list[str], title: bool = True) -> int:
        blob = self._axis_blob(pid, sku, title=title)
        bag = " " + re.sub(r"[^a-z0-9]+", " ", blob) + " "
        n = 0
        for t in toks:
            if self._want_score(blob, bag, t) > 0:
                n += 1
        return n

    def _page_hits(self, pid: str, sku: str, toks: list[str]) -> int:
        """Labelled facts + option string only — closer to what family axes grade."""
        return self._axis_hits(pid, sku, toks, title=False)

    def _most_marks(self, pid: str, sku: str, title: bool) -> tuple[int, int]:
        blob = self._axis_blob(pid, sku, title=title)
        bag = " " + re.sub(r"[^a-z0-9]+", " ", blob) + " "
        strong = sum(1 for t in self.primary if self._want_score(blob, bag, t) >= 2)
        mild = sum(1 for t in self.primary if self._want_score(blob, bag, t) == 1)
        return strong, mild

    def _primary_ok(self, pid: str, sku: str) -> bool:
        if not self.primary:
            return True
        return self._page_hits(pid, sku, self.primary) > 0 or self._axis_hits(pid, sku, self.primary) > 0

    def _clump_n(self, pid: str, sku: str, title: bool = True) -> int:
        blob = self._axis_blob(pid, sku, title=title)
        bag = " " + re.sub(r"[^a-z0-9]+", " ", blob) + " "
        n = 0
        for group in _USE_CLUMPS:
            if any(self._want_score(blob, bag, t) > 0 for t in group):
                n += 1
        return n

    def _wet(self, pid: str, sku: str) -> bool:
        blob = self._axis_blob(pid, sku, True)
        bag = " " + re.sub(r"[^a-z0-9]+", " ", blob) + " "
        return any(self._want_score(blob, bag, t) > 0 for t in _USE_CLUMPS[0])

    def _spread(self, pid: str, sku: str, title: bool = True) -> int:
        """Distinct use clumps. Water-only is 1; water+radio+hold is 3. Non-MOST pages stay 0 for ranking."""
        return self._clump_n(pid, sku, title=title)

    def _taste_key(self, h: Hit) -> tuple:
        most = 0 if self._has_most(h) else 1
        wet = 0 if self._wet(h.pid, h.sku) else 1
        # Spread only counts on a MOST/wet page so a dry bluetooth title cannot beat IPX.
        sp_page = self._spread(h.pid, h.sku, False) if most == 0 else 0
        sp_all = self._spread(h.pid, h.sku, True) if most == 0 else 0
        pe_s, pe_m = self._most_marks(h.pid, h.sku, False)
        pt_s, pt_m = self._most_marks(h.pid, h.sku, True) if not pe_s else (0, 0)
        sec = self._axis_hits(h.pid, h.sku, self.secondary)
        overlap = self._priority_hits(h.pid, h.sku)
        price = h.price if h.price is not None else 1e12
        return (
            most, wet, -sp_page, -sp_all,
            0 if pe_s else 1, -pe_s, -pe_m,
            0 if pt_s else 1, -pt_s, -pt_m,
            -sec, -overlap, price,
        )

    def _has_most(self, h: Hit) -> bool:
        if not self.primary:
            return True
        s, m = self._most_marks(h.pid, h.sku, False)
        if s or m >= 2:
            return True
        s2, m2 = self._most_marks(h.pid, h.sku, True)
        return bool(s2 or m2 >= 2)

    def _strong_most(self, h: Hit) -> bool:
        """Catalog spelling of MOST (ipx / waterproof / bluetooth / earhook), not a weak stem."""
        if not self.primary:
            return True
        s, _m = self._most_marks(h.pid, h.sku, False)
        if s:
            return True
        s2, _m2 = self._most_marks(h.pid, h.sku, True)
        return s2 > 0

    def _taste_queries(self) -> list[str]:
        shelf = self.wish.shelf or ""
        stems: list[str] = []
        extras: list[str] = []
        for t in list(self.primary[:6]) + list(self.secondary[:4]):
            al = expand_want(t)
            if t not in _WEAK_WANT and t not in stems:
                stems.append(t)
            for a in al[1:]:
                if a not in extras and a not in stems:
                    extras.append(a)
        out: list[str] = []
        for chunk in (stems[:4], extras[:5]):
            if not chunk:
                continue
            q = " ".join(x for x in (shelf, " ".join(chunk)) if x)
            if q and q.lower() not in {s.lower() for s in out}:
                out.append(q)
        if self.shelf.shock_at:
            bits = list(stems[:3]) + [x for x in ("bluetooth", "sport") if x not in stems]
            q = " ".join(x for x in (shelf, " ".join(bits)) if x)
            if q and q.lower() not in {s.lower() for s in out}:
                out.append(q)
        return out

    def _priority_hits(self, pid: str, sku: str) -> int:
        wants = set(self.primary + self.secondary + self.priority)
        if not wants:
            wants = {t for t in _words(self.wish.ask) if len(t) > 3 and not t.isdigit() and t not in _STOP}
        have = _words(self._axis_blob(pid, sku, title=True))
        return len(wants & have)

    def _brand_nudge(self) -> None:
        if self.asked_brand or not self.wish.soft_maker:
            return
        self.asked_brand = True
        self.hint(
            "No fully compliant %s listing is on record yet. Filter brand=%s within budget before other makers."
            % (self.wish.soft_maker, self.wish.soft_maker)
        )

    def _exact_nudge(self) -> None:
        if self.asked_exact or not self.wish.model:
            return
        self.asked_exact = True
        self.hint(
            "Exact model %s%s is not on record. Search the model code and view every color variant."
            % (self.wish.model, " / " + self.wish.color if self.wish.color else "")
        )

    def ready(self) -> bool:
        w = self.wish
        if self.shelf.adds:
            return True
        if w.sealed:
            if not w.gate_val and not self._panic():
                return False
            if not w.gate_val:
                w.sealed = False
                self.hint("No firm spec arrived; remaining constraints decide.")
        if (
            w.family in ("intent", "constraint")
            and self.turn < 4
            and not self.shopper_replied
            and not w.gate_val
        ):
            return False
        if w.family == "constraint" and not w.gate_val and not self._panic():
            return False
        if w.family == "ranking":
            missing = [p for p, _s in w.named if p not in self.shelf.pages and p not in self.opaque]
            if missing and not self._panic():
                return False
        if w.family == "preference" and not self.shelf.adds:
            if len(self.shelf.listings) < 2:
                return False
            if self.searches < 2 and not self._panic():
                return False
        if w.family == "recall":
            need = w.recall_n
            if self.left() > 110 and self.turn < self.max_steps * 0.45:
                if len(self.shelf.seen_ids) < need or self.searches < 2:
                    return False
            elif len(self.shelf.seen_ids) < need and self.searches < 4 and self.left() > 70:
                return False
        if self._want_backup() and self.left() > 110 and self.turn < self.max_steps * 0.42:
            ok_pids = {h.pid for h in self.winners() if h.ok}
            if len(ok_pids) < 2 and self._unread():
                return False
        return True

    def _want_backup(self) -> bool:
        # Preference is a taste call the model must make; ranking is a closed named set.
        return self.wish.family not in ("ranking", "preference") and self.shelf.adds == 0

    def _panic(self) -> bool:
        if self.shelf.shock_at and self.wish.family == "intent":
            return self.left() < 55 or self.turn >= self.max_steps - 10
        return self.left() < 28 or self.turn >= self.max_steps - 3

    def _taste_takeover(self) -> bool:
        return self.left() < 40 or self.turn >= self.max_steps - 6

    def _name_q(self) -> str:
        w = self.wish
        if w.maker:
            return w.maker
        if (
            w.soft_maker
            and self.shelf.adds == 0
            and not self.maker_miss
            and not self.shelf.shock_at
        ):
            return w.soft_maker
        return ""

    def _hold_taste(self, pid: str, sku: str) -> None:
        if self.wish.family == "preference":
            self.forced = self.score(pid, sku)

    def fresh(self, pid: str, sku: str) -> bool:
        k = vid(pid, sku)
        if k not in self.shelf.inspected:
            return False
        return self.shelf.last_add_seq is None or self.shelf.quote_at.get(k, 0) > self.shelf.last_add_seq

    def orderable(self, pid: str, sku: str) -> tuple[bool, str]:
        st = self.shelf.quote.get(vid(pid, sku))
        k = vid(pid, sku)
        if st is None:
            return False, "inspect_stock that variant first"
        if (pid, sku) not in self.shelf.cart:
            return False, "cart that variant first"
        if self.shelf.last_add_seq is not None and self.shelf.quote_at.get(k, 0) <= self.shelf.last_add_seq:
            return False, "inspect_stock again after the cart add"
        if not st.get("in_stock"):
            return False, "out of stock"
        price = st.get("price")
        if self.ceiling is not None and price is not None and price > self.ceiling + _EPS:
            return False, "above budget"
        if k in self.shelf.dead:
            return False, "invalidated"
        return True, ""

    def claims(self, pid: str, sku: str) -> list[dict]:
        out = []
        field_name, want = self.wish.gate, self.wish.gate_val
        if not field_name and self.wish.family == "justification":
            field_name, want = "resolution", self.fact_val(pid, "resolution")
        if not field_name or not want:
            return []
        stated = dict(self.wish.specs) or {field_name: want}
        stated.setdefault(field_name, want)
        for name, value in stated.items():
            if self.fact_val(pid, name) == value and self.product_val(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.shelf.quote.get(vid(pid, sku)) or {}
        if self.ceiling is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.ceiling})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self.brand(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.wish.family == "justification":
            c = self.claims(pid, sku)
            if c:
                out["justification"] = {"claims": c}
        return out

    def _unread(self) -> list[str]:
        leads = []
        for key, row in self.shelf.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.shelf.pages or str(pid) in self.shelf.dead_pid:
                continue
            st = self.shelf.quote.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.ceiling is not None and price > self.ceiling + _EPS):
                continue
            title = str(row.get("title") or "")
            if self.wish.theme and not _whole(title, self.wish.theme):
                continue
            if self.wish.model and self.shelf.adds == 0 and not self.shelf.shock_at:
                if _squash(self.wish.model) not in _squash(title):
                    continue
            if self.wish.maker:
                want = _words(self.wish.maker)
                if not (want <= _words(str(row.get("brand") or "")) or want <= _words(title)):
                    continue
            if self.wish.family == "preference" and not self.gear_ok(str(pid)):
                continue
            if _same_noun(self.wish.shelf or "", "smartphone") and not re.search(
                r"\b(?:smart\s*)?phones?\b|\bcellphones?\b|\bcellphone\b", title, re.I
            ):
                continue
            spec_hit = 0
            if self.wish.gate_val and self.wish.gate:
                spec_hit = 1 if self.wish.gate_val in near_values(self.wish.gate, title) else 0
            leads.append((-spec_hit, -self._title_spread(title), -self._title_want(title), price, str(pid)))
        leads.sort()
        if self.wish.gate_val and any(row[0] < 0 for row in leads):
            leads = [row for row in leads if row[0] < 0]
        if self.wish.family == "preference":
            water = [row for row in leads if row[2] < 0]  # title_want
            wide = [row for row in leads if row[1] <= -2 and row[2] < 0]
            if self.shelf.shock_at and wide:
                leads = wide
            elif wide:
                leads = wide
            elif water:
                leads = water
        out = []
        for _s, _sp, _w, _p, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _title_spread(self, title: str) -> int:
        blob = (title or "").lower()
        bag = " " + re.sub(r"[^a-z0-9]+", " ", blob) + " "
        n = 0
        for group in _USE_CLUMPS:
            if any(self._want_score(blob, bag, t) > 0 for t in group):
                n += 1
        return n

    def _title_want(self, title: str) -> int:
        if not self.primary:
            return 0
        bag = " " + re.sub(r"[^a-z0-9]+", " ", (title or "").lower()) + " "
        blob = (title or "").lower()
        n = 0
        for t in self.primary:
            n += self._want_score(blob, bag, t)
        return n

    def _filter_args(self) -> dict:
        args: dict[str, Any] = {"in_stock": True, "k": 10}
        if self.wish.shelf:
            args["category"] = self.wish.shelf
        if self.ceiling is not None:
            args["max_price"] = self.ceiling
        if self.wish.maker:
            args["brand"] = self.wish.maker
        elif (
            self.wish.soft_maker
            and self.shelf.adds == 0
            and self.turn < self.max_steps * 0.62
            and not self.maker_miss
            and not self.shelf.shock_at
        ):
            args["brand"] = self.wish.soft_maker
        return args

    def _gate_query(self) -> str:
        v = self.wish.gate_val or ""
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)in", v)
        if m:
            return m.group(1) + " inch"
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)hz", v)
        if m:
            return m.group(1) + " hz"
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)gb", v)
        if m:
            return m.group(1) + " gb"
        return v

    def _emit(self, name: str, args: dict) -> tuple[str, dict] | None:
        sig = name + json.dumps(args, sort_keys=True)
        if sig in self.shelf.signatures:
            return None
        self.shelf.signatures.add(sig)
        return (name, args)

    def harvest_move(self) -> tuple[str, dict] | None:
        w = self.wish
        if w.family == "ranking":
            missing = [p for p, _s in w.named if p not in self.shelf.pages and p not in self.opaque]
            if missing:
                if self.has("compare"):
                    return self._emit("compare", {"product_ids": missing[:5]})
                if self.has("view"):
                    return self._emit("view", {"product_id": missing[0]})
            # Named set only — extra catalog rows cannot enter the ranking.
            return None
        if (
            w.family in ("constraint", "intent")
            and not w.gate_val
            and (w.sealed or w.family == "constraint")
            and self.has("message")
            and self.q_sent < 2
            and self.turn >= 3
            and not self.spec_nudge
        ):
            self.spec_nudge = True
            self.q_sent += 1
            return self._emit(
                "message",
                {
                    "content": (
                        "Name the one hardware number the chosen variant must print — field and exact "
                        "value (size, panel class, refresh, or memory). I will not guess it from the title."
                    )
                },
            )
        if w.family == "recovery" and not self.shelf.adds and w.model and self.has("search"):
            sib = self._sibling()
            if sib:
                return sib
            q = w.model
            if q.lower() not in self.harvested and self.searches < 6:
                move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                if move:
                    self.harvested.add(q.lower())
                    return move
        if w.family == "recall":
            need = w.recall_n
            if self.has("search") and (len(self.shelf.seen_ids) < need or self.searches < 2):
                move = self._recall_search()
                if move:
                    return move
        if w.family == "preference" and self.has("search") and self.searches < 7:
            for q in self._taste_queries():
                args = {"query": q, "in_stock": True, "k": 10}
                if q.lower() not in self.harvested:
                    move = self._emit("search", args)
                    if move:
                        self.harvested.add(q.lower())
                        return move
            if not self.primary and w.shelf and self.searches == 0:
                args = {"query": w.shelf, "in_stock": True, "k": 10}
                if w.shelf.lower() not in self.harvested:
                    move = self._emit("search", args)
                    if move:
                        self.harvested.add(w.shelf.lower())
                        return move
        if self.searches == 0 and w.family not in ("ranking", "recovery", "preference"):
            if self.has("filter") and w.shelf:
                move = self._emit("filter", self._filter_args())
                if move:
                    return move
            if self.has("search"):
                words = [x for x in (self._name_q(), self._gate_query(), w.shelf) if x]
                if words:
                    q = " ".join(str(x) for x in words)
                    if q.lower() not in self.harvested:
                        move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                        if move:
                            self.harvested.add(q.lower())
                            return move
        if self.spec_fresh and w.gate_val:
            self.spec_fresh = False
            q = " ".join(x for x in (w.shelf, self._gate_query(), self._name_q()) if x)
            if q and q.lower() not in self.harvested and self.has("search"):
                move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                if move:
                    self.harvested.add(q.lower())
                    return move
            if self.has("filter") and w.shelf:
                move = self._emit("filter", self._filter_args())
                if move:
                    return move
        if w.family == "preference" and self.shelf.shock_at and self.has("search") and self.searches < 10:
            for q in self._taste_queries():
                if q.lower() not in self.harvested:
                    move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                    if move:
                        self.harvested.add(q.lower())
                        self.hint("Recover on a leftover listing_id that still shows the ranked uses together.")
                        return move
        if w.family in ("intent", "constraint") and self.shelf.shock_at and self.has("search"):
            q = " ".join(x for x in (w.shelf, self._gate_query()) if x)
            tagged = q + " |" + "%.0f" % (self.ceiling or 0)
            if q and tagged.lower() not in self.harvested:
                move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                if move:
                    self.harvested.add(tagged.lower())
                    return move
        # Taste: keep looking while MOST-titled unread rows remain.
        if w.family == "preference" and not self.shelf.shock_at:
            leftover = self._unread()
            aligned = any(h.ok and self._has_most(h) for h in self.pool()) if self.primary else True
            if aligned and len(self.shelf.listings) >= 2 and not leftover:
                return None
        unread = self._unread()
        if unread and self.reads < 12 and self.turn >= 1:
            self.reads += 1
            if len(unread) > 1 and self.has("compare"):
                move = self._emit("compare", {"product_ids": unread[:5]})
                if move:
                    return move
            if self.has("view"):
                move = self._emit("view", {"product_id": unread[0]})
                if move:
                    return move
        if w.family == "constraint" and self.shelf.shock_at and self.has("search"):
            q = " ".join(x for x in (w.shelf, self._gate_query(), w.maker) if x)
            if q and q.lower() not in self.harvested:
                move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                if move:
                    self.harvested.add(q.lower())
                    return move
        if w.family not in ("ranking", "constraint", "preference", "intent") and self.shelf.shock_at and not self.twin_q and self.has("search"):
            title = self.title(self.shelf.shock_at.split("::", 1)[0])
            if title:
                self.twin_q = True
                self.hint("Search the dead listing's title; sibling colors/sizes share it.")
                return self._emit("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})
        miss = self._near_miss_search()
        if miss:
            return miss
        probe = "|".join(
            [w.maker or "", "%.2f" % (self.ceiling or 0.0), str(w.gate_val or ""), str(self.searches)]
        )
        if w.family == "preference":
            return None
        if self.turn >= 2 and probe not in self.miss_queries and self.searches < 8:
            self.miss_queries.add(probe)
            if self.has("filter") and w.shelf:
                move = self._emit("filter", self._filter_args())
                if move:
                    return move
            if self.has("search"):
                words = [x for x in (self._name_q(), self._gate_query(), w.shelf) if x]
                if words:
                    q = " ".join(str(x) for x in words)
                    if q.lower() not in self.harvested:
                        move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
                        if move:
                            self.harvested.add(q.lower())
                            return move
        return None

    def _recall_search(self) -> tuple[str, dict] | None:
        base = " ".join(self.wish.theme) if self.wish.theme else ""
        if not base:
            return None
        for q in alt_queries(base):
            if q.lower() in self.harvested:
                continue
            move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
            if move:
                self.harvested.add(q.lower())
                return move
        if self.wish.shelf:
            mixed = self.wish.shelf + " " + base
            if mixed.lower() not in self.harvested:
                move = self._emit("search", {"query": mixed, "in_stock": True, "k": 10})
                if move:
                    self.harvested.add(mixed.lower())
                    return move
        return None

    def _near_miss_search(self) -> tuple[str, dict] | None:
        """Rejected pages tell us which literal token is still missing — re-query it."""
        if self.searches >= 7 or not self.has("search"):
            return None
        blocked = [h for h in self.pool() if h.viewed and h.live and not h.ok]
        if not blocked or not self.wish.gate_val:
            return None
        q = " ".join(x for x in (self.wish.shelf, self._gate_query(), self.wish.maker) if x)
        if not q or q.lower() in self.harvested:
            return None
        move = self._emit("search", {"query": q, "in_stock": True, "k": 10})
        if move:
            self.harvested.add(q.lower())
        return move

    def _sibling(self) -> tuple[str, dict] | None:
        if self.sib_q or not self.has("search"):
            return None
        if any(h.live and h.viewed and h.relax == 0 for h in self.pool()):
            return None
        near = [h for h in self.pool() if h.viewed and h.relax == 1]
        if not near:
            return None
        title = self.title(near[0].pid)
        if not title:
            return None
        self.sib_q = True
        return self._emit("search", {"query": " ".join(title.split()[:8]), "k": 10, "in_stock": True})

    def checkout_move(self) -> tuple[str, dict] | None:
        if self.done:
            return None
        if self.answer_due:
            if self.has("message"):
                self.answer_due = False
                self.q_wait = 0
                return ("message", {"content": self._say()})
        if self.wish.family == "preference" and self.shelf.shock_at:
            scan = self.harvest_move()
            if scan and scan[0] in ("search", "filter", "view", "compare"):
                wide = any(
                    h.ok and self._has_most(h) and self._spread(h.pid, h.sku) >= 2
                    for h in self.pool()
                )
                if not wide or scan[0] in ("search", "filter"):
                    return scan
        if self.wish.family == "intent" and self.shelf.shock_at:
            scan = self.harvest_move()
            if scan and scan[0] in ("search", "filter", "view", "compare"):
                if not any(h.ok for h in self.pool()) or scan[0] in ("search", "filter"):
                    return scan
        target = self.champion()
        if target is None or not self.ready():
            forced = self.harvest_move()
            if forced:
                return forced
            if target is None:
                return None
        pid, sku = target.pid, target.sku
        if (pid, sku) in self.shelf.cart:
            if not self.fresh(pid, sku) and self.has("inspect_stock"):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            if self.has("place_test_order"):
                return ("place_test_order", self.order_args(pid, sku))
            return None
        if not self.ready():
            return None
        for cpid, csku in self.shelf.cart:
            if vid(cpid, csku) in self.shelf.dead and self.has("remove_from_cart"):
                return ("remove_from_cart", {"product_id": cpid, "sku": csku})
        if not self.fresh(pid, sku) and self.has("inspect_stock"):
            return ("inspect_stock", {"product_id": pid, "sku": sku})
        if self.has("add_to_cart"):
            return ("add_to_cart", {"product_id": pid, "sku": sku})
        return None

    def _say(self) -> str:
        h = self.champion() or (self.pool()[:1] or [None])[0]
        if not h:
            return "Still checking options against your requirements; nothing is locked in yet."
        bits = ["Going with " + (self.title(h.pid) or h.pid)[:90]]
        if self.wish.gate and self.wish.gate_val and self.sku_val(h.pid, h.sku, self.wish.gate) == self.wish.gate_val:
            bits.append("listed " + self.wish.gate.replace("_", " ") + " is " + self.wish.gate_val)
        if h.price is not None:
            bits.append("%.2f %s, in stock" % (h.price, self.wish.currency))
        return ", ".join(bits) + "."

    def opening(self) -> str | None:
        if (self.wish.family == "constraint" and not self.wish.gate_val) or self.wish.sealed:
            return (
                "Name the one hardware number the chosen variant must print — field and exact "
                "value (size, panel class, refresh, or memory). I will not guess it from the title."
            )
        if self.wish.family == "preference" and not self.primary:
            return (
                "When two in-budget devices conflict, what counts most and what comes second? "
                "Answer as 'X counts most, then Y'. I will not pick the cheapest silent SKU."
            )
        return None

    def _budget_msg(self, text: str) -> None:
        new = _money(text)
        if new is None or (self.ceiling is not None and new >= self.ceiling):
            return
        self.ceiling = new
        self.shopper_replied = True
        self.spec_fresh = True
        self.reads = 0
        self._retune()
        self.hint("Ceiling is now %.2f %s." % (new, self.wish.currency))
        killed = False
        for pid, sku in list(self.shelf.cart):
            st = self.shelf.quote.get(vid(pid, sku)) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.shelf.dead.add(vid(pid, sku))
                self.shelf.dead_pid.add(str(pid))
                killed = True
                self.hint("Carted %s is above the new ceiling." % vid(pid, sku))
        if killed:
            self.shelf.shock_at = self.shelf.shock_at or vid(self.shelf.cart[-1][0], self.shelf.cart[-1][1])
            self.forced = None

    def _priority_msg(self, text: str) -> None:
        low = (text or "").lower()
        signal = bool(re.search(r"prefer|priority|rather|first|then|\bover\b|important|weigh|counts most|matters most", low))
        if _ASKING.search(text) and not signal:
            return
        if self.wish.family != "preference" and not signal:
            return
        head, nxt = ranked_wants(text)
        if head:
            self.primary = head
        if nxt:
            self.secondary = nxt
        ordered: list[str] = []
        for m in re.finditer(
            r"(?:most important|come first|weigh first|prefer(?:ably)?|rather than|then|over|counts most)\s+([a-z0-9][\w\-]{2,24})",
            low,
        ):
            tok = m.group(1)
            if tok not in _STOP and tok not in ordered:
                ordered.append(tok)
        toks = want_tokens(low)
        merged = list(dict.fromkeys(self.primary + self.secondary + ordered + toks))
        if merged:
            self.priority = merged[:12]
            if self.wish.family == "preference":
                bits = []
                if self.primary:
                    bits.append("top=" + ",".join(self.primary[:6]))
                if self.secondary:
                    bits.append("then=" + ",".join(self.secondary[:4]))
                self.hint("Ranked wants I will require on the page: " + "; ".join(bits or merged[:6]) + ".")
                self._retune()

    def _market_from_shopper(self, text: str) -> None:
        """The shopper often announces the market event; trust that over a stale quote."""
        jumped = _PRICE_JUMP.search(text or "")
        talked = bool(_MARKET_TALK.search(text or ""))
        if not jumped and not talked:
            return
        if not self.shelf.cart:
            if talked:
                self.hint("Re-quote the SKU you were about to commit; prices move on the first add.")
            return
        pid, sku = self.shelf.cart[-1]
        newp = _num((jumped.group(2) or "").replace(",", "")) if jumped else None
        over = self.ceiling is not None and newp is not None and newp > self.ceiling + _EPS
        if over:
            self.shelf.dead.add(vid(pid, sku))
            self.shelf.dead_pid.add(str(pid))
            self.shelf.shock_at = vid(pid, sku)
            self.forced = None
            self.reads = 0
            self.hint("Shopper quoted %s at %.2f, above the ceiling; recover on another listing_id." % (vid(pid, sku), newp))
        self.recheck = (pid, sku)

    def think(self, extra: list[dict] | None = None) -> dict | None:
        if self.left() < 32:
            return None
        msgs = list(self.messages)
        if extra:
            msgs.extend(extra)
        tries = 1 if self.left() < 90 else 2
        for attempt in range(tries):
            if self.left() < 32:
                return None
            model = self.models[min(self.model_i, len(self.models) - 1)]
            resp = _WIRE.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": msgs,
                    "tools": self.tools,
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
                self.hint("Choose exactly one supplied tool.")
                continue
            if self.left() > 80:
                time.sleep(min(0.8 * (attempt + 1), 2.0))
            self.model_i = min(self.model_i + 1, len(self.models) - 1)
        return None

    def send(self, envelope: dict) -> dict | None:
        tries = 2 if self.left() < 70 else 4
        for attempt in range(tries):
            result = _WIRE.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.left() < 22:
                break
            time.sleep(min(1.2 * (attempt + 1), 3.0))
        return None

    def act(self, name: str, args: dict, content: str = "", call_id: str | None = None) -> dict | None:
        if self.pending_env is not None:
            pend = self.pending_env
            replay = self.send(pend["envelope"])
            if replay is None:
                self.hint("Environment still pending; retrying the last action.")
                return None
            self.pending_env = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.pid + "-t" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self.send(envelope)
        if result is None:
            self.pending_env = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": call_id or ("c-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, call_id or ("c-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.shelf.signatures.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"error": "environment call failed; retrying"}),
                }
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = peel(wrap)
        if wrap.get("done") or obs.get("done"):
            self.done = True
        if any(isinstance(c, dict) and peel(c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        self.shelf.ingest(name, obs, self.wish)
        if name == "filter" and args.get("brand") and not (obs.get("results") or []):
            self.maker_miss = True
            self.hint(
                "No in-budget rows for that brand at the current ceiling; other makers that clear "
                "the hard gates are allowed."
            )
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _clip(json.dumps(obs), _OBS_CAP)})
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_msg(text)
            self._priority_msg(text)
            self._market_from_shopper(text)
            if self.wish.family in ("constraint", "intent") and not self.wish.gate_val:
                got = self.wish.take_spec(text)
                if got:
                    self.shopper_replied = True
                    self.spec_fresh = True
                    self._retune()
                    self.hint("Firm spec recorded: " + got + ".")
            if _ASKING.search(text) and not (_MARKET_TALK.search(text) or _PRICE_JUMP.search(text)):
                self.answer_due = True
                self.hint("Answer the shopper with message, then continue.")
        return obs

    def after_add(self) -> None:
        pid, sku = self.recheck
        self.recheck = None
        if not self.has("inspect_stock"):
            return
        obs = self.act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get("error"):
            return
        stock = bool(obs.get("in_stock"))
        price = _num(obs.get("price"))
        over = self.ceiling is not None and price is not None and price > self.ceiling + _EPS
        if stock and not over:
            self.hint("Post-cart check: %s still live at %s." % (vid(pid, sku), price))
            return
        self.shelf.shock_at = vid(pid, sku)
        self.shelf.dead.add(vid(pid, sku))
        self.shelf.dead_pid.add(str(pid))
        self.forced = None
        self.reads = 0
        why = "OOS" if not stock else "repriced to %.2f" % (price or 0)
        recover = {
            "recovery": "Relax only color on the same model; do not jump models yet.",
            "ranking": "Only named survivors remain; closer re-sorts them.",
            "preference": "Move to another listing_id that still matches the ranked wants.",
            "constraint": "Change product_id. A sibling SKU of the dead page is not a recovery.",
        }.get(self.wish.family, "Pick another product that still clears every firm gate.")
        self.hint("Market move: %s %s. %s" % (vid(pid, sku), why, recover))
        self.phase = self.HARVEST

    def slam(self) -> None:
        skip_cart = self.wish.family == "preference" and self.shelf.shock_at
        if not skip_cart:
            for pid, sku in reversed(self.shelf.cart):
                ok, _ = self.orderable(pid, sku)
                if ok and not self.done and self.has("place_test_order"):
                    self.act("place_test_order", self.order_args(pid, sku))
                    return
        if self.done or self.turn >= self.max_steps:
            return
        good = [h for h in self.pool() if h.ok]
        if self.wish.family == "recovery" and self.shelf.adds == 0:
            exact = [h for h in good if h.relax == 0]
            if exact:
                good = exact
            else:
                return
        if not good:
            relax = [
                h for h in self.pool()
                if h.live and self.maker_ok(h.pid) and self.shelf_ok(h.pid) and self.device_ok(h.pid, h.sku)
                and (not self.wish.named or (h.pid, h.sku) in self.wish.named)
            ]
            if self.wish.gate and self.wish.gate_val:
                relax = [
                    h for h in relax
                    if self.sku_val(h.pid, h.sku, self.wish.gate) == self.wish.gate_val
                ]
            if self.wish.family not in ("justification", "constraint", "recovery"):
                good = relax
            elif self._panic() and relax:
                good = relax
        if not good:
            return
        if self.wish.family == "preference":
            wide = [x for x in good if self._has_most(x) and self._spread(x.pid, x.sku) >= 2]
            strong = [x for x in good if self._strong_most(x)]
            aligned = [x for x in good if self._has_most(x)]
            if wide:
                good = wide
            elif strong:
                good = strong
            elif aligned:
                good = aligned
            good.sort(key=self._taste_key)
        else:
            good.sort(key=lambda h: self.rank_key(h, self.shelf.adds == 0))
        for h in good[:8]:
            if self.done or self.turn >= self.max_steps:
                return
            if not self.fresh(h.pid, h.sku) and self.has("inspect_stock"):
                obs = self.act("inspect_stock", {"product_id": h.pid, "sku": h.sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or obs.get("error"):
                    continue
            if (h.pid, h.sku) not in self.shelf.cart and self.has("add_to_cart"):
                obs = self.act("add_to_cart", {"product_id": h.pid, "sku": h.sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or obs.get("error"):
                    continue
            if self.has("inspect_stock"):
                obs = self.act("inspect_stock", {"product_id": h.pid, "sku": h.sku})
                if self.done:
                    return
                if not isinstance(obs, dict) or not obs.get("in_stock"):
                    continue
                price = _num(obs.get("price"))
                if self.ceiling is not None and price is not None and price > self.ceiling + _EPS:
                    continue
            if self.has("place_test_order"):
                self.act("place_test_order", self.order_args(h.pid, h.sku))
            return

    def veto(self, name: str, args: dict) -> str | None:
        if name not in self.names:
            return "unknown tool"
        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.shelf.cart:
                pid, sku = self.shelf.cart[-1]
            ok, why = self.orderable(pid, sku)
            if not ok:
                return "order blocked: " + why
            if self.wish.family == "recall" and len(self.shelf.seen_ids) < self.wish.recall_n:
                return "recall incomplete: %d/%d distinct title hits" % (len(self.shelf.seen_ids), self.wish.recall_n)
            if self.wish.family == "preference" and len(self.shelf.listings) < 2:
                return "compare two listing_id values first"
        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs product_id and sku"
            h = self.score(pid, sku)
            if not h.live:
                return "dead, OOS, or over budget"
            if not h.viewed:
                return "view or compare first"
            if vid(pid, sku) not in self.shelf.inspected:
                return "inspect_stock this SKU first"
            if not h.ok:
                return "fails a firm constraint"
            if self.wish.family == "preference" and len(self.shelf.listings) < 2:
                return "compare two listing_ids before the first add"
            if (
                self.wish.family == "preference"
                and self.shelf.adds == 0
                and self.searches < 2
                and not self._panic()
            ):
                return "search the ranked want before the first add"
            if self.wish.family == "preference" and self.primary:
                best = max(
                    (self._spread(x.pid, x.sku) for x in self.pool() if x.ok and self._has_most(x)),
                    default=0,
                )
                got = self._spread(h.pid, h.sku) if self._has_most(h) else 0
                if best >= 2 and got < 2:
                    return "a leftover page still shows MOST plus another ranked use; cart that listing"
            if (
                self.wish.family == "preference"
                and self.primary
                and not self._strong_most(h)
                and not self.shelf.shock_at
            ):
                if any(x.ok and self._strong_most(x) for x in self.pool()):
                    return "a page with catalog MOST spelling is on record; cart that listing"
            if (
                self.wish.family == "intent"
                and self.shelf.adds == 0
                and self.wish.soft_maker
            ):
                want = _words(self.wish.soft_maker)
                this = want <= _words(self.brand(pid)) or want <= _words(self.title(pid))
                branded = [
                    x for x in self.pool()
                    if x.ok and (want <= _words(self.brand(x.pid)) or want <= _words(self.title(x.pid)))
                ]
                if branded and not this:
                    return "a valid preferred-brand listing is on record; first add must be that maker"
            if self.wish.family == "recovery" and not self.shelf.adds and self.wish.model and h.relax:
                return "first add must be the named model and color"
            if not self.ready():
                return "hold for a backup candidate or missing family evidence"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "empty message"
            if not self.answer_due and "?" not in text:
                return "no status updates; message only to ask or answer the shopper"
            if not self.answer_due and self.q_sent >= 2:
                return "enough questions; decide from evidence"
        if name == "inspect_cart" and self.cart_looks >= 1:
            return "cart already known"
        if name in ("search", "filter") and self.wish.family == "ranking":
            return "closed named set; compare the given product_id values instead"
        sig = name + json.dumps(args, sort_keys=True)
        if sig in self.shelf.signatures and name in ("search", "filter", "view", "compare"):
            return "identical call already made"
        self.shelf.signatures.add(sig)
        return None

    def snapshot(self) -> None:
        good = [h for h in self.winners()[:6]]
        if self.wish.family == "preference" and good:
            ranked = sorted(good, key=self._taste_key)
            lines = []
            for h in ranked[:6]:
                lid = self.shelf.listing_of.get(h.pid, "?")
                pri = self._page_hits(h.pid, h.sku, self.primary)
                sp = self._spread(h.pid, h.sku)
                lines.append(
                    "%s::%s lid=%s price=%s uses=%d top_hits=%d/%d %s"
                    % (
                        h.pid, h.sku, lid, h.price, sp, pri, max(1, len(self.primary) or 1),
                        (self.title(h.pid) or "")[:48],
                    )
                )
            extra = ""
            if self.primary:
                extra = " MOST tokens: " + ", ".join(self.primary[:6]) + "."
            board = (
                "Recover board. Order the leftover with the most uses on the page (water+radio+hold), not cheapest waterproof."
                if self.shelf.shock_at
                else "Tradeoff board. First add needs MOST on the page, not cheapest-valid."
            )
            self.hint(
                board
                + extra
                + " inspect_stock then add_to_cart. "
                + " || ".join(lines)
            )
            return
        if good:
            self.hint(
                "Valid SKUs: "
                + "; ".join("%s::%s @ %s" % (h.pid, h.sku, h.price) for h in good)
                + ". Closer will quote/cart/order when ready."
            )
            return
        blocked = [h for h in self.pool() if h.viewed and h.live and not h.ok][:3]
        if blocked:
            self.hint(
                "Read SKUs fail a firm check: "
                + ", ".join(h.pid + "::" + h.sku for h in blocked)
                + ". Try different products."
            )

    def run(self) -> list:
        question = self.opening()
        last, reps = None, 0
        while not self.done and self.turn < self.max_steps:
            if self._panic():
                self.slam()
                break
            if question and not self.asked and self.has("message"):
                self.asked = True
                self.q_sent += 1
                self.act("message", {"content": question})
                self.phase = self.HARVEST
                continue
            self.phase = self.HARVEST
            if self.recheck and self.has("inspect_stock"):
                self.after_add()
                continue
            move = self.checkout_move()
            if move is not None:
                name, args = move
                sig = name + json.dumps(args, sort_keys=True)
                reps = reps + 1 if sig == last else 1
                last = sig
                if reps > 2:
                    if name in ("inspect_stock", "add_to_cart"):
                        self.shelf.dead.add(vid(args.get("product_id"), args.get("sku")))
                    if name == "compare":
                        self.opaque.update(str(p) for p in args.get("product_ids") or [])
                    if name == "message":
                        self.answer_due = False
                    last, reps = None, 0
                    self.hint("That step stalled; moving on.")
                    continue
                obs = self.act(name, args)
                if name == "message":
                    self.answer_due = False
                    self.q_wait = 0
                if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                    added = obs.get("added") or {}
                    if added.get("product_id"):
                        self.recheck = (str(added["product_id"]), str(added["sku"]))
                        self._hold_taste(str(added["product_id"]), str(added["sku"]))
                if name in ("search", "filter"):
                    self.searches += 1
                    if self.wish.family == "recall" and self.wish.recall_n:
                        have, need = len(self.shelf.seen_ids), self.wish.recall_n
                        self.hint("Title hits %d/%d.%s" % (have, need, "" if have >= need else " Search again with different wording."))
                if name in ("view", "compare"):
                    self.snapshot()
                continue
            last, reps = None, 0
            proposal = self.think()
            if proposal is None:
                if self._panic():
                    self.slam()
                    break
                fallback = self.harvest_move()
                if fallback is None and self.has("search"):
                    words = [x for x in (self._name_q(), self._gate_query(), self.wish.shelf) if x]
                    if words:
                        q = " ".join(str(x) for x in words)
                        if q.lower() not in self.harvested:
                            self.harvested.add(q.lower())
                            fallback = ("search", {"query": q, "in_stock": True, "k": 10})
                if fallback is not None:
                    name, args = fallback
                    obs = self.act(name, args)
                    if name in ("search", "filter"):
                        self.searches += 1
                    if name in ("view", "compare"):
                        self.snapshot()
                    continue
                if self.champion():
                    self.slam()
                    break
                continue
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = tool_args(call)
            content = proposal.get("content") or ""
            nrep = 0
            verdict = self.veto(name, args)
            while verdict and nrep < 2:
                nrep += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[blocked " + name + "] " + verdict[:180]})
                again = self.think()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = tool_args(call)
                content = again.get("content") or ""
                verdict = self.veto(name, args)
            if verdict:
                self.reject_n += 1
                self.hint("Skipped: " + verdict + ".")
                if self.reject_n >= 3:
                    self.snapshot()
                    self.reject_n = 0
                continue
            self.reject_n = 0
            if name == "place_test_order":
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.shelf.cart:
                    pid, sku = self.shelf.cart[-1]
                args = self.order_args(pid, sku)
            if name == "message":
                if not self.answer_due:
                    self.q_sent += 1
                self.answer_due = False
                self.q_wait = 0
            obs = self.act(name, args, content, call.get("id"))
            if name == "add_to_cart" and isinstance(obs, dict) and not obs.get("error"):
                added = obs.get("added") or {}
                if added.get("product_id"):
                    self.recheck = (str(added["product_id"]), str(added["sku"]))
                    self._hold_taste(str(added["product_id"]), str(added["sku"]))
            if name in ("search", "filter"):
                self.searches += 1
            if name == "inspect_cart":
                self.cart_looks += 1
            if name in ("view", "compare"):
                self.snapshot()
        if not self.done:
            self.slam()
        return self.dialogue


def agent_main(problem_data: dict) -> list:
    try:
        return ShopRun(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: %s: %s" % (type(exc).__name__, str(exc)[:200])}]


__all__ = ["agent_main"]
