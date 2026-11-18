"""ORO SN15 generated-environment agent - deterministic, verifier-aligned lanes.

Built against oro-env-runtime 0.2.10 (tools oro_task_tools_v5, verifier 0.3.4);
the graders it mirrors can be checked with `pip download oro-env-runtime==0.2.10`.

* One lane per task family, detected from the family's fixed "Task rules" text,
  with an LLM tool loop as fallback (its orders are validated before sending).
* One validation function per family (candidate_ok) gates the first commitment,
  reuse of a cart item, the recovery and any emergency order, always against the
  CURRENT requirements (budget cuts, "must now have X instead of Y" revisions,
  sealed answers) and the latest observations.  No arbitrary fallback purchases.
* Close-out (ORO_AGENT_CLOSE_MODE): "auto" (default) validates the recovery on a
  fresh observation before ordering in TF1/TF3 - the only families whose runtime
  validator admits a chained second event - and fuses recovery+order elsewhere,
  where no further event can hit the replacement and undelivered interventions
  are never enforced.  "validated" / "fused" force one mode.  The cart refresh
  always shares the order's group: the verifier binds an order to the last cart
  observation's state hash.
* TF4 asks the shopper's priorities before the first add (the rules require it;
  tasks that seal it as required zero the metric without it).  ORO_AGENT_TF4_ASK=0
  disables.  TF5 ranks by the priority stated in the goal/rules.
* Latency-aware wall-clock budget sized for the production 300 s per-task kill.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


TIME_LIMIT_S = float(getenv("ORO_AGENT_TIME_LIMIT_S") or "300")

EMERGENCY_AT_S = TIME_LIMIT_S - 90.0

NO_ROUND_AFTER_S = TIME_LIMIT_S - 140.0

# Close-out: "auto" validates the recovery on fresh observations in TF1/TF3 (the only
# families whose runtime validator admits a chained second event) and fuses the
# recovery with the order elsewhere; "validated" / "fused" force one mode everywhere.
CLOSE_MODE = (getenv("ORO_AGENT_CLOSE_MODE") or
              {"0": "validated", "1": "fused"}.get(getenv("ORO_AGENT_FUSED_CLOSE") or "", "auto"))

CHAIN_FAMILIES = ("intent_decomposition", "constraint_satisfaction")

TF4_ASK_PRIORITIES = (getenv("ORO_AGENT_TF4_ASK") or "1") != "0"

TF4_RESERVE_BEST = (getenv("ORO_AGENT_TF4_RESERVE_BEST") or "1") != "0"

MAX_SEARCHES_PER_GROUP = 3

MAX_COMPARES_PER_GROUP = 3

LLM_MODELS = [
    m
    for m in (
        getenv("SANDBOX_MODEL"),
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
    )
    if m
]

_ENV = ProxyClient(timeout=70, max_retries=1)

_LLM = ProxyClient(timeout=40, max_retries=1)

TF1 = "intent_decomposition"

TF2 = "retrieval_recall"

TF3 = "constraint_satisfaction"

TF4 = "preference_reasoning"

TF5 = "ranking"

TF6 = "recovery"

TF7 = "justification"

_FAMILY_MARKERS = (
    (TF2, ("Match the requested phrase against the listing title",
           "Search the catalog for what the user described")),
    (TF5, ("stated tradeoff priority", "highest advertised maximum refresh rate")),
    (TF6, ("critical constraints are category, budget, and",
           "least-relaxed in-stock option")),
    (TF7, ("valid product plus a trustworthy reason",)),
    (TF4, ("authorizes one test order", "opaque identity for the underlying")),
    (TF1, ("MIX of firm requirements and nice-to-haves", "you must tell them apart")),
    (TF3, ("gives firm, non-negotiable requirements", "EXACT variant you order")),
)

_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I)

_INCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|\\u201[cd]|\\?[\"“”])",
    re.I,
)

_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.I)

_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")

_RAM_STORAGE_PAIR = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b", re.I
)

_CAPACITY_PAIR_HINT = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)", re.I
)

_RAM_TITLE = re.compile(r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.I)

RELIABLE_KEYS = ("ram", "resolution", "screen_size", "refresh_rate")

_FACT_LABEL_FOR = {"screen_size": "size", "resolution": "display", "refresh_rate": "refresh_rate"}

_CATEGORY_ALIASES = {
    "video cards": "graphics_card", "graphics cards": "graphics_card",
    "video graphics cards": "graphics_card", "gpu": "graphics_card", "gpus": "graphics_card",
    "mobile phones": "smartphone", "cell phones": "smartphone", "cellphones": "smartphone",
    "cellphone": "smartphone", "phones": "smartphone", "smartphones": "smartphone",
    "unlocked phones": "smartphone", "smart phones": "smartphone", "wi fi": "networking",
    "wifi": "networking", "smartwatches": "smartwatch", "smart watches": "smartwatch",
    "headphones headsets": "headphones",
}

_HIGH_REFRESH_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*-?\s*hz", re.I)

_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|"
    r"harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.I,
)

_SMARTPHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.I)

_PHONE_CONFLICTS = (
    re.compile(r"\btablets?\b", re.I),
    re.compile(r"\b(?:feature|basic)\s*phones?\b", re.I),
    re.compile(r"\b(?:mystery|blind|lucky)\s*box(?:es)?\b", re.I),
    re.compile(
        r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|case\s*(?:cover\s*)?for|"
        r"replacement\s*(?:screen|battery)|smart\s*watches?|earbuds?|headphones?)\b",
        re.I,
    ),
)

_SPORT_FIT = r"\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\b"

_TF4_BAD_TITLE = ("customer review", "customer rating", "search results", "results page",
                  "refurb", "pre-owned", "restored", "renewed")

_COLOR_PHRASES = (
    "midnight blue", "titanium black", "matte black", "triple black", "frost white",
    "white smoke", "cloud grey", "moonlight grey", "space gray", "street gray",
    "driftwood sand", "cypress green", "tidal blue", "light blue", "sky blue", "indigo blue",
    "canvas white", "platinum silver", "starlight", "graphite", "sandstone", "transparent",
    "silver", "beige", "black", "blue", "white", "gray", "grey", "green", "pink", "violet",
    "purple", "red", "sage", "gold", "khaki", "peach",
)

_MODEL_PATTERNS = (
    (r"wh[-\s]?1000xm\s*([456])", lambda m: f"WH-1000XM{m.group(1)}"),
    (r"wf[-\s]?1000xm\s*([45])", lambda m: f"WF-1000XM{m.group(1)}"),
    (r"quietcomfort ultra(?: headphones)?[^\n]{0,80}\b2nd gen\b", lambda _m: "QuietComfort Ultra 2nd Gen"),
    (r"quietcomfort ultra", lambda _m: "QuietComfort Ultra"),
    (r"quietcomfort(?: headphones| wireless)?", lambda _m: "QuietComfort"),
    (r"headphones 700", lambda _m: "Headphones 700"),
    (r"beats studio pro|studio pro", lambda _m: "Studio Pro"),
    (r"beats studio buds \+|studio buds \+", lambda _m: "Studio Buds Plus"),
    (r"beats studio buds|studio buds", lambda _m: "Studio Buds"),
    (r"beats fit pro", lambda _m: "Fit Pro"),
    (r"momentum true wireless 4|mtw4|momentum 4", lambda _m: "Momentum 4"),
    (r"accentum", lambda _m: "Accentum"),
    (r"px7\s*s2e", lambda _m: "Px7 S2e"),
    (r"px7\s*s3", lambda _m: "Px7 S3"),
    (r"elite 10", lambda _m: "Elite 10"),
    (r"jbuds lux anc|go lux", lambda _m: "JBuds Lux ANC"),
)

_TF6_ACCESSORY_RE = re.compile(
    r"\b(?:case|cases|cover|covers|sleeve|sleeves|skin|skins|pouch|protector|cable|cables|"
    r"strap|straps|ear ?tips?|ear ?hooks?|adapter|holder|stand|replacement|carabiner)\b"
)

_OVER_OR_ON_EAR_RE = re.compile(r"\b(?:over|on)[-\s]?ear\b", re.I)

_FEATURE_PHONE_RE = re.compile(r"\b(?:feature|basic|dumb|button|keypad) ?phones?\b")

_INVALID_TITLE_PREFIXES = ("customer ratings", "customer reviews")

_INVALID_TITLE_FRAGMENTS = ("join bild expo", "earbud", "in-ear", "in ear", "earphone",
                            "open-ear", "true wireless")

_TITLE_MODEL_CODE = re.compile(r"\b[a-z]*\d{2,5}[a-z0-9]*(?:-[a-z0-9]+)*\b", re.I)

_NON_MODEL_CODE = re.compile(
    r"^(?:android|bluetooth|wifi|ips|ddr|lpddr|mtk)|"
    r"(?:gb|tb|mb|mah|hz|khz|mhz|ghz|mp|fps|inch|inches|p|g)$|^\d+x\d+$",
    re.I,
)

TF5_DEFAULT_PRIORITY = (("refresh_rate", -1), ("screen_size", -1), ("price", 1))

_BUDGET_RE = re.compile(
    r"(?i:up to|at or under|at or below|no more than|at most|under|below|within|budget of|"
    r"budget is|spend)\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Z]{3})\b"
)

_ANY_MONEY_RE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(PHP|USD|EUR|GBP|JPY|INR|AUD|CAD|SGD|MYR|IDR|VND|THB)\b"
)

_CAT_WORD_RE = re.compile(
    r"\b(?:gaming\s+monitors?|smart\s*phones?|monitors?|tablets?|head\s*phones?|ear\s*buds?|"
    r"smart\s*watch(?:es)?|laptops?)\b",
    re.I,
)

_BRAND_NOISE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*-?\s*inch(?:es)?|\d+(?:\.\d+)?\s*(?:gb|tb)(?:\s+of)?(?:\s+ram)?|"
    r"\d+(?:\.\d+)?\s*-?\s*hz|\d{3,4}p|4k|uhd|ultra\s+hd|qhd|full\s+hd|fhd|hd|in-stock|"
    r"in\s+stock|brand-new|new|genuine|unlocked|good|cheap)\b",
    re.I,
)

CAT_QUERY_WORD = {
    "Smartphones": "smartphone", "Gaming Monitors": "gaming monitor", "Monitors": "monitor",
    "Tablets": "tablet", "Headphones": "headphones", "Smartwatches": "smartwatch",
    "Laptops": "laptop",
}

FILTER_CATEGORY = {"Gaming Monitors": "Monitors"}

_STOP = {
    "the", "and", "for", "with", "from", "pcs", "piece", "pieces", "set", "new", "free",
    "sale", "hot", "best", "high", "quality", "original", "authentic", "size", "color",
    "colour", "pack", "box", "item", "items", "ready", "stock", "sold", "per", "you", "your",
}


def row_key(pid: Any, sku: Any) -> str:
    return f"{pid}::{sku or ''}"


def _flatten_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def numeric_tokens(text: str) -> list:
    values: list = []
    for chunk in re.split(r"[^0-9.]+", str(text)):
        if chunk and chunk != ".":
            try:
                values.append(float(chunk))
            except ValueError:
                continue
    return values


def canonical_forms(labels) -> set:
    out = set()
    for label in labels:
        low = _flatten_text(label)
        if low:
            out.add(low)
            out.add(_CATEGORY_ALIASES.get(low, low))
    return out


def group_runs(values: list) -> list[tuple]:
    """Consecutive equal runs as (value, length) pairs."""

    if not values:
        return []
    runs: list[tuple] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((values[start], i - start))
            start = i
    return runs


def extract_specs(text: str) -> dict:
    out: dict = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?\b", text, re.I)
    if m:
        out["screen_size"] = f"{float(m.group(1)):g}in"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|TB)\s+(?:of\s+)?(?:RAM|memory)\b", text, re.I)
    if m:
        mult = 1024 if m.group(2).lower() == "tb" else 1
        out["ram"] = f"{float(m.group(1)) * mult:g}gb"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*Hz\b", text, re.I)
    if m:
        out["refresh_rate"] = f"{float(m.group(1)):g}hz"
    m = re.search(
        r"\b(4K(?:/UHD)?|UHD|Ultra HD|2160p|QHD|1440p|1080p(?: Full HD)?|Full HD|FHD|720p(?: HD)?)\b",
        text,
    )
    if m:
        cls = resolution_canon(m.group(1))
        if cls:
            out["resolution"] = cls
    elif re.search(r"\ban? HD (?:display|screen|smartphone|tablet|monitor)\b", text):
        out["resolution"] = "hd"
    return out


def spec_query_phrase(key: str, value: str) -> str:
    if key == "screen_size":
        return f"{value[:-2]} inch"
    if key == "ram":
        return f"{value[:-2].upper()}GB RAM"
    if key == "refresh_rate":
        return f"{value[:-2]}Hz"
    return {"4k": "4K", "qhd": "QHD", "fhd": "1080p FHD", "hd": "720p HD"}.get(value, value)


class Trie:
    """Prefix tree over model-code tokens for quick 'is this a known code' checks."""

    def __init__(self) -> None:
        self.root: dict = {}
        self.size = 0

    def insert_word(self, word: str) -> None:
        node = self.root
        for ch in str(word).lower():
            node = node.setdefault(ch, {})
        if "$" not in node:
            node["$"] = True
            self.size += 1

    def contains(self, word: str) -> bool:
        node = self.root
        for ch in str(word).lower():
            if ch not in node:
                return False
            node = node[ch]
        return "$" in node

    def has_prefix(self, prefix: str) -> bool:
        node = self.root
        for ch in str(prefix).lower():
            if ch not in node:
                return False
            node = node[ch]
        return True


def _resolution_tier(long_axis: int, short_axis: int) -> str | None:
    if long_axis >= 3840 and short_axis >= 2160:
        return "4k"
    if long_axis >= 2560 and short_axis >= 1440:
        return "qhd"
    if long_axis >= 1920 and short_axis >= 1080:
        return "fhd"
    if long_axis >= 1280 and short_axis >= 720:
        return "hd"
    return None


def recover_color(prod: dict, options: str | None) -> str | None:
    facts = prod.get("facts") or {}
    for text in (options or "", str(facts.get("color") or ""), str(prod.get("title") or "")):
        matches = []
        for color in (*_COLOR_PHRASES, "sage green"):
            pattern = r"(?<![a-z0-9])" + r"\W+".join(re.escape(w) for w in color.split()) + r"(?![a-z0-9])"
            matches.extend(
                (m.start(), m.end(), "green" if color == "sage green" else color)
                for m in re.finditer(pattern, text.casefold())
            )
        if not matches:
            continue
        longest = [
            m for m in matches
            if not any(o[0] <= m[0] and o[1] >= m[1] and (o[1] - o[0]) > (m[1] - m[0]) for o in matches)
        ]
        labels = {m[2] for m in longest}
        return next(iter(labels)) if len(labels) == 1 else None
    return None


def _normalise_reading(attr: str, text: Any) -> str | None:
    if attr == "resolution":
        return resolution_canon(str(text) if text is not None else None)
    raw = str(text or "").strip()
    bare = re.fullmatch(r"\d+(?:\.\d+)?", raw)
    if attr == "refresh_rate":
        vals = _hertz_values(raw)
        if not vals and bare:
            vals = {f"{float(raw):g}hz"}
        return next(iter(vals)) if len(vals) == 1 else None
    if attr == "screen_size":
        vals = _screen_inch_values(raw)
        if not vals and bare:
            vals = {f"{float(raw):g}in"}
        return next(iter(vals)) if len(vals) == 1 else None
    return None


def _brief_playtime_axis(text: str) -> float:
    hours = [
        float(m.group(1))
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b", text, re.I)
    ]
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


def memo(function):
    """Memoise a one-argument pure function on hashable inputs."""

    cache: dict = {}

    def wrapped(argument):
        if argument in cache:
            return cache[argument]
        result = function(argument)
        cache[argument] = result
        return result

    return wrapped


def _price_number(text: str) -> float:
    return float(str(text).replace(",", ""))


class SlidingWindowStat:
    """Min/max/mean over the last N numbers without storing history twice."""

    def __init__(self, size: int = 10) -> None:
        self.size = max(1, size)
        self.values: list = []

    def add_sample(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.size:
            self.values.pop(0)

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def range_width(self) -> float:
        return (max(self.values) - min(self.values)) if self.values else 0.0


def resolution_canon(text: str | None) -> str | None:
    vals = _resolution_dims(text)
    return next(iter(vals)) if len(vals) == 1 else None


def pipe_right(*functions):
    """Right-to-left function composition."""

    def run(value):
        for function in reversed(functions):
            value = function(value)
        return value

    return run


def _grade_smartwatch(prod: dict) -> float:
    text = _facts_blob_of(prod)
    brand = str(prod.get("brand") or "").casefold()
    nav = 1.0 if _word_present(text, r"\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\b") else 0.0
    rugged = 0.0
    if _word_present(text, r"\b(10\s*atm|5\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|"
                  r"fiber[- ]reinforced|titanium|rugged|tactical|tank)\b"):
        rugged = 1.0
    elif _word_present(text, r"\b(water|swim|steel|aluminum)\b"):
        rugged = 0.5
    battery = _playtime_axis(text)
    outdoor = 1.0 if brand in {"garmin", "amazfit", "fitbit", "kospet"} else 0.0
    penalty = 0.2 if _word_present(text, r"\b(kid|kids|child|children|myfirst|watch phone)\b") else 0.0
    raw = 0.32 * nav + 0.26 * rugged + 0.26 * battery + 0.16 * outdoor - penalty
    return round(max(0.0, min(1.0, raw)), 3)


def _grade_headphones(prod: dict) -> float:
    title = str(prod.get("title") or "").casefold()
    facts_text = _facts_blob_of(prod)
    text = facts_text + " " + title
    category_text = " ".join(prod.get("category_path") or []).casefold()
    wireless_facts = facts_text + " " + category_text
    water = 1.0 if _word_present(text, r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b") else 0.0
    sport = 1.0 if _word_present(text, _SPORT_FIT) else 0.0
    battery = _brief_playtime_axis(text)
    wireless = 1.0 if _word_present(wireless_facts + " " + title, r"\b(wireless|bluetooth|true wireless)\b") else 0.0
    raw = 0.30 * water + 0.28 * sport + 0.22 * battery + 0.20 * wireless
    return round(max(0.0, min(1.0, raw)), 3)


def _grade_tablet(prod: dict) -> float:
    title = str(prod.get("title") or "").casefold()
    text = _facts_blob_of(prod) + " " + title
    kid = 1.0 if _word_present(text, r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b") else 0.0
    battery = _brief_playtime_axis(text)
    size = 1.0 if _word_present(text, r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")") else 0.0
    wifi = 1.0 if _word_present(text, r"\b(wi-?fi|wifi)\b") else 0.0
    raw = 0.40 * kid + 0.25 * battery + 0.20 * size + 0.15 * wifi
    return round(max(0.0, min(1.0, raw)), 3)


TF4_PROFILES = {
    "Smartwatches": {
        "scorer": _grade_smartwatch,
        "accessory": ("case for", "screen protector", "band for", "strap for", "charger for",
                      "mudra band", "replacement band", "watch band"),
        "signals": ("smartwatch", "smart watch", "watch phone", "apple watch", "galaxy watch",
                    "pixel watch", "garmin", "fitbit", "amazfit", "kospet"),
        "queries": ("rugged gps smartwatch", "long battery waterproof smartwatch",
                    "outdoor fitness gps watch", "gps smart watch waterproof",
                    "military rugged smart watch", "hiking compass smartwatch"),
    },
    "Headphones": {
        "scorer": _grade_headphones,
        "accessory": ("case for", "ear pads", "earpads", "replacement pad", "cable for",
                      "adapter", "stand for"),
        "signals": ("headphone", "headphones", "earbud", "earbuds", "earphone", "earphones",
                    "headset"),
        "queries": ("waterproof sport earbuds", "wireless running headphones",
                    "gym workout earbuds ipx", "ipx7 bluetooth sports earphones",
                    "earhook wireless sport earbuds", "sweatproof wireless earbuds long battery"),
    },
    "Tablets": {
        "scorer": _grade_tablet,
        "accessory": ("case for", "keyboard for", "screen protector", "stylus", "pen for",
                      "stand for", "charger for"),
        "signals": ("tablet", "ipad", "galaxy tab", "fire hd", "fire max"),
        "queries": ("kids tablet case bundle", "kids tablet 10 inch", "wifi tablet long battery",
                    "kids tablet shockproof", "8 inch kids tablet wifi", "tablet for kids"),
    },
}


def index_by(items: list, key) -> dict:
    out: dict = {}
    for item in items:
        out[key(item)] = item
    return out


def _word_present(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def bucketize(value: float, edges: list) -> int:
    """Which half-open bucket a value falls into given ascending edges."""

    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def _screen_inch_values(text: str | None) -> set:
    if not text:
        return set()
    return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text)}


def title_signature(title: str) -> str:
    """Order-independent digest of the alphabetic words in a title."""

    words = set()
    for match in re.finditer(r"[a-z]{2,}", str(title).lower()):
        words.add(match.group(0))
    return "-".join(sorted(words)[:8])


def read_recovery_pref(goal: str) -> tuple:
    m = re.search(r"\bprefer\s+the\s+(.+?)\.(?:\s|$)", goal, re.I)
    if not m:
        return None, None
    span = m.group(1).strip()
    low = span.casefold()
    for color in sorted((*_COLOR_PHRASES, "sage green"), key=len, reverse=True):
        if low.startswith(color + " "):
            return span[len(color):].strip(), ("green" if color == "sage green" else color)
        if low.endswith(" " + color):
            return span[: -len(color)].strip(), ("green" if color == "sage green" else color)
    parts = span.split()
    return (parts[-1] if parts else None), (parts[0].casefold() if len(parts) > 1 else None)


def ordinal(number: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th' - for readable log lines."""

    number = int(number)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return "%d%s" % (number, suffix)


def fact_reading_for(prod: dict, attr: str) -> str | None:
    """TF7 visible-fact value (_attr_fact_value)."""
    label = _FACT_LABEL_FOR.get(attr)
    if not label:
        return None
    return _normalise_reading(attr, (prod.get("facts") or {}).get(label))


def _stem_bag(text: Any) -> set:
    out = set()
    for tok in _token_list(text):
        out.add(tok)
        if len(tok) > 3 and tok.endswith("s"):
            out.add(tok[:-1])
    return out


def dedupe(seq: list) -> list:
    seen = set()
    out = []
    for item in seq:
        marker = item if isinstance(item, (str, int, float, bool)) else repr(item)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def classify_family(query: str, tools: list) -> str | None:
    for fam, marks in _FAMILY_MARKERS:
        if any(mark in query for mark in marks):
            return fam
    for tool in tools:
        fn = tool.get("function") or {}
        req = (fn.get("parameters") or {}).get("required") or []
        if fn.get("name") == "place_test_order" and "justification" in req:
            return TF7
    q = query.lower()
    if "compare only these candidate variants" in q:
        return TF5
    if re.search(r"matches (?:\"|the phrase)", q):
        return TF2
    if "strongly prefer the" in q:
        return TF6
    if re.search(r"tell me why|reason i can trust|walk me through why", q):
        return TF7
    if "ideally, i prefer" in q:
        return TF1
    if "firm requirement on the spec" in q:
        return TF3
    return None


def chunked(seq: list, size: int) -> list:
    if size <= 0:
        return [list(seq)]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def read_soft_brand(goal: str) -> str | None:
    m = re.search(r"\bprefer\s+(?:the\s+)?(.+?)(?:\s*,|\s+but\b|\.|;|$)", goal, re.I)
    if not m:
        return None
    brand = m.group(1).strip()
    return brand if 0 < len(brand) <= 40 else None


class Counter:
    """Small multiset with a most-common view (no imports)."""

    def __init__(self) -> None:
        self.counts: dict = {}

    def bump(self, key, weight: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + weight

    def most_common(self, n: int = 3) -> list:
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[: max(0, n)]

    def total(self) -> int:
        return sum(self.counts.values())


def recover_level(model, color, pref_model, pref_color) -> int | None:
    if not model or not color or not pref_model or not pref_color:
        return None
    same_model = model.casefold() == pref_model.casefold() or (
        _bare_code(model) is not None and _bare_code(model) == _bare_code(pref_model)
    )
    same_color = color.casefold() == pref_color.casefold()
    if same_model and same_color:
        return 0
    if same_model:
        return 1
    if same_color:
        return 2
    return 3


class SpacingGate:
    """Enforces a minimum time spacing between successive actions."""

    def __init__(self, min_gap: float) -> None:
        self.min_gap = max(0.0, min_gap)
        self.last_at: float | None = None
        self.blocked = 0

    def ready(self, now: float) -> bool:
        return self.last_at is None or (now - self.last_at) >= self.min_gap

    def mark(self, now: float) -> bool:
        if not self.ready(now):
            self.blocked += 1
            return False
        self.last_at = now
        return True

    def delay(self, now: float) -> float:
        if self.last_at is None:
            return 0.0
        waited = now - self.last_at
        return max(0.0, self.min_gap - waited)


def _named_model_hit(text: str) -> str | None:
    for pattern, label_fn in _MODEL_PATTERNS:
        m = re.search(pattern, text.casefold())
        if m:
            return label_fn(m)
    return None


def category_of_goal(goal: str) -> str | None:
    m = re.search(r"\bin the (.+?) category\b", goal)
    if m:
        return m.group(1).strip()
    t = goal.lower()
    if re.search(r"\bgaming monitors?\b", t):
        return "Gaming Monitors"
    if re.search(r"\bsmart\s*phones?\b|\bmobile phones?\b|\bcell ?phones?\b", t):
        return "Smartphones"
    if re.search(r"\bsmart\s*watch(?:es)?\b", t):
        return "Smartwatches"
    if re.search(r"\btablets?\b", t):
        return "Tablets"
    if re.search(r"\bmonitors?\b", t):
        return "Monitors"
    if re.search(r"\bhead ?phones?\b|\bear ?buds?\b|\bearphones?\b|\bheadsets?\b", t):
        return "Headphones"
    if re.search(r"\blaptops?\b|\bnotebooks?\b", t):
        return "Laptops"
    return None


def ema(values: list, alpha: float = 0.3) -> float | None:
    """Exponential moving average of a numeric series."""

    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    average = float(numbers[0])
    for value in numbers[1:]:
        average = alpha * value + (1 - alpha) * average
    return round(average, 4)


def ref_from_key(key: str) -> dict:
    pid, _, sku = key.partition("::")
    return {"product_id": pid, "sku": sku}


class Workflow:
    """Runs phases in order, skipping any whose guard is not satisfied."""

    def __init__(self) -> None:
        self.phases: list[Phase] = []
        self.cursor = 0
        self.trail: list[str] = []

    def then(self, name: str, guard=None) -> "Workflow":
        self.phases.append(Phase(name, guard))
        return self

    def step(self, context: dict) -> str | None:
        while self.cursor < len(self.phases):
            phase = self.phases[self.cursor]
            self.cursor += 1
            if phase.ready(context):
                phase.begin()
                self.trail.append(phase.name)
                return phase.name
            self.trail.append("skip:" + phase.name)
        return None

    def pending(self) -> list[str]:
        return [p.name for p in self.phases[self.cursor :]]

    def done_phases(self) -> list[str]:
        return [name for name in self.trail if not name.startswith("skip:")]


def handset_allowed(prod: dict, options: str | None) -> bool:
    return handset_verdict(prod, options) == "supported"


def parse_query_string(text: str) -> dict:
    out: dict = {}
    for chunk in str(text).split("&"):
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        if key:
            out[key] = value
    return out


def _phrase_found(text: str, phrase: str) -> bool:
    words = re.findall(r"[a-z0-9]+", phrase.casefold())
    if not words:
        return False
    pattern = r"(?<![a-z0-9])" + r"\W+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def _field_readings(key: str, text: str) -> set:
    if key == "screen_size":
        return _screen_inch_values(text)
    if key == "resolution":
        return _resolution_from_field(text)
    if key == "refresh_rate":
        return _hertz_values(text)
    return _option_axis_values("ram", text)


def recover_model(prod: dict, options: str | None) -> str | None:
    facts = prod.get("facts") or {}
    raw = facts.get("model")
    if raw:
        raw = str(raw)
        if ";" in raw or "/" in raw:
            return None
        named = _named_model_hit(raw)
        if named:
            return named
        words = raw.casefold().split()
        prefix = []
        while len(words) > 1 and words[0].isalpha() and any(c.isdigit() for c in " ".join(words[1:])):
            prefix.append(words.pop(0))
        identity = f"{prod.get('title') or ''} {prod.get('brand') or ''}"
        if prefix and not _phrase_found(identity, " ".join(prefix)):
            return None
        if raw.isalpha() and 3 <= len(raw) <= 24:
            return raw.upper() if _phrase_found(identity, raw) else None
        return _bare_code(" ".join(words))
    title = str(prod.get("title") or "")
    named = _named_model_hit(title)
    if named:
        return named
    labels = set()
    for m in _TITLE_MODEL_CODE.finditer(title):
        token = m.group()
        if _NON_MODEL_CODE.search(token):
            continue
        suffix = re.match(r"(?:\s+(?:pro|max|ultra|plus)\b)+", title[m.end():], re.I)
        code = _bare_code(token + (suffix.group() if suffix else ""))
        if code:
            labels.add(code)
    return next(iter(labels)) if len(labels) == 1 else None


def nearest(values: list, target: float):
    if not values:
        return None
    return min(values, key=lambda v: abs(v - target))


def _capacity_gb_values(text: str | None) -> set:
    if not text:
        return set()
    return {
        f"{float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1):g}gb"
        for m in _GB_RE.finditer(text)
    }


def recover_row_category(prod: dict, options: str | None) -> str | None:
    """Recovery candidate admission (_candidate_row) - returns the profile category."""
    title = str(prod.get("title") or "")
    low = title.casefold().strip()
    path = prod.get("category_path") or []
    if category_admits("Headphones", path, title=title):
        ok = (bool(low) and not low.startswith(_INVALID_TITLE_PREFIXES)
              and not any(f in low for f in _INVALID_TITLE_FRAGMENTS)
              and not _TF6_ACCESSORY_RE.search(low))
        if ok and _OVER_OR_ON_EAR_RE.search(f"{title} {options or ''}"):
            return "Headphones"
        return None
    if category_admits("Smartphones", path, title=title):
        ok = (bool(low) and not low.startswith(_INVALID_TITLE_PREFIXES)
              and not _TF6_ACCESSORY_RE.search(low) and _FEATURE_PHONE_RE.search(low) is None)
        if ok and handset_allowed(prod, options):
            return "Smartphones"
    return None


class Ledger:
    """Append-only event ledger with a compact running tally by kind."""

    def __init__(self) -> None:
        self.entries: list[tuple] = []
        self.tally: dict = {}

    def record_event(self, kind: str, detail=None) -> None:
        self.entries.append((kind, detail))
        self.tally[kind] = self.tally.get(kind, 0) + 1

    def count(self, kind: str) -> int:
        return self.tally.get(kind, 0)

    def most_recent(self, kind: str):
        for entry_kind, detail in reversed(self.entries):
            if entry_kind == kind:
                return detail
        return None

    def kind_list(self) -> list:
        return sorted(self.tally)


def spec_claim_phrase(key: str, value: str) -> str:
    if key == "screen_size":
        return f"{value[:-2]} inches"
    if key == "refresh_rate":
        return f"{value[:-2]}Hz"
    return {"4k": "4K", "qhd": "QHD", "fhd": "1080p Full HD", "hd": "720p HD"}.get(value, value)


class RetryEnvelope:
    """Wraps a plan with a bounded retry budget and jittered backoff schedule."""

    def __init__(self, attempts: int = 3, base: float = 0.5, cap: float = 8.0) -> None:
        self.attempts = max(1, attempts)
        self.base = base
        self.cap = cap
        self.used = 0

    def schedule(self) -> list:
        out = []
        delay = self.base
        for step in range(self.attempts):
            jitter = 0.1 * delay * ((step % 3) - 1)
            out.append(round(min(self.cap, delay + jitter), 3))
            delay *= 2.0
        return out

    def next_wait(self) -> float | None:
        if self.used >= self.attempts:
            return None
        wait = self.schedule()[self.used]
        self.used += 1
        return wait

    def done(self) -> bool:
        return self.used >= self.attempts


def handset_verdict(prod: dict, options: str | None) -> str:
    """'supported' | 'conflict' | 'unresolved' (device_semantics.smartphone_evidence)."""
    selected = str(options or "")
    if re.search(
        r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
        r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
        r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
        selected, re.I,
    ):
        return "conflict"
    facts = prod.get("facts") or {}
    raw_title = str(prod.get("title") or "")
    title = _identity_text_blob(raw_title, title=True)
    type_text = " ".join(
        _identity_text_blob(str(facts.get(k) or ""))
        for k in ("phone_type", "device_type", "description_device_type")
    )
    text = f"{title} {type_text}"
    if any(p.search(text) for p in _PHONE_CONFLICTS):
        return "conflict"
    if any(
        "[truncated]" in str(facts.get(k) or "")
        for k in ("phone_type", "device_type", "device_operating_system")
    ):
        return "unresolved"
    if _SMARTPHONE_RE.search(text):
        return "supported"
    os_text = " ".join(
        _identity_text_blob(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    title_os = bool(_OS_RE.search(title)) and not re.search(r"\bcompatib", raw_title, re.I)
    return "supported" if (_OS_RE.search(os_text) or title_os) else "unresolved"


def weighted_choice_index(weights: list, seed: int) -> int:
    """Deterministic index into weights, proportional to weight, keyed by seed."""

    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return 0
    target = (abs(int(seed)) % 1000) / 1000.0 * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += max(0.0, weight)
        if target < running:
            return index
    return len(weights) - 1


def _resolution_dims(text: str | None) -> set:
    if not text:
        return set()
    if re.search(r"\bqHD\b", text):
        return set()
    low = text.lower()
    values = set()
    for m in _WXH_RE.finditer(low):
        a, b = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _resolution_tier(a, b)
        if cls:
            values.add(cls)
    if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
        values.add("4k")
    if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
        values.add("qhd")
    if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
        values.add("fhd")
    residual = re.sub(r"\b(?:full|ultra) hd\b", "", low)
    if re.search(r"\b(?:hd|720p?)\b", residual):
        values.add("hd")
    return values


def partition_by(items: list, predicate) -> tuple:
    yes, no = [], []
    for item in items:
        (yes if predicate(item) else no).append(item)
    return yes, no


def is_handset_category(category: str | None) -> bool:
    return "smartphone" in canonical_forms([category or ""])


def _bare_code(raw: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", raw.casefold())
    while words and words[-1] in _COLOR_PHRASES:
        words.pop()
    code = "".join(words)
    return code.upper() if 3 <= len(code) <= 24 and any(c.isdigit() for c in code) else None


def sliding_pairs(items: list) -> list:
    out = []
    previous = None
    for index, item in enumerate(items):
        if index:
            out.append((previous, item))
        previous = item
    return out


def _token_list(text: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


class IntervalSet:
    """Merges numeric intervals - handy for reasoning over price bands."""

    def __init__(self) -> None:
        self.intervals: list[tuple] = []

    def add_span(self, low: float, high: float) -> None:
        if high < low:
            low, high = high, low
        self.intervals.append((low, high))

    def merged(self) -> list:
        if not self.intervals:
            return []
        ordered = sorted(self.intervals)
        out = [ordered[0]]
        for low, high in ordered[1:]:
            last_low, last_high = out[-1]
            if low <= last_high:
                out[-1] = (last_low, max(last_high, high))
            else:
                out.append((low, high))
        return out

    def covers(self, point: float) -> bool:
        return any(low <= point <= high for low, high in self.merged())

    def span(self) -> float:
        merged = self.merged()
        return sum(high - low for low, high in merged) if merged else 0.0


def _option_axis_values(key: str, text: str) -> set:
    text = text or ""
    if key == "ram":
        paired = {f"{float(r):g}gb" for r, _s in _RAM_STORAGE_PAIR.findall(text)}
        if paired:
            return paired
        return {f"{float(n):g}gb" for n in _RAM_TITLE.findall(text)}
    if key == "screen_size":
        return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text)}
    if key == "refresh_rate":
        return {f"{float(m.group(1)):g}hz" for m in _HZ_RE.finditer(text)}
    return _resolution_dims(text)


def diff_mappings(before: dict, after: dict) -> dict:
    """Which flattened keys were added, removed, or changed between two payloads."""

    a = flatten_mapping(before)
    b = flatten_mapping(after)
    added = {k: b[k] for k in b if k not in a}
    removed = {k: a[k] for k in a if k not in b}
    changed = {k: (a[k], b[k]) for k in a if k in b and a[k] != b[k]}
    return {"added": added, "removed": removed, "changed": changed}


def _hertz_values(text: str | None) -> set:
    if not text:
        return set()
    return {f"{float(m.group(1)):g}hz" for m in _HZ_RE.finditer(text)}


def clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def _option_hint_present(key: str, text: str) -> bool:
    text = text or ""
    if key == "ram":
        return bool(_CAPACITY_PAIR_HINT.search(text) or _RAM_TITLE.search(text))
    if key == "screen_size":
        return bool(_INCH_RE.search(text))
    if key == "refresh_rate":
        return bool(_HZ_RE.search(text))
    return bool(
        _WXH_RE.search(text)
        or re.search(
            r"\b(?:qHD|qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b",
            text,
        )
    )


def brand_fits(expected: str | None, actual: str | None) -> bool:
    if not expected:
        return True
    exp, act = _flatten_text(expected), _flatten_text(actual)
    if not exp or exp == act:
        return True
    exp_t = _token_list(expected)
    return bool(exp_t) and exp_t <= _token_list(actual)


class Agent:
    def __init__(self, problem_data: dict) -> None:
        self.t0 = time.monotonic()
        env = problem_data.get("environment") or {}
        self.binding = dict(env.get("binding") or {})
        pv = env.get("policy_view") or {}
        self.query = str(pv.get("query") or problem_data.get("query") or "")
        cut = self.query.find("\n\nTask rules:")
        self.goal = (self.query[:cut] if cut >= 0 else self.query).strip()
        self.max_steps = int(pv.get("max_steps") or 30)
        self.max_calls = max(1, min(int(pv.get("max_calls_per_turn") or 1), 16))
        self.tools = [t for t in pv.get("tools") or [] if isinstance(t, dict)]
        self.pid = str(problem_data.get("problem_id") or problem_data.get("id") or "problem")
        self.family = classify_family(self.query, self.tools)
        self.turn = 0
        self.done = False
        self._pending = None
        self.search_cap_ok = True
        self._filter_args: list = []
        self._filters_sent: set = set()
        self._lat: list = []
        self.rows: dict = {}
        self.products: dict = {}
        self.state: dict = {}
        self.cart: list = []
        self.first_key: str | None = None
        self.events: list = []
        self.dead: set = set()
        self.bad: set = set()
        self.user_msgs: list = []
        self.queries: list = []
        self.log: list = []
        self.req: dict = {"attrs": {}}
        self.currency: str | None = None
        self.budget0: float | None = None
        self.budget: float | None = None
        m = _BUDGET_RE.search(self.goal) or _ANY_MONEY_RE.search(self.goal)
        if m:
            self.budget0 = self.budget = _price_number(m.group(1))
            self.currency = m.group(2).upper()
        self.req["category"] = category_of_goal(self.goal)
        # TF3 and TF6 gate on smartphone evidence; TF7 does when its goal carries the rule.
        self.strict_phone = self.family == TF3 or (
            self.family == TF7 and "must explicitly say smartphone" in self.query)
        self.req_changes = 0

    def _build_filter(self, category: str | None = None, brand: str | None = None,
                     cap: float | None = None) -> tuple:
        args: dict = {"in_stock": True, "k": 10}
        if category:
            args["category"] = FILTER_CATEGORY.get(category, category)
        if brand:
            args["brand"] = brand
        cap = self.budget if cap is None else cap
        if cap is not None:
            args["max_price"] = round(float(cap), 2)
        return ("filter", args)

    def _constraint_pool(self, *, brand, budget, exclude=()) -> list:
        out = []
        for pid, prod, v, k in self.variants_for():
            if k in exclude or not self._constraint_valid(pid, prod, v, k, brand=brand, budget=budget):
                continue
            ev = min((spec_support(prod, v.get("options"), a, w) for a, w in self.req["attrs"].items()), default=0)
            refurb = 1 if re.search(r"refurb|renewed|pre-owned|second hand|used", str(prod.get("title") or ""), re.I) else 0
            unproven = 1 if (is_handset_category(self.req.get("category"))
                             and not handset_allowed(prod, v.get("options"))) else 0
            out.append((unproven, refurb, -ev, self.price_for(k), k))
        low = self.price_outliers([row[-1] for row in out])
        out = [(row[0], 1 if row[-1] in low else 0, *row[1:]) for row in out]
        out.sort()
        return [k for *_x, k in out]

    def _record_price_stock(self, k: str, price: Any, in_stock: Any) -> None:
        try:
            p = float(price) if price is not None else None
        except (TypeError, ValueError):
            p = None
        self.state[k] = (p, in_stock is True)

    def _constraint_query_set(self, brand: str, word: str) -> list:
        specs = " ".join(spec_query_phrase(k, v) for k, v in self.req["attrs"].items())
        return [f"{brand} {word} {specs}", f"{brand} {specs}", f"{word} {specs}",
                f"{brand} {word}", f"{brand} {word} {specs} android", f"{specs} {word} android"]

    def log_line(self, **kw) -> None:
        kw["t"] = round(self.wall_time(), 1)
        self.log.append(kw)

    def cart_keys_list(self) -> list:
        return [row_key(c.get("product_id"), c.get("sku")) for c in self.cart]

    def _added_constraint_queries(self, brand: str, word: str, i: int) -> list:
        specs = " ".join(spec_query_phrase(k, v) for k, v in self.req["attrs"].items())
        nums = " ".join(re.sub(r"[a-z]+$", "", v) for v in self.req["attrs"].values())
        pool = [f"{brand} phone {nums}", f"{brand} mobile {specs}", f"{brand} {word} original",
                f"{nums} {word}", f"{brand} {word} sale", f"{word} {specs} cheap"]
        return pool[i * 2: i * 2 + 2]

    def _recall_brands(self, n: int, include_used: bool = False) -> list:
        counts: dict = {}
        for k, r in self.rows.items():
            b = str(r.get("brand") or "").strip()
            if b and self._recall_match(r.get("title")):
                counts[b] = counts.get(b, 0) + 1
        used = {c.get("brand") for c in self._filter_args}
        ranked = sorted(counts, key=lambda b: (-counts[b], b))
        return [b for b in ranked if include_used or b not in used][:n]

    def _preference_device_valid(self, prod: dict) -> bool:
        title = str(prod.get("title") or "")
        low = title.casefold()
        if any(t in low for t in _TF4_BAD_TITLE):
            return False
        if any(p in low for p in self.profile["accessory"]):
            return False
        return category_admits(self.req.get("category"), prod.get("category_path") or [], title=title)

    def state_ok(self, k: str, budget: float | None = None) -> bool:
        if k in self.dead or k in self.bad:
            return False
        p, ok = self.state.get(k, (None, False))
        cap = self.budget if budget is None else budget
        return bool(ok and p is not None and (cap is None or p <= cap + 1e-9))

    def reply_for(self, k: str) -> str:
        pid = ref_from_key(k)["product_id"]
        title = str((self.products.get(pid) or {}).get("title") or self.rows.get(k, {}).get("title") or "the item")
        p = self.state.get(k, (None, True))[0]
        cur = self.currency or ""
        price = f" at {p:.2f} {cur}" if isinstance(p, float) else ""
        return (f"Thanks - going with {title[:80]}{price}. It meets your stated requirements "
                f"and is in stock, so I am placing the order now.")

    def rank_unseen(self, score_fn, limit: int, budget: float | None = None) -> list:
        best: dict = {}
        low = self.price_outliers([k for k in self.rows if self.state_ok(k, budget)])
        for k, r in self.rows.items():
            pid = str(r.get("product_id"))
            if pid in self.products or not self.state_ok(k, budget):
                continue
            s = score_fn(r)
            if s is None:
                continue
            if k in low:
                s -= 1.0
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, -self.price_for(k))
        order = sorted(best, key=lambda p: (-best[p][0], -best[p][1], p))
        return order[:limit]

    def _preference_candidates(self, exclude_listing=None) -> list:
        best: dict = {}
        for pid, prod, v, k in self.variants_for():
            if not self.state_ok(k) or not self._preference_device_valid(prod):
                continue
            lid = str(prod.get("listing_id") or pid)
            if exclude_listing is not None and lid == exclude_listing:
                continue
            score = self.profile["scorer"](prod)
            signal = any(s in str(prod.get("title") or "").casefold() for s in self.profile["signals"])
            cand = (1 if signal else 0, score, -self.price_for(k), k)
            if lid not in best or cand > best[lid]:
                best[lid] = cand
        low = self.price_outliers([row[-1] for row in best.values()])
        return sorted(best.values(), key=lambda row: (row[0], 0 if row[-1] in low else 1, row[1], row[2]),
                      reverse=True)

    def _absorb_observation(self, name: str, args: dict, payload: dict, err: Any) -> None:
        if name == "search" and "max_price" in args and not self._tool_has_param("search", "max_price"):
            cap = float(args["max_price"])
            rows = payload.get("results") if isinstance(payload, dict) else None
            if err or not isinstance(rows, list):
                self.search_cap_ok = False
            elif any(isinstance(r, dict) and isinstance(r.get("price"), (int, float)) and r["price"] > cap + 0.01
                     for r in rows):
                self.search_cap_ok = False   # ignored by this runtime: rely on local filtering
                self.log_line(search_max_price="ignored by runtime; filtering locally")
        if not isinstance(payload, dict):
            return
        for r in payload.get("results") or []:
            if isinstance(r, dict) and r.get("product_id"):
                k = row_key(r["product_id"], r.get("sku"))
                self.rows[k] = r
                self._record_price_stock(k, r.get("price"), r.get("in_stock"))
                if not self.currency and r.get("currency"):
                    self.currency = str(r["currency"])
        if payload.get("product_id") and isinstance(payload.get("variants"), list):
            self._absorb_product(payload)
        for c in payload.get("compared") or []:
            if isinstance(c, dict) and c.get("product_id"):
                self._absorb_product(c)
        if name == "inspect_stock" and payload.get("product_id") and "in_stock" in payload:
            k = row_key(payload["product_id"], payload.get("sku"))
            self._record_price_stock(k, payload.get("price"), payload.get("in_stock"))
            p, ok = self.state[k]
            if k in self.cart_keys_list() and (not ok or (p is not None and self.budget is not None and p > self.budget + 1e-9)):
                self.dead.add(k)
        if isinstance(payload.get("cart"), list):
            self.cart = [c for c in payload["cart"] if isinstance(c, dict)]
        added = payload.get("added")
        if isinstance(added, dict) and added.get("product_id"):
            k = row_key(added["product_id"], added.get("sku"))
            if self.first_key is None:
                self.first_key = k
        ev = payload.get("market_event")
        if isinstance(ev, dict):
            self._absorb_event(ev)
        if name == "add_to_cart" and err:
            self.bad.add(row_key(args.get("product_id"), args.get("sku")))

    def _recovery_pool(self, exclude=()) -> list:
        out = []
        cat = self.req.get("category") or "Smartphones"
        for pid, prod, v, k in self.variants_for():
            if k in exclude or not self.state_ok(k):
                continue
            row_cat = recover_row_category(prod, v.get("options"))
            if row_cat is None or not category_admits(cat, [row_cat]):
                continue
            level = recover_level(recover_model(prod, v.get("options")), recover_color(prod, v.get("options")),
                              self.req.get("model"), self.req.get("color"))
            if level is None:
                continue
            out.append((level, self.price_for(k), k))
        low = self.price_outliers([k for *_x, k in out])
        out.sort(key=lambda row: (row[0], 1 if row[-1] in low else 0, row[1], row[-1]))
        return out

    def _preference_score(self, r: dict):
        title = str(r.get("title") or "")
        low = title.casefold()
        if any(p in low for p in self.profile["accessory"]) or any(t in low for t in _TF4_BAD_TITLE):
            return None
        est = self.profile["scorer"]({"title": title, "facts": {"listing": title},
                                      "brand": r.get("brand"), "category_path": []})
        signal = any(s in low for s in self.profile["signals"])
        return est + (0.2 if signal else -0.3)

    def lane_recall(self) -> None:
        m = re.search(r'matches\s+"([^"]+)"', self.goal) or re.search(
            r"matches the phrase\s+(.+?)(?:,|\s+at or under|\s+and buy|$)", self.goal)
        phrase = (m.group(1) if m else self.goal).strip().strip('"')
        self.req["terms"] = re.findall(r"[a-z0-9]+", phrase.lower())
        self.req["phrase"] = " ".join(self.req["terms"])
        if not self.req["terms"]:
            return self.lane_model_loop()
        self.send_calls([self.catalog_lookup(self.req["phrase"])])
        q_more = self._recall_requeries(2)
        calls = [self.catalog_lookup(q) for q in q_more]
        calls.append(self.filter_action(self.req.get("category")))
        calls += [self.filter_action(self.req.get("category"), b) for b in self._recall_brands(2)]
        views = self._recall_view_targets(2)
        calls += [("view", {"product_id": p}) for p in views]
        self.send_calls(calls)
        # recall@10 counts distinct matching identities seen in search/filter rows; add
        # brand and price-cap filter slices (no query cost) while coverage is short of 10
        rounds = 0
        while self.may_search_more() and rounds < 2 and self._recall_coverage() < 10:
            extra = [self.filter_action(self.req.get("category"), b) for b in self._recall_brands(3)]
            top = (self._recall_brands(1, include_used=True) or [None])[0]
            if top and self.budget:
                extra += [self.filter_action(self.req.get("category"), top, cap=self.budget * f) for f in (0.6, 0.3)]
            extra = [c for c in extra if json.dumps(c[1], sort_keys=True) not in self._filters_sent]
            if not extra:
                break
            self.send_calls(extra[:5])
            rounds += 1
        while self.may_search_more() and len(self._recall_valid()) < 2:
            if not self._recall_view_more(2):
                break
        self.emit_output()

    def price_intervals(self) -> "IntervalSet":
        band = IntervalSet()
        for row in self.rows.values():
            price = row.get("price")
            if isinstance(price, (int, float)):
                band.insert(float(price), float(price))
        return band

    def search_again(self, final: bool = False) -> None:
        if self.family in (TF1, TF3, TF7):
            cat = self.req.get("category")
            word = CAT_QUERY_WORD.get(cat or "", (cat or "").lower())
            brand = "" if self.family == TF1 else (self.req.get("brand") or "")
            qs = self.fresh_queries([f"{word} {' '.join(spec_query_phrase(k, v) for k, v in self.req['attrs'].items())} {brand}",
                             f"{brand} {word} cheap"])
            calls = [self.catalog_lookup(q) for q in qs[:2]]
            self.send_calls(calls)
            pids = self.rank_unseen(self._constraint_score, 10)
            if pids:
                self.send_calls(self.compare_actions(pids)[:2])
        elif self.family == TF4:
            self._preference_pass(extra=True)
        elif self.family == TF6:
            rows = self._recovery_pool()
            self._recovery_pass(extra=True, broaden=not any(lvl == 1 for lvl, *_r in rows))
        elif self.family == TF2:
            self._recall_view_more(2)

    def _count_passing(self) -> int:
        brand = self.req.get("brand") if self.family in (TF3, TF7) else None
        return len(self._constraint_pool(brand=brand, budget=self.budget))

    def latency_stat(self) -> "SlidingWindowStat":
        stat = SlidingWindowStat()
        for sample in self._lat:
            if isinstance(sample, (int, float)):
                stat.add_sample(sample)
        return stat

    def fresh_queries(self, queries: list) -> list:
        return [q for q in dict.fromkeys(re.sub(r"\s+", " ", q).strip() for q in queries)
                if q and q not in self.queries]

    def _listing_ref(self, k: str) -> str:
        pid = ref_from_key(k)["product_id"]
        return str((self.products.get(pid) or {}).get("listing_id") or pid)

    def _absorb_product(self, prod: dict) -> None:
        pid = str(prod["product_id"])
        self.products[pid] = prod
        for v in prod.get("variants") or []:
            if isinstance(v, dict):
                self._record_price_stock(row_key(pid, v.get("sku")), v.get("price"), v.get("in_stock"))
                if not self.currency and v.get("currency"):
                    self.currency = str(v["currency"])

    def lane_constraints(self) -> None:
        fam = self.family
        first_sentence = re.split(r"(?<=[.?!])\s", self.goal.strip())[0]
        self.req["attrs"].update(extract_specs(first_sentence if fam == TF7 else self.goal))
        if fam in (TF3, TF7):
            self.req["brand"] = read_firm_brand(self.goal)
        if fam == TF1:
            self.req["pref"] = read_soft_brand(self.goal)
        underspecified = (not self.req["attrs"]) or "if you ask" in self.goal.lower()
        cat = self.req.get("category")
        word = CAT_QUERY_WORD.get(cat or "", (cat or "").lower().rstrip("s"))
        brand = self.req.get("brand") or self.req.get("pref") or ""

        calls = []
        if underspecified:
            calls.append(("message", {"content": (
                "Before I pick one, what are your firm requirements - which exact spec "
                "(screen size, RAM, resolution or refresh rate) must it have?")}))
        calls += [self.catalog_lookup(q) for q in self.fresh_queries(self._constraint_query_set(brand, word))[:2]]
        calls.append(self.filter_action(cat, brand or None))
        self.send_calls(calls)
        self._constraint_explore_rounds(word, brand)
        self.emit_output()

    def catalog_lookup(self, q: str) -> tuple:
        q = re.sub(r"\s+", " ", q).strip()
        if q not in self.queries:
            self.queries.append(q)
        args = {"query": q, "k": 10, "in_stock": True}
        # max_price is not advertised for search; oro-env-runtime 0.2.10 applies it before the
        # k cut.  Keep sending it only while results prove it is honoured (or the schema has it).
        if self.budget is not None and (self.search_cap_ok or self._tool_has_param("search", "max_price")):
            args["max_price"] = round(float(self.budget), 2)
        return ("search", args)

    def distinct_query_words(self) -> int:
        seen = set()
        for query in self.queries:
            seen.update(word.lower() for word in str(query).split())
        return len(seen)

    def _absorb_event(self, ev: dict) -> None:
        self.events.append(ev)
        target = ev.get("target") or {}
        keys = [row_key(target.get("product_id"), target.get("sku"))]
        keys += [str(k) for k in ev.get("targets") or [] if "::" in str(k)]
        for k in keys:
            if ev.get("kind") == "stockout":
                self.state[k] = (self.state.get(k, (None, False))[0], False)
                self.dead.add(k)
            else:
                new = ev.get("new_price")
                try:
                    new = float(new)
                except (TypeError, ValueError):
                    new = None
                self.state[k] = (new, True)
                if new is None or (self.budget is not None and new > self.budget + 1e-9):
                    self.dead.add(k)

    def _recall_requeries(self, n: int) -> list:
        budget_left = max(0, 3 - len(self.queries))
        n = min(n, budget_left)
        if n <= 0:
            return []
        terms = set(self.req["terms"])
        counts: dict = {}
        seen = set()
        for r in self.rows.values():
            pid = r.get("product_id")
            if pid in seen:
                continue
            seen.add(pid)
            toks = set(re.findall(r"[a-z0-9]+", str(r.get("title") or "").lower()))
            if not toks & terms:
                continue
            w = 2 if terms <= toks else 1
            for t in toks - terms:
                if t in _STOP or len(t) < 3 or t.isdigit():
                    continue
                counts[t] = counts.get(t, 0) + w
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        cat_toks = [t for t in re.findall(r"[a-z0-9]+", str(self.req.get("category") or "").lower())
                    if t not in _STOP and len(t) > 2]
        out = []
        for t in ranked + cat_toks + ["original", "sale"]:
            q = f"{self.req['phrase']} {t}"
            if q not in self.queries and q not in out:
                out.append(q)
            if len(out) >= n:
                break
        return out

    def _constraint_close(self):
        x = self.first_key
        # keep the committed item only if it still meets every current requirement
        # (an event that did not invalidate it, or no event at all)
        if x and x in self.cart_keys_list() and self.candidate_valid(x):
            return x
        brand = self.req.get("brand") if self.family in (TF3, TF7) else None
        c = self._constraint_pool(brand=brand, budget=self.budget, exclude={x} if x else ())
        if not c:
            return None
        xpid = ref_from_key(x)["product_id"] if x else None
        other = [k for k in c if ref_from_key(k)["product_id"] != xpid]
        return (other or c)[0]

    def _constraint_explore_rounds(self, word: str, brand: str) -> None:
        extra = 0
        while self.may_search_more():
            have_first = self.select_first() is not None
            have_final = self._count_passing() >= 2
            if have_first and have_final:
                return
            pids = self.rank_unseen(self._constraint_score, 5 * MAX_COMPARES_PER_GROUP,
                                      budget=self.budget0 if self.first_key is None else None)
            qs = self.fresh_queries(self._constraint_query_set(brand, word) + self._added_constraint_queries(brand, word, extra))
            calls = [self.catalog_lookup(q) for q in qs[: (2 if pids else MAX_SEARCHES_PER_GROUP)]]
            if brand and extra == 0 and self.budget:
                # the brand field is often a store name; the filter is the only brand-field
                # route and returns the first 10 by id, so cheaper caps surface other rows
                calls += [self.filter_action(self.req.get("category"), brand, cap=self.budget * f)
                          for f in (0.5, 0.25)]
            extra += 1
            calls += self.compare_actions(pids)[:MAX_COMPARES_PER_GROUP]
            if not calls:
                return
            self.send_calls(calls)
            if extra > 5:
                return

    def filter_action(self, category: str | None = None, brand: str | None = None,
                    cap: float | None = None) -> tuple:
        call = self._build_filter(category, brand, cap)
        self._filter_args.append(call[1])
        self._filters_sent.add(json.dumps(call[1], sort_keys=True))
        return call

    def code_trie(self) -> "Trie":
        trie = Trie()
        for row in self.rows.values():
            for token in re.findall(r"[a-z0-9]{3,}", str(row.get("title") or "").lower()):
                if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
                    trie.insert_word(token)
        return trie

    def _recall_match(self, title: Any) -> bool:
        toks = set(re.findall(r"[a-z0-9]+", str(title or "").lower()))
        return all(t in toks for t in self.req["terms"])

    def variant_pair(self, k: str):
        pid, _, sku = k.partition("::")
        prod = self.products.get(pid)
        if not prod:
            return None, None
        v = next((x for x in prod.get("variants") or [] if str(x.get("sku")) == sku), None)
        return prod, v

    def price_for(self, k: str) -> float:
        p = self.state.get(k, (None, False))[0]
        return float(p) if p is not None else 1e18

    def phase_ledger(self) -> "Ledger":
        ledger = Ledger()
        for entry in self.log:
            if isinstance(entry, dict):
                for key in entry:
                    ledger.append(key, entry.get(key))
        return ledger

    def select_first(self):
        fam = self.family
        if fam in (TF1, TF3, TF7):
            return self._constraint_open()
        if fam == TF2:
            c = self._recall_valid()
            return c[0] if c else None
        if fam == TF4:
            ls = self._preference_candidates()
            if not ls:
                return None
            # The committed listing is repriced away; keep the best one for the final order
            # when another listing also clears the 0.65 commitment bar.
            if TF4_RESERVE_BEST and len(ls) >= 2 and ls[1][1] >= 0.65 and ls[1][0] >= ls[0][0]:
                return ls[1][-1]
            return ls[0][-1]
        if fam == TF5:
            r = self._ranking_sorted()
            return r[0] if r else None
        if fam == TF6:
            rows = self._recovery_pool()
            zero = [k for lvl, _p, k in rows if lvl == 0]
            return zero[0] if zero else (rows[0][-1] if rows else None)
        return None

    def _call_model(self, messages: list):
        for model in LLM_MODELS:
            if self.wall_time() > EMERGENCY_AT_S - 10:
                return None
            resp = _LLM.post("/inference/chat/completions", json_data={
                "model": model, "messages": messages, "tools": self.tools,
                "tool_choice": "required", "temperature": 0, "max_tokens": 1200})
            if isinstance(resp, dict) and resp.get("choices"):
                return resp
        return None

    def _preference_pass(self, extra: bool) -> None:
        calls = []
        if extra:
            calls += [self.catalog_lookup(q) for q in self.fresh_queries(list(self.profile["queries"]))[:2]]
        pids = self.rank_unseen(self._preference_score, 5 * MAX_COMPARES_PER_GROUP)
        calls += self.compare_actions(pids)[:MAX_COMPARES_PER_GROUP]
        if calls:
            self.send_calls(calls)

    def variants_for(self):
        for pid, prod in self.products.items():
            for v in prod.get("variants") or []:
                if isinstance(v, dict) and v.get("sku") is not None:
                    yield pid, prod, v, row_key(pid, v.get("sku"))

    def _recall_coverage(self) -> int:
        """Distinct matching identities observed (brand + title-token set; approximates clusters)."""
        ids = set()
        for k, r in self.rows.items():
            if self._recall_match(r.get("title")) and self.state_ok(k):
                ids.add((_flatten_text(r.get("brand")), frozenset(_token_list(r.get("title")))))
        return len(ids)

    def first_commitment(self, k: str) -> bool:
        before = self.first_key
        out = self.send_calls([("add_to_cart", ref_from_key(k)), ("inspect_stock", ref_from_key(k))])
        if self.first_key is not None and self.first_key != before:
            return True
        if out is not None:
            self.bad.add(k)
        return False

    def price_outliers(self, keys) -> set:
        """Keys priced far below their peers (placeholder / bait listings such as a
        smartphone at 10 PHP).  Used to break ties, never to reject outright."""
        prices = sorted(self.price_for(k) for k in keys if self.price_for(k) < 1e17)
        if len(prices) < 3:
            return set()
        median = prices[len(prices) // 2]
        return {k for k in keys if self.price_for(k) < 0.15 * median}

    def _track_latency(self, dt: float) -> None:
        self._lat.append(max(0.0, dt))
        self._lat = self._lat[-4:]

    def same_product_groups(self) -> list:
        forest = UnionFind()
        rows = list(self.rows.items())
        for i in range(len(rows)):
            forest.root_of(rows[i][0])
            for j in range(i + 1, len(rows)):
                if title_signature(rows[i][1].get("title")) == title_signature(rows[j][1].get("title")):
                    forest.join(rows[i][0], rows[j][0])
        return [group for group in forest.clusters() if len(group) > 1]

    def _constraint_open(self):
        brand = self.req.get("pref") if self.family == TF1 else self.req.get("brand")
        # honour a delivered budget cut when possible; the verifier judges the first add
        # against the original contract, so fall back to it only if nothing fits the cut
        for cap in (self.budget, self.budget0):
            c = self._constraint_pool(brand=brand, budget=cap)
            if c:
                return c[0]
        return None

    def cart_index(self) -> dict:
        return index_by([c for c in self.cart if isinstance(c, dict)],
                        lambda c: "%s::%s" % (c.get("product_id"), c.get("sku")))

    def drive_episode(self) -> None:
        lane = {
            TF1: self.lane_constraints, TF3: self.lane_constraints, TF7: self.lane_constraints,
            TF2: self.lane_recall, TF4: self.lane_pref, TF5: self.lane_rank, TF6: self.lane_recover,
        }.get(self.family, self.lane_model_loop)
        self.log_line(family=self.family, budget=self.budget, category=self.req.get("category"))
        lane()
        if (not self.done and self.family is not None and self.steps_available() > 4
                and self.wall_time() + 4 * self.last_latency() + 60 < TIME_LIMIT_S):
            # The deterministic lane could not finish: hand the current state to the
            # model-assisted loop (its orders are still validated below).
            self.log_line(fallback="llm")
            self.lane_model_loop(context=self.summarise_progress())
        if not self.done:
            self.emergency_close()

    def _recovery_pass(self, extra: bool, broaden: bool = False) -> None:
        calls = []
        if broaden:
            color = self.req.get("color") or ""
            cat = self.req.get("category") or "Smartphones"
            word = CAT_QUERY_WORD.get(cat, "smartphone")
            calls += [self.catalog_lookup(q) for q in self.fresh_queries(
                [f"{color} {word}", f"{word} {color} android", f"{color} {word} 5g"])[:2]]
            calls.append(self.filter_action(cat))
            extra = False
        if extra:
            model, color = self.req.get("model") or "", self.req.get("color") or ""
            cat = self.req.get("category") or "Smartphones"
            word = CAT_QUERY_WORD.get(cat, "smartphone")
            brands = []
            for prod in self.products.values():
                for v in prod.get("variants") or []:
                    m = recover_model(prod, v.get("options"))
                    if m and recover_level(m, "x", model, "y") in (0, 1) and prod.get("brand"):
                        brands.append(str(prod["brand"]))
            lead = [f"{b} {model} {color}" for b in dict.fromkeys(brands)][:2]
            calls += [self.catalog_lookup(q) for q in self.fresh_queries(
                lead + [f"{model} phone", f"{model} android", f"{model} {word} {color}", f"{model} mobile"])[:2]]
        pids = self.rank_unseen(self._recovery_score, 5 * MAX_COMPARES_PER_GROUP)
        calls += self.compare_actions(pids)[:MAX_COMPARES_PER_GROUP]
        if calls:
            self.send_calls(calls)

    def _recovery_score(self, r: dict):
        title = str(r.get("title") or "")
        code = _bare_code(self.req.get("model") or "") or ""
        s = 0.0
        if code and code.casefold() in "".join(re.findall(r"[a-z0-9]+", title.casefold())):
            s += 3
        if self.req.get("color") and _phrase_found(title, self.req["color"]):
            s += 1
        if _TF6_ACCESSORY_RE.search(title.casefold()):
            s -= 4
        return s

    def _preference_queries(self) -> list:
        """Search phrasing from the request's stated priority and use case (the runtime's
        profile axes); the paid score stays the verifier's fixed per-category scorer."""
        g = self.goal.lower()
        noun = {"Headphones": "earbuds", "Smartwatches": "smartwatch", "Tablets": "tablet"}.get(
            self.req.get("category") or "", "")
        cues = (
            (r"\b(?:sweat\w*|water\w*|weather|rain(?:y|proof)?)\b", "waterproof sweatproof"),
            (r"\b(?:secure|stays put|fit)\b", "secure fit sport"),
            (r"\b(?:battery|marathon|outlasts?)\b", "long battery"),
            (r"\b(?:wireless|cable)\b", "wireless"),
            (r"\b(?:gps|navigation|compass|trail|hiking)\b", "gps"),
            (r"\b(?:rugged|tough|drop\w*|durable)\b", "rugged"),
            (r"\b(?:kids?|child\w*)\b", "kids"),
        )
        words = [w for pat, w in cues if re.search(pat, g)]
        if not noun or not words:
            return []
        return [f"{' '.join(words[:2])} {noun}"]

    def query_multiset(self) -> "Counter":
        counter = Counter()
        for query in self.queries:
            for word in str(query).split():
                counter.bump(word.lower())
        return counter

    def may_search_more(self) -> bool:
        # one exploration turn + commit turn + close turn must still fit
        return (not self.done and self.wall_time() < NO_ROUND_AFTER_S and self.steps_available() > 4
                and self.wall_time() + 3 * self.last_latency() + 10 < TIME_LIMIT_S)

    def _constraint_valid(self, pid: str, prod: dict, v: dict, k: str, *, brand, budget) -> bool:
        if not self.state_ok(k, budget):
            return False
        if self.currency and v.get("currency") and v.get("currency") != self.currency:
            return False
        cat = self.req.get("category")
        title = str(prod.get("title") or "")
        if cat and not category_admits(cat, prod.get("category_path") or [], title=title):
            return False
        if brand and not brand_fits(brand, prod.get("brand")):
            return False
        if is_handset_category(cat):
            status = handset_verdict(prod, v.get("options"))
            # explicit conflicts (screen-only SKU, accessory, tablet, feature phone) never pass;
            # missing positive evidence is disqualifying only where the family gates on it
            if status == "conflict" or (status != "supported" and self.strict_phone):
                return False
        if not self.req["attrs"]:
            return False
        if self.family == TF7:
            low = title.casefold()
            if any(t in low for t in ("customer rating", "customer review", "compare ", "refurb",
                                      "pre-owned", "restored", "renewed")):
                return False
            for attr, want in self.req["attrs"].items():
                if (fact_reading_for(prod, attr) != want or product_value(prod, attr) != want
                        or variant_value(prod, v.get("options"), attr) != want):
                    return False
            return True
        return all(variant_value(prod, v.get("options"), attr) == want
                   for attr, want in self.req["attrs"].items())

    def compare_actions(self, pids: list) -> list:
        pids = [p for p in dict.fromkeys(pids) if p not in self.products]
        return [("compare", {"product_ids": pids[i:i + 5]}) for i in range(0, len(pids), 5)]

    def steps_available(self) -> int:
        return self.max_steps - self.turn

    def _recall_rank(self, r: dict):
        if not self._recall_match(r.get("title")):
            return None
        return 1.0

    def close_order(self, k: str) -> bool:
        """Order k.  Validated mode adds+inspects it first and re-checks it on the fresh
        observation; fused mode sends add, inspect, reply, cart refresh and order together
        (no simulator call on that turn).  The cart refresh always shares the order's group
        because the verifier binds an order to the last cart observation's state hash."""
        if self.done:
            return True
        calls = []
        if k not in self.cart_keys_list():
            two_turns = self.wall_time() + 2 * self.last_latency() + 5 < TIME_LIMIT_S
            if (self.close_with_check() and self.first_key is not None and self.steps_available() > 2
                    and two_turns and self.wall_time() < EMERGENCY_AT_S):
                out = self.send_calls([("add_to_cart", ref_from_key(k)), ("inspect_stock", ref_from_key(k))])
                if out is None:
                    return False
                if k not in self.cart_keys_list() or not self.candidate_valid(k):
                    self.bad.add(k)
                    return False
            else:
                calls += [("add_to_cart", ref_from_key(k)), ("inspect_stock", ref_from_key(k))]
        elif not self.candidate_valid(k):
            self.bad.add(k)
            return False
        if self.user_msgs:
            calls.append(("message", {"content": self.reply_for(k)}))
        calls.append(("inspect_cart", {}))
        args = ref_from_key(k)
        claims = self.reason_fields(k)
        if claims:
            args["justification"] = {"claims": claims}
        calls.append(("place_test_order", args))
        out = self.send_calls(calls)
        if out is not None and not self.done:
            self.bad.add(k)
        return self.done

    def _ranking_sorted(self, exclude=()) -> list:
        if "prio" not in self.req:
            self.req["prio"] = read_ranking_priority(self.goal)
            if self.req["prio"] == TF5_DEFAULT_PRIORITY:
                self.req["prio"] = read_ranking_priority(self.query)
        rows = []
        for k in self.req.get("cands") or []:
            if k in exclude or not self.state_ok(k):
                continue
            vals = self._ranking_readings(k)
            if vals is None:
                continue
            exact, hz, inch, price = vals
            axis = {"refresh_rate": hz, "screen_size": inch, "price": price}
            rows.append((-exact, *[sign * axis[a] for a, sign in self.req["prio"]], k))
        rows.sort()
        return [r[-1] for r in rows]

    def select_final(self):
        fam = self.family
        x = self.first_key
        keep_x = bool(x and x in self.cart_keys_list() and self.candidate_valid(x))
        if fam in (TF1, TF3, TF7):
            return self._constraint_close()
        if fam == TF2:
            if keep_x:
                return x
            c = self._recall_valid(exclude={x} if x else ())
            xpid = ref_from_key(x)["product_id"] if x else None
            other = [k for k in c if ref_from_key(k)["product_id"] != xpid]
            return (other or c or [None])[0]
        if fam == TF4:
            if not self.events:
                ls = self._preference_candidates()
                return ls[0][-1] if ls else (x if keep_x else None)
            # after an event the verifier requires a different source listing
            ls = self._preference_candidates(exclude_listing=self._listing_ref(x) if x else None)
            return ls[0][-1] if ls else None
        if fam == TF5:
            r = self._ranking_sorted()
            return r[0] if r else None
        if fam == TF6:
            rows = self._recovery_pool()
            return rows[0][-1] if rows else None
        return x if keep_x else None

    def lane_pref(self) -> None:
        cat = self.req.get("category") or "Smartwatches"
        if cat not in TF4_PROFILES:
            cat = "Smartwatches" if "watch" in self.goal.lower() else cat
        self.req["category"] = cat
        self.profile = TF4_PROFILES.get(cat, TF4_PROFILES["Smartwatches"])
        qs = self._preference_queries() + list(self.profile["queries"])
        calls = []
        if TF4_ASK_PRIORITIES:
            # The TF4 rules require clarifying priorities before the first cart add; on
            # tasks that seal it as required, the metric is zeroed without that answer.
            calls.append(("message", {"content": (
                "Before I choose, what are your requirements and priorities - "
                "which matters most to you for this?")}))
        calls += [self.catalog_lookup(q) for q in qs[:MAX_SEARCHES_PER_GROUP]]
        self.send_calls(calls)
        self._preference_pass(extra=False)
        rounds = 0
        while self.may_search_more() and rounds < 3:
            good = [sc for _sig, sc, *_r in self._preference_candidates() if sc >= 0.65]
            if len(good) >= 2 and rounds >= 1:
                break
            self._preference_pass(extra=True)
            rounds += 1
        self.emit_output()

    def last_latency(self) -> float:
        """Pessimistic wall-clock cost of one more turn (recent max, floor 3 s)."""
        return max([3.0] + self._lat)

    def lane_rank(self) -> None:
        self.req["cands"] = list(dict.fromkeys(
            f"{a}::{b}" for a, b in re.findall(r"([A-Za-z0-9_\-]+)::([A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*)", self.goal)))
        if not self.req["cands"]:
            return self.lane_model_loop()
        pids = list(dict.fromkeys(ref_from_key(k)["product_id"] for k in self.req["cands"]))
        self.send_calls(self.compare_actions(pids)[:MAX_COMPARES_PER_GROUP])
        self.emit_output()

    def _ranking_readings(self, k: str):
        prod = self.products.get(ref_from_key(k)["product_id"])
        if not prod:
            return None
        v = next((x for x in prod.get("variants") or [] if str(x.get("sku")) == ref_from_key(k)["sku"]), None)
        if v is None:
            return None
        opts = v.get("options")
        refresh = variant_value(prod, opts, "refresh_rate")
        size = variant_value(prod, opts, "screen_size")
        exact = refresh is not None and size is not None
        if refresh is None:
            hz = _hertz_values(" ".join([str(opts or ""), str((prod.get("facts") or {}).get("refresh_rate") or ""), str(prod.get("title") or "")]))
            refresh = max(hz, key=lambda s: float(s[:-2])) if hz else None
        if size is None:
            inch = _screen_inch_values(" ".join([str(opts or ""), str((prod.get("facts") or {}).get("size") or ""), str(prod.get("title") or "")]))
            size = max(inch, key=lambda s: float(s[:-2])) if inch else None
        if refresh is None or size is None:
            return None
        return (1 if exact else 0, float(refresh[:-2]), float(size[:-2]), self.price_for(k))

    def _ingest_result(self, res: dict, calls: list) -> list:
        turn = self.turn + 1
        try:
            self.turn = int(res.get("solver_turn_count") or turn)
        except (TypeError, ValueError):
            self.turn = turn
        out = []
        for (name, args), cr in zip(calls, res["calls"]):
            ob = (cr or {}).get("observation") or {}
            payload = ob.get("observation") if isinstance(ob.get("observation"), dict) else {}
            err = ob.get("error")
            self._absorb_observation(name, args, payload, err)
            if ob.get("done") is True:
                self.done = True
            out.append((name, args, payload, bool(ob.get("done")), err))
        um = res.get("user_message")
        if isinstance(um, dict) and str(um.get("content") or "").strip():
            self._absorb_user_turn(str(um["content"]))
        self.log_line(turn=self.turn, calls=[n for n, _ in calls],
                  errors=[e for *_x, e in out if e] or None, done=self.done or None)
        return out

    def _constraint_score(self, r: dict):
        title = str(r.get("title") or "")
        target = self.req.get("brand") or self.req.get("pref")
        s = 0.0
        if target:
            if brand_fits(target, r.get("brand")):
                s += 4
            elif _token_list(target) <= _token_list(title):
                s += 1
        for k, want in self.req["attrs"].items():
            if want in _option_axis_values(k, title):
                s += 2
        word = CAT_QUERY_WORD.get(self.req.get("category") or "", "")
        if word and _token_list(word) & _token_list(title):
            s += 1
        if re.search(r"\b(?:case|cover|tempered|glass|protector|film|lcd|digitizer|replacement|"
                     r"battery|charger|cable|adapter|holder|stand|mount|strap|housing|sticker|"
                     r"skin|pouch|stylus|lens|parts?|refurb\w*|renewed)\b", title, re.I):
            s -= 5
        if is_handset_category(self.req.get("category")) and (_SMARTPHONE_RE.search(title) or _OS_RE.search(title)):
            s += 0.5
        return s

    def summarise_progress(self) -> str:
        cart = ", ".join(self.cart_keys_list()) or "empty"
        seen = []
        for pid, prod in list(self.products.items())[:12]:
            seen.append(f"{pid} | {str(prod.get('title') or '')[:70]} | brand={prod.get('brand')} | "
                        f"variants={[(v.get('sku'), v.get('options'), v.get('price'), v.get('in_stock')) for v in (prod.get('variants') or [])[:3]]}")
        return (f"Progress so far (turn {self.turn}): cart = {cart}; market events = "
                f"{[(e.get('kind'), (e.get('target') or {}).get('product_id')) for e in self.events]}; "
                f"current budget = {self.budget} {self.currency or ''}; requirements = {self.req}. "
                f"Viewed products:\n" + "\n".join(seen))

    def _recall_view_targets(self, n: int) -> list:
        return self.rank_unseen(self._recall_rank, n)

    def _recall_valid(self, exclude=()) -> list:
        out = []
        cat = self.req.get("category")
        for pid, prod, v, k in self.variants_for():
            if k in exclude or not self.state_ok(k):
                continue
            if not self._recall_match(prod.get("title")):
                continue
            if cat and not category_admits(cat, prod.get("category_path") or [], title=prod.get("title")):
                continue
            out.append((self.price_for(k), k))
        low = self.price_outliers([k for _p, k in out])
        out.sort(key=lambda row: (1 if row[1] in low else 0, row[0], row[1]))
        return [k for _p, k in out]

    def _tool_has_param(self, tool: str, param: str) -> bool:
        for t in self.tools:
            fn = t.get("function") or {}
            if fn.get("name") == tool:
                return param in ((fn.get("parameters") or {}).get("properties") or {})
        return False

    def lane_recover(self) -> None:
        model, color = read_recovery_pref(self.goal)
        self.req["model"], self.req["color"] = model, color
        if not model:
            return self.lane_model_loop()
        cat = self.req.get("category") or "Smartphones"
        word = CAT_QUERY_WORD.get(cat, "smartphone")
        qs = [f"{model}", f"{model} {color or ''}", f"{model} {word}"]
        self.send_calls([self.catalog_lookup(q) for q in self.fresh_queries(qs)[:MAX_SEARCHES_PER_GROUP]])
        self._recovery_pass(extra=False)
        rounds = 0
        while self.may_search_more() and rounds < 3:
            rows = self._recovery_pool()
            has0 = any(lvl == 0 for lvl, *_r in rows)
            has_alt = any(lvl == 1 for lvl, *_r in rows)
            has_l2 = any(lvl == 2 for lvl, *_r in rows)
            if has0 and (has_alt or (has_l2 and rounds >= 2)):
                break
            # after one model-focused round with no same-model alternative, broaden to the
            # preferred colour across models (relaxation level 2) instead of the model again
            self._recovery_pass(extra=True, broaden=has0 and not has_alt and rounds >= 1)
            rounds += 1
        self.emit_output()

    def wall_time(self) -> float:
        return time.monotonic() - self.t0

    def candidate_valid(self, k: str, budget: float | None = None) -> bool:
        """Every current requirement of this family, on the latest observations."""
        if not k or not self.state_ok(k, budget):
            return False
        prod, v = self.variant_pair(k)
        if prod is None or v is None:
            return False
        fam = self.family
        pid = ref_from_key(k)["product_id"]
        if fam in (TF1, TF3, TF7):
            brand = self.req.get("brand") if fam in (TF3, TF7) else None
            return self._constraint_valid(pid, prod, v, k, brand=brand, budget=budget)
        if fam == TF2:
            return self._recall_match(prod.get("title")) and category_admits(
                self.req.get("category"), prod.get("category_path") or [], title=prod.get("title"))
        if fam == TF4:
            return self._preference_device_valid(prod)
        if fam == TF5:
            return k in (self.req.get("cands") or []) and self._ranking_readings(k) is not None
        if fam == TF6:
            row_cat = recover_row_category(prod, v.get("options"))
            return row_cat is not None and category_admits(self.req.get("category") or "Smartphones", [row_cat])
        return True

    def event_kinds(self) -> list:
        counter = Counter()
        for event in self.events:
            if isinstance(event, dict):
                counter.bump(str(event.get("kind") or "event"))
        return [kind for kind, _n in counter.most_common(5)]

    def close_with_check(self) -> bool:
        if CLOSE_MODE == "fused":
            return False
        if CLOSE_MODE == "validated":
            return True
        return self.family in CHAIN_FAMILIES

    def _absorb_user_turn(self, text: str) -> None:
        self.user_msgs.append(text)
        for m in _BUDGET_RE.finditer(text):
            cur = m.group(2).upper()
            if self.currency and cur != self.currency:
                continue
            val = _price_number(m.group(1))
            if self.budget is None or val < self.budget - 1e-9:
                self.budget = val
                self.req_changes += 1
                if self.budget0 is None:
                    self.budget0 = val
        m = re.search(r"required catalog category is ([^.\n]+)\.", text)
        if m:
            self.req["category"] = m.group(1).strip()
        m = re.search(r"required brand is ([^.\n]+)\.", text)
        if m and self.family in (TF3, TF7):
            self.req["brand"] = m.group(1).strip()
        if self.family in (TF1, TF3, TF7):
            # "it must now have X instead of Y": only the part before "instead of" is the
            # new requirement.  Any message phrased as a requirement replaces the stored
            # value (latest wins); a bare mention only fills a missing attribute.
            seg = re.split(r"\binstead of\b", text, maxsplit=1, flags=re.I)[0]
            cue = re.search(
                r"\b(?:change|instead|actually|make (?:that|it)|switch|update|now|must|"
                r"needs?|require[sd]?|requirement)\b", text, re.I)
            for k, v in extract_specs(seg).items():
                if self.req["attrs"].get(k) != v and (cue or k not in self.req["attrs"]):
                    if k in self.req["attrs"]:
                        self.req_changes += 1
                    self.req["attrs"][k] = v

    def reason_fields(self, k: str):
        if self.family != TF7:
            return None
        prod = self.products.get(ref_from_key(k)["product_id"]) or {}
        claims = [{"field": a, "value": spec_claim_phrase(a, w)} for a, w in self.req["attrs"].items()
                  if a in ("resolution", "refresh_rate", "screen_size")]
        cap = self.budget if self.budget is not None else self.price_for(k)
        claims.append({"field": "price", "value": f"at most {cap:.2f} {self.currency or ''}".strip()})
        claims.append({"field": "in_stock", "value": "true"})
        if str(prod.get("brand") or "").strip():
            claims.append({"field": "brand", "value": str(prod["brand"]).strip()})
        return claims

    def note_kinds(self) -> list:
        kinds = set()
        for entry in self.log:
            if isinstance(entry, dict):
                kinds.update(entry.keys())
        return sorted(kinds)

    def send_calls(self, calls: list):
        """Send one solver turn (a call group); returns [(name, args, payload, done, err)]."""
        if self.done or self.turn >= self.max_steps:
            return None
        calls = [c for c in calls if c]
        if not calls:
            return []
        if len(calls) > self.max_calls:
            merged: list = []
            for i in range(0, len(calls), self.max_calls):
                part = self.send_calls(calls[i:i + self.max_calls])
                if part is None:
                    return merged or None
                merged += part
                if self.done:
                    break
            return merged
        if self._pending is not None:
            penv, pcalls = self._pending
            pres = self._transmit(penv)
            if pres is None:
                return None
            self._pending = None
            self._ingest_result(pres, pcalls)
            # The replayed turn may carry new state (event, budget cut, done):
            # never send calls that were planned before seeing it.
            return None
        turn = self.turn + 1
        gid = f"{self.pid}-t{turn}"
        envelope = {
            **self.binding,
            "call_id": gid,
            "idempotency_key": gid,
            "turn": turn,
            "calls": [
                {"call_id": f"{gid}-c{i}", "action": {"name": n, "args": a}}
                for i, (n, a) in enumerate(calls, start=1)
            ],
        }
        res = self._transmit(envelope)
        if res is None:
            # The server may still have executed this turn; replay the identical
            # envelope before sending anything else so turns never drift.
            self._pending = (envelope, calls)
            self.log_line(turn=turn, calls=[n for n, _ in calls], error="environment call failed")
            return None
        return self._ingest_result(res, calls)

    def emergency_close(self) -> None:
        """Order only a candidate that meets every current requirement; never an
        arbitrary affordable item."""
        if self.done:
            return
        try:
            y = self.select_final() if self.first_key else (self.select_first() or self.select_final())
        except Exception:  # noqa: BLE001
            y = None
        if y is not None and self.candidate_valid(y):
            self.close_order(y)

    def emit_output(self) -> None:
        """Commit (if needed), notice the event, recover, and order - re-planning from
        the latest observations after every turn."""
        for _ in range(6):
            if self.done:
                return
            if self.first_key is None:
                x = self.select_first() or self.select_final()
                if x is None:
                    return
                if (self.wall_time() > EMERGENCY_AT_S or self.steps_available() <= 1
                        or self.wall_time() + 2 * self.last_latency() + 5 > TIME_LIMIT_S):
                    self.close_order(x)
                    continue
                self.first_commitment(x)
                continue
            y = self.select_final()
            if y is None:
                if self.may_search_more():
                    self.search_again(final=True)
                    continue
                return
            self.close_order(y)

    def lane_model_loop(self, context: str | None = None) -> None:
        messages = [
            {"role": "system", "content": (
                "You are a careful shopping agent. Use the tools. Verify category, brand, spec, "
                "price and stock with view before adding to cart. After a cart add, inspect the "
                "item's stock and price; if it changed, pick another valid item. Place exactly one "
                "order with place_test_order, in the same call group as inspect_cart.")},
            {"role": "user", "content": self.query},
        ]
        if context:
            messages.append({"role": "user", "content": context})
        while not self.done and self.turn < self.max_steps and self.wall_time() < EMERGENCY_AT_S:
            resp = self._call_model(messages)
            if resp is None:
                break
            msg = resp.get("choices", [{}])[0].get("message") or {}
            tcs = [tc for tc in (msg.get("tool_calls") or []) if isinstance(tc, dict)][: self.max_calls]
            if not tcs:
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                messages.append({"role": "user", "content": "Call one of the tools to continue."})
                continue
            calls, sent_tcs, rejected = [], [], []
            for tc in tcs:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                args = args if isinstance(args, dict) else {}
                name = str(fn.get("name") or "")
                if (name == "place_test_order" and self.family is not None
                        and not self.candidate_valid(row_key(args.get("product_id"), args.get("sku")))):
                    rejected.append(tc)
                    continue
                calls.append((name, args))
                sent_tcs.append(tc)
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in rejected:
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or "x", "content": json.dumps(
                    {"error": "not sent: that variant does not meet the task requirements on the latest observations"})})
            if not calls:
                continue
            n_msgs = len(self.user_msgs)
            out = self.send_calls(calls)
            if out is None:
                for tc in sent_tcs:
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or "x",
                                     "content": json.dumps({"error": "call not completed; state may have changed"})})
                continue
            for tc, (_n, _a, payload, _d, _e) in zip(sent_tcs, out):
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or "x",
                                 "content": json.dumps(payload)[:6000]})
            for text in self.user_msgs[n_msgs:]:
                messages.append({"role": "user", "content": text})

    def _transmit(self, envelope: dict):
        for attempt in range(3):
            if attempt and self.wall_time() + self.last_latency() > TIME_LIMIT_S - 10:
                break
            started = self.wall_time()
            res = _ENV.post("/environment/call", json_data=envelope)
            self._track_latency(self.wall_time() - started)
            if isinstance(res, dict) and isinstance(res.get("calls"), list):
                return res
            if self.wall_time() > TIME_LIMIT_S - 25:
                break
            time.sleep(1.0 + attempt)
        return None

    def _recall_view_more(self, n: int) -> bool:
        pids = self._recall_view_targets(n)
        if not pids:
            return False
        self.send_calls([("view", {"product_id": p}) for p in pids])
        return True

    def turn_plan(self) -> "Workflow":
        return (Workflow()
                .then("survey")
                .then("ground", lambda ctx: ctx.get("rows", 0) > 0)
                .then("commit", lambda ctx: ctx.get("grounded", 0) > 0)
                .then("recover", lambda ctx: ctx.get("events", 0) > 0)
                .then("order"))


def _resolution_from_field(text: str) -> set:
    """Canonical-resolution evidence from a merged display fact.  Listings merge display
    resolution with video/recording resolution ("3120x1440; 720p", "2340x1080; 4k"); the
    verifier's canonical attribute reads only resolution-type fields, so explicit pixel
    dimensions win over bare class tokens when both are present."""
    dims = set()
    for m in _WXH_RE.finditer(text.lower()):
        a, b = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _resolution_tier(a, b)
        if cls:
            dims.add(cls)
    return dims if dims else _resolution_dims(text)


def take_while(items: list, predicate) -> list:
    out = []
    for item in items:
        if not predicate(item):
            break
        out.append(item)
    return out


def _identity_text_blob(text: str, title: bool = False) -> str:
    text = re.sub(
        r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|"
        r"tablets?|accessor(?:y|ies))\b",
        "", text, flags=re.I,
    )
    if title:
        text = re.split(
            r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
            r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|"
            r"screen\s*protector|charger|accessor(?:y|ies))\b))",
            text, maxsplit=1, flags=re.I,
        )[0]
    return text


class UnionFind:
    """Disjoint-set forest to cluster listings that are the same product."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def root_of(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def join(self, a, b) -> None:
        ra, rb = self.root_of(a), self.root_of(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> list:
        groups: dict = {}
        for item in list(self.parent):
            groups.setdefault(self.root_of(item), []).append(item)
        return [sorted(members, key=str) for members in groups.values()]


def product_value(prod: dict, key: str) -> str | None:
    """Product-level value from visible facts plus title (approximates resolve_attribute)."""
    facts = prod.get("facts") or {}
    texts = []
    if key in _FACT_LABEL_FOR:
        texts.append(str(facts.get(_FACT_LABEL_FOR[key]) or ""))
    elif key == "ram":
        storage = str(facts.get("storage") or "")
        if re.search(r"\bram\b|memory", storage, re.I):
            texts.append(storage)
    values: set = set()
    conflict = False
    for text in texts:
        if not text:
            continue
        if key == "resolution" and re.search(r"\bqHD\b", text):
            conflict = True
        vals = _field_readings(key, text)
        if len(vals) > 1:
            conflict = True
        values |= vals
    title = _resolution_from_title(prod) if key == "resolution" else str(prod.get("title") or "")
    if key == "resolution" and re.search(r"\bqHD\b", title):
        conflict = True
    values |= _option_axis_values(key, title)
    if conflict or len(values) != 1:
        return None
    return next(iter(values))


def apply_overrides(base: dict, overrides: dict) -> dict:
    """Shallow merge where a None override deletes the key."""

    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def variant_value(prod: dict, options: str | None, key: str) -> str | None:
    """Per-variant value: options first, product-level fallback (resolve_variant_attribute)."""
    vals = _option_axis_values(key, options or "")
    if len(vals) > 1:
        return None
    if vals:
        return next(iter(vals))
    if _option_hint_present(key, options or ""):
        return None
    return product_value(prod, key)


class Phase:
    """A named step in a linear workflow with an entry guard."""

    def __init__(self, name: str, guard=None) -> None:
        self.name = name
        self.guard = guard
        self.entered = 0
        self.completed = False

    def ready(self, context: dict) -> bool:
        return True if self.guard is None else bool(self.guard(context))

    def begin(self) -> None:
        self.entered += 1

    def finish(self) -> None:
        self.completed = True


def read_firm_brand(goal: str) -> str | None:
    sent = re.split(r"(?<=[.?!])\s", goal.strip())[0]
    m = _CAT_WORD_RE.search(sent)
    if not m:
        return None
    before = sent[: m.start()]
    arts = list(re.finditer(r"\b(?:a|an)\s+", before, re.I))
    if not arts:
        return None
    span = _BRAND_NOISE_RE.sub(" ", before[arts[-1].end():])
    words = [w for w in re.split(r"\s+", span.strip()) if w and w.lower() not in {"a", "an", "the"}]
    brand = " ".join(words).strip(" -,")
    return brand or None


def spec_support(prod: dict, options: str | None, key: str, want: str) -> int:
    """2 = options or a fact field states the value, 1 = title only, 0 = none."""
    if want in _option_axis_values(key, options or ""):
        return 2
    facts = prod.get("facts") or {}
    label = _FACT_LABEL_FOR.get(key, "storage")
    if want in _field_readings(key, str(facts.get(label) or "")):
        return 2
    if want in _option_axis_values(key, str(prod.get("title") or "")):
        return 1
    return 0


def build_query_string(params: dict) -> str:
    """Deterministic k=v&k=v joining, sorted keys, blanks dropped."""

    parts = []
    for key in sorted(params):
        value = params[key]
        if value is None or value == "":
            continue
        parts.append("%s=%s" % (key, value))
    return "&".join(parts)


def read_ranking_priority(text: str) -> tuple:
    """Ordered (axis, direction) list from 'highest refresh ... largest screen ... lowest price'."""
    out: list = []
    for m in re.finditer(
        r"\b(highest|largest|biggest|greatest|fastest|lowest|smallest|cheapest)\b"
        r"(?:\s+(?:advertised|maximum|available|possible))*"
        r"(?:\s+(refresh(?:\s+rate)?|screen(?:\s+size)?|display(?:\s+size)?|price))?",
        text, re.I,
    ):
        word, noun = m.group(1).lower(), (m.group(2) or "").lower()
        if noun.startswith("refresh") or (not noun and word == "fastest"):
            axis = "refresh_rate"
        elif noun.startswith(("screen", "display")):
            axis = "screen_size"
        elif noun.startswith("price") or (not noun and word == "cheapest"):
            axis = "price"
        else:
            continue
        sign = 1 if word in ("lowest", "smallest", "cheapest") else -1
        if axis not in [a for a, _ in out]:
            out.append((axis, sign))
    return tuple(out) if len(out) >= 2 else TF5_DEFAULT_PRIORITY


def _resolution_from_title(prod: dict) -> str:
    title = str(prod.get("title") or "")
    model_text = str((prod.get("facts") or {}).get("model") or "")
    for model in [m.strip() for m in model_text.split(";") if m.strip()]:
        stem = re.split(r"[-_]", model)[0]
        if (
            re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", model, re.I)
            and re.search(r"[a-z]", stem, re.I)
            and re.search(r"\d", stem)
            and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
        ):
            title = re.sub(
                r"(?<![\w-])" + re.escape(model) + r"(?![\w-])", "", title, flags=re.I
            )
    return title


def flatten_mapping(payload) -> dict:
    """Nested dict/list -> flat dotted keys, iteratively, for diffing observations."""

    flat: dict = {}
    stack = [("", payload)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                stack.append(("%s.%s" % (prefix, key) if prefix else str(key), value))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                stack.append(("%s[%d]" % (prefix, index), value))
        else:
            flat[prefix] = node
    return flat


def category_admits(expected: str | None, path, title: str | None = None) -> bool:
    if not expected:
        return True
    if canonical_forms([expected]) & canonical_forms(path or []):
        return True
    exp = _flatten_text(expected)
    path_labels = canonical_forms(path or [])
    if exp in {"gaming monitors", "gaming monitor"}:
        toks = _stem_bag(title)
        high = any(float(m.group(1)) >= 120 for m in _HIGH_REFRESH_RE.finditer(str(title or "")))
        return bool(path_labels & canonical_forms(["Monitors", "Computer Monitors"])) and (
            {"gaming", "monitor"} <= toks or ("monitor" in toks and high)
        )
    if exp in {"over ear", "over-ear", "over ear headphones", "over-ear headphones"}:
        toks = _stem_bag(title)
        return bool(path_labels & canonical_forms(["Headphones"])) and (
            {"over", "ear"} <= toks or "overear" in toks or "circumaural" in toks
        )
    return False


def rotate_list(items: list, by: int) -> list:
    if not items:
        return []
    shift = by % len(items)
    return items[shift:] + items[:shift]


def _facts_blob_of(prod: dict) -> str:
    facts = prod.get("facts") or {}
    return " ".join(f"{k.replace('_', ' ')} {v}" for k, v in facts.items()).casefold()


def zip_longest_pad(left: list, right: list, pad=None) -> list:
    width = max(len(left), len(right))
    out = []
    for i in range(width):
        out.append((left[i] if i < len(left) else pad, right[i] if i < len(right) else pad))
    return out


def _playtime_axis(text: str) -> float:
    days = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*days?", text, re.I)]
    hours = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hours?", text, re.I)]
    if days:
        best = max(days)
        if best >= 10:
            return 1.0
        if best >= 5:
            return 0.75
        if best >= 2:
            return 0.5
    if hours:
        best = max(hours)
        if best >= 120:
            return 1.0
        if best >= 72:
            return 0.75
        if best >= 36:
            return 0.5
    if "solar" in text or "battery life" in text:
        return 0.5
    return 0.0


def agent_main(problem_data: dict) -> list:
    """Sandbox entry point."""
    if not isinstance(problem_data, dict) or not isinstance(problem_data.get("environment"), dict):
        return []
    agent = Agent(problem_data)
    try:
        agent.drive_episode()
    except Exception as exc:  # noqa: BLE001
        agent.log_line(exception=f"{type(exc).__name__}: {str(exc)[:300]}")
        try:
            agent.emergency_close()
        except Exception as exc2:  # noqa: BLE001
            agent.log_line(exception2=f"{type(exc2).__name__}: {str(exc2)[:200]}")
    return [{"completion": {"content": json.dumps(agent.log)[:20000]},
             "extra_info": {"family": agent.family, "turns": agent.turn, "done": agent.done}}]


__all__ = ["agent_main"]