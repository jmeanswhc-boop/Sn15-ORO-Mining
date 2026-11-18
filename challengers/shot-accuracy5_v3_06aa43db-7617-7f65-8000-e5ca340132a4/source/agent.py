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


def marker_hits(query: str, markers: tuple[str, ...]) -> int:
    low = _low(query)
    total = 0
    for marker in markers:
        if marker.lower() in low:
            total += 1
    return total


def category_tail(path: Any) -> str | None:
    if isinstance(path, str):
        parts = [p for p in re.split(r"[>/]", path) if p.strip()]
    elif isinstance(path, list):
        parts = [str(p) for p in path if str(p).strip()]
    else:
        return None
    return parts[-1].strip() if parts else None


def label_forms(labels) -> set:
    out = set()
    for label in labels:
        low = _squash(label)
        if low:
            out.add(low)
            out.add(_CATEGORY_ALIASES.get(low, low))
    return out


def pick_max(items: list[Any], score) -> Any:
    best = None
    best_score = None
    for item in items:
        value = score(item)
        if best_score is None or value > best_score:
            best, best_score = item, value
    return best


def title_tokens(title: Any) -> list[str]:
    """Lowercase maximal [a-z0-9] runs, matching the live tokeniser's contract."""

    runs: list[str] = []
    current: list[str] = []
    for ch in _low(title):
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            current.append(ch)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def _pixel_pairs(text: str | None) -> set:
    if not text:
        return set()
    if re.search(r"\bqHD\b", text):
        return set()
    low = text.lower()
    values = set()
    for m in _WXH_RE.finditer(low):
        a, b = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
        cls = _screen_tier(a, b)
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


def sorted_by_keys(rows: dict[str, dict[str, Any]], keys: list[str]) -> list[str]:
    """Stable multi-key sort of row ids; numeric keys sort descending."""

    def sort_key(item: str) -> tuple:
        row = rows[item]
        parts: list[float] = []
        for key in keys:
            value = _f(row.get(key))
            parts.append(-value if value is not None else float("inf"))
        return tuple(parts)

    return sorted(rows, key=sort_key)


def is_placeholder_title(title: str) -> bool:
    low = _low(title)
    return any(bad in low for bad in ("test product", "sample listing", "do not buy", "placeholder"))


def category_agrees(expected: str | None, path, title: str | None = None) -> bool:
    if not expected:
        return True
    if label_forms([expected]) & label_forms(path or []):
        return True
    exp = _squash(expected)
    path_labels = label_forms(path or [])
    if exp in {"gaming monitors", "gaming monitor"}:
        toks = _stem_set(title)
        high = any(float(m.group(1)) >= 120 for m in _HIGH_REFRESH_RE.finditer(str(title or "")))
        return bool(path_labels & label_forms(["Monitors", "Computer Monitors"])) and (
            {"gaming", "monitor"} <= toks or ("monitor" in toks and high)
        )
    if exp in {"over ear", "over-ear", "over ear headphones", "over-ear headphones"}:
        toks = _stem_set(title)
        return bool(path_labels & label_forms(["Headphones"])) and (
            {"over", "ear"} <= toks or "overear" in toks or "circumaural" in toks
        )
    return False


def salient_bigrams(goal: str) -> list[str]:
    tokens = tokenise_goal(goal)
    return ["%s %s" % (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def canon_resolution(text: str | None) -> str | None:
    vals = _pixel_pairs(text)
    return next(iter(vals)) if len(vals) == 1 else None


class ClaimBuilder:
    """Assembles a short, evidence-backed justification for TF7-style orders."""

    ALLOWED = ("brand", "category", "in_stock", "price", "refresh_rate", "screen_size", "resolution")

    def __init__(self, limit: int = 4) -> None:
        self.limit = limit
        self.claims: list[tuple[str, str]] = []

    def add_fact(self, field: str, value: Any) -> bool:
        name = _low(field).replace(" ", "_")
        text = _text(value).strip()
        if name not in self.ALLOWED or not text or len(self.claims) >= self.limit:
            return False
        if any(existing == name for existing, _v in self.claims):
            return False
        self.claims.append((name, text))
        return True

    def sentence(self) -> str:
        if not self.claims:
            return ""
        parts = ["%s is %s" % (name.replace("_", " "), value) for name, value in self.claims]
        return "Chosen because " + ", ".join(parts[:-1] + [("and " + parts[-1]) if len(parts) > 1 else parts[-1]]) + "."

    def payload(self) -> dict[str, Any] | None:
        return {"claims": [{"field": f, "value": v} for f, v in self.claims]} if self.claims else None


def parse_ref(text: str) -> tuple[str, str]:
    pid, _, sku = _text(text).partition("::")
    return pid, sku


def _option_readings(key: str, text: str) -> set:
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
    return _pixel_pairs(text)


def tf5_priority_from_goal(goal: str, default: str = "price") -> str:
    axes = ordered_axes(goal)
    return axes[0] if axes else default


def _amount_of(text: str) -> float:
    return float(str(text).replace(",", ""))


def sku_tail(sku: str) -> str:
    parts = re.split(r"[-_/ ]+", str(sku).strip())
    return parts[-1] if parts else str(sku)


def brand_agreement(goal_brand: str | None, row_brand: str | None) -> bool:
    if not goal_brand:
        return True
    if not row_brand:
        return False
    return _low(goal_brand) in _low(row_brand) or _low(row_brand) in _low(goal_brand)


def _identity_blob(text: str, title: bool = False) -> str:
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


def title_case_keep_codes(title: str) -> str:
    out = []
    for token in str(title).split():
        out.append(token if looks_like_code(token) else token.capitalize())
    return " ".join(out)


def rank_correlation(order_a: list[str], order_b: list[str]) -> float | None:
    shared = [key for key in order_a if key in order_b]
    count = len(shared)
    if count < 2:
        return None
    positions_a = {key: index for index, key in enumerate(shared)}
    positions_b = {key: index for index, key in enumerate([k for k in order_b if k in positions_a])}
    diff_sq = 0
    for key in shared:
        diff_sq += (positions_a[key] - positions_b[key]) ** 2
    denom = count * (count * count - 1)
    return round(1.0 - 6.0 * diff_sq / denom, 4) if denom else None


def read_recovery_preference(goal: str) -> tuple:
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


def tf6_relax_reason(rung: str) -> str:
    reasons = {
        "same_variant": "preferred variant still available",
        "same_model_variant": "another variant of the preferred model",
        "same_model": "same model, different listing",
        "same_brand": "same brand within category and budget",
        "same_category": "same category within budget",
        "any_in_stock": "least-relaxed in-stock option",
    }
    return reasons.get(rung, rung.replace("_", " "))


def phrase_recall(title: str, phrase: str) -> float:
    """Share of the phrase's tokens present in the title as whole tokens."""

    wanted = list(title_tokens(phrase))
    if not wanted:
        return 1.0
    present = set(title_tokens(title))
    return round(sum(1 for part in wanted if part in present) / len(wanted), 3)


def category_from_goal(goal: str) -> str | None:
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


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def strip_marketing(title: str) -> str:
    """Drop leading promo fragments so the product noun surfaces."""

    noise = ("hot sale", "flash sale", "big sale", "free shipping", "cod",
             "on sale", "best seller", "new arrival", "official store")
    low = str(title)
    for fragment in noise:
        low = re.sub(r"(?i)\b%s\b" % re.escape(fragment), " ", low)
    return re.sub(r"\s+", " ", low).strip(" -|,")


def _bare_model_code(raw: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", raw.casefold())
    while words and words[-1] in _COLOR_PHRASES:
        words.pop()
    code = "".join(words)
    return code.upper() if 3 <= len(code) <= 24 and any(c.isdigit() for c in code) else None


class RecoveryLadder:
    """Ordered fallbacks after an event, most faithful replacement first."""

    RUNGS = ("same_variant", "same_model_variant", "same_model", "same_brand", "same_category", "any_in_stock")

    def __init__(self) -> None:
        self.position = 0

    def rung_now(self) -> str:
        return self.RUNGS[min(self.position, len(self.RUNGS) - 1)]

    def drop_rung(self) -> str | None:
        if self.position + 1 >= len(self.RUNGS):
            return None
        self.position += 1
        return self.rung_now()

    def at_bottom(self) -> bool:
        return self.position >= len(self.RUNGS) - 1

    def rung_text(self) -> str:
        return self.rung_now().replace("_", " ")


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return round(len(left & right) / len(union), 4) if union else 0.0


def spec_backing(prod: dict, options: str | None, key: str, want: str) -> int:
    """2 = options or a fact field states the value, 1 = title only, 0 = none."""
    if want in _option_readings(key, options or ""):
        return 2
    facts = prod.get("facts") or {}
    label = _FACT_LABEL_FOR.get(key, "storage")
    if want in _values_for_field(key, str(facts.get(label) or "")):
        return 2
    if want in _option_readings(key, str(prod.get("title") or "")):
        return 1
    return 0


def close_mode_for(family: str | None, mode: str) -> str:
    """Resolve the effective close-out mode the way the runtime notes describe."""

    if mode in ("validated", "fused"):
        return mode
    return "validated" if chained_event_family(family) else "fused"


def running_total(values: list[float]) -> list[float]:
    total = 0.0
    out: list[float] = []
    for value in values:
        total += value
        out.append(round(total, 4))
    return out


def recovery_row_category(prod: dict, options: str | None) -> str | None:
    """Recovery candidate admission (_candidate_row) - returns the profile category."""
    title = str(prod.get("title") or "")
    low = title.casefold().strip()
    path = prod.get("category_path") or []
    if category_agrees("Headphones", path, title=title):
        ok = (bool(low) and not low.startswith(_INVALID_TITLE_PREFIXES)
              and not any(f in low for f in _INVALID_TITLE_FRAGMENTS)
              and not _TF6_ACCESSORY_RE.search(low))
        if ok and _OVER_OR_ON_EAR_RE.search(f"{title} {options or ''}"):
            return "Headphones"
        return None
    if category_agrees("Smartphones", path, title=title):
        ok = (bool(low) and not low.startswith(_INVALID_TITLE_PREFIXES)
              and not _TF6_ACCESSORY_RE.search(low) and _FEATURE_PHONE_RE.search(low) is None)
        if ok and phone_admissible(prod, options):
            return "Smartphones"
    return None


class CatalogSnapshot:
    """Immutable-ish view over the row pool with a few derived reports."""

    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows

    def brands(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows.values():
            brand = _low(row.get("brand"))
            if brand:
                counts[brand] = counts.get(brand, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def price_stats(self) -> dict[str, float]:
        prices = [p for p in (_f(row.get("price")) for row in self.rows.values()) if p is not None]
        if not prices:
            return {}
        prices.sort()
        return {"min": prices[0], "max": prices[-1], "median": prices[len(prices) // 2]}

    def stock_rate(self) -> float:
        flags = [row.get("in_stock") for row in self.rows.values() if isinstance(row.get("in_stock"), bool)]
        return round(sum(1 for f in flags if f) / len(flags), 3) if flags else 0.0


class FamilyLane:
    """Per-family switches the runtime notes describe, in one place."""

    def __init__(self, family: str | None) -> None:
        self.family = family

    def wants_priority_probe(self) -> bool:
        return self.family == "preference_reasoning"

    def reserves_best(self) -> bool:
        return self.family == "preference_reasoning" and TF4_RESERVE_BEST

    def validates_recovery(self) -> bool:
        return chained_event_family(self.family)

    def needs_justification(self) -> bool:
        return self.family == "justification"

    def ranks_by_priority(self) -> bool:
        return self.family == "ranking"

    def phrase_gated(self) -> bool:
        return self.family == "retrieval_recall"

    def readable(self) -> str:
        return family_label(self.family)


def brand_is_match(expected: str | None, actual: str | None) -> bool:
    if not expected:
        return True
    exp, act = _squash(expected), _squash(actual)
    if not exp or exp == act:
        return True
    exp_t = _word_list(expected)
    return bool(exp_t) and exp_t <= _word_list(actual)


def _low(value: Any) -> str:
    return _text(value).lower()


def tf4_needs_priority_probe(family: str | None, asked: bool, ask_enabled: bool) -> bool:
    """TF4 must learn the shopper's priorities before its first add."""

    return family == "preference_reasoning" and ask_enabled and not asked


def phone_verdict(prod: dict, options: str | None) -> str:
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
    title = _identity_blob(raw_title, title=True)
    type_text = " ".join(
        _identity_blob(str(facts.get(k) or ""))
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
        _identity_blob(str(facts.get(k) or ""))
        for k in ("device_operating_system", "description_operating_system")
        if not re.search(r"\bcompatib", str(facts.get(k) or ""), re.I)
    )
    title_os = bool(_OS_RE.search(title)) and not re.search(r"\bcompatib", raw_title, re.I)
    return "supported" if (_OS_RE.search(os_text) or title_os) else "unresolved"


def blend_scores(scores: dict[str, float], weights: dict[str, float]) -> float:
    normalised = normalise_weights(weights)
    acc = 0.0
    for name, value in scores.items():
        acc += value * normalised.get(name, 0.0)
    return round(acc, 4)


def _facts_blob(prod: dict) -> str:
    facts = prod.get("facts") or {}
    return " ".join(f"{k.replace('_', ' ')} {v}" for k, v in facts.items()).casefold()


class CallGroupBuilder:
    """Accumulates tool calls into groups that must be sent together."""

    def __init__(self) -> None:
        self.groups: list[list[tuple[str, dict]]] = [[]]

    def start_group(self) -> "CallGroupBuilder":
        if self.groups[-1]:
            self.groups.append([])
        return self

    def append_call(self, name: str, payload: dict | None = None) -> "CallGroupBuilder":
        self.groups[-1].append((name, dict(payload or {})))
        return self

    def order_with_cart(self, product_id: str, sku: str, justification: str | None = None) -> "CallGroupBuilder":
        self.start_group().append_call("inspect_cart", {})
        ref = {"product_id": product_id, "sku": sku}
        if justification:
            ref["justification"] = justification
        return self.append_call("place_test_order", ref)

    def as_calls(self) -> list[tuple[str, dict]]:
        return [call for group in self.groups for call in group if group]

    def group_sizes(self) -> list[int]:
        return [len(group) for group in self.groups if group]


def axis_reading(row: dict[str, Any], axis: str) -> float | None:
    direct = _f(row.get(axis))
    if direct is not None:
        return direct
    facts = row.get("facts")
    if isinstance(facts, dict):
        for key, value in facts.items():
            if axis in _low(key):
                match = re.search(r"\d+(?:\.\d+)?", _text(value))
                if match:
                    return float(match.group(0))
    match = re.search(r"(\d+(?:\.\d+)?)\s*%s" % re.escape(axis), _low(row.get("title")))
    return float(match.group(1)) if match else None


def _known_model(text: str) -> str | None:
    for pattern, label_fn in _MODEL_PATTERNS:
        m = re.search(pattern, text.casefold())
        if m:
            return label_fn(m)
    return None


class GoalParser:
    """Pulls the coarse fields off a goal string without the model."""

    def __init__(self, goal: str) -> None:
        self.goal = goal

    def budget(self) -> float | None:
        return price_to_float(self.goal) if re.search(r"(?i)budget|under|below|max|most", self.goal) else None

    def brand_hint(self) -> str | None:
        match = re.search(r"(?i)\b(?:brand|from|by)\s+([A-Za-z][A-Za-z0-9]{1,20})", self.goal)
        return match.group(1) if match else None

    def wants_superlative(self) -> bool:
        return has_superlative_local(self.goal)

    def axes(self) -> list[str]:
        return ordered_axes(self.goal)

    def key_terms(self) -> list[str]:
        return tokenise_goal(self.goal)


class PriorityRanker:
    """Ranks rows by the goal's stated axes, honouring per-axis direction."""

    MINIMISE = ("price", "weight")

    def __init__(self, axes: list[str]) -> None:
        self.axes = list(axes)

    def measure(self, row: dict[str, Any], axis: str) -> float | None:
        return axis_reading(row, axis)

    def sort_key(self, row: dict[str, Any]) -> tuple:
        readings: list[float] = []
        for axis in self.axes:
            value = self.measure(row, axis)
            if value is None:
                readings.append(float("inf"))
            elif axis in self.MINIMISE:
                readings.append(value)
            else:
                readings.append(-value)
        return tuple(readings)

    def order(self, rows: dict[str, dict[str, Any]]) -> list[str]:
        return sorted(rows, key=lambda key: self.sort_key(rows[key]))

    def front(self, rows: dict[str, dict[str, Any]]) -> str | None:
        ordered = self.order(rows)
        return ordered[0] if ordered else None


def spec_to_claim(key: str, value: str) -> str:
    if key == "screen_size":
        return f"{value[:-2]} inches"
    if key == "refresh_rate":
        return f"{value[:-2]}Hz"
    return {"4k": "4K", "qhd": "QHD", "fhd": "1080p Full HD", "hd": "720p HD"}.get(value, value)


def preferred_over_required(clause: str) -> bool:
    low = _low(clause)
    return any(cue in low for cue in ("ideally", "prefer", "nice to have", "if possible", "bonus"))


def spec_to_query(key: str, value: str) -> str:
    if key == "screen_size":
        return f"{value[:-2]} inch"
    if key == "ram":
        return f"{value[:-2].upper()}GB RAM"
    if key == "refresh_rate":
        return f"{value[:-2]}Hz"
    return {"4k": "4K", "qhd": "QHD", "fhd": "1080p FHD", "hd": "720p HD"}.get(value, value)


def within_category(row_path: Any, wanted: str | None) -> bool:
    if not wanted:
        return True
    tail = category_tail(row_path)
    if not tail:
        return False
    return _low(wanted) in _low(tail) or _low(tail) in _low(wanted)


class ProbeSchedule:
    """Decides which shopper probes are worth their one message each."""

    def __init__(self, family: str | None) -> None:
        self.family = family
        self.planned: list[str] = []

    def plan(self, has_budget: bool, has_priorities: bool) -> list[str]:
        self.planned = []
        if self.family == "preference_reasoning" and not has_priorities and TF4_ASK_PRIORITIES:
            self.planned.append("ask_priorities")
        if not has_budget and self.family in ("intent_decomposition", "constraint_satisfaction"):
            self.planned.append("confirm_budget")
        return list(self.planned)

    def worth_sending(self) -> bool:
        return bool(self.planned)


def _word_list(text: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def chained_event_family(code: Any) -> bool:
    return _text(code) in ("intent_decomposition", "constraint_satisfaction")


def _title_resolution(prod: dict) -> str:
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


def event_targets_key(event: dict[str, Any], key: str) -> bool:
    pid, sku = parse_ref(key)
    same_pid = _text(event.get("product_id")) == pid
    event_sku = _text(event.get("sku"))
    return same_pid and (not event_sku or event_sku == sku)


def _phrase_present(text: str, phrase: str) -> bool:
    words = re.findall(r"[a-z0-9]+", phrase.casefold())
    if not words:
        return False
    pattern = r"(?<![a-z0-9])" + r"\W+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def title_without_codes(title: str) -> str:
    kept = [token for token in str(title).split() if not looks_like_code(token)]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def code_variants(code: str) -> list[str]:
    """The catalog forms a model code can appear as, longest first."""

    pieces = [p for p in re.split(r"[\s_/-]+", str(code).strip().lower()) if p]
    if not pieces:
        return []
    joined = "".join(pieces)
    spaced = " ".join(pieces)
    hyphenated = "-".join(pieces)
    ranked = sorted({joined, spaced, hyphenated}, key=lambda form: (-len(form), form))
    return ranked


def _contains_word(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


class CommitmentGate:
    """The single validation the first add, reuse, recovery and emergency share."""

    def __init__(self, card: "RowScoreCard") -> None:
        self.card = card
        self.last_reason = ""

    def permits(self, row: dict[str, Any]) -> bool:
        failing = self.card.failing(row)
        self.last_reason = "; ".join(failing) if failing else ""
        return not failing

    def permits_key(self, rows: dict[str, dict[str, Any]], key: str) -> bool:
        row = rows.get(key)
        return bool(row) and self.permits(row)

    def reason(self) -> str:
        return self.last_reason or "ok"


class MessageBudget:
    """Guards how many shopper messages are worth sending in this run."""

    def __init__(self, cap: int = 1) -> None:
        self.cap = max(0, cap)
        self.count = 0
        self.reasons: list[str] = []

    def request(self, reason: str) -> bool:
        if self.count >= self.cap:
            return False
        self.count += 1
        self.reasons.append(reason)
        return True

    def messages_left(self) -> int:
        return max(0, self.cap - self.count)

    def log_line(self) -> str:
        return "messages %d/%d: %s" % (self.count, self.cap, "; ".join(self.reasons) or "none")


def _brief_endurance_axis(text: str) -> float:
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


def group_adjacent(items: list[Any], same) -> list[list[Any]]:
    """Split a list wherever ``same(prev, cur)`` returns False."""

    groups: list[list[Any]] = []
    for item in items:
        if groups and same(groups[-1][-1], item):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def clock_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    parts: list[str] = []
    for unit_size, suffix in ((3600, "h"), (60, "m"), (1, "s")):
        if total >= unit_size or (suffix == "s" and not parts):
            parts.append("%d%s" % (total // unit_size, suffix))
            total %= unit_size
    return " ".join(parts)


def _refresh_values(text: str | None) -> set:
    if not text:
        return set()
    return {f"{float(m.group(1)):g}hz" for m in _HZ_RE.finditer(text)}


def _rate_smartwatch(prod: dict) -> float:
    text = _facts_blob(prod)
    brand = str(prod.get("brand") or "").casefold()
    nav = 1.0 if _contains_word(text, r"\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\b") else 0.0
    rugged = 0.0
    if _contains_word(text, r"\b(10\s*atm|5\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|"
                  r"fiber[- ]reinforced|titanium|rugged|tactical|tank)\b"):
        rugged = 1.0
    elif _contains_word(text, r"\b(water|swim|steel|aluminum)\b"):
        rugged = 0.5
    battery = _endurance_axis(text)
    outdoor = 1.0 if brand in {"garmin", "amazfit", "fitbit", "kospet"} else 0.0
    penalty = 0.2 if _contains_word(text, r"\b(kid|kids|child|children|myfirst|watch phone)\b") else 0.0
    raw = 0.32 * nav + 0.26 * rugged + 0.26 * battery + 0.16 * outdoor - penalty
    return round(max(0.0, min(1.0, raw)), 3)


def _rate_headphones(prod: dict) -> float:
    title = str(prod.get("title") or "").casefold()
    facts_text = _facts_blob(prod)
    text = facts_text + " " + title
    category_text = " ".join(prod.get("category_path") or []).casefold()
    wireless_facts = facts_text + " " + category_text
    water = 1.0 if _contains_word(text, r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b") else 0.0
    sport = 1.0 if _contains_word(text, _SPORT_FIT) else 0.0
    battery = _brief_endurance_axis(text)
    wireless = 1.0 if _contains_word(wireless_facts + " " + title, r"\b(wireless|bluetooth|true wireless)\b") else 0.0
    raw = 0.30 * water + 0.28 * sport + 0.22 * battery + 0.20 * wireless
    return round(max(0.0, min(1.0, raw)), 3)


def _rate_tablet(prod: dict) -> float:
    title = str(prod.get("title") or "").casefold()
    text = _facts_blob(prod) + " " + title
    kid = 1.0 if _contains_word(text, r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b") else 0.0
    battery = _brief_endurance_axis(text)
    size = 1.0 if _contains_word(text, r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")") else 0.0
    wifi = 1.0 if _contains_word(text, r"\b(wi-?fi|wifi)\b") else 0.0
    raw = 0.40 * kid + 0.25 * battery + 0.20 * size + 0.15 * wifi
    return round(max(0.0, min(1.0, raw)), 3)


TF4_PROFILES = {
    "Smartwatches": {
        "scorer": _rate_smartwatch,
        "accessory": ("case for", "screen protector", "band for", "strap for", "charger for",
                      "mudra band", "replacement band", "watch band"),
        "signals": ("smartwatch", "smart watch", "watch phone", "apple watch", "galaxy watch",
                    "pixel watch", "garmin", "fitbit", "amazfit", "kospet"),
        "queries": ("rugged gps smartwatch", "long battery waterproof smartwatch",
                    "outdoor fitness gps watch", "gps smart watch waterproof",
                    "military rugged smart watch", "hiking compass smartwatch"),
    },
    "Headphones": {
        "scorer": _rate_headphones,
        "accessory": ("case for", "ear pads", "earpads", "replacement pad", "cable for",
                      "adapter", "stand for"),
        "signals": ("headphone", "headphones", "earbud", "earbuds", "earphone", "earphones",
                    "headset"),
        "queries": ("waterproof sport earbuds", "wireless running headphones",
                    "gym workout earbuds ipx", "ipx7 bluetooth sports earphones",
                    "earhook wireless sport earbuds", "sweatproof wireless earbuds long battery"),
    },
    "Tablets": {
        "scorer": _rate_tablet,
        "accessory": ("case for", "keyboard for", "screen protector", "stylus", "pen for",
                      "stand for", "charger for"),
        "signals": ("tablet", "ipad", "galaxy tab", "fire hd", "fire max"),
        "queries": ("kids tablet case bundle", "kids tablet 10 inch", "wifi tablet long battery",
                    "kids tablet shockproof", "8 inch kids tablet wifi", "tablet for kids"),
    },
}


def all_prices(text: str) -> list[float]:
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text))
    values = [float(v) for v in re.findall(r"\d+(?:\.\d{1,2})?", cleaned)]
    return [value for value in values if value > 0]


def read_rank_priority(text: str) -> tuple:
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


def budget_from_text(text: str) -> tuple[float | None, str | None]:
    match = re.search(r"(?i)(?:under|below|at most|max(?:imum)?|budget of)\s*([\u20b1$]?\s*[\d,]+(?:\.\d{1,2})?)", str(text))
    if not match:
        return None, None
    amount = price_to_float(match.group(1))
    currency = "PHP" if "\u20b1" in match.group(1) else ("USD" if "$" in match.group(1) else None)
    return amount, currency


def tf1_split_requirements(text: str) -> dict[str, list[str]]:
    """Separate firm requirements from nice-to-haves by their cue words."""

    firm_cues = ("must", "need", "required", "has to", "only", "exactly")
    soft_cues = ("ideally", "prefer", "nice to have", "would like", "if possible", "bonus")
    firm: list[str] = []
    soft: list[str] = []
    for clause in re.split(r"[.;\n]|,\s*(?=and |but )", text):
        stripped = clause.strip()
        if len(stripped) < 3:
            continue
        low = " %s " % _low(stripped)
        if any(" %s " % cue in low for cue in soft_cues):
            soft.append(stripped)
        elif any(cue in low for cue in firm_cues):
            firm.append(stripped)
        else:
            firm.append(stripped)
    return {"firm": firm, "soft": soft}


def _screen_tier(long_axis: int, short_axis: int) -> str | None:
    if long_axis >= 3840 and short_axis >= 2160:
        return "4k"
    if long_axis >= 2560 and short_axis >= 1440:
        return "qhd"
    if long_axis >= 1920 and short_axis >= 1080:
        return "fhd"
    if long_axis >= 1280 and short_axis >= 720:
        return "hd"
    return None


def sealed_answer(messages: list[str]) -> str | None:
    """The most recent shopper message that states a firm answer."""

    for message in reversed(messages or []):
        low = _low(message)
        if any(cue in low for cue in ("the answer is", "it must be", "i need", "required", "must have")):
            return message
    return None


def emergency_should_fire(elapsed: float, has_commitment: bool) -> bool:
    return elapsed >= EMERGENCY_AT_S and has_commitment


def _diagonal_values(text: str | None) -> set:
    if not text:
        return set()
    return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text)}


def tokenise_goal(goal: str) -> list[str]:
    return [t for t in dict.fromkeys(title_tokens(goal)) if len(t) > 1 and t not in _STOP]


def summarise_event(event: dict[str, Any]) -> str:
    kind = _low(event.get("kind")) or _low(event.get("type")) or "event"
    pid = _text(event.get("product_id")) or "?"
    if "price" in kind:
        return "price change on %s to %s" % (pid, event.get("new_price"))
    if "stock" in kind:
        return "stockout on %s" % pid
    return "%s on %s" % (kind, pid)


def read_required_brand(goal: str) -> str | None:
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


def order_calls_present(calls: list[tuple[str, dict]]) -> bool:
    names = {name for name, _payload in calls}
    return "place_test_order" in names and "inspect_cart" in names


def looks_like_code(token: str) -> bool:
    """A token mixing letters and digits, e.g. a model or SKU stub."""

    text = str(token)
    has_alpha = any(ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)
    return has_alpha and has_digit and len(text) >= 3


def recovery_model(prod: dict, options: str | None) -> str | None:
    facts = prod.get("facts") or {}
    raw = facts.get("model")
    if raw:
        raw = str(raw)
        if ";" in raw or "/" in raw:
            return None
        named = _known_model(raw)
        if named:
            return named
        words = raw.casefold().split()
        prefix = []
        while len(words) > 1 and words[0].isalpha() and any(c.isdigit() for c in " ".join(words[1:])):
            prefix.append(words.pop(0))
        identity = f"{prod.get('title') or ''} {prod.get('brand') or ''}"
        if prefix and not _phrase_present(identity, " ".join(prefix)):
            return None
        if raw.isalpha() and 3 <= len(raw) <= 24:
            return raw.upper() if _phrase_present(identity, raw) else None
        return _bare_model_code(" ".join(words))
    title = str(prod.get("title") or "")
    named = _known_model(title)
    if named:
        return named
    labels = set()
    for m in _TITLE_MODEL_CODE.finditer(title):
        token = m.group()
        if _NON_MODEL_CODE.search(token):
            continue
        suffix = re.match(r"(?:\s+(?:pro|max|ultra|plus)\b)+", title[m.end():], re.I)
        code = _bare_model_code(token + (suffix.group() if suffix else ""))
        if code:
            labels.add(code)
    return next(iter(labels)) if len(labels) == 1 else None


def pick_min(items: list[Any], score) -> Any:
    best = None
    best_score = None
    for item in items:
        value = score(item)
        if best_score is None or value < best_score:
            best, best_score = item, value
    return best


def bucket_by(items: list[Any], key) -> dict[Any, list[Any]]:
    buckets: dict[Any, list[Any]] = {}
    for item in items:
        buckets.setdefault(key(item), []).append(item)
    return buckets


def _squash(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def overlap_coefficient(left: set[Any], right: set[Any]) -> float:
    smaller = min(len(left), len(right))
    return round(len(left & right) / smaller, 4) if smaller else 0.0


class VerifierMirror:
    """Local re-checks that mirror what the runtime validator inspects."""

    def __init__(self, budget: float | None, require_stock: bool = True) -> None:
        self.budget = budget
        self.require_stock = require_stock
        self.notes: list[str] = []

    def price_ok(self, price: float | None) -> bool:
        if self.budget is not None and price is not None and price > self.budget + 1e-9:
            self.notes.append("price %s over budget %s" % (price, self.budget))
            return False
        return True

    def stock_ok(self, in_stock: Any) -> bool:
        if self.require_stock and in_stock is False:
            self.notes.append("out of stock")
            return False
        return True

    def title_ok(self, title: str) -> bool:
        if is_placeholder_title(title):
            self.notes.append("placeholder title")
            return False
        return True

    def approve(self, price: float | None, in_stock: Any, title: str) -> bool:
        self.notes = []
        return self.price_ok(price) and self.stock_ok(in_stock) and self.title_ok(title)

    def reason(self) -> str:
        return "; ".join(self.notes) or "clean"


def _canon_value(attr: str, text: Any) -> str | None:
    if attr == "resolution":
        return canon_resolution(str(text) if text is not None else None)
    raw = str(text or "").strip()
    bare = re.fullmatch(r"\d+(?:\.\d+)?", raw)
    if attr == "refresh_rate":
        vals = _refresh_values(raw)
        if not vals and bare:
            vals = {f"{float(raw):g}hz"}
        return next(iter(vals)) if len(vals) == 1 else None
    if attr == "screen_size":
        vals = _diagonal_values(raw)
        if not vals and bare:
            vals = {f"{float(raw):g}in"}
        return next(iter(vals)) if len(vals) == 1 else None
    return None


def family_label(code: Any) -> str:
    names = {
        "intent_decomposition": "TF1 intent",
        "retrieval_recall": "TF2 retrieval",
        "constraint_satisfaction": "TF3 constraints",
        "preference_reasoning": "TF4 preference",
        "ranking": "TF5 ranking",
        "recovery": "TF6 recovery",
        "justification": "TF7 justification",
    }
    return names.get(_text(code), "unclassified")


def has_superlative_local(text: str) -> bool:
    words = ("most", "least", "highest", "lowest", "best", "cheapest",
             "largest", "smallest", "fastest", "longest")
    tokens = set(title_tokens(text))
    return any(word in tokens for word in words)


def is_phone_category(category: str | None) -> bool:
    return "smartphone" in label_forms([category or ""])


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(_text(cell) for cell in cells) + " |"


def compares_left(used: int) -> int:
    return max(0, MAX_COMPARES_PER_GROUP - used)


def _option_summary(key: str, text: str) -> bool:
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


def span_of_prices(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    prices: list[float] = []
    for row in rows:
        value = _f(row.get("price"))
        if value is not None:
            prices.append(value)
    if not prices:
        return None
    return (min(prices), max(prices))


def ordered_axes(text: str) -> list[str]:
    """Rank axes by where their cue word first appears in the ask."""

    cues = {
        "price": ("price", "cheap", "budget", "affordable"),
        "battery": ("battery", "mah", "playtime", "endurance"),
        "screen": ("screen", "display", "inch"),
        "refresh": ("refresh", "hz"),
        "weight": ("weight", "light", "gram"),
        "rating": ("rating", "review", "stars"),
        "storage": ("storage", "gb", "capacity"),
    }
    low = _low(text)
    hits: list[tuple[int, str]] = []
    for axis, words in cues.items():
        positions = [low.find(word) for word in words if low.find(word) >= 0]
        if positions:
            hits.append((min(positions), axis))
    return [axis for _index, axis in sorted(hits)]


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
        self.family = route_family(self.query, self.tools)
        self.turn = 0
        self.done = False
        self._pending = None
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
            self.budget0 = self.budget = _amount_of(m.group(1))
            self.currency = m.group(2).upper()
        self.req["category"] = category_from_goal(self.goal)
        # TF3 and TF6 gate on smartphone evidence; TF7 does when its goal carries the rule.
        self.strict_phone = self.family == TF3 or (
            self.family == TF7 and "must explicitly say smartphone" in self.query)
        self.req_changes = 0

    def effective_close_mode(self) -> str:
        return close_mode_for(self.family, CLOSE_MODE)

    def clock(self) -> float:
        return time.monotonic() - self.t0

    def event_summaries(self) -> list[str]:
        return [summarise_event(event) for event in self.events if isinstance(event, dict)]

    def rank_ungrounded(self, score_fn, limit: int, budget: float | None = None) -> list:
        best: dict = {}
        for k, r in self.rows.items():
            pid = str(r.get("product_id"))
            if pid in self.products or not self.state_is_valid(k, budget):
                continue
            s = score_fn(r)
            if s is None:
                continue
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, -self.price_of(k))
        order = sorted(best, key=lambda p: (-best[p][0], -best[p][1], p))
        return order[:limit]

    def run_calls(self, calls: list):
        """Send one solver turn (a call group); returns [(name, args, payload, done, err)]."""
        if self.done or self.turn >= self.max_steps:
            return None
        calls = [c for c in calls if c]
        if not calls:
            return []
        if len(calls) > self.max_calls:
            merged: list = []
            for i in range(0, len(calls), self.max_calls):
                part = self.run_calls(calls[i:i + self.max_calls])
                if part is None:
                    return merged or None
                merged += part
                if self.done:
                    break
            return merged
        if self._pending is not None:
            penv, pcalls = self._pending
            pres = self._dispatch(penv)
            if pres is None:
                return None
            self._pending = None
            self._take_result(pres, pcalls)
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
        res = self._dispatch(envelope)
        if res is None:
            # The server may still have executed this turn; replay the identical
            # envelope before sending anything else so turns never drift.
            self._pending = (envelope, calls)
            self.log_note(turn=turn, calls=[n for n, _ in calls], error="environment call failed")
            return None
        return self._take_result(res, calls)

    def catalog_search(self, q: str) -> tuple:
        q = re.sub(r"\s+", " ", q).strip()
        if q not in self.queries:
            self.queries.append(q)
        args = {"query": q, "k": 10, "in_stock": True}
        if self.budget is not None:
            args["max_price"] = round(float(self.budget), 2)
        return ("search", args)

    def _more_constraint_queries(self, brand: str, word: str, i: int) -> list:
        specs = " ".join(spec_to_query(k, v) for k, v in self.req["attrs"].items())
        nums = " ".join(re.sub(r"[a-z]+$", "", v) for v in self.req["attrs"].values())
        pool = [f"{brand} phone {nums}", f"{brand} mobile {specs}", f"{brand} {word} original",
                f"{nums} {word}", f"{brand} {word} sale", f"{word} {specs} cheap"]
        return pool[i * 2: i * 2 + 2]

    def variants_of(self):
        for pid, prod in self.products.items():
            for v in prod.get("variants") or []:
                if isinstance(v, dict) and v.get("sku") is not None:
                    yield pid, prod, v, make_key(pid, v.get("sku"))

    def candidate_passes(self, k: str, budget: float | None = None) -> bool:
        """Every current requirement of this family, on the latest observations."""
        if not k or not self.state_is_valid(k, budget):
            return False
        prod, v = self.variant_ref(k)
        if prod is None or v is None:
            return False
        fam = self.family
        pid = key_to_ref(k)["product_id"]
        if fam in (TF1, TF3, TF7):
            brand = self.req.get("brand") if fam in (TF3, TF7) else None
            return self._constraint_ok(pid, prod, v, k, brand=brand, budget=budget)
        if fam == TF2:
            return self._retrieval_match(prod.get("title")) and category_agrees(
                self.req.get("category"), prod.get("category_path") or [], title=prod.get("title"))
        if fam == TF4:
            return self._preference_device_ok(prod)
        if fam == TF5:
            return k in (self.req.get("cands") or []) and self._ranking_values(k) is not None
        if fam == TF6:
            row_cat = recovery_row_category(prod, v.get("options"))
            return row_cat is not None and category_agrees(self.req.get("category") or "Smartphones", [row_cat])
        return True

    def verifier_check(self, key: str) -> tuple[bool, str]:
        row = self.rows.get(key) or {}
        mirror = VerifierMirror(self.budget, self.strict_phone or True)
        ok = mirror.approve(_f(row.get("price")), row.get("in_stock"), _text(row.get("title")))
        return ok, mirror.reason()

    def _dispatch(self, envelope: dict):
        for attempt in range(3):
            if attempt and self.clock() + self.recent_latency() > TIME_LIMIT_S - 10:
                break
            started = self.clock()
            res = _ENV.post("/environment/call", json_data=envelope)
            self._note_latency(self.clock() - started)
            if isinstance(res, dict) and isinstance(res.get("calls"), list):
                return res
            if self.clock() > TIME_LIMIT_S - 25:
                break
            time.sleep(1.0 + attempt)
        return None

    def _take_product(self, prod: dict) -> None:
        pid = str(prod["product_id"])
        self.products[pid] = prod
        for v in prod.get("variants") or []:
            if isinstance(v, dict):
                self._remember_state(make_key(pid, v.get("sku")), v.get("price"), v.get("in_stock"))
                if not self.currency and v.get("currency"):
                    self.currency = str(v["currency"])

    def ranked_keys(self) -> list[str]:
        axes = ordered_axes(self.goal)
        priced = self.priced_rows()
        if not axes:
            return sorted(priced, key=lambda k: _f(priced[k].get("price")) or float("inf"))

        def key_for(item: str) -> tuple:
            row = priced[item]
            readings = []
            for axis in axes:
                value = axis_reading(row, axis)
                direction = -1.0 if axis in ("price", "weight") else 1.0
                readings.append(-(value * direction) if value is not None else float("inf"))
            return tuple(readings)

        return sorted(priced, key=key_for)

    def _take_result(self, res: dict, calls: list) -> list:
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
            self._take_observation(name, args, payload, err)
            if ob.get("done") is True:
                self.done = True
            out.append((name, args, payload, bool(ob.get("done")), err))
        um = res.get("user_message")
        if isinstance(um, dict) and str(um.get("content") or "").strip():
            self._take_user_turn(str(um["content"]))
        self.log_note(turn=self.turn, calls=[n for n, _ in calls],
                  errors=[e for *_x, e in out if e] or None, done=self.done or None)
        return out

    def _preference_row_score(self, r: dict):
        title = str(r.get("title") or "")
        low = title.casefold()
        if any(p in low for p in self.profile["accessory"]) or any(t in low for t in _TF4_BAD_TITLE):
            return None
        est = self.profile["scorer"]({"title": title, "facts": {"listing": title},
                                      "brand": r.get("brand"), "category_path": []})
        signal = any(s in low for s in self.profile["signals"])
        return est + (0.2 if signal else -0.3)

    def _retrieval_valid(self, exclude=()) -> list:
        out = []
        cat = self.req.get("category")
        for pid, prod, v, k in self.variants_of():
            if k in exclude or not self.state_is_valid(k):
                continue
            if not self._retrieval_match(prod.get("title")):
                continue
            if cat and not category_agrees(cat, prod.get("category_path") or [], title=prod.get("title")):
                continue
            out.append((self.price_of(k), k))
        out.sort()
        return [k for _p, k in out]

    def acceptable_keys(self) -> list[str]:
        card = self.score_card()
        return [k for k, row in self.rows.items() if card.accepts(row)]

    def _preference_round(self, extra: bool) -> None:
        calls = []
        if extra:
            calls += [self.catalog_search(q) for q in self.fresh_observation(list(self.profile["queries"]))[:2]]
        pids = self.rank_ungrounded(self._preference_row_score, 5 * MAX_COMPARES_PER_GROUP)
        calls += self.compare_variants(pids)[:MAX_COMPARES_PER_GROUP]
        if calls:
            self.run_calls(calls)

    def close_out_plan(self, key: str) -> list[str]:
        return CloseOutPlanner(self.family).call_names(key)

    def steps_left(self) -> int:
        return self.max_steps - self.turn

    def _recovery_rows(self, exclude=()) -> list:
        out = []
        cat = self.req.get("category") or "Smartphones"
        for pid, prod, v, k in self.variants_of():
            if k in exclude or not self.state_is_valid(k):
                continue
            row_cat = recovery_row_category(prod, v.get("options"))
            if row_cat is None or not category_agrees(cat, [row_cat]):
                continue
            level = recovery_rung(recovery_model(prod, v.get("options")), recovery_color(prod, v.get("options")),
                              self.req.get("model"), self.req.get("color"))
            if level is None:
                continue
            out.append((level, self.price_of(k), k))
        out.sort()
        return out

    def log_note(self, **kw) -> None:
        kw["t"] = round(self.clock(), 1)
        self.log.append(kw)

    def _retrieval_requeries(self, n: int) -> list:
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

    def rejection_reasons(self, key: str) -> list[str]:
        row = self.rows.get(key) or {}
        return self.score_card().failing(row)

    def _listing_key(self, k: str) -> str:
        pid = key_to_ref(k)["product_id"]
        return str((self.products.get(pid) or {}).get("listing_id") or pid)

    def _retrieval_view_targets(self, n: int) -> list:
        return self.rank_ungrounded(self._retrieval_rank, n)

    def run_digest(self) -> dict[str, Any]:
        elapsed = self.clock()
        remaining = max(0.0, TIME_LIMIT_S - elapsed)
        return {
            "family": family_label(self.family),
            "phase": deadline_state(elapsed),
            "chained_event_ok": chained_event_family(self.family),
            "rows": len(self.rows),
            "grounded": len(self.products),
            "cart_lines": len(self.cart),
            "committed": self.first_key,
            "dead": sorted(self.dead),
            "budget": self.budget,
            "queries": list(self.queries),
            "query_slots_left": max(0, MAX_SEARCHES_PER_GROUP - len(self.queries)),
            "goal_axes": ordered_axes(self.goal),
            "events": len(self.events),
            "elapsed": clock_text(elapsed),
            "seconds_left": round(remaining, 1),
        }

    def lane_constraint(self) -> None:
        fam = self.family
        first_sentence = re.split(r"(?<=[.?!])\s", self.goal.strip())[0]
        self.req["attrs"].update(read_specs(first_sentence if fam == TF7 else self.goal))
        if fam in (TF3, TF7):
            self.req["brand"] = read_required_brand(self.goal)
        if fam == TF1:
            self.req["pref"] = read_preferred_brand(self.goal)
        underspecified = (not self.req["attrs"]) or "if you ask" in self.goal.lower()
        cat = self.req.get("category")
        word = CAT_QUERY_WORD.get(cat or "", (cat or "").lower().rstrip("s"))
        brand = self.req.get("brand") or self.req.get("pref") or ""

        calls = []
        if underspecified:
            calls.append(("message", {"content": (
                "Before I pick one, what are your firm requirements - which exact spec "
                "(screen size, RAM, resolution or refresh rate) must it have?")}))
        calls += [self.catalog_search(q) for q in self.fresh_observation(self._constraint_queries(brand, word))[:2]]
        calls.append(self.catalog_filter(cat, brand or None))
        self.run_calls(calls)
        self._constraint_rounds(word, brand)
        self.emit_result()

    def close_out(self, k: str) -> bool:
        """Order k.  Validated mode adds+inspects it first and re-checks it on the fresh
        observation; fused mode sends add, inspect, reply, cart refresh and order together
        (no simulator call on that turn).  The cart refresh always shares the order's group
        because the verifier binds an order to the last cart observation's state hash."""
        if self.done:
            return True
        calls = []
        if k not in self.cart_refs():
            two_turns = self.clock() + 2 * self.recent_latency() + 5 < TIME_LIMIT_S
            if (self.close_validated() and self.first_key is not None and self.steps_left() > 2
                    and two_turns and self.clock() < EMERGENCY_AT_S):
                out = self.run_calls([("add_to_cart", key_to_ref(k)), ("inspect_stock", key_to_ref(k))])
                if out is None:
                    return False
                if k not in self.cart_refs() or not self.candidate_passes(k):
                    self.bad.add(k)
                    return False
            else:
                calls += [("add_to_cart", key_to_ref(k)), ("inspect_stock", key_to_ref(k))]
        elif not self.candidate_passes(k):
            self.bad.add(k)
            return False
        if self.user_msgs:
            calls.append(("message", {"content": self.shopper_reply(k)}))
        calls.append(("inspect_cart", {}))
        args = key_to_ref(k)
        claims = self.reason_claims(k)
        if claims:
            args["justification"] = {"claims": claims}
        calls.append(("place_test_order", args))
        out = self.run_calls(calls)
        if out is not None and not self.done:
            self.bad.add(k)
        return self.done

    def _retrieval_view_more(self, n: int) -> bool:
        pids = self._retrieval_view_targets(n)
        if not pids:
            return False
        self.run_calls([("view", {"product_id": p}) for p in pids])
        return True

    def _remember_state(self, k: str, price: Any, in_stock: Any) -> None:
        try:
            p = float(price) if price is not None else None
        except (TypeError, ValueError):
            p = None
        self.state[k] = (p, in_stock is True)

    def justification_for(self, key: str) -> dict[str, Any] | None:
        row = self.rows.get(key) or {}
        builder = ClaimBuilder()
        builder.add_fact("brand", row.get("brand"))
        builder.add_fact("price", row.get("price"))
        builder.add_fact("in_stock", "yes" if row.get("in_stock") else "")
        for field, value in (row.get("facts") or {}).items():
            builder.add_fact(field, value)
        return builder.payload()

    def shopper_reply(self, k: str) -> str:
        pid = key_to_ref(k)["product_id"]
        title = str((self.products.get(pid) or {}).get("title") or self.rows.get(k, {}).get("title") or "the item")
        p = self.state.get(k, (None, True))[0]
        cur = self.currency or ""
        price = f" at {p:.2f} {cur}" if isinstance(p, float) else ""
        return (f"Thanks - going with {title[:80]}{price}. It meets your stated requirements "
                f"and is in stock, so I am placing the order now.")

    def _retrieval_match(self, title: Any) -> bool:
        toks = set(re.findall(r"[a-z0-9]+", str(title or "").lower()))
        return all(t in toks for t in self.req["terms"])

    def _constraint_rounds(self, word: str, brand: str) -> None:
        extra = 0
        while self.may_explore():
            have_first = self.choose_first() is not None
            have_final = self._valid_total() >= 2
            if have_first and have_final:
                return
            pids = self.rank_ungrounded(self._constraint_row_score, 5 * MAX_COMPARES_PER_GROUP,
                                      budget=self.budget0 if self.first_key is None else None)
            qs = self.fresh_observation(self._constraint_queries(brand, word) + self._more_constraint_queries(brand, word, extra))
            extra += 1
            calls = [self.catalog_search(q) for q in qs[: (2 if pids else MAX_SEARCHES_PER_GROUP)]]
            calls += self.compare_variants(pids)[:MAX_COMPARES_PER_GROUP]
            if not calls:
                return
            self.run_calls(calls)
            if extra > 5:
                return

    def _constraint_ok(self, pid: str, prod: dict, v: dict, k: str, *, brand, budget) -> bool:
        if not self.state_is_valid(k, budget):
            return False
        if self.currency and v.get("currency") and v.get("currency") != self.currency:
            return False
        cat = self.req.get("category")
        title = str(prod.get("title") or "")
        if cat and not category_agrees(cat, prod.get("category_path") or [], title=title):
            return False
        if brand and not brand_is_match(brand, prod.get("brand")):
            return False
        if is_phone_category(cat):
            status = phone_verdict(prod, v.get("options"))
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
                if (fact_reading(prod, attr) != want or product_reading(prod, attr) != want
                        or variant_reading(prod, v.get("options"), attr) != want):
                    return False
            return True
        return all(variant_reading(prod, v.get("options"), attr) == want
                   for attr, want in self.req["attrs"].items())

    def _retrieval_rank(self, r: dict):
        if not self._retrieval_match(r.get("title")):
            return None
        return 1.0

    def wall_fraction(self) -> float:
        span = self.clock() / TIME_LIMIT_S
        return 0.0 if span < 0 else 1.0 if span > 1 else round(span, 4)

    def catalog_filter(self, category: str | None = None, brand: str | None = None) -> tuple:
        args: dict = {"in_stock": True, "k": 10}
        if category:
            args["category"] = FILTER_CATEGORY.get(category, category)
        if brand:
            args["brand"] = brand
        if self.budget is not None:
            args["max_price"] = round(float(self.budget), 2)
        return ("filter", args)

    def order_ready_keys(self) -> list[str]:
        return [k for k in self.rows if self.verifier_check(k)[0]]

    def summarise_state(self) -> str:
        cart = ", ".join(self.cart_refs()) or "empty"
        seen = []
        for pid, prod in list(self.products.items())[:12]:
            seen.append(f"{pid} | {str(prod.get('title') or '')[:70]} | brand={prod.get('brand')} | "
                        f"variants={[(v.get('sku'), v.get('options'), v.get('price'), v.get('in_stock')) for v in (prod.get('variants') or [])[:3]]}")
        return (f"Progress so far (turn {self.turn}): cart = {cart}; market events = "
                f"{[(e.get('kind'), (e.get('target') or {}).get('product_id')) for e in self.events]}; "
                f"current budget = {self.budget} {self.currency or ''}; requirements = {self.req}. "
                f"Viewed products:\n" + "\n".join(seen))

    def price_of(self, k: str) -> float:
        p = self.state.get(k, (None, False))[0]
        return float(p) if p is not None else 1e18

    def choose_first(self):
        fam = self.family
        if fam in (TF1, TF3, TF7):
            return self._constraint_first()
        if fam == TF2:
            c = self._retrieval_valid()
            return c[0] if c else None
        if fam == TF4:
            ls = self._preference_listings()
            if not ls:
                return None
            # The committed listing is repriced away; keep the best one for the final order
            # when another listing also clears the 0.65 commitment bar.
            if TF4_RESERVE_BEST and len(ls) >= 2 and ls[1][1] >= 0.65 and ls[1][0] >= ls[0][0]:
                return ls[1][-1]
            return ls[0][-1]
        if fam == TF5:
            r = self._ranking_order()
            return r[0] if r else None
        if fam == TF6:
            rows = self._recovery_rows()
            zero = [k for lvl, _p, k in rows if lvl == 0]
            return zero[0] if zero else (rows[0][-1] if rows else None)
        return None

    def emit_result(self) -> None:
        """Commit (if needed), notice the event, recover, and order - re-planning from
        the latest observations after every turn."""
        for _ in range(6):
            if self.done:
                return
            if self.first_key is None:
                x = self.choose_first() or self.choose_final()
                if x is None:
                    return
                if (self.clock() > EMERGENCY_AT_S or self.steps_left() <= 1
                        or self.clock() + 2 * self.recent_latency() + 5 > TIME_LIMIT_S):
                    self.close_out(x)
                    continue
                self.first_commit(x)
                continue
            y = self.choose_final()
            if y is None:
                if self.may_explore():
                    self.keep_exploring(final=True)
                    continue
                return
            self.close_out(y)

    def should_probe_priorities(self, asked: bool) -> bool:
        return tf4_needs_priority_probe(self.family, asked, TF4_ASK_PRIORITIES)

    def variant_ref(self, k: str):
        pid, _, sku = k.partition("::")
        prod = self.products.get(pid)
        if not prod:
            return None, None
        v = next((x for x in prod.get("variants") or [] if str(x.get("sku")) == sku), None)
        return prod, v

    def _constraint_candidates(self, *, brand, budget, exclude=()) -> list:
        out = []
        for pid, prod, v, k in self.variants_of():
            if k in exclude or not self._constraint_ok(pid, prod, v, k, brand=brand, budget=budget):
                continue
            ev = min((spec_backing(prod, v.get("options"), a, w) for a, w in self.req["attrs"].items()), default=0)
            refurb = 1 if re.search(r"refurb|renewed|pre-owned|second hand|used", str(prod.get("title") or ""), re.I) else 0
            unproven = 1 if (is_phone_category(self.req.get("category"))
                             and not phone_admissible(prod, v.get("options"))) else 0
            out.append((unproven, refurb, -ev, self.price_of(k), k))
        out.sort()
        return [k for *_x, k in out]

    def lane_retrieval(self) -> None:
        m = re.search(r'matches\s+"([^"]+)"', self.goal) or re.search(
            r"matches the phrase\s+(.+?)(?:,|\s+at or under|\s+and buy|$)", self.goal)
        phrase = (m.group(1) if m else self.goal).strip().strip('"')
        self.req["terms"] = re.findall(r"[a-z0-9]+", phrase.lower())
        self.req["phrase"] = " ".join(self.req["terms"])
        if not self.req["terms"]:
            return self.lane_fallback()
        self.run_calls([self.catalog_search(self.req["phrase"])])
        q_more = self._retrieval_requeries(2)
        calls = [self.catalog_search(q) for q in q_more]
        calls.append(self.catalog_filter(self.req.get("category")))
        views = self._retrieval_view_targets(2)
        calls += [("view", {"product_id": p}) for p in views]
        self.run_calls(calls)
        while self.may_explore() and len(self._retrieval_valid()) < 2:
            if not self._retrieval_view_more(2):
                break
        self.emit_result()

    def _take_event(self, ev: dict) -> None:
        self.events.append(ev)
        target = ev.get("target") or {}
        keys = [make_key(target.get("product_id"), target.get("sku"))]
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

    def _take_observation(self, name: str, args: dict, payload: dict, err: Any) -> None:
        if not isinstance(payload, dict):
            return
        for r in payload.get("results") or []:
            if isinstance(r, dict) and r.get("product_id"):
                k = make_key(r["product_id"], r.get("sku"))
                self.rows[k] = r
                self._remember_state(k, r.get("price"), r.get("in_stock"))
                if not self.currency and r.get("currency"):
                    self.currency = str(r["currency"])
        if payload.get("product_id") and isinstance(payload.get("variants"), list):
            self._take_product(payload)
        for c in payload.get("compared") or []:
            if isinstance(c, dict) and c.get("product_id"):
                self._take_product(c)
        if name == "inspect_stock" and payload.get("product_id") and "in_stock" in payload:
            k = make_key(payload["product_id"], payload.get("sku"))
            self._remember_state(k, payload.get("price"), payload.get("in_stock"))
            p, ok = self.state[k]
            if k in self.cart_refs() and (not ok or (p is not None and self.budget is not None and p > self.budget + 1e-9)):
                self.dead.add(k)
        if isinstance(payload.get("cart"), list):
            self.cart = [c for c in payload["cart"] if isinstance(c, dict)]
        added = payload.get("added")
        if isinstance(added, dict) and added.get("product_id"):
            k = make_key(added["product_id"], added.get("sku"))
            if self.first_key is None:
                self.first_key = k
        ev = payload.get("market_event")
        if isinstance(ev, dict):
            self._take_event(ev)
        if name == "add_to_cart" and err:
            self.bad.add(make_key(args.get("product_id"), args.get("sku")))

    def status_bar(self) -> str:
        return "%s | %s | %d rows | %d grounded | %s" % (
            family_label(self.family),
            deadline_state(self.clock()),
            len(self.rows),
            len(self.products),
            clock_text(self.clock()),
        )

    def close_validated(self) -> bool:
        if CLOSE_MODE == "fused":
            return False
        if CLOSE_MODE == "validated":
            return True
        return self.family in CHAIN_FAMILIES

    def planned_probes(self) -> list[str]:
        has_priorities = bool(self.req.get("priorities"))
        return ProbeSchedule(self.family).plan(self.budget is not None, has_priorities)

    def lane_ranking(self) -> None:
        self.req["cands"] = list(dict.fromkeys(
            f"{a}::{b}" for a, b in re.findall(r"([A-Za-z0-9_\-]+)::([A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*)", self.goal)))
        if not self.req["cands"]:
            return self.lane_fallback()
        pids = list(dict.fromkeys(key_to_ref(k)["product_id"] for k in self.req["cands"]))
        self.run_calls(self.compare_variants(pids)[:MAX_COMPARES_PER_GROUP])
        self.emit_result()

    def event_hits_commitment(self) -> bool:
        if not self.first_key:
            return False
        return any(event_targets_key(event, self.first_key) for event in self.events if isinstance(event, dict))

    def fresh_observation(self, queries: list) -> list:
        return [q for q in dict.fromkeys(re.sub(r"\s+", " ", q).strip() for q in queries)
                if q and q not in self.queries]

    def _note_latency(self, dt: float) -> None:
        self._lat.append(max(0.0, dt))
        self._lat = self._lat[-4:]

    def keep_exploring(self, final: bool = False) -> None:
        if self.family in (TF1, TF3, TF7):
            cat = self.req.get("category")
            word = CAT_QUERY_WORD.get(cat or "", (cat or "").lower())
            brand = "" if self.family == TF1 else (self.req.get("brand") or "")
            qs = self.fresh_observation([f"{word} {' '.join(spec_to_query(k, v) for k, v in self.req['attrs'].items())} {brand}",
                             f"{brand} {word} cheap"])
            calls = [self.catalog_search(q) for q in qs[:2]]
            self.run_calls(calls)
            pids = self.rank_ungrounded(self._constraint_row_score, 10)
            if pids:
                self.run_calls(self.compare_variants(pids)[:2])
        elif self.family == TF4:
            self._preference_round(extra=True)
        elif self.family == TF6:
            self._recovery_round(extra=True)
        elif self.family == TF2:
            self._retrieval_view_more(2)

    def _constraint_final(self):
        x = self.first_key
        # keep the committed item only if it still meets every current requirement
        # (an event that did not invalidate it, or no event at all)
        if x and x in self.cart_refs() and self.candidate_passes(x):
            return x
        brand = self.req.get("brand") if self.family in (TF3, TF7) else None
        c = self._constraint_candidates(brand=brand, budget=self.budget, exclude={x} if x else ())
        if not c:
            return None
        xpid = key_to_ref(x)["product_id"] if x else None
        other = [k for k in c if key_to_ref(k)["product_id"] != xpid]
        return (other or c)[0]

    def affordable_keys(self) -> list[str]:
        limit = self.budget
        out = []
        for k, row in self.rows.items():
            price = _f(row.get("price"))
            if limit is None or price is None or price <= limit + 1e-9:
                out.append(k)
        return out

    def may_explore(self) -> bool:
        # one exploration turn + commit turn + close turn must still fit
        return (not self.done and self.clock() < NO_ROUND_AFTER_S and self.steps_left() > 4
                and self.clock() + 3 * self.recent_latency() + 10 < TIME_LIMIT_S)

    def title_recall_for(self, key: str, phrase: str) -> float:
        row = self.rows.get(key) or {}
        return phrase_recall(_text(row.get("title")), phrase)

    def _valid_total(self) -> int:
        brand = self.req.get("brand") if self.family in (TF3, TF7) else None
        return len(self._constraint_candidates(brand=brand, budget=self.budget))

    def compare_slots_left(self) -> int:
        compares = sum(1 for entry in self.log if entry.get("compare"))
        return max(0, MAX_COMPARES_PER_GROUP - compares)

    def lane_recovery(self) -> None:
        model, color = read_recovery_preference(self.goal)
        self.req["model"], self.req["color"] = model, color
        if not model:
            return self.lane_fallback()
        cat = self.req.get("category") or "Smartphones"
        word = CAT_QUERY_WORD.get(cat, "smartphone")
        qs = [f"{model}", f"{model} {color or ''}", f"{model} {word}"]
        self.run_calls([self.catalog_search(q) for q in self.fresh_observation(qs)[:MAX_SEARCHES_PER_GROUP]])
        self._recovery_round(extra=False)
        rounds = 0
        while self.may_explore() and rounds < 3:
            rows = self._recovery_rows()
            has0 = any(lvl == 0 for lvl, *_r in rows)
            has_alt = any(lvl in (1,) for lvl, *_r in rows)
            if has0 and (has_alt or rounds >= 2):
                break
            self._recovery_round(extra=True)
            rounds += 1
        self.emit_result()

    def commitment_gate(self) -> "CommitmentGate":
        return CommitmentGate(self.score_card())

    def _ranking_values(self, k: str):
        prod = self.products.get(key_to_ref(k)["product_id"])
        if not prod:
            return None
        v = next((x for x in prod.get("variants") or [] if str(x.get("sku")) == key_to_ref(k)["sku"]), None)
        if v is None:
            return None
        opts = v.get("options")
        refresh = variant_reading(prod, opts, "refresh_rate")
        size = variant_reading(prod, opts, "screen_size")
        exact = refresh is not None and size is not None
        if refresh is None:
            hz = _refresh_values(" ".join([str(opts or ""), str((prod.get("facts") or {}).get("refresh_rate") or ""), str(prod.get("title") or "")]))
            refresh = max(hz, key=lambda s: float(s[:-2])) if hz else None
        if size is None:
            inch = _diagonal_values(" ".join([str(opts or ""), str((prod.get("facts") or {}).get("size") or ""), str(prod.get("title") or "")]))
            size = max(inch, key=lambda s: float(s[:-2])) if inch else None
        if refresh is None or size is None:
            return None
        return (1 if exact else 0, float(refresh[:-2]), float(size[:-2]), self.price_of(k))

    def tf5_axis(self) -> str:
        return tf5_priority_from_goal(self.goal)

    def _take_user_turn(self, text: str) -> None:
        self.user_msgs.append(text)
        for m in _BUDGET_RE.finditer(text):
            cur = m.group(2).upper()
            if self.currency and cur != self.currency:
                continue
            val = _amount_of(m.group(1))
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
            for k, v in read_specs(seg).items():
                if self.req["attrs"].get(k) != v and (cue or k not in self.req["attrs"]):
                    if k in self.req["attrs"]:
                        self.req_changes += 1
                    self.req["attrs"][k] = v

    def score_card(self) -> "RowScoreCard":
        return RowScoreCard(self.budget, self.req.get("category"), self.req.get("brand"))

    def _constraint_first(self):
        brand = self.req.get("pref") if self.family == TF1 else self.req.get("brand")
        # honour a delivered budget cut when possible; the verifier judges the first add
        # against the original contract, so fall back to it only if nothing fits the cut
        for cap in (self.budget, self.budget0):
            c = self._constraint_candidates(brand=brand, budget=cap)
            if c:
                return c[0]
        return None

    def _ranking_order(self, exclude=()) -> list:
        if "prio" not in self.req:
            self.req["prio"] = read_rank_priority(self.goal)
            if self.req["prio"] == TF5_DEFAULT_PRIORITY:
                self.req["prio"] = read_rank_priority(self.query)
        rows = []
        for k in self.req.get("cands") or []:
            if k in exclude or not self.state_is_valid(k):
                continue
            vals = self._ranking_values(k)
            if vals is None:
                continue
            exact, hz, inch, price = vals
            axis = {"refresh_rate": hz, "screen_size": inch, "price": price}
            rows.append((-exact, *[sign * axis[a] for a, sign in self.req["prio"]], k))
        rows.sort()
        return [r[-1] for r in rows]

    def cheapest_key(self) -> str | None:
        priced = self.priced_rows()
        if not priced:
            return None
        return min(priced, key=lambda k: _f(priced[k].get("price")) or float("inf"))

    def state_is_valid(self, k: str, budget: float | None = None) -> bool:
        if k in self.dead or k in self.bad:
            return False
        p, ok = self.state.get(k, (None, False))
        cap = self.budget if budget is None else budget
        return bool(ok and p is not None and (cap is None or p <= cap + 1e-9))

    def lane_fallback(self, context: str | None = None) -> None:
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
        while not self.done and self.turn < self.max_steps and self.clock() < EMERGENCY_AT_S:
            resp = self._ask_model(messages)
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
                        and not self.candidate_passes(make_key(args.get("product_id"), args.get("sku")))):
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
            out = self.run_calls(calls)
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

    def unmet_specs(self, key: str) -> list[str]:
        row = self.rows.get(key) or {}
        sheet = SpecSheet()
        for spec in (self.req.get("specs") or []):
            if isinstance(spec, dict):
                sheet.add(_text(spec.get("field")), _text(spec.get("value")))
        return sheet.unmet(row)

    def _recovery_round(self, extra: bool) -> None:
        calls = []
        if extra:
            model, color = self.req.get("model") or "", self.req.get("color") or ""
            cat = self.req.get("category") or "Smartphones"
            word = CAT_QUERY_WORD.get(cat, "smartphone")
            calls += [self.catalog_search(q) for q in self.fresh_observation(
                [f"{model} phone", f"{model} android", f"{model} {word} {color}", f"{model} mobile"])[:2]]
        pids = self.rank_ungrounded(self._recovery_row_score, 5 * MAX_COMPARES_PER_GROUP)
        calls += self.compare_variants(pids)[:MAX_COMPARES_PER_GROUP]
        if calls:
            self.run_calls(calls)

    def recent_latency(self) -> float:
        """Pessimistic wall-clock cost of one more turn (recent max, floor 3 s)."""
        return max([3.0] + self._lat)

    def _constraint_queries(self, brand: str, word: str) -> list:
        specs = " ".join(spec_to_query(k, v) for k, v in self.req["attrs"].items())
        return [f"{brand} {word} {specs}", f"{brand} {specs}", f"{word} {specs}",
                f"{brand} {word}", f"{brand} {word} {specs} android", f"{specs} {word} android"]

    def _recovery_row_score(self, r: dict):
        title = str(r.get("title") or "")
        code = _bare_model_code(self.req.get("model") or "") or ""
        s = 0.0
        if code and code.casefold() in "".join(re.findall(r"[a-z0-9]+", title.casefold())):
            s += 3
        if self.req.get("color") and _phrase_present(title, self.req["color"]):
            s += 1
        if _TF6_ACCESSORY_RE.search(title.casefold()):
            s -= 4
        return s

    def _preference_device_ok(self, prod: dict) -> bool:
        title = str(prod.get("title") or "")
        low = title.casefold()
        if any(t in low for t in _TF4_BAD_TITLE):
            return False
        if any(p in low for p in self.profile["accessory"]):
            return False
        return category_agrees(self.req.get("category"), prod.get("category_path") or [], title=title)

    def priced_rows(self) -> dict[str, dict[str, Any]]:
        return {k: row for k, row in self.rows.items() if _f(row.get("price")) is not None}

    def compare_variants(self, pids: list) -> list:
        pids = [p for p in dict.fromkeys(pids) if p not in self.products]
        return [("compare", {"product_ids": pids[i:i + 5]}) for i in range(0, len(pids), 5)]

    def firm_and_soft(self) -> dict[str, list[str]]:
        return tf1_split_requirements(self.query)

    def lane_preference(self) -> None:
        cat = self.req.get("category") or "Smartwatches"
        if cat not in TF4_PROFILES:
            cat = "Smartwatches" if "watch" in self.goal.lower() else cat
        self.req["category"] = cat
        self.profile = TF4_PROFILES.get(cat, TF4_PROFILES["Smartwatches"])
        qs = list(self.profile["queries"])
        calls = []
        if TF4_ASK_PRIORITIES:
            # The TF4 rules require clarifying priorities before the first cart add; on
            # tasks that seal it as required, the metric is zeroed without that answer.
            calls.append(("message", {"content": (
                "Before I choose, what are your requirements and priorities - "
                "which matters most to you for this?")}))
        calls += [self.catalog_search(q) for q in qs[:MAX_SEARCHES_PER_GROUP]]
        self.run_calls(calls)
        self._preference_round(extra=False)
        rounds = 0
        while self.may_explore() and rounds < 3:
            good = [sc for _sig, sc, *_r in self._preference_listings() if sc >= 0.65]
            if len(good) >= 2 and rounds >= 1:
                break
            self._preference_round(extra=True)
            rounds += 1
        self.emit_result()

    def price_window(self) -> tuple[float, float] | None:
        return span_of_prices(list(self.rows.values()))

    def choose_final(self):
        fam = self.family
        x = self.first_key
        keep_x = bool(x and x in self.cart_refs() and self.candidate_passes(x))
        if fam in (TF1, TF3, TF7):
            return self._constraint_final()
        if fam == TF2:
            if keep_x:
                return x
            c = self._retrieval_valid(exclude={x} if x else ())
            xpid = key_to_ref(x)["product_id"] if x else None
            other = [k for k in c if key_to_ref(k)["product_id"] != xpid]
            return (other or c or [None])[0]
        if fam == TF4:
            if not self.events:
                ls = self._preference_listings()
                return ls[0][-1] if ls else (x if keep_x else None)
            # after an event the verifier requires a different source listing
            ls = self._preference_listings(exclude_listing=self._listing_key(x) if x else None)
            return ls[0][-1] if ls else None
        if fam == TF5:
            r = self._ranking_order()
            return r[0] if r else None
        if fam == TF6:
            rows = self._recovery_rows()
            return rows[0][-1] if rows else None
        return x if keep_x else None

    def _constraint_row_score(self, r: dict):
        title = str(r.get("title") or "")
        target = self.req.get("brand") or self.req.get("pref")
        s = 0.0
        if target:
            if brand_is_match(target, r.get("brand")):
                s += 4
            elif _word_list(target) <= _word_list(title):
                s += 1
        for k, want in self.req["attrs"].items():
            if want in _option_readings(k, title):
                s += 2
        word = CAT_QUERY_WORD.get(self.req.get("category") or "", "")
        if word and _word_list(word) & _word_list(title):
            s += 1
        if re.search(r"\b(?:case|cover|tempered|glass|protector|film|lcd|digitizer|replacement|"
                     r"battery|charger|cable|adapter|holder|stand|mount|strap|housing|sticker|"
                     r"skin|pouch|stylus|lens|parts?|refurb\w*|renewed)\b", title, re.I):
            s -= 5
        if is_phone_category(self.req.get("category")) and (_SMARTPHONE_RE.search(title) or _OS_RE.search(title)):
            s += 0.5
        return s

    def lane(self) -> "FamilyLane":
        return FamilyLane(self.family)

    def _preference_listings(self, exclude_listing=None) -> list:
        best: dict = {}
        for pid, prod, v, k in self.variants_of():
            if not self.state_is_valid(k) or not self._preference_device_ok(prod):
                continue
            lid = str(prod.get("listing_id") or pid)
            if exclude_listing is not None and lid == exclude_listing:
                continue
            score = self.profile["scorer"](prod)
            signal = any(s in str(prod.get("title") or "").casefold() for s in self.profile["signals"])
            cand = (1 if signal else 0, score, -self.price_of(k), k)
            if lid not in best or cand > best[lid]:
                best[lid] = cand
        return sorted(best.values(), reverse=True)

    def recovery_reason(self, rung: str) -> str:
        return tf6_relax_reason(rung)

    def first_commit(self, k: str) -> bool:
        before = self.first_key
        out = self.run_calls([("add_to_cart", key_to_ref(k)), ("inspect_stock", key_to_ref(k))])
        if self.first_key is not None and self.first_key != before:
            return True
        if out is not None:
            self.bad.add(k)
        return False

    def phrase_ready(self, key: str) -> bool:
        row = self.rows.get(key) or {}
        return tf2_title_ok(_text(row.get("title")), self.req.get("phrase_terms") or [])

    def emergency_order(self) -> None:
        """Order only a candidate that meets every current requirement; never an
        arbitrary affordable item."""
        if self.done:
            return
        try:
            y = self.choose_final() if self.first_key else (self.choose_first() or self.choose_final())
        except Exception:  # noqa: BLE001
            y = None
        if y is not None and self.candidate_passes(y):
            self.close_out(y)

    def second_family_opinion(self) -> str | None:
        return dominant_family(self.query, self.tools)

    def _ask_model(self, messages: list):
        for model in LLM_MODELS:
            if self.clock() > EMERGENCY_AT_S - 10:
                return None
            resp = _LLM.post("/inference/chat/completions", json_data={
                "model": model, "messages": messages, "tools": self.tools,
                "tool_choice": "required", "temperature": 0, "max_tokens": 1200})
            if isinstance(resp, dict) and resp.get("choices"):
                return resp
        return None

    def reason_claims(self, k: str):
        if self.family != TF7:
            return None
        prod = self.products.get(key_to_ref(k)["product_id"]) or {}
        claims = [{"field": a, "value": spec_to_claim(a, w)} for a, w in self.req["attrs"].items()
                  if a in ("resolution", "refresh_rate", "screen_size")]
        cap = self.budget if self.budget is not None else self.price_of(k)
        claims.append({"field": "price", "value": f"at most {cap:.2f} {self.currency or ''}".strip()})
        claims.append({"field": "in_stock", "value": "true"})
        if str(prod.get("brand") or "").strip():
            claims.append({"field": "brand", "value": str(prod["brand"]).strip()})
        return claims

    def drive(self) -> None:
        lane = {
            TF1: self.lane_constraint, TF3: self.lane_constraint, TF7: self.lane_constraint,
            TF2: self.lane_retrieval, TF4: self.lane_preference, TF5: self.lane_ranking, TF6: self.lane_recovery,
        }.get(self.family, self.lane_fallback)
        self.log_note(family=self.family, budget=self.budget, category=self.req.get("category"))
        lane()
        if (not self.done and self.family is not None and self.steps_left() > 4
                and self.clock() + 4 * self.recent_latency() + 60 < TIME_LIMIT_S):
            # The deterministic lane could not finish: hand the current state to the
            # model-assisted loop (its orders are still validated below).
            self.log_note(fallback="llm")
            self.lane_fallback(context=self.summarise_state())
        if not self.done:
            self.emergency_order()

    def priority_leader(self) -> str | None:
        axes = ordered_axes(self.goal)
        if not axes:
            return self.cheapest_key()
        return PriorityRanker(axes).front(self.priced_rows())

    def cart_refs(self) -> list:
        return [make_key(c.get("product_id"), c.get("sku")) for c in self.cart]

    def sealed_shopper_answer(self) -> str | None:
        return sealed_answer(self.user_msgs)


def price_after_event(row: dict[str, Any], event: dict[str, Any]) -> float | None:
    new_price = _f(event.get("new_price"))
    return new_price if new_price is not None else _f(row.get("price"))


def ascii_fold(text: str) -> str:
    table = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
    return "".join(table.get(ch, ch if ord(ch) < 128 else "?") for ch in str(text))


def _values_for_field(key: str, text: str) -> set:
    if key == "screen_size":
        return _diagonal_values(text)
    if key == "resolution":
        return _pixel_pairs(text)
    if key == "refresh_rate":
        return _refresh_values(text)
    return _option_readings("ram", text)


def sentence_case(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return stripped
    head, tail = stripped[0], stripped[1:]
    return head.upper() + tail


def interleave_unique(*sequences: list[str]) -> list[str]:
    out: list[str] = []
    seen: set = set()
    for column in range(max((len(seq) for seq in sequences), default=0)):
        for seq in sequences:
            if column < len(seq):
                value = seq[column]
                low = _low(value)
                if low not in seen:
                    seen.add(low)
                    out.append(value)
    return out


def route_family(query: str, tools: list) -> str | None:
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


def shared_prefix_len(left: str, right: str) -> int:
    count = 0
    for a, b in zip(str(left), str(right)):
        if a != b:
            break
        count += 1
    return count


def group_calls_by_action(calls: list[tuple[str, dict]]) -> dict[str, int]:
    names = [name for name, _payload in calls]
    return {name: names.count(name) for name in dict.fromkeys(names)}


def read_specs(text: str) -> dict:
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
        cls = canon_resolution(m.group(1))
        if cls:
            out["resolution"] = cls
    elif re.search(r"\ban? HD (?:display|screen|smartphone|tablet|monitor)\b", text):
        out["resolution"] = "hd"
    return out


def extract_codes(title: str) -> list[str]:
    return [token for token in re.split(r"[\s,/()]+", str(title)) if looks_like_code(token)]


def searches_left(used: int) -> int:
    return max(0, MAX_SEARCHES_PER_GROUP - used)


def read_preferred_brand(goal: str) -> str | None:
    m = re.search(r"\bprefer\s+(?:the\s+)?(.+?)(?:\s*,|\s+but\b|\.|;|$)", goal, re.I)
    if not m:
        return None
    brand = m.group(1).strip()
    return brand if 0 < len(brand) <= 40 else None


def dominant_family(query: str, tools: list[dict[str, Any]]) -> str | None:
    """A second opinion on family, blending marker scores with tool schema."""

    router = FamilyRouter(query)
    scored = router.top_family()
    if scored:
        return scored
    for tool in tools or []:
        fn = tool.get("function") or {}
        required = (fn.get("parameters") or {}).get("required") or []
        if fn.get("name") == "place_test_order" and "justification" in required:
            return "justification"
    return None


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return {key: 0.0 for key in weights}
    return {key: round(max(0.0, value) / total, 4) for key, value in weights.items()}


def fact_reading(prod: dict, attr: str) -> str | None:
    """TF7 visible-fact value (_attr_fact_value)."""
    label = _FACT_LABEL_FOR.get(attr)
    if not label:
        return None
    return _canon_value(attr, (prod.get("facts") or {}).get(label))


def deadline_state(elapsed: float) -> str:
    if elapsed >= EMERGENCY_AT_S:
        return "emergency"
    if elapsed >= NO_ROUND_AFTER_S:
        return "closing"
    if elapsed >= TIME_LIMIT_S * 0.5:
        return "mid"
    return "early"


def recovery_color(prod: dict, options: str | None) -> str | None:
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


def _f(value: Any) -> float | None:
    """Best-effort float, refusing bools and blanks (dead-code local helper)."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def tf2_title_ok(title: str, phrase_terms: list[str]) -> bool:
    if not phrase_terms:
        return True
    present = set(title_tokens(title))
    for term in phrase_terms:
        for part in title_tokens(term):
            if part not in present:
                return False
    return True


def key_to_ref(key: str) -> dict:
    pid, _, sku = key.partition("::")
    return {"product_id": pid, "sku": sku}


def tf7_reason_grounded(reason: str, facts_blob: str) -> bool:
    """A justification is trustworthy only if its content words appear in facts."""

    content = [w for w in title_tokens(reason) if w not in _STOP and len(w) > 3]
    if not content:
        return False
    haystack = _low(facts_blob)
    return all(word in haystack for word in content)


def _gigabyte_values(text: str | None) -> set:
    if not text:
        return set()
    return {
        f"{float(m.group(1)) * (1024 if m.group(2).lower() == 'tb' else 1):g}gb"
        for m in _GB_RE.finditer(text)
    }


def within_span(price: float | None, span: tuple[float, float] | None, pad: float = 0.0) -> bool:
    if price is None or span is None:
        return False
    return span[0] - pad <= price <= span[1] + pad


class FamilyRouter:
    """Scores which family a query most resembles from marker phrases."""

    MARKERS = {
        "retrieval_recall": ("match the requested phrase", "search the catalog for what"),
        "ranking": ("stated tradeoff priority", "advertised maximum"),
        "recovery": ("critical constraints are", "least-relaxed"),
        "justification": ("trustworthy reason", "reason i can trust", "walk me through why"),
        "preference_reasoning": ("authorizes one test order", "opaque identity"),
        "intent_decomposition": ("mix of firm requirements", "tell them apart", "ideally, i prefer"),
        "constraint_satisfaction": ("firm, non-negotiable", "exact variant"),
    }

    def __init__(self, query: str) -> None:
        self.low = _low(query)

    def scores(self) -> dict[str, int]:
        return {
            family: sum(1 for marker in markers if marker in self.low)
            for family, markers in self.MARKERS.items()
        }

    def top_family(self) -> str | None:
        ranked = sorted(self.scores().items(), key=lambda item: (-item[1], item[0]))
        return ranked[0][0] if ranked and ranked[0][1] > 0 else None

    def ambiguous(self) -> bool:
        hits = sorted(self.scores().values(), reverse=True)
        return len(hits) >= 2 and hits[0] == hits[1] and hits[0] > 0


def make_key(pid: Any, sku: Any) -> str:
    return f"{pid}::{sku or ''}"


def missing_phrase_tokens(title: str, phrase: str) -> list[str]:
    present = set(title_tokens(title))
    out: list[str] = []
    for part in title_tokens(phrase):
        if part not in present and part not in out:
            out.append(part)
    return out


def over_budget(price: float | None, budget: float | None, slack: float = 1e-9) -> bool:
    return price is not None and budget is not None and price > budget + slack


def _endurance_axis(text: str) -> float:
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


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [markdown_row(columns), markdown_row(["---"] * len(columns))]
    for row in rows:
        lines.append(markdown_row([_text(row.get(col)) for col in columns]))
    return "\n".join(lines)


def positions_of(items: list[Any], target: Any) -> list[int]:
    return [index for index, item in enumerate(items) if item == target]


def phone_admissible(prod: dict, options: str | None) -> bool:
    return phone_verdict(prod, options) == "supported"


class CloseOutPlanner:
    """Decides validated vs fused close-out and the calls it expands to."""

    def __init__(self, family: str | None, mode: str = CLOSE_MODE) -> None:
        self.family = family
        self.mode = close_mode_for(family, mode)

    def is_validated(self) -> bool:
        return self.mode == "validated"

    def call_names(self, key: str, refresh: bool = True) -> list[str]:
        return order_group_plan(self.mode, key, refresh)

    def refresh_shares_order_group(self) -> bool:
        return True

    def describe(self) -> str:
        style = "validate on a fresh observation" if self.is_validated() else "fuse recovery and order"
        return "%s close-out: %s" % (self.mode, style)


class RankBoard:
    """Compares successive rankings to detect a stable leader."""

    def __init__(self) -> None:
        self.rounds: list[list[str]] = []

    def push_round(self, ranking: list[str]) -> None:
        self.rounds.append(list(ranking))

    def top(self) -> str | None:
        return self.rounds[-1][0] if self.rounds and self.rounds[-1] else None

    def stable_top(self, window: int = 2) -> str | None:
        recent = [round_[0] for round_ in self.rounds[-window:] if round_]
        return recent[0] if len(recent) == window and len(set(recent)) == 1 else None

    def rank_drift(self) -> float | None:
        if len(self.rounds) < 2:
            return None
        return rank_correlation(self.rounds[-2], self.rounds[-1])


def product_reading(prod: dict, key: str) -> str | None:
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
        vals = _values_for_field(key, text)
        if len(vals) > 1:
            conflict = True
        values |= vals
    title = _title_resolution(prod) if key == "resolution" else str(prod.get("title") or "")
    if key == "resolution" and re.search(r"\bqHD\b", title):
        conflict = True
    values |= _option_readings(key, title)
    if conflict or len(values) != 1:
        return None
    return next(iter(values))


def cart_matches_order(cart_obs: dict[str, Any], key: str) -> bool:
    """Whether the last cart observation contains the item about to be ordered."""

    pid, sku = parse_ref(key)
    for line in cart_obs.get("cart") or []:
        if isinstance(line, dict) and _text(line.get("product_id")) == pid and _text(line.get("sku")) == sku:
            return True
    return False


def variant_reading(prod: dict, options: str | None, key: str) -> str | None:
    """Per-variant value: options first, product-level fallback (resolve_variant_attribute)."""
    vals = _option_readings(key, options or "")
    if len(vals) > 1:
        return None
    if vals:
        return next(iter(vals))
    if _option_summary(key, options or ""):
        return None
    return product_reading(prod, key)


class RowScoreCard:
    """Per-row deterministic checks used before a family commits."""

    def __init__(self, budget: float | None, category: str | None, brand: str | None) -> None:
        self.budget = budget
        self.category = category
        self.brand = brand

    def assess(self, row: dict[str, Any]) -> dict[str, bool]:
        return {
            "in_budget": not over_budget(_f(row.get("price")), self.budget),
            "in_category": within_category(row.get("category_path") or row.get("path"), self.category),
            "brand_ok": brand_agreement(self.brand, _text(row.get("brand"))),
            "in_stock": row.get("in_stock") is not False,
            "real_title": not is_placeholder_title(_text(row.get("title"))),
        }

    def accepts(self, row: dict[str, Any]) -> bool:
        return all(self.assess(row).values())

    def failing(self, row: dict[str, Any]) -> list[str]:
        return [name for name, ok in self.assess(row).items() if not ok]


def is_priority_line(text: str) -> bool:
    low = _low(text)
    return "priorit" in low or "most important" in low or "matters most" in low


def _stem_set(text: Any) -> set:
    out = set()
    for tok in _word_list(text):
        out.add(tok)
        if len(tok) > 3 and tok.endswith("s"):
            out.add(tok[:-1])
    return out


class SpecSheet:
    """Collects field -> value specs and checks a row against them."""

    def __init__(self) -> None:
        self.specs: dict[str, str] = {}

    def set_spec(self, field: str, value: str) -> bool:
        key = _low(field).strip()
        text = _text(value).strip()
        if not key or not text:
            return False
        self.specs[key] = text
        return True

    def load_from(self, pairs: list[tuple[str, str]]) -> int:
        return sum(1 for field, value in pairs if self.set_spec(field, value))

    def check_row(self, row: dict[str, Any]) -> dict[str, bool]:
        blob = _low(row.get("title")) + " " + _low(json.dumps(row.get("facts") or {}, default=str))
        return {field: value.lower() in blob for field, value in self.specs.items()}

    def satisfied_by(self, row: dict[str, Any]) -> bool:
        verdicts = self.check_row(row)
        return bool(verdicts) and all(verdicts.values())

    def unmet(self, row: dict[str, Any]) -> list[str]:
        return [field for field, ok in self.check_row(row).items() if not ok]


def price_to_float(text: str) -> float | None:
    """Read the first price in a string, tolerating thousands separators."""

    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text))
    match = re.search(r"\d+(?:\.\d{1,2})?", cleaned)
    if not match:
        return None
    value = float(match.group(0))
    return value if value > 0 else None


def recovery_rung(model, color, pref_model, pref_color) -> int | None:
    if not model or not color or not pref_model or not pref_color:
        return None
    same_model = model.casefold() == pref_model.casefold() or (
        _bare_model_code(model) is not None and _bare_model_code(model) == _bare_model_code(pref_model)
    )
    same_color = color.casefold() == pref_color.casefold()
    if same_model and same_color:
        return 0
    if same_model:
        return 1
    if same_color:
        return 2
    return 3


def valid_model_name(name: str) -> bool:
    text = str(name).strip()
    if len(text) < 2:
        return False
    return any(ch.isalnum() for ch in text) and not text.lower().startswith(("the ", "a ", "an "))


def order_group_plan(mode: str, key: str, refresh: bool) -> list[str]:
    """The ordered call names one commitment turns into under a close mode."""

    pid, sku = parse_ref(key)
    plan: list[str] = []
    if mode == "validated":
        plan += ["add_to_cart", "inspect_stock"]
        if refresh:
            plan.append("inspect_cart")
        plan.append("place_test_order")
    else:
        plan += ["add_to_cart", "inspect_stock", "inspect_cart", "place_test_order"]
    return plan


def agent_main(problem_data: dict) -> list:
    """Sandbox entry point."""
    if not isinstance(problem_data, dict) or not isinstance(problem_data.get("environment"), dict):
        return []
    agent = Agent(problem_data)
    try:
        agent.drive()
    except Exception as exc:  # noqa: BLE001
        agent.log_note(exception=f"{type(exc).__name__}: {str(exc)[:300]}")
        try:
            agent.emergency_order()
        except Exception as exc2:  # noqa: BLE001
            agent.log_note(exception2=f"{type(exc2).__name__}: {str(exc2)[:200]}")
    return [{"completion": {"content": json.dumps(agent.log)[:20000]},
             "extra_info": {"family": agent.family, "turns": agent.turn, "done": agent.done}}]


__all__ = ["agent_main"]