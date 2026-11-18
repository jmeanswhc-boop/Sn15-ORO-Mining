"""ORO generated-environment shopping agent (v4).

Design notes
------------
The scored object is the *process*, not only the final order:

* every task fires one market event on the FIRST successful cart add, so the
  first ``add_to_cart`` is a graded commitment.  A later correction does not
  repair a weak initial choice.
* the family name and the acceptance criteria are never exposed.  Behaviour is
  therefore derived from the public rules text that ships inside the query,
  from the tool schema, and from observations - never from task identifiers.
* a run that never places an order scores zero, so turn budget is managed
  explicitly and there is always a terminal fallback.

The agent keeps an explicit requirement state (hard constraints, soft
preferences, relaxation ladder), grounds every commitment in raw product facts
(``view`` / ``compare``), and runs a deterministic recovery ladder after an
event.  The model is used for semantics (parsing intent, scoring preference
fit, choosing among grounded candidates); all gating is deterministic.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

_CHUTES_CHAIN = [
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
]
_OPENROUTER = {
    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
}

# a killed run scores zero, so the terminal action has to happen well before
# any harness limit; these are wall-clock seconds for one task
_DEADLINE_S = float(getenv("ORO_AGENT_DEADLINE", "220"))
_FORCE_ORDER_AT = float(getenv("ORO_AGENT_FORCE_ORDER_AT", "0.55"))
_LLM_BUDGET = int(getenv("ORO_AGENT_LLM_CALLS", "10"))
_LLM_TIME_BUDGET = float(getenv("ORO_AGENT_LLM_SECONDS", "90"))
_LLM_TIMEOUT = float(getenv("ORO_AGENT_LLM_TIMEOUT", "40"))

# words that mark an accessory / non-device listing when a device is wanted
_ACCESSORY = (
    "case", "cover", "protector", "tempered glass", "charger", "cable",
    "adapter", "holder", "stand", "strap", "lanyard", "pouch", "sleeve",
    "skin", "sticker", "replacement part", "spare part", "screen film",
    "lens film", "mount", "dock", "stylus", "sim tray", "battery cover",
    "mystery box", "surprise box", "review", "manual", "voucher",
)
_MOBILE_OS = (
    "android", "ios", "iphone os", "blackberry", "windows phone", "symbian",
    "harmonyos", "hyperos", "sailfish", "ubuntu touch", "tizen",
)
_NON_SMARTPHONE = (
    "tablet", "feature phone", "basic phone", "keypad phone", "flip phone",
    "senior phone", "kid phone", "smart watch", "smartwatch",
)
_FEATURE_SYNONYMS = (
    {
        "pref": ("sweat", "water", "waterproof", "rain", "splash"),
        "facts": ("ipx", "ip6", "ip5", "waterproof", "water resistant", "sweatproof", "sweat"),
        "query": "ipx7 waterproof sweatproof",
    },
    {
        "pref": ("fit", "secure", "running", "gym", "sport", "workout"),
        "facts": ("ear hook", "earhook", "sport", "fin", "wing", "secure fit", "over ear"),
        "query": "sport secure fit ear hook",
    },
    {
        "pref": ("battery", "long", "endurance", "playtime", "hours"),
        # a bare cell capacity ("battery: 400") is not evidence of endurance;
        # the scored axis wants stated playtime
        "facts": ("hour", "hrs", "playtime", "playback", "standby"),
        "query": "long playtime hours battery life",
    },
    {
        "pref": ("noise", "anc", "cancel"),
        "facts": ("anc", "noise cancel", "enc"),
        "query": "active noise cancelling anc",
    },
    {
        "pref": ("mic", "call", "microphone"),
        "facts": ("mic", "microphone", "enc"),
        "query": "built in microphone clear call",
    },
)
_STOPWORDS = {
    "find", "place", "one", "test", "order", "buy", "get", "need", "want",
    "looking", "please", "for", "and", "the", "with", "that", "this", "has",
    "have", "can", "spend", "budget", "under", "below", "over", "most", "more",
    "than", "least", "price", "priced", "cost", "costs", "php", "usd", "stock",
    "instock", "available", "item", "listing", "listings", "product", "products",
    "option", "options", "something", "anything", "good", "best", "nice",
    "should", "must", "would", "like", "prefer", "ideally", "matters", "matter",
    "actually", "really", "just", "also", "but", "not", "its", "your", "you",
    "me", "my", "our", "their", "from", "into", "about", "within", "using",
    "make", "sure", "keep", "still", "then", "when", "while", "what", "which",
}
_COLORS = (
    "matte black", "jet black", "midnight black", "space grey", "space gray",
    "rose gold", "sage green", "sky blue", "navy blue", "light blue",
    "dark blue", "mint green", "dark green", "light green", "off white",
    "black", "white", "silver", "gold", "grey", "gray", "blue", "green",
    "red", "pink", "purple", "violet", "yellow", "orange", "brown", "beige",
    "cream", "titanium", "graphite", "lavender", "teal", "cyan", "bronze",
)

_NAMED_CLASSES = (
    ("4k", "4k"), ("uhd", "4k"), ("2160p", "4k"),
    ("qhd", "qhd"), ("1440p", "qhd"), ("2k", "qhd"),
    ("full hd", "fhd"), ("fhd", "fhd"), ("1080p", "fhd"),
    ("hd+", "hd"), ("720p", "hd"),
)

_EVENT_NOTICE = (
    "still available", "more expensive", "re-check", "recheck", "sold out",
    "out of stock", "went up", "price change", "just get", "still in stock",
)

_KEY_RE = re.compile(r"\b(\d{3,25})::([A-Za-z0-9._-]+)\b")
_AXIS_WORDS = (
    ("refresh_rate", ("refresh rate", "refresh", "hz")),
    ("screen_size", ("screen", "display size", "largest screen", "inch")),
    ("price", ("price", "cheapest", "lowest price")),
    ("battery", ("battery", "mah")),
    ("storage", ("storage", "rom", "capacity")),
    ("ram", ("ram", "memory")),
)
_AXIS_DIRECTION = {
    "price": "min",
    "refresh_rate": "max",
    "screen_size": "max",
    "battery": "max",
    "storage": "max",
    "ram": "max",
}


def tradeoff_order(text: str) -> list[str]:
    """Ordered tradeoff axes, taken from the order they appear in the ask."""

    low = _lc(text)
    hits: list[tuple[int, str]] = []
    for axis, words in _AXIS_WORDS:
        best = None
        for word in words:
            index = low.find(word)
            if index >= 0 and (best is None or index < best):
                best = index
        if best is not None:
            hits.append((best, axis))
    hits.sort()
    ordered: list[str] = []
    for _index, axis in hits:
        if axis not in ordered:
            ordered.append(axis)
    return ordered


def count_hits(text: str, words: tuple[str, ...]) -> int:
    low = _lc(text)
    return sum(1 for word in words if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(word), low))


def leaf_category(path: list[str] | None) -> str | None:
    for part in reversed(path or []):
        text = _as_str(part).strip()
        if text:
            return text
    return None


def model_norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _lc(value))


def ordering_disagreement(first: list[str], second: list[str]) -> float:
    shared = [item for item in first if item in second]
    if len(shared) < 2:
        return 0.0
    position = {item: second.index(item) for item in shared}
    disagree = total = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            total += 1
            if position[shared[i]] > position[shared[j]]:
                disagree += 1
    return disagree / float(total)


class RetrievalMetrics:
    """Coverage and query-count bookkeeping for the retrieval-scored families."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.filters = 0
        self.returned: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}

    def note_search(self, query: str, product_ids: list[str]) -> int:
        key = collapse_spaces(_lc(query))
        self.queries.append(key)
        fresh = 0
        for pid in product_ids:
            pid = _as_str(pid)
            if pid and pid not in self.first_seen:
                self.first_seen[pid] = len(self.queries)
                fresh += 1
        self.returned[key] = self.returned.get(key, 0) + len(product_ids)
        return fresh

    def note_filter(self) -> None:
        self.filters += 1

    def distinct_queries(self) -> int:
        return len(set(self.queries))

    def coverage(self, universe: int) -> float:
        if universe <= 0:
            return 0.0
        return min(1.0, len(self.first_seen) / float(universe))

    def penalty(self, target: int = 3) -> float:
        distinct = self.distinct_queries()
        return min(1.0, target / float(distinct)) if distinct else 1.0

    def estimated_reward(self, universe: int, target: int = 3) -> float:
        return round(self.coverage(universe) * self.penalty(target), 4)

    def summary(self) -> str:
        return "queries=%d distinct=%d filters=%d products=%d" % (
            len(self.queries), self.distinct_queries(), self.filters, len(self.first_seen)
        )


def brand_verified(candidate: Candidate, brand: str | None) -> bool:
    """Structured brand first - a brand word in the title proves nothing."""

    if not brand:
        return False
    if candidate.brand:
        return brand.lower() in candidate.brand.lower()
    return False


class LadderPlan:
    """Named rungs of the replacement ladder, most faithful first."""

    RUNGS = ("exact", "same_model", "same_color", "same_brand", "same_category", "any")

    def __init__(self, requirements: "Requirements") -> None:
        self.requirements = requirements
        self.rung = 0

    def rung_for(self, candidate: "Candidate", variant: dict[str, Any]) -> int:
        req = self.requirements
        model = model_norm(req.preferred_model or "")
        color = _lc(req.preferred_color)
        code = model_norm(catalog_fact(candidate, "model") or candidate.model_guess() or "")
        model_hit = bool(model) and bool(code) and (model in code or code in model)
        color_hit = bool(color) and (candidate.colour_of_variant(variant) or "") == color
        if model_hit and (color_hit or not color):
            return 0
        if model_hit:
            return 1
        if color_hit:
            return 2
        if req.brand and brand_verified(candidate, req.brand):
            return 3
        if req.category and category_matches(candidate, req.category):
            return 4
        return 5

    def label(self, rung: int) -> str:
        if 0 <= rung < len(self.RUNGS):
            return self.RUNGS[rung].replace("_", " ")
        return "unranked"

    def descend(self) -> str | None:
        if self.rung + 1 >= len(self.RUNGS):
            return None
        self.rung += 1
        return self.RUNGS[self.rung]


def relative_saving(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return round((before - after) / before, 4)


def is_market_ping(text: str) -> bool:
    """Does this shopper turn just surface the market event?

    Those need a corrected commitment, not a reply. Answering them spends a
    simulator call for nothing, and every simulator call can fail the run.
    """

    low = _lc(text)
    return any(phrase in low for phrase in _EVENT_NOTICE)


class ConstraintTally:
    """Per-candidate spec verdict grid (pass / fail / unresolved)."""

    def __init__(self, specs: list[dict[str, str]]) -> None:
        self.specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("field")]
        self.grid: dict[str, dict[str, bool | None]] = {}

    def evaluate(self, candidates: list["Candidate"]) -> None:
        for candidate in candidates:
            row = self.grid.setdefault(candidate.product_id, {})
            for spec in self.specs:
                row[_lc(spec.get("field"))] = spec_verdict(candidate, spec)

    def verdict(self, product_id: str, field: str) -> bool | None:
        return (self.grid.get(product_id) or {}).get(_lc(field))

    def passes(self, product_id: str) -> int:
        return sum(1 for value in (self.grid.get(product_id) or {}).values() if value is True)

    def fails(self, product_id: str) -> bool:
        return any(value is False for value in (self.grid.get(product_id) or {}).values())

    def unresolved(self, product_id: str) -> int:
        return sum(1 for value in (self.grid.get(product_id) or {}).values() if value is None)

    def leader(self) -> str | None:
        standing = sorted(
            (pid for pid in self.grid if not self.fails(pid)),
            key=lambda pid: (-self.passes(pid), self.unresolved(pid)),
        )
        return standing[0] if standing else None


def category_depth(path: list[str] | None) -> int:
    return len([part for part in (path or []) if _as_str(part).strip()])


def select_variant(
    candidate: Candidate, requirements: Requirements
) -> dict[str, Any] | None:
    """Best eligible variant, preferring the requested colour."""

    eligible = [v for v in candidate.variants if variant_allowed(candidate, v, requirements)]
    if not eligible and not candidate.variants:
        if candidate.sku and (
            candidate.price is None
            or requirements.budget is None
            or candidate.price <= requirements.budget + 1e-9
        ):
            if not requirements.require_in_stock or candidate.in_stock is not False:
                return {
                    "sku": candidate.sku,
                    "price": candidate.price,
                    "in_stock": candidate.in_stock,
                    "options": None,
                }
        return None
    if not eligible:
        return None
    wanted_color = _lc(requirements.preferred_color)
    if wanted_color:
        exact = [
            v
            for v in eligible
            if (candidate.colour_of_variant(v) or "") == wanted_color
        ]
        if exact:
            return exact[0]
    return eligible[0]


class RoundBoard:
    """Records the leader of each selection round to detect churn."""

    def __init__(self, width: int = 3) -> None:
        self.width = max(1, width)
        self.rounds: list[list[str]] = []

    def submit(self, pairs: list[tuple["Candidate", dict[str, Any]]]) -> None:
        self.rounds.append(
            ["%s::%s" % (c.product_id, _as_str(v.get("sku"))) for c, v in pairs[: self.width]]
        )

    def frontrunners(self) -> list[str]:
        return [row[0] for row in self.rounds if row]

    def settled(self) -> str | None:
        leaders = self.frontrunners()
        if len(leaders) >= 2 and leaders[-1] == leaders[-2]:
            return leaders[-1]
        return None

    def churn(self) -> float:
        if len(self.rounds) < 2:
            return 0.0
        return ordering_disagreement(self.rounds[-2], self.rounds[-1])

    def clear(self) -> None:
        self.rounds = []


def title_has_all(title: str, words: list[str]) -> bool:
    present = set(split_title(title))
    return all(word in present for word in words)


def pixel_tier(width: float, height: float) -> str | None:
    """Highest pixel class whose BOTH thresholds are met, orientation-free."""

    long_side, short_side = max(width, height), min(width, height)
    if long_side >= 3840 and short_side >= 2160:
        return "4k"
    if long_side >= 2560 and short_side >= 1440:
        return "qhd"
    if long_side >= 1920 and short_side >= 1080:
        return "fhd"
    if long_side >= 1280 and short_side >= 720:
        return "hd"
    return None


def option_value(variant: dict[str, Any], name: str) -> str | None:
    wanted = _lc(name)
    for key, value in parse_option_pairs(variant.get("options")):
        if key == wanted:
            return value
    return None


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def blank_option(variant: dict[str, Any]) -> bool:
    return not _as_str(variant.get("options")).strip()


def abbreviate(text: str, limit: int = 64) -> str:
    text = strip_bracketed(text)
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0]
    return head.rstrip(" -|,") + "\u2026"


class Candidate:
    def __init__(self, product_id: str) -> None:
        self.product_id = product_id
        self.title = ""
        self.brand: str | None = None
        self.price: float | None = None
        self.sku: str | None = None
        self.in_stock: bool | None = None
        self.category_path: list[str] = []
        self.facts: dict[str, Any] = {}
        self.variants: list[dict[str, Any]] = []
        self.grounded = False
        self.pref_score: float | None = None
        self.rejected: str | None = None

    def price_range(self) -> tuple[float, float] | None:
        prices = [_as_float(v.get("price")) for v in self.variants]
        known = [p for p in prices if p is not None]
        if not known:
            return None if self.price is None else (self.price, self.price)
        return min(known), max(known)

    def absorb_detail_row(self, row: dict[str, Any]) -> None:
        self.title = _as_str(row.get("title")) or self.title
        if row.get("brand"):
            self.brand = _as_str(row.get("brand"))
        path = row.get("category_path")
        if isinstance(path, list):
            self.category_path = [_as_str(p) for p in path]
        facts = row.get("facts")
        if isinstance(facts, dict):
            self.facts = facts
        variants = row.get("variants")
        if isinstance(variants, list):
            self.variants = [v for v in variants if isinstance(v, dict)]
        self.grounded = True

    def model_guess(self) -> str | None:
        fact = _lc(self.facts.get("model")) if self.facts else ""
        if fact.strip():
            return fact.strip()
        for token in split_title(self.title):
            if re.fullmatch(r"[a-z]{1,3}\d{1,4}[a-z]?", token):
                return token
        return None

    def distinct_colors(self) -> list[str]:
        colors = {self.colour_of_variant(v) for v in self.variants}
        return sorted(c for c in colors if c)

    def cheapest_price(self) -> float | None:
        span = self.price_range()
        return span[0] if span else None

    def facts_blob(self) -> str:
        chunks = [self.title]
        for key, value in (self.facts or {}).items():
            chunks.append("%s: %s" % (key, value))
        for variant in self.variants:
            chunks.append(_as_str(variant.get("options")))
        return " | ".join(c for c in chunks if c)

    def states_fact(self, name: str) -> bool:
        wanted = _lc(name)
        return any(
            _lc(key) == wanted and _as_str(value).strip() for key, value in (self.facts or {}).items()
        )

    def option_texts(self) -> list[str]:
        labels = []
        for variant in self.variants:
            text = _as_str(variant.get("options")).strip()
            if text and text not in labels:
                labels.append(text)
        return labels

    def colour_of_variant(self, variant: dict[str, Any]) -> str | None:
        options = _lc(variant.get("options"))
        if options:
            # the selected variant's own options win; a colour word elsewhere in
            # the listing says nothing about THIS sku
            sources = [options]
        else:
            sources = [_lc(self.facts.get("color")), _lc(self.title)]
        for source in sources:
            if not source:
                continue
            found = [color for color in _COLORS if color in source]
            if not found:
                continue
            longest = max(found, key=len)
            shorter = {c for c in found if c in longest or longest in c}
            if len(set(found) - shorter) > 0:
                return None  # several distinct colors -> unresolved
            return longest
        return None

    def spec_line(self) -> str:
        bits = []
        for field in ("model", "screen_size", "ram", "refresh_rate", "resolution"):
            value = catalog_fact(self, field)
            if value:
                bits.append("%s=%s" % (field, value))
        return "; ".join(bits) or "no facts"

    def has_any_stock(self) -> bool:
        if self.in_stock is True:
            return True
        return any(v.get("in_stock") is True for v in self.variants)

    def absorb_search_row(self, row: dict[str, Any]) -> None:
        self.title = _as_str(row.get("title")) or self.title
        if row.get("brand"):
            self.brand = _as_str(row.get("brand"))
        price = _as_float(row.get("price"))
        if price is not None:
            self.price = price
        if row.get("sku"):
            self.sku = _as_str(row.get("sku"))
        if isinstance(row.get("in_stock"), bool):
            self.in_stock = row["in_stock"]

    def lowest_variant(self, in_stock_only: bool = True) -> dict[str, Any] | None:
        pool = [
            variant
            for variant in self.variants
            if _as_float(variant.get("price")) is not None
            and (not in_stock_only or variant.get("in_stock") is True)
        ]
        if not pool:
            return None
        return min(pool, key=lambda variant: _as_float(variant.get("price")) or 0.0)

    def stock_resolved(self) -> bool:
        if self.in_stock is not None:
            return True
        return any(isinstance(v.get("in_stock"), bool) for v in self.variants)


def spare_budget(price: float | None, ceiling: float | None) -> float | None:
    if price is None or ceiling is None:
        return None
    return round(ceiling - price, 2)


def cap_text(value: Any, width: int = 90) -> str:
    text = _as_str(value).strip()
    return text[:width]


def device_matches(candidate: Candidate, requirements: Requirements) -> bool:
    if not requirements.device_required:
        return True
    if seems_accessory(candidate.title):
        return False
    if not requirements.smartphone_required:
        return True
    blob = _lc(candidate.facts_blob())
    if any(bad in blob for bad in _NON_SMARTPHONE) and "smartphone" not in blob:
        return False
    if "smartphone" in _lc(candidate.title):
        return True
    device_type = _lc(candidate.facts.get("phone_type")) + " " + _lc(
        candidate.facts.get("device_type")
    )
    if "smartphone" in device_type:
        return True
    os_text = " ".join(
        _lc(candidate.facts.get(key))
        for key in ("device_operating_system", "operating_system", "os")
    )
    return any(name in os_text for name in _MOBILE_OS)


def currency_symbol(code: str) -> str:
    return {"PHP": "\u20b1", "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3"}.get(_as_str(code).upper(), "")


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", _as_str(text)).strip()


def shopper_request(query: str) -> str:
    """Return the shopper-facing request without the appended task rules."""

    cut = query.find("Task rules:")
    return (query[:cut] if cut > 0 else query).strip()


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _lc(value)
    if text in ("true", "yes", "in stock", "available", "1"):
        return True
    if text in ("false", "no", "sold out", "out of stock", "0"):
        return False
    return None


def resolution_label(width: float, height: float) -> str:
    klass = pixel_tier(width, height)
    names = {"4k": "4K UHD", "qhd": "QHD", "fhd": "Full HD", "hd": "HD"}
    return names.get(klass or "", "sub-HD")


class Requirements:
    """Hard constraints, soft preferences and the relaxation ladder."""

    def __init__(self) -> None:
        self.category: str | None = None
        self.budget: float | None = None
        self.currency: str = "PHP"
        self.brand: str | None = None
        self.brand_is_soft: bool = False
        self.specs: list[dict[str, str]] = []
        self.phrase_terms: list[str] = []
        self.require_in_stock: bool = True
        self.device_required: bool = True
        self.smartphone_required: bool = False
        self.preferred_model: str | None = None
        self.preferred_color: str | None = None
        self.soft_prefs: list[str] = []
        self.requested_spec_field: str | None = None
        self.ranking_rule: str | None = None
        self.axes: list[str] = []
        self.listed: list[tuple[str, str]] = []

    def hard_spec_count(self) -> int:
        return len(self.specs)

    def spec_field_names(self) -> list[str]:
        return [_lc(spec.get("field")) for spec in self.specs if spec.get("field")]

    def named_targets(self) -> int:
        return len(self.listed)

    def describe(self) -> str:
        parts = []
        if self.category:
            parts.append("category=%s" % self.category)
        if self.budget is not None:
            parts.append("budget<=%.2f %s" % (self.budget, self.currency))
        if self.brand:
            parts.append(
                "brand=%s(%s)" % (self.brand, "soft" if self.brand_is_soft else "firm")
            )
        for spec in self.specs:
            parts.append("%s=%s" % (spec.get("field"), spec.get("value")))
        if self.phrase_terms:
            parts.append("title must contain tokens %s" % self.phrase_terms)
        if self.preferred_model:
            parts.append("preferred model=%s" % self.preferred_model)
        if self.preferred_color:
            parts.append("preferred color=%s" % self.preferred_color)
        if self.soft_prefs:
            parts.append("soft prefs: %s" % "; ".join(self.soft_prefs))
        parts.append("in_stock required" if self.require_in_stock else "stock optional")
        return ", ".join(parts)

    def soft_line(self) -> str:
        bits = list(self.soft_prefs)
        if self.brand and self.brand_is_soft:
            bits.append("brand %s (soft)" % self.brand)
        if self.preferred_color:
            bits.append("color %s" % self.preferred_color)
        return "; ".join(bits) or "none"

    def blank(self) -> bool:
        return not (
            self.category or self.brand or self.specs or self.phrase_terms
            or self.preferred_model or self.soft_prefs or self.budget is not None
        )

    def duplicate(self) -> "Requirements":
        other = Requirements()
        for key, value in self.__dict__.items():
            other.__dict__[key] = list(value) if isinstance(value, list) else value
        return other

    def summary_fields(self) -> list[str]:
        fields = []
        if self.category:
            fields.append("category")
        if self.brand:
            fields.append("brand")
        if self.budget is not None:
            fields.append("budget")
        if self.specs:
            fields.append("specs")
        if self.preferred_model:
            fields.append("model")
        return fields


def same_model_line(left: str | None, right: str | None) -> bool:
    left_letters, left_digits = model_series(left)
    right_letters, right_digits = model_series(right)
    if not left_digits or not right_digits:
        return False
    return left_letters[:2] == right_letters[:2] and left_digits[:2] == right_digits[:2]


class PriceWatch:
    """Tracks how a variant's price moves across a session."""

    def __init__(self) -> None:
        self.series: dict[tuple[str, str], list[tuple[int, float]]] = {}

    def observe(self, product_id: str, sku: str, price: Any, turn: int = 0) -> None:
        value = _as_float(price)
        if value is None:
            return
        rows = self.series.setdefault((product_id, sku), [])
        if not rows or abs(rows[-1][1] - value) > 1e-9:
            rows.append((turn, value))

    def observe_rows(self, rows: list[dict[str, Any]], turn: int = 0) -> int:
        count = 0
        for row in rows or []:
            if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                self.observe(_as_str(row["product_id"]), _as_str(row["sku"]), row.get("price"), turn)
                count += 1
        return count

    def current(self, product_id: str, sku: str) -> float | None:
        rows = self.series.get((product_id, sku)) or []
        return rows[-1][1] if rows else None

    def opening(self, product_id: str, sku: str) -> float | None:
        rows = self.series.get((product_id, sku)) or []
        return rows[0][1] if rows else None

    def rose(self, product_id: str, sku: str, tolerance: float = 0.01) -> bool:
        start = self.opening(product_id, sku)
        end = self.current(product_id, sku)
        return start is not None and end is not None and end > start + tolerance

    def drift(self, product_id: str, sku: str) -> float | None:
        return relative_saving(self.opening(product_id, sku), self.current(product_id, sku))

    def movers(self, limit: int = 5) -> str:
        moved = [
            "%s/%s %.2f->%.2f" % (pid, sku, rows[0][1], rows[-1][1])
            for (pid, sku), rows in self.series.items() if len(rows) > 1
        ]
        return "; ".join(moved[:limit]) or "no movement"


def passes_hard_gate(candidate: Candidate, requirements: Requirements) -> bool:
    """Product-level gate; variant-level checks happen separately."""

    if requirements.phrase_terms and not title_covers_phrase(candidate.title, requirements.phrase_terms):
        return False
    if not category_matches(candidate, requirements.category):
        return False
    if not device_matches(candidate, requirements):
        return False
    if requirements.brand and not requirements.brand_is_soft:
        brand = _lc(candidate.brand)
        if brand:
            if requirements.brand.lower() not in brand:
                return False
        elif requirements.brand.lower() not in _lc(candidate.title):
            return False
    if candidate.grounded:
        for spec in requirements.specs:
            if spec_verdict(candidate, spec) is not True:
                return False
    return True


def facts_conflict(left: "Candidate", right: "Candidate", field: str) -> bool:
    first = _lc(catalog_fact(left, field))
    second = _lc(catalog_fact(right, field))
    return bool(first) and bool(second) and first != second


def cart_lines(cart: list[dict[str, Any]]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for line in cart or []:
        if isinstance(line, dict) and line.get("product_id"):
            keys.append((_as_str(line.get("product_id")), _as_str(line.get("sku"))))
    return keys


def playtime_hours(candidate: "Candidate") -> float:
    """Largest stated playtime in hours, 0.0 when the listing never says."""

    low = _lc(candidate.facts_blob())
    values = [
        float(m.group(1))
        for m in re.finditer(r"(\d{1,3})\s*(?:hours|hour|hrs|hr)\b", low)
    ]
    return max(values) if values else 0.0


def is_meaningful_saving(before: float | None, after: float | None, floor: float = 0.05) -> bool:
    ratio = relative_saving(before, after)
    return ratio is not None and ratio >= floor


def normalise_whitespace_list(items: list[str]) -> list[str]:
    return [collapse_spaces(item) for item in items if collapse_spaces(item)]


def fact_backed_specs(candidate: "Candidate", specs: list[dict[str, str]]) -> int:
    """Specs confirmed by a catalog fact rather than by title text alone.

    Gold sets are fact-backed, so a listing whose title says "27 inch" while the
    catalog states nothing is a weaker pick than one whose facts say it.
    """

    count = 0
    for spec in specs:
        field = _lc(spec.get("field"))
        if catalog_fact(candidate, field) and spec_verdict(candidate, spec) is True:
            count += 1
    return count


def hertz_value(text: str) -> float | None:
    rates = [float(m.group(1)) for m in re.finditer(r"(\d{2,3})\s*hz", _lc(text))]
    plausible = [rate for rate in rates if 24 <= rate <= 540]
    return max(plausible) if plausible else None


def sku_family(sku: str | None) -> str:
    text = _as_str(sku).strip()
    return text.split("-", 1)[0] if "-" in text else text


def title_covers_phrase(title: str, terms: list[str]) -> bool:
    if not terms:
        return True
    toks = set(split_title(title))
    for term in terms:
        for part in split_title(term):
            if part not in toks:
                return False
    return True


def milliamp_hours(text: str) -> int | None:
    values = [int(m.group(1)) for m in re.finditer(r"(\d{3,5})\s*mah", _lc(text))]
    plausible = [value for value in values if 200 <= value <= 40000]
    return max(plausible) if plausible else None


def soft_pref_terms(prefs: list[str]) -> list[str]:
    terms: list[str] = []
    for pref in prefs or []:
        for word in split_title(pref):
            if len(word) > 3 and word not in _STOPWORDS and word not in terms:
                terms.append(word)
    return terms


class Agent:
    def __init__(self, problem: dict[str, Any]) -> None:
        environment = problem["environment"]
        self.policy = environment["policy_view"]
        self.problem_id = _as_str(
            problem.get("problem_id") or problem.get("id") or "problem"
        )
        self.dialogue: list[dict[str, Any]] = []
        self.log: list[str] = []
        self.client = Client()
        self.llm = LLM(self.client, self.log)
        self.env = Env(
            self.client,
            environment["binding"],
            self.policy,
            self.problem_id,
            self.dialogue,
            self.log,
        )
        self.query = _as_str(self.policy.get("query"))
        self.request = shopper_request(self.query)
        self.flags = Flags(self.query)
        self.req = Requirements()
        self.pool: dict[str, Candidate] = {}
        self.queries_run: list[str] = []
        self.started = time.monotonic()
        self.committed: tuple[str, str] | None = None  # product_id, sku
        self.commit_price: float | None = None
        self.dead: set[tuple[str, str]] = set()
        self.replacement_round = 0
        self.pending_question: str | None = None
        self.wants_endurance = False
        self.viewed_after_event: set[str] = set()
        self.search_budget = int(getenv("ORO_AGENT_SEARCHES", "10"))
        # retrieval scoring divides coverage by the number of distinct queries,
        # so extra searches actively destroy the reward once coverage is there
        self.retrieval_mode = bool(self.flags.phrase_rule or self.flags.family == "retrieval_recall")
        if self.retrieval_mode:
            self.search_budget = int(getenv("ORO_AGENT_RECALL_SEARCHES", "3"))
        # keep a couple of queries back for the post-event recovery
        self.search_reserve = 0 if self.retrieval_mode else 2
        # the event sequence after the commitment needs roughly eight turns
        self.commit_deadline = max(8, self.env.max_steps - 12)

    def process_shopper_turns(self) -> None:
        """Apply shopper clarifications / interventions to the live state."""

        while self.env.messages:
            content = self.env.messages.pop(0)
            self.log.append("shopper: %s" % content[:160])
            if "?" in content and not is_market_ping(content):
                self.pending_question = content
            budget = read_budget(content)
            if budget is not None:
                if self.req.budget is None or budget < self.req.budget:
                    self.req.budget = budget
            match = re.search(
                r"required catalog category is ([A-Za-z &'/-]+)", content
            )
            if match:
                self.req.category = match.group(1).strip().rstrip(".")
            match = re.search(r"required brand is ([A-Za-z0-9 &'-]+)", content)
            if match:
                self.req.brand = match.group(1).strip().rstrip(".")
                self.req.brand_is_soft = False
            # a shopper turn may be the only place a firm spec is ever stated
            known = {_lc(s.get("field")) for s in self.req.specs}
            for spec in specs_from_text(content):
                if spec["field"] not in known:
                    self.req.specs.append(spec)
                    self.log.append("spec from shopper: %s=%s" % (spec["field"], spec["value"]))
                else:
                    for existing in self.req.specs:
                        if _lc(existing.get("field")) == spec["field"]:
                            existing["value"] = spec["value"]
            caught = budget is not None or bool(specs_from_text(content))
            if not caught and not self.order_clock_hit and len(content) > 20:
                payload = self.llm.query_model(
                    "You update shopping requirements. Reply with JSON only, no prose.",
                    _UPDATE_PROMPT % (self.req.describe(), content),
                    max_tokens=500,
                )
                if isinstance(payload, dict):
                    self.fold_in_payload(payload)
            self.log.append("requirements now: %s" % self.req.describe())

    @property
    def deadline_passed(self) -> bool:
        return self.seconds_in > _DEADLINE_S

    def guaranteed_order(self) -> None:
        """Last resort: a valid order always beats no terminal action."""

        if self.env.ordered or self.env.done or self.env.steps_left < 1:
            return
        # a transient proxy failure earlier must not stop us ordering now
        self.env.failed = False
        self.env.consecutive_failures = 0
        if self.committed and self.committed not in self.dead:
            if self.submit_order():
                return
        seen: list[tuple[Candidate, dict[str, Any]]] = []
        ranked = sorted(
            self.pool.values(),
            key=lambda c: (
                -spec_standing(c, self.req.specs),
                -fact_backed_specs(c, self.req.specs),
                -self.cheap_rank(c),
            ),
        )
        for candidate in ranked:
            variant = select_variant(candidate, self.req)
            if variant is not None:
                seen.append((candidate, variant))
        if not seen:
            # drop every soft filter but never the budget or stock requirement
            for candidate in self.pool.values():
                for variant in candidate.variants or []:
                    if variant_allowed(candidate, variant, self.req):
                        seen.append((candidate, variant))
                        break
        for candidate, variant in seen[:3]:
            key = (candidate.product_id, _as_str(variant.get("sku")))
            if key in self.dead or self.env.steps_left < 2:
                break
            if self.committed and self.committed != key:
                self.env.single_call(
                    "remove_from_cart",
                    {"product_id": self.committed[0], "sku": self.committed[1]},
                )
            if not self.env.single_call("add_to_cart", {"product_id": key[0], "sku": key[1]}):
                continue
            self.committed = key
            self.commit_price = _as_float(variant.get("price"))
            if self.submit_order():
                return

    def grade_preferences(self, pairs: list[tuple[Candidate, dict[str, Any]]]) -> None:
        need = [p for p in pairs if p[0].pref_score is None]
        if not need:
            return
        if not self.req.soft_prefs and not self.flags.preference_scored:
            for candidate, _variant in need:
                candidate.pref_score = 0.5
            return
        items = []
        for candidate, variant in need[:8]:
            items.append(
                {
                    "product_id": candidate.product_id,
                    "title": candidate.title[:180],
                    "brand": candidate.brand,
                    "price": variant.get("price"),
                    "facts": {k: _as_str(v)[:90] for k, v in list((candidate.facts or {}).items())[:12]},
                    "variant": _as_str(variant.get("options"))[:60],
                }
            )
        if self.order_clock_hit:
            for candidate, _variant in need:
                candidate.pref_score = self._heuristic_pref(candidate)
            return
        payload = self.llm.query_model(
            "You score how well products match a shopper's stated needs. JSON only.",
            _SCORE_PROMPT
            % (
                self.request[:700],
                "; ".join(self.req.soft_prefs) or "(none stated)",
                self.req.ranking_rule or "(none)",
                json.dumps(items, ensure_ascii=False)[:6000],
            ),
            max_tokens=700,
        )
        scores: dict[str, float] = {}
        if isinstance(payload, dict):
            payload = payload.get("scores", payload)
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    value = _as_float(row.get("score"))
                    key = _as_str(row.get("product_id"))
                    if key and value is not None:
                        scores[key] = max(0.0, min(1.0, value))
        elif isinstance(payload, dict):
            for key, value in payload.items():
                number = _as_float(value)
                if number is not None:
                    scores[_as_str(key)] = max(0.0, min(1.0, number))
        for candidate, _variant in need:
            candidate.pref_score = scores.get(candidate.product_id, self._heuristic_pref(candidate))

    def preference_axes_seen(self) -> int:
        return sum(1 for c in self.pool.values() if c.grounded and self.pref_hit_count(c) > 0)

    def read_requirements(self) -> None:
        req = self.req
        req.budget = read_budget(self.request)
        if "php" in _lc(self.request):
            req.currency = "PHP"
        quoted = re.findall(r'"([^"]+)"', self.request) + re.findall(
            r"matches the phrase ([a-z0-9 ]+?)(?:,| at| in| and)", _lc(self.request)
        )
        if self.flags.phrase_rule and quoted:
            req.phrase_terms = [t for t in split_title(quoted[0])]
        request_tokens = set(split_title(self.request))
        req.smartphone_required = bool(
            request_tokens & {"smartphone", "smartphones", "cellphone", "phone", "phones"}
        )
        req.listed = explicit_variants(self.request)
        req.axes = tradeoff_order(self.request) if req.listed or self.flags.ranking else []
        for spec in specs_from_text(self.request):
            if spec["field"] not in {_lc(s.get("field")) for s in req.specs}:
                req.specs.append(spec)
        payload = self.llm.query_model(
            "You extract shopping requirements. Reply with JSON only, no prose.",
            _EXTRACT_PROMPT % (self.request, self.flags.text[:1800]),
            max_tokens=700,
        )
        if isinstance(payload, dict):
            self.fold_in_payload(payload)
        if req.budget is None:
            req.budget = read_budget(self.query)
        self.wants_endurance = any(
            any(word in _lc(pref) for word in ("battery", "playtime", "long-lasting", "endurance", "hours"))
            for pref in req.soft_prefs
        )
        self.log.append("requirements: %s" % req.describe())

    def select_best(self) -> tuple[Candidate, dict[str, Any]] | None:
        """Ground the shortlist, gate it, and return the genuine best choice."""

        wide = bool(self.req.specs or self.req.preferred_model or self.req.soft_prefs)
        for _round in range(4):
            shortlist = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)
            if not shortlist:
                break
            self.fetch_facts(shortlist[:5])
            if wide and _round == 0 and not self.should_wrap_up(10):
                # gold sets are fact-backed, so look at more raw facts before
                # the graded first commitment
                self.fetch_facts([c for c in shortlist[5:10] if not c.grounded])
            self.process_shopper_turns()
            pairs = self.valid_pairs()
            if pairs and self.flags.preference_scored and _round == 0:
                # a first commitment with no support for any stated preference
                # is what the verifier calls an unintended commitment
                if not any(self.pref_hit_count(c) for c, _v in pairs):
                    extra = self.feature_targeted_queries()
                    if extra and not self.should_wrap_up(9):
                        self.do_searches(extra)
                        more = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)
                        self.fetch_facts([c for c in more if not c.grounded][:5])
                        pairs = self.valid_pairs()
            if pairs and self.req.preferred_model and _round == 0:
                if not any(self.relaxation_rung(c, v) == 0 for c, v in pairs):
                    self.chase_model_code()
                    pairs = self.valid_pairs()
            if pairs:
                self.grade_preferences(pairs)
                ranked = self.order_by_tradeoffs(pairs) if self.req.axes else self.order_pairs(pairs)
                if self.req.brand and self.req.brand_is_soft:
                    # honour a named soft brand when a fully compliant item exists
                    preferred = [
                        p for p in ranked if brand_verified(p[0], self.req.brand)
                    ]
                    if preferred:
                        ranked = preferred + [p for p in ranked if p not in preferred]
                best = ranked[0]
                room = (
                    self.search_budget > 0
                    and self.env.turn < self.commit_deadline
                    and not self.should_wrap_up(10)
                    and not self.order_clock_hit
                    and _round < 2
                )
                if self.good_enough_to_commit(best) or not room:
                    return best
                self.log.append("holding out for a stronger first commitment")
            if self.should_wrap_up(8) or self.env.done:
                break
            if self.search_budget <= 0 or self.env.turn >= self.commit_deadline:
                break
            extra: list[str] = []
            if self.req.specs:
                extra = self.spec_targeted_queries()
            if not extra and self.flags.preference_scored:
                extra = self.feature_targeted_queries()
            if not extra:
                extra = self.model_suggested_queries(self.outstanding_needs())
            if not extra:
                extra = self.rule_based_queries()
            if not extra:
                break
            self.do_searches(extra)
            shortlist = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)
            self.fetch_facts([c for c in shortlist if not c.grounded][:5])
        best_effort = self.valid_pairs()
        if best_effort:
            self.grade_preferences(best_effort)
            ranked = (
                self.order_by_tradeoffs(best_effort) if self.req.axes else self.order_pairs(best_effort)
            )
            return ranked[0]
        return None

    def keywords_of_request(self, limit: int = 8) -> list[str]:
        """Content words of the shopper request, in order, minus boilerplate."""

        out: list[str] = []
        for token in split_title(self.request):
            if token in _STOPWORDS or len(token) < 3 or token.isdigit():
                continue
            if token not in out:
                out.append(token)
        return out[:limit]

    def grounded_fraction(self) -> float:
        if not self.pool:
            return 0.0
        return round(sum(1 for c in self.pool.values() if c.grounded) / float(len(self.pool)), 3)

    def chase_model_code(self) -> None:
        """Chase an explicitly named model code before committing to anything.

        The code usually lives in the catalog's model fact rather than the
        title, so the bare token is the query that reaches it.
        """

        model = self.req.preferred_model
        if not model or self.env.done or self.should_wrap_up(8):
            return
        for pair in self.valid_pairs():
            if self.relaxation_rung(*pair) == 0:
                return  # a full match is already in hand
        queries = [model]
        if self.req.preferred_color:
            queries.append("%s %s" % (model, self.req.preferred_color))
        if self.req.category:
            queries.append("%s %s" % (model, self.req.category))
        queries = [q for q in queries if q.lower() not in {x.lower() for x in self.queries_run}]
        if not queries:
            return
        self.do_searches(queries[:2])
        self.process_shopper_turns()
        shortlist = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)
        self.fetch_facts([c for c in shortlist if not c.grounded][:5])

    def named_variant_count(self) -> int:
        return len(self.req.listed)

    def cheap_rank(self, candidate: Candidate) -> float:
        req = self.req
        blob = _lc(candidate.title + " " + " ".join(map(_as_str, (candidate.facts or {}).values())))
        score = 0.0
        if req.brand:
            if brand_verified(candidate, req.brand):
                score += 3.0
            elif req.brand.lower() in blob:
                score += 0.5
        if req.preferred_model and req.preferred_model.lower() in blob:
            score += 2.0
        if req.preferred_color and req.preferred_color.lower() in blob:
            score += 1.0
        for spec in req.specs:
            value = _lc(spec.get("value"))
            head = value.split()[0] if value else ""
            if head and head in blob:
                score += 1.5
        for pref in req.soft_prefs:
            words = [w for w in split_title(pref) if len(w) > 3]
            hits = sum(1 for w in words if w in blob)
            if words:
                score += 1.2 * hits / len(words)
        if candidate.grounded:
            score += 0.4
        return score

    def handle_named_variants(self) -> bool:
        """The shopper named the exact variants: compare them and rank."""

        listed = self.req.listed
        if not listed:
            return False
        wanted_skus = {key for key in listed}
        product_ids: list[str] = []
        for product_id, _sku in listed:
            if product_id not in product_ids:
                product_ids.append(product_id)
        for start in range(0, len(product_ids), 5):
            if self.env.done or self.should_wrap_up(6):
                break
            batch = product_ids[start : start + 5]
            if len(batch) == 1:
                observation = self.env.single_call("view", {"product_id": batch[0]})
                if observation and observation.get("product_id"):
                    self.pool.setdefault(batch[0], Candidate(batch[0])).absorb_detail_row(observation)
            else:
                observation = self.env.single_call("compare", {"product_ids": batch})
                for row in observation.get("compared") or []:
                    if isinstance(row, dict) and row.get("product_id"):
                        product_id = _as_str(row["product_id"])
                        self.pool.setdefault(product_id, Candidate(product_id)).absorb_detail_row(row)
        self.process_shopper_turns()

        def live_options() -> list[tuple[Candidate, dict[str, Any]]]:
            out = []
            for product_id, sku in listed:
                candidate = self.pool.get(product_id)
                if candidate is None:
                    continue
                for variant in candidate.variants:
                    if _as_str(variant.get("sku")) != sku:
                        continue
                    if (product_id, sku) in self.dead:
                        continue
                    if variant_allowed(candidate, variant, self.req):
                        out.append((candidate, variant))
            return self.order_by_tradeoffs(out)

        options = live_options()
        if not options:
            return False
        candidate, variant = options[0]
        if not self.add_first_to_cart(candidate, variant):
            for candidate, variant in options[1:]:
                if self.add_first_to_cart(candidate, variant):
                    break
        if not self.committed:
            return False
        state = self.check_after_commit()
        if state != "ok":
            kind = _lc((self.env.pending_event or {}).get("kind")) or "stockout"
            self.log_event(kind)
            # the market moved: re-rank the named candidates on live state
            for product_id, sku in listed:
                if (product_id, sku) in self.dead or self.should_wrap_up(5):
                    continue
                if (product_id, sku) == self.committed:
                    continue
                fresh = self.live_stock_check(product_id, sku)
                candidate = self.pool.get(product_id)
                if candidate is None:
                    continue
                for row in candidate.variants:
                    if _as_str(row.get("sku")) == sku:
                        if _as_float(fresh.get("price")) is not None:
                            row["price"] = _as_float(fresh.get("price"))
                        if isinstance(fresh.get("in_stock"), bool):
                            row["in_stock"] = fresh["in_stock"]
            options = live_options()
            for candidate, variant in options:
                key = (candidate.product_id, _as_str(variant.get("sku")))
                if key in self.dead:
                    continue
                if self.committed and self.committed[0] != key[0]:
                    self.env.single_call(
                        "remove_from_cart",
                        {"product_id": self.committed[0], "sku": self.committed[1]},
                    )
                observation = self.env.single_call(
                    "add_to_cart", {"product_id": key[0], "sku": key[1]}
                )
                if observation:
                    self.committed = key
                    self.commit_price = _as_float(variant.get("price"))
                    break
        self.settle_order()
        return bool(self.env.ordered)

    def do_filter(self) -> int:
        req = self.req
        args: dict[str, Any] = {"k": 10}
        if req.category:
            args["category"] = req.category
        if req.brand and not req.brand_is_soft:
            args["brand"] = req.brand
        if req.budget is not None:
            args["max_price"] = req.budget
        if req.require_in_stock:
            args["in_stock"] = True
        if len(args) <= 1:
            return 0
        before = len(self.pool)
        observation = self.env.single_call("filter", args)
        self.absorb_rows(observation.get("results"))
        return len(self.pool) - before

    def model_suggested_queries(self, missing: str) -> list[str]:
        payload = self.llm.query_model(
            "You write catalog search queries. Reply with JSON only.",
            _QUERY_PROMPT
            % (
                self.request[:600],
                self.req.describe(),
                json.dumps(self.queries_run[-6:]),
                missing,
            ),
            max_tokens=350,
        )
        out: list[str] = []
        if isinstance(payload, dict):
            payload = payload.get("queries")
        if isinstance(payload, list):
            for item in payload:
                text = _as_str(item).strip()
                if text and text.lower() not in {q.lower() for q in self.queries_run}:
                    out.append(text)
        return out[:3]

    def spec_targeted_queries(self) -> list[str]:
        """Searches aimed at the exact attribute value the task gates on."""

        parts: list[str] = []
        category = self.req.category or ""
        brand = self.req.brand or ""
        for spec in self.req.specs[:2]:
            value = _as_str(spec.get("value"))
            if not value:
                continue
            parts.append(" ".join(p for p in (brand, category, value) if p))
            parts.append(" ".join(p for p in (category, value) if p))
            if _lc(spec.get("field")) == "resolution":
                classes = {"hd": "720p hd", "fhd": "1080p full hd", "qhd": "1440p 2k", "4k": "4k uhd"}
                klass = requested_pixel_tier(value)
                if klass:
                    parts.append(" ".join(p for p in (category, classes[klass]) if p))
        tried = {q.lower() for q in self.queries_run}
        out: list[str] = []
        for query in parts:
            query = re.sub(r"\s+", " ", query).strip()
            if query and query.lower() not in tried and query.lower() not in {o.lower() for o in out}:
                out.append(query)
        return out[:2]

    def runner_up(self, product_id: str, limit: int = 3):
        pairs = [pair for pair in self.valid_pairs() if pair[0].product_id != product_id]
        if not pairs:
            return []
        ranked = self.order_by_tradeoffs(pairs) if self.req.axes else self.order_pairs(pairs)
        return ranked[:limit]

    def close_dialogue(self) -> list[dict[str, Any]]:
        if not self.env.ordered:
            try:
                self.guaranteed_order()
            except Exception as exc:
                self.log.append("close_out failed: %s" % type(exc).__name__)
        self.dialogue.append(
            {"role": "assistant", "content": "summary: %s" % " | ".join(self.log[-25:])}
        )
        return self.dialogue

    def dead_count(self) -> int:
        return len(self.dead)

    def build_justification(self) -> dict[str, Any] | None:
        if not self.flags.justification or not self.committed:
            return None
        product_id, _sku = self.committed
        candidate = self.pool.get(product_id)
        allowed = {
            "brand",
            "category",
            "in_stock",
            "price",
            "refresh_rate",
            "resolution",
            "screen_size",
        }
        claims: list[dict[str, str]] = []
        field = _lc(self.req.requested_spec_field)
        if not field:
            for spec in self.req.specs:
                candidate_field = _lc(spec.get("field"))
                if candidate_field in allowed:
                    field = candidate_field
                    break
        if candidate is not None and field in allowed:
            # claim the spec only when the catalog actually states it
            value = ""
            canonical = catalog_fact(candidate, field)
            if field == "resolution":
                classes = pixel_tiers_asserted(canonical or candidate.facts_blob())
                if len(set(classes)) == 1:
                    value = classes[0]
            elif canonical:
                value = canonical
            if not value:
                for spec in self.req.specs:
                    if _lc(spec.get("field")) == field and spec_verdict(candidate, spec) is True:
                        value = _as_str(spec.get("value"))
            if value:
                claims.append({"field": field, "value": value})
        if self.commit_price is not None:
            claims.append({"field": "price", "value": "%.2f" % self.commit_price})
        claims.append({"field": "in_stock", "value": "true"})
        if candidate is not None and candidate.brand and candidate.grounded:
            claims.append({"field": "brand", "value": candidate.brand})
        elif candidate is not None and candidate.category_path:
            claims.append(
                {"field": "category", "value": candidate.category_path[-1]}
            )
        if not claims:
            return None
        return {"claims": claims[:5]}

    def coarse_filter(self) -> list[Candidate]:
        req = self.req
        out = []
        for candidate in self.pool.values():
            if candidate.rejected:
                continue
            if req.phrase_terms and not title_covers_phrase(candidate.title, req.phrase_terms):
                continue
            if req.budget is not None and candidate.price is not None:
                if candidate.price > req.budget + 1e-9 and not candidate.variants:
                    continue
            if req.require_in_stock and candidate.in_stock is False and not candidate.variants:
                continue
            if req.device_required and seems_accessory(candidate.title):
                continue
            out.append(candidate)
        return out

    def ask_to_clarify(self) -> None:
        if self.env.done or self.should_wrap_up(6):
            return
        if not (self.flags.hidden_spec or self.flags.clarify_prefs or self.req.budget is None):
            return
        asks = []
        if self.flags.hidden_spec:
            asks.append("which exact hardware spec is the firm requirement (and its value)")
        if self.flags.clarify_prefs:
            asks.append("which of your preferences matters most, in order")
        if self.req.budget is None:
            asks.append("your maximum budget")
        if self.req.category is None:
            asks.append("the exact catalog category")
        if not asks:
            return
        content = "Before I search, could you confirm %s? I will then pick one item that meets everything." % (
            ", and ".join(asks[:3])
        )
        self.env.single_call("message", {"content": sanitised_message(content, 300)})
        self.process_shopper_turns()

    def eligible_colors(self) -> list[str]:
        colors: set[str] = set()
        for candidate, variant in self.valid_pairs():
            color = candidate.colour_of_variant(variant)
            if color:
                colors.add(color)
        return sorted(colors)

    def rule_based_queries(self) -> list[str]:
        """Reformulations that need no model: reorder, split, singularise."""

        keywords = self.keywords_of_request()
        tried = {q.lower() for q in self.queries_run}
        options: list[str] = []
        if keywords:
            options.append(" ".join(keywords[:3]))
            options.append(" ".join(keywords[1:4]))
            options.append(keywords[0])
            options.append(" ".join(keywords[-3:]))
            options.append(" ".join(k.rstrip("s") for k in keywords[:3]))
        if self.req.category:
            for pref in self.req.soft_prefs[:2]:
                options.append("%s %s" % (self.req.category, pref))
            if self.req.brand:
                options.append("%s %s" % (self.req.brand, self.req.category))
        out: list[str] = []
        for option in options:
            option = re.sub(r"\s+", " ", option).strip()
            if option and option.lower() not in tried and option.lower() not in {
                o.lower() for o in out
            }:
                out.append(option)
        return out[:3]

    @property
    def order_clock_hit(self) -> bool:
        """Time to stop improving and place the order."""

        return self.seconds_in > _DEADLINE_S * _FORCE_ORDER_AT

    def pref_hit_count(self, candidate: Candidate) -> int:
        """How many stated preference axes the raw facts actually support."""

        if not self.req.soft_prefs:
            return 0
        blob = _lc(candidate.facts_blob())
        hits = 0
        for pref in self.req.soft_prefs:
            words = [w for w in split_title(pref) if len(w) > 3 and w not in _STOPWORDS]
            if any(word in blob for word in words):
                hits += 1
                continue
            for family in _FEATURE_SYNONYMS:
                if any(word in _lc(pref) for word in family["pref"]) and any(
                    token in blob for token in family["facts"]
                ):
                    hits += 1
                    break
        return hits

    def check_after_commit(self) -> str:
        """The unaided turn after the commitment. Never terminal.

        Returns 'ok', 'replace' or 'gone'.
        """

        if not self.committed or self.env.done:
            return "gone"
        product_id, sku = self.committed
        observations = self.env.run_turn(
            [
                ("inspect_stock", {"product_id": product_id, "sku": sku}),
                ("inspect_cart", {}),
            ]
        )
        self.process_shopper_turns()
        state = observations[0] if observations else {}
        price = _as_float(state.get("price"))
        in_stock = state.get("in_stock")
        event = self.env.pending_event or {}
        kind = _lc(event.get("kind"))
        if kind == "stockout" or in_stock is False:
            self.dead.add((product_id, sku))
            return "replace"
        if price is not None:
            self.commit_price = price
            if self.req.budget is not None and price > self.req.budget + 1e-9:
                self.dead.add((product_id, sku))
                return "replace"
        if kind == "price_change" and price is None:
            new_price = _as_float(event.get("new_price"))
            if new_price is not None and self.req.budget is not None:
                if new_price > self.req.budget + 1e-9:
                    self.dead.add((product_id, sku))
                    return "replace"
                self.commit_price = new_price
        return "ok"

    def queries_look_stale(self, window: int = 2) -> bool:
        recent = [q.lower() for q in self.queries_run[-window:]]
        if len(recent) < window:
            return False
        return all(shared_keyword_count(recent[0], other) >= max(1, len(keyword_set(recent[0])) - 1)
                   for other in recent[1:])

    def type_words(self) -> list[str]:
        """Distinctive product-type words in the ask ("gaming", "foldable").

        They are not the category and not the spec, but the acceptance contract
        still counts them: a plain monitor is not a gaming monitor.
        """

        drop = set(_STOPWORDS)
        for source in (self.req.category or "", self.req.brand or ""):
            drop.update(split_title(source))
            drop.update(t.rstrip("s") for t in split_title(source))
        for spec in self.req.specs:
            drop.update(split_title(_as_str(spec.get("value"))))
        for pref in self.req.soft_prefs:
            drop.update(split_title(pref))
        out: list[str] = []
        for token in split_title(self.request):
            if token in drop or token.rstrip("s") in drop:
                continue
            if len(token) < 4 or token.isdigit() or any(ch.isdigit() for ch in token):
                continue
            if token not in out:
                out.append(token)
        return out[:4]

    def relaxation_rung(self, candidate: "Candidate", variant: dict[str, Any]) -> int:
        """0 full model+colour, 1 same model, 2 same colour, 3 neither."""

        want_model = model_norm(self.req.preferred_model or "")
        want_color = _lc(self.req.preferred_color or "")
        model_hit = False
        if want_model:
            code = model_norm(
                catalog_fact(candidate, "model") or candidate.model_guess() or ""
            )
            model_hit = bool(code) and (want_model in code or code in want_model)
            if not model_hit and want_model in model_norm(candidate.title):
                model_hit = True
        color_hit = False
        if want_color:
            color_hit = (candidate.colour_of_variant(variant) or "") == want_color
        if not want_model and not want_color:
            return 0
        if model_hit and (color_hit or not want_color):
            return 0
        if model_hit:
            return 1
        if color_hit:
            return 2
        return 3

    def state_dump(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "turn": self.env.turn,
            "remaining": self.env.steps_left,
            "pool": len(self.pool),
            "grounded": sum(1 for c in self.pool.values() if c.grounded),
            "queries": list(self.queries_run),
            "committed": list(self.committed) if self.committed else None,
            "commit_price": self.commit_price,
            "dead": sorted("%s/%s" % pair for pair in self.dead),
            "event": describe_market_move(self.env.pending_event),
            "requirements": self.req.describe(),
        }

    def should_wrap_up(self, need: int = 3) -> bool:
        return self.env.steps_left <= need or self.order_clock_hit

    def log_event(self, kind: str) -> None:
        """No-op by design.

        The event is closed by committing to the right replacement, not by
        talking about it, and each message risks a simulator failure that would
        invalidate the evaluation.
        """

        self.log.append("event noticed: %s" % (kind or "unknown"))

    def brand_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.pool.values():
            key = brand_key(candidate.brand) if candidate.brand else ""
            if key:
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def answer_open_question(self) -> None:
        """Answer a direct shopper question before the terminal action."""

        if not self.pending_question or not self.committed:
            return
        if self.env.done or self.should_wrap_up(3):
            return
        product_id, sku = self.committed
        candidate = self.pool.get(product_id)
        bits: list[str] = []
        if candidate and candidate.title:
            bits.append("Going with %s" % candidate.title[:110])
        for spec in self.req.specs:
            if candidate and spec_verdict(candidate, spec) is True:
                bits.append(
                    "its %s is %s in the product facts"
                    % (_as_str(spec.get("field")).replace("_", " "), _as_str(spec.get("value")))
                )
        if self.commit_price is not None:
            bits.append("current price %.2f %s" % (self.commit_price, self.req.currency))
        bits.append("verified in stock")
        self.env.single_call(
            "message",
            {"content": sanitised_message("; ".join(bits) + ". Placing the order now.")},
        )
        self.pending_question = None
        self.process_shopper_turns()

    def spec_scoreboard(self) -> list[str]:
        board = []
        for candidate in self.pool.values():
            if candidate.grounded:
                board.append(
                    "%s status=%d backed=%d"
                    % (candidate.product_id, spec_standing(candidate, self.req.specs),
                       fact_backed_specs(candidate, self.req.specs))
                )
        return board

    def live_stock_check(self, product_id: str, sku: str) -> dict[str, Any]:
        observation = self.env.single_call(
            "inspect_stock", {"product_id": product_id, "sku": sku}
        )
        return observation if isinstance(observation, dict) else {}

    def top_of_pool(self, limit: int = 6) -> str:
        ranked = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)[:limit]
        return " | ".join(
            "%s %.1f %s" % (c.product_id, self.cheap_rank(c), abbreviate(c.title, 44))
            for c in ranked
        ) or "empty pool"

    def _heuristic_pref(self, candidate: Candidate) -> float:
        if not self.req.soft_prefs:
            return 0.5
        blob = _lc(candidate.facts_blob())
        words: list[str] = []
        for pref in self.req.soft_prefs:
            words.extend(w for w in split_title(pref) if len(w) > 3)
        if not words:
            return 0.5
        hits = sum(1 for w in set(words) if w in blob)
        return min(1.0, 0.25 + 0.75 * hits / max(1, len(set(words))))

    def absorb_rows(self, rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            product_id = _as_str(row.get("product_id"))
            if not product_id:
                continue
            candidate = self.pool.get(product_id)
            if candidate is None:
                candidate = Candidate(product_id)
                self.pool[product_id] = candidate
            candidate.absorb_search_row(row)

    def do_searches(self, queries: list[str]) -> int:
        """Run up to max_calls searches in one turn; returns new candidates."""

        if not queries or self.env.done:
            return 0
        floor = 0 if self.committed or self.env.event_seen else self.search_reserve
        if self.search_budget - floor <= 0:
            return 0
        batch = queries[: min(2, self.env.max_calls, max(1, self.search_budget - floor))]
        self.search_budget -= len(batch)
        actions: list[tuple[str, dict[str, Any]]] = []
        for query in batch:
            actions.append(
                (
                    "search",
                    {
                        "query": query[:200],
                        "k": 10,
                        **({"in_stock": True} if self.req.require_in_stock else {}),
                    },
                )
            )
            self.queries_run.append(query)
        before = len(self.pool)
        for observation in self.env.run_turn(actions):
            self.absorb_rows(observation.get("results"))
        return len(self.pool) - before

    @property
    def seconds_in(self) -> float:
        return time.monotonic() - self.started

    def type_word_fit(self, candidate: Candidate) -> float:
        words = self.type_words()
        if not words:
            return 0.0
        blob = _lc(candidate.title + " " + candidate.facts_blob())
        return sum(1 for word in words if word in blob) / float(len(words))

    def fold_in_payload(self, payload: dict[str, Any]) -> None:
        req = self.req
        category = _as_str(payload.get("category")).strip()
        if category:
            req.category = category
        budget = _as_float(payload.get("budget"))
        if budget is not None and (req.budget is None or budget < req.budget):
            req.budget = budget
        currency = _as_str(payload.get("currency")).strip()
        if currency:
            req.currency = currency
        brand = _as_str(payload.get("brand")).strip()
        if brand:
            req.brand = brand
            req.brand_is_soft = bool(payload.get("brand_is_soft")) or self.flags.soft_brand
        specs = payload.get("specs")
        if isinstance(specs, list):
            merged = {}
            for spec in req.specs:
                merged[_lc(spec.get("field"))] = spec
            for spec in specs:
                if isinstance(spec, dict) and spec.get("field") and spec.get("value"):
                    merged[_lc(spec["field"])] = {
                        "field": _as_str(spec["field"]),
                        "value": _as_str(spec["value"]),
                    }
            req.specs = list(merged.values())
        terms = payload.get("phrase_terms")
        if isinstance(terms, list) and terms and self.flags.phrase_rule:
            req.phrase_terms = [_as_str(t) for t in terms if _as_str(t)]
        if isinstance(payload.get("device_required"), bool):
            req.device_required = payload["device_required"]
        if isinstance(payload.get("smartphone_required"), bool):
            req.smartphone_required = payload["smartphone_required"]
        model = _as_str(payload.get("preferred_model")).strip()
        if model:
            req.preferred_model = model
        color = _as_str(payload.get("preferred_color")).strip()
        if color:
            req.preferred_color = color
        prefs = payload.get("soft_prefs")
        if isinstance(prefs, list):
            for pref in prefs:
                text = _as_str(pref).strip()
                if text and text not in req.soft_prefs:
                    req.soft_prefs.append(text)
        field = _as_str(payload.get("requested_spec_field")).strip()
        if field:
            req.requested_spec_field = field
        rule = _as_str(payload.get("ranking_rule")).strip()
        if rule:
            req.ranking_rule = rule

    def commit_line(self) -> str:
        if not self.committed:
            return "nothing committed"
        product_id, sku = self.committed
        candidate = self.pool.get(product_id)
        title = abbreviate(candidate.title, 56) if candidate else product_id
        return "%s (%s) at %s" % (title, sku, self.commit_price)

    def pref_coverage(self, candidate: Candidate) -> float:
        if not self.req.soft_prefs:
            return 0.0
        return self.pref_hit_count(candidate) / float(len(self.req.soft_prefs))

    def fetch_facts(self, candidates: list[Candidate], limit: int = 5) -> None:
        """compare/view the shortlist so decisions rest on raw facts."""

        todo = [c for c in candidates if not c.grounded][:limit]
        if not todo or self.env.done:
            return
        ids = [c.product_id for c in todo]
        if len(ids) == 1:
            observation = self.env.single_call("view", {"product_id": ids[0]})
            if observation:
                todo[0].absorb_detail_row(observation)
            return
        observation = self.env.single_call("compare", {"product_ids": ids[:5]})
        rows = observation.get("compared")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = self.pool.get(_as_str(row.get("product_id")))
                if candidate is not None:
                    candidate.absorb_detail_row(row)

    def gather_options(self) -> None:
        """Broad-then-narrow retrieval with distinct reformulations."""

        if self.req.preferred_model:
            # the code usually lives in a catalog fact; the bare token reaches it
            self.chase_model_code()
        seeds = self.initial_queries()
        rounds = 0
        max_rounds = 5 if self.flags.family == "retrieval_recall" else 4
        while rounds < max_rounds and not self.env.done and not self.should_wrap_up(8):
            if rounds == 0 and self.req.category:
                self.do_filter()
            batch = seeds[:2]
            seeds = seeds[2:]
            if not batch:
                batch = self.model_suggested_queries(self.outstanding_needs())
            if not batch:
                batch = self.rule_based_queries()
            if not batch:
                break
            fresh = self.do_searches(batch)
            self.process_shopper_turns()
            rounds += 1
            shortlist = self.coarse_filter()
            distinct = {c.title[:40] for c in shortlist}
            if len(shortlist) >= 6 and len(distinct) >= 4 and rounds >= 2:
                break
            if fresh == 0 and rounds >= 2 and shortlist:
                break  # queries have stopped reaching new products
            if self.search_budget - (0 if self.committed else self.search_reserve) <= 0:
                break
            if self.env.turn >= self.commit_deadline or self.order_clock_hit:
                break

    def cheapest_eligible(self):
        return pick_cheapest(self.valid_pairs(), in_stock_only=True)

    def settle_order(self) -> None:
        """Answer, re-validate against the latest requirements, then order."""

        self.answer_open_question()
        if not self.committed:
            return
        over_budget = (
            self.req.budget is not None
            and self.commit_price is not None
            and self.commit_price > self.req.budget + 1e-9
        )
        if over_budget:
            # a late clarification tightened the budget under our commitment
            self.log.append("commitment now over budget; recovering")
            self.dead.add(self.committed)
            if self.env.steps_left >= 5:
                self.swap_commitment()
        if self.committed and self.committed not in self.dead:
            self.submit_order()

    def submit_order(self) -> bool:
        if not self.committed or self.env.done:
            return False
        if self.committed in self.dead:
            # the cart holds an item the event invalidated; do not order it
            self.committed = None
            return False
        product_id, sku = self.committed
        args: dict[str, Any] = {"product_id": product_id, "sku": sku}
        payload = self.build_justification()
        if payload:
            args["justification"] = payload
        observation = self.env.single_call("place_test_order", args)
        if observation and (
            observation.get("status") == "placed" or observation.get("order")
        ):
            return True
        # a rejected order is usually a stale price/stock read
        if observation and not self.env.done and not self.should_wrap_up(2):
            fresh = self.live_stock_check(product_id, sku)
            if fresh.get("in_stock") is True:
                observation = self.env.single_call("place_test_order", args)
                return bool(observation and observation.get("status") == "placed")
        return False

    def add_first_to_cart(self, candidate: Candidate, variant: dict[str, Any]) -> bool:
        """Verify, then place the graded first cart addition."""

        sku = _as_str(variant.get("sku"))
        if not sku or self.env.done:
            return False
        if not candidate.grounded and not self.should_wrap_up(5):
            # the verifier requires the commitment target to have been observed
            observation = self.env.single_call("view", {"product_id": candidate.product_id})
            if observation:
                candidate.absorb_detail_row(observation)
                refreshed = select_variant(candidate, self.req)
                if refreshed is not None and _as_str(refreshed.get("sku")) == sku:
                    variant = refreshed
        fresh = self.live_stock_check(candidate.product_id, sku)
        price = _as_float(fresh.get("price"))
        in_stock = fresh.get("in_stock")
        if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
            self.dead.add((candidate.product_id, sku))
            return False
        if self.req.require_in_stock and in_stock is False:
            self.dead.add((candidate.product_id, sku))
            return False
        observation = self.env.single_call(
            "add_to_cart", {"product_id": candidate.product_id, "sku": sku}
        )
        if not observation:
            return False
        self.committed = (candidate.product_id, sku)
        self.commit_price = price if price is not None else _as_float(variant.get("price"))
        self.process_shopper_turns()
        return True

    def good_enough_to_commit(self, pair: tuple[Candidate, dict[str, Any]]) -> bool:
        """Is this good enough to be the graded first cart add?

        A later correction never repairs a weak commitment, so anything short of
        the best obtainable item is worth more searching while budget remains.
        """

        candidate, variant = pair
        if self.retrieval_mode:
            return True
        if self.req.specs and fact_backed_specs(candidate, self.req.specs) < len(self.req.specs):
            return False
        if self.type_words() and self.type_word_fit(candidate) < 0.999:
            return False
        if self.req.preferred_model and self.relaxation_rung(candidate, variant) != 0:
            return False
        if self.req.brand and self.req.brand_is_soft and not brand_verified(candidate, self.req.brand):
            return False
        if (
            self.flags.preference_scored
            and self.req.soft_prefs
            and self.pref_coverage(candidate) < 0.999
        ):
            return False
        return True

    def select_after_relaxing(self) -> tuple[Candidate, dict[str, Any]] | None:
        """Last-resort widening: drop soft constraints, never hard ones."""

        saved_specs = list(self.req.specs)
        saved_brand, saved_soft = self.req.brand, self.req.brand_is_soft
        if self.req.brand_is_soft:
            self.req.brand = None
        self.req.preferred_color = None
        pick = self.select_best()
        if pick is None and saved_specs:
            # a spec the catalog never states should not block a purchase, but a
            # listing that contradicts it must still be excluded
            pairs = self.valid_pairs(allow_unproven=True)
            if pairs:
                self.grade_preferences(pairs)
                ranked = self.order_by_tradeoffs(pairs) if self.req.axes else self.order_pairs(pairs)
                pick = ranked[0]
        self.req.specs = saved_specs
        self.req.brand, self.req.brand_is_soft = saved_brand, saved_soft
        return pick

    def salvage_order(self) -> None:
        """Never end a run without a terminal action when one is possible."""

        if self.env.done or self.env.ordered:
            return
        if self.committed and self.committed not in self.dead:
            if self.submit_order():
                return
        if self.committed and self.committed in self.dead:
            if self.env.steps_left >= 4 and self.swap_commitment():
                if self.submit_order():
                    return
            self.committed = None
        pairs = self.valid_pairs()
        if not pairs:
            shortlist = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)
            if shortlist and not self.env.done:
                self.fetch_facts(shortlist[:5])
                pairs = self.valid_pairs()
        if not pairs:
            # nothing proves the spec: accept listings that do not contradict it
            pairs = self.valid_pairs(allow_unproven=True)
        if not pairs:
            # last resort: anything affordable and in stock that we have seen
            ranked = sorted(
                self.pool.values(),
                key=lambda c: (-spec_standing(c, self.req.specs), -self.cheap_rank(c)),
            )
            for candidate in ranked:
                variant = select_variant(candidate, self.req)
                if variant is not None:
                    pairs = [(candidate, variant)]
                    break
        if not pairs or self.env.done:
            return
        candidate, variant = self.order_pairs(pairs)[0]
        if self.add_first_to_cart(candidate, variant):
            if self.env.steps_left >= 2:
                state = self.check_after_commit()
                if state == "replace" and self.env.steps_left >= 4:
                    self.swap_commitment()
            self.submit_order()

    def pool_price_spread(self) -> tuple[float, float] | None:
        prices = [c.price for c in self.pool.values() if c.price is not None]
        return (min(prices), max(prices)) if prices else None

    def feature_targeted_queries(self) -> list[str]:
        """Searches aimed at the features the shopper actually asked for."""

        out: list[str] = []
        category = self.req.category or ""
        for pref in self.req.soft_prefs[:3]:
            for family in _FEATURE_SYNONYMS:
                if any(word in _lc(pref) for word in family["pref"]):
                    out.append(("%s %s" % (category, family["query"])).strip())
                    break
            else:
                out.append(("%s %s" % (category, pref)).strip())
        tried = {q.lower() for q in self.queries_run}
        seen: list[str] = []
        for query in out:
            query = re.sub(r"\s+", " ", query).strip()
            if query and query.lower() not in tried and query.lower() not in {
                s.lower() for s in seen
            }:
                seen.append(query)
        return seen[:2]

    def spec_coverage_ratio(self, candidate: "Candidate") -> float:
        if not self.req.specs:
            return 1.0
        return fact_backed_specs(candidate, self.req.specs) / float(len(self.req.specs))

    def replacement_ladder(self, exclude_product: str | None = None) -> list[tuple["Candidate", dict[str, Any]]]:
        """Eligible options ordered by least relaxation, then quality."""

        pairs = [
            pair
            for pair in self.valid_pairs()
            if (pair[0].product_id, _as_str(pair[1].get("sku"))) not in self.dead
        ]
        if not pairs:
            return []
        self.grade_preferences(pairs)
        ranked = self.order_by_tradeoffs(pairs) if self.req.axes else self.order_pairs(pairs)
        return sorted(ranked, key=lambda pair: self.relaxation_rung(*pair))

    def budget_ratio(self) -> float | None:
        if self.req.budget is None or self.commit_price is None or self.req.budget <= 0:
            return None
        return round(self.commit_price / self.req.budget, 3)

    def outstanding_needs(self) -> str:
        bits = []
        if self.req.phrase_terms:
            bits.append("title must contain the tokens %s" % self.req.phrase_terms)
        if self.req.category:
            bits.append("catalog category %s" % self.req.category)
        if self.req.budget is not None:
            bits.append("price at or under %.2f" % self.req.budget)
        for spec in self.req.specs:
            bits.append("%s = %s" % (spec.get("field"), spec.get("value")))
        if self.req.preferred_model:
            bits.append("model %s" % self.req.preferred_model)
        return "; ".join(bits) or "in-stock items matching the request"

    def order_pairs(self, pairs: list[tuple[Candidate, dict[str, Any]]]):
        req = self.req

        def key(item: tuple[Candidate, dict[str, Any]]):
            candidate, variant = item
            brand_hit = 1 if (req.brand and brand_verified(candidate, req.brand)) else 0
            model_hit = 0
            if req.preferred_model:
                code = candidate.model_guess() or ""
                model_hit = 1 if req.preferred_model.lower().replace(" ", "") in code.replace(" ", "") else 0
            color_hit = 0
            if req.preferred_color:
                color_hit = 1 if (candidate.colour_of_variant(variant) or "") == req.preferred_color.lower() else 0
            pref = candidate.pref_score if candidate.pref_score is not None else 0.4
            price = _as_float(variant.get("price")) or 0.0
            evidence = self.pref_hit_count(candidate)
            backed = fact_backed_specs(candidate, req.specs)
            descriptor = self.type_word_fit(candidate)
            endurance = playtime_hours(candidate) if self.wants_endurance else 0.0
            return (
                -(model_hit * 3 + color_hit * 2 + brand_hit * 2),
                -backed,
                -round(descriptor, 3),
                -evidence,
                -round(pref, 3),
                -endurance,
                price,
            )

        return sorted(pairs, key=key)

    def order_by_tradeoffs(self, pairs):
        """Lexicographic order over the shopper's stated tradeoff axes."""

        axes = self.req.axes or ["price"]

        def key(item):
            candidate, variant = item
            out = []
            for axis in axes:
                value = axis_measure(candidate, variant, axis)
                if value is None:
                    out.append(float("inf"))
                elif _AXIS_DIRECTION.get(axis, "max") == "min":
                    out.append(value)
                else:
                    out.append(-value)
            return tuple(out)

        return sorted(pairs, key=key)

    def initial_queries(self) -> list[str]:
        req = self.req
        seeds: list[str] = []
        if req.phrase_terms:
            seeds.append(" ".join(req.phrase_terms))
            if req.category:
                seeds.append("%s %s" % (" ".join(req.phrase_terms), req.category))
        core = " ".join(
            part
            for part in (
                req.brand or "",
                req.category or "",
                " ".join(_as_str(s.get("value")) for s in req.specs),
            )
            if part
        ).strip()
        if core:
            seeds.append(core)
        if req.preferred_model:
            seeds.append(
                " ".join(
                    p
                    for p in (req.brand or "", req.preferred_model, req.preferred_color or "")
                    if p
                )
            )
        if req.soft_prefs:
            seeds.append(
                ("%s %s" % (req.category or "", " ".join(req.soft_prefs[:3]))).strip()
            )
        if req.category:
            seeds.append(req.category)
            singular = req.category.rstrip("s")
            if singular and singular != req.category:
                seeds.append(singular)
        tried = {q.lower() for q in self.queries_run}
        out: list[str] = []
        for seed in seeds:
            seed = re.sub(r"\s+", " ", seed).strip()
            if not seed or seed.lower() in tried:
                continue
            if seed.lower() not in {q.lower() for q in out}:
                out.append(seed)
        if len(out) < 2:
            # extraction was thin (or inference is down): fall back to the text
            keywords = self.keywords_of_request()
            if keywords:
                for extra in (" ".join(keywords[:4]), " ".join(keywords[:2])):
                    extra = extra.strip()
                    if not extra or extra.lower() in tried:
                        continue
                    if extra.lower() not in {q.lower() for q in out}:
                        out.append(extra)
        return out[:4]

    def solve(self) -> list[dict[str, Any]]:
        try:
            self.read_requirements()
            self.ask_to_clarify()
            if self.req.listed:
                # the shopper named the exact variants: no search needed
                if self.handle_named_variants():
                    return self.close_dialogue()
                if self.env.ordered:
                    return self.close_dialogue()
            self.gather_options()
            pick = self.select_best()
            if pick is None:
                pick = self.select_after_relaxing()
            if pick is not None and not self.env.done:
                if not self.add_first_to_cart(*pick):
                    alternatives = [
                        p
                        for p in self.order_pairs(self.valid_pairs())
                        if (p[0].product_id, _as_str(p[1].get("sku"))) not in self.dead
                    ]
                    if alternatives:
                        self.add_first_to_cart(*alternatives[0])
            if self.committed and not self.env.done:
                if self.order_clock_hit:
                    # no time to improve on this; secure a scored outcome
                    self.log.append("clock: ordering the current commitment")
                    self.submit_order()
                    return self.close_dialogue()
                state = self.check_after_commit()
                if state != "ok":
                    kind = _lc((self.env.pending_event or {}).get("kind")) or "stockout"
                    self.log_event(kind)
                    if not self.swap_commitment():
                        self.log.append("recovery ladder exhausted")
                self.settle_order()
            if not self.env.ordered:
                self.salvage_order()
            if not self.env.ordered:
                self.guaranteed_order()
        except Exception as exc:  # a raised agent = agent_error = zero
            self.log.append("fatal: %s: %s" % (type(exc).__name__, exc))
            try:
                self.salvage_order()
                if not self.env.ordered:
                    self.guaranteed_order()
            except Exception as inner:
                self.log.append("fallback failed: %s" % type(inner).__name__)
        return self.close_dialogue()

    def valid_pairs(self, allow_unproven: bool = False) -> list[tuple[Candidate, dict[str, Any]]]:
        out = []
        for candidate in self.pool.values():
            if candidate.rejected or not candidate.grounded:
                continue
            if allow_unproven:
                # keep candidates whose spec is merely unstated, never one that
                # the catalog contradicts
                if spec_standing(candidate, self.req.specs) == 0:
                    continue
                probe = Requirements()
                probe.__dict__.update(self.req.__dict__)
                probe.specs = []
                if not passes_hard_gate(candidate, probe):
                    continue
            elif not passes_hard_gate(candidate, self.req):
                continue
            variant = select_variant(candidate, self.req)
            if variant is None:
                continue
            if (candidate.product_id, _as_str(variant.get("sku"))) in self.dead:
                continue
            out.append((candidate, variant))
        return out

    def swap_commitment(self) -> bool:
        """Least-relaxed replacement, re-grounded after the event."""

        if not self.committed or self.env.done:
            return False
        dead_product, dead_sku = self.committed
        self.replacement_round += 1
        dead_candidate = self.pool.get(dead_product)

        # refresh the dead listing: another variant of it may still qualify
        if dead_candidate is not None and not self.should_wrap_up(6):
            observation = self.env.single_call("view", {"product_id": dead_product})
            if observation:
                dead_candidate.absorb_detail_row(observation)
                self.viewed_after_event.add(dead_product)

        options = self.replacement_ladder()
        want_model = self.req.preferred_model
        if want_model and not self.should_wrap_up(8):
            has_full = any(self.relaxation_rung(*pair) == 0 for pair in options)
            if not has_full:
                self.chase_model_code()
                options = self.replacement_ladder()
        if not options and not self.should_wrap_up(8):
            self.gather_options()
            shortlist = sorted(self.coarse_filter(), key=self.cheap_rank, reverse=True)
            self.fetch_facts([c for c in shortlist if not c.grounded][:5])
            options = self.replacement_ladder()

        for candidate, variant in options:
            key = (candidate.product_id, _as_str(variant.get("sku")))
            if key in self.dead or self.env.done or self.should_wrap_up(4):
                break
            # the replacement's identity must be observed AFTER the event
            if candidate.product_id not in self.viewed_after_event:
                observation = self.env.single_call("view", {"product_id": candidate.product_id})
                if observation:
                    candidate.absorb_detail_row(observation)
                    self.viewed_after_event.add(candidate.product_id)
                if not passes_hard_gate(candidate, self.req):
                    continue
                refreshed = select_variant(candidate, self.req)
                if refreshed is not None and _as_str(refreshed.get("sku")) == key[1]:
                    variant = refreshed
            fresh = self.live_stock_check(candidate.product_id, key[1])
            price = _as_float(fresh.get("price"))
            if self.req.require_in_stock and fresh.get("in_stock") is False:
                self.dead.add(key)
                continue
            if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
                self.dead.add(key)
                continue
            if self.committed and self.committed[0] == dead_product:
                self.env.single_call(
                    "remove_from_cart", {"product_id": dead_product, "sku": dead_sku}
                )
            observation = self.env.single_call(
                "add_to_cart", {"product_id": candidate.product_id, "sku": key[1]}
            )
            if not observation:
                continue
            self.committed = key
            self.commit_price = price
            self.log.append(
                "recovered at relaxation level %d" % self.relaxation_rung(candidate, variant)
            )
            self.process_shopper_turns()
            return True
        return False

    def could_ask_budget(self) -> bool:
        if self.req.budget is not None or self.env.done:
            return False
        return self.env.has_tool("message") and not self.should_wrap_up(6)


def percent(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def as_price(value: Any) -> float | None:
    """Coerce a listing price into a positive float, or None when unusable."""

    number = _as_float(value)
    if number is None or number <= 0:
        return None
    return round(number, 2)


def variant_allowed(
    candidate: Candidate,
    variant: dict[str, Any],
    requirements: Requirements,
) -> bool:
    price = _as_float(variant.get("price"))
    if requirements.budget is not None and (price is None or price > requirements.budget + 1e-9):
        return False
    if requirements.require_in_stock and variant.get("in_stock") is not True:
        return False
    return True


def sniff_currency(text: str) -> str | None:
    low = _lc(text)
    for needle, code in (("php", "PHP"), ("peso", "PHP"), ("usd", "USD"),
                         ("dollar", "USD"), ("eur", "EUR"), ("euro", "EUR")):
        if needle in low:
            return code
    return None


def dominant_color(candidate: "Candidate") -> str | None:
    tally: dict[str, int] = {}
    for variant in candidate.variants:
        color = candidate.colour_of_variant(variant)
        if color:
            tally[color] = tally.get(color, 0) + 1
    if not tally:
        return None
    return max(tally.items(), key=lambda item: (item[1], item[0]))[0]


def axis_measure(candidate: "Candidate", variant: dict[str, Any], axis: str) -> float | None:
    """Numeric value for a tradeoff axis, variant-specific value first."""

    if axis == "price":
        return _as_float(variant.get("price"))
    scoped = _as_str(variant.get("options")) + " | " + candidate.facts_blob()
    low = _lc(scoped)
    if axis == "refresh_rate":
        rates = [float(m.group(1)) for m in re.finditer(r"(\d{2,3})\s*hz", low)]
        return max(rates) if rates else None
    if axis == "screen_size":
        sizes = [
            float(m.group(1))
            for m in re.finditer(r"(\d{1,2}(?:\.\d+)?)\s*(?:inch|inches|in\b|\")", low)
        ]
        return max(sizes) if sizes else None
    if axis == "battery":
        values = [float(m.group(1)) for m in re.finditer(r"(\d{3,6})\s*mah", low)]
        return max(values) if values else None
    if axis in ("storage", "ram"):
        values = [float(m.group(1)) for m in re.finditer(r"(\d{1,4})\s*(?:gb|tb)", low)]
        return max(values) if values else None
    return None


def title_signal(candidate: "Candidate", words: list[str]) -> float:
    if not words:
        return 0.0
    blob = _lc(candidate.title + " " + candidate.facts_blob())
    return sum(1 for word in words if word in blob) / float(len(words))


def pick_cheapest(pairs: list[tuple["Candidate", dict[str, Any]]], in_stock_only: bool = True):
    pool = [
        (candidate, variant)
        for candidate, variant in pairs
        if _as_float(variant.get("price")) is not None
        and (not in_stock_only or variant.get("in_stock") is True)
    ]
    if not pool:
        return None
    return min(pool, key=lambda pair: _as_float(pair[1].get("price")) or 0.0)


def read_budget(text: str) -> float | None:
    low = text.replace(",", "")
    patterns = (
        r"(?:at or under|at or below|no more than|not more than|at most|up to|"
        r"under|below|max(?:imum)?(?: of)?|budget(?: is)?(?: at most)?)\s*"
        r"(?:php|usd|\$)?\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*php\b",
    )
    for pattern in patterns:
        found = re.findall(pattern, low, flags=re.I)
        if found:
            return min(float(x) for x in found)
    return None


def describe_market_move(event: dict[str, Any] | None) -> str:
    if not isinstance(event, dict):
        return "no market event"
    kind = _lc(event.get("kind")) or "unknown"
    product = _as_str(event.get("product_id")) or "?"
    if kind == "price_change":
        before = _as_float(event.get("old_price"))
        after = _as_float(event.get("new_price"))
        if before is not None and after is not None:
            return "price change on %s: %.2f -> %.2f" % (product, before, after)
        return "price change on %s" % product
    if kind == "stockout":
        return "stockout on %s/%s" % (product, _as_str(event.get("sku")) or "?")
    return "%s on %s" % (kind, product)


class VariantTable:
    """Flattened (candidate, variant) rows with a few convenience filters."""

    def __init__(self) -> None:
        self.rows: list[tuple["Candidate", dict[str, Any]]] = []

    def ingest(self, candidate: "Candidate") -> int:
        added = 0
        for variant in candidate.variants:
            if isinstance(variant, dict) and variant.get("sku"):
                self.rows.append((candidate, variant))
                added += 1
        return added

    def ingest_all(self, candidates: list["Candidate"]) -> int:
        return sum(self.ingest(candidate) for candidate in candidates)

    def for_product(self, product_id: str) -> list[dict[str, Any]]:
        return [variant for candidate, variant in self.rows if candidate.product_id == product_id]

    def with_color(self, color: str) -> list[tuple["Candidate", dict[str, Any]]]:
        wanted = _lc(color)
        return [
            (candidate, variant)
            for candidate, variant in self.rows
            if (candidate.colour_of_variant(variant) or "") == wanted
        ]

    def affordable(self, ceiling: float | None) -> list[tuple["Candidate", dict[str, Any]]]:
        return [pair for pair in self.rows if within_budget(_as_float(pair[1].get("price")), ceiling)]

    def in_stock(self) -> list[tuple["Candidate", dict[str, Any]]]:
        return [pair for pair in self.rows if pair[1].get("in_stock") is True]

    def cheapest(self):
        return pick_cheapest(self.rows, in_stock_only=True)


class Env:
    """One generated-environment session: strict turn order, one turn per call."""

    def __init__(
        self,
        client: Client,
        binding: dict[str, Any],
        policy: dict[str, Any],
        problem_id: str,
        dialogue: list[dict[str, Any]],
        log: list[str],
    ) -> None:
        self.client = client
        self.binding = binding
        self.problem_id = problem_id
        self.dialogue = dialogue
        self.log = log
        self.max_steps = int(policy.get("max_steps") or 30)
        self.max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
        self.tool_names = {
            (tool.get("function") or {}).get("name")
            for tool in (policy.get("tools") or [])
        }
        self.turn = 0
        self.done = False
        self.failed = False
        self.consecutive_failures = 0
        self.cart: list[dict[str, Any]] = []
        self.pending_event: dict[str, Any] | None = None
        self.event_seen = False
        self.messages: list[str] = []
        self.ordered = False

    def cart_pairs(self) -> list[tuple[str, str]]:
        return cart_lines(self.cart)

    def tool_list(self) -> str:
        return ", ".join(sorted(name for name in self.tool_names if name)) or "all tools"

    def run_turn(self, actions: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Run one solver turn. Returns the observations, or [] on failure."""

        actions = [a for a in actions if self.has_tool(a[0])][: self.max_calls]
        if not actions or self.done or self.failed or self.steps_left <= 0:
            return []
        self.turn += 1
        group = "%s-t%d" % (self.problem_id, self.turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [
                {
                    "call_id": "%s-%d" % (group, index),
                    "action": {"name": name, "args": args},
                }
                for index, (name, args) in enumerate(actions, start=1)
            ],
        }
        result = None
        for attempt in range(3):
            try:
                result = self.client.dispatch("/environment/call", envelope)
            except Exception as exc:
                self.log.append("env call error: %s" % type(exc).__name__)
                result = None
            if isinstance(result, dict) and result.get("calls"):
                break
            time.sleep(1.0 + attempt)
        if not isinstance(result, dict) or not result.get("calls"):
            # the turn did not advance server-side; step back so the next
            # attempt replays the same turn number rather than skipping one
            self.turn -= 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.failed = True
            return []
        self.consecutive_failures = 0

        raw = [call.get("observation") or {} for call in result["calls"]]
        observations: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                observations.append({})
                continue
            # the runtime wraps every payload: {"observation": ..., "done", "error"}
            inner = item.get("observation")
            payload = inner if isinstance(inner, dict) else item
            if item.get("done") is True or payload.get("done") is True:
                self.done = True
            if item.get("error"):
                self.log.append("observation error: %s" % _as_str(item["error"])[:120])
            observations.append(payload)
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "turn %d: %s"
                % (self.turn, ", ".join(name for name, _ in actions)),
                "tool_calls": [
                    {"name": name, "args": args} for name, args in actions
                ],
                "environment_result": {
                    "turn": self.turn,
                    "calls": result.get("calls"),
                    "user_message": result.get("user_message"),
                },
            }
        )

        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if observation.get("done") is True:
                self.done = True
            if isinstance(observation.get("cart"), list):
                self.cart = observation["cart"]
            event = observation.get("market_event")
            if isinstance(event, dict) and not self.event_seen:
                self.event_seen = True
                self.pending_event = event
                self.log.append("market_event: %s" % _as_str(event.get("kind")))
            if observation.get("status") == "placed" or observation.get("order"):
                self.ordered = True
        user_message = result.get("user_message")
        if isinstance(user_message, dict):
            content = _as_str(user_message.get("content")).strip()
            if content:
                self.messages.append(content)
        if self.turn >= self.max_steps:
            self.done = True
        return observations

    def newest_message(self) -> str | None:
        return self.messages[-1] if self.messages else None

    def within_step_budget(self, need: int = 1) -> bool:
        return self.steps_left >= need and not self.done and not self.failed

    def has_tool(self, name: str) -> bool:
        return (not self.tool_names) or name in self.tool_names

    def cart_size(self) -> int:
        return len([line for line in self.cart if isinstance(line, dict)])

    def event_kind(self) -> str | None:
        event = self.pending_event or {}
        kind = _lc(event.get("kind"))
        return kind or None

    def single_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        observations = self.run_turn([(name, args)])
        return observations[0] if observations else {}

    def has_ordered(self) -> bool:
        return bool(self.ordered)

    @property
    def steps_left(self) -> int:
        return self.max_steps - self.turn

    def progress(self) -> float:
        return round(self.turn / float(self.max_steps), 3) if self.max_steps else 1.0


def variant_key(variant: dict[str, Any] | None) -> str:
    if not isinstance(variant, dict):
        return ""
    return "%s|%s" % (_as_str(variant.get("sku")), collapse_spaces(_lc(variant.get("options"))))


def median_price(prices: list[float]) -> float | None:
    clean = sorted(value for value in prices if value is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return round((clean[mid - 1] + clean[mid]) / 2.0, 2)


def _lc(value: Any) -> str:
    return _as_str(value).lower()


def intent_cues(text: str) -> list[str]:
    cues = ("gaming", "office", "travel", "running", "gym", "school", "outdoor",
            "photography", "streaming", "editing", "commute")
    low = _lc(text)
    return [cue for cue in cues if re.search(r"\b%s\b" % cue, low)]


def first_number(text: str) -> float | None:
    values = numeric_tokens(_as_str(text))
    return values[0] if values else None


def explicit_variants(text: str) -> list[tuple[str, str]]:
    """Variant keys the shopper named explicitly, e.g. ``123::456-7``."""

    out: list[tuple[str, str]] = []
    for match in _KEY_RE.finditer(text):
        pair = (match.group(1), match.group(2))
        if pair not in out:
            out.append(pair)
    return out


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def phrase_coverage(title: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    present = set(split_title(title))
    hit = sum(1 for term in terms if all(part in present for part in split_title(term)))
    return hit / float(len(terms))


class Flags:
    """Behavioural switches read out of the public rules text."""

    def __init__(self, rules: str) -> None:
        low = rules.lower()
        self.text = rules
        self.hidden_spec = (
            "tell you if you ask" in low
            or "will tell you if you ask" in low
            or "i will tell you" in low
        )
        self.clarify_prefs = (
            "clarify the user's priorities" in low
            or "clarify the user" in low
            or "ask useful questions" in low
        )
        self.phrase_rule = "match the requested phrase against the listing title" in low
        self.justification = (
            "structured justification" in low and "optional" not in low.split("structured justification")[1][:60]
        ) or "walk me through why" in low or "trustworthy reason" in low
        self.ladder = "another color of the same model" in low or "least-relaxed" in low
        self.soft_brand = (
            "soft preference" in low or "nice-to-haves" in low or "ideally" in low
        )
        self.preference_scored = (
            "soft preferences matter" in low
            or "sacrificial placeholder" in low
            or "best matches the user's intended use" in low
        )
        self.exact_variant = "exact variant" in low or "specific sku" in low
        self.ranking = (
            "tradeoff priority" in low
            or "strictly worse" in low
            or "best-ranked" in low
            or "among ties" in low
        )
        self.survey = "survey the realistic options" in low or "shortlist" in low
        self.family = self._detect_family(low)

    @staticmethod
    def _detect_family(low: str) -> str:
        signatures = {
            "retrieval_recall": (
                "match the requested phrase against the listing title",
                "complete title token",
                "survey the realistic options",
            ),
            "recovery": (
                "another color of the same model",
                "least-relaxed",
                "relaxable only after",
            ),
            "justification": (
                "structured justification",
                "walk me through",
                "trustworthy reason",
            ),
            "preference_reasoning": (
                "sacrificial placeholder",
                "soft preferences matter",
                "one test order",
            ),
            "intent_decomposition": (
                "nice-to-haves",
                "soft preference, subordinate",
                "initial choice and your later recovery are assessed separately",
            ),
            "constraint_satisfaction": (
                "every requirement must hold for the exact variant",
                "firm, non-negotiable",
                "non-negotiable requirements",
            ),
            "ranking": ("ranking rule", "rank the", "in the required order"),
        }
        best, score = "generic", 0
        for family, needles in signatures.items():
            hits = sum(1 for needle in needles if needle in low)
            if hits > score:
                best, score = family, hits
        return best

    def enabled(self) -> list[str]:
        switches = ("hidden_spec", "clarify_prefs", "phrase_rule", "justification",
                    "ladder", "soft_brand", "preference_scored", "exact_variant",
                    "ranking", "survey")
        return [name for name in switches if self.__dict__.get(name)]


class PurchaseChecklist:
    """Post-selection audit that mirrors the deterministic verifier gates."""

    def __init__(self, requirements: "Requirements") -> None:
        self.requirements = requirements
        self.issues: list[str] = []

    def run(self, candidate: "Candidate", variant: dict[str, Any], price: float | None) -> bool:
        self.issues = []
        req = self.requirements
        if not within_budget(price, req.budget):
            self.issues.append("price %s over budget %s" % (price, req.budget))
        if req.require_in_stock and variant.get("in_stock") is False:
            self.issues.append("variant sold out")
        if not passes_hard_gate(candidate, req):
            self.issues.append("hard gate failed")
        for spec in req.specs:
            if spec_verdict(candidate, spec) is False:
                self.issues.append("spec %s contradicted" % _as_str(spec.get("field")))
        if req.phrase_terms and not title_covers_phrase(candidate.title, req.phrase_terms):
            self.issues.append("title misses phrase terms")
        return not self.issues

    def clean(self) -> bool:
        return not self.issues

    def report(self) -> str:
        return "ok" if not self.issues else "; ".join(self.issues)


def clause_split(text: str) -> list[str]:
    parts = re.split(r"(?:[.;!?]\s+|,\s*(?:and|but)\s+|\s+and also\s+)", _as_str(text))
    return [part.strip(" ,.") for part in parts if part and part.strip(" ,.")]


def catalog_fact(candidate: "Candidate", field: str) -> str | None:
    """Catalog-stated value for a field, preferring facts over the title."""

    facts = candidate.facts or {}
    wanted = {
        "screen_size": ("screen_size", "screen size", "size", "display_size", "display size"),
        "ram": ("ram", "memory"),
        "refresh_rate": ("refresh_rate", "refresh rate", "refresh"),
        "resolution": ("resolution", "display", "screen_resolution"),
        "model": ("model",),
        "color": ("color", "colour"),
    }.get(field, (field,))
    for key, value in facts.items():
        if _lc(key) in wanted:
            text = _as_str(value).strip()
            if text:
                return text
    return None


class OutcomeTape:
    """Ordered log of terminal-relevant events for offline replay."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add(self, phase: str, detail: str, turn: int = 0) -> None:
        self.entries.append({"phase": phase, "detail": cap_text(detail, 120), "turn": turn})

    def by_phase(self, phase: str) -> list[str]:
        return [entry["detail"] for entry in self.entries if entry["phase"] == phase]

    def last(self) -> dict[str, Any] | None:
        return self.entries[-1] if self.entries else None

    def ordered_yet(self) -> bool:
        return any(entry["phase"] == "order" for entry in self.entries)

    def render(self, limit: int = 20) -> str:
        return " | ".join(
            "%s:%s" % (entry["phase"], entry["detail"]) for entry in self.entries[-limit:]
        )


def split_title(title: str) -> list[str]:
    """Tokenise exactly as the phrase rule specifies.

    Lowercase, then split into maximal runs of ``[a-z0-9]``.  No stemming, no
    accent folding, no spell correction - a term found only inside a longer
    token does not count.
    """

    out: list[str] = []
    run: list[str] = []
    for ch in title.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            run.append(ch)
        elif run:
            out.append("".join(run))
            run = []
    if run:
        out.append("".join(run))
    return out


def confirmed_spec_labels(candidate: "Candidate", specs: list[dict[str, str]]) -> list[str]:
    labels: list[str] = []
    for spec in specs:
        if spec_verdict(candidate, spec) is True:
            labels.append("%s=%s" % (_as_str(spec.get("field")), _as_str(spec.get("value"))))
    return labels


def stock_label(state: dict[str, Any] | None) -> str:
    if not isinstance(state, dict) or not state:
        return "unknown"
    if state.get("in_stock") is True:
        remaining = _as_float(state.get("quantity") or state.get("stock"))
        return "low" if remaining is not None and remaining <= 2 else "available"
    if state.get("in_stock") is False:
        return "sold_out"
    return "unknown"


def pixel_tiers_asserted(text: str) -> list[str]:
    """Every pixel class the text asserts, from dimensions AND from names.

    A listing that states both "720 x 1600" and "1080p" asserts two different
    classes; the rules call that unresolved, so both have to be collected.
    """

    low = _lc(text)
    found: list[str] = []
    for match in re.finditer(r"(\d{3,5})\s*[x*×]\s*(\d{3,5})", low):
        klass = pixel_tier(float(match.group(1)), float(match.group(2)))
        if klass and klass not in found:
            found.append(klass)
    stripped = re.sub(r"\d{3,5}\s*[x*×]\s*\d{3,5}", " ", low)
    # longest name first, blanking each match so "full hd" is not also read
    # as a bare "hd"
    for token, klass in sorted(_NAMED_CLASSES, key=lambda pair: -len(pair[0])):
        pattern = r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(token)
        if re.search(pattern, stripped):
            stripped = re.sub(pattern, " ", stripped)
            if klass not in found:
                found.append(klass)
    if re.search(r"(?<![a-z0-9+])hd(?![a-z0-9+])", stripped) and "hd" not in found:
        found.append("hd")
    return found


def digits_only(text: str) -> str:
    return re.sub(r"[^0-9]", "", _as_str(text))


def keyword_set(text: str) -> set[str]:
    return {token for token in split_title(text) if token not in _STOPWORDS and len(token) > 2}


def requested_pixel_tier(value: str) -> str | None:
    low = _lc(value)
    if "qhd" in low and "qhd" == low.strip():  # ambiguous case handled below
        pass
    if re.search(r"\bq\s*hd\b", low) and "quarter" not in low:
        # "qHD" lowercase means quarter HD; the rules only treat uppercase QHD
        # as 2560x1440.  Callers pass the original casing, so check that.
        if "QHD" in value:
            return "qhd"
        return None
    if "4k" in low or "uhd" in low:
        return "4k"
    if "full hd" in low or "fhd" in low or "1080" in low:
        return "fhd"
    if "1440" in low or "2560" in low:
        return "qhd"
    if "hd" in low or "720" in low:
        return "hd"
    return None


def gather_errors(observations: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for index, item in enumerate(observations or []):
        if not isinstance(item, dict):
            problems.append("#%d not an object" % index)
        elif item.get("error"):
            problems.append("#%d %s" % (index, cap_text(item["error"], 80)))
        elif not item:
            problems.append("#%d empty" % index)
    return problems


class CompletionMemo:
    """Small bounded memo so an identical prompt is not re-sent within a task."""

    def __init__(self, capacity: int = 24) -> None:
        self.capacity = max(1, capacity)
        self.store: dict[str, Any] = {}
        self.order: list[str] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def digest(system: str, user: str) -> str:
        body = _as_str(system) + "\x1f" + _as_str(user)
        return "%d:%s:%s" % (len(body), body[:40], body[-40:])

    def fetch(self, system: str, user: str) -> Any:
        key = self.digest(system, user)
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def remember(self, system: str, user: str, value: Any) -> None:
        if value is None:
            return
        key = self.digest(system, user)
        if key not in self.store:
            self.order.append(key)
        self.store[key] = value
        while len(self.order) > self.capacity:
            self.store.pop(self.order.pop(0), None)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / float(total), 3) if total else 0.0


def spec_verdict(candidate: Candidate, spec: dict[str, str]) -> bool | None:
    """True / False, or None when the evidence does not resolve the spec."""

    field = _lc(spec.get("field"))
    value = _as_str(spec.get("value"))
    canonical = catalog_fact(candidate, field)
    # the catalog fact is authoritative; a title number may describe another
    # model in a multi-model listing
    blob = canonical if canonical else candidate.facts_blob()
    low_blob = _lc(blob)

    if "screen" in field or "size" in field or "inch" in field:
        wanted = numeric_tokens(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [
            float(m.group(1))
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in\b|\")", low_blob)
        ]
        if not found:
            return None
        # the gate compares against the catalog's own value, so 6.6 does not
        # satisfy a 6.64 requirement
        return any(abs(size - target) < 0.005 for size in found)

    if "ram" in field or "memory" in field:
        wanted = numeric_tokens(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [
            float(m.group(1))
            for m in re.finditer(r"(\d+)\s*gb\s*(?:ram)?", low_blob)
        ]
        if not found:
            return None
        return any(abs(size - target) < 0.01 for size in found)

    if "refresh" in field or "hz" in field:
        wanted = numeric_tokens(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [float(m.group(1)) for m in re.finditer(r"(\d{2,3})\s*hz", low_blob)]
        if not found:
            return None
        return any(abs(rate - target) < 0.01 for rate in found)

    if "resolution" in field or "display" in field:
        want = requested_pixel_tier(value)
        if not want:
            return None
        classes = pixel_tiers_asserted(blob)
        if not classes:
            named = requested_pixel_tier(blob)
            return None if named is None else named == want
        if len(set(classes)) > 1:
            return None  # conflicting classes -> unresolved
        return classes[0] == want

    if not value:
        return None
    return value.lower() in low_blob


def variant_in_budget_count(candidate: "Candidate", ceiling: float | None) -> int:
    return sum(1 for variant in candidate.variants if within_budget(_as_float(variant.get("price")), ceiling))


def shared_keyword_count(left: str, right: str) -> int:
    return len(keyword_set(left) & keyword_set(right))


def numeric_tokens(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]


def prefer_facts(primary: dict[str, Any] | None, backup: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (backup or {}, primary or {}):
        for key, value in source.items():
            if _as_str(value).strip():
                merged[_lc(key)] = value
    return merged


def strip_bracketed(text: str) -> str:
    """Remove [SALE], (2024) and similar promo fragments from a title."""

    cleaned = re.sub(r"[\[\(\{][^\]\)\}]{0,48}[\]\)\}]", " ", _as_str(text))
    return collapse_spaces(cleaned).strip(" -|,")


def spec_standing(candidate: "Candidate", specs: list[dict[str, str]]) -> int:
    """2 every spec confirmed, 1 nothing contradicted, 0 a spec is violated."""

    if not specs:
        return 2
    verdicts = [spec_verdict(candidate, spec) for spec in specs]
    if any(v is False for v in verdicts):
        return 0
    return 2 if all(v is True for v in verdicts) else 1


def evidence_snippet(candidate: "Candidate", width: int = 120) -> str:
    return cap_text(candidate.facts_blob(), width)


def brand_key(name: str | None) -> str:
    low = re.sub(r"[^a-z0-9 ]", " ", _lc(name))
    low = collapse_spaces(low)
    table = {
        "hewlett packard": "hp", "samsung electronics": "samsung",
        "lg electronics": "lg", "apple inc": "apple", "tp link": "tplink",
    }
    return table.get(low, low)


def category_matches(candidate: Candidate, wanted: str | None) -> bool:
    if not wanted:
        return True
    want = re.sub(r"[^a-z0-9 ]", " ", wanted.lower()).strip()
    want_tokens = [t.rstrip("s") for t in want.split() if t]
    if not want_tokens:
        return True
    if not candidate.category_path:
        return True  # not yet grounded; judged again after view/compare
    path = " ".join(candidate.category_path).lower()
    path_tokens = [t.rstrip("s") for t in re.split(r"[^a-z0-9]+", path) if t]
    return all(token in path_tokens for token in want_tokens)


def parse_option_pairs(options: Any) -> list[tuple[str, str]]:
    """"Color: Black / Storage: 128GB" -> [("color","Black"),("storage","128GB")]."""

    if isinstance(options, dict):
        return [(_lc(k), _as_str(v).strip()) for k, v in options.items() if _as_str(v).strip()]
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"\s*[/|;]\s*", _as_str(options)):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key, value = _lc(key).strip(), value.strip()
        if key and value:
            out.append((key, value))
    return out


def gigabytes(text: str) -> list[int]:
    """Every storage or memory size a title states, normalised to GB."""

    sizes: list[int] = []
    for match in re.finditer(r"(\d{1,4})\s*(gb|tb)\b", _lc(text)):
        value = int(match.group(1))
        if match.group(2) == "tb":
            value *= 1024
        if value >= 2 and value not in sizes:
            sizes.append(value)
    return sorted(sizes)


def _load_json_blob(text: str) -> Any:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"```\s*$", "", body).strip()
    try:
        return json.loads(body)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = body.find(opener)
        end = body.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(body[start : end + 1])
            except Exception:
                continue
    return None


def within_budget(price: float | None, ceiling: float | None) -> bool:
    if price is None:
        return False
    if ceiling is None:
        return True
    return price <= ceiling + 1e-9


def unique_preserving(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = _lc(item).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


class LLM:
    """Model calls with a priority chain, JSON repair and a hard call budget."""

    def __init__(self, client: Client, log: list[str]) -> None:
        self.client = client
        self.log = log
        self.calls = 0
        self.misses = 0
        self.dead = False
        self.spent = 0.0
        provider = _lc(getenv("INFERENCE_PROVIDER", "chutes"))
        chain = list(_CHUTES_CHAIN)
        override = getenv("SANDBOX_MODEL")
        if override:
            chain = [override] + [m for m in chain if m != override]
        if provider == "openrouter":
            chain = [_OPENROUTER.get(m, m) for m in chain]
        self.chain = chain

    def usage_report(self) -> str:
        state = "disabled" if self.dead else "live"
        return "llm %s: %d calls, %d misses" % (state, self.calls, self.misses)

    def query_model(
        self,
        system: str,
        user: str,
        max_tokens: int = 900,
        want_json: bool = True,
    ) -> Any:
        if self.calls >= _LLM_BUDGET or self.dead or self.spent >= _LLM_TIME_BUDGET:
            return None
        self.calls += 1
        started = time.monotonic()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for model in self.chain[:2]:
            if time.monotonic() - started > _LLM_TIMEOUT * 1.5:
                break
            try:
                reply = self.client.dispatch(
                    "/inference/chat/completions",
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    },
                )
            except Exception as exc:  # network / provider hiccup
                self.log.append("llm error %s: %s" % (model, type(exc).__name__))
                continue
            if not isinstance(reply, dict):
                continue
            choices = reply.get("choices") or []
            if not choices:
                continue
            content = _as_str((choices[0].get("message") or {}).get("content"))
            if not content.strip():
                continue
            self.misses = 0
            self.spent += time.monotonic() - started
            if not want_json:
                return content.strip()
            parsed = _load_json_blob(content)
            if parsed is not None:
                return parsed
        self.spent += time.monotonic() - started
        if self.spent >= _LLM_TIME_BUDGET:
            self.dead = True
            self.log.append("model time budget spent; deterministic path only")
        self.misses += 1
        if self.misses >= 3:
            self.dead = True
            self.log.append("inference unavailable; deterministic path only")
        return None

    def is_healthy(self) -> bool:
        return not self.dead and self.misses < 2

    def budget_left(self) -> int:
        return 0 if self.dead else max(0, _LLM_BUDGET - self.calls)


def looks_like_question(text: str) -> bool:
    return "?" in _as_str(text) and not is_market_ping(text)


class Client:
    """Thin wrapper over the sandbox proxy.

    Model calls get a short timeout of their own: a hung completion is the
    difference between a scored order and a killed run.
    """

    def __init__(self) -> None:
        self._proxy = ProxyClient(timeout=110, max_retries=2)
        try:
            self._fast = ProxyClient(timeout=int(_LLM_TIMEOUT), max_retries=1)
        except Exception:
            self._fast = self._proxy

    def dispatch(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        client = self._fast if path.startswith("/inference") else self._proxy
        return client.post(path, json_data=payload)


class ShopperTranscript:
    """Every shopper turn, with light indexing for later phases."""

    def __init__(self) -> None:
        self.turns: list[tuple[int, str]] = []

    def append(self, text: str, turn: int = 0) -> None:
        content = collapse_spaces(text)
        if content:
            self.turns.append((turn, content))

    def all_text(self) -> str:
        return " ".join(text for _turn, text in self.turns)

    def open_questions(self) -> list[str]:
        return [text for _turn, text in self.turns if looks_like_question(text)]

    def latest_question(self) -> str | None:
        found = self.open_questions()
        return found[-1] if found else None

    def budgets(self) -> list[float]:
        found = []
        for _turn, text in self.turns:
            value = read_budget(text)
            if value is not None:
                found.append(value)
        return found

    def mentions(self, word: str) -> bool:
        needle = _lc(word)
        return any(needle in _lc(text) for _turn, text in self.turns)


def specs_from_text(text: str) -> list[dict[str, str]]:
    """Firm hardware specs stated anywhere in shopper text, without a model."""

    specs: list[dict[str, str]] = []
    match = re.search(r"(\d{1,2}(?:\.\d+)?)\s*-?\s*(?:inch|inches|in\b|\")", text, re.I)
    if match:
        specs.append({"field": "screen_size", "value": match.group(1) + " inch"})
    match = re.search(r"(\d{1,3})\s*gb\s*(?:of\s*)?ram", text, re.I)
    if match:
        specs.append({"field": "ram", "value": match.group(1) + "GB"})
    match = re.search(r"(\d{2,3})\s*hz", text, re.I)
    if match:
        specs.append({"field": "refresh_rate", "value": match.group(1) + "Hz"})
    for token, value in (
        ("4k", "4K"), ("uhd", "4K"), ("qhd", "QHD"), ("full hd", "Full HD"),
        ("fhd", "Full HD"), ("1080p", "Full HD"), ("1440p", "QHD"), ("720p", "HD"),
    ):
        if re.search(r"\b%s\b" % re.escape(token), text, re.I):
            if token == "qhd" and "QHD" not in text:
                continue
            specs.append({"field": "resolution", "value": value})
            break
    return specs


class DeadlineClock:
    """Wall-clock phase budgeting mirroring the forced-order deadline."""

    def __init__(self, deadline: float, force_at: float = 0.55) -> None:
        self.deadline = max(1.0, deadline)
        self.force_at = min(max(force_at, 0.1), 0.95)
        self.marks: list[tuple[str, float]] = []

    def stamp(self, label: str, moment: float) -> None:
        self.marks.append((label, moment))

    def elapsed(self, now: float) -> float:
        if not self.marks:
            return 0.0
        return max(0.0, now - self.marks[0][1])

    def fraction(self, now: float) -> float:
        return min(1.0, self.elapsed(now) / self.deadline)

    def should_force(self, now: float) -> bool:
        return self.fraction(now) >= self.force_at

    def expired(self, now: float) -> bool:
        return self.elapsed(now) >= self.deadline

    def phase_durations(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for (label, start), (_next_label, end) in zip(self.marks, self.marks[1:]):
            out.append((label, round(end - start, 3)))
        return out


def budget_bracket(price: float | None, ceiling: float | None) -> str:
    if price is None:
        return "unknown"
    if ceiling is None or ceiling <= 0:
        return "unbounded"
    ratio = price / ceiling
    if ratio > 1.0 + 1e-9:
        return "over"
    if ratio >= 0.9:
        return "near_cap"
    if ratio >= 0.6:
        return "mid"
    return "low"


def sanitised_message(text: str, limit: int = 200) -> str:
    """Short, plain-ASCII message text: the simulator is the fragile part."""

    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


def model_series(code: str | None) -> tuple[str, str]:
    key = model_norm(code or "")
    letters = "".join(ch for ch in key if ch.isalpha())
    digits = "".join(ch for ch in key if ch.isdigit())
    return letters, digits


def cart_value(cart: list[dict[str, Any]]) -> float:
    total = 0.0
    for line in cart or []:
        if not isinstance(line, dict):
            continue
        price = _as_float(line.get("price"))
        if price is None:
            continue
        total += price * (_as_float(line.get("quantity")) or 1.0)
    return round(total, 2)


def rules_section(query: str) -> str:
    cut = query.find("Task rules:")
    return query[cut:] if cut > 0 else query


def condense_facts(facts: dict[str, Any] | None, limit: int = 10, width: int = 80) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in list((facts or {}).items())[:limit]:
        text = cap_text(value, width)
        if text:
            out[_as_str(key)] = text
    return out


class ModelCallBook:
    """Ledger of inference calls: cost, latency and repair outcomes."""

    def __init__(self, budget: int, seconds: float) -> None:
        self.budget = budget
        self.seconds = seconds
        self.rows: list[dict[str, Any]] = []

    def record(self, model: str, latency: float, ok: bool, repaired: bool = False) -> None:
        self.rows.append(
            {"model": _as_str(model), "latency": round(latency, 3), "ok": bool(ok), "repaired": bool(repaired)}
        )

    def calls(self) -> int:
        return len(self.rows)

    def calls_left(self) -> int:
        return max(0, self.budget - self.calls())

    def spent(self) -> float:
        return round(sum(row["latency"] for row in self.rows), 3)

    def seconds_left(self) -> float:
        return max(0.0, self.seconds - self.spent())

    def success_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for row in self.rows if row["ok"]) / float(len(self.rows))

    def slowest(self) -> float:
        return max((row["latency"] for row in self.rows), default=0.0)

    def exhausted(self) -> bool:
        return self.calls_left() <= 0 or self.seconds_left() <= 0.0

    def summary(self) -> str:
        return "calls=%d/%d spent=%.1fs ok=%.0f%%" % (
            self.calls(), self.budget, self.spent(), self.success_rate() * 100
        )


def seems_accessory(title: str) -> bool:
    low = _lc(title)
    return any(word in low for word in _ACCESSORY)


class SearchAgenda:
    """Builds and dispenses query variants, avoiding near-duplicates."""

    ALIASES = {
        "phone": ("smartphone", "mobile phone"),
        "earbuds": ("wireless earphones", "tws"),
        "monitor": ("display", "gaming monitor"),
        "laptop": ("notebook",),
        "backpack": ("bag",),
    }

    def __init__(self, requirements: "Requirements", tried: list[str] | None = None) -> None:
        self.requirements = requirements
        self.tried = [collapse_spaces(_lc(q)) for q in (tried or [])]
        self.pending: list[str] = []

    def anchor_terms(self) -> list[str]:
        req = self.requirements
        parts = [req.brand or "", req.category or ""]
        parts.extend(_as_str(spec.get("value")) for spec in req.specs[:2])
        return [collapse_spaces(part) for part in parts if collapse_spaces(part)]

    def expand(self, query: str) -> list[str]:
        out = [query]
        for word, alts in self.ALIASES.items():
            if re.search(r"\b%s\b" % word, _lc(query)):
                out.extend(re.sub(r"\b%s\b" % word, alt, _lc(query)) for alt in alts)
        return out

    def distinct(self, queries: list[str]) -> list[str]:
        out: list[str] = []
        for query in queries:
            text = collapse_spaces(query)
            if not text or text.lower() in self.tried:
                continue
            if any(shared_keyword_count(text, kept) >= len(keyword_set(text)) for kept in out):
                continue
            out.append(text)
        return out

    def compile(self) -> list[str]:
        base = " ".join(self.anchor_terms())
        raw = self.expand(base) if base else []
        if self.requirements.preferred_model:
            raw.insert(0, self.requirements.preferred_model)
        self.pending = self.distinct(raw)
        return list(self.pending)

    def take(self, size: int = 2) -> list[str]:
        if not self.pending:
            self.compile()
        batch, self.pending = self.pending[:size], self.pending[size:]
        self.tried.extend(collapse_spaces(_lc(q)) for q in batch)
        return batch


def diagonal_inches(text: str) -> float | None:
    matches = [
        float(m.group(1))
        for m in re.finditer(r"(\d{1,2}(?:\.\d+)?)\s*(?:inch|inches|in\b|\")", _lc(text))
    ]
    plausible = [value for value in matches if 2.0 <= value <= 120.0]
    return max(plausible) if plausible else None


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

_EXTRACT_PROMPT = """Shopper request:
%s

Task rules (abridged):
%s

Extract the requirements as JSON with exactly these keys:
{"category": string|null,            // catalog category actually required
 "budget": number|null,              // maximum price, numeric only
 "currency": string|null,
 "brand": string|null,               // named brand, if any
 "brand_is_soft": boolean,           // true when the brand is only preferred
 "specs": [{"field": "screen_size|ram|resolution|refresh_rate|other",
            "value": string}],       // firm hardware specs only
 "phrase_terms": [string],           // exact words that must appear in the title
 "device_required": boolean,         // true when an actual device is wanted
 "smartphone_required": boolean,
 "preferred_model": string|null,
 "preferred_color": string|null,
 "soft_prefs": [string],             // use-case preferences, one short phrase each
 "requested_spec_field": string|null,// the spec the shopper named, for claims
 "ranking_rule": string|null}        // any stated ordering rule
Use null / [] when something is not stated. Never invent values. JSON only."""

_UPDATE_PROMPT = """Current requirements: %s

The shopper just said: "%s"

Return JSON with ONLY the fields that changed, using the same schema as before
(category, budget, currency, brand, brand_is_soft, specs, phrase_terms,
device_required, smartphone_required, preferred_model, preferred_color,
soft_prefs, requested_spec_field, ranking_rule). Requirements the shopper did
not mention must be omitted, not reset. JSON only."""

_QUERY_PROMPT = """Shopper request: %s
Requirements: %s
Queries already tried: %s
Still needed: %s

Write up to 3 NEW full-text catalog queries that would reach DIFFERENT products
than the ones already tried. Vary the wording: synonyms, product-type words,
brand or spec terms, use-case words. Keep each query under 8 words. Do not
repeat a previous query. Reply as JSON: {"queries": ["...", "..."]}"""

_SCORE_PROMPT = """Shopper request: %s
Stated soft preferences: %s
Stated ranking rule: %s

Candidates (raw catalog facts):
%s

Score each candidate from 0.0 to 1.0 on how well its RAW FACTS satisfy the
shopper's preferences and intended use. Ground the score in the facts, not in
marketing words in the title, and do not reward a low price by itself.
Reply as JSON: {"scores": [{"product_id": "...", "score": 0.0}]}"""


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Solve one generated-environment shopping task."""

    return Agent(problem_data).solve()


__all__ = ["agent_main"]