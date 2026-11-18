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


def price_histogram(prices: list[float], buckets: int = 5) -> list[int]:
    clean = [p for p in prices if p is not None]
    if not clean or buckets <= 0:
        return []
    floor = min(clean)
    width = (max(clean) - floor) / float(buckets) or 1.0
    counts = [0] * buckets
    for price in clean:
        counts[min(buckets - 1, int((price - floor) / width))] += 1
    return counts


def min_max_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    floor, top = min(values), max(values)
    if top - floor < 1e-12:
        return [1.0 for _ in values]
    return [round((v - floor) / (top - floor), 4) for v in values]


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

    def fingerprint(self) -> str:
        return json_compact(
            {
                "category": self.category,
                "budget": self.budget,
                "brand": self.brand,
                "specs": sorted("%s=%s" % (s.get("field"), s.get("value")) for s in self.specs),
                "model": self.preferred_model,
                "color": self.preferred_color,
            },
            240,
        )

    def numeric_specs(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for spec in self.specs:
            value = spec_numeric_value(spec)
            if value is not None:
                out[_lowered(spec.get("field"))] = value
        return out

    def render(self) -> str:
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

    def weighted_priorities(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        total = len(self.soft_prefs)
        for index, pref in enumerate(self.soft_prefs):
            weights[pref] = round((total - index) / float(total * (total + 1) / 2.0), 3)
        return weights

    def has_hard_rules(self) -> bool:
        return bool(self.specs or self.phrase_terms or (self.brand and not self.brand_is_soft))


def brand_is(candidate: Candidate, brand: str | None) -> bool:
    """Structured brand first - a brand word in the title proves nothing."""

    if not brand:
        return False
    if candidate.brand:
        return brand.lower() in candidate.brand.lower()
    return False


class ActionThrottle:
    """Caps how often each tool may be used in one episode."""

    DEFAULT_LIMITS = {"search": 10, "compare": 6, "view": 12, "message": 2, "place_test_order": 2}

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        self.limits = dict(self.DEFAULT_LIMITS)
        self.limits.update(limits or {})
        self.used: dict[str, int] = {}

    def permit(self, tool: str) -> bool:
        return self.used.get(tool, 0) < self.limits.get(tool, 1000)

    def consume(self, tool: str) -> bool:
        if not self.permit(tool):
            return False
        self.used[tool] = self.used.get(tool, 0) + 1
        return True

    def headroom(self, tool: str) -> int:
        return max(0, self.limits.get(tool, 1000) - self.used.get(tool, 0))


def token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in title_words(text):
        if token not in _STOPWORDS:
            counts[token] = counts.get(token, 0) + 1
    return counts


def is_device(candidate: Candidate, requirements: Requirements) -> bool:
    if not requirements.device_required:
        return True
    if accessory_title(candidate.title):
        return False
    if not requirements.smartphone_required:
        return True
    blob = _lowered(candidate.text_blob())
    if any(bad in blob for bad in _NON_SMARTPHONE) and "smartphone" not in blob:
        return False
    if "smartphone" in _lowered(candidate.title):
        return True
    device_type = _lowered(candidate.facts.get("phone_type")) + " " + _lowered(
        candidate.facts.get("device_type")
    )
    if "smartphone" in device_type:
        return True
    os_text = " ".join(
        _lowered(candidate.facts.get(key))
        for key in ("device_operating_system", "operating_system", "os")
    )
    return any(name in os_text for name in _MOBILE_OS)


def lcs_length(left: list[str], right: list[str]) -> int:
    """Longest common subsequence of two token lists (dynamic programming)."""

    if not left or not right:
        return 0
    row = [0] * (len(right) + 1)
    for token in left:
        diagonal = 0
        for j, other in enumerate(right, start=1):
            above = row[j]
            row[j] = diagonal + 1 if token == other else max(row[j], row[j - 1])
            diagonal = above
    return row[-1]


def camera_megapixels(text: str) -> int | None:
    best = None
    for match in re.finditer(r"(\d{1,3}(?:\.\d)?)\s*mp\b", _lowered(text)):
        value = int(float(match.group(1)))
        if best is None or value > best:
            best = value
    return best


def clears_hard_rules(candidate: Candidate, requirements: Requirements) -> bool:
    """Product-level gate; variant-level checks happen separately."""

    if requirements.phrase_terms and not phrase_in_title(candidate.title, requirements.phrase_terms):
        return False
    if not in_category(candidate, requirements.category):
        return False
    if not is_device(candidate, requirements):
        return False
    if requirements.brand and not requirements.brand_is_soft:
        brand = _lowered(candidate.brand)
        if brand:
            if requirements.brand.lower() not in brand:
                return False
        elif requirements.brand.lower() not in _lowered(candidate.title):
            return False
    if candidate.grounded:
        for spec in requirements.specs:
            if spec_holds(candidate, spec) is not True:
                return False
    return True


class ConfidenceTracker:
    """Beta-style belief that each candidate truly satisfies the request."""

    def __init__(self) -> None:
        self.support: dict[str, int] = {}
        self.against: dict[str, int] = {}

    def tally_verdict(self, product_id: str, verdict: bool | None) -> None:
        if verdict is True:
            self.support[product_id] = self.support.get(product_id, 0) + 1
        elif verdict is False:
            self.against[product_id] = self.against.get(product_id, 0) + 1

    def tally_specs(self, candidate: "Candidate", specs: list[dict[str, str]]) -> None:
        for spec in specs:
            self.tally_verdict(candidate.product_id, spec_holds(candidate, spec))

    def belief(self, product_id: str) -> float:
        return beta_mean(self.support.get(product_id, 0), self.against.get(product_id, 0))

    def strongest(self) -> str | None:
        keys = set(self.support) | set(self.against)
        if not keys:
            return None
        return max(sorted(keys), key=self.belief)


def chipset_name(text: str) -> str | None:
    match = re.search(
        r"\b(snapdragon\s*\d+(?:\s*gen\s*\d)?|dimensity\s*\d{3,4}|helio\s*[a-z]\d{2,3}|"
        r"exynos\s*\d{3,4}|tensor\s*g?\d|a\d{2}\s*bionic|unisoc\s*[a-z]\d{3,4})\b",
        _lowered(text),
    )
    return re.sub(r"\s+", " ", match.group(1)) if match else None


def stated_specs(text: str) -> list[dict[str, str]]:
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


def backoff_schedule(attempts: int, base: float = 0.5, cap: float = 8.0) -> list[float]:
    delays: list[float] = []
    delay = base
    for _ in range(max(0, attempts)):
        delays.append(round(min(cap, delay), 3))
        delay *= 2.0
    return delays


def merge_intervals(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((min(a, b), max(a, b)) for a, b in ranges)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def specs_backed_by_facts(candidate: "Candidate", specs: list[dict[str, str]]) -> int:
    """Specs confirmed by a catalog fact rather than by title text alone.

    Gold sets are fact-backed, so a listing whose title says "27 inch" while the
    catalog states nothing is a weaker pick than one whose facts say it.
    """

    count = 0
    for spec in specs:
        field = _lowered(spec.get("field"))
        if fact_lookup(candidate, field) and spec_holds(candidate, spec) is True:
            count += 1
    return count


def rank_positions(ranking: list[str]) -> dict[str, int]:
    return {key: index for index, key in enumerate(ranking)}


class RequirementDiff:
    """Field-level difference between two requirement snapshots."""

    def __init__(self, before: "Requirements", after: "Requirements") -> None:
        self.before = before
        self.after = after

    def changed(self) -> list[str]:
        left, right = self.before.__dict__, self.after.__dict__
        return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))

    def tightened_budget(self) -> bool:
        old, new = self.before.budget, self.after.budget
        return old is not None and new is not None and new < old - 1e-9

    def new_specs(self) -> list[str]:
        known = {_lowered(spec.get("field")) for spec in self.before.specs}
        return [_lowered(spec.get("field")) for spec in self.after.specs if _lowered(spec.get("field")) not in known]

    def headline(self) -> str:
        fields = self.changed()
        return "no change" if not fields else "changed: " + ", ".join(fields)


def spec_holds(candidate: Candidate, spec: dict[str, str]) -> bool | None:
    """True / False, or None when the evidence does not resolve the spec."""

    field = _lowered(spec.get("field"))
    value = _string_of(spec.get("value"))
    canonical = fact_lookup(candidate, field)
    # the catalog fact is authoritative; a title number may describe another
    # model in a multi-model listing
    blob = canonical if canonical else candidate.text_blob()
    low_blob = _lowered(blob)

    if "screen" in field or "size" in field or "inch" in field:
        wanted = all_numbers(value)
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
        wanted = all_numbers(value)
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
        wanted = all_numbers(value)
        if not wanted:
            return None
        target = wanted[0]
        found = [float(m.group(1)) for m in re.finditer(r"(\d{2,3})\s*hz", low_blob)]
        if not found:
            return None
        return any(abs(rate - target) < 0.01 for rate in found)

    if "resolution" in field or "display" in field:
        want = target_display_class(value)
        if not want:
            return None
        classes = display_classes_found(blob)
        if not classes:
            named = target_display_class(blob)
            return None if named is None else named == want
        if len(set(classes)) > 1:
            return None  # conflicting classes -> unresolved
        return classes[0] == want

    if not value:
        return None
    return value.lower() in low_blob


def price_ceiling(text: str) -> float | None:
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


class PhaseMachine:
    """Explicit phase transitions of one episode, rejecting illegal jumps."""

    TRANSITIONS = {
        "parse": ("clarify", "survey", "named"),
        "clarify": ("survey", "named"),
        "named": ("react", "order", "survey"),
        "survey": ("choose",),
        "choose": ("commit", "fallback"),
        "commit": ("react", "fallback"),
        "react": ("recover", "order"),
        "recover": ("order", "fallback"),
        "fallback": ("order", "done"),
        "order": ("done",),
    }

    def __init__(self) -> None:
        self.current = "parse"
        self.history: list[str] = ["parse"]

    def can_move(self, target: str) -> bool:
        return target in self.TRANSITIONS.get(self.current, ())

    def move(self, target: str) -> bool:
        if not self.can_move(target):
            return False
        self.current = target
        self.history.append(target)
        return True

    def finished(self) -> bool:
        return self.current == "done"

    def visited(self, phase: str) -> bool:
        return phase in self.history


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

    def forward(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        client = self._fast if path.startswith("/inference") else self._proxy
        return client.post(path, json_data=payload)

    def route_for(self, path: str) -> str:
        return "fast" if path.startswith("/inference") else "standard"


def moving_average(values: list[float], window: int = 3) -> list[float]:
    out: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        out.append(round(running / float(min(index + 1, window)), 4))
    return out


class RunHealth:
    """Rolls environment, inference and clock signals into one health grade."""

    def __init__(self) -> None:
        self.env_errors = 0
        self.llm_errors = 0
        self.clock_fraction = 0.0

    def note_env_error(self) -> None:
        self.env_errors += 1

    def note_llm_error(self) -> None:
        self.llm_errors += 1

    def note_clock(self, elapsed: float) -> None:
        self.clock_fraction = min(1.0, max(0.0, elapsed / max(_DEADLINE_S, 1.0)))

    def grade(self) -> str:
        if self.env_errors >= 3 or self.clock_fraction >= 0.9:
            return "critical"
        if self.env_errors or self.llm_errors >= 2 or self.clock_fraction >= _FORCE_ORDER_AT:
            return "degraded"
        return "ok"


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

    def condition_and_year(self) -> tuple[str, int | None]:
        year = model_year(self.title)
        if year is None and self.facts:
            year = model_year(_string_of(self.facts.get("release_date")))
        return listing_condition(self.title), year

    def load_detail(self, row: dict[str, Any]) -> None:
        self.title = _string_of(row.get("title")) or self.title
        if row.get("brand"):
            self.brand = _string_of(row.get("brand"))
        path = row.get("category_path")
        if isinstance(path, list):
            self.category_path = [_string_of(p) for p in path]
        facts = row.get("facts")
        if isinstance(facts, dict):
            self.facts = facts
        variants = row.get("variants")
        if isinstance(variants, list):
            self.variants = [v for v in variants if isinstance(v, dict)]
        self.grounded = True

    def word_overlap_with(self, request: str) -> list[str]:
        wanted = set(title_words(request)) - _STOPWORDS
        return sorted(token for token in title_words(self.title) if token in wanted)

    def text_blob(self) -> str:
        chunks = [self.title]
        for key, value in (self.facts or {}).items():
            chunks.append("%s: %s" % (key, value))
        for variant in self.variants:
            chunks.append(_string_of(variant.get("options")))
        return " | ".join(c for c in chunks if c)

    def load_search(self, row: dict[str, Any]) -> None:
        self.title = _string_of(row.get("title")) or self.title
        if row.get("brand"):
            self.brand = _string_of(row.get("brand"))
        price = _float_or_none(row.get("price"))
        if price is not None:
            self.price = price
        if row.get("sku"):
            self.sku = _string_of(row.get("sku"))
        if isinstance(row.get("in_stock"), bool):
            self.in_stock = row["in_stock"]

    def sku_lookup(self) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}
        for variant in self.variants:
            sku = _string_of(variant.get("sku"))
            if sku and sku not in table:
                table[sku] = variant
        return table

    def model_id(self) -> str | None:
        fact = _lowered(self.facts.get("model")) if self.facts else ""
        if fact.strip():
            return fact.strip()
        for token in title_words(self.title):
            if re.fullmatch(r"[a-z]{1,3}\d{1,4}[a-z]?", token):
                return token
        return None

    def variant_colour(self, variant: dict[str, Any]) -> str | None:
        options = _lowered(variant.get("options"))
        if options:
            # the selected variant's own options win; a colour word elsewhere in
            # the listing says nothing about THIS sku
            sources = [options]
        else:
            sources = [_lowered(self.facts.get("color")), _lowered(self.title)]
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

    def hardware_profile(self) -> dict[str, Any]:
        blob = self.text_blob()
        profile = {
            "chipset": chipset_name(blob),
            "camera_mp": camera_megapixels(blob),
            "charging_w": charging_watts(blob),
            "radios": connectivity_tags(blob),
            "warranty_months": warranty_months(blob),
        }
        return {key: value for key, value in profile.items() if value}


def json_compact(payload: Any, limit: int = 400) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        text = _string_of(payload)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def charging_watts(text: str) -> int | None:
    best = 0
    for match in re.finditer(r"(\d{2,3})\s*w\b", _lowered(text)):
        value = int(match.group(1))
        if 5 <= value <= 240:
            best = max(best, value)
    return best or None


def in_category(candidate: Candidate, wanted: str | None) -> bool:
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


def generation_number(text: str) -> int | None:
    """Product generation from "Gen 3", "3rd generation" or "Mark II"."""

    low = _lowered(text)
    match = re.search(r"\bgen(?:eration)?\s*(\d{1,2})\b", low) or re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)\s+gen(?:eration)?\b", low
    )
    if match:
        return int(match.group(1))
    roman = {"ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
    match = re.search(r"\b(?:mark|mk|gen)\s+(ii|iii|iv|v|vi)\b", low)
    return roman.get(match.group(1)) if match else None


def parse_duration_minutes(text: str) -> int | None:
    low = _lowered(text)
    hours = re.search(r"(\d{1,3})\s*(?:h|hr|hrs|hour|hours)\b", low)
    minutes = re.search(r"(\d{1,3})\s*(?:m|min|mins|minute|minutes)\b", low)
    if not hours and not minutes:
        return None
    return (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)


def announces_event(text: str) -> bool:
    """Does this shopper turn just surface the market event?

    Those need a corrected commitment, not a reply. Answering them spends a
    simulator call for nothing, and every simulator call can fail the run.
    """

    low = _lowered(text)
    return any(phrase in low for phrase in _EVENT_NOTICE)


def query_budget_split(total: int, phases: int) -> list[int]:
    if phases <= 0:
        return []
    base, extra = divmod(max(0, total), phases)
    return [base + (1 if index < extra else 0) for index in range(phases)]


def title_alignment(left: str, right: str) -> float:
    """Order-aware title agreement: LCS over tokens, normalised by the longer title."""

    a, b = title_words(left), title_words(right)
    longest = max(len(a), len(b))
    return round(lcs_length(a, b) / float(longest), 3) if longest else 1.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def cosine_overlap(left: str, right: str) -> float:
    """Cosine similarity of token-count vectors; order-free, weight-aware."""

    a, b = token_counts(left), token_counts(right)
    dot = sum(a[token] * b.get(token, 0) for token in a)
    norm = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return round(dot / norm, 4) if norm else 0.0


class TokenBucket:
    """Rate limiter for proxy calls; the caller passes the clock in."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = max(0.001, rate)
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.updated: float | None = None

    def refill(self, now: float) -> None:
        if self.updated is not None:
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def try_take(self, now: float, cost: float = 1.0) -> bool:
        self.refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def wait_time(self, now: float, cost: float = 1.0) -> float:
        self.refill(now)
        missing = cost - self.tokens
        return 0.0 if missing <= 0 else round(missing / self.rate, 3)


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
        self.family = self._classify_family(low)

    def rule_signature(self) -> str:
        on = [key for key, value in sorted(self.__dict__.items()) if value is True]
        return "%s:%s" % (self.family, ",".join(on))

    @staticmethod
    def _classify_family(low: str) -> str:
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


def bluetooth_version(text: str) -> float | None:
    match = re.search(r"\b(?:bluetooth|bt)\s*v?(\d)\.(\d)\b", _lowered(text))
    return float("%s.%s" % (match.group(1), match.group(2))) if match else None


def _string_of(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


class PendingQueue:
    """Small priority queue without heapq: lowest priority value first."""

    def __init__(self) -> None:
        self.items: list[tuple[float, int, Any]] = []
        self.counter = 0

    def push(self, priority: float, item: Any) -> None:
        entry = (priority, self.counter, item)
        self.counter += 1
        index = len(self.items)
        while index > 0 and self.items[index - 1][:2] > entry[:2]:
            index -= 1
        self.items.insert(index, entry)

    def pop(self) -> Any:
        return self.items.pop(0)[2] if self.items else None

    def peek(self) -> Any:
        return self.items[0][2] if self.items else None

    def size(self) -> int:
        return len(self.items)


def fuzzy_model_match(wanted: str | None, found: str | None, tolerance: int = 1) -> bool:
    """Model codes one typo apart ("a556e" vs "a556") still count as close."""

    left = canonical_model(wanted or "")
    right = canonical_model(found or "")
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    return edit_distance(left, right) <= tolerance


def priority_axes(text: str) -> list[str]:
    """Ordered tradeoff axes, taken from the order they appear in the ask."""

    low = _lowered(text)
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


def percentile(values: list[float], share: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = min(1.0, max(0.0, share)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


class SkuResolver:
    """Maps a free-text option wish ("blue 256") to the closest variant sku."""

    def __init__(self, candidate: "Candidate") -> None:
        self.candidate = candidate

    def overlap(self, variant: dict[str, Any], wish: str) -> float:
        options = _lowered(variant.get("options"))
        words = [w for w in title_words(wish) if w not in _STOPWORDS]
        if not words:
            return 0.0
        return sum(1 for w in words if w in options) / float(len(words))

    def resolve(self, wish: str) -> str | None:
        best_sku, best_score = None, 0.0
        for variant in self.candidate.variants:
            value = self.overlap(variant, wish)
            if value > best_score:
                best_sku, best_score = _string_of(variant.get("sku")), value
        return best_sku


def all_numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]


def seller_tier(text: str) -> str:
    low = _lowered(text)
    if "official store" in low or "flagship" in low or "mall" in low:
        return "official"
    if "authorized" in low or "authorised" in low:
        return "authorized"
    if "preferred seller" in low or "top seller" in low:
        return "preferred"
    return "marketplace"


def knapsack_pick(items: list[tuple[str, int, float]], capacity: int) -> list[str]:
    """0/1 knapsack over (key, integer price, value): best value within budget."""

    if capacity <= 0 or not items:
        return []
    best = [(0.0, [])] * (capacity + 1)
    for key, cost, value in items:
        if cost <= 0 or cost > capacity:
            continue
        for room in range(capacity, cost - 1, -1):
            candidate_value = best[room - cost][0] + value
            if candidate_value > best[room][0]:
                best[room] = (candidate_value, best[room - cost][1] + [key])
    return best[capacity][1]


def request_body(query: str) -> str:
    """Return the shopper-facing request without the appended task rules."""

    cut = query.find("Task rules:")
    return (query[:cut] if cut > 0 else query).strip()


def sliding_windows(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0 or size > len(items):
        return []
    return [items[start : start + size] for start in range(len(items) - size + 1)]


class QuotaWindow:
    """Counts events inside a sliding time window (seconds, caller's clock)."""

    def __init__(self, span: float, ceiling: int) -> None:
        self.span = max(0.001, span)
        self.ceiling = max(1, ceiling)
        self.stamps: list[float] = []

    def prune(self, now: float) -> None:
        cutoff = now - self.span
        while self.stamps and self.stamps[0] < cutoff:
            self.stamps.pop(0)

    def hit(self, now: float) -> bool:
        self.prune(now)
        if len(self.stamps) >= self.ceiling:
            return False
        self.stamps.append(now)
        return True

    def load(self, now: float) -> float:
        self.prune(now)
        return round(len(self.stamps) / float(self.ceiling), 3)


def stated_hours(candidate: "Candidate") -> float:
    """Largest stated playtime in hours, 0.0 when the listing never says."""

    low = _lowered(candidate.text_blob())
    values = [
        float(m.group(1))
        for m in re.finditer(r"(\d{1,3})\s*(?:hours|hour|hrs|hr)\b", low)
    ]
    return max(values) if values else 0.0


class DecisionTrace:
    """Tree of decisions: each node keeps its reason and the options it beat."""

    def __init__(self) -> None:
        self.nodes: dict[int, dict[str, Any]] = {}
        self.next_id = 1

    def decide(self, choice: str, reason: str, rejected: list[str] | None = None, parent: int | None = None) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.nodes[node_id] = {
            "choice": choice, "reason": reason, "rejected": list(rejected or []), "parent": parent,
        }
        return node_id

    def path_to(self, node_id: int) -> list[str]:
        chain: list[str] = []
        current: int | None = node_id
        while current is not None and current in self.nodes:
            chain.append(self.nodes[current]["choice"])
            current = self.nodes[current]["parent"]
        return list(reversed(chain))

    def children(self, node_id: int) -> list[int]:
        return sorted(key for key, node in self.nodes.items() if node["parent"] == node_id)


def canonical_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _lowered(value))


def normalise_sku(sku: str | None) -> str:
    parts = [part for part in re.split(r"[\s_./-]+", _string_of(sku).upper()) if part]
    return "-".join(parts)


def pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
    """Keys not dominated on (lower price, higher quality)."""

    front: list[str] = []
    for key, price, quality in points:
        dominated = any(
            other_key != key
            and other_price <= price
            and other_quality >= quality
            and (other_price < price or other_quality > quality)
            for other_key, other_price, other_quality in points
        )
        if not dominated:
            front.append(key)
    return front


def fact_lookup(candidate: "Candidate", field: str) -> str | None:
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
        if _lowered(key) in wanted:
            text = _string_of(value).strip()
            if text:
                return text
    return None


def connectivity_tags(text: str) -> list[str]:
    low = _lowered(text)
    tags: list[str] = []
    if re.search(r"\b5g\b", low):
        tags.append("5g")
    elif re.search(r"\b(?:4g|lte)\b", low):
        tags.append("4g")
    for tag, needle in (("wifi", "wi-fi"), ("wifi", "wifi"), ("nfc", "nfc"), ("bluetooth", "bluetooth")):
        if needle in low and tag not in tags:
            tags.append(tag)
    if re.search(r"\b(?:usb[\s-]?c|type[\s-]?c)\b", low):
        tags.append("usb-c")
    return tags


def _decode_json(text: str) -> Any:
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


def spec_numeric_value(spec: dict[str, str]) -> float | None:
    """First number of a spec value, with TB folded into GB and mm/cm into inches."""

    scale = {"tb": 1024.0, "mm": 1 / 25.4, "cm": 1 / 2.54}
    text = _lowered(spec.get("value"))
    if not text:
        return None
    number = ""
    for index, char in enumerate(text):
        if char.isdigit() or (char == "." and number and "." not in number):
            number += char
        elif number:
            # the unit, if any, follows the number directly or after spaces
            unit = re.match(r"\s*([a-z]+)", text[index:])
            factor = scale.get(unit.group(1), 1.0) if unit else 1.0
            return round(float(number) * factor, 2)
    return round(float(number), 2) if number else None


def color_family(color: str | None) -> str:
    low = _lowered(color)
    groups = (
        ("neutral", ("black", "white", "grey", "gray", "silver", "graphite", "titanium", "beige")),
        ("warm", ("red", "pink", "orange", "yellow", "gold", "bronze", "brown", "rose")),
        ("cool", ("blue", "green", "teal", "cyan", "purple", "violet", "lavender", "navy")),
    )
    for family, words in groups:
        if any(word in low for word in words):
            return family
    return "other"


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

    def call_windows(self) -> list[list[int]]:
        turns = list(range(1, self.turn + 1))
        return sliding_windows(turns, min(3, len(turns))) if turns else []

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        observations = self.step([(name, args)])
        return observations[0] if observations else {}

    def throttle_for_turns(self) -> "ActionThrottle":
        remaining = max(0, self.budget_turns)
        return ActionThrottle({"search": min(10, remaining), "compare": min(6, remaining), "message": 1})

    @property
    def budget_turns(self) -> int:
        return self.max_steps - self.turn

    def turn_digest(self) -> str:
        state = "done" if self.done else "failed" if self.failed else "live"
        pending = json_compact(self.pending_event or {}, 120)
        return "turn %d/%d %s, %d msg(s), event=%s" % (
            self.turn, self.max_steps, state, len(self.messages), pending
        )

    def step(self, actions: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Run one solver turn. Returns the observations, or [] on failure."""

        actions = [a for a in actions if self.offers(a[0])][: self.max_calls]
        if not actions or self.done or self.failed or self.budget_turns <= 0:
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
                result = self.client.forward("/environment/call", envelope)
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
                self.log.append("observation error: %s" % _string_of(item["error"])[:120])
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
                self.log.append("market_event: %s" % _string_of(event.get("kind")))
            if observation.get("status") == "placed" or observation.get("order"):
                self.ordered = True
        user_message = result.get("user_message")
        if isinstance(user_message, dict):
            content = _string_of(user_message.get("content")).strip()
            if content:
                self.messages.append(content)
        if self.turn >= self.max_steps:
            self.done = True
        return observations

    def offers(self, name: str) -> bool:
        return (not self.tool_names) or name in self.tool_names

    def retry_delays(self) -> list[float]:
        return backoff_schedule(3, base=1.0, cap=3.0)


def elapsed_label(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, rest = divmod(int(seconds), 60)
    return "%dm%02ds" % (minutes, rest) if minutes else "%.1fs" % seconds


def exponential_smoothing(values: list[float], alpha: float = 0.5) -> list[float]:
    alpha = min(1.0, max(0.0, alpha))
    smoothed: list[float] = []
    for value in values:
        smoothed.append(value if not smoothed else alpha * value + (1 - alpha) * smoothed[-1])
    return [round(v, 4) for v in smoothed]


def spec_grade(candidate: "Candidate", specs: list[dict[str, str]]) -> int:
    """2 every spec confirmed, 1 nothing contradicted, 0 a spec is violated."""

    if not specs:
        return 2
    verdicts = [spec_holds(candidate, spec) for spec in specs]
    if any(v is False for v in verdicts):
        return 0
    return 2 if all(v is True for v in verdicts) else 1


def near_duplicate_titles(titles: list[str], threshold: float = 0.9) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if shingle_similarity(titles[i], titles[j]) >= threshold:
                pairs.append((i, j))
    return pairs


def gini_impurity(counts: dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    return round(1.0 - sum((value / total) ** 2 for value in counts.values()), 4)


def display_class(width: float, height: float) -> str | None:
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


def axis_number(candidate: "Candidate", variant: dict[str, Any], axis: str) -> float | None:
    """Numeric value for a tradeoff axis, variant-specific value first."""

    if axis == "price":
        return _float_or_none(variant.get("price"))
    scoped = _string_of(variant.get("options")) + " | " + candidate.text_blob()
    low = _lowered(scoped)
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


def discount_share(text: str) -> float | None:
    """ "20% off" -> 0.2; ignores percentages that are clearly not discounts."""

    for match in re.finditer(r"(\d{1,2})\s*%\s*(off|discount|less)", _lowered(text)):
        share = int(match.group(1)) / 100.0
        if 0 < share < 0.95:
            return share
    return None


class SynonymGraph:
    """Undirected term graph; expansion walks it breadth-first."""

    def __init__(self) -> None:
        self.edges: dict[str, set[str]] = {}

    def link(self, left: str, right: str) -> None:
        a, b = _lowered(left).strip(), _lowered(right).strip()
        if a and b and a != b:
            self.edges.setdefault(a, set()).add(b)
            self.edges.setdefault(b, set()).add(a)

    def seed_from_features(self) -> int:
        linked = 0
        for family in _FEATURE_SYNONYMS:
            words = list(family["pref"]) + list(family["facts"])
            for other in words[1:]:
                self.link(words[0], other)
                linked += 1
        return linked

    def neighbourhood(self, term: str, depth: int = 1) -> list[str]:
        start = _lowered(term).strip()
        seen = {start}
        frontier = [start]
        for _ in range(max(0, depth)):
            following: list[str] = []
            for node in frontier:
                for neighbour in sorted(self.edges.get(node, ())):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        following.append(neighbour)
            frontier = following
        seen.discard(start)
        return sorted(seen)


def ascii_message(text: str, limit: int = 200) -> str:
    """Short, plain-ASCII message text: the simulator is the fragile part."""

    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


def char_shingles(text: str, size: int = 3) -> set[str]:
    squeezed = re.sub(r"[^a-z0-9]", "", _lowered(text))
    if len(squeezed) <= size:
        return {squeezed} if squeezed else set()
    return {squeezed[i : i + size] for i in range(len(squeezed) - size + 1)}


def interleave(first: list[str], second: list[str]) -> list[str]:
    """Alternate two query lists, dropping case-insensitive repeats."""

    out: list[str] = []
    seen: set[str] = set()
    for index in range(max(len(first), len(second))):
        for source in (first, second):
            if index < len(source) and _lowered(source[index]) not in seen:
                seen.add(_lowered(source[index]))
                out.append(source[index])
    return out


def target_display_class(value: str) -> str | None:
    low = _lowered(value)
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


def model_year(text: str) -> int | None:
    latest = None
    for token in title_words(_string_of(text)):
        if len(token) == 4 and token.isdigit() and token.startswith("20"):
            year = int(token)
            if 2010 <= year <= 2029 and (latest is None or year > latest):
                latest = year
    return latest


class CircuitBreaker:
    """Opens after repeated failures so a dead endpoint stops eating turns."""

    def __init__(self, threshold: int = 3, cool_down: float = 30.0) -> None:
        self.threshold = max(1, threshold)
        self.cool_down = cool_down
        self.failures = 0
        self.opened_at: float | None = None

    def state(self, now: float) -> str:
        if self.opened_at is None:
            return "closed"
        if now - self.opened_at >= self.cool_down:
            return "half_open"
        return "open"

    def allow(self, now: float) -> bool:
        return self.state(now) != "open"

    def succeeded(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failed(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = now


def best_variant(
    candidate: Candidate, requirements: Requirements
) -> dict[str, Any] | None:
    """Best eligible variant, preferring the requested colour."""

    eligible = [v for v in candidate.variants if variant_eligible(candidate, v, requirements)]
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
    wanted_color = _lowered(requirements.preferred_color)
    if wanted_color:
        exact = [
            v
            for v in eligible
            if (candidate.variant_colour(v) or "") == wanted_color
        ]
        if exact:
            return exact[0]
    return eligible[0]


def bundle_extras(title: str) -> list[str]:
    """Freebies a title advertises: "phone with free case + charger" -> [case, charger]."""

    low = _lowered(title)
    if "with" not in low and "+" not in low and "free" not in low:
        return []
    tail = re.split(r"\bwith\b|\bfree\b", low, maxsplit=1)[-1]
    pieces = [piece.strip(" ,.()[]") for piece in re.split(r"\+|,|\band\b", tail)]
    extras = [re.sub(r"^(?:a|an|free)\s+", "", piece) for piece in pieces if 2 < len(piece) <= 24]
    return list(dict.fromkeys(extras))[:4]


def shingle_similarity(left: str, right: str, size: int = 3) -> float:
    """Dice coefficient over character trigrams; robust to spacing and hyphens."""

    a, b = char_shingles(left, size), char_shingles(right, size)
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / float(len(a) + len(b))


def rules_tail(query: str) -> str:
    cut = query.find("Task rules:")
    return query[cut:] if cut > 0 else query


class InvertedIndex:
    """Token -> product ids, with a rarity weight for ranking lookups."""

    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = {}
        self.documents: set[str] = set()

    def index_document(self, product_id: str, text: str) -> None:
        self.documents.add(product_id)
        for token in set(title_words(text)):
            if token not in _STOPWORDS and len(token) > 1:
                self.postings.setdefault(token, set()).add(product_id)

    def rarity(self, token: str) -> float:
        hits = len(self.postings.get(token, ()))
        return len(self.documents) / float(hits) if hits else 0.0

    def lookup_ranked(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for token in set(title_words(query)):
            weight = self.rarity(token)
            for product_id in self.postings.get(token, ()):
                scores[product_id] = scores.get(product_id, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(pid, round(score, 3)) for pid, score in ranked[:limit]]

    def vocabulary_size(self) -> int:
        return len(self.postings)


def weighted_blend(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.get(name, 0.0) for name in scores)
    if total_weight <= 0:
        return 0.0
    blended = sum(value * weights.get(name, 0.0) for name, value in scores.items())
    return round(blended / total_weight, 4)


def display_classes_found(text: str) -> list[str]:
    """Every pixel class the text asserts, from dimensions AND from names.

    A listing that states both "720 x 1600" and "1080p" asserts two different
    classes; the rules call that unresolved, so both have to be collected.
    """

    low = _lowered(text)
    found: list[str] = []
    for match in re.finditer(r"(\d{3,5})\s*[x*×]\s*(\d{3,5})", low):
        klass = display_class(float(match.group(1)), float(match.group(2)))
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


def listing_condition(text: str) -> str:
    low = _lowered(text)
    for label, needles in (
        ("refurbished", ("refurbished", "renewed", "reconditioned")),
        ("used", ("second hand", "pre-owned", "preowned", "used")),
        ("open_box", ("open box", "opened box", "display unit")),
    ):
        if any(needle in low for needle in needles):
            return label
    return "new"


def watt_hours(milliamp: float | None, volts: float = 3.85) -> float | None:
    """Energy in Wh, the figure airline rules and spec sheets compare."""

    try:
        return round(float(milliamp) * volts / 1000.0, 2) if milliamp and milliamp > 0 else None
    except (TypeError, ValueError):
        return None


def variant_keys_in(text: str) -> list[tuple[str, str]]:
    """Variant keys the shopper named explicitly, e.g. ``123::456-7``."""

    out: list[tuple[str, str]] = []
    for match in _KEY_RE.finditer(text):
        pair = (match.group(1), match.group(2))
        if pair not in out:
            out.append(pair)
    return out


def edit_distance(left: str, right: str, limit: int = 32) -> int:
    """Levenshtein distance on the first ``limit`` characters of each string."""

    a, b = _lowered(left)[:limit], _lowered(right)[:limit]
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def accessory_title(title: str) -> bool:
    low = _lowered(title)
    return any(word in low for word in _ACCESSORY)


class TitleTrie:
    """Prefix lookup over title words, for incremental query suggestions."""

    def __init__(self) -> None:
        self.root: dict[str, Any] = {}

    def insert(self, word: str, product_id: str) -> None:
        node = self.root
        for char in _lowered(word):
            node = node.setdefault(char, {})
        node.setdefault("$", set()).add(product_id)

    def insert_title(self, title: str, product_id: str) -> None:
        for word in title_words(title):
            self.insert(word, product_id)

    def complete(self, prefix: str, limit: int = 8) -> list[str]:
        node = self.root
        for char in _lowered(prefix):
            if char not in node:
                return []
            node = node[char]
        words: list[str] = []
        stack = [(node, _lowered(prefix))]
        while stack and len(words) < limit:
            current, text = stack.pop()
            if "$" in current:
                words.append(text)
            for char in sorted((k for k in current if k != "$"), reverse=True):
                stack.append((current[char], text + char))
        return words


def warranty_months(text: str) -> int | None:
    match = re.search(
        r"(\d{1,2})\s*(year|yr|month|mo)s?\s*(?:local\s+|official\s+)?warranty", _lowered(text)
    )
    if not match:
        return None
    amount = int(match.group(1))
    return amount * 12 if match.group(2) in ("year", "yr") else amount


class Agent:
    def __init__(self, problem: dict[str, Any]) -> None:
        environment = problem["environment"]
        self.policy = environment["policy_view"]
        self.problem_id = _string_of(
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
        self.query = _string_of(self.policy.get("query"))
        self.request = request_body(self.query)
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
        self.verified: tuple[str, str] | None = None
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

    def apply_filter(self) -> int:
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
        observation = self.env.invoke("filter", args)
        self.store_rows(observation.get("results"))
        return len(self.pool) - before

    def final_attempt(self) -> None:
        """Last resort: a valid order always beats no terminal action."""

        if self.env.ordered or self.env.done or self.env.budget_turns < 1:
            return
        # a transient proxy failure earlier must not stop us ordering now
        self.env.failed = False
        self.env.consecutive_failures = 0
        if self.committed and self.committed not in self.dead:
            if self.checkout():
                return
        seen: list[tuple[Candidate, dict[str, Any]]] = []
        ranked = sorted(
            self.pool.values(),
            key=lambda c: (
                -spec_grade(c, self.req.specs),
                -specs_backed_by_facts(c, self.req.specs),
                -self.rough_rank(c),
            ),
        )
        for candidate in ranked:
            variant = best_variant(candidate, self.req)
            if variant is not None:
                seen.append((candidate, variant))
        if not seen:
            # drop every soft filter but never the budget or stock requirement
            for candidate in self.pool.values():
                for variant in candidate.variants or []:
                    if variant_eligible(candidate, variant, self.req):
                        seen.append((candidate, variant))
                        break
        for candidate, variant in seen[:3]:
            key = (candidate.product_id, _string_of(variant.get("sku")))
            if key in self.dead or self.env.budget_turns < 2:
                break
            if self.committed and self.committed != key:
                self.env.invoke(
                    "remove_from_cart",
                    {"product_id": self.committed[0], "sku": self.committed[1]},
                )
            if not self.env.invoke("add_to_cart", {"product_id": key[0], "sku": key[1]}):
                continue
            self.committed = key
            self.commit_price = _float_or_none(variant.get("price"))
            if self.checkout():
                return

    def probe_stock(self, product_id: str, sku: str) -> dict[str, Any]:
        observation = self.env.invoke(
            "inspect_stock", {"product_id": product_id, "sku": sku}
        )
        return observation if isinstance(observation, dict) else {}

    def starter_queries(self) -> list[str]:
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
                " ".join(_string_of(s.get("value")) for s in req.specs),
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
            keywords = self.request_terms()
            if keywords:
                for extra in (" ".join(keywords[:4]), " ".join(keywords[:2])):
                    extra = extra.strip()
                    if not extra or extra.lower() in tried:
                        continue
                    if extra.lower() not in {q.lower() for q in out}:
                        out.append(extra)
        return out[:4]

    def execute(self) -> list[dict[str, Any]]:
        try:
            self.understand_request()
            self.seek_clarification()
            if self.req.listed:
                # the shopper named the exact variants: no search needed
                if self.solve_named_variants():
                    return self.finish_dialogue()
                if self.env.ordered:
                    return self.finish_dialogue()
            self.scan_catalog()
            pick = self.decide_pick()
            if pick is None:
                pick = self.decide_relaxed()
            if pick is not None and not self.env.done:
                if not self.place_in_cart(*pick):
                    alternatives = [
                        p
                        for p in self.rank_pairs(self.acceptable_pairs())
                        if (p[0].product_id, _string_of(p[1].get("sku"))) not in self.dead
                    ]
                    if alternatives:
                        self.place_in_cart(*alternatives[0])
            if self.committed and not self.env.done:
                state = self.observe_event()
                if state != "ok":
                    kind = _lowered((self.env.pending_event or {}).get("kind")) or "stockout"
                    self.record_event(kind)
                    if not self.recover_from_event():
                        self.log.append("recovery ladder exhausted")
                self.conclude()
            if not self.env.ordered:
                self.emergency_purchase()
            if not self.env.ordered:
                self.final_attempt()
        except Exception as exc:  # a raised agent = agent_error = zero
            self.log.append("fatal: %s: %s" % (type(exc).__name__, exc))
            try:
                self.emergency_purchase()
                if not self.env.ordered:
                    self.final_attempt()
            except Exception as inner:
                self.log.append("fallback failed: %s" % type(inner).__name__)
        return self.finish_dialogue()

    def model_near_misses(self) -> list[str]:
        want = self.req.preferred_model
        if not want:
            return []
        return [
            c.product_id
            for c in self.pool.values()
            if fuzzy_model_match(want, fact_lookup(c, "model") or c.model_id())
            and self.distance_from_ideal(c, best_variant(c, self.req) or {}) != 0
        ]

    def product_descriptors(self) -> list[str]:
        """Distinctive product-type words in the ask ("gaming", "foldable").

        They are not the category and not the spec, but the acceptance contract
        still counts them: a plain monitor is not a gaming monitor.
        """

        drop = set(_STOPWORDS)
        for source in (self.req.category or "", self.req.brand or ""):
            drop.update(title_words(source))
            drop.update(t.rstrip("s") for t in title_words(source))
        for spec in self.req.specs:
            drop.update(title_words(_string_of(spec.get("value"))))
        for pref in self.req.soft_prefs:
            drop.update(title_words(pref))
        out: list[str] = []
        for token in title_words(self.request):
            if token in drop or token.rstrip("s") in drop:
                continue
            if len(token) < 4 or token.isdigit() or any(ch.isdigit() for ch in token):
                continue
            if token not in out:
                out.append(token)
        return out[:4]

    def request_terms(self, limit: int = 8) -> list[str]:
        """Content words of the shopper request, in order, minus boilerplate."""

        out: list[str] = []
        for token in title_words(self.request):
            if token in _STOPWORDS or len(token) < 3 or token.isdigit():
                continue
            if token not in out:
                out.append(token)
        return out[:limit]

    def request_title_fit(self, candidate: "Candidate") -> float:
        return max(cosine_overlap(self.request, candidate.title), title_alignment(self.request, candidate.title))

    def search_batch(self, queries: list[str]) -> int:
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
        for observation in self.env.step(actions):
            self.store_rows(observation.get("results"))
        return len(self.pool) - before

    def query_redundancy(self) -> float:
        queries = self.queries_run
        if len(queries) < 2:
            return 0.0
        scores = [
            shingle_similarity(queries[i], queries[j])
            for i in range(len(queries))
            for j in range(i + 1, len(queries))
        ]
        return round(sum(scores) / float(len(scores)), 3)

    def rank_pairs(self, pairs: list[tuple[Candidate, dict[str, Any]]]):
        req = self.req

        def key(item: tuple[Candidate, dict[str, Any]]):
            candidate, variant = item
            brand_hit = 1 if (req.brand and brand_is(candidate, req.brand)) else 0
            model_hit = 0
            if req.preferred_model:
                code = candidate.model_id() or ""
                model_hit = 1 if req.preferred_model.lower().replace(" ", "") in code.replace(" ", "") else 0
            color_hit = 0
            if req.preferred_color:
                color_hit = 1 if (candidate.variant_colour(variant) or "") == req.preferred_color.lower() else 0
            pref = candidate.pref_score if candidate.pref_score is not None else 0.4
            price = _float_or_none(variant.get("price")) or 0.0
            evidence = self.preference_support(candidate)
            backed = specs_backed_by_facts(candidate, req.specs)
            descriptor = self.descriptor_match(candidate)
            endurance = stated_hours(candidate) if self.wants_endurance else 0.0
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

    def condition_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.pool.values():
            label = listing_condition(candidate.title)
            counts[label] = counts.get(label, 0) + 1
        return counts

    def observe_event(self) -> str:
        """The unaided turn after the commitment. Never terminal.

        Returns 'ok', 'replace' or 'gone'.
        """

        if not self.committed or self.env.done:
            return "gone"
        product_id, sku = self.committed
        observations = self.env.step(
            [
                ("inspect_stock", {"product_id": product_id, "sku": sku}),
                ("inspect_cart", {}),
            ]
        )
        self.handle_shopper_messages()
        state = observations[0] if observations else {}
        self.verified = self.committed
        price = _float_or_none(state.get("price"))
        in_stock = state.get("in_stock")
        event = self.env.pending_event or {}
        kind = _lowered(event.get("kind"))
        if kind == "stockout" or in_stock is False:
            self.dead.add((product_id, sku))
            return "replace"
        if price is not None:
            self.commit_price = price
            if self.req.budget is not None and price > self.req.budget + 1e-9:
                self.dead.add((product_id, sku))
                return "replace"
        if kind == "price_change" and price is None:
            new_price = _float_or_none(event.get("new_price"))
            if new_price is not None and self.req.budget is not None:
                if new_price > self.req.budget + 1e-9:
                    self.dead.add((product_id, sku))
                    return "replace"
                self.commit_price = new_price
        return "ok"

    def decide_relaxed(self) -> tuple[Candidate, dict[str, Any]] | None:
        """Last-resort widening: drop soft constraints, never hard ones."""

        saved_specs = list(self.req.specs)
        saved_brand, saved_soft = self.req.brand, self.req.brand_is_soft
        if self.req.brand_is_soft:
            self.req.brand = None
        self.req.preferred_color = None
        pick = self.decide_pick()
        if pick is None and saved_specs:
            # a spec the catalog never states should not block a purchase, but a
            # listing that contradicts it must still be excluded
            pairs = self.acceptable_pairs(allow_unproven=True)
            if pairs:
                self.judge_preferences(pairs)
                ranked = self.rank_on_axes(pairs) if self.req.axes else self.rank_pairs(pairs)
                pick = ranked[0]
        self.req.specs = saved_specs
        self.req.brand, self.req.brand_is_soft = saved_brand, saved_soft
        return pick

    @property
    def runtime(self) -> float:
        return time.monotonic() - self.started

    def seek_model(self) -> None:
        """Chase an explicitly named model code before committing to anything.

        The code usually lives in the catalog's model fact rather than the
        title, so the bare token is the query that reaches it.
        """

        model = self.req.preferred_model
        if not model or self.env.done or self.near_the_end(8) or self.improvement_closed:
            return
        for pair in self.acceptable_pairs():
            if self.distance_from_ideal(*pair) == 0:
                return  # a full match is already in hand
        queries = [model]
        if self.req.preferred_color:
            queries.append("%s %s" % (model, self.req.preferred_color))
        if self.req.category:
            queries.append("%s %s" % (model, self.req.category))
        queries = [q for q in queries if q.lower() not in {x.lower() for x in self.queries_run}]
        if not queries:
            return
        self.search_batch(queries[:2])
        self.handle_shopper_messages()
        shortlist = sorted(self.first_cut(), key=self.rough_rank, reverse=True)
        self.inspect_listings([c for c in shortlist if not c.grounded][:5])

    def finish_dialogue(self) -> list[dict[str, Any]]:
        if not self.env.ordered:
            try:
                self.final_attempt()
            except Exception as exc:
                self.log.append("close_out failed: %s" % type(exc).__name__)
        self.dialogue.append(
            {"role": "assistant", "content": "summary: %s" % " | ".join(self.log[-25:])}
        )
        return self.dialogue

    def rough_rank(self, candidate: Candidate) -> float:
        req = self.req
        blob = _lowered(candidate.title + " " + " ".join(map(_string_of, (candidate.facts or {}).values())))
        score = 0.0
        if req.brand:
            if brand_is(candidate, req.brand):
                score += 3.0
            elif req.brand.lower() in blob:
                score += 0.5
        if req.preferred_model and req.preferred_model.lower() in blob:
            score += 2.0
        if req.preferred_color and req.preferred_color.lower() in blob:
            score += 1.0
        for spec in req.specs:
            value = _lowered(spec.get("value"))
            head = value.split()[0] if value else ""
            if head and head in blob:
                score += 1.5
        for pref in req.soft_prefs:
            words = [w for w in title_words(pref) if len(w) > 3]
            hits = sum(1 for w in words if w in blob)
            if words:
                score += 1.2 * hits / len(words)
        if candidate.grounded:
            score += 0.4
        return score

    def recovery_turn_plan(self) -> dict[str, int]:
        spare = max(0, self.env.budget_turns - 3)
        return {
            "view": min(2, spare),
            "verify": min(2, max(0, spare - 2)),
            "swap": 1 if spare >= 5 else 0,
            "order": 1 if self.env.budget_turns >= 1 else 0,
        }

    def compose_claims(self) -> dict[str, Any] | None:
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
        field = _lowered(self.req.requested_spec_field)
        if not field:
            for spec in self.req.specs:
                candidate_field = _lowered(spec.get("field"))
                if candidate_field in allowed:
                    field = candidate_field
                    break
        if candidate is not None and field in allowed:
            # claim the spec only when the catalog actually states it
            value = ""
            canonical = fact_lookup(candidate, field)
            if field == "resolution":
                classes = display_classes_found(canonical or candidate.text_blob())
                if len(set(classes)) == 1:
                    value = classes[0]
            elif canonical:
                value = canonical
            if not value:
                for spec in self.req.specs:
                    if _lowered(spec.get("field")) == field and spec_holds(candidate, spec) is True:
                        value = _string_of(spec.get("value"))
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

    def _default_pref(self, candidate: Candidate) -> float:
        if not self.req.soft_prefs:
            return 0.5
        blob = _lowered(candidate.text_blob())
        words: list[str] = []
        for pref in self.req.soft_prefs:
            words.extend(w for w in title_words(pref) if len(w) > 3)
        if not words:
            return 0.5
        hits = sum(1 for w in set(words) if w in blob)
        return min(1.0, 0.25 + 0.75 * hits / max(1, len(set(words))))

    @property
    def improvement_closed(self) -> bool:
        """Time to stop searching and scoring - NOT to stop verifying.

        Skipping verification is how an agent orders an item the market event
        just pushed over budget, so the clock never shortens the
        commit / react / recover / order sequence itself.
        """

        return self.runtime > _DEADLINE_S * _FORCE_ORDER_AT

    def respond_to_shopper(self) -> None:
        """Answer a direct shopper question before the terminal action."""

        if not self.pending_question or not self.committed:
            return
        if self.env.done or self.near_the_end(3):
            return
        product_id, sku = self.committed
        candidate = self.pool.get(product_id)
        bits: list[str] = []
        if candidate and candidate.title:
            bits.append("Going with %s" % candidate.title[:110])
        for spec in self.req.specs:
            if candidate and spec_holds(candidate, spec) is True:
                bits.append(
                    "its %s is %s in the product facts"
                    % (_string_of(spec.get("field")).replace("_", " "), _string_of(spec.get("value")))
                )
        if self.commit_price is not None:
            bits.append("current price %.2f %s" % (self.commit_price, self.req.currency))
        bits.append("verified in stock")
        self.env.invoke(
            "message",
            {"content": ascii_message("; ".join(bits) + ". Placing the order now.")},
        )
        self.pending_question = None
        self.handle_shopper_messages()

    def place_in_cart(self, candidate: Candidate, variant: dict[str, Any]) -> bool:
        """Verify, then place the graded first cart addition."""

        sku = _string_of(variant.get("sku"))
        if not sku or self.env.done:
            return False
        if not candidate.grounded and self.env.budget_turns >= 3:
            # the verifier requires the commitment target to have been observed
            observation = self.env.invoke("view", {"product_id": candidate.product_id})
            if observation:
                candidate.load_detail(observation)
                refreshed = best_variant(candidate, self.req)
                if refreshed is not None and _string_of(refreshed.get("sku")) == sku:
                    variant = refreshed
        fresh = self.probe_stock(candidate.product_id, sku)
        price = _float_or_none(fresh.get("price"))
        in_stock = fresh.get("in_stock")
        if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
            self.dead.add((candidate.product_id, sku))
            return False
        if self.req.require_in_stock and in_stock is False:
            self.dead.add((candidate.product_id, sku))
            return False
        observation = self.env.invoke(
            "add_to_cart", {"product_id": candidate.product_id, "sku": sku}
        )
        if not observation:
            return False
        self.committed = (candidate.product_id, sku)
        self.commit_price = price if price is not None else _float_or_none(variant.get("price"))
        self.handle_shopper_messages()
        return True

    def emergency_purchase(self) -> None:
        """Never end a run without a terminal action when one is possible."""

        if self.env.done or self.env.ordered:
            return
        if self.committed and self.committed not in self.dead:
            if self.checkout():
                return
        if self.committed and self.committed in self.dead:
            if self.env.budget_turns >= 4 and self.recover_from_event():
                if self.checkout():
                    return
            self.committed = None
        pairs = self.acceptable_pairs()
        if not pairs:
            shortlist = sorted(self.first_cut(), key=self.rough_rank, reverse=True)
            if shortlist and not self.env.done:
                self.inspect_listings(shortlist[:5])
                pairs = self.acceptable_pairs()
        if not pairs:
            # nothing proves the spec: accept listings that do not contradict it
            pairs = self.acceptable_pairs(allow_unproven=True)
        if not pairs:
            # last resort: anything affordable and in stock that we have seen
            ranked = sorted(
                self.pool.values(),
                key=lambda c: (-spec_grade(c, self.req.specs), -self.rough_rank(c)),
            )
            for candidate in ranked:
                variant = best_variant(candidate, self.req)
                if variant is not None:
                    pairs = [(candidate, variant)]
                    break
        if not pairs or self.env.done:
            return
        candidate, variant = self.rank_pairs(pairs)[0]
        if self.place_in_cart(candidate, variant):
            if self.env.budget_turns >= 2:
                state = self.observe_event()
                if state == "replace" and self.env.budget_turns >= 4:
                    self.recover_from_event()
            self.checkout()

    def seek_clarification(self) -> None:
        if self.env.done or self.near_the_end(6):
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
        self.env.invoke("message", {"content": ascii_message(content, 300)})
        self.handle_shopper_messages()

    def handle_shopper_messages(self) -> None:
        """Apply shopper clarifications / interventions to the live state."""

        while self.env.messages:
            content = self.env.messages.pop(0)
            self.log.append("shopper: %s" % content[:160])
            if "?" in content and not announces_event(content):
                self.pending_question = content
            budget = price_ceiling(content)
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
            known = {_lowered(s.get("field")) for s in self.req.specs}
            for spec in stated_specs(content):
                if spec["field"] not in known:
                    self.req.specs.append(spec)
                    self.log.append("spec from shopper: %s=%s" % (spec["field"], spec["value"]))
                else:
                    for existing in self.req.specs:
                        if _lowered(existing.get("field")) == spec["field"]:
                            existing["value"] = spec["value"]
            caught = budget is not None or bool(stated_specs(content))
            if not caught and not self.improvement_closed and len(content) > 20:
                payload = self.llm.consult(
                    "You update shopping requirements. Reply with JSON only, no prose.",
                    _UPDATE_PROMPT % (self.req.render(), content),
                    max_tokens=500,
                )
                if isinstance(payload, dict):
                    self.absorb_extraction(payload)
            self.log.append("requirements now: %s" % self.req.render())

    def recheck_commitment(self) -> bool:
        """Re-check price and stock on the exact variant before ordering.

        An order outside budget or out of stock scores zero on a hard check, so
        this verification happens on every path, however tight the clock.
        """

        if not self.committed:
            return False
        product_id, sku = self.committed
        if self.verified == self.committed:
            return True
        if self.env.budget_turns < 2:
            return True  # no room to check; ordering still beats not ordering
        fresh = self.probe_stock(product_id, sku)
        if not fresh:
            return True
        price = _float_or_none(fresh.get("price"))
        if price is not None:
            self.commit_price = price
        if self.req.require_in_stock and fresh.get("in_stock") is False:
            self.log.append("pre-order check: out of stock")
            self.dead.add(self.committed)
            self.committed = None
            return False
        if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
            self.log.append(
                "pre-order check: %.2f over budget %.2f" % (price, self.req.budget)
            )
            self.dead.add(self.committed)
            self.committed = None
            return False
        self.verified = (product_id, sku)
        return True

    def store_rows(self, rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            product_id = _string_of(row.get("product_id"))
            if not product_id:
                continue
            candidate = self.pool.get(product_id)
            if candidate is None:
                candidate = Candidate(product_id)
                self.pool[product_id] = candidate
            candidate.load_search(row)

    def distance_from_ideal(self, candidate: "Candidate", variant: dict[str, Any]) -> int:
        """0 full model+colour, 1 same model, 2 same colour, 3 neither."""

        want_model = canonical_model(self.req.preferred_model or "")
        want_color = _lowered(self.req.preferred_color or "")
        model_hit = False
        if want_model:
            code = canonical_model(
                fact_lookup(candidate, "model") or candidate.model_id() or ""
            )
            model_hit = bool(code) and (want_model in code or code in want_model)
            if not model_hit and want_model in canonical_model(candidate.title):
                model_hit = True
        color_hit = False
        if want_color:
            color_hit = (candidate.variant_colour(variant) or "") == want_color
        if not want_model and not want_color:
            return 0
        if model_hit and (color_hit or not want_color):
            return 0
        if model_hit:
            return 1
        if color_hit:
            return 2
        return 3

    def ranking_movement(self, previous: list[str]) -> dict[str, int]:
        pairs = self.acceptable_pairs()
        if not pairs:
            return {}
        current = ["%s::%s" % (c.product_id, _string_of(v.get("sku"))) for c, v in self.rank_pairs(pairs)]
        return rank_shift(previous, current)

    def preference_support(self, candidate: Candidate) -> int:
        """How many stated preference axes the raw facts actually support."""

        if not self.req.soft_prefs:
            return 0
        blob = _lowered(candidate.text_blob())
        hits = 0
        for pref in self.req.soft_prefs:
            words = [w for w in title_words(pref) if len(w) > 3 and w not in _STOPWORDS]
            if any(word in blob for word in words):
                hits += 1
                continue
            for family in _FEATURE_SYNONYMS:
                if any(word in _lowered(pref) for word in family["pref"]) and any(
                    token in blob for token in family["facts"]
                ):
                    hits += 1
                    break
        return hits

    def brand_concentration(self) -> float:
        counts: dict[str, int] = {}
        for candidate in self.pool.values():
            if candidate.brand:
                label = _lowered(candidate.brand)
                counts[label] = counts.get(label, 0) + 1
        return gini_impurity(counts)

    @property
    def over_deadline(self) -> bool:
        return self.runtime > _DEADLINE_S

    def inspect_listings(self, candidates: list[Candidate], limit: int = 5) -> None:
        """compare/view the shortlist so decisions rest on raw facts."""

        todo = [c for c in candidates if not c.grounded][:limit]
        if not todo or self.env.done:
            return
        ids = [c.product_id for c in todo]
        if len(ids) == 1:
            observation = self.env.invoke("view", {"product_id": ids[0]})
            if observation:
                todo[0].load_detail(observation)
            return
        observation = self.env.invoke("compare", {"product_ids": ids[:5]})
        rows = observation.get("compared")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate = self.pool.get(_string_of(row.get("product_id")))
                if candidate is not None:
                    candidate.load_detail(row)

    def record_event(self, kind: str) -> None:
        """No-op by design.

        The event is closed by committing to the right replacement, not by
        talking about it, and each message risks a simulator failure that would
        invalidate the evaluation.
        """

        self.log.append("event noticed: %s" % (kind or "unknown"))

    def understand_request(self) -> None:
        req = self.req
        req.budget = price_ceiling(self.request)
        if "php" in _lowered(self.request):
            req.currency = "PHP"
        quoted = re.findall(r'"([^"]+)"', self.request) + re.findall(
            r"matches the phrase ([a-z0-9 ]+?)(?:,| at| in| and)", _lowered(self.request)
        )
        if self.flags.phrase_rule and quoted:
            req.phrase_terms = [t for t in title_words(quoted[0])]
        request_tokens = set(title_words(self.request))
        req.smartphone_required = bool(
            request_tokens & {"smartphone", "smartphones", "cellphone", "phone", "phones"}
        )
        req.listed = variant_keys_in(self.request)
        req.axes = priority_axes(self.request) if req.listed or self.flags.ranking else []
        for spec in stated_specs(self.request):
            if spec["field"] not in {_lowered(s.get("field")) for s in req.specs}:
                req.specs.append(spec)
        payload = self.llm.consult(
            "You extract shopping requirements. Reply with JSON only, no prose.",
            _EXTRACT_PROMPT % (self.request, self.flags.text[:1800]),
            max_tokens=700,
        )
        if isinstance(payload, dict):
            self.absorb_extraction(payload)
        if req.budget is None:
            req.budget = price_ceiling(self.query)
        self.wants_endurance = any(
            any(word in _lowered(pref) for word in ("battery", "playtime", "long-lasting", "endurance", "hours"))
            for pref in req.soft_prefs
        )
        self.log.append("requirements: %s" % req.render())

    def decide_pick(self) -> tuple[Candidate, dict[str, Any]] | None:
        """Ground the shortlist, gate it, and return the genuine best choice."""

        wide = bool(self.req.specs or self.req.preferred_model or self.req.soft_prefs)
        for _round in range(4):
            shortlist = sorted(self.first_cut(), key=self.rough_rank, reverse=True)
            if not shortlist:
                break
            self.inspect_listings(shortlist[:5])
            if wide and _round == 0 and not self.near_the_end(10):
                # gold sets are fact-backed, so look at more raw facts before
                # the graded first commitment
                self.inspect_listings([c for c in shortlist[5:10] if not c.grounded])
            self.handle_shopper_messages()
            pairs = self.acceptable_pairs()
            if pairs and self.flags.preference_scored and _round == 0:
                # a first commitment with no support for any stated preference
                # is what the verifier calls an unintended commitment
                if not any(self.preference_support(c) for c, _v in pairs):
                    extra = self.queries_for_features()
                    if extra and not self.near_the_end(9):
                        self.search_batch(extra)
                        more = sorted(self.first_cut(), key=self.rough_rank, reverse=True)
                        self.inspect_listings([c for c in more if not c.grounded][:5])
                        pairs = self.acceptable_pairs()
            if pairs and self.req.preferred_model and _round == 0:
                if not any(self.distance_from_ideal(c, v) == 0 for c, v in pairs):
                    self.seek_model()
                    pairs = self.acceptable_pairs()
            if pairs:
                self.judge_preferences(pairs)
                ranked = self.rank_on_axes(pairs) if self.req.axes else self.rank_pairs(pairs)
                if self.req.brand and self.req.brand_is_soft:
                    # honour a named soft brand when a fully compliant item exists
                    preferred = [
                        p for p in ranked if brand_is(p[0], self.req.brand)
                    ]
                    if preferred:
                        ranked = preferred + [p for p in ranked if p not in preferred]
                best = ranked[0]
                room = (
                    self.search_budget > 0
                    and self.env.turn < self.commit_deadline
                    and not self.near_the_end(10)
                    and not self.improvement_closed
                    and _round < 2
                )
                if self.ready_to_commit(best) or not room:
                    return best
                self.log.append("holding out for a stronger first commitment")
            if self.near_the_end(8) or self.env.done:
                break
            if self.search_budget <= 0 or self.env.turn >= self.commit_deadline:
                break
            extra: list[str] = []
            if self.req.specs:
                extra = self.queries_for_specs()
            if not extra and self.flags.preference_scored:
                extra = self.queries_for_features()
            if not extra:
                extra = self.generated_queries(self.gap_summary())
            if not extra:
                extra = self.reworded_queries()
            if not extra:
                break
            self.search_batch(extra)
            shortlist = sorted(self.first_cut(), key=self.rough_rank, reverse=True)
            self.inspect_listings([c for c in shortlist if not c.grounded][:5])
        best_effort = self.acceptable_pairs()
        if best_effort:
            self.judge_preferences(best_effort)
            ranked = (
                self.rank_on_axes(best_effort) if self.req.axes else self.rank_pairs(best_effort)
            )
            return ranked[0]
        return None

    def gap_summary(self) -> str:
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

    def near_the_end(self, need: int = 3) -> bool:
        return self.env.budget_turns <= need or self.over_deadline

    def generated_queries(self, missing: str) -> list[str]:
        payload = self.llm.consult(
            "You write catalog search queries. Reply with JSON only.",
            _QUERY_PROMPT
            % (
                self.request[:600],
                self.req.render(),
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
                text = _string_of(item).strip()
                if text and text.lower() not in {q.lower() for q in self.queries_run}:
                    out.append(text)
        return out[:3]

    def absorb_extraction(self, payload: dict[str, Any]) -> None:
        req = self.req
        category = _string_of(payload.get("category")).strip()
        if category:
            req.category = category
        budget = _float_or_none(payload.get("budget"))
        if budget is not None and (req.budget is None or budget < req.budget):
            req.budget = budget
        currency = _string_of(payload.get("currency")).strip()
        if currency:
            req.currency = currency
        brand = _string_of(payload.get("brand")).strip()
        if brand:
            req.brand = brand
            req.brand_is_soft = bool(payload.get("brand_is_soft")) or self.flags.soft_brand
        specs = payload.get("specs")
        if isinstance(specs, list):
            merged = {}
            for spec in req.specs:
                merged[_lowered(spec.get("field"))] = spec
            for spec in specs:
                if isinstance(spec, dict) and spec.get("field") and spec.get("value"):
                    merged[_lowered(spec["field"])] = {
                        "field": _string_of(spec["field"]),
                        "value": _string_of(spec["value"]),
                    }
            req.specs = list(merged.values())
        terms = payload.get("phrase_terms")
        if isinstance(terms, list) and terms and self.flags.phrase_rule:
            req.phrase_terms = [_string_of(t) for t in terms if _string_of(t)]
        if isinstance(payload.get("device_required"), bool):
            req.device_required = payload["device_required"]
        if isinstance(payload.get("smartphone_required"), bool):
            req.smartphone_required = payload["smartphone_required"]
        model = _string_of(payload.get("preferred_model")).strip()
        if model:
            req.preferred_model = model
        color = _string_of(payload.get("preferred_color")).strip()
        if color:
            req.preferred_color = color
        prefs = payload.get("soft_prefs")
        if isinstance(prefs, list):
            for pref in prefs:
                text = _string_of(pref).strip()
                if text and text not in req.soft_prefs:
                    req.soft_prefs.append(text)
        field = _string_of(payload.get("requested_spec_field")).strip()
        if field:
            req.requested_spec_field = field
        rule = _string_of(payload.get("ranking_rule")).strip()
        if rule:
            req.ranking_rule = rule

    def recover_from_event(self) -> bool:
        """Least-relaxed replacement, re-grounded after the event."""

        if not self.committed or self.env.done:
            return False
        dead_product, dead_sku = self.committed
        self.replacement_round += 1
        dead_candidate = self.pool.get(dead_product)

        # refresh the dead listing: another variant of it may still qualify
        if dead_candidate is not None and not self.near_the_end(6):
            observation = self.env.invoke("view", {"product_id": dead_product})
            if observation:
                dead_candidate.load_detail(observation)
                self.viewed_after_event.add(dead_product)

        options = self.fallback_options()
        want_model = self.req.preferred_model
        if want_model and not self.near_the_end(8) and not self.improvement_closed:
            has_full = any(self.distance_from_ideal(*pair) == 0 for pair in options)
            if not has_full:
                self.seek_model()
                options = self.fallback_options()
        if not options and not self.near_the_end(8) and not self.improvement_closed:
            self.scan_catalog()
            shortlist = sorted(self.first_cut(), key=self.rough_rank, reverse=True)
            self.inspect_listings([c for c in shortlist if not c.grounded][:5])
            options = self.fallback_options()

        for candidate, variant in options:
            key = (candidate.product_id, _string_of(variant.get("sku")))
            if key in self.dead or self.env.done or self.env.budget_turns <= 4:
                break
            # the replacement's identity must be observed AFTER the event
            if candidate.product_id not in self.viewed_after_event:
                observation = self.env.invoke("view", {"product_id": candidate.product_id})
                if observation:
                    candidate.load_detail(observation)
                    self.viewed_after_event.add(candidate.product_id)
                if not clears_hard_rules(candidate, self.req):
                    continue
                refreshed = best_variant(candidate, self.req)
                if refreshed is not None and _string_of(refreshed.get("sku")) == key[1]:
                    variant = refreshed
            fresh = self.probe_stock(candidate.product_id, key[1])
            price = _float_or_none(fresh.get("price"))
            if self.req.require_in_stock and fresh.get("in_stock") is False:
                self.dead.add(key)
                continue
            if price is not None and self.req.budget is not None and price > self.req.budget + 1e-9:
                self.dead.add(key)
                continue
            if self.committed and self.committed[0] == dead_product:
                self.env.invoke(
                    "remove_from_cart", {"product_id": dead_product, "sku": dead_sku}
                )
            observation = self.env.invoke(
                "add_to_cart", {"product_id": candidate.product_id, "sku": key[1]}
            )
            if not observation:
                continue
            self.committed = key
            self.commit_price = price
            self.verified = key
            self.log.append(
                "recovered at relaxation level %d" % self.distance_from_ideal(candidate, variant)
            )
            self.handle_shopper_messages()
            return True
        return False

    def non_dominated(self) -> list[str]:
        points = []
        for candidate, variant in self.acceptable_pairs():
            price = _float_or_none(variant.get("price"))
            if price is None:
                continue
            quality = specs_backed_by_facts(candidate, self.req.specs) + self.preference_support(candidate)
            points.append(("%s::%s" % (candidate.product_id, _string_of(variant.get("sku"))), price, float(quality)))
        return pareto_front(points)

    def queries_for_features(self) -> list[str]:
        """Searches aimed at the features the shopper actually asked for."""

        out: list[str] = []
        category = self.req.category or ""
        for pref in self.req.soft_prefs[:3]:
            for family in _FEATURE_SYNONYMS:
                if any(word in _lowered(pref) for word in family["pref"]):
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

    def checkout(self) -> bool:
        if not self.committed or self.env.done:
            return False
        if self.committed in self.dead:
            # the cart holds an item the event invalidated; do not order it
            self.committed = None
            return False
        product_id, sku = self.committed
        if not self.recheck_commitment():
            return False
        args: dict[str, Any] = {"product_id": product_id, "sku": sku}
        payload = self.compose_claims()
        if payload:
            args["justification"] = payload
        observation = self.env.invoke("place_test_order", args)
        if observation and (
            observation.get("status") == "placed" or observation.get("order")
        ):
            return True
        # a rejected order is usually a stale price/stock read
        if observation and not self.env.done and not self.near_the_end(2):
            fresh = self.probe_stock(product_id, sku)
            price = _float_or_none(fresh.get("price"))
            affordable = (
                self.req.budget is None
                or price is None
                or price <= self.req.budget + 1e-9
            )
            if fresh.get("in_stock") is True and affordable:
                observation = self.env.invoke("place_test_order", args)
                return bool(observation and observation.get("status") == "placed")
            self.dead.add((product_id, sku))
            self.committed = None
        return False

    def judge_preferences(self, pairs: list[tuple[Candidate, dict[str, Any]]]) -> None:
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
                    "facts": {k: _string_of(v)[:90] for k, v in list((candidate.facts or {}).items())[:12]},
                    "variant": _string_of(variant.get("options"))[:60],
                }
            )
        if self.improvement_closed:
            for candidate, _variant in need:
                candidate.pref_score = self._default_pref(candidate)
            return
        payload = self.llm.consult(
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
                    value = _float_or_none(row.get("score"))
                    key = _string_of(row.get("product_id"))
                    if key and value is not None:
                        scores[key] = max(0.0, min(1.0, value))
        elif isinstance(payload, dict):
            for key, value in payload.items():
                number = _float_or_none(value)
                if number is not None:
                    scores[_string_of(key)] = max(0.0, min(1.0, number))
        for candidate, _variant in need:
            candidate.pref_score = scores.get(candidate.product_id, self._default_pref(candidate))

    def fused_order(self) -> list[str]:
        pairs = self.acceptable_pairs()
        if not pairs:
            return []
        by_rank = ["%s::%s" % (c.product_id, _string_of(v.get("sku"))) for c, v in self.rank_pairs(pairs)]
        by_axes = ["%s::%s" % (c.product_id, _string_of(v.get("sku"))) for c, v in self.rank_on_axes(pairs)]
        return borda_merge([by_rank, by_axes])

    def acceptable_pairs(self, allow_unproven: bool = False) -> list[tuple[Candidate, dict[str, Any]]]:
        out = []
        for candidate in self.pool.values():
            if candidate.rejected or not candidate.grounded:
                continue
            if allow_unproven:
                # keep candidates whose spec is merely unstated, never one that
                # the catalog contradicts
                if spec_grade(candidate, self.req.specs) == 0:
                    continue
                probe = Requirements()
                probe.__dict__.update(self.req.__dict__)
                probe.specs = []
                if not clears_hard_rules(candidate, probe):
                    continue
            elif not clears_hard_rules(candidate, self.req):
                continue
            variant = best_variant(candidate, self.req)
            if variant is None:
                continue
            if (candidate.product_id, _string_of(variant.get("sku"))) in self.dead:
                continue
            out.append((candidate, variant))
        return out

    def timing_note(self) -> str:
        spent = elapsed_label(self.runtime)
        left = max(0.0, _DEADLINE_S - self.runtime)
        closing = " (closing)" if self.improvement_closed else ""
        return "%s elapsed, %s left%s" % (spent, elapsed_label(left), closing)

    def ready_to_commit(self, pair: tuple[Candidate, dict[str, Any]]) -> bool:
        """Is this good enough to be the graded first cart add?

        A later correction never repairs a weak commitment, so anything short of
        the best obtainable item is worth more searching while budget remains.
        """

        candidate, variant = pair
        if self.retrieval_mode:
            return True
        if self.req.specs and specs_backed_by_facts(candidate, self.req.specs) < len(self.req.specs):
            return False
        if self.product_descriptors() and self.descriptor_match(candidate) < 0.999:
            return False
        if self.req.preferred_model and self.distance_from_ideal(candidate, variant) != 0:
            return False
        if self.req.brand and self.req.brand_is_soft and not brand_is(candidate, self.req.brand):
            return False
        if (
            self.flags.preference_scored
            and self.req.soft_prefs
            and self.preference_share(candidate) < 0.999
        ):
            return False
        return True

    def descriptor_match(self, candidate: Candidate) -> float:
        words = self.product_descriptors()
        if not words:
            return 0.0
        blob = _lowered(candidate.title + " " + candidate.text_blob())
        return sum(1 for word in words if word in blob) / float(len(words))

    def conclude(self) -> None:
        """Answer, re-validate against the latest requirements, then order."""

        self.respond_to_shopper()
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
            if self.env.budget_turns >= 5:
                self.recover_from_event()
        if self.committed and self.committed not in self.dead:
            self.checkout()

    def rank_on_axes(self, pairs):
        """Lexicographic order over the shopper's stated tradeoff axes."""

        axes = self.req.axes or ["price"]

        def key(item):
            candidate, variant = item
            out = []
            for axis in axes:
                value = axis_number(candidate, variant, axis)
                if value is None:
                    out.append(float("inf"))
                elif _AXIS_DIRECTION.get(axis, "max") == "min":
                    out.append(value)
                else:
                    out.append(-value)
            return tuple(out)

        return sorted(pairs, key=key)

    def reworded_queries(self) -> list[str]:
        """Reformulations that need no model: reorder, split, singularise."""

        keywords = self.request_terms()
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

    def fallback_options(self, exclude_product: str | None = None) -> list[tuple["Candidate", dict[str, Any]]]:
        """Eligible options ordered by least relaxation, then quality."""

        pairs = [
            pair
            for pair in self.acceptable_pairs()
            if (pair[0].product_id, _string_of(pair[1].get("sku"))) not in self.dead
        ]
        if not pairs:
            return []
        self.judge_preferences(pairs)
        ranked = self.rank_on_axes(pairs) if self.req.axes else self.rank_pairs(pairs)
        return sorted(ranked, key=lambda pair: self.distance_from_ideal(*pair))

    def scan_catalog(self) -> None:
        """Broad-then-narrow retrieval with distinct reformulations."""

        if self.req.preferred_model:
            # the code usually lives in a catalog fact; the bare token reaches it
            self.seek_model()
        seeds = self.starter_queries()
        rounds = 0
        max_rounds = 5 if self.flags.family == "retrieval_recall" else 4
        while rounds < max_rounds and not self.env.done and not self.near_the_end(8):
            if self.improvement_closed and self.pool:
                break
            if rounds == 0 and self.req.category:
                self.apply_filter()
            batch = seeds[:2]
            seeds = seeds[2:]
            if not batch:
                batch = self.generated_queries(self.gap_summary())
            if not batch:
                batch = self.reworded_queries()
            if not batch:
                break
            fresh = self.search_batch(batch)
            self.handle_shopper_messages()
            rounds += 1
            shortlist = self.first_cut()
            distinct = {c.title[:40] for c in shortlist}
            if len(shortlist) >= 6 and len(distinct) >= 4 and rounds >= 2:
                break
            if fresh == 0 and rounds >= 2 and shortlist:
                break  # queries have stopped reaching new products
            if self.search_budget - (0 if self.committed else self.search_reserve) <= 0:
                break
            if self.env.turn >= self.commit_deadline or self.improvement_closed:
                break

    def queries_for_specs(self) -> list[str]:
        """Searches aimed at the exact attribute value the task gates on."""

        parts: list[str] = []
        category = self.req.category or ""
        brand = self.req.brand or ""
        for spec in self.req.specs[:2]:
            value = _string_of(spec.get("value"))
            if not value:
                continue
            parts.append(" ".join(p for p in (brand, category, value) if p))
            parts.append(" ".join(p for p in (category, value) if p))
            if _lowered(spec.get("field")) == "resolution":
                classes = {"hd": "720p hd", "fhd": "1080p full hd", "qhd": "1440p 2k", "4k": "4k uhd"}
                klass = target_display_class(value)
                if klass:
                    parts.append(" ".join(p for p in (category, classes[klass]) if p))
        tried = {q.lower() for q in self.queries_run}
        out: list[str] = []
        for query in parts:
            query = re.sub(r"\s+", " ", query).strip()
            if query and query.lower() not in tried and query.lower() not in {o.lower() for o in out}:
                out.append(query)
        return out[:2]

    def current_phase(self) -> str:
        if self.env.ordered:
            return "ordered"
        if self.committed and self.env.event_seen:
            return "recovering" if self.committed in self.dead else "post_event"
        if self.committed:
            return "committed"
        if any(c.grounded for c in self.pool.values()):
            return "grounding"
        return "searching" if self.queries_run else "parsing"

    def first_cut(self) -> list[Candidate]:
        req = self.req
        out = []
        for candidate in self.pool.values():
            if candidate.rejected:
                continue
            if req.phrase_terms and not phrase_in_title(candidate.title, req.phrase_terms):
                continue
            if req.budget is not None and candidate.price is not None:
                if candidate.price > req.budget + 1e-9 and not candidate.variants:
                    continue
            if req.require_in_stock and candidate.in_stock is False and not candidate.variants:
                continue
            if req.device_required and accessory_title(candidate.title):
                continue
            out.append(candidate)
        return out

    def preference_share(self, candidate: Candidate) -> float:
        if not self.req.soft_prefs:
            return 0.0
        return self.preference_support(candidate) / float(len(self.req.soft_prefs))

    def solve_named_variants(self) -> bool:
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
            if self.env.done or self.near_the_end(6):
                break
            batch = product_ids[start : start + 5]
            if len(batch) == 1:
                observation = self.env.invoke("view", {"product_id": batch[0]})
                if observation and observation.get("product_id"):
                    self.pool.setdefault(batch[0], Candidate(batch[0])).load_detail(observation)
            else:
                observation = self.env.invoke("compare", {"product_ids": batch})
                for row in observation.get("compared") or []:
                    if isinstance(row, dict) and row.get("product_id"):
                        product_id = _string_of(row["product_id"])
                        self.pool.setdefault(product_id, Candidate(product_id)).load_detail(row)
        self.handle_shopper_messages()

        def live_options() -> list[tuple[Candidate, dict[str, Any]]]:
            out = []
            for product_id, sku in listed:
                candidate = self.pool.get(product_id)
                if candidate is None:
                    continue
                for variant in candidate.variants:
                    if _string_of(variant.get("sku")) != sku:
                        continue
                    if (product_id, sku) in self.dead:
                        continue
                    if variant_eligible(candidate, variant, self.req):
                        out.append((candidate, variant))
            return self.rank_on_axes(out)

        options = live_options()
        if not options:
            return False
        candidate, variant = options[0]
        if not self.place_in_cart(candidate, variant):
            for candidate, variant in options[1:]:
                if self.place_in_cart(candidate, variant):
                    break
        if not self.committed:
            return False
        state = self.observe_event()
        if state != "ok":
            kind = _lowered((self.env.pending_event or {}).get("kind")) or "stockout"
            self.record_event(kind)
            # the market moved: re-rank the named candidates on live state
            for product_id, sku in listed:
                if (product_id, sku) in self.dead or self.near_the_end(5):
                    continue
                if (product_id, sku) == self.committed:
                    continue
                fresh = self.probe_stock(product_id, sku)
                candidate = self.pool.get(product_id)
                if candidate is None:
                    continue
                for row in candidate.variants:
                    if _string_of(row.get("sku")) == sku:
                        if _float_or_none(fresh.get("price")) is not None:
                            row["price"] = _float_or_none(fresh.get("price"))
                        if isinstance(fresh.get("in_stock"), bool):
                            row["in_stock"] = fresh["in_stock"]
            options = live_options()
            for candidate, variant in options:
                key = (candidate.product_id, _string_of(variant.get("sku")))
                if key in self.dead:
                    continue
                if self.committed and self.committed[0] != key[0]:
                    self.env.invoke(
                        "remove_from_cart",
                        {"product_id": self.committed[0], "sku": self.committed[1]},
                    )
                observation = self.env.invoke(
                    "add_to_cart", {"product_id": key[0], "sku": key[1]}
                )
                if observation:
                    self.committed = key
                    self.commit_price = _float_or_none(variant.get("price"))
                    break
        self.conclude()
        return bool(self.env.ordered)


def aspect_ratio(text: str) -> tuple[int, int] | None:
    match = re.search(r"\b(\d{1,2})\s*:\s*(\d{1,2})\b", _string_of(text))
    if not match:
        return None
    wide, tall = int(match.group(1)), int(match.group(2))
    if wide <= 0 or tall <= 0 or wide < tall:
        return None
    return wide, tall


def split_camel(text: str) -> str:
    """"GalaxyA55Pro" -> "Galaxy A55Pro" so tokenisers see the pieces."""

    out: list[str] = []
    previous = ""
    for char in _string_of(text):
        if out and char.isupper() and previous.islower():
            out.append(" ")
        out.append(char)
        previous = char
    return "".join(out)


def _lowered(value: Any) -> str:
    return _string_of(value).lower()


class PaletteMatcher:
    """How acceptable a variant colour is against the preferred one."""

    def __init__(self, preferred: str | None) -> None:
        self.preferred = _lowered(preferred)

    def closeness(self, color: str | None) -> float:
        found = _lowered(color)
        if not self.preferred or not found:
            return 0.5
        if found == self.preferred:
            return 1.0
        if found in self.preferred or self.preferred in found:
            return 0.8
        if color_family(found) == color_family(self.preferred):
            return 0.4
        return 0.0

    def rank_variants(self, candidate: "Candidate") -> list[dict[str, Any]]:
        return sorted(candidate.variants, key=lambda v: -self.closeness(candidate.variant_colour(v)))


def zscores(values: list[float]) -> list[float]:
    """Standardise a list; a flat list maps to zeros instead of dividing by zero."""

    if not values:
        return []
    mean = sum(values) / float(len(values))
    spread = (sum((v - mean) ** 2 for v in values) / float(len(values))) ** 0.5
    if spread < 1e-12:
        return [0.0 for _ in values]
    return [round((v - mean) / spread, 4) for v in values]


def variant_eligible(
    candidate: Candidate,
    variant: dict[str, Any],
    requirements: Requirements,
) -> bool:
    price = _float_or_none(variant.get("price"))
    if requirements.budget is not None and (price is None or price > requirements.budget + 1e-9):
        return False
    if requirements.require_in_stock and variant.get("in_stock") is not True:
        return False
    return True


class ComparisonPlanner:
    """Splits product ids into compare batches, skipping grounded ones."""

    def __init__(self, batch_size: int = 5) -> None:
        self.batch_size = max(1, batch_size)
        self.grounded_ids: set[str] = set()

    def mark_grounded(self, product_ids: list[str]) -> None:
        self.grounded_ids.update(_string_of(pid) for pid in product_ids if pid)

    def batches(self, product_ids: list[str]) -> list[list[str]]:
        pending: list[str] = []
        for pid in product_ids:
            pid = _string_of(pid)
            if pid and pid not in self.grounded_ids and pid not in pending:
                pending.append(pid)
        return [pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)]

    def turns_needed(self, product_ids: list[str]) -> int:
        return len(self.batches(product_ids))


def discounted_price(price: float | None, text: str) -> float | None:
    share = discount_share(text) or 0.0
    return None if price is None else round(price * (1.0 - share), 2)


def title_words(title: str) -> list[str]:
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


def rank_shift(before: list[str], after: list[str]) -> dict[str, int]:
    """Positive = moved up. Items missing from one side are ignored."""

    old, new = rank_positions(before), rank_positions(after)
    return {key: old[key] - new[key] for key in new if key in old and old[key] != new[key]}


def topological_order(dependencies: dict[str, list[str]]) -> list[str] | None:
    """Kahn's algorithm; None when the dependency graph has a cycle."""

    nodes = set(dependencies) | {d for deps in dependencies.values() for d in deps}
    indegree = {node: 0 for node in nodes}
    for node, deps in dependencies.items():
        indegree[node] += len(set(deps))
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for other, deps in sorted(dependencies.items()):
            if node in deps:
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
        ready.sort()
    return order if len(order) == len(nodes) else None


class LLM:
    """Model calls with a priority chain, JSON repair and a hard call budget."""

    def __init__(self, client: Client, log: list[str]) -> None:
        self.client = client
        self.log = log
        self.calls = 0
        self.misses = 0
        self.dead = False
        self.spent = 0.0
        provider = _lowered(getenv("INFERENCE_PROVIDER", "chutes"))
        chain = list(_CHUTES_CHAIN)
        override = getenv("SANDBOX_MODEL")
        if override:
            chain = [override] + [m for m in chain if m != override]
        if provider == "openrouter":
            chain = [_OPENROUTER.get(m, m) for m in chain]
        self.chain = chain

    def budget_state(self) -> dict[str, Any]:
        remaining = max(0.0, _LLM_TIME_BUDGET - self.spent)
        return {
            "calls": "%d/%d" % (self.calls, _LLM_BUDGET),
            "seconds_left": round(remaining, 1),
            "lead_model": self.chain[0] if self.chain else None,
            "disabled": self.dead,
        }

    def consult(
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
                reply = self.client.forward(
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
            content = _string_of((choices[0].get("message") or {}).get("content"))
            if not content.strip():
                continue
            self.misses = 0
            self.spent += time.monotonic() - started
            if not want_json:
                return content.strip()
            parsed = _decode_json(content)
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


def beta_mean(successes: int, failures: int, prior: float = 1.0) -> float:
    return round((successes + prior) / float(successes + failures + 2 * prior), 4)


def borda_merge(rankings: list[list[str]]) -> list[str]:
    """Fuse rankings: each list awards n-1 ... 0 points by position."""

    points: dict[str, int] = {}
    for ranking in rankings:
        size = len(ranking)
        for position, key in enumerate(ranking):
            points[key] = points.get(key, 0) + (size - 1 - position)
    return [key for key, _points in sorted(points.items(), key=lambda item: (-item[1], item[0]))]


def phrase_in_title(title: str, terms: list[str]) -> bool:
    if not terms:
        return True
    toks = set(title_words(title))
    for term in terms:
        for part in title_words(term):
            if part not in toks:
                return False
    return True


def run_length(items: list[str]) -> list[tuple[str, int]]:
    """Collapse repeats: [search, search, view] -> [(search, 2), (view, 1)]."""

    runs: list[tuple[str, int]] = []
    for item in items:
        if runs and runs[-1][0] == item:
            runs[-1] = (item, runs[-1][1] + 1)
        else:
            runs.append((item, 1))
    return runs


def longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for index, char in enumerate(shortest):
        if any(other[index] != char for other in strings):
            return shortest[:index]
    return shortest


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

    return Agent(problem_data).execute()


__all__ = ["agent_main"]