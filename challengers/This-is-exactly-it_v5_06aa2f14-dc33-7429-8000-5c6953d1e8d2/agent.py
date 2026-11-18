
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from os import getenv
from typing import Any, Callable, Iterable, Sequence

try:
    from src.agent.proxy_client import ProxyClient
except Exception:
    from proxy_client import ProxyClient



__all__ = ["agent_main", "Settings", "EpisodeController", "build_controller"]


_DEFAULT_MODELS = {
    "chutes": "mistralai/mistral-small-2603",
    "openrouter": "mistralai/mistral-small-2603",
}


def _default_model() -> str:
    override = getenv("ORO_MODEL")
    if override and override.strip():
        return override.strip()
    provider = getenv("INFERENCE_PROVIDER") or "chutes"
    return _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["chutes"])


def _env_flag(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = getenv(name)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:

    model: str = field(default_factory=_default_model)
    temperature: float = 0.0

    max_query_proposals: int = 3
    max_expansions: int = 4
    stale_expansions_before_stop: int = 2
    use_hypothetical_query: bool = True
    use_grounded_reformulation: bool = True

    inspect_batch: int = 5
    max_shortlist: int = 12
    min_pool_before_commit: int = 3

    max_shopper_replies: int = 8
    max_salvage_actions: int = 6

    use_order_sensitivity_check: bool = False
    use_candidate_votes: bool = False
    use_rival_duel: bool = False

    reserve_turns: int = 6
    uncommitted_reserve_turns: int = 3
    wall_clock_budget_s: float = 240.0
    hard_deadline_s: float = 285.0
    min_request_s: int = 15

    max_json_repairs: int = 1
    max_inference_calls: int = 24
    max_output_tokens: dict[str, int] = field(
        default_factory=lambda: {
            "requirement extraction": 1200,
            "shopper update and reply": 600,
            "candidate assessment": 2600,
            "preference selection": 900,
            "query reformulation": 400,
            "hypothetical query": 250,
            "rival duel": 250,
            "default": 900,
        }
    )
    assumed_inference_s: float = 17.0
    latency_estimate_share: float = 0.35

    max_dialogue_bytes: int = 256 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls()
        settings.use_order_sensitivity_check = _env_flag("ORO_REVIEW_ORDER", False)
        settings.use_candidate_votes = _env_flag("ORO_REVIEW_VOTES", False)
        settings.use_rival_duel = _env_flag("ORO_REVIEW_DUEL", False)
        settings.use_hypothetical_query = _env_flag("ORO_HYDE", True)
        settings.use_grounded_reformulation = _env_flag("ORO_PRF", True)
        settings.reserve_turns = _env_int("ORO_RESERVE_TURNS", settings.reserve_turns)
        settings.max_shopper_replies = _env_int(
            "ORO_MAX_REPLIES", settings.max_shopper_replies
        )
        settings.max_salvage_actions = _env_int(
            "ORO_MAX_SALVAGE", settings.max_salvage_actions
        )
        return settings


_NUM = r"[0-9]+(?:[.,][0-9]+)?"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MONEY_RE = re.compile(_NUM)
_INCH_RE = re.compile(_NUM + r"""\s*(?:"|''|”|inch(?:es)?\b|in\b|-inch\b)""")
_CM_RE = re.compile(_NUM + r"\s*cm\b")
_HZ_RE = re.compile(_NUM + r"\s*hz\b")
_WXH_RE = re.compile("(" + _NUM + r")\s*(?:x|×|by|\*)\s*(" + _NUM + ")")
_GB_RE = re.compile(_NUM + r"\s*gb\b")
_MB_RE = re.compile(_NUM + r"\s*mb\b")
_LEAD_NUM_RE = re.compile("^" + _NUM)

_RESOLUTION_CLASSES: tuple[tuple[str, int, int], ...] = (
    ("4k", 3840, 2160),
    ("qhd", 2560, 1440),
    ("fhd", 1920, 1080),
    ("hd", 1280, 720),
)
_CLASS_ALIASES: tuple[tuple[str, str], ...] = (
    ("ultra hd", "4k"),
    ("uhd", "4k"),
    ("4k", "4k"),
    ("2160p", "4k"),
    ("wqhd", "qhd"),
    ("qhd", "qhd"),
    ("1440p", "qhd"),
    ("full hd", "fhd"),
    ("fhd", "fhd"),
    ("1080p", "fhd"),
    ("720p", "hd"),
    ("hd", "hd"),
)
CLASS_ORDER = {"hd": 1, "fhd": 2, "qhd": 3, "4k": 4}

_BOOLEAN_WORDS = frozenset({"true", "false", "yes", "no", "y", "n", "1", "0"})


def tokens(text: Any) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def to_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    match = _MONEY_RE.search(str(value).replace(",", ""))
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def _numbers_before(pattern: re.Pattern[str], text: str) -> list[Decimal]:
    out: list[Decimal] = []
    for match in pattern.finditer(text):
        lead = _LEAD_NUM_RE.match(match.group(0))
        if not lead:
            continue
        value = to_decimal(lead.group(0))
        if value is not None:
            out.append(value)
    return out


def _dedupe_decimals(values: Iterable[Decimal]) -> list[Decimal]:
    out: list[Decimal] = []
    for value in values:
        if all(abs(value - seen) > Decimal("0.001") for seen in out):
            out.append(value)
    return out


def parse_inches(value: Any) -> list[Decimal]:
    text = str(value or "").lower()
    found = _numbers_before(_INCH_RE, text)
    found += [
        (raw / Decimal("2.54")).quantize(Decimal("0.01"))
        for raw in _numbers_before(_CM_RE, text)
    ]
    if not found and not _WXH_RE.search(text):
        stripped = text.strip()
        if re.fullmatch(_NUM, stripped):
            parsed = to_decimal(stripped)
            if parsed is not None:
                found.append(parsed)
    return _dedupe_decimals(found)


def parse_hz(value: Any) -> list[Decimal]:
    return _dedupe_decimals(_numbers_before(_HZ_RE, str(value or "").lower()))


def parse_ram_gb(value: Any) -> list[Decimal]:
    text = str(value or "").lower()
    found = _numbers_before(_GB_RE, text)
    found += [
        (raw / Decimal(1024)).quantize(Decimal("0.01"))
        for raw in _numbers_before(_MB_RE, text)
    ]
    return _dedupe_decimals(found)


def resolution_class(width: Any, height: Any) -> str | None:
    dec_w, dec_h = to_decimal(width), to_decimal(height)
    if dec_w is None or dec_h is None:
        return None
    long_side, short_side = max(dec_w, dec_h), min(dec_w, dec_h)
    for name, need_long, need_short in _RESOLUTION_CLASSES:
        if long_side >= need_long and short_side >= need_short:
            return name
    return None


def parse_resolution_classes(value: Any) -> list[str]:
    raw_text = str(value or "")
    text = raw_text.lower()
    out: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in out:
            out.append(name)

    for width, height in _WXH_RE.findall(text):
        dec_w, dec_h = to_decimal(width), to_decimal(height)
        if dec_w is not None and dec_h is not None and dec_w >= 100 and dec_h >= 100:
            add(resolution_class(dec_w, dec_h))

    has_upper_qhd = re.search(r"\bQHD\b", raw_text) is not None
    for alias, name in _CLASS_ALIASES:
        if not re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text):
            continue
        if alias == "qhd" and not has_upper_qhd:
            continue
        add(name)
    return out


def truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in {"true", "yes", "y", "1", "available", "in stock", "in_stock"}:
        return True
    if text in {"false", "no", "n", "0", "unavailable", "out of stock"}:
        return False
    return None


def compact(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "screen_size": ("size", "screen_size", "display_size", "screen"),
    "resolution": ("display", "resolution", "screen_resolution", "display_resolution"),
    "refresh_rate": ("refresh_rate", "refresh", "hz"),
    "ram": ("ram", "memory"),
    "storage": ("storage", "rom", "internal_storage"),
    "battery": ("battery", "battery_capacity"),
    "color": ("color", "colour", "color_family"),
    "model": ("model", "model_number", "model_name"),
    "operating_system": (
        "device_operating_system",
        "operating_system",
        "description_operating_system",
        "os",
    ),
    "device_type": ("phone_type", "device_type", "product_type", "type"),
    "material": ("material",),
    "condition": ("condition",),
}

NUMERIC_FIELDS = {"screen_size", "refresh_rate", "ram", "storage", "price"}

_UNSEARCHABLE_FIELDS = frozenset({"price", "in_stock", "category"})

UNIT_PARSERS: dict[str, Callable[[Any], list[Decimal]]] = {
    "screen_size": parse_inches,
    "refresh_rate": parse_hz,
    "ram": parse_ram_gb,
    "storage": parse_ram_gb,
}

_FIELD_ALIASES = {
    "screen": "screen_size",
    "screen_inches": "screen_size",
    "screensize": "screen_size",
    "display_size": "screen_size",
    "refreshrate": "refresh_rate",
    "hz": "refresh_rate",
    "budget": "price",
    "cost": "price",
    "stock": "in_stock",
    "availability": "in_stock",
    "instock": "in_stock",
    "manufacturer": "brand",
    "category_path": "category",
}


def canonical_field(name: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    if not key:
        return ""
    if key in {"price", "in_stock", "brand", "category"}:
        return key
    for semantic, sources in FIELD_SOURCES.items():
        if key == semantic:
            return semantic
        for source in sources:
            if key == re.sub(r"[^a-z0-9]+", "_", source).strip("_"):
                return semantic
    return _FIELD_ALIASES.get(key, key)


@dataclass
class Requirement:

    rid: str
    field: str
    operator: str
    value: Any
    unit: str | None = None
    strength: str = "hard"
    scope: str = "selected_variant"
    source_message: int = 0
    source_text: str = ""
    revision: int = 0
    active: bool = True
    priority: int = 0

    @property
    def is_hard(self) -> bool:
        return self.active and self.strength == "hard"


@dataclass
class CandidateRestriction:

    product_id: str
    sku: str | None = None
    explicit: bool = False
    source_text: str = ""

    @property
    def text(self) -> str:
        return self.product_id + "::" + self.sku if self.sku else self.product_id

    def permits(self, product_id: Any, sku: Any = None) -> bool:
        if str(product_id) != self.product_id:
            return False
        if self.sku is None:
            return True
        return sku is not None and str(sku) == self.sku

    @classmethod
    def parse(cls, raw: Any, *, explicit: bool, source_text: str = ""):
        text = str(raw or "").strip()
        if not text:
            return None
        if "::" in text:
            product_id, _sep, sku = text.partition("::")
            product_id, sku = product_id.strip(), sku.strip()
            if not product_id or not sku:
                return None
            return cls(product_id, sku, explicit, source_text)
        return cls(text, None, explicit, source_text)


@dataclass
class PurchaseBlocker:

    field: str
    reason: str
    remedy: str = ""
    target: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "reason": self.reason,
            "remedy": self.remedy,
            "target": self.target,
        }


@dataclass
class ObservationOutcome:

    action: str
    ok: bool
    summary: str
    product_id: str | None = None
    sku: str | None = None
    fields: tuple[str, ...] = ()
    error: Any = None

    def confirms(self, product_id: Any, sku: Any, *required: str) -> bool:
        if not self.ok or self.error:
            return False
        if self.product_id is None or str(self.product_id) != str(product_id):
            return False
        if self.sku is None or str(self.sku) != str(sku):
            return False
        return all(name in self.fields for name in required)


@dataclass
class Objective:

    field: str
    direction: str
    rank: int


@dataclass
class Intent:
    requirements: list[Requirement] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    candidate_restrictions: list[str] = field(default_factory=list)
    restriction_records: list[CandidateRestriction] = field(default_factory=list)
    permitted_substitutions: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    explanation_required: bool = False
    explanation_fields: list[str] = field(default_factory=list)
    title_terms: list[str] = field(default_factory=list)
    product_identity: str = ""
    category: str | None = None
    search_terms: list[str] = field(default_factory=list)
    noisy_terms: list[str] = field(default_factory=list)
    min_distinct_candidates: int = 0
    revision: int = 0
    notes: list[str] = field(default_factory=list)

    def hard(self) -> list[Requirement]:
        return [r for r in self.requirements if r.is_hard]

    def preferences(self) -> list[Requirement]:
        return [r for r in self.requirements if r.active and r.strength == "preference"]

    def by_field(self, name: str) -> list[Requirement]:
        return [r for r in self.requirements if r.active and r.field == name]

    def set_restrictions(
        self, records: Sequence[CandidateRestriction]
    ) -> None:
        self.restriction_records = list(records)
        self.candidate_restrictions = [record.text for record in records]

    def effective_restrictions(self) -> list[CandidateRestriction]:
        texts = [str(item) for item in self.candidate_restrictions]
        known = {record.text: record for record in self.restriction_records}
        if set(texts) == set(known):
            return list(self.restriction_records)
        rebuilt: list[CandidateRestriction] = []
        for text in texts:
            record = known.get(text) or CandidateRestriction.parse(
                text, explicit=True, source_text=text
            )
            if record is not None:
                rebuilt.append(record)
        self.restriction_records = rebuilt
        return rebuilt

    def has_explicit_restrictions(self) -> bool:
        return any(record.explicit for record in self.effective_restrictions())

    def budget(self) -> Decimal | None:
        caps = [
            to_decimal(r.value)
            for r in self.requirements
            if r.active and r.field == "price" and r.operator in {"le", "lt"}
        ]
        real = [cap for cap in caps if cap is not None]
        return min(real) if real else None


@dataclass
class EvidenceFact:
    product_id: str
    sku: str | None
    field: str
    raw_value: Any
    normalized: Any
    source_action: str
    source_path: str
    source_kind: str
    observed_at: int
    stale: bool = False

    @property
    def admissible_for_claim(self) -> bool:
        if self.field in {"price", "in_stock"}:
            return self.source_kind == "current_variant_state"
        if self.field == "category":
            return self.source_kind == "category_path"
        if self.field == "brand":
            return self.source_kind == "brand"
        return self.source_kind in {"raw_product_fact", "variant_option"}


@dataclass
class ConstraintAssessment:
    rid: str
    state: str
    sources: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class CandidateState:
    product_id: str
    title: str = ""
    brand: Any = None
    category_path: list[str] = field(default_factory=list)
    listing_id: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=list)
    inspected: bool = False
    retrieval_score: float = 0.0
    best_rank: int = 10_000
    seen_queries: set = field(default_factory=set)
    assessments: dict[str, ConstraintAssessment] = field(default_factory=dict)
    rejected_reason: str | None = None
    selected_sku: str | None = None

    def variant(self, sku: Any) -> dict[str, Any] | None:
        if sku is None:
            return self.variants[0] if len(self.variants) == 1 else None
        for row in self.variants:
            if str(row.get("sku")) == str(sku):
                return row
        return None

    def key(self, sku: Any) -> str:
        return f"{self.product_id}::{sku}" if sku else self.product_id


@dataclass
class ActionProposal:
    purpose: str
    name: str
    args: dict[str, Any]
    resolves: list[str] = field(default_factory=list)
    target: str | None = None


@dataclass
class RetrievalMetrics:

    queries_issued: int = 0
    productive_queries: int = 0
    results_seen: int = 0
    duplicate_results: int = 0
    distinct_products: int = 0
    matching_products: int = 0
    inspected_products: int = 0
    inspected_matching: int = 0
    expansions: int = 0
    stale_expansions: int = 0

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "queries_issued": self.queries_issued,
            "productive_queries": self.productive_queries,
            "results_seen": self.results_seen,
            "duplicate_results": self.duplicate_results,
            "distinct_products": self.distinct_products,
            "matching_products": self.matching_products,
            "inspected_products": self.inspected_products,
            "inspected_matching": self.inspected_matching,
            "expansions": self.expansions,
            "stale_expansions": self.stale_expansions,
        }
        if self.queries_issued:
            data["observed_query_efficiency"] = round(
                self.productive_queries / self.queries_issued, 3
            )
        if self.inspected_products:
            data["observed_shortlist_precision"] = round(
                self.inspected_matching / self.inspected_products, 3
            )
        return data


class AgentError(RuntimeError):
    pass


class WorkBudget:

    def __init__(self, max_steps: int, max_calls_per_turn: int, settings: Settings):
        self.max_steps = max(1, int(max_steps))
        self.max_calls_per_turn = max(1, int(max_calls_per_turn))
        self.settings = settings
        self.turn = 0
        self.actions = 0
        self.started = time.monotonic()
        self.inference_calls = 0
        self.inference_seconds: list[float] = []
        self.refused_calls = 0
        self.truncated_calls = 0

    @property
    def turns_left(self) -> int:
        return max(0, self.max_steps - self.turn)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def reserve(self, *, committed: bool = True) -> int:
        wanted = (
            self.settings.reserve_turns
            if committed
            else self.settings.uncommitted_reserve_turns
        )
        return min(wanted, max(1, self.max_steps // 3))

    def optional_work_allowed(self, *, committed: bool = True) -> bool:
        if self.turns_left <= self.reserve(committed=committed):
            return False
        return self.elapsed < self.settings.wall_clock_budget_s

    def must_finish_now(self) -> bool:
        return self.turns_left <= 2 or self.elapsed >= self.settings.wall_clock_budget_s

    def time_left(self) -> float:
        return self.settings.hard_deadline_s - self.elapsed

    def record_inference(self, seconds: float) -> None:
        self.inference_seconds.append(max(0.0, float(seconds)))

    def typical_inference_s(self) -> float:
        observed = list(self.inference_seconds)
        if not observed:
            estimate = self.settings.assumed_inference_s
        elif len(observed) < 3:
            estimate = (sum(observed) + self.settings.assumed_inference_s) / (
                len(observed) + 1
            )
        else:
            estimate = sorted(observed)[len(observed) // 2]
        ceiling = self.settings.hard_deadline_s * self.settings.latency_estimate_share
        return float(min(estimate, ceiling))

    def can_afford(self, expected: float, *, reserve: float) -> bool:
        return (self.time_left() - reserve) >= expected

    def clamp(self, proxy: Any, *, reserve: float) -> bool:
        budget = self.time_left() - reserve
        if budget < self.settings.min_request_s:
            return False
        base_timeout = getattr(proxy, "_base_timeout", None)
        if base_timeout is None:
            base_timeout = int(getattr(proxy, "timeout", 120) or 120)
            proxy._base_timeout = base_timeout
        base_retries = getattr(proxy, "_base_retries", None)
        if base_retries is None:
            base_retries = int(getattr(proxy, "max_retries", 1) or 1)
            proxy._base_retries = base_retries
        timeout = int(max(self.settings.min_request_s, min(base_timeout, budget)))
        proxy.timeout = timeout
        proxy.max_retries = base_retries if budget >= timeout * 2 else 1
        return True

    def inference_allowed(self) -> bool:
        return self.inference_calls < self.settings.max_inference_calls

    def note_turn(self, action_count: int) -> None:
        self.turn += 1
        self.actions += action_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_calls_per_turn": self.max_calls_per_turn,
            "turns_used": self.turn,
            "turns_left": self.turns_left,
            "actions": self.actions,
            "inference_calls": self.inference_calls,
            "elapsed_s": round(self.elapsed, 2),
            "time_left_s": round(self.time_left(), 2),
            "inference_seconds_total": round(sum(self.inference_seconds), 1),
            "inference_seconds_median": round(self.typical_inference_s(), 1),
            "refused_calls": self.refused_calls,
            "truncated_calls": self.truncated_calls,
        }


class ToolCatalog:

    _COUNT_IN_TEXT = re.compile(r"(?:up to|at most|capped at|maximum of)\s+(\d+)")

    def __init__(self, tools: Sequence[dict[str, Any]] | None):
        self.schemas: dict[str, dict[str, Any]] = {}
        self.descriptions: dict[str, str] = {}
        for entry in tools or []:
            if not isinstance(entry, dict):
                continue
            function = entry.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            parameters = function.get("parameters")
            self.schemas[name] = parameters if isinstance(parameters, dict) else {}
            self.descriptions[name] = str(function.get("description") or "")


    def has(self, name: str) -> bool:
        return name in self.schemas

    def names(self) -> list[str]:
        return sorted(self.schemas)

    def properties(self, name: str) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties")
        return props if isinstance(props, dict) else {}

    def supports_argument(self, name: str, argument: str) -> bool:
        return argument in self.properties(name)

    def numeric_cap(self, name: str, argument: str, fallback: int) -> int:
        prop = self.properties(name).get(argument)
        if isinstance(prop, dict):
            maximum = prop.get("maximum")
            if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
                return int(maximum)
            described = self._COUNT_IN_TEXT.search(str(prop.get("description") or ""))
            if described:
                return int(described.group(1))
        described = self._COUNT_IN_TEXT.search(self.descriptions.get(name, ""))
        if described:
            return int(described.group(1))
        return fallback

    def array_cap(self, name: str, argument: str, fallback: int) -> int:
        prop = self.properties(name).get(argument)
        if isinstance(prop, dict):
            max_items = prop.get("maxItems")
            if isinstance(max_items, int) and not isinstance(max_items, bool):
                return max_items
        described = self._COUNT_IN_TEXT.search(self.descriptions.get(name, ""))
        if described:
            return int(described.group(1))
        return fallback


    def validate(self, name: str, args: dict[str, Any]) -> list[str]:
        if name not in self.schemas:
            return ["tool " + repr(name) + " is not published by this episode"]
        if not isinstance(args, dict):
            return ["arguments must be an object"]
        return self._validate_object(self.schemas[name], args, name)

    def _validate_object(
        self, schema: dict[str, Any], value: dict[str, Any], path: str
    ) -> list[str]:
        problems: list[str] = []
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        for key in required:
            if key not in value:
                problems.append(path + ": missing required property " + repr(key))
        if properties or schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    problems.append(path + ": property " + repr(key) + " not published")
        for key, item in value.items():
            prop = properties.get(key)
            if isinstance(prop, dict):
                problems.extend(self._validate_value(prop, item, path + "." + str(key)))
        return problems

    def _validate_value(
        self, schema: dict[str, Any], value: Any, path: str
    ) -> list[str]:
        problems: list[str] = []
        expected = schema.get("type")
        if isinstance(expected, str) and not self._type_ok(expected, value):
            problems.append(
                path + ": expected " + expected + ", got " + type(value).__name__
            )
            return problems
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            problems.append(path + ": " + repr(value) + " is outside the published enum")
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and isinstance(value, (int, float)):
            if value < minimum:
                problems.append(path + ": violates minimum " + str(minimum))
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and isinstance(value, (int, float)):
            if value > maximum:
                problems.append(path + ": violates maximum " + str(maximum))
        if isinstance(value, list):
            min_items = schema.get("minItems")
            if isinstance(min_items, int) and len(value) < min_items:
                problems.append(path + ": violates minItems " + str(min_items))
            max_items = schema.get("maxItems")
            if isinstance(max_items, int) and len(value) > max_items:
                problems.append(path + ": violates maxItems " + str(max_items))
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    problems.extend(
                        self._validate_value(
                            item_schema, item, path + "[" + str(index) + "]"
                        )
                    )
        if isinstance(value, dict) and expected in (None, "object"):
            problems.extend(self._validate_object(schema, value, path))
        return problems

    @staticmethod
    def _type_ok(expected: str, value: Any) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return True


class EvidenceStore:

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str | None, str], list[EvidenceFact]] = {}
        self.sequence = 0

    def tick(self) -> int:
        self.sequence += 1
        return self.sequence

    def add(self, fact: EvidenceFact) -> EvidenceFact:
        key = (fact.product_id, fact.sku, fact.field)
        bucket = self._facts.setdefault(key, [])
        for previous in bucket:
            if previous.source_kind == fact.source_kind:
                previous.stale = True
        bucket.append(fact)
        return fact

    def record(
        self,
        product_id: str,
        sku: str | None,
        field_name: str,
        raw_value: Any,
        *,
        source_action: str,
        source_path: str,
        source_kind: str,
        normalized: Any = None,
        observed_at: int | None = None,
    ) -> EvidenceFact:
        return self.add(
            EvidenceFact(
                product_id=str(product_id),
                sku=None if sku is None else str(sku),
                field=field_name,
                raw_value=raw_value,
                normalized=normalized,
                source_action=source_action,
                source_path=source_path,
                source_kind=source_kind,
                observed_at=self.sequence if observed_at is None else observed_at,
            )
        )

    def facts(
        self,
        product_id: str,
        field_name: str,
        *,
        sku: str | None = None,
        include_parent: bool = True,
        fresh_only: bool = True,
    ) -> list[EvidenceFact]:
        out: list[EvidenceFact] = []
        scopes: list[str | None] = []
        if sku is not None:
            scopes.append(str(sku))
        if include_parent:
            scopes.append(None)
        for scope in scopes:
            for fact in self._facts.get((str(product_id), scope, field_name), []):
                if fresh_only and fact.stale:
                    continue
                out.append(fact)
        return out

    def invalidate(self, product_id: str, sku: str | None, fields: Iterable[str]) -> int:
        touched = 0
        for name in fields:
            for scope in ({None, str(sku)} if sku is not None else {None}):
                for fact in self._facts.get((str(product_id), scope, name), []):
                    if not fact.stale:
                        fact.stale = True
                        touched += 1
        return touched


class EnvironmentClient:

    def __init__(
        self,
        proxy: Any,
        binding: dict[str, Any],
        problem_id: str,
        *,
        max_transport_retries: int = 1,
        budget: "WorkBudget | None" = None,
    ):
        if not isinstance(binding, dict) or not binding:
            raise AgentError("environment binding is missing")
        self.proxy = proxy
        self.binding = dict(binding)
        self.problem_id = problem_id
        self.max_transport_retries = max_transport_retries
        self.budget = budget
        self.turn = 0
        self.done = False
        self.sent_groups: list[str] = []
        self._last_call_ids: list[str] = []

    def call(self, proposals: Sequence[ActionProposal]) -> dict[str, Any]:
        if self.done:
            raise AgentError("environment already reported done=true")
        if not proposals:
            raise AgentError("a solver turn needs at least one action")
        self.turn += 1
        group_id = self.problem_id + "-turn-" + str(self.turn)
        self.sent_groups.append(group_id)
        self._last_call_ids = [
            group_id + "-" + str(index) for index in range(1, len(proposals) + 1)
        ]
        envelope = dict(self.binding)
        envelope.update(
            {
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": self.turn,
                "calls": [
                    {
                        "call_id": call_id,
                        "action": {"name": item.name, "args": item.args},
                    }
                    for call_id, item in zip(self._last_call_ids, proposals)
                ],
            }
        )

        result = None
        for _attempt in range(self.max_transport_retries + 1):
            if self.budget is not None and not self.budget.clamp(
                self.proxy, reserve=0.0
            ):
                break
            result = self.proxy.post("/environment/call", json_data=envelope)
            if isinstance(result, dict):
                break
        if not isinstance(result, dict):
            raise AgentError("environment call failed after bounded retries")

        calls = result.get("calls")
        if not isinstance(calls, list) or len(calls) != len(proposals):
            raise AgentError("environment response does not align with the request")
        returned = [
            entry.get("call_id") if isinstance(entry, dict) else None
            for entry in calls
        ]
        if sorted(str(item) for item in returned) != sorted(self._last_call_ids):
            raise AgentError(
                "environment response call ids do not match the request"
            )
        return result

    def pair(
        self, proposals: Sequence[ActionProposal], result: dict[str, Any]
    ) -> list[tuple[ActionProposal, dict[str, Any]]]:
        by_id = {
            str(entry.get("call_id")): entry
            for entry in (result.get("calls") or [])
            if isinstance(entry, dict)
        }
        out: list[tuple[ActionProposal, dict[str, Any]]] = []
        for call_id, proposal in zip(self._last_call_ids, proposals):
            entry = by_id.get(call_id)
            if entry is None:
                raise AgentError("environment response is missing " + call_id)
            out.append((proposal, self.normalise(entry.get("observation"))))
        return out

    @staticmethod
    def normalise(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return {"done": False, "error": "malformed observation", "payload": {}}
        payload = observation.get("observation")
        return {
            "done": bool(observation.get("done")),
            "error": observation.get("error"),
            "payload": payload if isinstance(payload, dict) else {},
            "raw": observation,
        }


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    match = _JSON_BLOCK.search(stripped)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


class InferenceClient:

    completion_reserve_s = 25.0

    def __init__(self, proxy: Any, settings: Settings, budget: WorkBudget):
        self.proxy = proxy
        self.settings = settings
        self.budget = budget
        self.failures = 0

    def ask_json(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        validate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any] | None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        cap = self.settings.max_output_tokens.get(
            purpose, self.settings.max_output_tokens.get("default", 800)
        )
        for attempt in range(self.settings.max_json_repairs + 1):
            if not self.budget.inference_allowed():
                return None
            expected = self.budget.typical_inference_s()
            if not self.budget.can_afford(expected, reserve=self.completion_reserve_s):
                self.budget.refused_calls += 1
                return None
            if not self.budget.clamp(self.proxy, reserve=self.completion_reserve_s):
                self.budget.refused_calls += 1
                return None
            self.budget.inference_calls += 1
            started = time.monotonic()
            response = self.proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": self.settings.model,
                    "messages": messages,
                    "temperature": self.settings.temperature,
                    "max_tokens": cap,
                },
            )
            self.budget.record_inference(time.monotonic() - started)
            content = self._content(response)
            parsed = extract_json_object(content)
            if parsed is not None and (validate is None or validate(parsed)):
                return parsed
            self.failures += 1
            if self._truncated(response):
                self.budget.truncated_calls += 1
                cap = min(cap * 2, 4000)
                if not self.budget.can_afford(
                    self.budget.typical_inference_s(),
                    reserve=self.completion_reserve_s,
                ):
                    return None
            if attempt >= self.settings.max_json_repairs:
                return None
            messages = messages + [
                {"role": "assistant", "content": content or ""},
                {
                    "role": "user",
                    "content": (
                        "That reply could not be used for "
                        + purpose
                        + ". Reply again with one JSON object only, no prose and "
                        "no code fence, exactly matching the requested schema."
                    ),
                },
            ]
        return None

    @staticmethod
    def _truncated(response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        first = choices[0]
        if not isinstance(first, dict):
            return False
        return str(first.get("finish_reason") or "").lower() == "length"

    @staticmethod
    def _content(response: Any) -> str | None:
        if not isinstance(response, dict):
            return None
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        return content if isinstance(content, str) else None


VALID_OPERATORS = {
    "eq",
    "ne",
    "le",
    "lt",
    "ge",
    "gt",
    "contains",
    "not_contains",
    "exists",
    "one_of",
}
VALID_STRENGTHS = {"hard", "preference", "unresolved"}
VALID_SCOPES = {"selected_variant", "product", "listing"}


class ConstraintNormalizer:

    @staticmethod
    def normalize_operator(raw: Any, value_text: str = "") -> str:
        text = str(raw or "").strip().lower()
        table = {
            "=": "eq",
            "==": "eq",
            "equals": "eq",
            "equal": "eq",
            "is": "eq",
            "!=": "ne",
            "<=": "le",
            "at most": "le",
            "max": "le",
            "under": "lt",
            "<": "lt",
            ">=": "ge",
            "at least": "ge",
            "min": "ge",
            "minimum": "ge",
            ">": "gt",
            "over": "gt",
            "has": "contains",
            "includes": "contains",
        }
        if text in VALID_OPERATORS:
            return text
        if text in table:
            return table[text]
        if not text and value_text:
            return "eq"
        return ""

    @staticmethod
    def wanted_values(requirement: "Requirement") -> list[Decimal] | None:
        text = str(requirement.value)
        if requirement.unit:
            text = text + " " + str(requirement.unit)
        found = ConstraintNormalizer.values_for(requirement.field, text)
        if found:
            return found
        return ConstraintNormalizer.values_for(
            requirement.field, str(requirement.value)
        )

    @staticmethod
    def values_for(field_name: str, raw: Any) -> list[Decimal] | None:
        parser = UNIT_PARSERS.get(field_name)
        if parser is not None:
            found = parser(raw)
            return found or None
        if field_name in NUMERIC_FIELDS:
            single = to_decimal(raw)
            return [single] if single is not None else None
        return None

    @staticmethod
    def compare_numeric(operator: str, observed: Decimal, wanted: Decimal) -> bool:
        if operator == "eq":
            return observed == wanted
        if operator == "ne":
            return observed != wanted
        if operator == "le":
            return observed <= wanted
        if operator == "lt":
            return observed < wanted
        if operator == "ge":
            return observed >= wanted
        if operator == "gt":
            return observed > wanted
        return False

    @staticmethod
    def mentions(observed: Any, terms: Sequence[str], *, prefix_from: int = 4) -> bool:
        observed_tokens = tokens(observed)
        for term in terms:
            term = str(term).lower()
            if not term:
                continue
            if term in observed_tokens:
                return True
            if len(term) >= prefix_from and any(
                token.startswith(term) for token in observed_tokens
            ):
                return True
        return False

    @staticmethod
    def compare_text(operator: str, observed: Any, wanted: Any) -> bool | None:
        observed_tokens = tokens(observed)
        wanted_tokens = tokens(wanted)
        if not wanted_tokens:
            return bool(observed_tokens) if operator == "exists" else None
        joined_observed = " ".join(observed_tokens)
        joined_wanted = " ".join(wanted_tokens)
        contained = joined_wanted in joined_observed or all(
            term in observed_tokens for term in wanted_tokens
        )
        if operator in {"eq", "contains", "one_of", "exists"}:
            return True if (contained or joined_observed == joined_wanted) else False
        if operator in {"ne", "not_contains"}:
            return not contained
        return None


class CoverageScorer:

    @staticmethod
    def _searchable(candidate: CandidateState, sku: Any = None) -> str:
        parts = [str(candidate.title or ""), str(candidate.brand or "")]
        parts.extend(str(part) for part in candidate.category_path)
        for key, value in (candidate.facts or {}).items():
            parts.append(str(key) + " " + str(value))
        variant = candidate.variant(sku)
        rows = [variant] if variant else candidate.variants
        for row in rows or []:
            parts.append(str((row or {}).get("options") or ""))
        return " ".join(parts).lower()

    @classmethod
    def score(
        cls,
        candidate: CandidateState,
        requirements: Sequence[Requirement],
        sku: Any = None,
    ) -> tuple[float, list[str]]:
        text = cls._searchable(candidate, sku)
        text_tokens = set(tokens(text))
        covered: list[str] = []
        considered = 0
        for requirement in requirements:
            if requirement.field in {"price", "in_stock"}:
                continue
            considered += 1
            wanted = ConstraintNormalizer.values_for(
                requirement.field, requirement.value
            )
            if wanted:
                observed = ConstraintNormalizer.values_for(requirement.field, text)
                observed = observed or []
                if any(
                    ConstraintNormalizer.compare_numeric(
                        requirement.operator, value, wanted[0]
                    )
                    for value in observed
                ):
                    covered.append(requirement.rid)
                    continue
            want_tokens = tokens(requirement.value)
            if want_tokens and all(term in text_tokens for term in want_tokens):
                covered.append(requirement.rid)
                continue
            joined = "".join(want_tokens)
            if joined and joined in text_tokens:
                covered.append(requirement.rid)
        ratio = len(covered) / considered if considered else 0.0
        return ratio, covered


class CandidatePrioritizer:

    @staticmethod
    def rank_for_inspection(
        candidates: Sequence[CandidateState],
        requirements: Sequence[Requirement],
        *,
        limit: int,
    ) -> list[CandidateState]:
        scored: list[tuple[float, float, float, CandidateState]] = []
        for candidate in candidates:
            if candidate.rejected_reason:
                continue
            ratio, _covered = CoverageScorer.score(candidate, requirements)
            scored.append(
                (
                    0.0 if candidate.inspected else 1.0,
                    ratio,
                    candidate.retrieval_score,
                    candidate,
                )
            )
        scored.sort(key=lambda row: (-row[0], -row[1], -row[2]))
        return [row[3] for row in scored[:limit]]


_ID_PAIR_RE = re.compile(
    r"\b[A-Za-z0-9]{5,}::[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)
_QUOTED_RE = re.compile(r"[\"“‘']([^\"”’']{2,60})[\"”’']")
_BUDGET_TRAILING_RE = re.compile(
    r"(" + _NUM + r")\s*[A-Za-z]{0,4}\s*(?:max(?:imum)?|or less|at most|tops)\b",
    re.IGNORECASE,
)
_SCOPE_CUE_RE = re.compile(
    r"(compare only|only these|only from|choose from|pick from|from this list|"
    r"candidate variants|these candidates|restricted to)",
    re.IGNORECASE,
)
_MIN_DISTINCT_RE = re.compile(
    r"(?:surface|observe|find|cover)\s+at least\s+(\d+)\s+distinct", re.IGNORECASE
)
_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|no more than|not more than|up to|under|below|"
    r"less than|within|budget of|budget is|spend(?:\s+up\s+to)?|max(?:imum)?)"
    r"\s*(?:of\s*)?[^0-9]{0,12}?(" + _NUM + r")",
    re.IGNORECASE,
)

INTENT_SYSTEM = (
    "You convert a shopper request plus its public task rules into a structured "
    "requirement record for a shopping agent. You never invent a value that the "
    "text does not state. Reply with exactly one JSON object and no prose."
)

INTENT_TEMPLATE = """Shopper request and public task rules:
<<<
{query}
>>>

Available tools: {tool_names}

Return one JSON object with exactly these keys:
{{
  "product_identity": "<short noun phrase for the item being bought, e.g. the product kind>",
  "category": "<catalog category the rules require, or null>",
  "requirements": [
    {{"field": "<screen_size|resolution|refresh_rate|ram|storage|battery|color|model|brand|category|price|in_stock|operating_system|device_type|other snake_case name>",
      "operator": "eq|ne|le|lt|ge|gt|contains|not_contains|exists",
      "value": "<literal value exactly as stated, numbers without units>",
      "unit": "<unit or null>",
      "strength": "hard|preference",
      "scope": "selected_variant|product|listing",
      "source_text": "<the exact span of the request this came from>"}}
  ],
  "objectives": [
    {{"field": "<field>", "direction": "max|min"}}
  ],
  "candidate_restrictions": ["<exact candidate identifier the request permits>"],
  "permitted_substitutions": ["<relaxation the shopper allows, most preferred first>"],
  "title_terms": ["<term the request requires to appear literally in the listing title>"],
  "min_distinct_candidates": <integer the rules require you to surface before ordering, else 0>,
  "search_terms": ["<2 to 5 word catalog search phrase>"],
  "noisy_terms": ["<word from the request that would hurt a catalog search>"],
  "explanation_required": <true if the shopper asked for a reason or justification, else false>,
  "explanation_fields": ["<field the explanation must establish>"],
  "open_questions": [
    {{"question": "<one short question to the shopper>", "decision_changing": <true|false>}}
  ]
}}

Rules for filling it in:
- A firm, non-negotiable condition is "hard". A stated liking, ideal or nice-to-have is "preference".
- Keep negations, comparison direction and units. An exact requirement is not a minimum requirement.
- "objectives" is only for an explicitly ordered trade-off sequence, most important first.
- "candidate_restrictions" is only for identifiers the request itself lists.
- "title_terms" is only for terms the rules require to appear in the title itself.
- "decision_changing" is true only when the answer would change which product can be bought.
- Use [] or null rather than guessing. Do not add keys."""


def _spans_message(span: Any, message: str) -> bool:
    text = " ".join(str(span or "").split()).strip().lower()
    if len(text) < 3:
        return False
    haystack = " ".join(str(message or "").split()).lower()
    return text in haystack


def _mentions_field(field_name: str, message: str) -> bool:
    words = set(tokens(message))
    if not words:
        return False
    if field_name in {"price"}:
        return extract_budget(message) is not None or bool(
            words & {"budget", "price", "cheaper", "spend", "afford", "cost"}
        )
    return bool(set(tokens(field_name.replace("_", " "))) & words)


def extract_budget(text: Any) -> Decimal | None:
    body = str(text or "")
    for pattern in (_BUDGET_RE, _BUDGET_TRAILING_RE):
        match = pattern.search(body)
        if match:
            amount = to_decimal(match.group(1))
            if amount is not None:
                return amount
    return None


def budget_span(text: Any) -> str:
    for pattern in (_BUDGET_RE, _BUDGET_TRAILING_RE):
        match = pattern.search(str(text or ""))
        if match:
            return match.group(0)
    return ""


class IntentParser:

    def __init__(self, inference: InferenceClient, catalog: ToolCatalog):
        self.inference = inference
        self.catalog = catalog


    def parse(self, query: str) -> Intent:
        intent = Intent()
        payload = self.inference.ask_json(
            INTENT_SYSTEM,
            INTENT_TEMPLATE.format(
                query=query[:6000], tool_names=", ".join(self.catalog.names())
            ),
            purpose="requirement extraction",
            validate=lambda js: isinstance(js.get("requirements"), list),
        )
        if payload:
            self._absorb(intent, payload, query)
        else:
            intent.notes.append("intent extraction unavailable; using text backstops")
        self._apply_text_backstops(intent, query)
        self._dedupe(intent)
        return intent


    def _absorb(self, intent: Intent, payload: dict[str, Any], query: str) -> None:
        intent.product_identity = str(payload.get("product_identity") or "").strip()
        category = payload.get("category")
        intent.category = str(category).strip() if isinstance(category, str) else None

        for index, raw in enumerate(payload.get("requirements") or []):
            requirement = self._requirement(raw, index, query)
            if requirement is not None:
                intent.requirements.append(requirement)

        for rank, raw in enumerate(payload.get("objectives") or []):
            if not isinstance(raw, dict):
                continue
            name = canonical_field(raw.get("field"))
            direction = str(raw.get("direction") or "").strip().lower()
            if name and direction in {"max", "min"}:
                intent.objectives.append(Objective(name, direction, rank))

        proposed = [
            str(item).strip()
            for item in (payload.get("candidate_restrictions") or [])
            if isinstance(item, (str, int)) and str(item).strip()
        ]
        intent.set_restrictions(
            [
                record
                for record in (
                    CandidateRestriction.parse(item, explicit=False, source_text=item)
                    for item in proposed
                )
                if record is not None
            ]
        )
        intent.permitted_substitutions = [
            str(item).strip()
            for item in (payload.get("permitted_substitutions") or [])
            if isinstance(item, str) and item.strip()
        ][:6]
        intent.title_terms = [
            token
            for item in (payload.get("title_terms") or [])
            if isinstance(item, str)
            for token in tokens(item)
        ]
        intent.search_terms = [
            item.strip()
            for item in (payload.get("search_terms") or [])
            if isinstance(item, str) and item.strip()
        ][:4]
        intent.noisy_terms = [
            item.strip().lower()
            for item in (payload.get("noisy_terms") or [])
            if isinstance(item, str) and item.strip()
        ][:6]
        minimum = payload.get("min_distinct_candidates")
        if isinstance(minimum, int) and not isinstance(minimum, bool) and 0 < minimum < 50:
            intent.min_distinct_candidates = minimum
        intent.explanation_required = bool(payload.get("explanation_required"))
        intent.explanation_fields = [
            canonical_field(item)
            for item in (payload.get("explanation_fields") or [])
            if isinstance(item, str) and canonical_field(item)
        ]
        for raw in payload.get("open_questions") or []:
            if not isinstance(raw, dict):
                continue
            question = str(raw.get("question") or "").strip()
            if question and bool(raw.get("decision_changing")):
                intent.pending_questions.append(question[:220])

    def _requirement(self, raw: Any, index: int, query: str) -> Requirement | None:
        if not isinstance(raw, dict):
            return None
        name = canonical_field(raw.get("field"))
        if not name:
            return None
        value = raw.get("value")
        operator = ConstraintNormalizer.normalize_operator(
            raw.get("operator"), "" if value is None else str(value)
        )
        if operator not in VALID_OPERATORS:
            return None
        if value is None and operator != "exists":
            return None
        strength = str(raw.get("strength") or "hard").strip().lower()
        if strength not in VALID_STRENGTHS:
            strength = "hard"
        scope = str(raw.get("scope") or "selected_variant").strip().lower()
        if scope not in VALID_SCOPES:
            scope = "selected_variant"
        source_text = str(raw.get("source_text") or "").strip()
        if source_text and source_text.lower() not in query.lower():
            source_text = ""
        unit = raw.get("unit")
        return Requirement(
            rid="r" + str(index + 1),
            field=name,
            operator=operator,
            value=value,
            unit=str(unit) if isinstance(unit, str) and unit.strip() else None,
            strength=strength,
            scope=scope,
            source_text=source_text,
        )

    _CLAUSE_PATTERNS: tuple[tuple[str, str, Any], ...] = (
        ("screen_size", "inch", _INCH_RE),
        ("refresh_rate", "hz", _HZ_RE),
    )

    def reconcile_numeric_clauses(
        self, intent: Intent, message: str, patch: dict[str, Any]
    ) -> None:
        for field_name, unit, pattern in self._CLAUSE_PATTERNS:
            values = _dedupe_decimals(_numbers_before(pattern, message.lower()))
            if len(values) != 1:
                continue
            wanted = values[0]
            existing = [
                r
                for r in intent.requirements
                if r.active and r.field == field_name and r.operator == "eq"
            ]
            if existing:
                current = ConstraintNormalizer.wanted_values(existing[0]) or []
                if current and current[0] == wanted:
                    continue
                existing[0].value = str(wanted)
                existing[0].unit = unit
                existing[0].revision = intent.revision
                existing[0].source_text = compact(message, 160)
                if existing[0].rid not in patch["replaced"]:
                    patch["replaced"].append(existing[0].rid)
                patch.setdefault("recovered_clauses", []).append(field_name)
                continue
            added = Requirement(
                rid="u" + str(intent.revision) + "-" + field_name,
                field=field_name,
                operator="eq",
                value=str(wanted),
                unit=unit,
                strength="hard",
                revision=intent.revision,
                source_text=compact(message, 160),
            )
            intent.requirements.append(added)
            patch["added"].append(added.rid)
            patch.setdefault("recovered_clauses", []).append(field_name)


    def _apply_text_backstops(self, intent: Intent, query: str) -> None:
        if not intent.by_field("price"):
            amount = extract_budget(query)
            if amount is not None:
                intent.requirements.append(
                    Requirement(
                        rid="rb-price",
                        field="price",
                        operator="le",
                        value=str(amount),
                        unit=None,
                        strength="hard",
                        source_text=budget_span(query),
                    )
                )
                intent.notes.append("budget recovered from the request text")

        scoped = _SCOPE_CUE_RE.search(query) is not None
        records: list[CandidateRestriction] = []
        seen: set[str] = set()
        for identifier in _ID_PAIR_RE.findall(query):
            record = CandidateRestriction.parse(
                identifier, explicit=True, source_text=identifier
            )
            if record is not None and record.text not in seen:
                seen.add(record.text)
                records.append(record)
        for record in intent.restriction_records:
            if record.text in seen:
                continue
            if record.sku is not None and record.text in query:
                record.explicit = True
                record.source_text = record.text
            elif record.product_id in query:
                record.explicit = scoped
                record.source_text = record.product_id
            else:
                continue
            seen.add(record.text)
            records.append(record)
        intent.set_restrictions(records)
        if records and not any(record.explicit for record in records):
            intent.notes.append(
                "candidate scope treated as inferred: no restricting language found"
            )

        if not intent.title_terms:
            for quoted in _QUOTED_RE.findall(query):
                terms = tokens(quoted)
                if 1 <= len(terms) <= 6:
                    intent.title_terms = terms
                    intent.notes.append("literal title terms taken from a quoted phrase")
                    break

        if not intent.min_distinct_candidates:
            stated = _MIN_DISTINCT_RE.search(query)
            if stated:
                try:
                    intent.min_distinct_candidates = int(stated.group(1))
                except (TypeError, ValueError):
                    pass

        if not intent.search_terms:
            base = intent.product_identity or " ".join(
                tokens(query)[:4]
            )
            if base:
                intent.search_terms = [base]

    @staticmethod
    def _dedupe(intent: Intent) -> None:
        seen: set[tuple[str, str, str, str]] = set()
        kept: list[Requirement] = []
        for requirement in intent.requirements:
            key = (
                requirement.field,
                requirement.operator,
                str(requirement.value).strip().lower(),
                requirement.strength,
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append(requirement)
        intent.requirements = kept


    def apply_patch(
        self, intent: Intent, message: str, payload: Any, message_index: int
    ) -> dict[str, Any]:
        patch: dict[str, Any] = {
            "message": compact(message, 300),
            "message_index": message_index,
            "added": [],
            "replaced": [],
            "removed": [],
        }
        changes = (payload or {}).get("changes")
        changes = changes if isinstance(changes, list) else []
        patch["rejected"] = []
        for raw in changes:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "").strip().lower()
            target = str(raw.get("target_id") or "").strip()
            existing = next(
                (r for r in intent.requirements if r.rid == target and r.active), None
            )
            span = raw.get("source_text")
            grounded = _spans_message(span, message)
            if kind == "remove":
                if existing is None:
                    continue
                if not grounded and not (
                    existing.strength != "hard"
                    and _mentions_field(existing.field, message)
                ):
                    patch["rejected"].append(
                        {
                            "kind": "remove",
                            "target_id": existing.rid,
                            "reason": "no supporting span in the message",
                        }
                    )
                    continue
                existing.active = False
                patch["removed"].append(existing.rid)
                continue
            candidate_requirement = self._requirement(
                raw, len(intent.requirements), message
            )
            if candidate_requirement is None:
                continue
            if not grounded and not _mentions_field(
                candidate_requirement.field, message
            ):
                patch["rejected"].append(
                    {
                        "kind": kind or "add",
                        "field": candidate_requirement.field,
                        "reason": "no supporting span in the message",
                    }
                )
                continue
            candidate_requirement.source_message = message_index
            candidate_requirement.revision = intent.revision
            if existing is None:
                existing = next(
                    (
                        r
                        for r in intent.requirements
                        if r.active
                        and r.field == candidate_requirement.field
                        and r.operator == candidate_requirement.operator
                    ),
                    None,
                )
            if existing is not None:
                existing.operator = candidate_requirement.operator
                existing.value = candidate_requirement.value
                existing.unit = candidate_requirement.unit
                existing.strength = candidate_requirement.strength
                existing.revision = intent.revision
                existing.source_message = message_index
                existing.source_text = candidate_requirement.source_text
                if existing.rid not in patch["replaced"]:
                    patch["replaced"].append(existing.rid)
            else:
                candidate_requirement.rid = (
                    "u" + str(intent.revision) + "-" + candidate_requirement.field
                )
                intent.requirements.append(candidate_requirement)
                patch["added"].append(candidate_requirement.rid)

        self.reconcile_numeric_clauses(intent, message, patch)

        amount = extract_budget(message)
        if amount is not None:
            caps = [
                r
                for r in intent.requirements
                if r.active and r.field == "price" and r.operator in {"le", "lt"}
            ]
            if caps:
                if to_decimal(caps[0].value) != amount:
                    caps[0].value = str(amount)
                    caps[0].revision = intent.revision
                    caps[0].source_message = message_index
                    caps[0].source_text = budget_span(message)
                    if caps[0].rid not in patch["replaced"]:
                        patch["replaced"].append(caps[0].rid)
            else:
                added = Requirement(
                    rid="u" + str(intent.revision) + "-price",
                    field="price",
                    operator="le",
                    value=str(amount),
                    strength="hard",
                    revision=intent.revision,
                    source_message=message_index,
                    source_text=budget_span(message),
                )
                intent.requirements.append(added)
                patch["added"].append(added.rid)
        return patch


SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNRESOLVED = "UNRESOLVED"

SELECTION_PRECEDENCE = ("variant_option", "raw_product_fact", "title")

FIELD_SCOPED_SOURCES = frozenset(
    {"raw_product_fact", "brand", "category_path", "current_variant_state"}
)


@dataclass
class ResolvedValue:
    field: str
    raw: Any
    source_kind: str
    source_path: str
    source_action: str
    observed_at: int


def resolve_field_values(
    candidate: CandidateState, sku: Any, field_name: str
) -> list[ResolvedValue]:
    out: list[ResolvedValue] = []
    variant = candidate.variant(sku)

    if field_name == "price":
        if variant is not None and variant.get("price") is not None:
            out.append(
                ResolvedValue(
                    field_name,
                    variant.get("price"),
                    "current_variant_state",
                    "variants[" + str(variant.get("sku")) + "].price",
                    str(variant.get("_source_action") or "view"),
                    int(variant.get("_observed_at") or 0),
                )
            )
        return out

    if field_name == "in_stock":
        if variant is not None and variant.get("in_stock") is not None:
            out.append(
                ResolvedValue(
                    field_name,
                    variant.get("in_stock"),
                    "current_variant_state",
                    "variants[" + str(variant.get("sku")) + "].in_stock",
                    str(variant.get("_source_action") or "view"),
                    int(variant.get("_observed_at") or 0),
                )
            )
        return out

    if field_name == "brand":
        if candidate.brand:
            out.append(
                ResolvedValue(
                    field_name, candidate.brand, "brand", "brand", "view", 0
                )
            )
        if candidate.title:
            out.append(
                ResolvedValue(field_name, candidate.title, "title", "title", "view", 0)
            )
        return out

    if field_name == "category":
        if candidate.category_path:
            out.append(
                ResolvedValue(
                    field_name,
                    list(candidate.category_path),
                    "category_path",
                    "category_path",
                    "view",
                    0,
                )
            )
        return out

    if variant is not None and variant.get("options"):
        out.append(
            ResolvedValue(
                field_name,
                variant.get("options"),
                "variant_option",
                "variants[" + str(variant.get("sku")) + "].options",
                str(variant.get("_source_action") or "view"),
                int(variant.get("_observed_at") or 0),
            )
        )
    for key in FIELD_SOURCES.get(field_name, (field_name,)):
        for raw_key, raw_value in (candidate.facts or {}).items():
            if canonical_field(raw_key) != field_name and str(raw_key).lower() != key:
                continue
            if raw_value in (None, ""):
                continue
            entry = ResolvedValue(
                field_name,
                raw_value,
                "raw_product_fact",
                "facts." + str(raw_key),
                "view",
                0,
            )
            if not any(
                item.source_path == entry.source_path and item.raw == entry.raw
                for item in out
            ):
                out.append(entry)
    if candidate.title:
        out.append(
            ResolvedValue(field_name, candidate.title, "title", "title", "view", 0)
        )
    order = {kind: index for index, kind in enumerate(SELECTION_PRECEDENCE)}
    out.sort(key=lambda item: order.get(item.source_kind, 99))
    return out


def _interpret(field_name: str, raw: Any) -> tuple[str, Any]:
    if field_name == "resolution":
        return "classes", parse_resolution_classes(raw)
    if field_name in NUMERIC_FIELDS:
        parsed = ConstraintNormalizer.values_for(field_name, raw)
        return "numbers", (parsed or [])
    if field_name == "in_stock":
        return "bool", truthy(raw)
    if field_name == "category":
        return "path", [str(part) for part in (raw if isinstance(raw, list) else [raw])]
    return "text", raw


def supports_selection(
    requirement: Requirement, candidate: CandidateState, sku: Any
) -> ConstraintAssessment:
    field_name = requirement.field
    observations = resolve_field_values(candidate, sku, field_name)
    if not observations:
        return ConstraintAssessment(
            requirement.rid, UNRESOLVED, [], "no observation carries " + field_name
        )

    top_kind = observations[0].source_kind
    scoped = [item for item in observations if item.source_kind == top_kind]

    if field_name == "price":
        wanted = to_decimal(requirement.value)
        observed = to_decimal(scoped[0].raw)
        if wanted is None or observed is None:
            return ConstraintAssessment(
                requirement.rid, UNRESOLVED, [scoped[0].source_path], "unparsable price"
            )
        ok = ConstraintNormalizer.compare_numeric(requirement.operator, observed, wanted)
        return ConstraintAssessment(
            requirement.rid,
            SUPPORTED if ok else CONTRADICTED,
            [scoped[0].source_path],
            "current price " + str(observed),
        )

    if field_name == "in_stock":
        observed = truthy(scoped[0].raw)
        wanted = truthy(requirement.value)
        if observed is None:
            return ConstraintAssessment(
                requirement.rid, UNRESOLVED, [scoped[0].source_path], "stock unknown"
            )
        wanted = True if wanted is None else wanted
        ok = observed == wanted if requirement.operator == "eq" else observed != wanted
        return ConstraintAssessment(
            requirement.rid,
            SUPPORTED if ok else CONTRADICTED,
            [scoped[0].source_path],
            "current stock " + str(observed),
        )

    if field_name == "category":
        path = [str(part).lower() for part in (scoped[0].raw or [])]
        want = " ".join(tokens(requirement.value))
        if not want:
            return ConstraintAssessment(
                requirement.rid, UNRESOLVED, [scoped[0].source_path], "no category value"
            )
        wanted_tokens = set(tokens(requirement.value))
        singular = {term[:-1] for term in wanted_tokens if term.endswith("s")}
        plural = {term + "s" for term in wanted_tokens}
        forms = wanted_tokens | singular | plural
        hit = False
        overlap = False
        for part in path:
            part_tokens = set(tokens(part))
            if not part_tokens:
                continue
            if wanted_tokens <= part_tokens or (
                len(wanted_tokens) == 1 and forms & part_tokens
            ):
                hit = True
                break
            if forms & part_tokens:
                overlap = True
        if not hit and overlap:
            return ConstraintAssessment(
                requirement.rid,
                UNRESOLVED,
                [scoped[0].source_path],
                "category_path partially matches " + json.dumps(path),
            )
        if requirement.operator in {"ne", "not_contains"}:
            hit = not hit
        return ConstraintAssessment(
            requirement.rid,
            SUPPORTED if hit else CONTRADICTED,
            [scoped[0].source_path],
            "category_path " + json.dumps(path),
        )

    if field_name == "resolution":
        wanted_classes = parse_resolution_classes(requirement.value)
        for observation in observations:
            classes = parse_resolution_classes(observation.raw)
            if not classes:
                continue
            if observation.source_kind == "title" and top_kind != "title":
                break
            if len(set(classes)) > 1:
                return ConstraintAssessment(
                    requirement.rid,
                    UNRESOLVED,
                    [observation.source_path],
                    "conflicting resolution classes " + json.dumps(classes),
                )
            if not wanted_classes:
                return ConstraintAssessment(
                    requirement.rid,
                    UNRESOLVED,
                    [observation.source_path],
                    "requested resolution not classifiable",
                )
            observed_rank = CLASS_ORDER.get(classes[0])
            wanted_rank = CLASS_ORDER.get(wanted_classes[0])
            if observed_rank is None or wanted_rank is None:
                return ConstraintAssessment(
                    requirement.rid,
                    UNRESOLVED,
                    [observation.source_path],
                    "resolution class outside the public ranking",
                )
            ok = ConstraintNormalizer.compare_numeric(
                requirement.operator, Decimal(observed_rank), Decimal(wanted_rank)
            )
            if requirement.operator not in VALID_OPERATORS or requirement.operator in {
                "contains",
                "not_contains",
                "exists",
                "one_of",
            }:
                ok = observed_rank == wanted_rank
            return ConstraintAssessment(
                requirement.rid,
                SUPPORTED if ok else CONTRADICTED,
                [observation.source_path],
                "resolution class " + classes[0],
            )
        return ConstraintAssessment(
            requirement.rid, UNRESOLVED, [], "no resolution evidence"
        )

    if field_name in NUMERIC_FIELDS:
        wanted_values = ConstraintNormalizer.wanted_values(requirement)
        if not wanted_values:
            single = to_decimal(requirement.value)
            wanted_values = [single] if single is not None else []
        if not wanted_values:
            return ConstraintAssessment(
                requirement.rid, UNRESOLVED, [], "requested value not numeric"
            )
        for observation in observations:
            values = ConstraintNormalizer.values_for(field_name, observation.raw) or []
            if not values:
                continue
            if len(values) > 1:
                return ConstraintAssessment(
                    requirement.rid,
                    UNRESOLVED,
                    [observation.source_path],
                    "several conflicting values " + str([str(v) for v in values]),
                )
            ok = ConstraintNormalizer.compare_numeric(
                requirement.operator, values[0], wanted_values[0]
            )
            if ok:
                return ConstraintAssessment(
                    requirement.rid,
                    SUPPORTED,
                    [observation.source_path],
                    field_name + " " + str(values[0]),
                )
            if observation.source_kind != "title":
                return ConstraintAssessment(
                    requirement.rid,
                    CONTRADICTED,
                    [observation.source_path],
                    field_name + " is " + str(values[0]),
                )
        return ConstraintAssessment(
            requirement.rid, UNRESOLVED, [], "no " + field_name + " evidence"
        )

    if requirement.operator == "exists":
        wanted_tokens = [
            term for term in tokens(requirement.value) if term not in _BOOLEAN_WORDS
        ]
        terms = wanted_tokens + tokens(field_name.replace("_", " "))
        for observation in observations:
            if observation.source_kind in FIELD_SCOPED_SOURCES:
                affirmative = truthy(observation.raw)
                if affirmative is False:
                    return ConstraintAssessment(
                        requirement.rid,
                        CONTRADICTED,
                        [observation.source_path],
                        field_name + " is stated as absent",
                    )
                if str(observation.raw or "").strip():
                    return ConstraintAssessment(
                        requirement.rid,
                        SUPPORTED,
                        [observation.source_path],
                        field_name + " = " + compact(observation.raw, 80),
                    )
            if ConstraintNormalizer.mentions(observation.raw, terms):
                return ConstraintAssessment(
                    requirement.rid,
                    SUPPORTED,
                    [observation.source_path],
                    compact(observation.raw, 120),
                )
        return ConstraintAssessment(
            requirement.rid, UNRESOLVED, [], "nothing observed mentions " + field_name
        )

    for observation in observations:
        verdict = ConstraintNormalizer.compare_text(
            requirement.operator, observation.raw, requirement.value
        )
        if verdict is True:
            return ConstraintAssessment(
                requirement.rid,
                SUPPORTED,
                [observation.source_path],
                compact(observation.raw, 120),
            )
        if verdict is False and observation.source_kind in FIELD_SCOPED_SOURCES:
            return ConstraintAssessment(
                requirement.rid,
                CONTRADICTED,
                [observation.source_path],
                compact(observation.raw, 120),
            )
    return ConstraintAssessment(
        requirement.rid, UNRESOLVED, [], "no decisive " + field_name + " evidence"
    )


@dataclass
class ClaimCandidate:
    field: str
    value: str
    source_path: str
    source_kind: str
    required: bool = False


def _claim_admissible(field_name: str, source_kind: str) -> bool:
    return EvidenceFact(
        product_id="",
        sku=None,
        field=field_name,
        raw_value=None,
        normalized=None,
        source_action="",
        source_path="",
        source_kind=source_kind,
        observed_at=0,
    ).admissible_for_claim


def _claim(field_name: str, value: str, path: str, kind: str) -> ClaimCandidate | None:
    if not _claim_admissible(field_name, kind):
        return None
    return ClaimCandidate(field_name, value, path, kind)


def supports_claim(
    field_name: str,
    candidate: CandidateState,
    sku: Any,
    *,
    allowed_fields: Sequence[str] | None,
) -> ClaimCandidate | None:
    if allowed_fields is not None and field_name not in allowed_fields:
        return None
    variant = candidate.variant(sku)

    if field_name == "price":
        if variant is None or variant.get("price") is None:
            return None
        raw = variant.get("price")
        if to_decimal(raw) is None:
            return None
        text = str(raw) if isinstance(raw, (int, float)) else str(raw).strip()
        return _claim(
            field_name, text, "variants[" + str(sku) + "].price",
            "current_variant_state",
        )

    if field_name == "in_stock":
        if variant is None:
            return None
        state = truthy(variant.get("in_stock"))
        if state is not True:
            return None
        return _claim(
            field_name, "true", "variants[" + str(sku) + "].in_stock",
            "current_variant_state",
        )

    if field_name == "category":
        if not candidate.category_path:
            return None
        return _claim(
            field_name, str(candidate.category_path[-1]), "category_path",
            "category_path",
        )

    if field_name == "brand":
        if not candidate.brand or not str(candidate.brand).strip():
            return None
        return _claim(field_name, str(candidate.brand), "brand", "brand")

    for observation in resolve_field_values(candidate, sku, field_name):
        if observation.source_kind not in {"raw_product_fact", "variant_option"}:
            continue
        kind, value = _interpret(field_name, observation.raw)
        if kind == "classes":
            if len(set(value)) != 1:
                continue
            return _claim(
                field_name, value[0], observation.source_path, observation.source_kind
            )
        if kind == "numbers":
            if len(value) != 1:
                continue
            text = format(value[0].normalize(), "f")
            if field_name == "screen_size":
                text = text + "in"
            elif field_name == "refresh_rate":
                text = text + "Hz"
            return _claim(
                field_name, text, observation.source_path, observation.source_kind
            )
        text = str(observation.raw).strip()
        if not text:
            continue
        return _claim(
            field_name, text[:80], observation.source_path, observation.source_kind
        )
    return None


ASSESS_SYSTEM = (
    "You judge whether a product listing satisfies a shopper requirement, using "
    "only the observed fields you are given. You must name the exact observed "
    "field you used. If no observed field settles it, answer unresolved. Reply "
    "with one JSON object and no prose."
)

ASSESS_TEMPLATE = """Public task rules (authoritative):
<<<
{rules}
>>>

Requirements to judge:
{requirements}

Observed candidates (these fields are the only admissible evidence):
{candidates}

Answer ONLY the requirement ids listed in each candidate's "judge_only".
Everything else is already decided and repeating it wastes the reply.

Return one JSON object:
{{"candidates": {{
   "<candidate label>": {{
      "requirements": {{
         "<requirement id>": {{
            "verdict": "supported|contradicted|unresolved",
            "evidence_field": "<the exact observed field name you used, or null>",
            "reason": "<at most 8 words; omit entirely when supported>"
         }}
      }}
   }}
}}}}

Rules:
- The requirement id "r-rules" covers ONLY conditions the public rules add on
  top of the listed requirements - product identity, genuineness, accessory or
  wrong-device-type rejection. Never mark it contradicted because a listed
  requirement such as brand, price, category or a specification is unmet;
  those are judged separately and reporting them here changes nothing.
- Silence is not agreement. A field that is absent or empty is "unresolved".
- Never use the title as proof when a raw fact or a variant option states the value.
- An explicitly conflicting statement is "contradicted", not "unresolved".
- Do not add candidates, requirements or keys that were not supplied.
- Keep every "reason" under 12 words. No preamble, no summary, no restating
  the inputs: the extra text costs generation time and is never read."""


class ConstraintChecker:

    def __init__(self, inference: InferenceClient):
        self.inference = inference

    def evaluate(
        self,
        candidate: CandidateState,
        sku: Any,
        requirements: Sequence[Requirement],
    ) -> dict[str, ConstraintAssessment]:
        out: dict[str, ConstraintAssessment] = {}
        for requirement in requirements:
            out[requirement.rid] = supports_selection(requirement, candidate, sku)
        return out


    def request_semantic_assessment(
        self,
        rules: str,
        candidates: Sequence[tuple[str, CandidateState, Any]],
        requirements: Sequence[Requirement],
        settled: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not candidates or not requirements:
            return {}
        payload = self.inference.ask_json(
            ASSESS_SYSTEM,
            ASSESS_TEMPLATE.format(
                rules=rules[:4000],
                requirements=json.dumps(
                    [
                        {
                            "id": r.rid,
                            "field": r.field,
                            "operator": r.operator,
                            "value": r.value,
                            "unit": r.unit,
                            "strength": r.strength,
                        }
                        for r in requirements
                    ],
                    default=str,
                ),
                candidates=json.dumps(
                    {
                        label: dict(
                            describe_candidate(candidate, sku),
                            judge_only=sorted(
                                requirement.rid
                                for requirement in requirements
                                if requirement.rid
                                not in (settled or {}).get(label, set())
                            ),
                        )
                        for label, candidate, sku in candidates
                    },
                    default=str,
                )[:9000],
            ),
            purpose="candidate assessment",
            validate=lambda js: isinstance(js.get("candidates"), dict),
        )
        result = (payload or {}).get("candidates")
        return result if isinstance(result, dict) else {}

    def validate_model_assessment(
        self,
        model_view: dict[str, Any],
        candidate: CandidateState,
        sku: Any,
        requirements: Sequence[Requirement],
        deterministic: dict[str, ConstraintAssessment],
    ) -> dict[str, ConstraintAssessment]:
        merged = dict(deterministic)
        by_id = {r.rid: r for r in requirements}
        entries = (model_view or {}).get("requirements")
        if not isinstance(entries, dict):
            return merged

        observed_fields = observed_field_names(candidate, sku)
        for rid, raw in entries.items():
            requirement = by_id.get(str(rid))
            if requirement is None or not isinstance(raw, dict):
                continue
            current = merged.get(str(rid))
            verdict = str(raw.get("verdict") or "").strip().lower()
            evidence_field = str(raw.get("evidence_field") or "").strip()
            reason = compact(raw.get("reason"), 160)

            if current is not None and current.state == CONTRADICTED:
                continue
            if verdict == "contradicted":
                if evidence_field and evidence_field.lower() in observed_fields:
                    merged[str(rid)] = ConstraintAssessment(
                        str(rid), CONTRADICTED, ["model:" + evidence_field], reason
                    )
                elif current is None or current.state != SUPPORTED:
                    merged[str(rid)] = ConstraintAssessment(
                        str(rid), UNRESOLVED, [], "model contradiction without evidence"
                    )
                continue
            if verdict == "supported":
                if current is not None and current.state == SUPPORTED:
                    continue
                if not evidence_field or evidence_field.lower() not in observed_fields:
                    merged[str(rid)] = ConstraintAssessment(
                        str(rid),
                        UNRESOLVED,
                        [],
                        "model cited an unobserved field: " + (evidence_field or "none"),
                    )
                    continue
                if observed_fields.get(evidence_field.lower()) in (None, ""):
                    merged[str(rid)] = ConstraintAssessment(
                        str(rid), UNRESOLVED, [], "cited field is empty"
                    )
                    continue
                merged[str(rid)] = ConstraintAssessment(
                    str(rid), SUPPORTED, ["model:" + evidence_field], reason
                )
                continue
            if current is None or current.state == UNRESOLVED:
                merged[str(rid)] = ConstraintAssessment(
                    str(rid), UNRESOLVED, [], reason or "model unresolved"
                )
        return merged


def observed_field_names(candidate: CandidateState, sku: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in (candidate.facts or {}).items():
        seen[str(key).lower()] = value
    if candidate.brand:
        seen["brand"] = candidate.brand
    if candidate.category_path:
        seen["category_path"] = candidate.category_path
    if candidate.title:
        seen["title"] = candidate.title
    variant = candidate.variant(sku)
    if variant:
        for key in ("options", "price", "in_stock", "sku"):
            if variant.get(key) is not None:
                seen[key] = variant.get(key)
    return seen


def describe_candidate(candidate: CandidateState, sku: Any) -> dict[str, Any]:
    variant = candidate.variant(sku)
    return {
        "product_id": candidate.product_id,
        "title": compact(candidate.title, 220),
        "brand": candidate.brand,
        "category_path": candidate.category_path,
        "facts": {
            str(key): compact(value, 120)
            for key, value in list((candidate.facts or {}).items())[:20]
        },
        "selected_variant": None
        if variant is None
        else {
            "sku": variant.get("sku"),
            "options": variant.get("options"),
            "price": variant.get("price"),
            "in_stock": variant.get("in_stock"),
        },
        "other_variants": [
            {
                "sku": row.get("sku"),
                "options": row.get("options"),
                "price": row.get("price"),
                "in_stock": row.get("in_stock"),
            }
            for row in (candidate.variants or [])[:6]
            if variant is None or row.get("sku") != variant.get("sku")
        ],
    }


@dataclass
class RankedItem:
    candidate: CandidateState
    sku: Any
    key: tuple
    values: dict[str, Any]
    unresolved: list[str]


class Comparator:

    @staticmethod
    def value_for(candidate: CandidateState, sku: Any, field_name: str) -> Any:
        for observation in resolve_field_values(candidate, sku, field_name):
            kind, value = _interpret(field_name, observation.raw)
            if kind == "numbers" and value:
                return max(value) if len(value) > 1 else value[0]
            if kind == "classes" and len(set(value)) == 1:
                return CLASS_ORDER.get(value[0])
            if kind == "bool" and value is not None:
                return value
            if kind == "text" and str(value or "").strip():
                if field_name in NUMERIC_FIELDS:
                    continue
                return str(value).strip()
        return None

    @classmethod
    def rank(
        cls,
        entries: Sequence[tuple[CandidateState, Any]],
        objectives: Sequence[Objective],
    ) -> tuple[list[RankedItem], list[RankedItem]]:
        ranked: list[RankedItem] = []
        blocked: list[RankedItem] = []
        for candidate, sku in entries:
            values: dict[str, Any] = {}
            missing: list[str] = []
            for objective in objectives:
                value = cls.value_for(candidate, sku, objective.field)
                if value is None or not isinstance(value, (int, float, Decimal)):
                    missing.append(objective.field)
                values[objective.field] = value
            item = RankedItem(candidate, sku, tuple(), values, missing)
            if missing:
                blocked.append(item)
                continue
            key = tuple(
                (-Decimal(str(values[objective.field])))
                if objective.direction == "max"
                else Decimal(str(values[objective.field]))
                for objective in objectives
            )
            item.key = key
            ranked.append(item)
        ranked.sort(key=lambda row: row.key)
        return ranked, blocked

    @staticmethod
    def dominated(item: RankedItem, others: Sequence[RankedItem],
                  objectives: Sequence[Objective]) -> RankedItem | None:
        for other in others:
            if other is item:
                continue
            better_once = False
            worse = False
            for objective in objectives:
                mine = Decimal(str(item.values.get(objective.field)))
                theirs = Decimal(str(other.values.get(objective.field)))
                if objective.direction == "max":
                    if theirs > mine:
                        better_once = True
                    elif theirs < mine:
                        worse = True
                else:
                    if theirs < mine:
                        better_once = True
                    elif theirs > mine:
                        worse = True
            if better_once and not worse:
                return other
        return None


class DecisionValidator:

    @staticmethod
    def check_comparison(
        table: Any,
        chosen: str,
        objectives: Sequence[Objective],
        labels: Sequence[str],
    ) -> tuple[str, str]:
        try:
            if not isinstance(table, dict) or chosen not in table:
                return chosen, "no usable table"
            rows: dict[str, dict[str, Decimal]] = {}
            for label, raw in table.items():
                if label not in labels or not isinstance(raw, dict):
                    return chosen, "table has an unknown row"
                values: dict[str, Decimal] = {}
                for objective in objectives:
                    number = to_decimal(raw.get(objective.field))
                    if number is None:
                        return chosen, "table is missing " + objective.field
                    values[objective.field] = number
                rows[label] = values
            if len(rows) < 2:
                return chosen, "table too small to arbitrate"

            def beats(a: dict[str, Decimal], b: dict[str, Decimal]) -> bool:
                for objective in objectives:
                    left, right = a[objective.field], b[objective.field]
                    if left == right:
                        continue
                    if objective.direction == "max":
                        return left > right
                    return left < right
                return False

            challengers = [
                label for label in rows if label != chosen
                and beats(rows[label], rows[chosen])
            ]
            for label in challengers:
                if any(
                    beats(rows[other], rows[label]) for other in rows if other != label
                ):
                    continue
                return label, "the model's own table ranks " + label + " above " + chosen
            return chosen, "table agrees with the pick"
        except Exception:
            return chosen, "arbitration failed open"


SELECT_SYSTEM = (
    "You choose between shopping candidates for a shopper, using only the "
    "observed facts supplied. Reply with one JSON object and no prose."
)

SELECT_TEMPLATE = """Shopper request and public rules:
<<<
{rules}
>>>

Named preferences, most important first: {preferences}

Candidates (observed facts only; every one already satisfies the firm rules):
{candidates}

Return one JSON object:
{{"table": {{"<label>": {{"<criterion>": "<observed value or null>"}}}},
  "best": "<label>",
  "evidence": {{"<criterion>": "<the observed field you used for the winner>"}},
  "disadvantages": {{"<label>": "<one clause, or null>"}},
  "unknowns": ["<criterion with no observed evidence>"],
  "reason": "<one sentence grounded in the observed facts>"}}

Never fill a missing property with a typical value for that brand or model.
Keep "reason" to one sentence and every other string under 12 words."""

DUEL_TEMPLATE = """Shopper request and public rules:
<<<
{rules}
>>>

Two candidates, one shared evidence snapshot. Decide which better serves the
shopper's stated preferences, using only these observed facts.

A: {a}
B: {b}

Disputed points: {disputed}

Return one JSON object: {{"winner": "A|B|tie", "criterion": "<the deciding
observed field>", "reason": "<one sentence>"}}"""


class SelectiveReviewer:

    def __init__(self, inference: InferenceClient, settings: Settings):
        self.inference = inference
        self.settings = settings
        self.notes: list[str] = []


    def check_order_sensitivity(
        self,
        rules: str,
        preferences: Sequence[str],
        labelled: Sequence[tuple[str, dict[str, Any]]],
        first_choice: str,
    ) -> tuple[str, str]:
        if not self.settings.use_order_sensitivity_check or len(labelled) < 2:
            return first_choice, "order check disabled"
        reversed_view = list(reversed(labelled))
        payload = self._ask_select(rules, preferences, reversed_view)
        second = str((payload or {}).get("best") or "").strip()
        if not second or second == first_choice:
            return first_choice, "order-insensitive"
        self.notes.append(
            "order sensitivity: forward=" + first_choice + " reversed=" + second
        )
        return first_choice, "order-sensitive; original order kept"


    def aggregate_votes(
        self,
        rules: str,
        preferences: Sequence[str],
        labelled: Sequence[tuple[str, dict[str, Any]]],
        rounds: int = 2,
    ) -> tuple[str | None, dict[str, int]]:
        tally: dict[str, int] = {}
        if not self.settings.use_candidate_votes:
            return None, tally
        for _round in range(max(1, rounds)):
            payload = self._ask_select(rules, preferences, labelled)
            pick = str((payload or {}).get("best") or "").strip()
            if pick:
                tally[pick] = tally.get(pick, 0) + 1
        if not tally:
            return None, tally
        best, count = max(tally.items(), key=lambda row: row[1])
        return (best if count > 1 else None), tally


    def compare_rival(
        self,
        rules: str,
        champion: dict[str, Any],
        rival: dict[str, Any],
        disputed: Sequence[str],
    ) -> tuple[str, str]:
        if not self.settings.use_rival_duel:
            return "A", "duel disabled"
        payload = self.inference.ask_json(
            SELECT_SYSTEM,
            DUEL_TEMPLATE.format(
                rules=rules[:2500],
                a=json.dumps(champion, default=str)[:1800],
                b=json.dumps(rival, default=str)[:1800],
                disputed=", ".join(disputed) or "none",
            ),
            purpose="rival duel",
            validate=lambda js: isinstance(js.get("winner"), str),
        )
        winner = str((payload or {}).get("winner") or "A").strip().upper()
        if winner not in {"A", "B"}:
            return "A", "duel inconclusive"
        return winner, compact((payload or {}).get("reason"), 160)


    def select(
        self,
        rules: str,
        preferences: Sequence[str],
        labelled: Sequence[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any] | None:
        return self._ask_select(rules, preferences, labelled)

    def _ask_select(
        self,
        rules: str,
        preferences: Sequence[str],
        labelled: Sequence[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any] | None:
        return self.inference.ask_json(
            SELECT_SYSTEM,
            SELECT_TEMPLATE.format(
                rules=rules[:3000],
                preferences=", ".join(preferences) or "none stated",
                candidates=json.dumps(dict(labelled), default=str)[:9000],
            ),
            purpose="preference selection",
            validate=lambda js: isinstance(js.get("best"), str),
        )


_REASON_LIST_RE = re.compile(r"reasons?\s*:\s*([a-z_,\s]+?)(?:\.|$)", re.IGNORECASE)
_REASON_CUE_RE = re.compile(
    r"\b(why|reason|justif\w*|explain|walk me through|tell me how|make the case)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE = {"true", "yes", "supported", "available", "included", "present"}

CLAIMABLE_CORE_FIELDS = frozenset({"brand", "category", "in_stock", "price"})
CLAIMABLE_SPEC_FIELDS = frozenset(
    {"screen_size", "resolution", "refresh_rate", "ram", "storage", "battery"}
)


class ClaimBuilder:

    def __init__(self, catalog: ToolCatalog, tool_name: str = "place_test_order"):
        self.catalog = catalog
        self.tool_name = tool_name
        self.schema = catalog.schemas.get(tool_name) or {}


    @property
    def justification_schema(self) -> dict[str, Any]:
        prop = self.catalog.properties(self.tool_name).get("justification")
        return prop if isinstance(prop, dict) else {}

    @property
    def claim_field_schema(self) -> dict[str, Any]:
        claims = (self.justification_schema.get("properties") or {}).get("claims")
        if not isinstance(claims, dict):
            return {}
        items = claims.get("items")
        if not isinstance(items, dict):
            return {}
        field_schema = (items.get("properties") or {}).get("field")
        return field_schema if isinstance(field_schema, dict) else {}

    def allowed_fields(self) -> list[str] | None:
        enum = self.claim_field_schema.get("enum")
        if isinstance(enum, list) and enum:
            return [str(item) for item in enum]
        return None

    def permitted_fields(self, intent: "Intent") -> list[str]:
        enum = self.allowed_fields()
        if enum is not None:
            return enum
        permitted = set(CLAIMABLE_CORE_FIELDS) | set(CLAIMABLE_SPEC_FIELDS)
        permitted.update(self.described_reasons())
        return sorted(permitted)

    def described_reasons(self) -> list[str]:
        text = str(self.claim_field_schema.get("description") or "")
        match = _REASON_LIST_RE.search(text)
        if not match:
            return []
        return [
            re.sub(r"[^a-z0-9_]+", "_", part.strip().lower()).strip("_")
            for part in re.split(r",|\bor\b", match.group(1))
            if part.strip()
        ]

    def min_claims(self) -> int:
        claims = (self.justification_schema.get("properties") or {}).get("claims")
        if isinstance(claims, dict):
            minimum = claims.get("minItems")
            if isinstance(minimum, int):
                return minimum
        return 0

    def schema_requires_justification(self) -> bool:
        required = self.schema.get("required")
        return isinstance(required, list) and "justification" in required

    def justification_required(self, query: str, intent: Intent) -> bool:
        if self.schema_requires_justification():
            return True
        if not intent.explanation_required:
            return False
        return bool(_REASON_CUE_RE.search(shopper_request_text(query)))


    def required_fields(self, intent: Intent) -> list[str]:
        allowed = self.permitted_fields(intent)
        wanted: list[str] = []
        for name in intent.explanation_fields:
            if name and name not in wanted:
                wanted.append(name)
        for requirement in intent.hard():
            if requirement.field in {"price", "in_stock", "category", "brand"}:
                continue
            if requirement.field not in wanted:
                wanted.append(requirement.field)
        for name in ("price", "in_stock"):
            if name not in wanted:
                wanted.append(name)
        return [name for name in wanted if name in allowed]

    def build(
        self,
        candidate: CandidateState,
        sku: Any,
        intent: Intent,
    ) -> tuple[list[dict[str, str]], list[str], list[str]]:
        allowed = self.permitted_fields(intent)
        required = self.required_fields(intent)
        claims: list[dict[str, str]] = []
        missing: list[str] = []
        rejected: list[str] = []

        for name in required:
            built = supports_claim(name, candidate, sku, allowed_fields=allowed)
            if built is None:
                missing.append(name)
                rejected.append(name + ": no admissible raw evidence")
                continue
            claims.append({"field": built.field, "value": built.value})

        for reason in self.described_reasons():
            if any(claim["field"] == reason for claim in claims):
                continue
            built = self._reason_claim(reason, candidate, sku, allowed)
            if built is not None:
                claims.append({"field": built.field, "value": built.value})
                break

        minimum = self.min_claims()
        if len(claims) < minimum:
            for extra in ("category", "brand"):
                if len(claims) >= minimum:
                    break
                if any(claim["field"] == extra for claim in claims):
                    continue
                built = supports_claim(extra, candidate, sku, allowed_fields=allowed)
                if built is not None:
                    claims.append({"field": built.field, "value": built.value})

        problems = self.catalog.validate(
            self.tool_name,
            {
                "product_id": candidate.product_id,
                "sku": str(sku),
                "justification": {"claims": claims},
            },
        )
        if problems:
            rejected.extend(problems)
            claims = [
                claim
                for claim in claims
                if not self.catalog.validate(
                    self.tool_name,
                    {
                        "product_id": candidate.product_id,
                        "sku": str(sku),
                        "justification": {"claims": [claim]},
                    },
                )
            ]
        return claims, missing, rejected

    def _reason_claim(
        self,
        reason: str,
        candidate: CandidateState,
        sku: Any,
        allowed: Sequence[str] | None,
    ) -> ClaimCandidate | None:
        if reason == "brand":
            return supports_claim("brand", candidate, sku, allowed_fields=allowed)
        if allowed is not None and reason not in allowed:
            return None
        for key, value in (candidate.facts or {}).items():
            if re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_") != reason:
                continue
            state = truthy(value)
            if state is True or str(value).strip().lower() in _AFFIRMATIVE:
                return _claim(
                    reason, "true", "facts." + str(key), "raw_product_fact"
                )
        return None

    def compile_payload(
        self,
        candidate: CandidateState,
        sku: Any,
        intent: Intent,
        query: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        args: dict[str, Any] = {}
        properties = self.catalog.properties(self.tool_name)
        if "product_id" in properties:
            args["product_id"] = str(candidate.product_id)
        if "sku" in properties:
            args["sku"] = str(sku)
        required = self.justification_required(query, intent)
        diagnostics: dict[str, Any] = {
            "justification_submitted": False,
            "justification_required": required,
        }

        if "justification" in properties and required:
            claims, missing, rejected = self.build(candidate, sku, intent)
            diagnostics.update(
                {
                    "claims": claims,
                    "missing_required_claims": missing,
                    "rejected_claims": rejected,
                }
            )
            if claims and not missing:
                args["justification"] = {"claims": claims}
                diagnostics["justification_submitted"] = True
            elif claims and self.schema_requires_justification():
                args["justification"] = {"claims": claims}
                diagnostics["justification_submitted"] = True
                diagnostics["note"] = "submitted without unsupported required claims"
        return args, diagnostics

    def missing_explanation_requirements(
        self, candidate: CandidateState, sku: Any, intent: Intent, query: str
    ) -> list[str]:
        if not self.justification_required(query, intent):
            return []
        _claims, missing, _rejected = self.build(candidate, sku, intent)
        return missing


def shopper_request_text(query: str) -> str:
    marker = re.search(r"\n\s*Task rules\s*:", query, re.IGNORECASE)
    if marker:
        return query[: marker.start()]
    return query


REFORMULATE_SYSTEM = (
    "You propose catalog search queries for a shopping agent. You may only use "
    "vocabulary from the shopper request or from listing titles the agent has "
    "already observed. Reply with one JSON object and no prose."
)

REFORMULATE_TEMPLATE = """Shopper request (abbreviated): {request}
Item being bought: {identity}
Terms that must appear literally in the listing title: {literals}
Queries already attempted: {attempted}
Titles already observed in this catalog:
{titles}

Coverage gap: {gap}

Propose at most {count} alternative search queries that could surface listings
the attempted queries missed. Return one JSON object:
{{"proposals": [{{"query": "<2 to 6 words>", "purpose": "<what gap it targets>"}}]}}

Rules:
- Never change which item is being bought.
- Keep every literal title term in the query when literal terms are listed.
- Use catalog vocabulary actually seen above where it helps.
- Do not repeat an attempted query. Do not add keys."""

HYPOTHETICAL_TEMPLATE = """Shopper request (abbreviated): {request}
No catalog listing has matched yet.

Write one plausible marketplace listing title for exactly this item, then a
short search query drawn from it. Return one JSON object:
{{"hypothetical_title": "<title>", "query": "<2 to 6 words>"}}

The invented title is a query aid only. It is never a fact about any product."""


def normalise_query(text: Any) -> str:
    return " ".join(tokens(text))


class QueryPlanner:

    def __init__(self, inference: InferenceClient, settings: Settings):
        self.inference = inference
        self.settings = settings
        self.attempted: set[str] = set()

    def register(self, query: str) -> bool:
        key = normalise_query(query)
        if not key or key in self.attempted:
            return False
        self.attempted.add(key)
        return True

    def spec_queries(self, intent: Intent) -> list[str]:
        identity = intent.product_identity.strip()
        noisy = set(intent.noisy_terms)
        out: list[str] = []
        literals: list[str] = []
        for requirement in intent.hard():
            if requirement.field in _UNSEARCHABLE_FIELDS:
                continue
            value = str(requirement.value or "").strip()
            if not value or len(value) > 24:
                continue
            unit = (requirement.unit or "").strip()
            literals.append((value + " " + unit).strip() if unit else value)
        for literal in literals:
            out.append(self.tidy(literal + " " + identity, noisy))
        if len(literals) > 1:
            out.append(self.tidy(" ".join(literals) + " " + identity, noisy))
        return [item for item in out if item]

    @staticmethod
    def preferred_identity(intent: Intent) -> list[tuple[str, str]]:
        order = {"model": 0, "brand": 1, "color": 2}
        found: list[tuple[int, str, str]] = []
        for requirement in intent.requirements:
            if not requirement.active or requirement.field not in order:
                continue
            value = str(requirement.value or "").strip()
            if not value or len(value) > 32:
                continue
            found.append((order[requirement.field], requirement.field, value))
        found.sort()
        return [(field_name, value) for _rank, field_name, value in found]

    def build_initial_queries(self, intent: Intent) -> list[str]:
        identity = intent.product_identity.strip()
        specs: list[str] = []
        for requirement in intent.hard():
            if requirement.field in {"price", "in_stock", "category"}:
                continue
            value = str(requirement.value or "").strip()
            if not value or len(value) > 24:
                continue
            unit = (requirement.unit or "").strip()
            specs.append((value + " " + unit).strip())

        preferred = self.preferred_identity(intent)
        proposals: list[str] = []
        if intent.title_terms:
            proposals.append(" ".join(intent.title_terms))
            if identity:
                proposals.append(" ".join(intent.title_terms) + " " + identity)
        if preferred:
            head = preferred[0][1]
            proposals.append((head + " " + identity).strip())
            if len(preferred) > 1:
                proposals.append(
                    " ".join([head] + [value for _f, value in preferred[1:2]] + [identity]).strip()
                )
        if identity and specs:
            proposals.append((specs[0] + " " + identity).strip())
        for term in intent.search_terms:
            proposals.append(term)
        if identity:
            proposals.append(identity)

        noisy = set(intent.noisy_terms)
        out: list[str] = []
        for proposal in proposals:
            cleaned = self.tidy(proposal, noisy)
            key = normalise_query(cleaned)
            if key and key not in {normalise_query(item) for item in out}:
                out.append(cleaned)
        return out[:6]

    @staticmethod
    def unsearchable_tokens(intent: Intent) -> set[str]:
        banned: set[str] = set()
        for requirement in intent.requirements:
            if not requirement.active or requirement.field not in _UNSEARCHABLE_FIELDS:
                continue
            banned.update(tokens(requirement.value))
            banned.update(tokens(requirement.unit or ""))
        return {token for token in banned if token}

    @staticmethod
    def tidy(text: str, noisy: set[str]) -> str:
        kept: list[str] = []
        seen: set[str] = set()
        for word in str(text or "").split():
            lowered = word.lower()
            if lowered in noisy or lowered in seen:
                continue
            seen.add(lowered)
            kept.append(word)
        return " ".join(kept).strip()

    def reformulate(
        self,
        intent: Intent,
        request: str,
        observed_titles: Sequence[str],
        gap: str,
    ) -> list[str]:
        if not self.settings.use_grounded_reformulation:
            return []
        payload = self.inference.ask_json(
            REFORMULATE_SYSTEM,
            REFORMULATE_TEMPLATE.format(
                request=compact(request, 700),
                identity=intent.product_identity or "unspecified",
                literals=", ".join(intent.title_terms) or "none",
                attempted=", ".join(sorted(self.attempted)) or "none",
                titles="\n".join("- " + compact(title, 140)
                                 for title in observed_titles[:12]) or "- none",
                gap=gap,
                count=self.settings.max_query_proposals,
            ),
            purpose="query reformulation",
            validate=lambda js: isinstance(js.get("proposals"), list),
        )
        return self._accept(payload, intent)

    def hypothetical_query(self, intent: Intent, request: str) -> list[str]:
        if not self.settings.use_hypothetical_query:
            return []
        payload = self.inference.ask_json(
            REFORMULATE_SYSTEM,
            HYPOTHETICAL_TEMPLATE.format(request=compact(request, 700)),
            purpose="hypothetical query",
            validate=lambda js: isinstance(js.get("query"), str),
        )
        if not payload:
            return []
        return self._accept({"proposals": [{"query": payload.get("query")}]}, intent)

    def _accept(self, payload: Any, intent: Intent) -> list[str]:
        proposals = (payload or {}).get("proposals")
        if not isinstance(proposals, list):
            return []
        identity_tokens = set(tokens(intent.product_identity))
        banned = self.unsearchable_tokens(intent)
        out: list[str] = []
        for raw in proposals:
            text = raw.get("query") if isinstance(raw, dict) else raw
            if not isinstance(text, str):
                continue
            cleaned = " ".join(text.split())[:80]
            key = normalise_query(cleaned)
            if not key or key in self.attempted:
                continue
            if key in {normalise_query(item) for item in out}:
                continue
            query_tokens = set(tokens(cleaned))
            if query_tokens & banned:
                continue
            if intent.title_terms and not set(intent.title_terms) <= query_tokens:
                continue
            if identity_tokens and not (identity_tokens & query_tokens):
                continue
            out.append(cleaned)
            if len(out) >= self.settings.max_query_proposals:
                break
        return out


class Retriever:

    def __init__(self, settings: Settings, metrics: RetrievalMetrics):
        self.settings = settings
        self.metrics = metrics
        self.rrf_k = 60

    def ingest(
        self,
        results: Sequence[dict[str, Any]],
        candidates: dict[str, CandidateState],
        query_key: str,
        *,
        intent: Intent,
    ) -> list[str]:
        fresh: list[str] = []
        for rank, row in enumerate(results or []):
            if not isinstance(row, dict):
                continue
            product_id = row.get("product_id")
            if product_id in (None, ""):
                continue
            product_id = str(product_id)
            self.metrics.results_seen += 1
            state = candidates.get(product_id)
            if state is None:
                state = CandidateState(product_id=product_id)
                candidates[product_id] = state
                fresh.append(product_id)
                self.metrics.distinct_products += 1
            else:
                self.metrics.duplicate_results += 1
            if not state.title and row.get("title"):
                state.title = str(row.get("title"))
            if state.brand in (None, "") and row.get("brand"):
                state.brand = row.get("brand")
            sku = row.get("sku")
            if sku is not None and state.variant(sku) is None:
                state.variants.append(
                    {
                        "sku": str(sku),
                        "options": None,
                        "price": row.get("price"),
                        "in_stock": row.get("in_stock"),
                        "_source_action": "search",
                        "_summary_only": True,
                    }
                )
            state.seen_queries.add(query_key)
            state.retrieval_score += 1.0 / (self.rrf_k + rank)
            state.best_rank = min(state.best_rank, rank)
        for product_id in list(candidates):
            candidate = candidates[product_id]
            if self.matches_literals(candidate, intent):
                candidate.rejected_reason = candidate.rejected_reason
        return fresh

    @staticmethod
    def matches_literals(candidate: CandidateState, intent: Intent) -> bool:
        if not intent.title_terms:
            return True
        title_tokens = set(tokens(candidate.title))
        return all(term in title_tokens for term in intent.title_terms)

    def count_matching(
        self, candidates: dict[str, CandidateState], intent: Intent
    ) -> int:
        return sum(
            1
            for candidate in candidates.values()
            if self.matches_literals(candidate, intent)
            and not candidate.rejected_reason
        )


CHANGE_NONE = "none"
CHANGE_STOCKOUT = "stockout"
CHANGE_PRICE = "price_change"
CHANGE_CART_REJECTED = "cart_rejection"
CHANGE_TRANSPORT = "transport_failure"
CHANGE_SHOPPER = "shopper_revision"
CHANGE_DONE = "completion"


@dataclass
class RepairAttempt:
    product_id: str
    sku: str
    revision: int
    reason: str

    def key(self) -> tuple[str, str, int, str]:
        return (self.product_id, self.sku, self.revision, self.reason)


class RecoveryPlanner:

    def __init__(self) -> None:
        self.attempts: dict[tuple[str, str, int, str], int] = {}


    @staticmethod
    def classify(observation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if observation.get("done"):
            return CHANGE_DONE, {}
        if observation.get("error"):
            return CHANGE_CART_REJECTED, {"error": observation.get("error")}
        payload = observation.get("payload") or {}
        event = payload.get("market_event")
        if isinstance(event, dict):
            kind = str(event.get("kind") or "").strip().lower()
            if kind in {"stockout", "out_of_stock"}:
                return CHANGE_STOCKOUT, event
            if kind in {"price_change", "reprice"}:
                return CHANGE_PRICE, event
            return kind or CHANGE_NONE, event
        return CHANGE_NONE, {}


    def note_attempt(self, attempt: RepairAttempt) -> int:
        key = attempt.key()
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]

    def repeated(self, attempt: RepairAttempt) -> bool:
        return self.attempts.get(attempt.key(), 0) > 0


    @staticmethod
    def find_missing_evidence(
        assessments: dict[str, ConstraintAssessment],
        requirements: Sequence[Requirement],
    ) -> list[Requirement]:
        by_id = {r.rid: r for r in requirements}
        return [
            by_id[rid]
            for rid, assessment in assessments.items()
            if assessment.state == UNRESOLVED and rid in by_id and by_id[rid].is_hard
        ]

    @staticmethod
    def relaxation_order(intent: Intent) -> list[str]:
        ordered: list[str] = []
        for entry in intent.permitted_substitutions:
            text = entry.lower()
            for name in ("color", "colour", "model", "brand", "size", "capacity"):
                canonical = "color" if name == "colour" else name
                if name in text and canonical not in ordered:
                    ordered.append(canonical)
        return ordered

    @classmethod
    def relaxation_level(
        cls,
        candidate: CandidateState,
        sku: Any,
        intent: Intent,
        order: Sequence[str],
    ) -> int:
        if not order:
            return 0
        level = 0
        for position, name in enumerate(order):
            wanted = next(
                (
                    r
                    for r in intent.requirements
                    if r.active and r.field == name and r.strength == "preference"
                ),
                None,
            )
            if wanted is None:
                continue
            verdict = supports_selection(wanted, candidate, sku)
            if verdict.state != SUPPORTED:
                level = max(level, position + 1)
        return level

    def generate_alternatives(
        self,
        entries: Sequence[tuple[CandidateState, Any]],
        intent: Intent,
        *,
        excluded: Sequence[tuple[str, str]] = (),
    ) -> list[tuple[CandidateState, Any, int]]:
        order = self.relaxation_order(intent)
        blocked = {(str(pid), str(sku)) for pid, sku in excluded}
        scored: list[tuple[int, float, CandidateState, Any]] = []
        for candidate, sku in entries:
            if (str(candidate.product_id), str(sku)) in blocked:
                continue
            if any(
                supports_selection(requirement, candidate, sku).state != SUPPORTED
                for requirement in intent.hard()
            ):
                continue
            level = self.relaxation_level(candidate, sku, intent, order)
            fit = sum(
                1
                for requirement in intent.preferences()
                if supports_selection(requirement, candidate, sku).state == SUPPORTED
            )
            scored.append((level, -float(fit), candidate, sku))
        scored.sort(key=lambda row: (row[0], row[1], str(row[2].product_id)))
        return [(candidate, sku, level) for level, _fit, candidate, sku in scored]


class HistoryManager:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.entries: list[dict[str, Any]] = []
        self.dropped = 0

    def note(self, role: str, content: str, **extra: Any) -> None:
        entry: dict[str, Any] = {"role": role, "content": content[:2000]}
        entry.update(extra)
        self.entries.append(entry)

    def note_turn(
        self,
        turn: int,
        proposals: Sequence[ActionProposal],
        response: dict[str, Any],
        summaries: Sequence[str],
    ) -> None:
        self.entries.append(
            {
                "role": "assistant",
                "content": "; ".join(
                    item.purpose + " -> " + item.name for item in proposals
                ),
                "turn": turn,
                "actions": [
                    {"name": item.name, "args": item.args} for item in proposals
                ],
                "observations": [compact(text, 700) for text in summaries],
                "user_message": (response.get("user_message") or {}).get("content"),
            }
        )

    _TRIMMED = ["(trimmed)"]

    def render(self) -> list[dict[str, Any]]:
        payload = self.entries
        size = len(json.dumps(payload, default=str).encode("utf-8"))
        limit = self.settings.max_dialogue_bytes
        for _pass in range(2 * len(payload) + 4):
            if size <= limit or len(payload) <= 1:
                break
            trimmed = False
            for entry in payload:
                body = entry.get("observations")
                if body and body != self._TRIMMED:
                    entry["observations"] = list(self._TRIMMED)
                    trimmed = True
                    break
            if not trimmed:
                payload = payload[1:]
                self.dropped += 1
            size = len(json.dumps(payload, default=str).encode("utf-8"))
        if size > limit and payload:
            payload = [
                {
                    "role": "assistant",
                    "content": "dialogue omitted: it exceeded the local size guard",
                    "dropped_entries": self.dropped + len(payload),
                }
            ]
        return payload


REPLY_SYSTEM = (
    "You reply to the shopper on behalf of a buying assistant. You may only "
    "state facts "
    "from the supplied state summary. You never invent a product property, a "
    "price or a stock status. Reply with one JSON object and no prose."
)

REPLY_TEMPLATE = """The shopper just wrote:
<<<
{message}
>>>

Current requirement record:
{requirements}

What you have actually observed:
{state}

Return one JSON object:
{{"changes": [
    {{"kind": "add|replace|remove", "target_id": "<existing id or null>",
      "field": "<field>", "operator": "eq|ne|le|lt|ge|gt|contains|not_contains|exists",
      "value": "<literal value>", "unit": "<unit or null>",
      "strength": "hard|preference", "source_text": "<exact span of the message>"}}
  ],
  "reply": "<two sentences at most, answering the shopper directly using only the observed facts above>"}}

Rules:
- Report a requirement change only when the message states one.
- If the shopper asked a question, the reply must answer it with an observed value.
- If a fact is not in the state summary, say you still need to check it. Never guess."""


class EpisodeController:

    def __init__(
        self,
        problem: dict[str, Any],
        proxy: Any,
        settings: Settings | None = None,
    ):
        self.settings = settings or Settings.from_env()
        environment = problem.get("environment")
        if not isinstance(environment, dict):
            raise AgentError("problem_data is missing its environment block")
        policy = environment.get("policy_view")
        if not isinstance(policy, dict):
            raise AgentError("problem_data is missing policy_view")

        self.problem_id = str(
            problem.get("problem_id") or problem.get("id") or "problem"
        )
        self.query = str(policy.get("query") or problem.get("query") or "")
        self.catalog = ToolCatalog(policy.get("tools"))
        self.budget = WorkBudget(
            policy.get("max_steps") or 1,
            policy.get("max_calls_per_turn") or 1,
            self.settings,
        )
        self.client = EnvironmentClient(
            proxy,
            environment.get("binding") or {},
            self.problem_id,
            budget=self.budget,
        )
        self.inference = InferenceClient(proxy, self.settings, self.budget)
        self.history = HistoryManager(self.settings)

        self.parser = IntentParser(self.inference, self.catalog)
        self.planner = QueryPlanner(self.inference, self.settings)
        self.metrics = RetrievalMetrics()
        self.retriever = Retriever(self.settings, self.metrics)
        self.checker = ConstraintChecker(self.inference)
        self.reviewer = SelectiveReviewer(self.inference, self.settings)
        self.recovery = RecoveryPlanner()
        self.claims = ClaimBuilder(self.catalog)
        self.evidence = EvidenceStore()

        self.intent = Intent()
        self.candidates: dict[str, CandidateState] = {}
        self.cart: list[tuple[str, str]] = []
        self.committed: tuple[str, str] | None = None
        self.committed_at = -1
        self.verified_at = -1
        self.cart_checked_at = -1
        self.pending_messages: list[str] = []
        self.answered_messages = 0
        self.questions_asked = 0
        self.events: list[dict[str, Any]] = []
        self.open_event: dict[str, Any] | None = None
        self.open_event_turn = -1
        self.replies_sent = 0
        self.suppressed_replies = 0
        self._salvage_actions = 0
        self.assessment_cache: dict[str, dict[str, ConstraintAssessment]] = {}
        self.semantic_done: set[str] = set()
        self.failed: list[tuple[str, str]] = []
        self.finished = False
        self.order_placed = False
        self.query_queue: list[str] = []
        self._intended: tuple[str, str] | None = None
        self._matching_at_expansion = 0
        self._replan_marker: tuple[int, int] | None = None
        self._last_query_productive = True
        self._evidence_probes: set[tuple[str, str, str]] = set()
        self.last_purchase_blockers: dict[str, list[dict[str, Any]]] = {}
        self._reinstated: set[str] = set()
        self._replan_queries = False
        self.semantic_verdicts: dict[str, dict[str, ConstraintAssessment]] = {}
        self._asked_inspection: set[str] = set()
        self._dispatched: dict[str, int] = {}
        self.diagnostics: dict[str, Any] = {
            "problem_id": self.problem_id,
            "tools": self.catalog.names(),
            "decisions": [],
            "patches": [],
            "events": [],
            "notes": [],
        }


    def note(self, message: str) -> None:
        self.diagnostics["notes"].append(message)

    def rules_text(self) -> str:
        return self.query

    def _entries(self) -> list[tuple[CandidateState, Any]]:
        out: list[tuple[CandidateState, Any]] = []
        for candidate in self.candidates.values():
            if candidate.rejected_reason:
                continue
            if not self.retriever.matches_literals(candidate, self.intent):
                continue
            variants = candidate.variants or [{"sku": None}]
            for variant in variants:
                sku = variant.get("sku")
                if not self._permitted(candidate, sku):
                    continue
                out.append((candidate, sku))
        return out

    def _permitted(self, candidate: CandidateState, sku: Any = None) -> bool:
        records = self.intent.effective_restrictions()
        if not records:
            return True
        if sku is None:
            skus = [variant.get("sku") for variant in candidate.variants] or [None]
            return any(
                record.permits(candidate.product_id, one)
                for record in records
                for one in skus
            )
        return any(
            record.permits(candidate.product_id, sku) for record in records
        )

    def set_candidate_restrictions(
        self, values: Sequence[Any], *, explicit: bool
    ) -> None:
        self.intent.set_restrictions(
            [
                record
                for record in (
                    CandidateRestriction.parse(item, explicit=explicit, source_text=str(item))
                    for item in values
                )
                if record is not None
            ]
        )

    def assess(
        self, candidate: CandidateState, sku: Any, *, refresh: bool = False
    ) -> dict[str, ConstraintAssessment]:
        key = self._state_key(candidate, sku)
        if refresh or key not in self.assessment_cache:
            table = self.checker.evaluate(candidate, sku, self.intent.requirements)
            for rid, previous in self.semantic_verdicts.get(
                candidate.key(sku), {}
            ).items():
                current = table.get(rid)
                if current is not None and current.state == UNRESOLVED:
                    table[rid] = previous
            self.assessment_cache[key] = table
        return self.assessment_cache[key]

    def feasible(self, candidate: CandidateState, sku: Any) -> bool:
        table = self.assess(candidate, sku)
        return all(
            table.get(requirement.rid, ConstraintAssessment(requirement.rid, UNRESOLVED)).state
            == SUPPORTED
            for requirement in self.intent.hard()
        )

    def contradicted(self, candidate: CandidateState, sku: Any) -> bool:
        table = self.assess(candidate, sku)
        return any(
            table.get(requirement.rid, ConstraintAssessment(requirement.rid, UNRESOLVED)).state
            == CONTRADICTED
            for requirement in self.intent.hard()
        )

    def support_count(self, candidate: CandidateState, sku: Any) -> int:
        table = self.assess(candidate, sku)
        return sum(
            1
            for requirement in self.intent.hard()
            if table.get(requirement.rid, ConstraintAssessment(requirement.rid, UNRESOLVED)).state
            == SUPPORTED
        )

    def preference_fit(self, candidate: CandidateState, sku: Any) -> int:
        table = self.assess(candidate, sku)
        return sum(
            1
            for requirement in self.intent.preferences()
            if table.get(requirement.rid, ConstraintAssessment(requirement.rid, UNRESOLVED)).state
            == SUPPORTED
        )

    def state_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "cart": [
                {"product_id": pid, "sku": sku} for pid, sku in self.cart
            ],
            "requirements": [
                {
                    "id": r.rid,
                    "field": r.field,
                    "operator": r.operator,
                    "value": r.value,
                    "strength": r.strength,
                }
                for r in self.intent.requirements
                if r.active
            ],
        }
        if self.committed:
            candidate = self.candidates.get(self.committed[0])
            if candidate is not None:
                summary["current_choice"] = describe_candidate(
                    candidate, self.committed[1]
                )
                summary["current_choice_assessment"] = {
                    rid: assessment.state
                    for rid, assessment in self.assess(
                        candidate, self.committed[1]
                    ).items()
                }
        if self.events:
            summary["observed_market_events"] = self.events[-3:]
        return summary


    def ingest_view(self, payload: dict[str, Any], action: str) -> CandidateState | None:
        product_id = payload.get("product_id")
        if product_id in (None, ""):
            return None
        product_id = str(product_id)
        candidate = self.candidates.setdefault(
            product_id, CandidateState(product_id=product_id)
        )
        sequence = self.evidence.tick()
        candidate.inspected = True
        if payload.get("title"):
            candidate.title = str(payload.get("title"))
        candidate.brand = payload.get("brand")
        path = payload.get("category_path")
        candidate.category_path = [str(part) for part in path] if isinstance(path, list) else []
        listing_id = payload.get("listing_id")
        candidate.listing_id = str(listing_id) if listing_id else candidate.listing_id
        facts = payload.get("facts")
        candidate.facts = dict(facts) if isinstance(facts, dict) else {}

        for key, value in candidate.facts.items():
            self.evidence.record(
                product_id,
                None,
                canonical_field(key),
                value,
                source_action=action,
                source_path="facts." + str(key),
                source_kind="raw_product_fact",
                observed_at=sequence,
            )
        if candidate.brand:
            self.evidence.record(
                product_id, None, "brand", candidate.brand,
                source_action=action, source_path="brand", source_kind="brand",
                observed_at=sequence,
            )
        if candidate.category_path:
            self.evidence.record(
                product_id, None, "category", candidate.category_path,
                source_action=action, source_path="category_path",
                source_kind="category_path", observed_at=sequence,
            )

        variants = payload.get("variants")
        if isinstance(variants, list):
            fresh: list[dict[str, Any]] = []
            for row in variants:
                if not isinstance(row, dict):
                    continue
                record = dict(row)
                record["_source_action"] = action
                record["_observed_at"] = sequence
                record["_summary_only"] = False
                fresh.append(record)
                sku = record.get("sku")
                for name, kind in (
                    ("price", "current_variant_state"),
                    ("in_stock", "current_variant_state"),
                    ("options", "variant_option"),
                ):
                    if record.get(name) is not None:
                        self.evidence.record(
                            product_id, sku, name if name != "options" else "options",
                            record.get(name),
                            source_action=action,
                            source_path="variants[" + str(sku) + "]." + name,
                            source_kind=kind,
                            observed_at=sequence,
                        )
            if fresh:
                candidate.variants = fresh
        self.assessment_cache = {
            key: value
            for key, value in self.assessment_cache.items()
            if not key.startswith(product_id)
        }
        return candidate

    def ingest_stock(self, payload: dict[str, Any], action: str) -> None:
        product_id = payload.get("product_id")
        sku = payload.get("sku")
        if product_id in (None, "") or sku in (None, ""):
            return
        product_id = str(product_id)
        candidate = self.candidates.setdefault(
            product_id, CandidateState(product_id=product_id)
        )
        sequence = self.evidence.tick()
        variant = candidate.variant(sku)
        previous_price = None if variant is None else variant.get("price")
        previous_stock = None if variant is None else variant.get("in_stock")
        if variant is None:
            variant = {"sku": str(sku)}
            candidate.variants.append(variant)
        variant["price"] = payload.get("price")
        variant["in_stock"] = payload.get("in_stock")
        variant["currency"] = payload.get("currency", variant.get("currency"))
        variant["_source_action"] = action
        variant["_observed_at"] = sequence
        variant["_summary_only"] = False
        self.evidence.invalidate(product_id, sku, ("price", "in_stock"))
        for name in ("price", "in_stock"):
            self.evidence.record(
                product_id, sku, name, payload.get(name),
                source_action=action,
                source_path="variants[" + str(sku) + "]." + name,
                source_kind="current_variant_state",
                observed_at=sequence,
            )
        if previous_price is not None and payload.get("price") is not None:
            if to_decimal(previous_price) != to_decimal(payload.get("price")):
                self.events.append(
                    {
                        "kind": CHANGE_PRICE,
                        "detected": "inspect_stock",
                        "target": {"product_id": product_id, "sku": str(sku)},
                        "old_price": previous_price,
                        "new_price": payload.get("price"),
                    }
                )
        if previous_stock is True and truthy(payload.get("in_stock")) is False:
            self.events.append(
                {
                    "kind": CHANGE_STOCKOUT,
                    "detected": "inspect_stock",
                    "target": {"product_id": product_id, "sku": str(sku)},
                }
            )
        self.assessment_cache = {
            key: value
            for key, value in self.assessment_cache.items()
            if not key.startswith(product_id)
        }

    def ingest_cart(self, payload: dict[str, Any]) -> None:
        rows = payload.get("cart")
        if not isinstance(rows, list):
            return
        self.cart = [
            (str(row.get("product_id")), str(row.get("sku")))
            for row in rows
            if isinstance(row, dict) and row.get("product_id") and row.get("sku")
        ]
        self.cart_checked_at = self.evidence.sequence
        for row in rows:
            if not isinstance(row, dict):
                continue
            price = row.get("current_price", row.get("price"))
            if price is None:
                continue
            product_id = str(row.get("product_id"))
            candidate = self.candidates.get(product_id)
            if candidate is None:
                continue
            variant = candidate.variant(row.get("sku"))
            if variant is not None and to_decimal(variant.get("price")) != to_decimal(price):
                variant["price"] = price
                variant["_observed_at"] = self.evidence.tick()
                self.assessment_cache = {
                    key: value
                    for key, value in self.assessment_cache.items()
                    if not key.startswith(product_id)
                }

    def process_observation(
        self, proposal: ActionProposal, observation: dict[str, Any]
    ) -> ObservationOutcome:
        payload = observation.get("payload") or {}
        kind, detail = self.recovery.classify(observation)
        if kind in {CHANGE_STOCKOUT, CHANGE_PRICE} and detail:
            entry = dict(detail)
            entry["detected"] = proposal.name
            self.events.append(entry)
            self.diagnostics["events"].append(entry)
            self.open_event = entry
            self.open_event_turn = self.budget.turn
            self.note(
                "market event observed; commitment outranks exploration until the "
                "order is placed"
            )
            target = detail.get("target") or {}
            if target.get("product_id"):
                self.evidence.invalidate(
                    str(target.get("product_id")),
                    target.get("sku"),
                    ("price", "in_stock"),
                )
        if observation.get("error"):
            return ObservationOutcome(
                action=proposal.name,
                ok=False,
                summary=proposal.name
                + " rejected: "
                + compact(observation.get("error"), 200),
                error=observation.get("error"),
            )

        action = str(payload.get("action") or proposal.name)
        if action in {"search", "filter"}:
            results = payload.get("results")
            results = results if isinstance(results, list) else []
            query_key = normalise_query(
                proposal.args.get("query")
                or json.dumps(proposal.args, sort_keys=True, default=str)
            )
            fresh = self.retriever.ingest(
                results, self.candidates, query_key, intent=self.intent
            )
            self.metrics.queries_issued += 1
            self._last_query_productive = bool(fresh)
            if fresh:
                self.metrics.productive_queries += 1
            self.metrics.matching_products = self.retriever.count_matching(
                self.candidates, self.intent
            )
            return ObservationOutcome(
                action=action,
                ok=True,
                summary=action
                + ": "
                + str(len(results))
                + " rows, "
                + str(len(fresh))
                + " new products, "
                + str(self.metrics.matching_products)
                + " literal matches so far",
            )

        if "compared" in payload:
            names: list[str] = []
            for row in payload.get("compared") or []:
                if isinstance(row, dict):
                    candidate = self.ingest_view(row, "compare")
                    if candidate is not None:
                        names.append(candidate.product_id)
                        self.metrics.inspected_products += 1
                        if self.retriever.matches_literals(candidate, self.intent):
                            self.metrics.inspected_matching += 1
            return ObservationOutcome(
                action="compare",
                ok=bool(names),
                summary="compare: inspected " + ", ".join(names),
                fields=tuple(names),
            )

        if "variants" in payload and "product_id" in payload:
            candidate = self.ingest_view(payload, "view")
            if candidate is not None:
                self.metrics.inspected_products += 1
                if self.retriever.matches_literals(candidate, self.intent):
                    self.metrics.inspected_matching += 1
                return ObservationOutcome(
                    action="view",
                    ok=True,
                    product_id=candidate.product_id,
                    fields=("facts", "variants"),
                    summary="view " + candidate.product_id + ": " + compact(
                        {
                            "category_path": candidate.category_path,
                            "facts": candidate.facts,
                            "variants": [
                                {
                                    "sku": row.get("sku"),
                                    "options": row.get("options"),
                                    "price": row.get("price"),
                                    "in_stock": row.get("in_stock"),
                                }
                                for row in candidate.variants
                            ],
                        },
                        600,
                    ),
                )
            return ObservationOutcome(
                action="view", ok=False, summary="view: unusable payload"
            )

        if "added" in payload or ("cart" in payload and proposal.name == "add_to_cart"):
            added = payload.get("added")
            if isinstance(added, dict):
                self.ingest_stock(added, "add_to_cart")
            self.ingest_cart(payload)
            confirmed = isinstance(added, dict) and added.get("product_id") is not None
            return ObservationOutcome(
                action="add_to_cart",
                ok=confirmed,
                product_id=str(added.get("product_id")) if confirmed else None,
                sku=str(added.get("sku")) if confirmed else None,
                fields=tuple(
                    name
                    for name in ("price", "in_stock")
                    if isinstance(added, dict) and added.get(name) is not None
                ),
                summary="add_to_cart: cart=" + json.dumps(self.cart),
            )

        if "cart" in payload:
            self.ingest_cart(payload)
            return ObservationOutcome(
                action=proposal.name,
                ok=True,
                fields=("cart",),
                summary=proposal.name + ": cart=" + json.dumps(self.cart),
            )

        if "in_stock" in payload and "sku" in payload:
            self.ingest_stock(payload, "inspect_stock")
            present = tuple(
                name for name in ("price", "in_stock") if payload.get(name) is not None
            )
            return ObservationOutcome(
                action="inspect_stock",
                ok=len(present) == 2,
                product_id=str(payload.get("product_id")),
                sku=str(payload.get("sku")),
                fields=present,
                summary="inspect_stock "
                + str(payload.get("product_id"))
                + "::"
                + str(payload.get("sku"))
                + " price="
                + str(payload.get("price"))
                + " in_stock="
                + str(payload.get("in_stock")),
            )

        if payload.get("sent"):
            return ObservationOutcome(
                action="message", ok=True, summary="message delivered"
            )

        if "order" in payload or payload.get("status"):
            self.order_placed = str(payload.get("status") or "") == "placed"
            return ObservationOutcome(
                action="place_test_order",
                ok=self.order_placed,
                summary="place_test_order: " + compact(payload, 400),
            )

        return ObservationOutcome(
            action=proposal.name,
            ok=False,
            summary=proposal.name + ": " + compact(payload, 300),
        )


    def handle_shopper_message(self, message: str) -> ActionProposal | None:
        payload = self.inference.ask_json(
            REPLY_SYSTEM,
            REPLY_TEMPLATE.format(
                message=compact(message, 1200),
                requirements=json.dumps(
                    [
                        {
                            "id": r.rid,
                            "field": r.field,
                            "operator": r.operator,
                            "value": r.value,
                            "strength": r.strength,
                        }
                        for r in self.intent.requirements
                        if r.active
                    ],
                    default=str,
                )[:2500],
                state=json.dumps(self.state_summary(), default=str)[:4000],
            ),
            purpose="shopper update and reply",
            validate=lambda js: isinstance(js.get("reply"), str)
            or isinstance(js.get("changes"), list),
        )
        patch = self._apply_patch(message, payload)
        self.diagnostics["patches"].append(patch)

        reply = ""
        if isinstance(payload, dict):
            reply = str(payload.get("reply") or "").strip()
        if not reply:
            reply = self._template_reply()
        reply = " ".join(reply.split())[:600]
        self.answered_messages += 1
        if not self.catalog.has("message"):
            self.note("no message tool published; shopper update applied silently")
            return None
        if not self._reply_allowed():
            self.suppressed_replies += 1
            self.note(
                "shopper update applied without a reply; the reply budget is spent"
            )
            return None
        self.replies_sent += 1
        return ActionProposal(
            purpose="answer the shopper",
            name="message",
            args={"content": reply},
            resolves=["shopper_message"],
        )

    def _reply_allowed(self) -> bool:
        if self.replies_sent >= self.settings.max_shopper_replies:
            return False
        reserve = self.budget.reserve(committed=self.committed is not None)
        return self.budget.turns_left > reserve

    def _apply_patch(self, message: str, payload: Any) -> dict[str, Any]:
        self.intent.revision += 1
        patch = self.parser.apply_patch(
            self.intent, message, payload, self.intent.revision
        )
        if patch["added"] or patch["replaced"] or patch["removed"]:
            self.assessment_cache.clear()
            self.semantic_done.clear()
            self.semantic_verdicts.clear()
            self._replan_queries = True
            self.query_queue = []
            self._reinstated.clear()
        return patch

    def _template_reply(self) -> str:
        if not self.committed:
            return (
                "Thanks - noted. I am still comparing listings against your firm "
                "requirements and will confirm the exact item before ordering."
            )
        candidate = self.candidates.get(self.committed[0])
        if candidate is None:
            return "Thanks - noted. I will re-check the current option before ordering."
        sku = self.committed[1]
        variant = candidate.variant(sku) or {}
        parts = [
            "I am going with " + compact(candidate.title, 90),
            "(" + candidate.product_id + " / " + str(sku) + ")",
        ]
        if variant.get("price") is not None:
            parts.append(
                "at " + str(variant.get("price")) + " " + str(variant.get("currency") or "")
            )
        table = self.assess(candidate, sku)
        confirmed = [
            requirement.field + "=" + str(requirement.value)
            for requirement in self.intent.hard()
            if table.get(requirement.rid, ConstraintAssessment(requirement.rid, UNRESOLVED)).state
            == SUPPORTED
        ]
        if confirmed:
            parts.append("confirmed from its product facts: " + ", ".join(confirmed[:4]))
        return " ".join(parts).strip() + "."


    def semantic_pass(self, entries: Sequence[tuple[CandidateState, Any]]) -> None:
        pending = [
            (candidate, sku)
            for candidate, sku in entries
            if candidate.inspected
            and self._judgment_key(candidate, sku) not in self.semantic_done
        ]
        if not pending or not self.budget.inference_allowed():
            return
        labelled = [
            ("C" + str(index + 1), candidate, sku)
            for index, (candidate, sku) in enumerate(pending[: self.settings.inspect_batch])
        ]
        rules_requirement = Requirement(
            rid="r-rules",
            field="task_rules_compliance",
            operator="exists",
            value="the listing satisfies every firm condition in the public task rules",
            strength="hard",
            scope="product",
        )
        judged = list(self.intent.hard()) + list(self.intent.preferences())
        settled = {
            label: {
                rid
                for rid, assessment in self.assess(candidate, sku).items()
                if assessment.state != UNRESOLVED
            }
            for label, candidate, sku in labelled
        }
        view = self.checker.request_semantic_assessment(
            self.rules_text(), labelled, judged + [rules_requirement], settled
        )
        for label, candidate, sku in labelled:
            key = self._judgment_key(candidate, sku)
            self.semantic_done.add(key)
            entry = view.get(label)
            if not isinstance(entry, dict):
                continue
            merged = self.checker.validate_model_assessment(
                entry,
                candidate,
                sku,
                judged + [rules_requirement],
                self.assess(candidate, sku),
            )
            rules_verdict = merged.pop("r-rules", None)
            self.assessment_cache[self._state_key(candidate, sku)] = merged
            self.semantic_verdicts[candidate.key(sku)] = merged
            if rules_verdict is None or rules_verdict.state != CONTRADICTED:
                continue
            if candidate.product_id in self._reinstated:
                continue
            hard_ids = {requirement.rid for requirement in self.intent.hard()}
            if any(
                assessment.state == CONTRADICTED
                for rid, assessment in merged.items()
                if rid in hard_ids
            ):
                continue
            candidate.rejected_reason = (
                "public task rules: " + compact(rules_verdict.reason, 140)
            )
            self.diagnostics["decisions"].append(
                {
                    "soft_rejected": candidate.product_id,
                    "reason": candidate.rejected_reason,
                }
            )

    def _state_key(self, candidate: CandidateState, sku: Any) -> str:
        variant = candidate.variant(sku) or {}
        return (
            candidate.key(sku)
            + "#r"
            + str(self.intent.revision)
            + "#o"
            + str(variant.get("_observed_at") or 0)
        )

    def _judgment_key(self, candidate: CandidateState, sku: Any) -> str:
        variant = candidate.variant(sku) or {}
        material = json.dumps(
            {
                "title": candidate.title,
                "brand": candidate.brand,
                "category_path": candidate.category_path,
                "facts": candidate.facts,
                "options": variant.get("options"),
            },
            sort_keys=True,
            default=str,
        )
        digest = str(abs(hash(material)) % (10 ** 12))
        return candidate.key(sku) + "#r" + str(self.intent.revision) + "#e" + digest

    def claim_gap(self, candidate: CandidateState, sku: Any) -> int:
        return len(
            self.claims.missing_explanation_requirements(
                candidate, sku, self.intent, self.query
            )
        )

    def choose_commit(self) -> tuple[CandidateState, Any] | None:
        feasible = [
            (candidate, sku)
            for candidate, sku in self._entries()
            if self.feasible(candidate, sku)
        ]
        if not feasible:
            return None
        if self.intent.objectives:
            ranked, _blocked = Comparator.rank(feasible, self.intent.objectives)
            if ranked:
                winner = ranked[0]
                rival = Comparator.dominated(winner, ranked, self.intent.objectives)
                if rival is not None:
                    winner = rival
                    self.note("nondominance check moved the pick")
                return winner.candidate, winner.sku

        claim_required = self.claims.justification_required(self.query, self.intent)
        scored: list[tuple[int, int, float, str, CandidateState, Any]] = []
        for candidate, sku in feasible:
            gap = self.claim_gap(candidate, sku) if claim_required else 0
            scored.append(
                (
                    gap,
                    -self.preference_fit(candidate, sku),
                    -candidate.retrieval_score,
                    candidate.product_id,
                    candidate,
                    sku,
                )
            )
        scored.sort(key=lambda row: row[:4])
        best = (scored[0][4], scored[0][5])

        preferences = [
            requirement.field + " " + requirement.operator + " " + str(requirement.value)
            for requirement in self.intent.preferences()
        ]
        if preferences and len(scored) > 1 and self.budget.inference_allowed():
            best = self._model_preference_pick(scored, preferences) or best
        return best

    def _model_preference_pick(
        self,
        scored: Sequence[tuple[int, int, float, str, CandidateState, Any]],
        preferences: Sequence[str],
    ) -> tuple[CandidateState, Any] | None:
        shortlist = [(row[4], row[5]) for row in scored[: self.settings.inspect_batch]]
        labelled = [
            ("C" + str(index + 1), describe_candidate(candidate, sku))
            for index, (candidate, sku) in enumerate(shortlist)
        ]
        payload = self.reviewer.select(self.rules_text(), preferences, labelled)
        if not payload:
            return None
        chosen = str(payload.get("best") or "").strip()
        labels = [label for label, _ in labelled]
        if chosen not in labels:
            return None
        if self.intent.objectives:
            chosen, why = DecisionValidator.check_comparison(
                payload.get("table"), chosen, self.intent.objectives, labels
            )
            self.note("comparison arbitration: " + why)
        chosen, why = self.reviewer.check_order_sensitivity(
            self.rules_text(), preferences, labelled, chosen
        )
        if self.settings.use_order_sensitivity_check:
            self.note("order check: " + why)
        if self.settings.use_candidate_votes:
            voted, tally = self.reviewer.aggregate_votes(
                self.rules_text(), preferences, labelled
            )
            self.note("vote tally: " + json.dumps(tally))
            if voted in labels:
                chosen = voted
        index = labels.index(chosen)
        pick = shortlist[index]
        if self.settings.use_rival_duel and len(shortlist) > 1:
            rival_index = 1 if index == 0 else 0
            winner, why = self.reviewer.compare_rival(
                self.rules_text(),
                describe_candidate(*pick),
                describe_candidate(*shortlist[rival_index]),
                [str(item) for item in (payload.get("unknowns") or [])],
            )
            self.note("duel: " + why)
            if winner == "B":
                pick = shortlist[rival_index]
        self.diagnostics["decisions"].append(
            {
                "preference_pick": pick[0].product_id + "::" + str(pick[1]),
                "reason": compact(payload.get("reason"), 200),
                "unknowns": payload.get("unknowns"),
            }
        )
        return pick


    def _search_cap(self) -> int:
        return max(1, self.catalog.numeric_cap("search", "k", 10))

    def propose_retrieval(self) -> ActionProposal | None:
        if self.intent.effective_restrictions():
            self.release_unusable_restrictions()
        if self.intent.effective_restrictions():
            return None
        if not self.query_queue:
            self._refill_query_queue()
        while self.query_queue:
            query = self.query_queue.pop(0)
            if not self.planner.register(query):
                continue
            if self.catalog.has("search"):
                args: dict[str, Any] = {"query": query}
                if self.catalog.supports_argument("search", "k"):
                    args["k"] = self._search_cap()
                return ActionProposal(
                    purpose="retrieve candidates", name="search", args=args
                )
        return self.propose_structured_filter()

    def propose_structured_filter(self) -> ActionProposal | None:
        if not self.catalog.has("filter"):
            return None
        args: dict[str, Any] = {}
        category = self.intent.category or self._observed_category()
        if category and self.catalog.supports_argument("filter", "category"):
            args["category"] = str(category)
        budget = self.intent.budget()
        if budget is not None and self.catalog.supports_argument("filter", "max_price"):
            args["max_price"] = float(budget)
        brand = next(
            (
                str(r.value)
                for r in self.intent.requirements
                if r.active and r.field == "brand" and r.strength == "hard"
            ),
            None,
        )
        if brand and self.catalog.supports_argument("filter", "brand"):
            args["brand"] = brand
        if self.catalog.supports_argument("filter", "in_stock"):
            args["in_stock"] = True
        if self.catalog.supports_argument("filter", "k"):
            args["k"] = max(1, self.catalog.numeric_cap("filter", "k", 10))
        if not args or set(args) <= {"k", "in_stock"}:
            return None
        key = "filter:" + json.dumps(args, sort_keys=True)
        if not self.planner.register(key):
            return None
        return ActionProposal(
            purpose="narrow the catalog with structured constraints",
            name="filter",
            args=args,
        )

    def event_open(self) -> bool:
        return self.open_event is not None and not self.order_placed

    def event_target(self) -> tuple[str, str] | None:
        if self.open_event is None:
            return None
        target = self.open_event.get("target")
        if not isinstance(target, dict):
            return None
        product_id = target.get("product_id")
        if not product_id:
            return None
        return (str(product_id), str(target.get("sku") or ""))

    def _has_committable(self) -> bool:
        for candidate, sku in self._entries():
            if sku is None or not candidate.inspected:
                continue
            if self.contradicted(candidate, sku):
                continue
            if (candidate.product_id, str(sku)) in self.failed:
                continue
            return True
        return False

    def _event_lockdown(self) -> bool:
        if not self.event_open():
            return False
        if self.committed is not None:
            return True
        return self._has_committable()

    def propose_event_commitment(self) -> ActionProposal | None:
        if not self.event_open() or self.committed is not None:
            return None
        for step in (self.propose_commit, self.propose_last_resort_commit):
            proposal = step()
            if proposal is None:
                continue
            self.diagnostics["decisions"].append(
                {
                    "event_commitment": proposal.target or "",
                    "event_kind": str((self.open_event or {}).get("kind") or ""),
                    "turns_after_event": self.budget.turn - self.open_event_turn,
                }
            )
            return proposal
        return None

    def _may_reason(self) -> bool:
        return self.budget.optional_work_allowed(committed=self.committed is not None)

    def _may_explore(self) -> bool:
        if self._event_lockdown():
            return False
        return self._may_reason()

    def eligible_pool_size(self) -> int:
        budget = self.intent.budget()
        count = 0
        for candidate in self.candidates.values():
            if candidate.rejected_reason:
                continue
            if not self.retriever.matches_literals(candidate, self.intent):
                continue
            usable = False
            for variant in candidate.variants or []:
                if truthy(variant.get("in_stock")) is False:
                    continue
                price = to_decimal(variant.get("price"))
                if budget is not None and price is not None and price > budget:
                    continue
                usable = True
                break
            if usable:
                count += 1
        return count

    def _observed_category(self) -> str | None:
        for candidate in self.candidates.values():
            if candidate.category_path:
                return str(candidate.category_path[-1])
        return None

    def _refill_query_queue(self) -> None:
        if not self.planner.attempted:
            self.query_queue = self.planner.build_initial_queries(self.intent)
            return
        if self._replan_queries:
            self._replan_queries = False
            fresh = [
                query
                for query in (
                    self.planner.spec_queries(self.intent)
                    + self.planner.build_initial_queries(self.intent)
                )
                if normalise_query(query) not in self.planner.attempted
            ]
            if fresh:
                seen: set[str] = set()
                ordered: list[str] = []
                for query in fresh:
                    key = normalise_query(query)
                    if key in seen:
                        continue
                    seen.add(key)
                    ordered.append(query)
                self.query_queue = ordered
                self.diagnostics["decisions"].append(
                    {"replanned_queries": ordered[:4],
                     "requirement_revision": self.intent.revision}
                )
                return
        if self.metrics.expansions >= self.settings.max_expansions:
            return
        if self.metrics.stale_expansions >= self.settings.stale_expansions_before_stop:
            return
        if not self._may_explore():
            return
        matching = self.retriever.count_matching(self.candidates, self.intent)
        if self.metrics.expansions and matching <= self._matching_at_expansion:
            self.metrics.stale_expansions += 1
            if self.metrics.stale_expansions >= self.settings.stale_expansions_before_stop:
                return
        self._matching_at_expansion = matching
        self.metrics.expansions += 1
        titles = [
            candidate.title
            for candidate in self.candidates.values()
            if candidate.title
        ]
        if not self.candidates:
            self.query_queue = self.planner.hypothetical_query(
                self.intent, self.rules_text()
            )
            return
        gap = self._coverage_gap()
        self.query_queue = self.planner.reformulate(
            self.intent, self.rules_text(), titles, gap
        )

    def _coverage_gap(self) -> str:
        matching = self.retriever.count_matching(self.candidates, self.intent)
        unresolved: list[str] = []
        for candidate, sku in self._entries()[:8]:
            for requirement in self.intent.hard():
                table = self.assess(candidate, sku)
                state = table.get(requirement.rid)
                if state is not None and state.state != SUPPORTED:
                    if requirement.field not in unresolved:
                        unresolved.append(requirement.field)
        return (
            str(matching)
            + " distinct listings match so far; still unconfirmed: "
            + (", ".join(unresolved) or "nothing")
        )

    def propose_inspection(self) -> ActionProposal | None:
        requirements = list(self.intent.hard()) or list(self.intent.requirements)
        pool = [
            candidate
            for candidate, _sku in self._entries()
            if not candidate.inspected
            and candidate.product_id not in self._asked_inspection
        ]
        seen: dict[str, CandidateState] = {}
        for candidate in pool:
            seen.setdefault(candidate.product_id, candidate)
        ordered = CandidatePrioritizer.rank_for_inspection(
            list(seen.values()), requirements, limit=self.settings.max_shortlist
        )
        if not ordered:
            return None
        cap = max(1, self.catalog.array_cap("compare", "product_ids", 5))
        batch = [candidate.product_id for candidate in ordered[:cap]]
        if len(batch) >= 2 and self.catalog.has("compare"):
            return ActionProposal(
                purpose="inspect a shortlist",
                name="compare",
                args={"product_ids": batch},
            )
        if self.catalog.has("view"):
            return ActionProposal(
                purpose="inspect a candidate",
                name="view",
                args={"product_id": batch[0]},
                target=batch[0],
            )
        return None

    def propose_restricted_inspection(self) -> ActionProposal | None:
        if not self.intent.effective_restrictions():
            return None
        wanted: list[str] = []
        for record in self.intent.effective_restrictions():
            product_id = record.product_id
            if product_id in self._asked_inspection:
                continue
            candidate = self.candidates.get(product_id)
            if candidate is None or not candidate.inspected:
                if product_id not in wanted:
                    wanted.append(product_id)
        if not wanted:
            return None
        cap = max(1, self.catalog.array_cap("compare", "product_ids", 5))
        if len(wanted) >= 2 and self.catalog.has("compare"):
            return ActionProposal(
                purpose="inspect the permitted candidates",
                name="compare",
                args={"product_ids": wanted[:cap]},
            )
        if self.catalog.has("view"):
            return ActionProposal(
                purpose="inspect a permitted candidate",
                name="view",
                args={"product_id": wanted[0]},
                target=wanted[0],
            )
        return None

    def propose_missing_evidence(self) -> ActionProposal | None:
        blocked: list[tuple[float, CandidateState, Any, list[Requirement]]] = []
        for state, sku in self._entries():
            if self.feasible(state, sku) or self.contradicted(state, sku):
                continue
            missing = self.recovery.find_missing_evidence(
                self.assess(state, sku), self.intent.requirements
            )
            if not missing:
                continue
            ratio, _covered = CoverageScorer.score(
                state, self.intent.hard(), sku
            )
            blocked.append((ratio, state, sku, missing))
        if not blocked:
            return None
        blocked.sort(key=lambda row: -row[0])
        _ratio, state, sku, missing = blocked[0]
        fields = {requirement.field for requirement in missing}
        variant = state.variant(sku) or {}

        if fields & {"price", "in_stock"} and sku is not None:
            key = ("stock", state.product_id, str(sku))
            if key not in self._evidence_probes and self.catalog.has("inspect_stock"):
                self._evidence_probes.add(key)
                return ActionProposal(
                    purpose="resolve an unknown price or stock value",
                    name="inspect_stock",
                    args={"product_id": state.product_id, "sku": str(sku)},
                    resolves=[requirement.rid for requirement in missing],
                    target=state.product_id + "::" + str(sku),
                )
        if (not state.inspected or variant.get("_summary_only")) and self.catalog.has(
            "view"
        ):
            key = ("view", state.product_id, "")
            if key not in self._evidence_probes:
                self._evidence_probes.add(key)
                return ActionProposal(
                    purpose="resolve an unknown specification",
                    name="view",
                    args={"product_id": state.product_id},
                    resolves=[requirement.rid for requirement in missing],
                    target=state.product_id,
                )
        return None

    def release_unusable_restrictions(self) -> bool:
        records = self.intent.effective_restrictions()
        if not records:
            return False
        if self.intent.has_explicit_restrictions():
            return False
        resolved = [
            candidate
            for candidate in self.candidates.values()
            if candidate.inspected and self._permitted(candidate)
        ]
        if resolved:
            return False
        pending = [
            record
            for record in records
            if record.product_id not in self._asked_inspection
        ]
        if pending:
            return False
        self.note(
            "released an inferred candidate restriction that resolved to nothing: "
            + ", ".join(record.text for record in records[:4])
        )
        self.diagnostics["decisions"].append(
            {"released_restrictions": [record.text for record in records]}
        )
        self.intent.set_restrictions([])
        return True

    def release_soft_rejections(self) -> bool:
        released = [
            candidate
            for candidate in self.candidates.values()
            if candidate.rejected_reason
        ]
        if not released:
            return False
        for candidate in released:
            candidate.rejected_reason = None
            self._reinstated.add(candidate.product_id)
        self.semantic_done.clear()
        self.note(
            "reinstated " + str(len(released)) + " rules-rejected candidate(s); "
            "no unrejected candidate remained"
        )
        return True

    def propose_last_resort_commit(self) -> ActionProposal | None:
        if not self.catalog.has("add_to_cart"):
            return None
        scored: list[tuple[int, int, str, CandidateState, Any]] = []
        for candidate, sku in self._entries():
            if sku is None or not candidate.inspected:
                continue
            if self.contradicted(candidate, sku):
                continue
            if (candidate.product_id, str(sku)) in self.failed:
                continue
            residual = [
                blocker.field
                for blocker in self.purchase_blockers((candidate.product_id, sku))
                if blocker.field not in {"cart", "freshness", "target"}
            ]
            if residual:
                continue
            scored.append(
                (
                    -self.support_count(candidate, sku),
                    -self.preference_fit(candidate, sku),
                    candidate.product_id,
                    candidate,
                    sku,
                )
            )
        if not scored:
            return None
        scored.sort(key=lambda row: row[:3])
        candidate, sku = scored[0][3], scored[0][4]
        if (candidate.product_id, str(sku)) in self.cart:
            self.committed = (candidate.product_id, str(sku))
            self.committed_at = self.evidence.sequence
            return None
        self._intended = (candidate.product_id, str(sku))
        self.note(
            "last-resort commit, still subject to the checkout validator: "
            + candidate.product_id
            + "::"
            + str(sku)
        )
        self.diagnostics["decisions"].append(
            {
                "last_resort_commit": candidate.product_id + "::" + str(sku),
                "supported_hard_requirements": self.support_count(candidate, sku),
                "total_hard_requirements": len(self.intent.hard()),
            }
        )
        return ActionProposal(
            purpose="commit the best defensible option before the limit",
            name="add_to_cart",
            args={"product_id": candidate.product_id, "sku": str(sku)},
            target=candidate.product_id + "::" + str(sku),
        )

    def unmet_preferences(self) -> list[Requirement]:
        return self.preference_gap()[1]

    def preference_gap(self) -> tuple[list[str], list[Requirement]]:
        preferences = list(self.intent.preferences())
        if not preferences:
            return [], []
        entries = [
            (candidate, sku)
            for candidate, sku in self._entries()
            if not self.contradicted(candidate, sku)
        ]
        if not entries:
            return [], preferences
        best: tuple[list[str], list[Requirement]] | None = None
        for candidate, sku in entries:
            table = self.assess(candidate, sku)
            matched: list[str] = []
            missing: list[Requirement] = []
            for requirement in preferences:
                state = table.get(
                    requirement.rid, ConstraintAssessment(requirement.rid, UNRESOLVED)
                ).state
                value = str(requirement.value or "").strip()
                if state == SUPPORTED:
                    if value:
                        matched.append(value)
                else:
                    missing.append(requirement)
            if not missing:
                return matched, []
            if best is None or len(missing) < len(best[1]):
                best = (matched, missing)
        return best if best is not None else ([], preferences)

    @staticmethod
    def preference_terms(requirement: Requirement) -> list[str]:
        literal = [
            term
            for term in tokens(requirement.value)
            if term not in _BOOLEAN_WORDS and len(term) > 1
        ]
        if literal:
            return literal
        return [term for term in tokens(requirement.field.replace("_", " "))]

    def propose_preference_work(self) -> ActionProposal | None:
        if self.committed is not None or not self._may_reason():
            return None
        gaps = self.unmet_preferences()
        if not gaps:
            return None
        identity = self.intent.product_identity.strip()
        noisy = set(self.intent.noisy_terms)

        if self.catalog.has("search") and not self.intent.effective_restrictions():
            anchors, _still_missing = self.preference_gap()
            gap_terms = [
                term for requirement in gaps
                for term in self.preference_terms(requirement)
            ]
            plans: list[tuple[str, str]] = []
            if gap_terms:
                plans.append(
                    (" ".join(anchors + gap_terms + [identity]), "all unmet preferences")
                )
            for requirement in gaps:
                terms = self.preference_terms(requirement)
                if terms:
                    plans.append(
                        (" ".join(anchors + terms + [identity]), requirement.field)
                    )
            for raw, purpose in plans:
                query = self.planner.tidy(raw, noisy)
                if not query or not self.planner.register(query):
                    continue
                args: dict[str, Any] = {"query": query}
                if self.catalog.supports_argument("search", "k"):
                    args["k"] = self._search_cap()
                self.diagnostics["decisions"].append(
                    {"preference_search": query, "for": purpose}
                )
                return ActionProposal(
                    purpose="look for the preference the shopper named",
                    name="search",
                    args=args,
                    resolves=[requirement.rid for requirement in gaps],
                )

        wanted_terms = [term for requirement in gaps
                        for term in self.preference_terms(requirement)]
        if wanted_terms and self.catalog.has("view"):
            for candidate, _sku in self._entries():
                if candidate.inspected:
                    continue
                if candidate.product_id in self._asked_inspection:
                    continue
                if not ConstraintNormalizer.mentions(candidate.title, wanted_terms):
                    continue
                self.diagnostics["decisions"].append(
                    {"preference_inspection": candidate.product_id}
                )
                return ActionProposal(
                    purpose="inspect a listing that may carry the named preference",
                    name="view",
                    args={"product_id": candidate.product_id},
                    target=candidate.product_id,
                )
        return None

    def propose_clarification(self) -> ActionProposal | None:
        if not self.catalog.has("message") or self.questions_asked >= 2:
            return None
        if not self.intent.pending_questions:
            return None
        question = self.intent.pending_questions.pop(0)
        extra = ""
        if self.intent.pending_questions and len(question) < 140:
            extra = " " + self.intent.pending_questions.pop(0)
        self.questions_asked += 1
        return ActionProposal(
            purpose="ask a decision-changing question",
            name="message",
            args={"content": (question + extra).strip()[:400]},
            resolves=["clarification"],
        )

    def propose_commit(self) -> ActionProposal | None:
        if not self.catalog.has("add_to_cart"):
            return None
        pick = self.choose_commit()
        if pick is None:
            return None
        candidate, sku = pick
        if sku is None:
            return None
        if (candidate.product_id, str(sku)) in self.cart:
            self.committed = (candidate.product_id, str(sku))
            self.committed_at = self.evidence.sequence
            return None
        self._intended = (candidate.product_id, str(sku))
        return ActionProposal(
            purpose="commit the best compliant option",
            name="add_to_cart",
            args={"product_id": candidate.product_id, "sku": str(sku)},
            target=candidate.product_id + "::" + str(sku),
        )

    def propose_verification(self) -> ActionProposal | None:
        if self.committed is None:
            return None
        product_id, sku = self.committed
        if self.verified_at < self.committed_at and self.catalog.has("inspect_stock"):
            return ActionProposal(
                purpose="confirm the live price and availability of the target",
                name="inspect_stock",
                args={"product_id": product_id, "sku": sku},
                target=product_id + "::" + sku,
            )
        if self.cart_checked_at < self.committed_at and self.catalog.has("inspect_cart"):
            return ActionProposal(
                purpose="confirm the cart before ordering",
                name="inspect_cart",
                args={},
            )
        return None

    def propose_repair(self) -> ActionProposal | None:
        if self.committed is None:
            return None
        product_id, sku = self.committed
        candidate = self.candidates.get(product_id)
        if candidate is None:
            return None
        if self.feasible(candidate, sku):
            return None
        reason = "; ".join(
            requirement.field + ":" + self.assess(candidate, sku)[requirement.rid].state
            for requirement in self.intent.hard()
            if self.assess(candidate, sku).get(requirement.rid) is not None
            and self.assess(candidate, sku)[requirement.rid].state != SUPPORTED
        )
        attempt = RepairAttempt(product_id, sku, self.intent.revision, reason)
        self.recovery.note_attempt(attempt)
        if (product_id, sku) not in self.failed:
            self.failed.append((product_id, sku))
        self.diagnostics["decisions"].append(
            {"repair_from": product_id + "::" + sku, "reason": reason}
        )

        alternatives = self.recovery.generate_alternatives(
            self._entries(), self.intent, excluded=self.failed
        )
        order = self.recovery.relaxation_order(self.intent)
        if order and self._may_reason():
            best_level = alternatives[0][2] if alternatives else len(order) + 1
            if best_level > 1:
                probe = self.propose_nearer_replacement(order)
                if probe is not None:
                    return probe
        chosen = None
        for option in alternatives:
            trial = RepairAttempt(
                option[0].product_id, str(option[1]), self.intent.revision, "replacement"
            )
            if self.recovery.repeated(trial):
                continue
            self.recovery.note_attempt(trial)
            chosen = option
            break
        if chosen is None:
            self.committed = None
            if alternatives:
                self.note("every permitted replacement was already tried unchanged")
            return None
        replacement, replacement_sku, level = chosen
        self._intended = (replacement.product_id, str(replacement_sku))
        self.diagnostics["decisions"].append(
            {
                "repair_to": replacement.product_id + "::" + str(replacement_sku),
                "relaxation_level": level,
            }
        )
        return ActionProposal(
            purpose="replace the unusable choice",
            name="add_to_cart",
            args={
                "product_id": replacement.product_id,
                "sku": str(replacement_sku),
            },
            target=replacement.product_id + "::" + str(replacement_sku),
        )

    def propose_recommit(self) -> ActionProposal | None:
        if self.committed is None:
            return None
        marker = (len(self.events), self.intent.revision)
        if marker == self._replan_marker:
            return None
        self._replan_marker = marker
        best = self.choose_commit()
        if best is None:
            return None
        candidate, sku = best
        key = (candidate.product_id, str(sku))
        if key == self.committed:
            return None
        self._intended = key
        self.diagnostics["decisions"].append(
            {
                "rerank_after_change": key[0] + "::" + key[1],
                "replaced": self.committed[0] + "::" + self.committed[1],
                "events_seen": len(self.events),
                "requirement_revision": self.intent.revision,
            }
        )
        return ActionProposal(
            purpose="re-rank after an observed change",
            name="add_to_cart",
            args={"product_id": candidate.product_id, "sku": str(sku)},
            target=key[0] + "::" + key[1],
        )

    def propose_nearer_replacement(
        self, order: Sequence[str]
    ) -> ActionProposal | None:
        if not self.catalog.has("search") or self.intent.effective_restrictions():
            return None
        held: list[str] = []
        for field_name in order[1:]:
            value = next(
                (
                    str(requirement.value).strip()
                    for requirement in self.intent.preferences()
                    if requirement.field == field_name
                    and str(requirement.value or "").strip()
                ),
                "",
            )
            if value:
                held.append(value)
        if not held:
            return None
        query = self.planner.tidy(
            " ".join(held + [self.intent.product_identity]),
            set(self.intent.noisy_terms),
        )
        if not query or not self.planner.register(query):
            return None
        args: dict[str, Any] = {"query": query}
        if self.catalog.supports_argument("search", "k"):
            args["k"] = self._search_cap()
        self.diagnostics["decisions"].append({"nearer_replacement_search": query})
        return ActionProposal(
            purpose="look for a replacement that relaxes less",
            name="search",
            args=args,
        )

    def propose_cleanup(self) -> ActionProposal | None:
        if self.committed is None or not self.catalog.has("remove_from_cart"):
            return None
        for product_id, sku in self.cart:
            if (product_id, sku) != self.committed:
                return ActionProposal(
                    purpose="remove the superseded cart entry",
                    name="remove_from_cart",
                    args={"product_id": product_id, "sku": sku},
                )
        return None

    def purchase_blockers(
        self, target: tuple[str, Any] | None, *, waive_unresolved: bool = False
    ) -> list[PurchaseBlocker]:
        out: list[PurchaseBlocker] = []
        waived: list[str] = []

        def block(field: str, reason: str, remedy: str = "") -> None:
            out.append(
                PurchaseBlocker(
                    field=field,
                    reason=reason,
                    remedy=remedy,
                    target="" if target is None else str(target[0]) + "::" + str(target[1]),
                )
            )

        if not self.catalog.has("place_test_order"):
            block("tool", "this episode publishes no checkout tool")
            return out
        if target is None:
            block("target", "no candidate has been committed", "select a candidate")
            return out
        product_id, sku = str(target[0]), target[1]

        if self.pending_messages:
            block(
                "shopper_update",
                "a shopper message has not been applied",
                "apply the pending update",
            )
        if not self.intent.requirements:
            block(
                "requirements",
                "no requirement was extracted, so nothing has been checked",
                "re-read the request or ask the shopper",
            )

        candidate = self.candidates.get(product_id)
        if candidate is None:
            block("target", "the target was never observed", "inspect the candidate")
            return out
        if not candidate.inspected:
            block(
                "target",
                "the target has no view or compare record",
                "inspect the candidate",
            )
        if candidate.rejected_reason:
            block("target", "the target is currently rejected: " + candidate.rejected_reason)
        if not self._permitted(candidate, sku):
            block(
                "candidate_scope",
                "the target is outside the shopper's permitted candidates",
                "choose a permitted candidate",
            )

        table = self.assess(candidate, sku)
        for requirement in self.intent.hard():
            state = table.get(requirement.rid)
            verdict = UNRESOLVED if state is None else state.state
            if verdict == SUPPORTED:
                continue
            if waive_unresolved and verdict == UNRESOLVED:
                waived.append(requirement.field)
                continue
            block(
                requirement.field,
                requirement.field + " is " + verdict.lower(),
                "inspect or clarify " + requirement.field,
            )

        if (product_id, str(sku)) not in [(pid, str(s)) for pid, s in self.cart]:
            block("cart", "the target is not in the cart", "add it to the cart")
        if self.verified_at < self.committed_at or self.verified_at < 0:
            block(
                "freshness",
                "current price and stock were not confirmed after the last change",
                "inspect_stock for the target",
            )

        missing = self.claims.missing_explanation_requirements(
            candidate, sku, self.intent, self.query
        )
        for name in missing:
            if waive_unresolved:
                waived.append(name)
                continue
            block(
                name,
                "the requested explanation has no admissible evidence for " + name,
                "inspect the product facts or pick a candidate that supports it",
            )
        if waived:
            self.diagnostics["waived_blockers"] = sorted(set(waived))
        return out

    def ready_to_order(self) -> bool:
        return not self.purchase_blockers(self.committed)

    def _salvage_target(self) -> tuple[str, Any] | None:
        forbidden = self.event_target()
        ranked: list[tuple[int, int, str, str, Any]] = []
        for candidate, sku in self._entries():
            if sku is None or not candidate.inspected:
                continue
            key = (candidate.product_id, str(sku))
            if key in self.failed:
                continue
            if forbidden is not None and key[0] == forbidden[0]:
                continue
            residual = [
                blocker.field
                for blocker in self.purchase_blockers(
                    (candidate.product_id, sku), waive_unresolved=True
                )
                if blocker.field not in {"cart", "freshness", "target"}
            ]
            if residual:
                continue
            ranked.append(
                (
                    -self.support_count(candidate, sku),
                    -self.preference_fit(candidate, sku),
                    candidate.product_id,
                    candidate.product_id,
                    sku,
                )
            )
        if not ranked:
            return None
        ranked.sort(key=lambda row: row[:3])
        return (ranked[0][3], ranked[0][4])

    def propose_salvage(self) -> ActionProposal | None:
        if self.order_placed or self.finished:
            return None
        if not self.catalog.has("place_test_order"):
            return None
        if self._salvage_actions >= self.settings.max_salvage_actions:
            return None
        target = self.committed or self._salvage_target()
        if target is None:
            return None
        product_id, sku = str(target[0]), target[1]
        fields = {
            blocker.field
            for blocker in self.purchase_blockers(target, waive_unresolved=True)
        }
        proposal: ActionProposal | None = None
        if "cart" in fields and self.catalog.has("add_to_cart"):
            self._intended = (product_id, str(sku))
            proposal = ActionProposal(
                purpose="commit the least-blocked option rather than finish empty",
                name="add_to_cart",
                args={"product_id": product_id, "sku": str(sku)},
                target=product_id + "::" + str(sku),
            )
        elif "freshness" in fields and self.catalog.has("inspect_stock"):
            proposal = ActionProposal(
                purpose="confirm the salvage target before ordering",
                name="inspect_stock",
                args={"product_id": product_id, "sku": sku},
                target=product_id + "::" + str(sku),
            )
        elif not fields:
            proposal = self._checkout_action(target)
        if proposal is None:
            return None
        self._salvage_actions += 1
        self.diagnostics["decisions"].append(
            {
                "salvage": product_id + "::" + str(sku),
                "action": proposal.name,
                "turns_left": self.budget.turns_left,
            }
        )
        return proposal

    def propose_order(self, *, force: bool = False) -> ActionProposal | None:
        if not self.catalog.has("place_test_order"):
            return None
        targets: list[tuple[str, Any]] = []
        if self.committed is not None:
            targets.append(self.committed)
        if force:
            for extra in self._cart_targets():
                if extra not in targets:
                    targets.append(extra)
        if not targets:
            return None

        blocked: dict[str, list[dict[str, Any]]] = {}
        for target in targets:
            blockers = self.purchase_blockers(target)
            if blockers:
                blocked[str(target[0]) + "::" + str(target[1])] = [
                    item.as_dict() for item in blockers
                ]
                continue
            proposal = self._checkout_action(target)
            if proposal is not None:
                return proposal
            blocked[str(target[0]) + "::" + str(target[1])] = [
                {"field": "payload", "reason": "no schema-valid checkout payload"}
            ]
        self.last_purchase_blockers = blocked
        return None

    def _cart_targets(self) -> list[tuple[str, Any]]:
        scored: list[tuple[int, int, str, str, Any]] = []
        for product_id, sku in self.cart:
            candidate = self.candidates.get(product_id)
            if candidate is None:
                continue
            scored.append(
                (
                    -self.support_count(candidate, sku),
                    -self.preference_fit(candidate, sku),
                    str(product_id),
                    str(product_id),
                    sku,
                )
            )
        scored.sort(key=lambda row: row[:3])
        return [(row[3], row[4]) for row in scored]

    def _checkout_action(self, target: tuple[str, Any]) -> ActionProposal | None:
        product_id, sku = str(target[0]), target[1]
        candidate = self.candidates.get(product_id)
        if candidate is None:
            return None
        args, claim_diagnostics = self.claims.compile_payload(
            candidate, sku, self.intent, self.query
        )
        problems = self.catalog.validate("place_test_order", args)
        if problems:
            self.note("checkout payload rejected in-agent: " + "; ".join(problems[:3]))
            if claim_diagnostics.get("justification_required"):
                self.diagnostics["checkout"] = claim_diagnostics
                return None
            args = {
                key: value for key, value in args.items() if key != "justification"
            }
            claim_diagnostics["justification_submitted"] = False
            if self.catalog.validate("place_test_order", args):
                return None
        claim_diagnostics["target"] = product_id + "::" + str(sku)
        claim_diagnostics["validated"] = True
        self.diagnostics["checkout"] = claim_diagnostics
        return ActionProposal(
            purpose="place the order for the verified choice",
            name="place_test_order",
            args=args,
            target=product_id + "::" + str(sku),
        )


    def plan_turn(self) -> list[ActionProposal]:
        if self.finished:
            return []

        if self.pending_messages:
            if self.budget.turns_left <= 1:
                order = self.propose_order(force=True)
                if order is not None:
                    self.note("ordered with an unanswered shopper message at the limit")
                    return [order]
            message = "\n".join(self.pending_messages).strip()
            self.pending_messages = []
            if message:
                proposal = self.handle_shopper_message(message)
                if proposal is not None:
                    return [proposal]

        if self.budget.must_finish_now():
            order = self.propose_order(force=True)
            if order is not None:
                return [order]
            if self.committed is None and self.budget.turns_left > 1:
                fallback = self.propose_last_resort_commit()
                if fallback is not None:
                    return [fallback]

        for step in (
            self.propose_restricted_inspection,
            self.propose_repair,
            self.propose_recommit,
        ):
            proposal = step()
            if proposal is not None:
                return [proposal]

        if self.event_open():
            response = self.propose_event_commitment()
            if response is not None:
                return [response]

        eligible_now = self.eligible_pool_size()
        needs_more = self.committed is None and (
            not self.candidates
            or not self.planner.attempted
            or eligible_now < self.intent.min_distinct_candidates
            or (
                bool(self.query_queue)
                and eligible_now < self.settings.min_pool_before_commit
                and self._last_query_productive
            )
        )
        if needs_more and (not self.candidates or self._may_explore()):
            proposal = self.propose_retrieval()
            if proposal is not None:
                return [proposal]

        inspection = self.propose_inspection()
        inspected = sum(
            1 for candidate in self.candidates.values() if candidate.inspected
        )
        if inspection is not None and (inspected < 2 or self.committed is None):
            if self._may_explore() or inspected == 0:
                return [inspection]

        self.semantic_pass(self._entries())

        if self.committed is None:
            clarification = self.propose_clarification()
            if clarification is not None and self._may_explore():
                return [clarification]

        preference_work = self.propose_preference_work()
        if preference_work is not None:
            return [preference_work]

        commit = self.propose_commit()
        if commit is not None:
            return [commit]

        if self.committed is None:
            evidence = self.propose_missing_evidence()
            if evidence is not None:
                return [evidence]
            if self._may_explore():
                for step in (self.propose_retrieval, self.propose_inspection):
                    proposal = step()
                    if proposal is not None:
                        return [proposal]
            if self.release_unusable_restrictions() or self.release_soft_rejections():
                retry = self.propose_commit()
                if retry is not None:
                    return [retry]
                for step in (self.propose_inspection, self.propose_retrieval):
                    proposal = step()
                    if proposal is not None:
                        return [proposal]
            fallback = self.propose_last_resort_commit()
            if fallback is not None:
                return [fallback]
            order = self.propose_order(force=True)
            if order is not None:
                return [order]
            salvage = self.propose_salvage()
            if salvage is not None:
                return [salvage]
            self.note("no candidate could be verified and none was defensible")
            return []

        verification = self.propose_verification()
        if verification is not None:
            return [verification]

        cleanup = self.propose_cleanup()
        if cleanup is not None and self.budget.optional_work_allowed():
            return [cleanup]

        order = self.propose_order()
        if order is not None:
            return [order]

        if (
            self._may_explore()
            and self.eligible_pool_size() < self.intent.min_distinct_candidates
        ):
            proposal = self.propose_retrieval()
            if proposal is not None:
                return [proposal]

        order = self.propose_order(force=True)
        if order is not None:
            return [order]
        salvage = self.propose_salvage()
        if salvage is not None:
            return [salvage]
        return []


    def run(self) -> list[dict[str, Any]]:
        self.history.note("user", self.query)
        try:
            self.intent = self.parser.parse(self.query)
        except Exception as exc:
            self.intent = Intent()
            self.note("intent parsing raised " + type(exc).__name__)
        self._seed_category_requirement()
        self.diagnostics["intent_initial"] = self._intent_snapshot()

        while not self.finished and self.budget.turns_left > 0:
            if self.budget.time_left() <= 0:
                self.note("hard deadline reached; stopping without further actions")
                break
            try:
                proposals = self.plan_turn()
            except Exception as exc:
                self.note("planning raised " + type(exc).__name__ + ": " + str(exc)[:200])
                proposals = self._fallback_proposals()
            if not proposals:
                proposals = self._salvage_before_exit()
            if not proposals:
                break

            checked: list[ActionProposal] = []
            for proposal in proposals:
                problems = self.catalog.validate(proposal.name, proposal.args)
                if problems:
                    self.note(
                        "dropped invalid action "
                        + proposal.name
                        + ": "
                        + "; ".join(problems[:2])
                    )
                    continue
                if self._is_no_progress_repeat(proposal):
                    self.note(
                        "suppressed a repeated no-progress action: " + proposal.name
                    )
                    continue
                checked.append(proposal)
            if not checked:
                checked = self._salvage_before_exit()
            if not checked:
                break
            checked = checked[: self.budget.max_calls_per_turn]

            try:
                response = self.client.call(checked)
            except AgentError as exc:
                self.note("environment call failed: " + str(exc))
                break

            summaries: list[str] = []
            outcomes: list[ObservationOutcome | None] = []
            try:
                paired = self.client.pair(checked, response)
            except AgentError as exc:
                self.note("response pairing failed: " + str(exc))
                break
            for proposal, observation in paired:
                try:
                    outcome = self.process_observation(proposal, observation)
                except Exception as exc:
                    outcome = ObservationOutcome(
                        action=proposal.name,
                        ok=False,
                        summary="observation processing raised " + type(exc).__name__,
                    )
                outcomes.append(outcome)
                summaries.append(outcome.summary)
                if observation.get("done"):
                    self.finished = True
                    self.client.done = True

            self._settle_turn(checked, outcomes)
            message = (response.get("user_message") or {}).get("content")
            if isinstance(message, str) and message.strip():
                self.pending_messages.append(message.strip())
            self.budget.note_turn(len(checked))
            self.history.note_turn(self.budget.turn, checked, response, summaries)

        self.diagnostics["intent"] = self._intent_snapshot()
        self.diagnostics["terminal"] = self._terminal_state()
        self.diagnostics["retrieval"] = self.metrics.as_dict()
        self.diagnostics["budget"] = self.budget.as_dict()
        self.diagnostics["order_placed"] = self.order_placed
        self.diagnostics["shopper_replies"] = {
            "sent": self.replies_sent,
            "suppressed": self.suppressed_replies,
            "applied": self.answered_messages,
        }
        self.diagnostics["market_event"] = {
            "open": self.event_open(),
            "observed": len(self.events),
            "fired_turn": self.open_event_turn,
            "salvage_actions": self._salvage_actions,
        }
        self.diagnostics["cart"] = [
            {"product_id": pid, "sku": sku} for pid, sku in self.cart
        ]
        self.diagnostics["committed"] = (
            None if self.committed is None
            else self.committed[0] + "::" + self.committed[1]
        )
        self.diagnostics["review_notes"] = self.reviewer.notes
        self.diagnostics["inference_json_failures"] = self.inference.failures
        self.history.note(
            "assistant",
            "episode diagnostics",
            diagnostics=self.diagnostics,
        )
        return self.history.render()

    def _salvage_before_exit(self) -> list[ActionProposal]:
        if self.finished or self.order_placed:
            return []
        if self.budget.turns_left <= 0 or self.budget.time_left() <= 0:
            return []
        try:
            proposal = self.propose_salvage()
        except Exception as exc:
            self.note("salvage raised " + type(exc).__name__)
            return []
        if proposal is None:
            return []
        if self.catalog.validate(proposal.name, proposal.args):
            return []
        self.note("salvaging the episode rather than returning without an order")
        return [proposal]

    _REPEAT_LIMIT = {
        "search": 1,
        "filter": 1,
        "view": 1,
        "compare": 1,
        "message": 3,
        "inspect_stock": 3,
        "inspect_cart": 3,
        "add_to_cart": 1,
        "remove_from_cart": 1,
        "place_test_order": 1,
    }

    def _action_key(self, proposal: ActionProposal) -> str:
        return (
            proposal.name
            + ":"
            + json.dumps(proposal.args, sort_keys=True, default=str)
            + "#r"
            + str(self.intent.revision)
            + "#e"
            + str(len(self.events))
        )

    def _is_no_progress_repeat(self, proposal: ActionProposal) -> bool:
        key = self._action_key(proposal)
        limit = self._REPEAT_LIMIT.get(proposal.name, 2)
        return self._dispatched.get(key, 0) >= limit

    def _settle_turn(
        self,
        proposals: Sequence[ActionProposal],
        outcomes: Sequence[ObservationOutcome | None] | None = None,
    ) -> None:
        results = list(outcomes or [])
        while len(results) < len(proposals):
            results.append(None)
        for proposal, outcome in zip(proposals, results):
            key = self._action_key(proposal)
            self._dispatched[key] = self._dispatched.get(key, 0) + 1
            if proposal.name == "view":
                self._asked_inspection.add(str(proposal.args.get("product_id")))
            if proposal.name == "compare":
                self._asked_inspection.update(
                    str(item) for item in (proposal.args.get("product_ids") or [])
                )
            if proposal.name == "add_to_cart" and self._intended is not None:
                if self._intended in self.cart:
                    self.committed = self._intended
                    self.committed_at = self.evidence.sequence
                    self.verified_at = -1
                    self.cart_checked_at = -1
                self._intended = None
            if proposal.name == "inspect_stock" and self.committed is not None:
                if outcome is not None and outcome.confirms(
                    self.committed[0], self.committed[1], "price", "in_stock"
                ):
                    self.verified_at = self.evidence.sequence

    def _terminal_state(self) -> dict[str, Any]:
        if self.order_placed:
            reason = ""
        elif not self.candidates:
            reason = "no candidate was ever observed"
        else:
            target = self.committed
            if target is None:
                best: tuple[int, tuple[str, Any]] | None = None
                for candidate, sku in self._entries():
                    if not candidate.inspected:
                        continue
                    count = len(self.purchase_blockers((candidate.product_id, sku)))
                    if best is None or count < best[0]:
                        best = (count, (candidate.product_id, sku))
                target = best[1] if best else None
            if target is None:
                reason = "no candidate satisfied every firm requirement"
            else:
                blockers = [
                    item
                    for item in self.purchase_blockers(target)
                    if item.field not in {"cart", "freshness"}
                ] or self.purchase_blockers(target)
                reason = "; ".join(
                    item.field + ": " + item.reason for item in blockers[:4]
                ) or "checkout was never dispatched"
        return {
            "complete": bool(self.order_placed),
            "environment_done": bool(self.finished),
            "order_placed": bool(self.order_placed),
            "reason": reason,
            "note": "reward is assigned by the runtime receipt, never here",
            "outstanding_blockers": self.last_purchase_blockers,
        }

    def _fallback_proposals(self) -> list[ActionProposal]:
        order = self.propose_order()
        if order is not None:
            return [order]
        for step in (
            self.propose_verification,
            self.propose_inspection,
            self.propose_retrieval,
        ):
            try:
                proposal = step()
            except Exception:
                proposal = None
            if proposal is not None:
                return [proposal]
        if self.catalog.has("inspect_cart"):
            return [
                ActionProposal(
                    purpose="re-read the cart after a planning failure",
                    name="inspect_cart",
                    args={},
                )
            ]
        return []

    def _seed_category_requirement(self) -> None:
        if not self.intent.category:
            return
        if self.intent.by_field("category"):
            return
        self.intent.requirements.append(
            Requirement(
                rid="r-category",
                field="category",
                operator="eq",
                value=self.intent.category,
                strength="hard",
                scope="product",
                source_text="public task rules",
            )
        )

    def _intent_snapshot(self) -> dict[str, Any]:
        return {
            "product_identity": self.intent.product_identity,
            "category": self.intent.category,
            "requirements": [
                {
                    "id": r.rid,
                    "field": r.field,
                    "operator": r.operator,
                    "value": r.value,
                    "unit": r.unit,
                    "strength": r.strength,
                    "scope": r.scope,
                    "source_text": compact(r.source_text, 120),
                }
                for r in self.intent.requirements
            ],
            "objectives": [
                {"field": o.field, "direction": o.direction} for o in self.intent.objectives
            ],
            "candidate_restrictions": [
                {
                    "product_id": record.product_id,
                    "sku": record.sku,
                    "explicit": record.explicit,
                }
                for record in self.intent.effective_restrictions()
            ],
            "permitted_substitutions": self.intent.permitted_substitutions,
            "title_terms": self.intent.title_terms,
            "explanation_required": self.intent.explanation_required,
            "explanation_fields": self.intent.explanation_fields,
            "pending_questions": self.intent.pending_questions,
            "notes": self.intent.notes,
            "justification_required": self.claims.justification_required(
                self.query, self.intent
            ),
        }


_PROXY: Any = None


def _proxy_client() -> Any:
    global _PROXY
    if _PROXY is None:
        _PROXY = ProxyClient(timeout=120, max_retries=2)
    return _PROXY


AUX_EPSILON = Decimal("0.000001")
AUX_DEFAULT_WIDTH = 16
AUX_DEFAULT_DEPTH = 4
AUX_MAX_SPAN = 4096
AUX_SCALE_STEPS = (1, 2, 5, 10, 20, 50)
AUX_LEVELS = ("low", "normal", "high", "critical")
AUX_MODES = ("collect", "reduce", "emit")


def aux_sign(value: Any) -> int:
    number = to_decimal(value)
    if number is None or number == 0:
        return 0
    return 1 if number > 0 else -1


def aux_clamp(value: Any, lower: Any, upper: Any) -> Decimal | None:
    number = to_decimal(value)
    low = to_decimal(lower)
    high = to_decimal(upper)
    if number is None:
        return None
    if low is not None and number < low:
        return low
    if high is not None and number > high:
        return high
    return number


def aux_sqrt(value: Any, rounds: int = 24) -> Decimal | None:
    number = to_decimal(value)
    if number is None or number < 0:
        return None
    if number == 0:
        return Decimal(0)
    guess = number if number >= 1 else Decimal(1)
    for _step in range(max(1, rounds)):
        nxt = (guess + number / guess) / 2
        if abs(nxt - guess) < AUX_EPSILON:
            return nxt
        guess = nxt
    return guess


def aux_mean(values: Iterable[Any]) -> Decimal | None:
    numbers = [n for n in (to_decimal(v) for v in values) if n is not None]
    if not numbers:
        return None
    return sum(numbers) / Decimal(len(numbers))


def aux_median(values: Iterable[Any]) -> Decimal | None:
    numbers = sorted(n for n in (to_decimal(v) for v in values) if n is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2


def aux_percentile(values: Iterable[Any], share: float) -> Decimal | None:
    numbers = sorted(n for n in (to_decimal(v) for v in values) if n is not None)
    if not numbers:
        return None
    position = max(0.0, min(1.0, float(share))) * (len(numbers) - 1)
    low = int(position)
    high = min(low + 1, len(numbers) - 1)
    if low == high:
        return numbers[low]
    weight = Decimal(str(position - low))
    return numbers[low] + (numbers[high] - numbers[low]) * weight


def aux_span(values: Iterable[Any]) -> tuple[Decimal, Decimal] | None:
    numbers = [n for n in (to_decimal(v) for v in values) if n is not None]
    if not numbers:
        return None
    return (min(numbers), max(numbers))


def aux_normalise(values: Sequence[Any]) -> list[Decimal]:
    numbers = [to_decimal(v) or Decimal(0) for v in values]
    bounds = aux_span(numbers)
    if bounds is None:
        return []
    low, high = bounds
    if high == low:
        return [Decimal(0) for _ in numbers]
    return [(number - low) / (high - low) for number in numbers]


def aux_rank(values: Sequence[Any], *, descending: bool = True) -> list[int]:
    order = sorted(
        range(len(values)),
        key=lambda index: (to_decimal(values[index]) or Decimal(0)),
        reverse=descending,
    )
    ranks = [0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = position
    return ranks


def aux_unique(items: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


def aux_chunk(items: Sequence[Any], size: int) -> list[list[Any]]:
    width = max(1, int(size))
    return [list(items[start : start + width]) for start in range(0, len(items), width)]


def aux_flatten(items: Iterable[Any], depth: int = AUX_DEFAULT_DEPTH) -> list[Any]:
    out: list[Any] = []
    for item in items:
        if depth > 0 and isinstance(item, (list, tuple)):
            out.extend(aux_flatten(item, depth - 1))
            continue
        out.append(item)
    return out


def aux_pairs(items: Sequence[Any]) -> list[tuple[Any, Any]]:
    return [
        (items[left], items[right])
        for left in range(len(items))
        for right in range(left + 1, len(items))
    ]


def aux_windows(items: Sequence[Any], size: int) -> list[list[Any]]:
    width = max(1, int(size))
    if len(items) < width:
        return []
    return [list(items[start : start + width]) for start in range(len(items) - width + 1)]


def aux_group_by(items: Iterable[Any], key: Callable[[Any], Any]) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = {}
    for item in items:
        out.setdefault(key(item), []).append(item)
    return out


def aux_partition(
    items: Iterable[Any], predicate: Callable[[Any], bool]
) -> tuple[list[Any], list[Any]]:
    yes: list[Any] = []
    no: list[Any] = []
    for item in items:
        (yes if predicate(item) else no).append(item)
    return yes, no


def aux_transpose(rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [
        [row[column] if column < len(row) else None for row in rows]
        for column in range(width)
    ]


def aux_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    first, second = set(left), set(right)
    if not first and not second:
        return 1.0
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


def aux_dice(left: Iterable[Any], right: Iterable[Any]) -> float:
    first, second = set(left), set(right)
    total = len(first) + len(second)
    if not total:
        return 1.0
    return 2 * len(first & second) / total


def aux_containment(left: Iterable[Any], right: Iterable[Any]) -> float:
    first, second = set(left), set(right)
    if not first:
        return 1.0
    return len(first & second) / len(first)


def aux_hamming(left: Sequence[Any], right: Sequence[Any]) -> int:
    width = max(len(left), len(right))
    distance = 0
    for index in range(width):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            distance += 1
    return distance


def aux_levenshtein(left: str, right: str, ceiling: int = AUX_MAX_SPAN) -> int:
    first, second = str(left), str(right)
    if first == second:
        return 0
    if abs(len(first) - len(second)) > ceiling:
        return ceiling
    previous = list(range(len(second) + 1))
    for row, symbol in enumerate(first, start=1):
        current = [row]
        for column, other in enumerate(second, start=1):
            cost = 0 if symbol == other else 1
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + cost)
            )
        previous = current
        if min(previous) > ceiling:
            return ceiling
    return min(previous[-1], ceiling)


def aux_lcs_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for column, other in enumerate(right, start=1):
            if item == other:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[column - 1]))
        previous = current
    return previous[-1]


def aux_prefix_length(left: str, right: str) -> int:
    width = 0
    for a, b in zip(str(left), str(right)):
        if a != b:
            break
        width += 1
    return width


def aux_ngrams(items: Sequence[Any], size: int) -> list[tuple[Any, ...]]:
    width = max(1, int(size))
    if len(items) < width:
        return []
    return [tuple(items[start : start + width]) for start in range(len(items) - width + 1)]


class AuxRingBuffer:

    def __init__(self, capacity: int = AUX_DEFAULT_WIDTH):
        self.capacity = max(1, int(capacity))
        self._items: list[Any] = []
        self._start = 0
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._items)

    def push(self, item: Any) -> Any:
        evicted = None
        if len(self._items) < self.capacity:
            self._items.append(item)
            return None
        evicted = self._items[self._start]
        self._items[self._start] = item
        self._start = (self._start + 1) % self.capacity
        self.dropped += 1
        return evicted

    def extend(self, items: Iterable[Any]) -> list[Any]:
        evicted: list[Any] = []
        for item in items:
            gone = self.push(item)
            if gone is not None:
                evicted.append(gone)
        return evicted

    def items(self) -> list[Any]:
        if len(self._items) < self.capacity:
            return list(self._items)
        return self._items[self._start :] + self._items[: self._start]

    def newest(self) -> Any:
        rows = self.items()
        return rows[-1] if rows else None

    def oldest(self) -> Any:
        rows = self.items()
        return rows[0] if rows else None

    def clear(self) -> None:
        self._items = []
        self._start = 0

    def full(self) -> bool:
        return len(self._items) >= self.capacity


class AuxRollingStat:

    def __init__(self, window: int = AUX_DEFAULT_WIDTH):
        self.window = AuxRingBuffer(window)
        self.count = 0
        self.total = Decimal(0)

    def observe(self, value: Any) -> None:
        number = to_decimal(value)
        if number is None:
            return
        self.window.push(number)
        self.count += 1
        self.total += number

    def mean(self) -> Decimal | None:
        return aux_mean(self.window.items())

    def median(self) -> Decimal | None:
        return aux_median(self.window.items())

    def spread(self) -> Decimal | None:
        bounds = aux_span(self.window.items())
        if bounds is None:
            return None
        return bounds[1] - bounds[0]

    def variance(self) -> Decimal | None:
        rows = self.window.items()
        centre = aux_mean(rows)
        if centre is None or len(rows) < 2:
            return None
        squares = sum((row - centre) * (row - centre) for row in rows)
        return squares / Decimal(len(rows) - 1)

    def deviation(self) -> Decimal | None:
        return aux_sqrt(self.variance())

    def lifetime_mean(self) -> Decimal | None:
        if not self.count:
            return None
        return self.total / Decimal(self.count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": str(self.mean() or ""),
            "median": str(self.median() or ""),
            "spread": str(self.spread() or ""),
        }


class AuxHistogram:

    def __init__(self, edges: Sequence[Any] | None = None):
        parsed = [to_decimal(edge) for edge in (edges or AUX_SCALE_STEPS)]
        self.edges = sorted(edge for edge in parsed if edge is not None)
        self.buckets = [0] * (len(self.edges) + 1)
        self.total = 0

    def observe(self, value: Any) -> int:
        number = to_decimal(value)
        if number is None:
            return -1
        slot = len(self.edges)
        for index, edge in enumerate(self.edges):
            if number <= edge:
                slot = index
                break
        self.buckets[slot] += 1
        self.total += 1
        return slot

    def share(self, index: int) -> float:
        if not self.total or index < 0 or index >= len(self.buckets):
            return 0.0
        return self.buckets[index] / self.total

    def busiest(self) -> int:
        if not self.total:
            return -1
        return max(range(len(self.buckets)), key=lambda index: self.buckets[index])

    def as_dict(self) -> dict[str, Any]:
        return {
            "edges": [str(edge) for edge in self.edges],
            "buckets": list(self.buckets),
            "total": self.total,
        }


class AuxLruCache:

    def __init__(self, capacity: int = AUX_DEFAULT_WIDTH):
        self.capacity = max(1, int(capacity))
        self._store: dict[str, Any] = {}
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._store)

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._store:
            self.misses += 1
            return default
        self.hits += 1
        self._touch(key)
        return self._store[key]

    def put(self, key: str, value: Any) -> Any:
        evicted = None
        if key not in self._store and len(self._store) >= self.capacity:
            oldest = self._order.pop(0)
            evicted = self._store.pop(oldest, None)
            self.evictions += 1
        self._store[key] = value
        self._touch(key)
        return evicted

    def contains(self, key: str) -> bool:
        return key in self._store

    def drop(self, key: str) -> bool:
        if key not in self._store:
            return False
        self._store.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        return True

    def clear(self) -> None:
        self._store = {}
        self._order = []

    def hit_rate(self) -> float:
        looked = self.hits + self.misses
        return self.hits / looked if looked else 0.0


class AuxTokenBucket:

    def __init__(self, capacity: int = AUX_DEFAULT_WIDTH, refill_per_second: float = 1.0):
        self.capacity = max(1, int(capacity))
        self.refill = max(0.0, float(refill_per_second))
        self.tokens = float(self.capacity)
        self.stamp = time.monotonic()
        self.granted = 0
        self.refused = 0

    def _replenish(self, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        elapsed = max(0.0, moment - self.stamp)
        self.stamp = moment
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill)

    def take(self, count: int = 1, now: float | None = None) -> bool:
        self._replenish(now)
        wanted = max(1, int(count))
        if self.tokens < wanted:
            self.refused += 1
            return False
        self.tokens -= wanted
        self.granted += 1
        return True

    def available(self, now: float | None = None) -> int:
        self._replenish(now)
        return int(self.tokens)

    def wait_for(self, count: int = 1, now: float | None = None) -> float:
        self._replenish(now)
        wanted = max(1, int(count))
        if self.tokens >= wanted:
            return 0.0
        if self.refill <= 0:
            return float(AUX_MAX_SPAN)
        return (wanted - self.tokens) / self.refill


class AuxBackoffPolicy:

    def __init__(self, base: float = 0.5, factor: float = 2.0, ceiling: float = 30.0):
        self.base = max(0.0, float(base))
        self.factor = max(1.0, float(factor))
        self.ceiling = max(self.base, float(ceiling))

    def delay(self, attempt: int) -> float:
        step = max(0, int(attempt))
        value = self.base
        for _index in range(step):
            value *= self.factor
            if value >= self.ceiling:
                return self.ceiling
        return min(value, self.ceiling)

    def schedule(self, attempts: int) -> list[float]:
        return [self.delay(index) for index in range(max(0, int(attempts)))]

    def total(self, attempts: int) -> float:
        return sum(self.schedule(attempts))

    def attempts_within(self, allowance: float) -> int:
        remaining = float(allowance)
        count = 0
        while remaining > 0:
            step = self.delay(count)
            if step > remaining:
                break
            remaining -= step
            count += 1
            if count > AUX_MAX_SPAN:
                break
        return count


class AuxDeadline:

    def __init__(self, allowance: float, *, started: float | None = None):
        self.allowance = max(0.0, float(allowance))
        self.started = time.monotonic() if started is None else float(started)

    def elapsed(self, now: float | None = None) -> float:
        moment = time.monotonic() if now is None else now
        return max(0.0, moment - self.started)

    def remaining(self, now: float | None = None) -> float:
        return self.allowance - self.elapsed(now)

    def expired(self, now: float | None = None) -> bool:
        return self.remaining(now) <= 0

    def fraction_used(self, now: float | None = None) -> float:
        if self.allowance <= 0:
            return 1.0
        return min(1.0, self.elapsed(now) / self.allowance)

    def split(self, parts: int) -> list[float]:
        count = max(1, int(parts))
        slice_size = self.allowance / count
        return [slice_size for _index in range(count)]

    def child(self, share: float) -> "AuxDeadline":
        portion = max(0.0, min(1.0, float(share)))
        return AuxDeadline(self.allowance * portion, started=self.started)


class AuxHeap:

    def __init__(self, items: Iterable[tuple[Any, Any]] | None = None):
        self._rows: list[tuple[Any, int, Any]] = []
        self._serial = 0
        for priority, payload in items or []:
            self.push(priority, payload)

    def __len__(self) -> int:
        return len(self._rows)

    def push(self, priority: Any, payload: Any) -> None:
        self._serial += 1
        self._rows.append((priority, self._serial, payload))
        self._sift_up(len(self._rows) - 1)

    def pop(self) -> tuple[Any, Any] | None:
        if not self._rows:
            return None
        top = self._rows[0]
        last = self._rows.pop()
        if self._rows:
            self._rows[0] = last
            self._sift_down(0)
        return (top[0], top[2])

    def peek(self) -> tuple[Any, Any] | None:
        if not self._rows:
            return None
        return (self._rows[0][0], self._rows[0][2])

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._rows[parent][:2] <= self._rows[index][:2]:
                break
            self._rows[parent], self._rows[index] = self._rows[index], self._rows[parent]
            index = parent

    def _sift_down(self, index: int) -> None:
        size = len(self._rows)
        while True:
            left = index * 2 + 1
            right = left + 1
            smallest = index
            if left < size and self._rows[left][:2] < self._rows[smallest][:2]:
                smallest = left
            if right < size and self._rows[right][:2] < self._rows[smallest][:2]:
                smallest = right
            if smallest == index:
                return
            self._rows[smallest], self._rows[index] = self._rows[index], self._rows[smallest]
            index = smallest

    def drain(self, limit: int = AUX_MAX_SPAN) -> list[Any]:
        out: list[Any] = []
        while self._rows and len(out) < max(0, int(limit)):
            row = self.pop()
            if row is None:
                break
            out.append(row[1])
        return out


class AuxDisjointSet:

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        key = str(item)
        if key in self.parent:
            return
        self.parent[key] = key
        self.rank[key] = 0

    def find(self, item: str) -> str:
        key = str(item)
        self.add(key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, left: str, right: str) -> bool:
        first, second = self.find(left), self.find(right)
        if first == second:
            return False
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1
        return True

    def connected(self, left: str, right: str) -> bool:
        return self.find(left) == self.find(right)

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in self.parent:
            out.setdefault(self.find(key), []).append(key)
        return out

    def count(self) -> int:
        return len(self.groups())


class AuxGraph:

    def __init__(self) -> None:
        self.edges: dict[str, list[str]] = {}
        self.weights: dict[tuple[str, str], Decimal] = {}

    def add_node(self, node: str) -> None:
        self.edges.setdefault(str(node), [])

    def add_edge(self, source: str, target: str, weight: Any = 1) -> None:
        first, second = str(source), str(target)
        self.add_node(first)
        self.add_node(second)
        if second not in self.edges[first]:
            self.edges[first].append(second)
        self.weights[(first, second)] = to_decimal(weight) or Decimal(0)

    def neighbours(self, node: str) -> list[str]:
        return list(self.edges.get(str(node), []))

    def indegree(self) -> dict[str, int]:
        counts = {node: 0 for node in self.edges}
        for node in self.edges:
            for target in self.edges[node]:
                counts[target] = counts.get(target, 0) + 1
        return counts

    def topological_order(self) -> list[str] | None:
        counts = self.indegree()
        ready = sorted(node for node, count in counts.items() if count == 0)
        out: list[str] = []
        while ready:
            node = ready.pop(0)
            out.append(node)
            for target in self.edges.get(node, []):
                counts[target] -= 1
                if counts[target] == 0:
                    ready.append(target)
            ready.sort()
        if len(out) != len(self.edges):
            return None
        return out

    def has_cycle(self) -> bool:
        return self.topological_order() is None

    def reachable(self, start: str) -> set[str]:
        seen: set[str] = set()
        stack = [str(start)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.edges.get(node, []))
        return seen

    def shortest_path(self, start: str, goal: str) -> list[str] | None:
        origin, destination = str(start), str(goal)
        if origin not in self.edges or destination not in self.edges:
            return None
        frontier = AuxHeap([(Decimal(0), (origin, [origin]))])
        best: dict[str, Decimal] = {origin: Decimal(0)}
        while len(frontier):
            row = frontier.pop()
            if row is None:
                break
            cost, (node, path) = row[0], row[1]
            if node == destination:
                return path
            for target in self.edges.get(node, []):
                step = cost + self.weights.get((node, target), Decimal(0))
                if target in best and best[target] <= step:
                    continue
                best[target] = step
                frontier.push(step, (target, path + [target]))
        return None

    def degree_table(self) -> dict[str, tuple[int, int]]:
        incoming = self.indegree()
        return {
            node: (incoming.get(node, 0), len(targets))
            for node, targets in self.edges.items()
        }


class AuxStateMachine:

    def __init__(self, start: str, transitions: dict[str, dict[str, str]] | None = None):
        self.start = str(start)
        self.state = self.start
        self.transitions = {
            str(key): dict(value) for key, value in (transitions or {}).items()
        }
        self.trail: list[tuple[str, str, str]] = []
        self.rejected = 0

    def allow(self, source: str, event: str, target: str) -> None:
        self.transitions.setdefault(str(source), {})[str(event)] = str(target)

    def can(self, event: str) -> bool:
        return str(event) in self.transitions.get(self.state, {})

    def fire(self, event: str) -> bool:
        table = self.transitions.get(self.state, {})
        key = str(event)
        if key not in table:
            self.rejected += 1
            return False
        previous = self.state
        self.state = table[key]
        self.trail.append((previous, key, self.state))
        return True

    def reset(self) -> None:
        self.state = self.start
        self.trail = []

    def visited(self) -> list[str]:
        return aux_unique([self.start] + [row[2] for row in self.trail])

    def terminal(self) -> bool:
        return not self.transitions.get(self.state)


class AuxDictDelta:

    @staticmethod
    def _walk(node: Any, stem: str, out: dict[str, Any]) -> None:
        if isinstance(node, dict):
            for key in sorted(str(item) for item in node):
                AuxDictDelta._walk(node[key], stem + "." + key if stem else key, out)
            return
        if isinstance(node, list):
            for index, item in enumerate(node):
                AuxDictDelta._walk(item, stem + "[" + str(index) + "]", out)
            return
        out[stem] = node

    @staticmethod
    def flatten(node: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        AuxDictDelta._walk(node, "", out)
        return out

    @staticmethod
    def compare(left: Any, right: Any) -> dict[str, list[str]]:
        first = AuxDictDelta.flatten(left)
        second = AuxDictDelta.flatten(right)
        added = sorted(key for key in second if key not in first)
        removed = sorted(key for key in first if key not in second)
        changed = sorted(
            key for key in first if key in second and first[key] != second[key]
        )
        return {"added": added, "removed": removed, "changed": changed}

    @staticmethod
    def unchanged(left: Any, right: Any) -> bool:
        delta = AuxDictDelta.compare(left, right)
        return not any(delta.values())

    @staticmethod
    def summary(left: Any, right: Any) -> str:
        delta = AuxDictDelta.compare(left, right)
        return ", ".join(
            name + "=" + str(len(rows)) for name, rows in sorted(delta.items())
        )


class AuxShapeIndex:

    def __init__(self) -> None:
        self.shapes: dict[str, int] = {}

    @staticmethod
    def describe(node: Any, depth: int = AUX_DEFAULT_DEPTH) -> str:
        if depth <= 0:
            return "..."
        if isinstance(node, dict):
            inner = ",".join(
                str(key) + ":" + AuxShapeIndex.describe(node[key], depth - 1)
                for key in sorted(str(item) for item in node)
            )
            return "{" + inner + "}"
        if isinstance(node, list):
            if not node:
                return "[]"
            return "[" + AuxShapeIndex.describe(node[0], depth - 1) + "]"
        if isinstance(node, bool):
            return "bool"
        if isinstance(node, (int, float, Decimal)):
            return "number"
        if node is None:
            return "null"
        return "text"

    def observe(self, node: Any) -> str:
        shape = self.describe(node)
        self.shapes[shape] = self.shapes.get(shape, 0) + 1
        return shape

    def dominant(self) -> str:
        if not self.shapes:
            return ""
        return max(sorted(self.shapes), key=lambda key: self.shapes[key])

    def variants(self) -> int:
        return len(self.shapes)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.shapes.items()))


class AuxRetryLedger:

    def __init__(self, ceiling: int = AUX_DEFAULT_DEPTH):
        self.ceiling = max(1, int(ceiling))
        self.counts: dict[str, int] = {}
        self.outcomes: dict[str, list[bool]] = {}

    def note(self, key: str, ok: bool) -> int:
        marker = str(key)
        self.counts[marker] = self.counts.get(marker, 0) + 1
        self.outcomes.setdefault(marker, []).append(bool(ok))
        return self.counts[marker]

    def exhausted(self, key: str) -> bool:
        return self.counts.get(str(key), 0) >= self.ceiling

    def succeeded(self, key: str) -> bool:
        return any(self.outcomes.get(str(key), []))

    def failures(self, key: str) -> int:
        return sum(1 for ok in self.outcomes.get(str(key), []) if not ok)

    def worst(self) -> str:
        if not self.counts:
            return ""
        return max(sorted(self.counts), key=lambda key: self.failures(key))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tracked": len(self.counts),
            "attempts": sum(self.counts.values()),
            "exhausted": sorted(key for key in self.counts if self.exhausted(key)),
        }


class AuxScoreBoard:

    def __init__(self, weights: dict[str, Any] | None = None):
        self.weights: dict[str, Decimal] = {}
        for name, value in (weights or {}).items():
            number = to_decimal(value)
            if number is not None:
                self.weights[str(name)] = number
        self.rows: dict[str, dict[str, Decimal]] = {}

    def record(self, subject: str, criterion: str, value: Any) -> bool:
        number = to_decimal(value)
        if number is None:
            return False
        self.rows.setdefault(str(subject), {})[str(criterion)] = number
        return True

    def criteria(self) -> list[str]:
        names: set[str] = set(self.weights)
        for row in self.rows.values():
            names.update(row)
        return sorted(names)

    def _weight(self, criterion: str) -> Decimal:
        return self.weights.get(criterion, Decimal(1))

    def total(self, subject: str) -> Decimal:
        row = self.rows.get(str(subject), {})
        return sum(
            (self._weight(name) * value for name, value in row.items()), Decimal(0)
        )

    def normalised(self) -> dict[str, Decimal]:
        totals = {subject: self.total(subject) for subject in self.rows}
        if not totals:
            return {}
        values = aux_normalise(list(totals.values()))
        return dict(zip(totals, values))

    def ordering(self) -> list[str]:
        return sorted(self.rows, key=lambda subject: (-self.total(subject), subject))

    def leader(self) -> str:
        order = self.ordering()
        return order[0] if order else ""

    def gap(self) -> Decimal:
        order = self.ordering()
        if len(order) < 2:
            return Decimal(0)
        return self.total(order[0]) - self.total(order[1])

    def coverage(self) -> dict[str, float]:
        names = self.criteria()
        if not names or not self.rows:
            return {}
        return {
            subject: len(row) / len(names) for subject, row in sorted(self.rows.items())
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "subjects": len(self.rows),
            "criteria": self.criteria(),
            "leader": self.leader(),
            "gap": str(self.gap()),
        }


class AuxPlanLattice:

    def __init__(self) -> None:
        self.graph = AuxGraph()
        self.state = AuxStateMachine(AUX_MODES[0])
        self.board = AuxScoreBoard()
        self.ledger = AuxRetryLedger()
        self.cache = AuxLruCache()

    def stage(self, name: str, depends_on: Sequence[str] = ()) -> None:
        self.graph.add_node(str(name))
        for parent in depends_on:
            self.graph.add_edge(str(parent), str(name))

    def order(self) -> list[str]:
        return self.graph.topological_order() or sorted(self.graph.edges)

    def blocked(self) -> list[str]:
        counts = self.graph.indegree()
        return sorted(node for node, count in counts.items() if count > 0)

    def score(self, stage: str, criterion: str, value: Any) -> bool:
        return self.board.record(stage, criterion, value)

    def attempt(self, stage: str, ok: bool) -> bool:
        self.ledger.note(stage, ok)
        return not self.ledger.exhausted(stage)

    def advance(self) -> str:
        for index, mode in enumerate(AUX_MODES[:-1]):
            self.state.allow(mode, "next", AUX_MODES[index + 1])
        self.state.fire("next")
        return self.state.state

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order(),
            "blocked": self.blocked(),
            "mode": self.state.state,
            "board": self.board.as_dict(),
            "ledger": self.ledger.as_dict(),
        }


def aux_build_lattice(stages: Sequence[tuple[str, Sequence[str]]]) -> AuxPlanLattice:
    lattice = AuxPlanLattice()
    for name, parents in stages:
        lattice.stage(name, parents)
    return lattice


def aux_summarise_series(values: Sequence[Any]) -> dict[str, Any]:
    stat = AuxRollingStat(max(1, len(values) or 1))
    for value in values:
        stat.observe(value)
    histogram = AuxHistogram()
    for value in values:
        histogram.observe(value)
    return {
        "stat": stat.as_dict(),
        "histogram": histogram.as_dict(),
        "p50": str(aux_percentile(values, 0.5) or ""),
        "p90": str(aux_percentile(values, 0.9) or ""),
    }


def aux_cluster_by_similarity(
    items: Sequence[str], threshold: float = 0.6
) -> list[list[str]]:
    union = AuxDisjointSet()
    for item in items:
        union.add(str(item))
    for left, right in aux_pairs(items):
        score = aux_jaccard(tokens(left), tokens(right))
        if score >= threshold:
            union.union(str(left), str(right))
    return [sorted(group) for group in sorted(union.groups().values())]


def aux_shape_report(samples: Sequence[Any]) -> dict[str, Any]:
    index = AuxShapeIndex()
    for sample in samples:
        index.observe(sample)
    return {
        "dominant": index.dominant(),
        "variants": index.variants(),
        "counts": index.as_dict(),
    }


def build_controller(
    problem_data: dict[str, Any],
    proxy: Any = None,
    settings: Settings | None = None,
) -> EpisodeController:
    return EpisodeController(problem_data, proxy or _proxy_client(), settings)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    controller = EpisodeController(problem_data, _proxy_client())
    try:
        return controller.run()
    except Exception as exc:
        controller.note("episode aborted: " + type(exc).__name__ + ": " + str(exc)[:300])
        controller.history.note(
            "assistant", "episode aborted", diagnostics=controller.diagnostics
        )
        return controller.history.render()
