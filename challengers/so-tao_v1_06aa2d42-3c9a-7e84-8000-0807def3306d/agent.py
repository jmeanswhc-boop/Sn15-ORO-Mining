from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from itertools import product
from os import getenv
from types import NoneType
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from src.agent.agent_interface import ToolCallResult, execute_tool_call
from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_FALLBACK_MODELS = {
    "chutes": [
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "zai-org/GLM-5.1-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "Qwen/Qwen3-32B-TEE",
    ],
    "openrouter": [
        "deepseek/deepseek-v3.2",
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.1",
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3-32b",
    ],
}

_proxy = ProxyClient(timeout=120, max_retries=3, retry_delay=1.0)


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _model_list() -> list[str]:
    preferred = getenv("SANDBOX_MODEL")
    provider = _provider()
    models = list(_FALLBACK_MODELS.get(provider, _FALLBACK_MODELS["chutes"]))
    if preferred and preferred not in models:
        models.insert(0, preferred)
    elif preferred:
        models.remove(preferred)
        models.insert(0, preferred)
    default = _DEFAULT_MODELS.get(provider)
    if default and default not in models:
        models.append(default)
    return models


def _parse_json_content(content: str) -> Any:
    text = (content or "").strip()
    if not text:
        raise ValueError("empty model content")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def llm_json(
    prompt: str,
    *,
    temperature: float = 0.0,
    expect: str = "dict",
) -> Any:
    last_err: Optional[Exception] = None
    temp = temperature
    for model in _model_list():
        for _ in range(2):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "stream": False,
                "max_tokens": 4096,
                "max_completion_tokens": 4096,
            }
            try:
                resp = _proxy.post("/inference/chat/completions", json_data=payload)
                if resp is None:
                    raise RuntimeError("inference returned None")
                content = resp["choices"][0]["message"].get("content") or ""
                parsed = _parse_json_content(content)
                if expect == "dict" and not isinstance(parsed, dict):
                    raise ValueError(f"expected dict got {type(parsed)}")
                if expect == "list" and not isinstance(parsed, list):
                    raise ValueError(f"expected list got {type(parsed)}")
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                temp = min(temp + 0.2, 0.8)
                continue
    raise RuntimeError(f"llm_json failed: {last_err}")


# ---------------------------------------------------------------------------
# Vocab — Newstart retail synonym / unit / stopword tables
# ---------------------------------------------------------------------------


class Vocab:
    """General retail vocabulary (not suite SKUs). Kept from Newstart_v11 spirit."""

    ID_KEYS = ("brand", "color", "model", "id", "compatibility", "color_family")
    SCHEMA_KEYS: frozenset = frozenset({
        "product_type", "color", "color_family", "material", "brand", "style",
        "size", "size_system", "gender", "user", "age", "pack", "set", "model",
        "compatibility", "pattern", "design", "shape", "origin", "usage",
        "fit", "feature", "category", "capacity", "volume", "length", "weight",
        "pre_config", "suitable", "condition", "warranty_days", "version",
        "option", "variation", "solution", "quality", "unit", "frame", "theme",
        "promotion", "inspiration", "interest", "vibe", "id",
    })
    NON_SEARCH_KEYS: frozenset = frozenset({"delivery", "service"})
    KW_STOPWORDS: tuple[str, ...] = (
        "color family", "suitable", "compatible",
        "about", "above", "after", "also", "and", "as", "at", "before", "below",
        "between", "by", "for", "from", "in",
        "less", "more", "of", "on", "or", "over", "per", "than", "to", "under",
        "up", "with", "within",
    )
    NO_WORDS: tuple[str, ...] = ("no", "without any", "without")
    YES_WORDS: tuple[str, ...] = ("yes", "enabled", "included", "exists", "true")
    COLOR_WORDS: tuple[str, ...] = (
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple",
        "violet", "orange", "brown", "gray", "grey", "beige", "navy", "gold",
        "silver", "bronze", "magenta", "cyan",
    )
    SEARCHABLE_UNITS: tuple[str, ...] = (
        "mm", "cm", "m", "km", "ft", "yd", "mg", "g", "kg", "oz", "lb",
        "ml", "l", "mah", "ah", "wh", "kwh", "v", "mv", "kv", "ma", "w", "kw",
        "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb",
        "bps", "kbps", "mbps", "gbps", "°c", "°f", "pcs", "pc",
        "pa", "kpa", "mpa", "mp", "dpi", "lux", "db", "rpm",
    )
    SYNONYMS: tuple[tuple[str, str], ...] = tuple(
        (min(a, b), max(a, b)) for a, b in [
            ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
            ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
            ("one", "single"), ("two", "double"), ("three", "triple"),
            ("1", "single"), ("2", "double"), ("3", "triple"),
            ("first", "1st"), ("second", "2nd"), ("third", "3rd"),
            ("hundred", "100"), ("thousand", "1000"), ("zero", "0"),
        ]
    )
    P_EXT_SEARCH_UNITS: tuple[tuple[str, str], ...] = (
        ("piece", "pcs"), ("pack", "pcs"),
    )

    @classmethod
    def synonym_partners(cls, token: str) -> list[str]:
        t = (token or "").lower().strip()
        out: list[str] = []
        for a, b in cls.SYNONYMS:
            if t == a and b not in out:
                out.append(b)
            elif t == b and a not in out:
                out.append(a)
        return out

    @classmethod
    def expand_query_tokens(cls, query: str) -> list[str]:
        """Build synonym + unit-extension variants of a free-text query."""
        base = " ".join((query or "").split()).strip()
        if not base:
            return []
        variants = [base]
        lower = base.lower()
        # Number-word swaps
        for a, b in cls.SYNONYMS:
            if re.search(rf"\b{re.escape(a)}\b", lower):
                variants.append(re.sub(rf"\b{re.escape(a)}\b", b, lower, flags=re.I))
            if re.search(rf"\b{re.escape(b)}\b", lower):
                variants.append(re.sub(rf"\b{re.escape(b)}\b", a, lower, flags=re.I))
        # Unit glued forms: "10 gb" <-> "10gb"
        for unit in cls.SEARCHABLE_UNITS:
            pat = rf"(\d+(?:\.\d+)?)\s+{re.escape(unit)}\b"
            if re.search(pat, lower, re.I):
                variants.append(re.sub(pat, rf"\1{unit}", lower, flags=re.I))
            pat2 = rf"(\d+(?:\.\d+)?){re.escape(unit)}\b"
            if re.search(pat2, lower, re.I):
                variants.append(re.sub(pat2, rf"\1 {unit}", lower, flags=re.I))
        # piece/pack → pcs
        for src, dst in cls.P_EXT_SEARCH_UNITS:
            if re.search(rf"\b1\s*{re.escape(src)}\b", lower):
                variants.append(re.sub(rf"\b1\s*{re.escape(src)}\b", f"1{dst}", lower))
        # de-dupe preserve order
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            key = " ".join(v.split()).lower()
            if key and key not in seen:
                seen.add(key)
                out.append(" ".join(v.split()))
        return out[:16]


# ---------------------------------------------------------------------------
# Weight — KEEP Newstart RANK_LOG_PROBS / SKU_LIN_PROBS / score_n
# ---------------------------------------------------------------------------


@dataclass
class Weight:
    """Newstart_v11 ranking confidence math (unchanged constants)."""

    SKU_LIN_PROBS = (0.1, 0.1, 0.08, 0.05, 0.02, 0.01)
    RANK_BONUS_C = 0.05
    RANK_LIMIT = 300
    RANK_LOG_PROBS = (
        0.6554774978466211, 0.09305071398002347, 0.04004714272557973,
        0.024146071349246602, 0.019434642793296042, 0.01943464179329533,
        0.018845714223802223, 0.005889285694938195, 0.005889284694937484,
        0.005005892840697467, 0.005005891840696755, 0.005005890840696044,
        0.005005889840695332, 0.002572687961473001, 0.00257268696147229,
        0.0025726859614715793, 0.002572684961470868, 0.0025726839614701575,
        0.0025726829614694463, 0.0025726819614687356, 0.0025726809614680245,
        0.0025726799614673138, 0.0025726789614666026, 0.002572677961465892,
        0.002572676961465181, 0.00257267596146447, 0.0025726749614637594,
        0.002572673961463048, 0.0025726729614623375, 0.0025726719614616264,
        0.0025726709614609156, 0.0025726699614602045, 0.002572668961459494,
        0.0025726679614587827, 0.002572666961458072, 0.0025726659614573612,
        0.00257266496145665, 0.0025726639614559394, 0.0025726629614552282,
        0.0025726619614545175, 0.0025726609614538064, 0.0025726599614530957,
        0.0025726589614523845, 0.002572657961451674, 0.0025726569614509627,
        0.002572655961450252, 0.0025726549614495412, 0.00257265396144883,
        0.0025726529614481194, 0.0025726519614474083, 0.0025726509614466975,
    )
    SCORE_N_THRES = 0.4

    detail: dict[str, float] = field(default_factory=dict)
    group: dict[str, int] = field(default_factory=dict)
    weighted_rank: int = -1
    rank: int = -1
    confidence: float = 0.0
    score: float = 0.0
    score_n: float = 0.0
    rank_bonus: float = 0.0
    sku_bonus: float = 0.0

    def _detail_n(self) -> int:
        return len(self.detail) or 1

    def normalize(self) -> None:
        idx = self.group.get("sku") if self.group.get("sku") is not None else -1
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            idx_i = -1
        self.sku_bonus = (
            Weight.SKU_LIN_PROBS[idx_i + 1]
            if idx_i < (len(Weight.SKU_LIN_PROBS) - 1)
            else 0.0
        )
        self.rank_bonus = max(1.0 - self.rank / Weight.RANK_LIMIT, 0.0) * Weight.RANK_BONUS_C
        self.score_n = self.score / self._detail_n() + self.rank_bonus + self.sku_bonus

    def update_confidence(self, weighted_rank: int) -> None:
        if self.weighted_rank >= 0:
            return
        self.weighted_rank = weighted_rank
        if weighted_rank >= len(Weight.RANK_LOG_PROBS):
            self.confidence = Weight.RANK_LOG_PROBS[-1] * (
                0.9 ** (weighted_rank - len(Weight.RANK_LOG_PROBS) + 1)
            )
        else:
            self.confidence = Weight.RANK_LOG_PROBS[weighted_rank]

    def update_with_sibling(self, sibling: "Weight") -> None:
        if self.score < sibling.score:
            self.detail = dict(sibling.detail)
            self.group = dict(sibling.group)
            self.score = sibling.score
        self.sku_bonus = max(self.sku_bonus, sibling.sku_bonus)
        self.score_n = self.score / self._detail_n() + self.rank_bonus + self.sku_bonus

    def absorb_raw(self, raw: float, *, rank: int = 0, sku_idx: int = -1) -> "Weight":
        """Fold an EnvPack heuristic score into Newstart Weight math."""
        self.score = float(raw)
        self.rank = int(rank)
        self.detail = {"envpack": float(raw)}
        self.group = {"sku": int(sku_idx)}
        self.normalize()
        self.update_confidence(max(0, rank))
        return self


# ---------------------------------------------------------------------------
# SchemaRouter — capability → policy_view tool name
# ---------------------------------------------------------------------------


class SchemaRouter:
    """Resolve logical shopping capabilities against this task's tool schema.

    Soft-skip missing tools; alias search↔filter, compare→view, skip absent
    stock/message/remove. Never invent names absent from policy_view.
    """

    _ALIASES: dict[str, tuple[str, ...]] = {
        "search": ("search", "find", "find_product", "catalog_search", "query_catalog"),
        "filter": ("filter", "filter_catalog", "browse", "list_products"),
        "view": ("view", "get_product", "product_detail", "inspect_product"),
        "compare": ("compare", "compare_products"),
        "inspect_stock": ("inspect_stock", "check_stock", "get_stock", "stock_status"),
        "inspect_cart": ("inspect_cart", "get_cart", "view_cart"),
        "message": ("message", "ask_user", "send_message", "reply_user"),
        "add_to_cart": ("add_to_cart", "cart_add", "add_item"),
        "remove_from_cart": ("remove_from_cart", "cart_remove", "remove_item"),
        "place_order": (
            "place_test_order",
            "place_order",
            "checkout",
            "buy",
            "submit_order",
        ),
    }

    def __init__(self, tools_schema: Optional[list[Any]] = None):
        self.names: set[str] = set()
        self.param_keys: dict[str, set[str]] = {}
        self._resolved: dict[str, Optional[str]] = {}
        for tool in tools_schema or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not name:
                continue
            name_s = str(name)
            self.names.add(name_s)
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
            self.param_keys[name_s] = {str(k) for k in props.keys()}
        for cap in self._ALIASES:
            self._resolved[cap] = self._match(cap)

    def _match(self, capability: str) -> Optional[str]:
        for candidate in self._ALIASES.get(capability, (capability,)):
            if candidate in self.names:
                return candidate
        if capability in self.names:
            return capability
        return None

    def has(self, capability: str) -> bool:
        return self.resolve(capability) is not None

    def resolve(self, capability_or_name: str) -> Optional[str]:
        if capability_or_name in self._resolved:
            return self._resolved[capability_or_name]
        if capability_or_name in self.names:
            return capability_or_name
        matched = self._match(capability_or_name)
        self._resolved[capability_or_name] = matched
        return matched

    def filter_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        allowed = self.param_keys.get(tool_name)
        if not allowed:
            return dict(args)
        return {k: v for k, v in args.items() if k in allowed}


# ---------------------------------------------------------------------------
# Desk — EnvPack session (Newstart SearchSession dialogue spirit)
# ---------------------------------------------------------------------------


@dataclass
class Desk:
    """Environment call desk: binding copy, turn counter, dialogue, shopper notes."""

    problem_id: str
    binding: dict[str, Any]
    policy: dict[str, Any]
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    finished: bool = False
    last_shopper_note: Optional[str] = None
    shopper_notes: list[str] = field(default_factory=list)
    router: SchemaRouter = field(default_factory=SchemaRouter)

    @classmethod
    def from_problem(cls, problem_data: dict[str, Any]) -> "Desk":
        environment = problem_data["environment"]
        policy = dict(environment["policy_view"])
        return cls(
            problem_id=str(
                problem_data.get("problem_id", problem_data.get("id", "problem"))
            ),
            binding=dict(environment["binding"]),
            policy=policy,
            router=SchemaRouter(policy.get("tools") or []),
        )

    @property
    def query(self) -> str:
        return str(self.policy.get("query") or "")

    @property
    def max_steps(self) -> int:
        return int(self.policy.get("max_steps") or 20)

    @property
    def max_calls_per_turn(self) -> int:
        return int(self.policy.get("max_calls_per_turn") or 1)

    @property
    def tool_names(self) -> set[str]:
        if self.router.names:
            return set(self.router.names)
        names: set[str] = set()
        for tool in self.policy.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.add(str(name))
        return names

    def remaining(self) -> int:
        return max(0, self.max_steps - self.turn)

    def can(self, capability: str) -> bool:
        return self.router.has(capability)

    def think(self, content: str, *, think: str = "") -> None:
        """Newstart-style dialogue entry: short assistant content + optional think."""
        entry: dict[str, Any] = {"role": "assistant", "content": content}
        if think:
            entry["think"] = think
        self.dialogue.append(entry)

    @staticmethod
    def _unwrap_observation(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if any(
            key in payload
            for key in (
                "results", "variants", "compared", "added", "cart",
                "product_id", "sent", "ordered", "error",
            )
        ) and not (
            "done" in payload
            and "observation" in payload
            and isinstance(payload.get("observation"), dict)
        ):
            return payload
        inner = payload.get("observation")
        if isinstance(inner, dict):
            return inner
        return payload

    @staticmethod
    def _call_done(call: dict[str, Any]) -> bool:
        if call.get("done") is True:
            return True
        obs = call.get("observation")
        if isinstance(obs, dict) and obs.get("done") is True:
            return True
        return False

    def call(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.finished:
            return {"calls": [], "user_message": None}
        if self.remaining() <= 0:
            raise RuntimeError("max_steps exhausted")
        allowed = self.tool_names
        cleaned: list[dict[str, Any]] = []
        for action in actions[: self.max_calls_per_turn]:
            raw_name = action.get("name")
            if not raw_name:
                continue
            name = self.router.resolve(str(raw_name)) or (
                str(raw_name) if str(raw_name) in allowed else None
            )
            if not name:
                continue
            if allowed and name not in allowed:
                continue
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            cleaned.append(
                {"name": name, "args": self.router.filter_args(name, args)}
            )
        if not cleaned:
            return {
                "calls": [],
                "user_message": None,
                "error": "tool_unavailable",
                "requested": [a.get("name") for a in actions],
                "tools": sorted(allowed),
            }

        self.turn += 1
        group_id = f"{self.problem_id}-turn-{self.turn}"
        envelope = {
            **self.binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": self.turn,
            "calls": [
                {"call_id": f"{group_id}-{index}", "action": action}
                for index, action in enumerate(cleaned, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        self.dialogue.append(
            {
                "role": "assistant",
                "content": f"step {self.turn}",
                "tool_calls": cleaned,
                "environment_result": result,
            }
        )
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            note = str(user_message["content"])
            self.last_shopper_note = note
            self.shopper_notes.append(note)
        for call in result.get("calls") or []:
            if self._call_done(call):
                self.finished = True
        return result

    def act(self, name: str, **args: Any) -> dict[str, Any]:
        if not self.router.resolve(name) and name not in self.router.names:
            return {"error": "tool_unavailable", "requested": name}
        result = self.call([{"name": name, "args": args}])
        if result.get("error") == "tool_unavailable":
            return {"error": "tool_unavailable", "requested": name}
        calls = result.get("calls") or []
        if not calls:
            return {}
        return self._unwrap_observation(calls[0].get("observation") or {})



# ---------------------------------------------------------------------------
# SkuCard + text helpers
# ---------------------------------------------------------------------------


@dataclass
class SkuCard:
    """Catalog SKU card with attached Newstart Weight."""

    product_id: str
    sku: str
    title: str = ""
    brand: str = ""
    price: Optional[float] = None
    currency: str = "PHP"
    in_stock: bool = True
    options: str = ""
    category_path: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    listing_id: Optional[str] = None
    viewed: bool = False
    source_brand: str = ""
    weight: Weight = field(default_factory=Weight)

    @property
    def key(self) -> str:
        return f"{self.product_id}::{self.sku}"

    def text_blob(self) -> str:
        parts = [
            self.title,
            self.brand,
            self.options,
            " ".join(self.category_path),
            json.dumps(self.facts, ensure_ascii=False),
        ]
        return " ".join(str(p) for p in parts if p).lower()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value).replace(",", ""))
    return float(m.group(1)) if m else None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t]


def _goal_only(query: str) -> str:
    split = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    return split[0].strip()


# ---------------------------------------------------------------------------
# BriefSheet — hard/soft constraints from public query
# ---------------------------------------------------------------------------


@dataclass
class BriefSheet:
    family: str = "generic"
    goal: str = ""
    category: Optional[str] = None
    brand: Optional[str] = None
    preferred_brand: Optional[str] = None
    budget: Optional[float] = None
    currency: str = "PHP"
    require_in_stock: bool = True
    attributes: dict[str, str] = field(default_factory=dict)
    soft_prefs: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    candidate_keys: list[str] = field(default_factory=list)
    preferred_model: Optional[str] = None
    preferred_color: Optional[str] = None
    needs_justification: bool = False
    needs_clarification: bool = False
    title_phrase: Optional[str] = None  # retrieval_recall quoted match


_FAMILY_HINTS = (
    (
        "ranking",
        (
            "largest screen",
            "best-ranked",
            "overclock",
            "candidate variants",
            "compare only these candidate",
            "highest refresh",
            "advertised maximum refresh",
            "tradeoff priority",
        ),
    ),
    (
        "preference_reasoning",
        ("listing_id", "soft preferences", "intended use", "sweat", "water resist"),
    ),
    (
        "recovery",
        ("relaxable", "another color", "least-relaxed", "preferred model"),
    ),
    (
        "retrieval_recall",
        ("title token", "distinct matching product", "complete title token"),
    ),
    (
        "constraint_satisfaction",
        (
            "exact variant",
            "pixel classes",
            "every requirement must hold",
            "check the specific sku",
            "conflicting resolution classes",
            "screen-only or replacement-screen",
        ),
    ),
    (
        "intent_decomposition",
        (
            "soft preference",
            "preferred brand",
            "nice-to-haves",
            "ideally, i prefer",
            "firm requirements matter more",
            "initial cart choice",
            "named preferred brand",
            "honor that named brand",
        ),
    ),
    (
        "justification",
        (
            "reason i can trust",
            "give me a reason",
            "trustworthy reason",
            "walk me through why",
            "tell me why",
            "why the one you pick",
            "why it's a good pick",
            "structured justification with claims",
        ),
    ),
)

EVENT_FAMILIES = frozenset(
    {"intent_decomposition", "constraint_satisfaction", "justification"}
)

_BRIEF_PROMPT = """\
You are a shopping planner for an interactive catalog environment.
Extract a structured purchase brief from the shopper text below.

Rules:
- Separate HARD requirements from SOFT preferences.
- Hard usually includes: category, budget, in-stock, and named hardware specs
  (screen_size, resolution, refresh_rate, ram, etc.).
- An explicitly preferred brand is SOFT unless the brief says the brand is mandatory.
- If a firm hardware spec will be told only when asked
  ("tell you if you ask" / "one firm requirement"), set needs_clarification=true
  and leave attributes EMPTY — do not invent specs.
- Keywords must be short BM25-friendly tokens (values, not schema keys).
- Provide 3-6 alternate search_queries that reformulate the request.
- If the brief lists candidate variants as product_id::sku, copy them into candidate_keys.
- needs_justification=true only if the shopper asks for a reason in their own request
  (ignore shared tool-footer text about optional claims).

Return JSON only:
{{
  "category": <str|null>,
  "brand": <str|null>,
  "preferred_brand": <str|null>,
  "budget": <number|null>,
  "currency": <str>,
  "require_in_stock": <bool>,
  "attributes": {{<attr>: <value>, ...}},
  "soft_prefs": [<str>, ...],
  "keywords": [<str>, ...],
  "search_queries": [<str>, ...],
  "candidate_keys": [<"product_id::sku">, ...],
  "preferred_model": <str|null>,
  "preferred_color": <str|null>,
  "needs_justification": <bool>,
  "needs_clarification": <bool>
}}

# Shopper brief
{query}
"""


def infer_family(query: str) -> str:
    """Detect EnvPack family from PUBLIC query wording only."""
    goal = _goal_only(query).lower()
    full = query.lower()
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    rules = parts[1].lower() if len(parts) > 1 else full
    if re.search(r"\b(ideally|nice[- ]to[- ]have)\b", goal) and re.search(
        r"\bprefer(?:\s+the)?\s+[a-z]", goal
    ):
        return "intent_decomposition"
    if "prefer" in goal and (
        "firm requirements matter" in goal
        or "nice-to-have" in full
        or "soft preference" in full
    ):
        return "intent_decomposition"
    for hint in (
        "reason i can trust",
        "give me a reason",
        "trustworthy reason",
        "walk me through why",
        "tell me why",
        "why the one you pick",
        "why it's a good pick",
        "structured justification with claims",
    ):
        if hint in goal:
            return "justification"
    for family, hints in _FAMILY_HINTS:
        if family == "justification":
            continue
        if family == "constraint_satisfaction":
            haystack = rules
        elif family == "intent_decomposition":
            haystack = full
        else:
            haystack = full
        if any(h in haystack for h in hints):
            return family
    if "prefer" in goal and ("firm" in full or "nice-to-have" in full):
        return "intent_decomposition"
    if re.search(r"\d{6,}::", query):
        return "ranking"
    return "generic"


def _extract_budget(text: str) -> Optional[float]:
    patterns = [
        r"(?:up to|under|below|at or under|budget(?:\s+is)?|spend(?:\s+up to)?)\s*"
        r"`?([0-9]+(?:\.[0-9]+)?)`?\s*(?:PHP|php|USD|usd)?",
        r"`?([0-9]+(?:\.[0-9]+)?)`?\s*(?:PHP|php)\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:PHP|php)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extract_candidate_keys(text: str) -> list[str]:
    keys = re.findall(r"\b(\d{6,}::[A-Za-z0-9_.\-]+)", text)
    cleaned = [k.rstrip(".,;:)") for k in keys]
    return list(dict.fromkeys(cleaned))


def shopper_needs_reply(note: Optional[str]) -> bool:
    if not note:
        return False
    text = note.strip()
    if not text:
        return False
    if "?" in text or "？" in text:
        return True
    if re.search(r"\b(tell me|which one|confirm|before you buy|can you)\b", text, re.I):
        return True
    return False


def is_market_event_note(note: Optional[str]) -> bool:
    if not note:
        return False
    return bool(
        re.search(
            r"\b(not available|out of stock|sold out|stockout|unavailable|"
            r"price (?:changed|increased|jumped)|now costs)\b",
            note,
            re.I,
        )
    )


# ---------------------------------------------------------------------------
# SpecLexicon — attribute matching
# ---------------------------------------------------------------------------


class SpecLexicon:
    """Match hardware attributes from facts / options / title."""

    _RES_ALIASES = {
        "720p": ("720p", "hd", "1280x720", "1280 x 720", "1280×720"),
        "1080p": ("1080p", "full hd", "fhd", "1920x1080", "1920 x 1080", "1920×1080"),
        "1440p": ("1440p", "qhd", "2560x1440", "2560 x 1440"),
        "4k": ("4k", "uhd", "3840x2160", "3840 x 2160"),
        "hd": ("hd", "720p", "1280x720", "1280 x 720"),
    }

    @staticmethod
    def parse_inches(text: Any) -> Optional[float]:
        if text is None:
            return None
        raw = str(text).strip().lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", raw)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\"", raw)
        if m:
            return float(m.group(1))
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:in)?", raw.replace(" ", ""))
        if m:
            return float(m.group(1))
        return None

    @classmethod
    def screen_inches(cls, card: SkuCard) -> Optional[float]:
        opt = (card.options or "").strip()
        if opt:
            m = re.search(r"\((\d+(?:\.\d+)?)\s*inch", opt, re.I) or re.search(
                r"(\d+(?:\.\d+)?)\s*inch", opt, re.I
            )
            if m:
                return float(m.group(1))
            num = cls.parse_inches(opt)
            if num is not None and not re.search(r"\d+\s*[-–/]\s*\d+", opt):
                return num
        for fk, fv in (card.facts or {}).items():
            if any(n in str(fk).lower() for n in ("size", "screen", "inch", "display")):
                raw = str(fv)
                if re.search(r"\d+(?:\.\d+)?\s*[-–/,]\s*\d+(?:\.\d+)?", raw):
                    continue
                num = cls.parse_inches(raw) or _as_float(raw)
                if num is not None and 3.0 <= num <= 100.0:
                    return num
        title = card.title or ""
        if re.search(r"\d+(?:\.\d+)?\s*[-–/]\s*\d+(?:\.\d+)?", title):
            return None
        return cls.parse_inches(title)

    @classmethod
    def resolution_token(cls, card: SkuCard) -> Optional[str]:
        blob = " ".join(
            [
                card.title or "",
                card.options or "",
                " ".join(f"{k} {v}" for k, v in (card.facts or {}).items()),
            ]
        ).lower()
        if re.search(r"\b4k\b|\buhd\b|3840\s*[x×]\s*2160", blob):
            return "4k"
        if re.search(r"\b1440p\b|\bqhd\b|2560\s*[x×]\s*1440", blob):
            return "1440p"
        if re.search(r"\b1080p\b|full\s*hd|\bfhd\b|1920\s*[x×]\s*1080", blob):
            return "1080p"
        if re.search(r"\b720p\b|1280\s*[x×]\s*720", blob):
            return "720p"
        if re.search(r"\bhd\b", blob) and not re.search(r"full\s*hd|fhd|1080", blob):
            return "hd"
        best: Optional[str] = None
        order = {"4k": 4, "1440p": 3, "1080p": 2, "720p": 1, "hd": 1}
        for m in re.finditer(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", blob):
            w, h = int(m.group(1)), int(m.group(2))
            a, b = max(w, h), min(w, h)
            token = None
            if a >= 3840 and b >= 2160:
                token = "4k"
            elif a >= 2560 and b >= 1440:
                token = "1440p"
            elif a >= 1920 and b >= 1080:
                token = "1080p"
            elif a >= 1280 and b >= 720:
                token = "720p"
            if token and (best is None or order[token] > order[best]):
                best = token
        return best

    @classmethod
    def refresh_hz(cls, card: SkuCard) -> Optional[float]:
        blob = " ".join(
            [
                card.title or "",
                card.options or "",
                " ".join(f"{k} {v}" for k, v in (card.facts or {}).items()),
            ]
        ).lower()
        vals = [float(m) for m in re.findall(r"(\d{2,3})\s*hz", blob)]
        return max(vals) if vals else None

    @classmethod
    def normalize_wanted(cls, key: str, value: str) -> tuple[str, Any]:
        key_l = key.lower().replace("-", "_")
        val = str(value).strip()
        val_l = val.lower().replace(" ", "")
        if val_l in cls._RES_ALIASES or re.fullmatch(
            r"\d{3,4}p|4k|uhd|qhd|hd|fullhd|fhd", val_l
        ):
            token = val_l
            if token in {"fullhd", "fhd"}:
                token = "1080p"
            return "resolution", token
        if "resolution" in key_l:
            token = val_l
            if token in {"fullhd", "fhd"}:
                token = "1080p"
            return "resolution", token
        if "refresh" in key_l or val_l.endswith("hz"):
            return "refresh_rate", _as_float(val)
        if "ram" in key_l:
            return "ram", _as_float(val)
        if "screen" in key_l or "size" in key_l or val_l.endswith("in"):
            inches = cls.parse_inches(val)
            if inches is None and re.fullmatch(r"\d+(?:\.\d+)?", val_l):
                inches = float(val_l)
            if inches is not None and not (3.0 <= float(inches) <= 100.0):
                inches = None
            return "screen_size", inches
        return key_l, val.lower()

    @classmethod
    def matches(cls, card: SkuCard, key: str, value: str) -> bool:
        kind, wanted = cls.normalize_wanted(key, value)
        if wanted is None or wanted == "":
            return False
        if kind == "screen_size":
            got = cls.screen_inches(card)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) < 0.005
        if kind == "resolution":
            got = cls.resolution_token(card)
            if not got:
                return False
            want = str(wanted).lower()
            if want in {"hd", "720p"} and got in {"hd", "720p"}:
                return True
            alts = cls._RES_ALIASES.get(want, (want,))
            return got in alts or got == want
        if kind == "refresh_rate":
            got = cls.refresh_hz(card)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) <= 1.0
        if kind == "ram":
            blob = card.text_blob()
            if not isinstance(wanted, (int, float)):
                return str(wanted).lower() in blob
            return bool(re.search(rf"\b{int(wanted)}\s*gb\b", blob))
        return _norm(str(wanted)) in _norm(card.text_blob())


# ---------------------------------------------------------------------------
# GuardRail — accessory rejection for device categories
# ---------------------------------------------------------------------------


class GuardRail:
    """Reject accessory / decoy listings when a real device category is required.

    Implementation wording intentionally differs from DeviceGuard / AccessoryFence.
    """

    _PARTS = (
        "case", "cover", "charger", "cable", "screen protector", "tempered glass",
        "replacement tip", "ear tip", "strap only", "mount only", "holder only",
        "stand only", "skin only", "bumper", "holster", "pouch", "digitizer",
        "digitiz", "panel assembly", "replacement part", "touch screen digitizer",
        "spare part", "repair part",
    )
    _PATH_BAD = (
        "accessor", "replacement", "spare part", "repair",
        "parts & tools", "parts and tools",
    )
    _BAD_OPT = re.compile(
        r"screen[- ]?only|replacement[- ]?screen|digitizer|lcd panel|touch panel only",
        re.I,
    )

    @classmethod
    def title_is_parts_kit(cls, title: str) -> bool:
        t = (title or "").lower()
        if not t:
            return False
        device_hit = bool(
            re.search(
                r"\b(smartphone|mobile phone|cellphone|tablet|ipad|monitor|"
                r"headphone|earbud|headset)\b",
                t,
            )
        )
        if any(b in t for b in cls._PARTS):
            if re.search(r"\b(with case|free case|include[sd]? case)\b", t) and device_hit:
                return False
            if re.search(r"\b(phone|tablet|monitor)\s+(case|cover|protector|bumper|pouch)\b", t):
                return True
            if re.search(r"\b(case|cover|protector|bumper)\b.*\b(phone|tablet)\b", t):
                return True
            return True
        if re.search(r"\b(lcd|digitizer)\b", t) and not device_hit:
            if re.search(r"\b(screen|display|panel|replacement|for)\b", t):
                return True
        if re.search(r"\bfor\b.+\b(lcd|digitizer|display|touch screen)\b", t) and not device_hit:
            return True
        if re.search(r"\b(lcd|digitizer)\b.+\bfor\b", t) and not device_hit:
            return True
        if re.search(r"\b(folding|foldable)\b", t) and re.search(
            r"\b(holder|stand|bracket|mount|support)\b", t
        ):
            return True
        return False

    @classmethod
    def path_is_parts(cls, category_path: list[str]) -> bool:
        path = _norm(" ".join(category_path or []))
        return any(b in path for b in cls._PATH_BAD)

    @classmethod
    def looks_phone(cls, card: SkuCard) -> bool:
        if cls.title_is_parts_kit(card.title or "") or cls.path_is_parts(card.category_path):
            return False
        title = card.title or ""
        path = _norm(" ".join(card.category_path))
        blob = card.text_blob()
        if re.search(
            r"\b(smartphone|cellphone|mobile phone|android phone|iphone|galaxy\s+[a-z]?\d)\b",
            title,
            re.I,
        ):
            return True
        if any(w in path for w in ("smartphone", "mobile phone", "cell phone")):
            return True
        for fk, fv in (card.facts or {}).items():
            fl = f"{fk} {fv}".lower()
            if re.search(r"\b(android|ios|iphone os|blackberry os|harmonyos)\b", fl):
                return True
        if re.search(r"\b(android|ios)\b", blob) and re.search(
            r"\b(gb ram|mp camera|dual sim)\b", blob
        ):
            return True
        return False

    @classmethod
    def looks_tablet(cls, card: SkuCard) -> bool:
        if cls.title_is_parts_kit(card.title or "") or cls.path_is_parts(card.category_path):
            return False
        title = card.title or ""
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            return False
        path = _norm(" ".join(card.category_path))
        if re.search(r"\b(tablet|ipad)\b", title, re.I):
            return True
        if "tablet" in path:
            return True
        for fk, fv in (card.facts or {}).items():
            if re.search(r"\b(tablet|ipados)\b", f"{fk} {fv}", re.I):
                return True
        return False

    @classmethod
    def looks_monitor(cls, card: SkuCard) -> bool:
        if cls.title_is_parts_kit(card.title or "") or cls.path_is_parts(card.category_path):
            return False
        title = card.title or ""
        path = _norm(" ".join(card.category_path))
        if re.search(r"\b(monitor|display)\b", title, re.I) and "lcd display" not in title.lower():
            if re.search(
                r"\b(digitizer|replacement|for vivo|for samsung|for xiaomi)\b", title, re.I
            ):
                return False
            return True
        return "monitor" in path or ("display" in path and "accessor" not in path)

    @classmethod
    def device_ok(cls, card: SkuCard, category: Optional[str]) -> bool:
        if not category:
            return True
        want = _norm(category).rstrip("s")
        title = card.title or ""
        if cls.title_is_parts_kit(title) or cls.path_is_parts(card.category_path):
            if any(
                w in want
                for w in ("phone", "smartphone", "tablet", "monitor", "headphone", "earbud")
            ):
                return False
        rich = bool(card.viewed or card.facts or card.category_path)
        if "smartphone" in want or want == "phone":
            return cls.looks_phone(card) if rich else True
        if "tablet" in want:
            return cls.looks_tablet(card) if rich else True
        if "monitor" in want:
            return cls.looks_monitor(card) if rich else True
        return True

    @classmethod
    def option_ok(cls, card: SkuCard, category: Optional[str]) -> bool:
        if not category:
            return True
        want = _norm(category)
        if not any(w in want for w in ("phone", "smartphone", "tablet", "monitor")):
            return True
        if cls._BAD_OPT.search(card.options or ""):
            return False
        return True



# ---------------------------------------------------------------------------
# Evidence / claims (visible facts only)
# ---------------------------------------------------------------------------


class EvidenceLedger:
    """Gate attribute must be visible on the selected SkuCard (TF7/TF3)."""

    @staticmethod
    def gate_from_goal(goal: str, attributes: dict[str, str]) -> Optional[str]:
        g = (goal or "").lower()
        if re.search(r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd)\b", g) or (
            re.search(r"\bhd\b", g) and "inch" not in g
        ):
            return "resolution"
        if re.search(r"\d+(?:\.\d+)?\s*-?\s*inch", g):
            return "screen_size"
        if re.search(r"\d{2,3}\s*hz", g):
            return "refresh_rate"
        for key in ("resolution", "screen_size", "refresh_rate", "ram"):
            if key in attributes:
                return key
        return None

    @classmethod
    def gate_key(cls, brief: BriefSheet) -> Optional[str]:
        return cls.gate_from_goal(brief.goal, brief.attributes) or (
            next(iter(brief.attributes), None) if brief.attributes else None
        )

    @classmethod
    def measured_value(cls, brief: BriefSheet, card: SkuCard) -> Optional[str]:
        gate = cls.gate_key(brief)
        if not gate:
            return None
        if gate == "resolution":
            return SpecLexicon.resolution_token(card)
        if gate == "screen_size":
            inches = SpecLexicon.screen_inches(card)
            return f"{inches:g}" if inches is not None else None
        if gate == "refresh_rate":
            hz = SpecLexicon.refresh_hz(card)
            return f"{int(hz)}Hz" if hz is not None else None
        if gate == "ram":
            blob = card.text_blob()
            m = re.search(r"\b(\d+)\s*gb\s*ram\b", blob, re.I) or re.search(
                r"\b(\d+)\s*gb\b", (card.options or "") + " " + blob, re.I
            )
            return f"{m.group(1)}GB" if m else None
        return None

    @classmethod
    def visible(cls, brief: BriefSheet, card: SkuCard) -> bool:
        gate = cls.gate_key(brief)
        if not gate:
            return True
        measured = cls.measured_value(brief, card)
        if measured is None:
            return False
        if gate in brief.attributes:
            return SpecLexicon.matches(card, gate, brief.attributes[gate])
        if gate == "resolution":
            goal = brief.goal.lower()
            got = measured.lower()
            if re.search(r"\b(720p|hd)\b", goal) and got in {"720p", "hd"}:
                return True
            if re.search(r"\b(1080p|full\s*hd|fhd)\b", goal) and got in {"1080p"}:
                return True
            if re.search(r"\b(1440p|qhd)\b", goal) and got in {"1440p"}:
                return True
            if re.search(r"\b(4k|uhd)\b", goal) and got in {"4k"}:
                return True
            return False
        return True

    @classmethod
    def claim_value(cls, brief: BriefSheet, card: SkuCard) -> Optional[str]:
        if not cls.visible(brief, card):
            return None
        gate = cls.gate_key(brief)
        measured = cls.measured_value(brief, card)
        if not gate or not measured:
            return None
        if gate == "resolution":
            fact_blob = " ".join(f"{k} {v}" for k, v in (card.facts or {}).items())
            surface = f"{card.title or ''} {card.options or ''} {fact_blob}".lower()
            goal = brief.goal.lower()
            lit_720 = "720p" in surface or "1280x720" in surface or "1280 x 720" in surface
            lit_hd = bool(re.search(r"\bhd\b", surface)) and "full hd" not in surface and "fhd" not in surface
            lit_1080 = (
                "1080p" in surface
                or "full hd" in surface
                or "fhd" in surface
                or "1920x1080" in surface
                or "1920 x 1080" in surface
            )
            if measured in {"hd", "720p"}:
                if "720p" in goal and lit_720:
                    return "720p"
                if re.search(r"\bhd\b", goal) and "1080" not in goal and "full" not in goal and (
                    lit_hd or lit_720
                ):
                    return "hd" if lit_hd and not lit_720 else ("720p" if lit_720 else None)
                if lit_720:
                    return "720p"
                if lit_hd:
                    return "hd"
                return None
            if measured == "1080p" and lit_1080:
                return "1080p"
            if measured.replace("p", "") and measured.lower() in surface:
                return measured
            return None
        return measured

    @classmethod
    def claimable(cls, brief: BriefSheet, card: SkuCard) -> bool:
        return cls.claim_value(brief, card) is not None


class ClaimSheet:
    """Build TF7 justification claims from viewed facts only."""

    @classmethod
    def build(cls, brief: BriefSheet, card: SkuCard) -> dict[str, Any]:
        claims: list[dict[str, str]] = []
        gate = EvidenceLedger.gate_from_goal(brief.goal, brief.attributes)
        claim_val = EvidenceLedger.claim_value(brief, card)
        if gate and claim_val:
            claims.append({"field": gate, "value": str(claim_val)})
        if card.price is not None:
            claims.append({"field": "price", "value": f"{float(card.price):g}"})
        claims.append({"field": "in_stock", "value": "true" if card.in_stock else "false"})
        soft_added = False
        want_brand = brief.brand or brief.preferred_brand
        if card.brand and (not want_brand or brand_hit(card, want_brand)):
            claims.append({"field": "brand", "value": str(card.brand)})
            soft_added = True
        elif want_brand and brand_hit(card, want_brand) and want_brand.lower() in _norm(
            card.title or ""
        ):
            claims.append({"field": "brand", "value": str(want_brand)})
            soft_added = True
        if not soft_added:
            blob = card.text_blob()
            for field_name, pat in (
                ("hdr", r"\bhdr\b"),
                ("microphone", r"\b(mic|microphone)\b"),
                ("streaming", r"\b(streaming|conference|zoom|teams)\b"),
                ("privacy_cover", r"\bprivacy (?:cover|shutter)\b"),
                ("autofocus", r"\bauto[- ]?focus\b"),
                ("gaming", r"\bgaming\b"),
            ):
                if re.search(pat, blob, re.I):
                    claims.append({"field": field_name, "value": "true"})
                    break
        uniq: dict[str, dict[str, str]] = {}
        for claim in claims:
            if re.search(r"\d+\s*[-–/]\s*\d+", claim["value"]):
                continue
            uniq.setdefault(claim["field"], claim)
        return {"claims": list(uniq.values())}


class OptionMention:
    """Prefer SKUs whose selected options explicitly state the firm attr."""

    @classmethod
    def hits(cls, brief: BriefSheet, card: SkuCard) -> bool:
        opt = (card.options or "").strip().lower()
        if not opt:
            return False
        for key, val in brief.attributes.items():
            kind, wanted = SpecLexicon.normalize_wanted(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                num = f"{float(wanted):g}"
                if re.search(
                    rf"\b{re.escape(num)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b", opt
                ) or re.search(rf"\({re.escape(num)}\s*inch", opt):
                    return True
            elif kind == "resolution" and wanted:
                want = str(wanted).lower()
                if want in opt or (
                    want in {"hd", "720p"} and ("720p" in opt or re.search(r"\bhd\b", opt))
                ):
                    return True
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                if re.search(rf"\b{int(wanted)}\s*hz\b", opt):
                    return True
            elif kind == "ram" and wanted:
                if _norm(str(wanted)) in _norm(opt):
                    return True
        return False


class CartPulse:
    """Apply cart-add observations and embedded market events to a SkuCard."""

    @staticmethod
    def apply_add(card: SkuCard, add_obs: dict[str, Any]) -> SkuCard:
        if not isinstance(add_obs, dict):
            return card
        added = add_obs.get("added") if isinstance(add_obs.get("added"), dict) else {}
        event = add_obs.get("market_event") if isinstance(add_obs.get("market_event"), dict) else {}
        cart = add_obs.get("cart") if isinstance(add_obs.get("cart"), list) else []
        if added:
            price = _as_float(added.get("price"))
            if price is not None:
                card.price = price
            if "in_stock" in added:
                card.in_stock = bool(added["in_stock"])
        for row in cart:
            if not isinstance(row, dict):
                continue
            if str(row.get("product_id")) != card.product_id:
                continue
            if str(row.get("sku") or "default") != (card.sku or "default"):
                continue
            if "in_stock" in row:
                card.in_stock = bool(row["in_stock"])
            cp = _as_float(row.get("current_price"))
            if cp is not None:
                card.price = cp
        if event.get("kind") == "stockout":
            card.in_stock = False
        if event.get("kind") == "price_change":
            new_p = _as_float(event.get("new_price"))
            if new_p is not None:
                card.price = new_p
            if "in_stock_after" in event:
                card.in_stock = bool(event["in_stock_after"])
        return card


# ---------------------------------------------------------------------------
# Predicate helpers + Weight scoring
# ---------------------------------------------------------------------------


def category_ok(card: SkuCard, category: Optional[str]) -> bool:
    if not category:
        return True
    if not GuardRail.device_ok(card, category):
        return False
    want = _norm(category).rstrip("s")
    path = _norm(" ".join(card.category_path))
    title = _norm(card.title)
    if want and (want in path or want in title):
        return True
    aliases = {
        "smartphone": ("phone", "mobile", "smartphone"),
        "phone": ("phone", "mobile", "smartphone"),
        "headphone": ("headphone", "earphone", "headset", "earbuds"),
        "monitor": ("monitor", "display"),
        "tablet": ("tablet", "ipad"),
        "grocery": ("grocery", "groceries", "food"),
        "beauty": ("beauty", "cosmetic", "skincare"),
    }
    for alias in aliases.get(want, ()):
        if alias in path or alias in title:
            return True
    return False


def brand_hit(card: SkuCard, brand: Optional[str]) -> bool:
    if not brand:
        return True
    want = _norm(brand)
    if not want:
        return True
    return want in _norm(card.brand or "") or want in _norm(card.source_brand or "") or want in _norm(
        card.title
    )


def brand_strength(card: SkuCard, brand: Optional[str]) -> int:
    if not brand:
        return 0
    want = _norm(brand)
    if not want:
        return 0
    if want in _norm(card.title or ""):
        return 3
    if want in _norm(card.brand or ""):
        return 2
    if want in _norm(card.source_brand or ""):
        return 1
    return 0


def budget_ok(card: SkuCard, budget: Optional[float]) -> bool:
    if budget is None or card.price is None:
        return True
    return float(card.price) <= float(budget) + 1e-6


def hard_ok(brief: BriefSheet, card: SkuCard) -> bool:
    if brief.require_in_stock and not card.in_stock:
        return False
    if not budget_ok(card, brief.budget):
        return False
    if not category_ok(card, brief.category):
        return False
    if not GuardRail.device_ok(card, brief.category):
        return False
    if not GuardRail.option_ok(card, brief.category):
        return False
    if brief.brand and not brand_hit(card, brief.brand):
        return False
    for key, value in brief.attributes.items():
        if not SpecLexicon.matches(card, key, value):
            return False
    if brief.family == "justification" and EvidenceLedger.gate_key(brief):
        if not EvidenceLedger.visible(brief, card):
            return False
    if brief.family == "retrieval_recall" and brief.title_phrase:
        if not title_phrase_match(card, brief.title_phrase):
            return False
    return True


def purchase_ok(brief: BriefSheet, card: SkuCard) -> bool:
    if brief.require_in_stock and not card.in_stock:
        return False
    if not budget_ok(card, brief.budget):
        return False
    if brief.brand and not brand_hit(card, brief.brand):
        return False
    if brief.category and not GuardRail.device_ok(card, brief.category):
        return False
    if brief.family == "retrieval_recall" and brief.title_phrase:
        if not title_phrase_match(card, brief.title_phrase):
            return False
    return True


def needs_event_recovery(brief: BriefSheet, card: SkuCard) -> bool:
    return not purchase_ok(brief, card)


def title_phrase_match(card: SkuCard, phrase: Optional[str]) -> bool:
    if not phrase:
        return True
    want = _norm(phrase)
    if not want:
        return True
    title = _norm(card.title or "")
    compact_want = want.replace(" ", "")
    compact_title = title.replace(" ", "")
    return want in title or compact_want in compact_title


def ranking_key(card: SkuCard) -> tuple:
    refresh = SpecLexicon.refresh_hz(card)
    size = SpecLexicon.screen_inches(card)
    price = float(card.price) if card.price is not None else float("inf")
    return (float(refresh or -1.0), float(size or -1.0), -price)


def preference_score(brief: BriefSheet, card: SkuCard) -> float:
    blob = card.text_blob()
    score = 0.0
    axes = " ".join(brief.soft_prefs + [brief.goal]).lower()

    def hit(*pats: str) -> bool:
        return any(re.search(p, blob, re.I) for p in pats)

    want_sweat = bool(re.search(r"sweat|water|rain|weather|waterproof|ipx?\d", axes, re.I))
    want_sport = bool(re.search(r"gym|run|sport|workout|fitness|training", axes, re.I))
    want_battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    want_wireless = bool(re.search(r"wireless|bluetooth|cordless|no.?wire", axes, re.I))
    want_noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))

    if want_sweat or want_sport:
        if hit(r"ipx?[0-9]", r"sweat", r"water.?resist", r"waterproof", r"splash"):
            score += 3.0
        if hit(r"sport", r"gym", r"running", r"workout", r"ear.?hook", r"secure.?fit"):
            score += 2.0
    if want_battery or want_sport:
        days = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*days?", blob)]
        hours = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*hours?", blob)]
        if days and max(days) >= 5:
            score += 2.0
        elif hours and max(hours) >= 20:
            score += 1.5
        elif hit(r"long.?battery", r"battery.?life"):
            score += 1.0
    if want_wireless or "headphone" in (brief.category or "").lower() or "earbud" in axes:
        if hit(r"bluetooth", r"wireless", r"tws", r"true.?wireless"):
            score += 1.5
        if hit(r"wired", r"3\.5\s*mm") and not hit(r"bluetooth", r"wireless"):
            score -= 1.0
    if want_noise and hit(r"\banc\b", r"noise.?cancel", r"active.?noise"):
        score += 1.5
    for pref in brief.soft_prefs:
        for tok in _tokens(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.2
    if brief.category and "headphone" in brief.category.lower():
        if hit(r"\bcase\b", r"cover", r"cable only", r"replacement tip") and not hit(
            r"earbud", r"headphone", r"headset", r"earphone"
        ):
            score -= 5.0
    return score


def soft_search_queries(brief: BriefSheet) -> list[str]:
    """Category-first soft queries from goal/soft_prefs axes only.

    Never emit pack-fixed preference phrases.
    """
    cat = (brief.category or "").strip()
    cat_l = cat.lower()
    axes = " ".join(brief.soft_prefs + [brief.goal]).lower()
    out: list[str] = []

    water = bool(re.search(r"sweat|water|rain|weather|waterproof|water.?resist", axes, re.I))
    sport = bool(re.search(r"gym|run|sport|workout|fitness|training", axes, re.I))
    battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    wireless = bool(re.search(r"wireless|bluetooth|cordless", axes, re.I))
    noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))
    audio = any(k in cat_l for k in ("headphone", "earbud", "headset", "earphone"))

    def push(parts: list[str]) -> None:
        q = " ".join(p for p in parts if p).strip()
        if q and q not in out:
            out.append(q)

    if audio:
        # Category noun first to avoid gym/fitness catalog drift.
        push([cat or "headphones", "wireless", "earbuds"])
        if water:
            push([cat or "earbuds", "waterproof", "sweat", "resistant"])
        if sport:
            push([cat or "earbuds", "sport", "wireless", "bluetooth"])
        if water and sport:
            push([cat or "earbuds", "waterproof", "sport", "bluetooth"])
        if battery:
            push([cat or "earbuds", "bluetooth", "battery"])
        if noise:
            push([cat or "earbuds", "noise", "cancelling", "wireless"])
        push([cat or "headphones", "bluetooth", "wireless"])
    else:
        seeds: list[str] = []
        if water:
            seeds.extend(["waterproof", "resistant"])
        if sport:
            seeds.append("sport")
        if battery:
            seeds.append("battery")
        if wireless:
            seeds.extend(["wireless", "bluetooth"])
        if noise:
            seeds.extend(["noise", "cancelling"])
        push(([cat] if cat else []) + seeds[:6])
        if cat and seeds:
            push(seeds[:4] + [cat])
    return out


def heuristic_raw(brief: BriefSheet, card: SkuCard, rank: int = 0) -> float:
    """EnvPack family heuristic before Weight.normalize."""
    score = 0.0
    if hard_ok(brief, card):
        score += 10.0
    else:
        if card.in_stock:
            score += 1.0
        if budget_ok(card, brief.budget):
            score += 1.0
        if category_ok(card, brief.category):
            score += 1.5
        for key, value in brief.attributes.items():
            if SpecLexicon.matches(card, key, value):
                score += 2.0
    prefer = brief.preferred_brand
    if prefer and brand_hit(card, prefer):
        bonus = 3.0 if brief.family == "intent_decomposition" else 1.5
        bonus += 0.5 * brand_strength(card, prefer)
        score += bonus
    if brief.brand and brand_hit(card, brief.brand):
        score += 1.0
        score += 0.25 * brand_strength(card, brief.brand)
    if brief.family in {"constraint_satisfaction", "justification"}:
        if EvidenceLedger.visible(brief, card):
            score += 2.5
        if OptionMention.hits(brief, card):
            score += 2.0
    blob = card.text_blob()
    for pref in brief.soft_prefs:
        for tok in _tokens(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.15
    for kw in brief.keywords:
        if _norm(kw) and _norm(kw) in _norm(blob):
            score += 0.2
    if brief.preferred_model and _norm(brief.preferred_model) in _norm(blob):
        score += 2.0
    if brief.preferred_color and _norm(brief.preferred_color) in _norm(
        blob + " " + card.options
    ):
        score += 1.0
    if brief.family == "ranking":
        refresh, size, neg_price = ranking_key(card)
        score += max(refresh, 0) * 10.0
        score += max(size, 0) * 1.0
        score += max(0.0, 5.0 + neg_price / 10000.0)
    if brief.family == "preference_reasoning":
        score += preference_score(brief, card) * 2.0
    if brief.family == "retrieval_recall" and brief.title_phrase:
        if title_phrase_match(card, brief.title_phrase):
            score += 8.0
    score -= rank * 0.01
    if card.price is not None and brief.budget and brief.family != "ranking":
        ratio = float(card.price) / float(brief.budget)
        if 0.4 <= ratio <= 0.95:
            score += 0.3
    if brief.candidate_keys and card.key in brief.candidate_keys:
        score += 50.0
    return score


def score_card(brief: BriefSheet, card: SkuCard, rank: int = 0) -> float:
    """Attach Weight and return score_n (primary ranking signal)."""
    raw = heuristic_raw(brief, card, rank=rank)
    # Approximate sku index from option richness for SKU_LIN_PROBS.
    sku_idx = -1
    if card.options:
        sku_idx = min(4, max(0, len(card.options.split("|")) - 1))
    card.weight = Weight().absorb_raw(raw, rank=rank, sku_idx=sku_idx)
    return card.weight.score_n


def first_cart_score(brief: BriefSheet, card: SkuCard) -> float:
    """Suite-agnostic first-cart ranking for event families."""
    s = float(heuristic_raw(brief, card))
    # listing ambiguity
    title = card.title or ""
    inch_hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", title, re.I)
    if len({h for h in inch_hits}) >= 2:
        s -= 2.5
    if re.search(r"\b\d+\s*(?:\"|in(?:ch)?)\s*/\s*\d+", title, re.I):
        s -= 2.0
    if re.search(r"\b[A-Za-z]?\d{2,}[A-Za-z0-9]*\s*/\s*[A-Za-z]?\d{2,}", title):
        s -= 1.5
    if title.count(" / ") >= 1 and len(title) > 80:
        s -= 0.5
    # budget position
    if brief.budget is not None and card.price is not None and float(brief.budget) > 0:
        ratio = float(card.price) / float(brief.budget)
        if ratio > 1.0:
            s -= 5.0
        elif ratio < 0.25:
            s -= 1.5
        elif 0.5 <= ratio <= 0.98:
            s += 1.2
        elif 0.35 <= ratio < 0.5:
            s += 0.3
    brand = soft_brand(brief)
    if brand:
        strength = brand_strength(card, brand)
        s += 1.0 * strength
        if strength >= 3:
            s += 1.0
        elif strength <= 1:
            s -= 0.8
    if brief.family == "justification":
        if EvidenceLedger.claimable(brief, card):
            s += 4.0
        elif EvidenceLedger.visible(brief, card):
            s += 1.0
        else:
            s -= 3.0
    if brief.attributes and OptionMention.hits(brief, card):
        s += 2.5
    if card.viewed:
        s += 0.8
    if not GuardRail.option_ok(card, brief.category):
        s -= 4.0
    return s


def soft_brand(brief: BriefSheet) -> Optional[str]:
    if brief.family == "justification":
        return brief.brand or brief.preferred_brand
    if brief.family == "intent_decomposition":
        return brief.preferred_brand
    if brief.family == "constraint_satisfaction":
        return brief.brand
    return brief.brand or brief.preferred_brand


def listing_ambiguity(card: SkuCard) -> float:
    title = card.title or ""
    pen = 0.0
    inch_hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", title, re.I)
    if len({h for h in inch_hits}) >= 2:
        pen -= 2.5
    if re.search(r"\b\d+\s*(?:\"|in(?:ch)?)\s*/\s*\d+", title, re.I):
        pen -= 2.0
    return pen


def card_from_summary(item: dict[str, Any]) -> SkuCard:
    return SkuCard(
        product_id=str(item.get("product_id") or ""),
        sku=str(item.get("sku") or "default"),
        title=str(item.get("title") or ""),
        brand=str(item.get("brand") or ""),
        price=_as_float(item.get("price")),
        currency=str(item.get("currency") or "PHP"),
        in_stock=bool(item.get("in_stock", True)),
    )


def cards_from_view(obs: dict[str, Any]) -> list[SkuCard]:
    if obs.get("error"):
        return []
    base = SkuCard(
        product_id=str(obs.get("product_id") or ""),
        sku="",
        title=str(obs.get("title") or ""),
        brand=str(obs.get("brand") or ""),
        category_path=list(obs.get("category_path") or []),
        facts=dict(obs.get("facts") or {}),
        listing_id=obs.get("listing_id"),
        viewed=True,
    )
    out: list[SkuCard] = []
    for row in obs.get("variants") or []:
        out.append(
            SkuCard(
                product_id=base.product_id,
                sku=str(row.get("sku") or "default"),
                title=base.title,
                brand=base.brand,
                price=_as_float(row.get("price")),
                currency=str(row.get("currency") or "PHP"),
                in_stock=bool(row.get("in_stock", True)),
                options=str(row.get("options") or ""),
                category_path=list(base.category_path),
                facts=dict(base.facts),
                listing_id=base.listing_id,
                viewed=True,
            )
        )
    return out or []



# ---------------------------------------------------------------------------
# QueryPipeline — classify family + BriefSheet from public query
# ---------------------------------------------------------------------------


class QueryPipeline:
    """Parse public query into BriefSheet (Newstart decompose spirit → EnvPack families)."""

    _WITHHELD = re.compile(
        r"(if you ask|will tell you|one firm requirement|tell you if you ask)",
        re.I,
    )

    @classmethod
    def is_withheld(cls, query: str) -> bool:
        return bool(cls._WITHHELD.search(query or ""))

    @classmethod
    def decompose(cls, query: str) -> BriefSheet:
        family = infer_family(query)
        goal = _goal_only(query)
        brief = BriefSheet(
            family=family,
            goal=goal,
            budget=_extract_budget(query),
            candidate_keys=_extract_candidate_keys(query),
        )
        # Retrieval title phrase from quotes.
        phrase_m = re.search(r'matches?\s+"([^"]+)"', goal, re.I)
        if phrase_m:
            brief.title_phrase = phrase_m.group(1).strip()

        try:
            js = llm_json(_BRIEF_PROMPT.format(query=query), expect="dict")
        except Exception:
            js = {}

        if isinstance(js, dict):
            brief.category = js.get("category") or brief.category
            brief.brand = js.get("brand") or brief.brand
            brief.preferred_brand = js.get("preferred_brand") or brief.preferred_brand
            if js.get("budget") is not None:
                try:
                    brief.budget = float(js["budget"])
                except (TypeError, ValueError):
                    pass
            brief.currency = str(js.get("currency") or brief.currency)
            if "require_in_stock" in js:
                brief.require_in_stock = bool(js["require_in_stock"])
            attrs = js.get("attributes") or {}
            if isinstance(attrs, dict):
                brief.attributes = {
                    str(k): str(v) for k, v in attrs.items() if v is not None and str(v)
                }
            soft = js.get("soft_prefs") or []
            if isinstance(soft, list):
                brief.soft_prefs = [str(x) for x in soft if str(x).strip()]
            kws = js.get("keywords") or []
            if isinstance(kws, list):
                brief.keywords = [str(x) for x in kws if str(x).strip()]
            sqs = js.get("search_queries") or []
            if isinstance(sqs, list):
                brief.search_queries = [str(x) for x in sqs if str(x).strip()]
            ckeys = js.get("candidate_keys") or []
            if isinstance(ckeys, list):
                for key in ckeys:
                    key_s = str(key)
                    if "::" in key_s and key_s not in brief.candidate_keys:
                        brief.candidate_keys.append(key_s)
            brief.preferred_model = js.get("preferred_model") or brief.preferred_model
            brief.preferred_color = js.get("preferred_color") or brief.preferred_color
            if "needs_clarification" in js:
                brief.needs_clarification = bool(js["needs_clarification"])

        cls._regex_overrides(brief, query, goal, family)
        return brief

    @classmethod
    def _regex_overrides(
        cls, brief: BriefSheet, query: str, goal: str, family: str
    ) -> None:
        if "screen_size" not in brief.attributes:
            inch_m = re.search(
                r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
                goal,
                re.I,
            )
            if inch_m:
                gate = EvidenceLedger.gate_from_goal(goal, brief.attributes)
                if gate in {None, "screen_size"} or family == "intent_decomposition":
                    if family != "justification" or gate == "screen_size":
                        brief.attributes["screen_size"] = f"{inch_m.group(1)}in"
        if family == "intent_decomposition":
            inch_m = re.search(
                r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
                goal,
                re.I,
            )
            if inch_m:
                brief.attributes["screen_size"] = f"{inch_m.group(1)}in"

        prefer_m = re.search(
            r"(?:ideally[, ]+)?(?:i\s+)?prefer(?:\s+the)?\s+([A-Za-z][A-Za-z0-9.\- ]{1,24})",
            goal,
            re.I,
        )
        if prefer_m:
            brand_guess = prefer_m.group(1).strip().rstrip(".,;")
            brand_guess = re.split(
                r"\b(?:but|and|if|,)\b", brand_guess, maxsplit=1
            )[0].strip()
            if brand_guess and brand_guess.lower() not in {"the", "a", "an", "black", "white"}:
                brief.preferred_brand = brief.preferred_brand or brand_guess
        if family in {"intent_decomposition", "preference_reasoning", "recovery", "generic"}:
            if (
                brief.brand
                and brief.preferred_brand
                and brief.brand.lower() == brief.preferred_brand.lower()
            ):
                brief.brand = None
            if brief.brand and re.search(
                r"\b(ideally|prefer|nice[- ]to[- ]have)\b", goal, re.I
            ):
                if "must be" not in goal.lower() and "only from" not in goal.lower():
                    brief.preferred_brand = brief.preferred_brand or brief.brand
                    brief.brand = None
            if family == "intent_decomposition" and brief.brand and "must" not in goal.lower():
                brief.preferred_brand = brief.preferred_brand or brief.brand
                brief.brand = None

        if family == "recovery":
            mc = re.search(
                r"prefer(?:\s+the)?\s+([a-z]+)\s+([A-Za-z0-9.\-]{4,})",
                goal,
                re.I,
            )
            if mc:
                brief.preferred_color = brief.preferred_color or mc.group(1)
                brief.preferred_model = brief.preferred_model or mc.group(2)

        if family == "justification":
            brand_m = re.match(
                r"(?:can you get me (?:an? )?in-stock\s+|find me (?:an?\s+)?|i want (?:an?\s+)?)"
                r"([A-Za-z][A-Za-z0-9.\- ]{1,24}?)"
                r"(?=\s+(?:\d|gaming|hd|monitor|smartphone|tablet|in-stock|at\b))",
                goal,
                re.I,
            )
            if brand_m:
                brief.preferred_brand = brief.preferred_brand or brand_m.group(1).strip()
                brief.brand = brief.brand or brand_m.group(1).strip()
            res_m = re.search(
                r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|720p|1080p|1440p|hd)\b", goal, re.I
            )
            if res_m and "resolution" not in brief.attributes:
                token = res_m.group(1).replace(" ", "").lower()
                brief.attributes["resolution"] = token
            inch_m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", goal, re.I)
            if inch_m and EvidenceLedger.gate_from_goal(goal, brief.attributes) == "screen_size":
                brief.attributes["screen_size"] = f"{inch_m.group(1)}in"
                brief.attributes.pop("resolution", None)

        if family == "preference_reasoning":
            for pat, label in (
                (r"sweat|water|rain|weather", "sweat/water resistance"),
                (r"gym|run|sport|workout|fitness", "sport secure fit"),
                (r"battery|all.?day|long.?lasting", "battery life"),
                (r"wireless|bluetooth", "wireless"),
                (r"noise|anc", "noise canceling"),
            ):
                if re.search(pat, goal, re.I):
                    brief.soft_prefs.append(label)

        withheld = cls.is_withheld(query)
        brief.needs_clarification = (
            family == "constraint_satisfaction" and withheld
        ) or (
            family == "preference_reasoning" and not brief.soft_prefs and withheld
        )
        if family == "constraint_satisfaction" and withheld:
            brief.attributes = {}

        brief.needs_justification = family == "justification"
        brief.require_in_stock = True

        normalized_attrs: dict[str, str] = {}
        for key, value in list(brief.attributes.items()):
            kind, wanted = SpecLexicon.normalize_wanted(key, str(value))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                normalized_attrs["screen_size"] = f"{float(wanted):g}in"
            elif kind == "resolution" and wanted:
                normalized_attrs["resolution"] = str(wanted)
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                normalized_attrs["refresh_rate"] = f"{int(wanted)}Hz"
            else:
                normalized_attrs[key] = str(value)
        brief.attributes = normalized_attrs

        if not brief.keywords:
            stop = {
                "i", "need", "want", "find", "looking", "for", "with", "and", "the",
                "a", "an", "to", "of", "in", "on", "at", "or", "my", "me", "can",
                "spend", "up", "under", "below", "php", "usd", "stock", "in-stock",
                "please",
            }
            tokens = [
                t
                for t in re.findall(r"[A-Za-z0-9.+\-]+", goal.lower())
                if t not in stop and len(t) > 1
            ]
            brief.keywords = tokens[:12]
        if not brief.search_queries:
            base = " ".join(brief.keywords[:8]) or goal[:80]
            brief.search_queries = [base]
            if brief.category:
                brief.search_queries.append(f"{brief.category} {base}")
            if brief.preferred_brand or brief.brand:
                brand = brief.brand or brief.preferred_brand
                brief.search_queries.append(f"{brand} {base}")
            for attr, val in list(brief.attributes.items())[:2]:
                brief.search_queries.append(f"{val} {base}")
        # Synonym expansions from Vocab (Newstart spirit).
        expanded: list[str] = []
        for q in brief.search_queries:
            expanded.extend(Vocab.expand_query_tokens(q)[:3])
        brief.search_queries = list(
            dict.fromkeys(q.strip() for q in (brief.search_queries + expanded) if q.strip())
        )

    @classmethod
    def merge_note(cls, brief: BriefSheet, note: str) -> BriefSheet:
        text = (note or "").strip()
        if not text:
            return brief
        new_budget = _extract_budget(text)
        if new_budget is not None:
            if brief.budget is None or new_budget < brief.budget:
                brief.budget = new_budget

        brand_m = re.search(
            r"(?:required brand is|brand is|must be(?: an?)?)\s+([A-Za-z][A-Za-z0-9.\-]{1,24})",
            text,
            re.I,
        )
        if brand_m and brief.family == "constraint_satisfaction":
            guess = brand_m.group(1).strip().rstrip(".,;")
            if guess and guess.lower() not in {"the", "a", "an", "smartphones", "smartphone"}:
                brief.brand = brief.brand or guess

        cat_m = re.search(
            r"(?:required catalog category is|category is)\s+([A-Za-z][A-Za-z0-9 &/\-]{2,40})",
            text,
            re.I,
        )
        if cat_m and not brief.category:
            cat = cat_m.group(1).strip().rstrip(".,;")
            cl = cat.lower()
            if "smartphone" in cl or "mobile" in cl:
                brief.category = "smartphone"
            elif "tablet" in cl:
                brief.category = "tablet"
            elif "monitor" in cl:
                brief.category = "monitor"
            elif "headphone" in cl or "earbud" in cl:
                brief.category = "headphone"
            else:
                brief.category = cat

        force = bool(re.search(r"\b(must now|instead of|required now|update)\b", text, re.I))
        reveal = bool(
            re.search(r"\b(exact|required|firm|must[- ]have|spec is)\b", text, re.I)
        )

        inch_m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?", text, re.I
        )
        if inch_m and (reveal or force or brief.family == "constraint_satisfaction"):
            brief.attributes["screen_size"] = f"{inch_m.group(1)}in"
        res_m = re.search(
            r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|720p|1080p|1440p|hd)\b", text, re.I
        )
        if res_m and (reveal or force or brief.family == "constraint_satisfaction"):
            token = res_m.group(1).replace(" ", "").lower()
            if token in {"fullhd", "fhd"}:
                token = "1080p"
            brief.attributes["resolution"] = token
        hz_m = re.search(r"\b(\d{2,3})\s*hz\b", text, re.I)
        if hz_m and (reveal or force):
            brief.attributes["refresh_rate"] = f"{hz_m.group(1)}Hz"
        ram_m = re.search(r"\b(\d+)\s*gb\s*ram\b", text, re.I)
        if ram_m and (reveal or force):
            brief.attributes["ram"] = f"{ram_m.group(1)}GB"

        if re.search(r"\b(prefer|rather|nice[- ]to[- ]have)\b", text, re.I):
            brief.soft_prefs.append(text[:120])

        # Renormalize
        normalized: dict[str, str] = {}
        for key, value in list(brief.attributes.items()):
            kind, wanted = SpecLexicon.normalize_wanted(key, str(value))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                normalized["screen_size"] = f"{float(wanted):g}in"
            elif kind == "resolution" and wanted:
                normalized["resolution"] = str(wanted)
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                normalized["refresh_rate"] = f"{int(wanted)}Hz"
            else:
                normalized[key] = str(value)
        brief.attributes = normalized
        return brief


# ---------------------------------------------------------------------------
# SweepEngine — search / filter harvest with synonym + unit extensions
# ---------------------------------------------------------------------------


class SweepEngine:
    """Harvest catalog hits via search/filter with Newstart-style query expansion."""

    def __init__(self, desk: Desk, brief: BriefSheet, pool: dict[str, SkuCard]):
        self.desk = desk
        self.brief = brief
        self.pool = pool
        self.observed_product_ids: list[str] = []
        self._requery_count = 0
        self._score_floors: dict[str, float] = {}

    def ingest(self, obs: dict[str, Any]) -> list[SkuCard]:
        out: list[SkuCard] = []
        for item in obs.get("results") or []:
            if not isinstance(item, dict) or not item.get("product_id"):
                continue
            card = card_from_summary(item)
            prev = self.pool.get(card.key)
            if prev and prev.viewed:
                prev.price = card.price if card.price is not None else prev.price
                prev.in_stock = card.in_stock
                card = prev
            self.pool[card.key] = card
            out.append(card)
            if card.product_id not in self.observed_product_ids:
                self.observed_product_ids.append(card.product_id)
        return out

    def search(
        self, query: str, *, k: int = 10, in_stock: Optional[bool] = True
    ) -> list[SkuCard]:
        if not self.desk.can("search"):
            if self.desk.can("filter"):
                return self.filter_catalog(
                    category=self.brief.category,
                    brand=self.brief.brand or self.brief.preferred_brand,
                    max_price=self.brief.budget,
                    k=k,
                    in_stock=in_stock,
                    _allow_search_fallback=False,
                )
            return []
        args: dict[str, Any] = {"query": query, "k": k}
        if in_stock is not None:
            args["in_stock"] = in_stock
        obs = self.desk.act("search", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        hits = self.ingest(obs)
        # Track Newstart-style score floor for this query.
        if hits:
            top = max(score_card(self.brief, c) for c in hits)
            self._score_floors[query.lower()] = top
        return hits

    def filter_catalog(
        self,
        *,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        max_price: Optional[float] = None,
        k: int = 10,
        in_stock: Optional[bool] = True,
        _allow_search_fallback: bool = True,
    ) -> list[SkuCard]:
        if not self.desk.can("filter"):
            if _allow_search_fallback and self.desk.can("search"):
                bits = [p for p in (brand, category) if p]
                if max_price is not None:
                    bits.append(str(max_price))
                q = " ".join(bits).strip() or (self.brief.category or "product")
                return self.search(q, k=k, in_stock=in_stock)
            return []
        args: dict[str, Any] = {"k": k}
        if category:
            args["category"] = category
        if brand:
            args["brand"] = brand
        if max_price is not None:
            args["max_price"] = max_price
        if in_stock is not None:
            args["in_stock"] = in_stock
        obs = self.desk.act("filter", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        out = self.ingest(obs)
        if brand:
            for c in out:
                c.source_brand = brand
                if not (c.brand or "").strip() and self.brief.brand:
                    c.brand = brand
                self.pool[c.key] = c
        return out

    def above_score_floor(self, card: SkuCard, query: str) -> bool:
        floor = self._score_floors.get(query.lower())
        if floor is None:
            return True
        return score_card(self.brief, card) >= floor - Weight.SCORE_N_THRES

    def firm_attr_queries(self) -> list[str]:
        brand = self.brief.brand or self.brief.preferred_brand or ""
        cat = self.brief.category or ""
        out: list[str] = []
        for key, val in self.brief.attributes.items():
            kind, wanted = SpecLexicon.normalize_wanted(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                inches = float(wanted)
                out.append(f"{brand} {inches:g} inch {cat}".strip())
                out.append(f"{inches:g} inch {cat} {brand}".strip())
                if inches >= 7.0 and cat and "phone" in cat.lower():
                    out.append(f"{brand} foldable {inches:g} inch smartphone".strip())
                    out.append(f"{brand} folding phone {inches:g}".strip())
            elif kind == "resolution" and wanted:
                token = str(wanted)
                out.append(f"{brand} {cat} {token}".strip())
                out.append(f"{brand} {token} {cat}".strip())
                if token in {"hd", "720p"}:
                    out.append(f"{brand} {cat} 720p HD".strip())
                    out.append(f"{brand} 1280x720 {cat}".strip())
                elif token in {"1080p", "fullhd", "fhd"}:
                    out.append(f"{brand} {cat} 1080p Full HD".strip())
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                out.append(f"{brand} {cat} {int(wanted)}Hz".strip())
            elif kind == "ram" and wanted:
                out.append(f"{brand} {cat} {wanted} RAM".strip())
            else:
                out.append(f"{brand} {cat} {val}".strip())
        seen: set[str] = set()
        uniq: list[str] = []
        for q in out:
            q = " ".join(q.split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        # Unit-extension variants (Newstart ProductSearchEngine spirit).
        extended: list[str] = []
        for q in uniq:
            extended.extend(Vocab.expand_query_tokens(q)[:2])
        for q in extended:
            if q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        return uniq

    def clarify_once(self) -> None:
        if self.brief.family == "constraint_satisfaction":
            content = (
                "Before I buy: what exact hardware spec must the variant meet "
                "(screen size, resolution, refresh rate, or RAM)?"
            )
        elif self.brief.family == "preference_reasoning":
            content = (
                "To pick the best match: which priorities matter most for your use case "
                "(battery, water/sweat resistance, fit, wireless, brand)?"
            )
        elif self.brief.family == "intent_decomposition":
            content = (
                "Quick check: which requirements are firm versus nice-to-have, "
                "and is the preferred brand optional if a firm-spec item is only "
                "available from another brand?"
            )
        else:
            content = "Before ordering, any must-have specs or budget updates I should apply?"
        self.desk.think("Asking shopper for withheld firm details.", think="clarify")
        if self.desk.can("message"):
            self.desk.act("message", content=content)
        if self.desk.last_shopper_note:
            self.brief = QueryPipeline.merge_note(self.brief, self.desk.last_shopper_note)
            self.desk.last_shopper_note = None
        if self.brief.family == "constraint_satisfaction" and self.brief.attributes:
            self.brief.search_queries = self.firm_attr_queries()[:4]
            self._requery_count = 0

    def spec_requery(self) -> None:
        if self._requery_count >= 2 or self.desk.remaining() <= 5:
            return
        self._requery_count += 1
        for q in self.firm_attr_queries()[:3]:
            if self.desk.remaining() <= 5:
                break
            self.search(q, k=10, in_stock=True)
        bits = [
            self.brief.brand or self.brief.preferred_brand or "",
            self.brief.category or "",
            " ".join(str(v) for v in self.brief.attributes.values()),
        ]
        q = " ".join(b for b in bits if b).strip()
        if q and self.desk.remaining() > 5:
            self.search(q, k=10, in_stock=True)
        if self.desk.remaining() > 5:
            brand = self.brief.brand
            if self.brief.preferred_brand and self.brief.family in {
                "justification",
                "constraint_satisfaction",
                "intent_decomposition",
            }:
                brand = brand or self.brief.preferred_brand
            self.filter_catalog(
                category=self.brief.category,
                brand=brand,
                max_price=self.brief.budget,
                k=10,
                in_stock=True,
            )

    def harvest(self) -> list[SkuCard]:
        found: list[SkuCard] = []
        for key in self.brief.candidate_keys:
            pid, _, sku = key.partition("::")
            if pid:
                card = SkuCard(product_id=pid, sku=sku or "default")
                found.append(card)
                self.pool.setdefault(card.key, card)

        if self.brief.family == "ranking" and self.brief.candidate_keys:
            self.desk.think(
                "Ranking family: restricting to candidate_keys from query.",
                think="sweep",
            )
            return list(found)

        queries = list(self.brief.search_queries)
        firm_qs = self.firm_attr_queries()
        if firm_qs and self.brief.family in EVENT_FAMILIES:
            queries = firm_qs + [q for q in queries if q not in firm_qs]
        if self.brief.family in {"constraint_satisfaction", "justification"}:
            brand = self.brief.brand or self.brief.preferred_brand or ""
            cat = self.brief.category or ""
            core = firm_qs[:3] if firm_qs else []
            if brand and cat:
                core.append(f"{brand} {cat}".strip())
            firm_toks = [str(v).lower() for v in self.brief.attributes.values()]
            for q in list(self.brief.search_queries):
                ql = q.lower()
                if any(t and t.replace("in", "")[:4] in ql for t in firm_toks) or any(
                    t in ql for t in ("720p", "1080p", "hd", "inch", "hz")
                ):
                    if q not in core:
                        core.append(q)
                if len(core) >= 4:
                    break
            queries = core[:5]
        if self.brief.family == "intent_decomposition" and self.brief.preferred_brand:
            bits = [
                self.brief.preferred_brand,
                self.brief.category or "smartphone",
                " ".join(str(v) for v in self.brief.attributes.values()),
            ]
            brand_q = " ".join(b for b in bits if b)
            queries = [brand_q, self.brief.preferred_brand] + [
                q for q in queries if q not in {brand_q, self.brief.preferred_brand}
            ]
            queries = firm_qs[:2] + [q for q in queries if q not in firm_qs[:2]]
        if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
            queries.insert(0, self.brief.title_phrase)
            queries.append(self.brief.title_phrase.replace(" ", ""))
        if self.brief.family == "preference_reasoning":
            extras = soft_search_queries(self.brief)
            queries = extras + [q for q in queries if q not in extras]
        if self.brief.family == "recovery" and self.brief.preferred_model:
            queries.insert(0, self.brief.preferred_model)
            if self.brief.preferred_color:
                queries.insert(
                    0, f"{self.brief.preferred_color} {self.brief.preferred_model}"
                )
        if self.brief.family == "justification":
            brand = self.brief.brand or self.brief.preferred_brand
            bits = [brand or "", self.brief.category or ""]
            for attr, val in self.brief.attributes.items():
                bits.append(str(val))
            queries.insert(0, " ".join(b for b in bits if b))

        # Synonym/unit expansion pass (cap to avoid step burn).
        expanded_queries: list[str] = []
        for q in queries:
            expanded_queries.append(q)
            for alt in Vocab.expand_query_tokens(q)[1:3]:
                if alt not in expanded_queries:
                    expanded_queries.append(alt)
        queries = expanded_queries

        self.desk.think(
            f"Sweep starting with {len(queries)} query variants.",
            think=json.dumps({"action": "search", "n": len(queries)}),
        )

        if (
            self.brief.family in EVENT_FAMILIES
            and firm_qs
            and self.desk.remaining() > 6
        ):
            for q in firm_qs[:2]:
                if self.desk.remaining() <= 6:
                    break
                found.extend(self.search(q, k=10, in_stock=True))
                if any(hard_ok(self.brief, c) for c in self.pool.values()):
                    break

        if (
            self.brief.family == "intent_decomposition"
            and self.brief.preferred_brand
            and self.desk.remaining() > 4
        ):
            found.extend(
                self.filter_catalog(
                    category=self.brief.category,
                    brand=self.brief.preferred_brand,
                    max_price=self.brief.budget,
                    k=10,
                    in_stock=True,
                )
            )

        for q in queries:
            if self.desk.remaining() <= 6:
                break
            if firm_qs and q in firm_qs[:2] and self.brief.family in EVENT_FAMILIES:
                continue
            found.extend(self.search(q, k=10, in_stock=True))
            if self.brief.family == "retrieval_recall" and len(self.observed_product_ids) >= 5:
                break
            if self.brief.family == "intent_decomposition":
                branded = [
                    c
                    for c in self.pool.values()
                    if self.brief.preferred_brand and brand_hit(c, self.brief.preferred_brand)
                ]
                if branded and any(hard_ok(self.brief, c) for c in branded):
                    break
                if branded and len(found) >= 10:
                    break
            elif self.brief.family in {"constraint_satisfaction", "justification"}:
                ok = [
                    c
                    for c in self.pool.values()
                    if hard_ok(self.brief, c) and EvidenceLedger.visible(self.brief, c)
                ]
                if ok:
                    break
            elif self.brief.family != "retrieval_recall" and len(found) >= 8:
                break

        if self.desk.remaining() > 6:
            brand = self.brief.brand
            if (
                not brand
                and self.brief.preferred_brand
                and self.brief.family != "intent_decomposition"
            ):
                brand = self.brief.preferred_brand
            found.extend(
                self.filter_catalog(
                    category=self.brief.category,
                    brand=brand,
                    max_price=self.brief.budget,
                    k=10,
                    in_stock=True,
                )
            )
            if (
                self.brief.family == "intent_decomposition"
                and self.brief.preferred_brand
                and self.desk.remaining() > 5
            ):
                found.extend(
                    self.filter_catalog(
                        category=self.brief.category,
                        brand=self.brief.preferred_brand,
                        max_price=self.brief.budget,
                        k=10,
                        in_stock=True,
                    )
                )

        uniq: dict[str, SkuCard] = {}
        for c in found:
            uniq[c.key] = self.pool.get(c.key, c)
        # Floor-filter when we have floors.
        kept = list(uniq.values())
        if self._score_floors:
            floored = [
                c
                for c in kept
                if any(self.above_score_floor(c, q) for q in self._score_floors)
            ]
            if floored:
                kept = floored
        return kept



# ---------------------------------------------------------------------------
# ViewBench — view / compare / inspect_stock survey
# ---------------------------------------------------------------------------


class ViewBench:
    """Inspect catalog cards: view/compare/stock with family-tuned depth."""

    def __init__(
        self,
        desk: Desk,
        brief: BriefSheet,
        pool: dict[str, SkuCard],
        sweep: SweepEngine,
    ):
        self.desk = desk
        self.brief = brief
        self.pool = pool
        self.sweep = sweep

    def view(self, product_id: str) -> list[SkuCard]:
        if not self.desk.can("view"):
            if self.desk.can("compare"):
                return self.compare([product_id])
            return []
        obs = self.desk.act("view", product_id=product_id)
        if obs.get("error") == "tool_unavailable":
            return []
        cards = cards_from_view(obs)
        prev_brand = ""
        prev_source = ""
        for prev in self.pool.values():
            if prev.product_id == product_id:
                prev_brand = prev.brand or prev_brand
                prev_source = prev.source_brand or prev_source
        for c in cards:
            if not (c.brand or "").strip():
                c.brand = prev_brand
            c.source_brand = c.source_brand or prev_source or c.brand
            self.pool[c.key] = c
            if c.product_id not in self.sweep.observed_product_ids:
                self.sweep.observed_product_ids.append(c.product_id)
        return cards

    def compare(self, product_ids: list[str]) -> list[SkuCard]:
        if not self.desk.can("compare"):
            out: list[SkuCard] = []
            for pid in product_ids[:5]:
                if self.desk.remaining() <= 0 or self.desk.finished:
                    break
                out.extend(self.view(pid))
            return out
        obs = self.desk.act("compare", product_ids=product_ids[:5])
        if obs.get("error") == "tool_unavailable":
            return []
        out: list[SkuCard] = []
        for row in obs.get("compared") or []:
            out.extend(cards_from_view(row))
        for c in out:
            self.pool[c.key] = c
            if c.product_id not in self.sweep.observed_product_ids:
                self.sweep.observed_product_ids.append(c.product_id)
        return out

    def inspect_stock(self, product_id: str, sku: str) -> SkuCard:
        key = f"{product_id}::{sku}"
        card = self.pool.get(key) or SkuCard(product_id=product_id, sku=sku)
        if not self.desk.can("inspect_stock"):
            return card
        obs = self.desk.act("inspect_stock", product_id=product_id, sku=sku)
        if obs.get("error"):
            return card
        card.price = _as_float(obs.get("price"))
        card.in_stock = bool(obs.get("in_stock", card.in_stock))
        card.currency = str(obs.get("currency") or card.currency)
        self.pool[key] = card
        return card

    def survey(self, candidates: list[SkuCard]) -> list[SkuCard]:
        product_ids: list[str] = []
        for c in candidates:
            if c.product_id not in product_ids:
                product_ids.append(c.product_id)

        min_views = 3 if self.brief.family == "retrieval_recall" else 2
        if self.brief.family == "preference_reasoning":
            min_views = 4
        if self.brief.family == "ranking" and self.brief.candidate_keys:
            min_views = min(
                5, len({k.split("::", 1)[0] for k in self.brief.candidate_keys})
            )
        if self.brief.family == "justification":
            min_views = 3
        if self.brief.family == "recovery":
            min_views = 2
        if self.brief.family == "intent_decomposition":
            min_views = 2
        if self.brief.family == "constraint_satisfaction":
            min_views = 1

        viewed: list[SkuCard] = []
        use_compare = (
            self.brief.family
            not in {"constraint_satisfaction", "justification", "intent_decomposition"}
            and self.desk.can("compare")
        )

        if self.brief.family == "intent_decomposition" and self.brief.preferred_brand:
            branded_ids = [
                pid
                for pid in product_ids
                if any(
                    brand_hit(c, self.brief.preferred_brand)
                    for c in self.pool.values()
                    if c.product_id == pid
                )
            ]
            other_ids = [pid for pid in product_ids if pid not in branded_ids]
            product_ids = branded_ids + other_ids

        if self.brief.family in EVENT_FAMILIES and self.brief.attributes:

            def _pid_firmness(pid: str) -> tuple[int, float]:
                vs = [c for c in self.pool.values() if c.product_id == pid]
                if not vs:
                    return (0, 0.0)
                best = max(vs, key=lambda c: score_card(self.brief, c))
                return (1 if hard_ok(self.brief, best) else 0, score_card(self.brief, best))

            product_ids.sort(key=_pid_firmness, reverse=True)

        self.desk.think(
            f"ViewBench surveying up to {min_views} product identities.",
            think="inspect",
        )

        while (
            product_ids
            and self.desk.remaining() > 4
            and len({c.product_id for c in viewed}) < min_views
        ):
            batch = product_ids[:5]
            product_ids = product_ids[5:]
            if use_compare and len(batch) >= 2:
                viewed.extend(self.compare(batch))
            else:
                for pid in batch:
                    if self.desk.remaining() <= 4:
                        break
                    viewed.extend(self.view(pid))
                    if len({c.product_id for c in viewed}) >= min_views:
                        break

        if len({c.product_id for c in viewed}) < min_views:
            ranked = sorted(
                self.pool.values(),
                key=lambda c: score_card(self.brief, c),
                reverse=True,
            )
            for c in ranked:
                if self.desk.remaining() <= 4:
                    break
                if any(x.product_id == c.product_id and x.viewed for x in viewed):
                    continue
                viewed.extend(self.view(c.product_id))
                if len({x.product_id for x in viewed}) >= min_views:
                    break

        out = [
            c
            for c in self.pool.values()
            if c.viewed or c.key in {x.key for x in candidates}
        ]
        if self.brief.family == "ranking" and self.brief.candidate_keys:
            out = [c for c in out if c.key in self.brief.candidate_keys]
            out.sort(key=ranking_key, reverse=True)
        else:
            out.sort(key=lambda c: score_card(self.brief, c), reverse=True)
        return out


# ---------------------------------------------------------------------------
# PickBench — Weight score_n ranking + optional LLM phase/tally select
# ---------------------------------------------------------------------------


class PickBench:
    """Select best SkuCard via Weight.score_n and optional Newstart-style tally."""

    def __init__(self, desk: Desk, brief: BriefSheet, pool: dict[str, SkuCard]):
        self.desk = desk
        self.brief = brief
        self.pool = pool

    def best_commit(self, cards: list[SkuCard]) -> Optional[SkuCard]:
        if not cards:
            return None
        buyable = [
            c
            for c in cards
            if hard_ok(self.brief, c)
            and purchase_ok(self.brief, c)
            and GuardRail.option_ok(c, self.brief.category)
        ]
        firm = buyable or [
            c
            for c in cards
            if hard_ok(self.brief, c) and GuardRail.option_ok(c, self.brief.category)
        ]
        if not firm:
            return None
        brand = soft_brand(self.brief)
        if brand:
            branded = [c for c in firm if brand_hit(c, brand)]
            if branded:
                strong = [c for c in branded if brand_strength(c, brand) >= 2]
                firm = strong or branded
        if self.brief.family == "justification":
            claimable = [c for c in firm if EvidenceLedger.claimable(self.brief, c)]
            if claimable:
                firm = claimable
            else:
                evidenced = [c for c in firm if EvidenceLedger.visible(self.brief, c)]
                if evidenced:
                    firm = evidenced
        if self.brief.family in EVENT_FAMILIES:
            firm.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
            return firm[0]
        firm.sort(
            key=lambda c: (
                brand_strength(c, brand) if brand else 0,
                score_card(self.brief, c),
            ),
            reverse=True,
        )
        return firm[0]

    def event_shortlist(self, cards: list[SkuCard], *, limit: int = 6) -> list[SkuCard]:
        pool = [
            c
            for c in cards
            if hard_ok(self.brief, c)
            and purchase_ok(self.brief, c)
            and GuardRail.option_ok(c, self.brief.category)
        ]
        if self.brief.family == "justification":
            claimable = [c for c in pool if EvidenceLedger.claimable(self.brief, c)]
            if claimable:
                pool = claimable
        brand = soft_brand(self.brief)
        if brand:
            strong = [c for c in pool if brand_strength(c, brand) >= 2]
            if strong:
                pool = strong
        pool.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
        out: list[SkuCard] = []
        seen: set[str] = set()
        for c in pool:
            if c.product_id in seen:
                continue
            seen.add(c.product_id)
            out.append(c)
            if len(out) >= limit:
                break
        return out

    def choose(self, shortlist: list[SkuCard]) -> Optional[SkuCard]:
        if self.brief.family == "ranking" and self.brief.candidate_keys:
            pool = [
                c
                for c in shortlist
                if c.key in self.brief.candidate_keys
                and c.in_stock
                and budget_ok(c, self.brief.budget)
            ]
            if not pool:
                pool = [
                    c
                    for c in shortlist
                    if c.key in self.brief.candidate_keys and c.in_stock
                ]
            if not pool:
                return None
            pool.sort(key=ranking_key, reverse=True)
            return pool[0]

        hard = [c for c in shortlist if hard_ok(self.brief, c)]
        pool = hard or [c for c in shortlist if c.in_stock]
        if not pool:
            return None

        if self.brief.family == "intent_decomposition" and self.brief.preferred_brand:
            branded = [c for c in hard if brand_hit(c, self.brief.preferred_brand)]
            if branded:
                pool = branded

        if self.brief.family == "recovery":
            exact = [
                c
                for c in hard
                if (
                    not self.brief.preferred_model
                    or _norm(self.brief.preferred_model) in _norm(c.text_blob())
                )
                and (
                    not self.brief.preferred_color
                    or _norm(self.brief.preferred_color)
                    in _norm(c.text_blob() + " " + c.options)
                )
            ]
            if exact:
                pool = exact

        if self.brief.family == "preference_reasoning":
            # Never fall back to off-category in_stock junk.
            pref_pool = [
                c
                for c in shortlist
                if hard_ok(self.brief, c) and purchase_ok(self.brief, c)
            ]
            if not pref_pool and self.desk.remaining() > 4:
                for q in soft_search_queries(self.brief)[:3]:
                    if self.desk.remaining() <= 3:
                        break
                    self.sweep.search(q, k=10, in_stock=True)
                pref_pool = [
                    c
                    for c in self.pool.values()
                    if hard_ok(self.brief, c) and purchase_ok(self.brief, c)
                ]
            if not pref_pool:
                return None
            viewed = [c for c in pref_pool if c.viewed]
            if viewed:
                pref_pool = viewed
            pref_pool = sorted(
                pref_pool,
                key=lambda c: (preference_score(self.brief, c), score_card(self.brief, c)),
                reverse=True,
            )
            return pref_pool[0] if pref_pool else None

        if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
            matched = [
                c for c in hard if title_phrase_match(c, self.brief.title_phrase)
            ]
            if not matched and self.desk.remaining() > 4:
                self.sweep.search(self.brief.title_phrase, k=10, in_stock=True)
                hard = [c for c in self.pool.values() if hard_ok(self.brief, c)]
                matched = [
                    c for c in hard if title_phrase_match(c, self.brief.title_phrase)
                ]
            if not matched:
                return None
            matched = sorted(
                matched,
                key=lambda c: (1 if c.viewed else 0, score_card(self.brief, c)),
                reverse=True,
            )
            return matched[0]

        if self.brief.family == "justification":
            want = self.brief.brand or self.brief.preferred_brand
            branded = [c for c in hard if want and brand_hit(c, want)]
            if branded:
                pool = branded
            claimable = [c for c in pool if EvidenceLedger.claimable(self.brief, c)]
            if claimable:
                pool = claimable
            else:
                evidenced = [c for c in pool if EvidenceLedger.visible(self.brief, c)]
                if evidenced:
                    pool = evidenced
            viewed = [c for c in pool if c.viewed]
            if viewed:
                pool = viewed

        if self.brief.family == "constraint_satisfaction":
            evidenced = [c for c in hard if EvidenceLedger.visible(self.brief, c)]
            if evidenced:
                pool = evidenced
            mentioned = [c for c in pool if OptionMention.hits(self.brief, c)]
            if mentioned:
                pool = mentioned
            viewed = [c for c in pool if c.viewed]
            if viewed:
                pool = viewed

        if self.brief.family in EVENT_FAMILIES:
            pool = sorted(pool, key=lambda c: first_cart_score(self.brief, c), reverse=True)
        else:
            pool = sorted(pool, key=lambda c: score_card(self.brief, c), reverse=True)

        top = pool[:5]
        # Newstart phase/tally LLM pick for ambiguous non-event families.
        if (
            len(top) >= 2
            and self.desk.remaining() > 8
            and self.brief.family
            not in {
                "ranking",
                "recovery",
                "justification",
                "preference_reasoning",
                "intent_decomposition",
                "constraint_satisfaction",
            }
        ):
            try:
                pick = self._phase_tally_select(top)
                if pick is not None:
                    return pick
            except Exception:
                pass

        if top:
            self.desk.think(
                f"Best-matching candidate selected: `{top[0].product_id}`",
                think=f"score_n={top[0].weight.score_n:.4f} conf={top[0].weight.confidence:.4f}",
            )
        return top[0] if top else None

    def _phase_tally_select(self, cards: list[SkuCard]) -> Optional[SkuCard]:
        """Newstart-style phase1 / tally LLM adjudication."""
        payload = []
        for i, c in enumerate(cards, start=1):
            payload.append(
                {
                    "id": i,
                    "product_id": c.product_id,
                    "sku": c.sku,
                    "title": c.title,
                    "brand": c.brand,
                    "price": c.price,
                    "in_stock": c.in_stock,
                    "options": c.options,
                    "facts": c.facts,
                    "category_path": c.category_path,
                    "score_n": c.weight.score_n,
                    "confidence": c.weight.confidence,
                }
            )
        prompt = (
            "Select the best purchasable variant. Hard constraints must all pass; "
            "soft preferences are secondary.\n"
            "Use Newstart phase/tally style:\n"
            "phase1: evidence quality per constraint "
            "(admissibility → structured → extra_tokens).\n"
            "tally: '<id>→<c1>:key:…,admissible:bool,structured:bool,extra_tokens:n; …'\n"
            f"Family: {self.brief.family}\n"
            f"Goal: {self.brief.goal}\n"
            f"Budget: {self.brief.budget} {self.brief.currency}\n"
            f"Category: {self.brief.category}\n"
            f"Mandatory brand: {self.brief.brand}\n"
            f"Preferred brand: {self.brief.preferred_brand}\n"
            f"Attributes: {json.dumps(self.brief.attributes)}\n"
            f"Soft prefs: {self.brief.soft_prefs}\n"
            f"Candidates:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            'Return JSON: {"best": <int|null>, "closest": <int|null>, '
            '"phase1": <str>, "tally": <str>, "recommend": <str>}'
        )
        js = llm_json(prompt, expect="dict")
        bid = js.get("best")
        if bid is None:
            bid = js.get("closest")
        try:
            bid_i = int(bid or 0)
        except (TypeError, ValueError):
            return None
        if 1 <= bid_i <= len(cards):
            pick = cards[bid_i - 1]
            self.desk.think(
                f"Best-matching candidate selected: `{pick.product_id}`",
                think=f"phase1={js.get('phase1','')}; tally={js.get('tally','')}",
            )
            return pick
        return None


# ---------------------------------------------------------------------------
# SealBench — first-cart → recover → place_order
# ---------------------------------------------------------------------------


class SealBench:
    """Seal purchase: add_to_cart, apply market events, recover, place_test_order."""

    def __init__(
        self,
        desk: Desk,
        brief: BriefSheet,
        pool: dict[str, SkuCard],
        sweep: SweepEngine,
        view: ViewBench,
        pick: PickBench,
    ):
        self.desk = desk
        self.brief = brief
        self.pool = pool
        self.sweep = sweep
        self.view = view
        self.pick = pick

    def message(self, content: str) -> None:
        if not self.desk.can("message"):
            return
        self.desk.act("message", content=content)
        if self.desk.last_shopper_note:
            self.brief = QueryPipeline.merge_note(self.brief, self.desk.last_shopper_note)
            self.desk.last_shopper_note = None

    def handle_shopper_note(self, *, pending: Optional[SkuCard] = None) -> None:
        note = self.desk.last_shopper_note
        if not note:
            return
        self.brief = QueryPipeline.merge_note(self.brief, note)
        if pending is not None and re.search(
            r"\b(not available|out of stock|sold out|still available|stockout|unavailable)\b",
            note,
            re.I,
        ):
            pending.in_stock = False
            self.pool[pending.key] = pending
        if pending is not None:
            price_jump = re.search(
                r"(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:to|->)\s*(\d+(?:\.\d+)?)\s*php",
                note,
                re.I,
            )
            if price_jump:
                pending.price = float(price_jump.group(2))
                self.pool[pending.key] = pending
        # Skip courtesy replies when market events fire and steps are tight.
        if is_market_event_note(note) and self.desk.remaining() <= 4:
            self.desk.last_shopper_note = None
            return
        if shopper_needs_reply(note) and self.desk.remaining() > 1:
            reply = self._compose_reply(note, pending)
            self.desk.last_shopper_note = None
            self.message(reply)
        else:
            self.desk.last_shopper_note = None

    def _compose_reply(self, note: str, choice: Optional[SkuCard]) -> str:
        if choice is not None:
            bits = [
                f"I'm going with {choice.title or choice.product_id}",
                f"sku {choice.sku}",
            ]
            if choice.price is not None:
                bits.append(f"at {choice.price} {choice.currency}")
            inches = SpecLexicon.screen_inches(choice)
            if inches is not None:
                bits.append(f"measured_screen={inches:g}in")
            if hard_ok(self.brief, choice):
                bits.append("firm-requirements: met")
            else:
                bits.append("firm-requirements: still checking")
            return "; ".join(bits) + ". I'll verify live price/stock before ordering."
        lower = note.lower()
        if "balance" in lower or "budget" in lower or "php" in lower:
            return (
                f"Understood — I'll stay within {self.brief.budget} {self.brief.currency} "
                "and re-check stock/price before ordering."
            )
        if "expensive" in lower or "price" in lower:
            return "Thanks for the heads-up — I'll re-check the live price and switch if needed."
        return (
            "Got it. I'll apply your latest requirements (specs + budget + in-stock) "
            "and only order a firm-compliant item."
        )

    def add_to_cart(self, product_id: str, sku: str) -> dict[str, Any]:
        if not self.desk.can("add_to_cart"):
            return {"error": "tool_unavailable", "requested": "add_to_cart"}
        obs = self.desk.act("add_to_cart", product_id=product_id, sku=sku)
        self.handle_shopper_note(pending=self.pool.get(f"{product_id}::{sku}"))
        return obs

    def remove_from_cart(self, product_id: str, sku: str) -> None:
        if self.desk.can("remove_from_cart"):
            self.desk.act("remove_from_cart", product_id=product_id, sku=sku)
            self.handle_shopper_note()

    def place_order(
        self,
        card: SkuCard,
        justification: Optional[dict] = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.desk.can("place_order"):
            return {"error": "tool_unavailable", "requested": "place_order"}
        if not force and not purchase_ok(self.brief, card):
            return {}
        if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
            if not title_phrase_match(card, self.brief.title_phrase):
                if not force:
                    return {}
        if (
            self.brief.family == "justification"
            and self.desk.remaining() > 0
            and not force
        ):
            card = self.view.inspect_stock(card.product_id, card.sku)
            if not purchase_ok(self.brief, card):
                return {}
            if not EvidenceLedger.claimable(self.brief, card):
                return {}
        args: dict[str, Any] = {"product_id": card.product_id, "sku": card.sku}
        if justification is not None:
            args["justification"] = justification
        elif self.brief.family == "justification":
            just = ClaimSheet.build(self.brief, card)
            fields = {c.get("field") for c in (just.get("claims") or [])}
            gate = EvidenceLedger.gate_key(self.brief)
            if gate and gate not in fields:
                if not force:
                    return {}
            args["justification"] = just
        return self.desk.act("place_order", **args)

    def prepare_first_cart(self, choice: SkuCard) -> Optional[SkuCard]:
        if self.brief.family not in EVENT_FAMILIES:
            return choice

        def pick_from_product(pid: str) -> Optional[SkuCard]:
            cards = self.view.view(pid) if self.desk.remaining() > 2 else []
            self.handle_shopper_note()
            best = self.pick.best_commit(cards)
            if best is None:
                return None
            if self.desk.remaining() > 2:
                best = self.view.inspect_stock(best.product_id, best.sku)
                self.handle_shopper_note(pending=best)
            if not (hard_ok(self.brief, best) and purchase_ok(self.brief, best)):
                return None
            if self.brief.family == "justification" and not EvidenceLedger.claimable(
                self.brief, best
            ):
                return None
            return best

        brand = soft_brand(self.brief)
        if brand and self.brief.family == "intent_decomposition":
            branded_hits = [c for c in self.pool.values() if brand_hit(c, brand)]
            if not branded_hits:
                self._brand_hunt()

        ranked = self.pick.event_shortlist(list(self.pool.values()), limit=6)
        try_pids = [c.product_id for c in ranked]
        if choice.product_id not in try_pids:
            try_pids.append(choice.product_id)

        picked_ok: list[SkuCard] = []
        for pid in try_pids:
            if self.desk.remaining() <= 5:
                break
            picked = pick_from_product(pid)
            if picked is None:
                continue
            picked_ok.append(picked)
            if (
                listing_ambiguity(picked) >= -0.5
                and (
                    self.brief.family != "justification"
                    or EvidenceLedger.claimable(self.brief, picked)
                )
                and (not brand or brand_strength(picked, brand) >= 2)
            ):
                if len(picked_ok) >= 2 or self.desk.remaining() <= 8:
                    break

        if not picked_ok:
            return choice if hard_ok(self.brief, choice) else None
        picked_ok.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
        best = picked_ok[0]
        if self.brief.family == "justification" and not EvidenceLedger.claimable(
            self.brief, best
        ):
            return None
        return best

    def _brand_hunt(self) -> None:
        brand = soft_brand(self.brief)
        if not brand or self.desk.remaining() <= 5:
            return
        for q in self.sweep.firm_attr_queries()[:2]:
            if self.desk.remaining() <= 4:
                break
            self.sweep.search(q, k=10, in_stock=True)
        bits = [
            brand,
            self.brief.category or "",
            " ".join(str(v) for v in self.brief.attributes.values()),
        ]
        q = " ".join(b for b in bits if b)
        self.sweep.search(q, k=10, in_stock=True)
        if self.desk.remaining() > 4:
            self.sweep.filter_catalog(
                category=self.brief.category,
                brand=brand,
                max_price=self.brief.budget,
                k=10,
                in_stock=True,
            )
        self.handle_shopper_note()

    def seal(self, choice: SkuCard) -> None:
        if (
            self.brief.family in {"intent_decomposition", "constraint_satisfaction"}
            and self.brief.attributes
            and self.desk.remaining() > 8
            and not hard_ok(self.brief, choice)
        ):
            self.sweep.spec_requery()
            alts = [c for c in self.pool.values() if hard_ok(self.brief, c)]
            if self.brief.family == "intent_decomposition" and self.brief.preferred_brand:
                branded = [c for c in alts if brand_hit(c, self.brief.preferred_brand)]
                if branded:
                    alts = branded
            if alts:
                alts.sort(key=lambda c: score_card(self.brief, c), reverse=True)
                choice = alts[0]

        self.handle_shopper_note(pending=choice)
        prepared = self.prepare_first_cart(choice)
        if prepared is None:
            alts = [c for c in self.pool.values() if hard_ok(self.brief, c)]
            if self.brief.family == "justification":
                want = self.brief.brand or self.brief.preferred_brand
                branded = [c for c in alts if want and brand_hit(c, want)]
                if branded:
                    alts = branded
                claimable = [c for c in alts if EvidenceLedger.claimable(self.brief, c)]
                if claimable:
                    alts = claimable
            if self.brief.family == "intent_decomposition" and self.brief.preferred_brand:
                branded = [c for c in alts if brand_hit(c, self.brief.preferred_brand)]
                if branded:
                    alts = branded
            if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
                alts = [c for c in alts if title_phrase_match(c, self.brief.title_phrase)]
            alts.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
            if not alts:
                return
            choice = alts[0]
            if not choice.viewed and self.desk.remaining() > 3:
                self.view.view(choice.product_id)
                choice = (
                    self.pick.best_commit(
                        [c for c in self.pool.values() if c.product_id == choice.product_id]
                    )
                    or choice
                )
            if self.brief.family == "justification" and not EvidenceLedger.claimable(
                self.brief, choice
            ):
                if self.desk.remaining() > 6:
                    self.sweep.spec_requery()
                    alts = [
                        c
                        for c in self.pool.values()
                        if hard_ok(self.brief, c)
                        and EvidenceLedger.claimable(self.brief, c)
                    ]
                    alts.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
                    if not alts:
                        return
                    choice = alts[0]
                else:
                    return
        else:
            choice = prepared

        if self.brief.family == "ranking":
            if not choice.viewed and self.desk.remaining() > 3:
                cards = self.view.view(choice.product_id)
                if cards:
                    cards.sort(key=ranking_key, reverse=True)
                    choice = cards[0]
            if self.desk.remaining() > 3:
                choice = self.view.inspect_stock(choice.product_id, choice.sku)
                self.handle_shopper_note(pending=choice)
        elif self.brief.family not in EVENT_FAMILIES:
            if not choice.viewed and self.desk.remaining() > 3:
                cards = self.view.view(choice.product_id)
                self.handle_shopper_note(pending=choice)
                picked = self.pick.best_commit(cards)
                if picked is not None:
                    choice = picked
                elif cards:
                    cards.sort(key=lambda c: score_card(self.brief, c), reverse=True)
                    choice = cards[0]
            if self.desk.remaining() > 3:
                choice = self.view.inspect_stock(choice.product_id, choice.sku)
                self.handle_shopper_note(pending=choice)

        if self.desk.remaining() < 2:
            return

        self.desk.think(
            f"First cart commitment: `{choice.product_id}` / {choice.sku}",
            think="seal",
        )
        add_obs = self.add_to_cart(choice.product_id, choice.sku)
        if self.desk.finished:
            return

        live = self.pool.get(choice.key) or choice
        live = CartPulse.apply_add(live, add_obs or {})
        self.pool[live.key] = live

        if self.desk.remaining() > 0:
            live = self.view.inspect_stock(choice.product_id, choice.sku)
            live = CartPulse.apply_add(live, add_obs or {})
            self.pool[live.key] = live
            self.handle_shopper_note(pending=live)
        else:
            self.handle_shopper_note(pending=live)

        if not needs_event_recovery(self.brief, live):
            if self.brief.family == "ranking":
                replacement = self.recover_ranking(exclude_keys=set(), previous=live)
                if replacement is not None and replacement.key != live.key:
                    self.remove_from_cart(choice.product_id, choice.sku)
                    if self.desk.remaining() > 1:
                        if not replacement.viewed:
                            self.view.view(replacement.product_id)
                        self.view.inspect_stock(replacement.product_id, replacement.sku)
                    self.add_to_cart(replacement.product_id, replacement.sku)
                    if self.desk.finished:
                        return
                    if self.desk.remaining() > 0:
                        replacement = self.view.inspect_stock(
                            replacement.product_id, replacement.sku
                        )
                    if self.desk.remaining() > 0 and purchase_ok(self.brief, replacement):
                        self.place_order(replacement)
                    return
            if purchase_ok(self.brief, live):
                self.place_order(live)
            return

        self.remove_from_cart(choice.product_id, choice.sku)
        if self.brief.family == "ranking":
            replacement = self.recover_ranking(exclude_keys={choice.key}, previous=choice)
        else:
            replacement = self.recover(exclude_keys={choice.key}, previous=choice)
        if replacement is None:
            if purchase_ok(self.brief, live):
                self.add_to_cart(live.product_id, live.sku)
                if not self.desk.finished and purchase_ok(self.brief, live):
                    self.place_order(live)
            elif self.desk.remaining() > 0:
                self.force_done()
            return

        if self.desk.remaining() > 1:
            if not replacement.viewed:
                self.view.view(replacement.product_id)
            self.view.inspect_stock(replacement.product_id, replacement.sku)
            replacement = self.pool.get(replacement.key, replacement)
        if not purchase_ok(self.brief, replacement):
            alt = self.recover(
                exclude_keys={choice.key, replacement.key}, previous=replacement
            )
            if alt is None or not purchase_ok(self.brief, alt):
                if self.desk.remaining() > 0:
                    self.force_done()
                return
            replacement = alt
        self.add_to_cart(replacement.product_id, replacement.sku)
        if self.desk.finished:
            return
        if self.desk.remaining() > 0:
            replacement = self.view.inspect_stock(replacement.product_id, replacement.sku)
            self.pool[replacement.key] = replacement
        if self.desk.remaining() > 0 and purchase_ok(self.brief, replacement):
            self.place_order(replacement)
        elif self.desk.remaining() > 0:
            self.force_done()

    def recover_ranking(
        self, *, exclude_keys: set[str], previous: SkuCard
    ) -> Optional[SkuCard]:
        keys = self.brief.candidate_keys or []
        pool: list[SkuCard] = []
        for key in keys:
            if key in exclude_keys:
                continue
            c = self.pool.get(key)
            if c is None:
                pid, _, sku = key.partition("::")
                c = SkuCard(product_id=pid, sku=sku or "default")
                self.pool[key] = c
            if not c.viewed and self.desk.remaining() > 3:
                self.view.view(c.product_id)
                c = self.pool.get(key, c)
            if self.desk.remaining() > 2:
                c = self.view.inspect_stock(c.product_id, c.sku)
            if c.in_stock and budget_ok(c, self.brief.budget):
                pool.append(c)
        if not pool:
            return None
        pool.sort(key=ranking_key, reverse=True)
        return pool[0]

    def recover(
        self, *, exclude_keys: set[str], previous: SkuCard
    ) -> Optional[SkuCard]:
        if previous.product_id and self.desk.remaining() > 2:
            cards = self.view.view(previous.product_id)
            siblings = [
                c
                for c in cards
                if c.key not in exclude_keys
                and hard_ok(self.brief, c)
                and purchase_ok(self.brief, c)
            ]
            if siblings:
                siblings.sort(key=lambda c: score_card(self.brief, c), reverse=True)
                return siblings[0]
            for c in cards:
                if c.key in exclude_keys or not hard_ok(self.brief, c):
                    continue
                if self.desk.remaining() <= 2:
                    break
                live = self.view.inspect_stock(c.product_id, c.sku)
                if purchase_ok(self.brief, live) and hard_ok(self.brief, live):
                    return live

        if self.desk.remaining() > 2:
            if self.brief.family == "recovery" and self.brief.preferred_model:
                self.sweep.search(self.brief.preferred_model, k=10, in_stock=True)
                if self.brief.preferred_color and self.desk.remaining() > 2:
                    self.sweep.search(
                        f"{self.brief.preferred_color} {self.brief.preferred_model}",
                        k=10,
                        in_stock=True,
                    )
            elif self.brief.family == "preference_reasoning":
                # Re-run soft_search_queries; require category gate.
                for q in soft_search_queries(self.brief)[:3]:
                    if self.desk.remaining() <= 2:
                        break
                    self.sweep.search(q, k=10, in_stock=True)
            elif self.brief.family == "retrieval_recall" and self.brief.title_phrase:
                self.sweep.search(self.brief.title_phrase, k=10, in_stock=True)
            elif self.brief.family in EVENT_FAMILIES:
                self.sweep.spec_requery()
            else:
                q = " ".join(self.brief.keywords[:6]) or self.brief.goal[:60]
                self.sweep.search(q, k=10, in_stock=True)
                if self.desk.remaining() > 2:
                    self.sweep.filter_catalog(
                        category=self.brief.category,
                        brand=self.brief.brand,
                        max_price=self.brief.budget,
                        k=10,
                        in_stock=True,
                    )
                self.handle_shopper_note()

        candidates = [
            c
            for c in self.pool.values()
            if c.key not in exclude_keys
            and hard_ok(self.brief, c)
            and purchase_ok(self.brief, c)
        ]
        if self.brief.family == "preference_reasoning":
            candidates = [
                c for c in candidates if category_ok(c, self.brief.category)
            ] or candidates
        if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
            candidates = [
                c for c in candidates if title_phrase_match(c, self.brief.title_phrase)
            ]

        if self.brief.family == "recovery":
            same_model = []
            if self.brief.preferred_model:
                same_model = [
                    c
                    for c in candidates
                    if _norm(self.brief.preferred_model)
                    in _norm(c.text_blob() + " " + c.options)
                ]
            if same_model and self.brief.preferred_color:
                other_color = [
                    c
                    for c in same_model
                    if _norm(self.brief.preferred_color)
                    not in _norm(c.text_blob() + " " + c.options)
                ]
                pool = other_color or same_model
            else:
                pool = same_model or candidates
        elif self.brief.family == "preference_reasoning":
            prev_listing = previous.listing_id
            diversed = [
                c
                for c in candidates
                if c.listing_id and c.listing_id != prev_listing
            ]
            pool = diversed or candidates
            pool = sorted(
                pool, key=lambda c: preference_score(self.brief, c), reverse=True
            )
            return pool[0] if pool else None
        elif self.brief.family == "justification":
            want = self.brief.brand or self.brief.preferred_brand
            branded = [c for c in candidates if want and brand_hit(c, want)]
            pool = branded or candidates
        elif self.brief.family == "intent_decomposition":
            if self.brief.preferred_brand:
                branded = [
                    c for c in candidates if brand_hit(c, self.brief.preferred_brand)
                ]
                pool = branded or candidates
            else:
                pool = candidates
        else:
            same = [c for c in candidates if c.product_id == previous.product_id]
            pool = same or candidates

        if not pool:
            if self.brief.family in EVENT_FAMILIES:
                return None
            pool = [
                c
                for c in self.pool.values()
                if c.key not in exclude_keys and purchase_ok(self.brief, c)
            ]
            if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
                pool = [
                    c for c in pool if title_phrase_match(c, self.brief.title_phrase)
                ]
        if not pool:
            return None

        if self.brief.family in EVENT_FAMILIES:
            pool.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
        else:
            pool.sort(key=lambda c: score_card(self.brief, c), reverse=True)
        best = pool[0]
        if not best.viewed and self.desk.remaining() > 2:
            cards = self.view.view(best.product_id)
            cards = [
                c
                for c in cards
                if (hard_ok(self.brief, c) or purchase_ok(self.brief, c))
                and c.key not in exclude_keys
            ]
            if cards:
                if self.brief.family in EVENT_FAMILIES:
                    cards.sort(
                        key=lambda c: first_cart_score(self.brief, c), reverse=True
                    )
                else:
                    cards.sort(key=lambda c: score_card(self.brief, c), reverse=True)
                best = cards[0]
        if not purchase_ok(self.brief, best):
            return None
        return best

    def force_done(self) -> None:
        """Always try to reach done=true before exiting."""
        if self.desk.finished or self.desk.remaining() <= 0:
            return
        if (
            self.brief.family in EVENT_FAMILIES
            and self.brief.attributes
            and self.desk.remaining() > 5
            and not any(
                hard_ok(self.brief, c) and EvidenceLedger.visible(self.brief, c)
                for c in self.pool.values()
            )
        ):
            self.sweep.spec_requery()
        pool = [c for c in self.pool.values() if purchase_ok(self.brief, c)]
        hard = [c for c in pool if hard_ok(self.brief, c)]
        if hard:
            pool = hard
        elif self.brief.family in EVENT_FAMILIES:
            if self.desk.remaining() > 4 and self.brief.attributes:
                self.sweep.spec_requery()
                pool = [c for c in self.pool.values() if purchase_ok(self.brief, c)]
                hard = [c for c in pool if hard_ok(self.brief, c)]
                if hard:
                    pool = hard
        if self.brief.family == "justification":
            claimable = [c for c in pool if EvidenceLedger.claimable(self.brief, c)]
            if claimable:
                pool = claimable
        if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
            matched = [
                c for c in pool if title_phrase_match(c, self.brief.title_phrase)
            ]
            if matched:
                pool = matched
            elif self.desk.remaining() > 3:
                self.sweep.search(self.brief.title_phrase, k=8, in_stock=True)
                pool = [
                    c
                    for c in self.pool.values()
                    if purchase_ok(self.brief, c)
                    and title_phrase_match(c, self.brief.title_phrase)
                ]
                if not pool:
                    return
            else:
                return
        if self.brief.family == "preference_reasoning":
            pool = [c for c in pool if hard_ok(self.brief, c)]
            if not pool:
                for q in soft_search_queries(self.brief)[:2]:
                    if self.desk.remaining() <= 2:
                        break
                    self.sweep.search(q, k=8, in_stock=True)
                pool = [
                    c
                    for c in self.pool.values()
                    if hard_ok(self.brief, c) and purchase_ok(self.brief, c)
                ]
                if not pool:
                    return
        brand = soft_brand(self.brief)
        if brand:
            branded = [c for c in pool if brand_hit(c, brand)]
            if branded:
                pool = branded
        if self.brief.candidate_keys:
            cand = [c for c in pool if c.key in self.brief.candidate_keys]
            if cand:
                pool = cand
        if not pool:
            pool = [
                c
                for c in self.pool.values()
                if c.in_stock and GuardRail.device_ok(c, self.brief.category)
            ]
        if not pool:
            pool = [c for c in self.pool.values() if c.in_stock]
        if not pool:
            pool = list(self.pool.values())
        if not pool:
            if self.desk.remaining() > 1:
                self.sweep.search(
                    " ".join(self.brief.keywords[:4]) or self.brief.goal[:40],
                    k=5,
                    in_stock=True,
                )
                pool = [
                    c
                    for c in self.pool.values()
                    if c.in_stock and GuardRail.device_ok(c, self.brief.category)
                ] or [c for c in self.pool.values() if c.in_stock] or list(
                    self.pool.values()
                )
            if not pool:
                return
        if self.brief.family == "ranking":
            pool.sort(key=ranking_key, reverse=True)
        elif self.brief.family in EVENT_FAMILIES:
            pool.sort(key=lambda c: first_cart_score(self.brief, c), reverse=True)
        else:
            pool.sort(key=lambda c: score_card(self.brief, c), reverse=True)
        best = pool[0]
        if not best.viewed and self.desk.remaining() > 2:
            cards = self.view.view(best.product_id)
            ok = [
                c for c in cards if hard_ok(self.brief, c) and purchase_ok(self.brief, c)
            ]
            if ok:
                ok.sort(key=lambda c: score_card(self.brief, c), reverse=True)
                best = ok[0]
            else:
                best = self.pool.get(best.key, best)

        if self.desk.remaining() > 1:
            self.add_to_cart(best.product_id, best.sku)
            if self.desk.finished:
                return
            best = self.pool.get(best.key, best)
            if self.desk.remaining() > 0:
                best = self.view.inspect_stock(best.product_id, best.sku)
                self.handle_shopper_note(pending=best)

        if self.desk.finished:
            return

        if self.desk.remaining() > 3 and not purchase_ok(self.brief, best):
            self.remove_from_cart(best.product_id, best.sku)
            alt = self.recover(exclude_keys={best.key}, previous=best)
            if alt is not None:
                best = alt
            if self.desk.remaining() > 1:
                self.add_to_cart(best.product_id, best.sku)
                if self.desk.finished:
                    return
                if self.desk.remaining() > 0:
                    best = self.view.inspect_stock(best.product_id, best.sku)

        if self.desk.finished:
            return
        if self.desk.remaining() > 0:
            if not purchase_ok(self.brief, best):
                alts = [
                    c
                    for c in self.pool.values()
                    if c.key != best.key and purchase_ok(self.brief, c)
                ]
                if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
                    alts = [
                        c
                        for c in alts
                        if title_phrase_match(c, self.brief.title_phrase)
                    ]
                hard_alts = [c for c in alts if hard_ok(self.brief, c)]
                pick_from = hard_alts or alts
                if pick_from:
                    pick_from.sort(
                        key=lambda c: score_card(self.brief, c), reverse=True
                    )
                    best = pick_from[0]
                    if self.desk.remaining() > 1:
                        self.add_to_cart(best.product_id, best.sku)
            self.place_order(best, force=True)
            if self.desk.finished:
                return
            if self.desk.remaining() > 1:
                self.add_to_cart(best.product_id, best.sku)
            if self.desk.remaining() > 0:
                self.place_order(best, force=True)


# ---------------------------------------------------------------------------
# NewstartAgent — pipeline orchestrator
# ---------------------------------------------------------------------------


class NewstartAgent:
    """classify → QueryPipeline → Sweep → View → Pick (Weight) → Seal."""

    def __init__(self, desk: Desk):
        self.desk = desk
        self.brief = QueryPipeline.decompose(desk.query)
        self.pool: dict[str, SkuCard] = {}
        self.sweep = SweepEngine(desk, self.brief, self.pool)
        self.view = ViewBench(desk, self.brief, self.pool, self.sweep)
        self.pick = PickBench(desk, self.brief, self.pool)
        self.seal = SealBench(
            desk, self.brief, self.pool, self.sweep, self.view, self.pick
        )

    def _sync_brief(self) -> None:
        """Keep engines pointing at the latest BriefSheet after merges."""
        self.sweep.brief = self.brief
        self.view.brief = self.brief
        self.pick.brief = self.brief
        self.seal.brief = self.brief

    def run(self) -> list[dict[str, Any]]:
        try:
            self.desk.think(
                "Initial query analysis completed.",
                think=(
                    f"family={self.brief.family}; "
                    f"attrs={json.dumps(self.brief.attributes)}; "
                    f"soft={self.brief.soft_prefs[:4]}"
                ),
            )
            if self.brief.needs_clarification and self.desk.remaining() > 10:
                self.sweep.clarify_once()
                self.brief = self.sweep.brief
                self._sync_brief()
            if self.desk.finished:
                return self.desk.dialogue

            candidates = self.sweep.harvest()
            self.brief = self.sweep.brief
            self._sync_brief()
            if self.desk.finished:
                return self.desk.dialogue

            shortlist = self.view.survey(candidates)
            if self.desk.finished:
                return self.desk.dialogue

            choice = self.pick.choose(shortlist)
            if choice is None:
                fallback = [
                    c
                    for c in self.pool.values()
                    if c.in_stock and budget_ok(c, self.brief.budget)
                ]
                if self.brief.family == "preference_reasoning":
                    fallback = [c for c in fallback if hard_ok(self.brief, c)]
                if self.brief.candidate_keys:
                    fallback = [
                        c for c in fallback if c.key in self.brief.candidate_keys
                    ] or fallback
                if self.brief.family == "retrieval_recall" and self.brief.title_phrase:
                    matched = [
                        c
                        for c in fallback
                        if title_phrase_match(c, self.brief.title_phrase)
                    ]
                    fallback = matched  # empty OK — force_done may re-search phrase
                if self.brief.family == "preference_reasoning":
                    fallback.sort(
                        key=lambda c: preference_score(self.brief, c), reverse=True
                    )
                else:
                    fallback.sort(
                        key=lambda c: score_card(self.brief, c), reverse=True
                    )
                choice = fallback[0] if fallback else None
            if choice is not None and not self.desk.finished:
                self.seal.seal(choice)
                self.brief = self.seal.brief
                self._sync_brief()

            if not self.desk.finished and self.desk.remaining() > 0:
                self.seal.force_done()
        except Exception as exc:  # noqa: BLE001
            self.desk.dialogue.append(
                {
                    "role": "system",
                    "content": f"Newstart_v12 error: {type(exc).__name__}: {exc}",
                }
            )
            try:
                if not self.desk.finished and self.desk.remaining() > 0:
                    self.seal.force_done()
            except Exception:
                pass
        return self.desk.dialogue


# ---------------------------------------------------------------------------
# LGTEnhancekit (embedded; not called by agent_main / NewstartAgent.run)
# Source: LGTEnhancekit.py — foundation + scoring helpers only.
# ---------------------------------------------------------------------------

class LGT_ShoppingTaskType(str, Enum):
    PRODUCT = "product"
    SHOP = "shop"
    VOUCHER = "voucher"
    def explain(self, is_multiple: bool) -> str:
        match self:
            case LGT_ShoppingTaskType.PRODUCT:
                return "a single product with specific constraints"
            case LGT_ShoppingTaskType.SHOP:
                return "multiple products from the same shop"
            case LGT_ShoppingTaskType.VOUCHER:
                pstr = "multiple products" if is_multiple else "a single product"
                return f"{pstr} within a specified budget and considering voucher options"
class LGT_Vocab:
    ID_KEYS = ["brand", "color", "model", "id", "compatibility", "color_family"]
    GATE_HARD_KEYS = ID_KEYS + ["material", "packaging", "storage", "pack", "set", "size", "capacity", "volume", "no"]
    ALPHABETS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    KW_STOPWORDS: List[str] = [
        "color family", "suitable", "compatible",
        "about", "above", "after", "also", "and", "as", "at", "before", "below", "between", "by", "for", "from", "in",
        "less", "more", "of", "on", "or", "over", "per", "than", "to", "under", "up", "with", "within"
    ]
    YES_WORDS: List[str] = ["yes", "enabled", "included", "exists", "true"]
    NO_WORDS: List[str] = ["no", "without any", "without"]
    COLOR_WORDS: List[str] = [
        "red", "blue", "green", "yellow", "black", "white", "pink", "purple", "violet", "orange", "brown", "gray", "grey",
        "beige", "navy", "gold", "silver", "bronze", "magenta", "cyan"
    ]
    SEARCHABLE_UNITS: List[str] = [
        "mm", "cm", "m", "km", "ft", "yd", "mg", "g", "kg", "oz", "lb",
        "ml", "l", "mah", "ah", "wh", "kwh", "v", "mv", "kv", "ma", "w", "kw",
        "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb",
        "bps", "kbps", "mbps", "gbps", "°c", "°f", "pcs", "pc",
        "pa", "kpa", "mpa", "mp", "dpi", "lux", "db", "rpm"
    ]
    SYNONYMS: List[Tuple[str, str]] = [
        (min(a, b), max(a, b)) for a, b in
        [
            ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
            ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("ten", "10"),
            ("one", "single"), ("two", "double"), ("three", "triple"),
            ("1", "single"), ("2", "double"), ("3", "triple"),
            ("first", "1st"), ("second", "2nd"), ("third", "3rd"),
            ("hundred", "100"), ("thousand", "1000"), ("zero", "0"),
            ("shock-resistant", "shockproof"), ("shock resistant", "shockproof"),
            ("shock-proof", "shockproof"),
            ("water-resistant", "waterproof"), ("water resistant", "waterproof"),
            ("water-proof", "waterproof"),
        ]
    ]
    P_EXT_SEARCH_UNITS: List[Tuple[str, str]] = [
        ("piece", "pcs"), ("pack", "pcs")
    ]

@dataclass
class LGT_JsonModel(object):
    @classmethod
    def from_json(cls, data: dict) -> "LGT_JsonModel":
        kwargs = {}
        for f in fields(cls):
            if f.name not in data: continue
            value = data[f.name]
            t = f.type
            kwargs[f.name] = cls.__convert_from__(value, t)
        return cls(**kwargs)
    def to_json(self) -> Dict[str, Any] | List[Any]:
        return self.__convert_to__(self)
    @staticmethod
    def __convert_from__(value, t) -> Any:
        origin = get_origin(t)
        args = get_args(t)
        if origin is Union and len(args) == 2 and NoneType in args:
            inner_t = next(a for a in args if a is not NoneType)
            if value is None: return None
            return LGT_JsonModel.__convert_from__(value, inner_t)
        if is_dataclass(t) and isinstance(value, dict):
            return t.from_json(value)
        if origin is list and isinstance(value, list):
            return [LGT_JsonModel.__convert_from__(v, args[0]) for v in value]
        if origin is tuple and isinstance(value, list):
            return tuple(LGT_JsonModel.__convert_from__(v, args[0]) for v in value)
        if origin is dict and isinstance(value, dict):
            kt, vt = args
            return {
                LGT_JsonModel.__convert_from__(k, kt): LGT_JsonModel.__convert_from__(v, vt)
                for k, v in value.items()
            }
        if t is float and isinstance(value, str): return LGT_MiscUtil.extract_number(value)
        if t is int and isinstance(value, str): return LGT_MiscUtil.extract_number(value, True)
        if t is str: return str(value)
        return value
    @staticmethod
    def __convert_to__(obj) -> Any:
        if is_dataclass(obj):
            result = {}
            for f in fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = LGT_JsonModel.__convert_to__(value)
            return result
        if isinstance(obj, list):
            return [LGT_JsonModel.__convert_to__(x) for x in obj]
        if isinstance(obj, dict):
            return {LGT_JsonModel.__convert_to__(k): LGT_JsonModel.__convert_to__(v) for k, v in obj.items()}
        return obj

@dataclass
class LGT_SubQuery(LGT_JsonModel):
    query: str = ""
    services: List[str] = field(default_factory = list, compare = False)
    price: Optional[str] = None
    kw: List[List[str]] = field(default_factory = list, compare = False)
    def __hash__(self):
        return hash((self.query, ''.join(sorted(self.services)), self.price))
    def __eq__(self, other):
        if not isinstance(other, LGT_SubQuery): return NotImplemented
        if self.query != other.query: return False
        if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
        if self.price != other.price: return False
        return True

@dataclass
class LGT_VoucherOption(LGT_JsonModel):
    budget: float
    apply_to: str
    threshold: Optional[float]
    is_fixed: bool
    discount: float
    cap: Optional[float]

@dataclass
class LGT_QueryInfo(LGT_JsonModel):
    challenge: LGT_ShoppingTaskType = ""
    query_f: str = ""
    query_e: str = ""
    items: List[LGT_SubQuery] = field(default_factory = list)
    vopt: Optional[LGT_VoucherOption] = None

@dataclass
class LGT_SearchOption(LGT_SubQuery):
    shop_id: Optional[str] = None
    q: str = ""
    def update_shop(self, shop_id: Optional[str]) -> "LGT_SearchOption":
        self.shop_id = shop_id
        return self
    def update_price(self, price: Optional[str]) -> "LGT_SearchOption":
        self.price = price
        return self
    def update_kw(self, id: int, kw: List[str]) -> "LGT_SearchOption":
        self.kw[id] = kw
        return self
    def update_q(self, q: Optional[str] = None) -> "LGT_SearchOption":
        if not q:
            segs = []
            for kw in self.kw:
                v = kw[-1]
                for m in v.split():
                    m = m.lower().strip()
                    if m and m not in segs: segs.append(m)
            self.q = " ".join(segs)
        else:
            self.q = q
        return self
    def __hash__(self):
        return hash((self.q, self.query, ''.join(sorted(self.services)), self.price, self.shop_id))
    def __eq__(self, other):
        if not isinstance(other, LGT_SearchOption): return NotImplemented
        if self.query != other.query: return False
        if ''.join(sorted(self.services)) != ''.join(sorted(other.services)): return False
        if self.price != other.price: return False
        if self.shop_id != other.shop_id: return False
        if self.q != other.q: return False
        return True

@dataclass
class LGT_Product(LGT_JsonModel):
    id: str
    title: str
    price: float
    service: Optional[List[str]] = None
    shop_id: Optional[str] = None
    attributes: Dict[str, List[str]] = field(default_factory = dict)
    sku: Dict[str, Dict[str, str]] = field(default_factory = dict)
    is_viewed: bool = False
    def to_prompt(self, q: str, for_reason: bool = False) -> Dict[str, Any]:
        attr = self.extract_attr(q, ':', False)
        joint_attr = []
        for _, group in attr.items():
            for lines in group:
                for l in lines:
                    joint_attr.append(l)
        js = {
            "title": self.title,
            "attributes": '\n'.join(joint_attr)
        }
        if for_reason: js["price"] = self.price
        return js
    def clone_from(self, other: "LGT_Product") -> None:
        self.id = other.id
        self.title = other.title
        self.price = other.price
        self.service = other.service
        self.shop_id = other.shop_id
        self.attributes = other.attributes
        self.sku = other.sku
        self.is_viewed = other.is_viewed
    def extract_attr(self, q: str, delimiter: str, norm: bool) -> Dict[str, List[List[str]]]:
        extracted = {"sku": []}
        is_valid = False
        for k, vs in self.attributes.items():
            extracted[f"att_{k}"] = []
            atoms = []
            for v in vs:
                if LGT_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                atoms.append((k, v))
            if atoms: extracted[f"att_{k}"].append(atoms)
        for _, kv in self.sku.items():
            atoms = []
            for k, v in kv.items():
                if LGT_EngineUtil.is_valid_attr(k, v, q): is_valid = True
                atoms.append((k, v))
            if atoms: extracted["sku"].append(atoms)
        if not is_valid: extracted.clear()
        return {
            g: [
                [
                    f"{LGT_MiscUtil.normalize_str(k) if norm else k}{delimiter} "
                    f"{LGT_MiscUtil.normalize_str(v) if norm else v}"
                    for k, v in kv
                ]
                for kv in kvs
            ]
            for g, kvs in extracted.items()
        }
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if not isinstance(other, LGT_Product): return NotImplemented
        return self.id == other.id

@dataclass
class LGT_Weight(LGT_JsonModel):
    SKU_LIN_PROBS = [0.1, 0.1, 0.08, 0.05, 0.02, 0.01]
    RANK_BONUS_C = 0.05
    RANK_LIMIT = 300
    RANK_LOG_PROBS = [
        0.6554774978466211, 0.09305071398002347, 0.04004714272557973, 0.024146071349246602, 0.019434642793296042, 0.01943464179329533,
        0.018845714223802223, 0.005889285694938195, 0.005889284694937484, 0.005005892840697467, 0.005005891840696755, 0.005005890840696044,
        0.005005889840695332, 0.002572687961473001, 0.00257268696147229, 0.0025726859614715793, 0.002572684961470868, 0.0025726839614701575,
        0.0025726829614694463, 0.0025726819614687356, 0.0025726809614680245, 0.0025726799614673138, 0.0025726789614666026,
        0.002572677961465892, 0.002572676961465181, 0.00257267596146447, 0.0025726749614637594, 0.002572673961463048, 0.0025726729614623375,
        0.0025726719614616264, 0.0025726709614609156, 0.0025726699614602045, 0.002572668961459494, 0.0025726679614587827,
        0.002572666961458072, 0.0025726659614573612, 0.00257266496145665, 0.0025726639614559394, 0.0025726629614552282,
        0.0025726619614545175, 0.0025726609614538064, 0.0025726599614530957, 0.0025726589614523845, 0.002572657961451674,
        0.0025726569614509627, 0.002572655961450252, 0.0025726549614495412, 0.00257265396144883, 0.0025726529614481194,
        0.0025726519614474083, 0.0025726509614466975
    ]
    detail: Dict[str, float] = field(default_factory = dict)
    group: Dict[str, int] = field(default_factory = dict)
    weighted_rank: int = -1
    rank: int = -1
    rank_bonus: float = 0
    sku_bonus: float = 0
    confidence: float = 0
    score: float = 0
    score_n: float = 0
    def normalize(self):
        idx = self.group.get("sku") or -1
        self.sku_bonus = LGT_Weight.SKU_LIN_PROBS[idx + 1] if idx < (len(LGT_Weight.SKU_LIN_PROBS) - 1) else 0
        self.rank_bonus = max(1.0 - self.rank / LGT_Weight.RANK_LIMIT, 0) * LGT_Weight.RANK_BONUS_C
        self.score_n = self.score / len(self.detail) + self.rank_bonus + self.sku_bonus
    def update_confidence(self, weighted_rank: int):
        if self.weighted_rank >= 0: return
        self.weighted_rank = weighted_rank
        if weighted_rank >= len(LGT_Weight.RANK_LOG_PROBS):
            self.confidence = LGT_Weight.RANK_LOG_PROBS[-1] * (0.9 ** (weighted_rank - len(LGT_Weight.RANK_LOG_PROBS) + 1))
        else:
            self.confidence = LGT_Weight.RANK_LOG_PROBS[weighted_rank]
    def update_with_sibling(self, sibling: "LGT_Weight"):
        if self.score < sibling.score:
            self.detail = sibling.detail
            self.group = sibling.group
            self.score = sibling.score
        self.sku_bonus = max(self.sku_bonus, sibling.sku_bonus)
        self.score_n = self.score / len(self.detail) + self.rank_bonus + self.sku_bonus

def LGT_cast_model(source: LGT_JsonModel, target_cls: Type[LGT_JsonModel]) -> LGT_JsonModel:

    return target_cls.from_json(source.to_json())

class LGT_TextCodec:
    @staticmethod
    def pull_number(s: str, as_int: bool = False) -> int | float:
        matches = re.findall(r"\d+\.?\d*", s); val = float(matches[0]) if matches else 0.0; return int(val) if as_int and matches else (0 if not matches else val)
    @staticmethod
    def coerce_number(v: Any, as_int: bool = False) -> int | float:
        if v is None: return 0
        if isinstance(v, (int, float)): return v
        if isinstance(v, str): return LGT_TextCodec.pull_number(v, as_int=as_int)
        raise ValueError(f"# cannot convert {v} to number")
    @staticmethod
    def pull_bool(s: str) -> bool:
        s = s.lower().strip()
        if not s: return False
        if s == "true": return True
        if s == "false": return False
        return int(float(s)) != 0 if s.isnumeric() else False
    @staticmethod
    def tidy_text(x: str) -> str:
        x = x.lower(); x = re.sub(r"\b(\d+)([a-z]+)\b", r"\1 \2", x); x = re.sub(r"\b([a-z]+)(\d+)\b", r"\1 \2", x); x = re.sub(r"[\(\)\[\]\{\}\+\*\_\:\;\,\-\~\/\<\>\=]", " ", x); x = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1 \2", x); return re.sub(r"\s+", " ", x).strip()
    @staticmethod
    def cartesian_slots(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        keys = list(cat.keys()) if isinstance(cat, dict) else list(range(len(cat))); choices = [([-1] if include_non else []) + list(range(len(cat[k]))) for k in keys]; opts = product(*choices); return [dict(zip(keys, opt)) for opt in opts] if isinstance(cat, dict) else list(opts)
    @staticmethod
    def dump_js(js: Dict[str, Any] | List[Any]) -> None:
        print(json.dumps(js, ensure_ascii=False, indent=2))
    @staticmethod
    def decode_js(s: str) -> dict[str, Any] | list[Any] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I); cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try: return json.loads(cleaned)
        except json.JSONDecodeError: pass
        decoder = json.JSONDecoder(); starts = []
        obj_start = cleaned.find("{"); arr_start = cleaned.find("[")
        if obj_start >= 0: starts.append(obj_start)
        if arr_start >= 0: starts.append(arr_start)
        for start in sorted(starts):
            try:
                obj, _ = decoder.raw_decode(cleaned[start:]); return obj
            except json.JSONDecodeError: pass
        return None

class LGT_MiscUtil:
    @staticmethod
    def extract_number(s: str, as_int: bool = False) -> int | float:
        return LGT_TextCodec.pull_number(s, as_int)
    @staticmethod
    def ensure_number(v: Any, as_int: bool = False) -> int | float:
        return LGT_TextCodec.coerce_number(v, as_int)
    @staticmethod
    def extract_bool(s: str) -> bool:
        return LGT_TextCodec.pull_bool(s)
    @staticmethod
    def normalize_str(x: str) -> str:
        return LGT_TextCodec.tidy_text(x)
    @staticmethod
    def get_choice_combs(cat: Dict | List, include_non: bool) -> List[Dict | List]:
        return LGT_TextCodec.cartesian_slots(cat, include_non)
    @staticmethod
    def print_json(js: Dict[str, Any] | List[Any]) -> None:
        LGT_TextCodec.dump_js(js)
    @staticmethod
    def parse_json(s: str) -> dict[str, Any] | list[Any] | None:
        return LGT_TextCodec.decode_js(s)

class LGT_EngineUtil:
    @staticmethod
    def get_spell_variations(s: str) -> List[str]:
        vars = []
        if len(s) >= 3 and not re.search(r"\d", s):
            for suffix in ["es", "ic", "n", "d", "ed", "y", "ing", "ily", "ied", "er", "est", "ier", "ren"]:
                vars.append(s + suffix)
                if s.endswith(suffix):
                    body = s[:-len(suffix)]
                    vars.append(body + "y")
                    vars.append(body)
                    if len(body) >= 4 and body[-1] == body[-2]: vars.append(body[:-1])
            for suffix in ["ed", "es", "y", "er", "est", "ing"]:
                vars.append(s + s[-1] + suffix)
            if s.endswith("s"):
                vars.append(s[:-1] + "es")
                vars.append(s[:-1])
            else:
                vars.append(s + "s")
            if s.endswith("y"):
                for suffix in ["", "ied", "ies", "ier", "ing", "ily"]:
                    vars.append(s[:-1] + suffix)
            if len(s) <= 5 and s.endswith("en"): vars.append(s[:-2] + "an")
            if len(s) <= 5 and s.endswith("an"): vars.append(s[:-2] + "en")
            if len(s) >= 5 and s.endswith("or"): vars.append(s[:-2] + "our")
            if len(s) >= 6 and s.endswith("our"): vars.append(s[:-3] + "or")
        vars = list(set(v for v in vars if v != s and v not in LGT_Vocab.KW_STOPWORDS and len(v) >= 3))
        return vars
    @staticmethod
    def _literal_kw_hit(kw: str, s: str) -> bool:
        return bool(re.search((r"\b" if re.match(r"\w", kw[0]) else "") + re.escape(kw) + (r"\b" if re.match(r"\w", kw[-1]) else ""), s, re.I))
    @staticmethod
    def _concat_seg_hit(kw: str, s: str) -> bool:
        if len(kw) < 6 or len(s) <= len(kw) + 1: return False
        segs = s.replace('🎈', ' ').split()
        return any(kw == segs[i] + segs[i + 1] for i in range(len(segs) - 1)) if len(segs) > 1 else False
    @staticmethod
    def _affix_hit(kw: str, s: str) -> bool:
        if len(kw) < 5: return False
        return any(len(seg) > len(kw) and (seg.startswith(kw) or seg.endswith(kw)) for seg in s.replace('🎈', ' ').split())
    @staticmethod
    def _syn_pair_hit(kw: str, s: str, synonyms: List[Tuple[str, str]]) -> bool:
        for from_s, to_s in synonyms:
            if kw == from_s or kw == f"{from_s}s" or from_s == f"{kw}s":
                if re.search(rf"\b{re.escape(to_s)}s?\b", s): return True
            elif kw == to_s or kw == f"{to_s}s" or to_s == f"{kw}s":
                if re.search(rf"\b{re.escape(from_s)}s?\b", s): return True
        return False
    @staticmethod
    def is_kw_in(kw: str, s: str, as_is: bool, synonyms: List[Tuple[str, str]], terminal: bool = False) -> bool:
        if LGT_EngineUtil._literal_kw_hit(kw, s): return True
        if as_is: return False
        if any(re.search(rf"\b{re.escape(v)}\b", s, re.I) for v in LGT_EngineUtil.get_spell_variations(kw)): return True
        if LGT_EngineUtil._concat_seg_hit(kw, s): return True
        if not as_is and not terminal and LGT_EngineUtil._affix_hit(kw, s): return True
        if kw in ["pcs", "pc", "pack"] and any(w in s for w in ["piece", "pack", "set"]): return True
        if LGT_EngineUtil._syn_pair_hit(kw, s, synonyms): return True
        if not terminal:
            spell = re.sub(rf"ss", "sss", kw, flags=re.I)
            if spell != kw and LGT_EngineUtil.is_kw_in(spell, s, False, synonyms, True): return True
        return False
    @staticmethod
    def is_valid_attr(k: str, v: str, q: str) -> bool:
        pairs: List[Tuple[str, str, str, str, bool]] = [
            (v, q, r'\b(\d+[a-z]*)\s*\*\s*(\d+[a-z]*)\b', r'{m1} by {m2}', True), (v, q, r'\b(\d+)\-(\d+)([a-z]+)\b', r'{m1}-{m2} {m3}', True),
            (v, q, r'\b(\d+)([a-z])\+', r'\b{m1}\s*{m2}([a-z]*)\sand\sup\b', True), (v, q, r'\b(\d+)\-(\d+)[a-z]+\b', r'\b{m1}\-{m2}\b', True),
            (v, q, r'\b(\d+)\-(\d+)y\b', r'\b{m1}\sto\s{m2}\syears\b', True), (q, v, r'\b(\d+)\s*\-\&\-\s*(\d+)\b', r'{m1}\sto\s{m2}', False),
            (v, q, r'\b(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'{m3}\sbetween\s{m1}\sand\s{m2}', True), (v, q, r'\b(\d+)\.0\b', r'\b{m1}\b', True),
            (v, q, r'\b([\d\.]+)[a-z]*\s*~\s*([\d\.]+)\b', r'\b{m1}\sto\s{m2}\b', True), (v, q, r'【(\d+\-\d+[a-z]+)】', r'\b{m1}\b', True),
            (v, q, r'\b(\d+)\s*-\s*(\d+)([a-z]*)\b', r'up to {m2}{m3}', True), (v, q, r'\b(\w+)\s(\w+)\b', r'\b{m1}\s(\w+)\s{m2}\b', True),
            (q, v, r'\b([a-z]+)\s+\((\d)\s+piece\)\s+', r'^{m1}\s+{m2}pcs$', False), (v, q, r'^(?:[a-z]+)\:\s*(\d+)$', r'\b{m1}\b', True),
            (v, q, r'\b(?:[a-z]+)\:\s*(\d+)\s*-\s*(\d+)\s*([a-z]+)\b', r'up\sto\s{m2}{m3}', True),
            (v, q, r'\b(\d+)\s*-\s*(\d+)\b', r'{m1} and {m2}', True)
        ]
        for src, dst, src_pattern, dst_pattern, yes in sorted(pairs, key = lambda p: p[-1]):
            if (m := re.search(src_pattern, src, re.I)):
                exists = re.search(
                    dst_pattern.format(
                        m1 = re.escape(m.group(1)) if m.lastindex and m.lastindex >= 1 else "",
                        m2 = re.escape(m.group(2)) if m.lastindex and m.lastindex >= 2 else "",
                        m3 = re.escape(m.group(3)) if m.lastindex and m.lastindex >= 3 else "",
                        k = k
                    ),
                    dst,
                    re.I
                )
                if not yes and not exists: return False
        return True
    @staticmethod
    def compact_kw(nested: str, q: str) -> Optional[List[str]]:
        splits = [s for s in [s.strip() for s in nested.lower().split("→")] if s]
        if len(splits) < 2: return None
        ks, v = splits[:-1], splits[-1]
        if v.lower() in LGT_Vocab.YES_WORDS:
            for i in range(len(ks)):
                t = " ".join([k.replace("_", " ") for k in ks[i:]])
                if t in q:
                    ks, v = [v], t
                    break
            else:
                return None
        for yes in LGT_Vocab.YES_WORDS:
            if re.search(rf"\b{re.escape(yes)}$", v, re.I):
                v = re.sub(rf"\b{re.escape(yes)}$", "", v, flags = re.I)
        for nw in LGT_Vocab.NO_WORDS:
            if (m := re.search(rf"^{re.escape(nw)}\s+([\w\s]+)$", v, re.I)):
                ks.append("no")
                v = m.group(1)
                break
        if f"{ks[-1]} {v}" in LGT_Vocab.KW_STOPWORDS: return None
        if f"{v} {ks[-1]}" in LGT_Vocab.KW_STOPWORDS: return None
        for sw in LGT_Vocab.KW_STOPWORDS:
            v = re.sub(rf"\b{re.escape(sw)}\b", "", v, flags = re.I)
        if len(ks) == 1 and '_' in ks[0]:
            prefix = ks[0].split('_')[0] + '-'
            if v.startswith(prefix): v = v[len(prefix):]
            prefix = prefix[:-1]
            if re.search(rf"^{re.escape(prefix)}[a-z]+$", v, re.I): v = v[len(prefix):]
        if (m := re.search(r"^(any|all)\s+([\w\s]+)$", v, re.I)):
            k = m.group(2).replace(' ', '_')
            v = m.group(1)
            if k not in ks: ks.append(k)
        if v.isnumeric() and ks[-1] in ["eu", "us", "int"]:
            v, ks = f"{ks[-1]} {v}", ["size"]
        v = re.sub(r"\bge[a-z]([a-z]{2}one)\b", r"\1", v, flags = re.I)
        v = re.sub(r"\b(\d+)in(\d+)\b", r"\1-in-\2", v, flags = re.I)
        v = re.sub(r"[\&\|]", " ", v)
        v = re.sub(r"\s+", " ", v).strip()
        return (ks + [v]) if v else None
    @staticmethod
    def is_prod_only(q: str) -> bool:
        prod_str = type(LGT_Product(id = "", title = "", price = 0)).__name__.lower()
        prod_suffix = "only"
        return any(
            re.search(pat, q, re.I) is not None for pat in [
                rf"\b(?:just|{prod_suffix})(?:\s+the)?\s+{prod_str}s?\b",
                rf"\b{prod_str}s?\s+{prod_suffix}\b"
            ]
        )
    @staticmethod
    def get_vouchered_cost(products: List[LGT_Product], vo: Optional[LGT_VoucherOption]) -> float:
        psum = sum([p.price for p in products])
        if vo is None: return psum
        if vo.apply_to == "shop":
            shops = set([p.shop_id for p in products])
            if len(shops) > 1: return psum
        if vo.threshold is not None:
            if psum < vo.threshold: return psum
        if vo.is_fixed: return psum - vo.discount
        discount = psum * vo.discount / 100.0
        if vo.cap is not None: discount = min(discount, vo.cap)
        return psum - discount
    @staticmethod
    def get_favor_range(vo: LGT_VoucherOption) -> Tuple[float, float]:
        high = (vo.threshold / 0.1) if vo.threshold is not None else math.inf
        low = (vo.threshold / 0.9) if vo.threshold is not None else 0.0
        if not vo.is_fixed:
            pct = 1.0 - vo.discount / 100.0
            l1 = vo.budget / 1.1 / pct if pct > 0 else math.inf
            l2 = vo.budget / 1.1 + (vo.cap if vo.cap is not None else math.inf)
            h1 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
            h2 = vo.budget / pct if pct > 0 else math.inf
            low = max(low, min(l1, l2))
            high = min([high, h1, h2])
        else:
            low = max(low, vo.budget / 1.1 + vo.discount)
            high = min(high, vo.budget + vo.discount)
        return (math.floor(low) - 1, math.ceil(high) + 1)
    @staticmethod
    def is_favor_cost(vo: LGT_VoucherOption, psum: float) -> bool:


        if psum < 100.0: return False
        if vo.threshold is not None:
            if vo.threshold < int(psum * 0.1) or vo.threshold > int(psum * 0.9): return False
        if not vo.is_fixed:
            ac = max(
                psum * (1.0 - vo.discount / 100.0),
                psum - (vo.cap if vo.cap is not None else math.inf)
            )
            if vo.budget < math.floor(ac) or vo.budget > math.ceil(1.1 * ac): return False
        return True
    @staticmethod
    def get_allowed_total(vo: LGT_VoucherOption) -> float:
        if vo.is_fixed:
            return vo.budget + vo.discount
        upper_1 = vo.budget / (1.0 - vo.discount / 100.0) if vo.discount < 100 else math.inf
        upper_2 = vo.budget + (vo.cap if vo.cap is not None else math.inf)
        return min(upper_1, upper_2)
    @staticmethod
    def get_marginal_band(
        sidx: int,
        n_specs: int,
        threshold: float,
        allowed_total: float,
        minima: List[float],
        maxima: List[float],
        price_range: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        other = [i for i in range(n_specs) if i != sidx]
        floor_raw = max(0.0, threshold - sum(maxima[i] for i in other))
        ceiling_raw = allowed_total - sum(minima[i] for i in other)
        lo, hi = floor_raw, ceiling_raw
        if price_range:
            pr_lo, pr_hi = LGT_EngineUtil.numerify_price(price_range)
            lo = max(lo, pr_lo)
            hi = min(hi, pr_hi)
        if lo > hi: return None
        return (lo, hi)
    @staticmethod
    def voucher_within_budget(products: List[LGT_Product], vo: LGT_VoucherOption) -> bool:
        return LGT_EngineUtil.get_vouchered_cost(products, vo) <= vo.budget
    @staticmethod
    def promptfy_price(price: Optional[str]) -> str:
        if not price: return ""
        segs = [s.strip() for s in price.split('-')]
        if len(segs) != 2: return ""
        try:
            low = int(segs[0]) if segs[0] else None
            high = int(segs[1]) if segs[1] else None
            if low and high: return f"price between {low} and {high}"
            if low: return f"price greater than {low}"
            return f"price less than {high}"
        except Exception:
            return ""
    @staticmethod
    def numerify_price(price: Optional[str]) -> Tuple[int, int]:
        low_str, high_str = price.split('-', 1)
        low, high = 0, math.inf
        if low_str: low = LGT_MiscUtil.extract_number(low_str, True)
        if high_str: high = LGT_MiscUtil.extract_number(high_str, True)
        return low, high
    @staticmethod
    def stringfy_price(low: int, high: int) -> str:
        return (str(low) if low > 0 else "") + "-" + (str(high) if high < math.inf else "")

class LGT_ProxyUtil:
    DEADLINE = 0.0
    LLM_PRIORITIES = [
        "moonshotai/Kimi-K2.6-TEE", "moonshotai/Kimi-K2.5-TEE",
        "zai-org/GLM-5.1-TEE", "zai-org/GLM-5-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "MiniMaxAI/MiniMax-M2.5-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE", "deepseek-ai/DeepSeek-V3.1-TEE", "deepseek-ai/DeepSeek-V3-0324-TEE", "deepseek-ai/DeepSeek-R1-0528-TEE",
        "google/gemma-4-31B-turbo-TEE", "Qwen/Qwen3-32B-TEE", "XiaomiMiMo/MiMo-V2-Flash-TEE"
    ]
    OPENROUTER_LLM_PRIORITIES = [
        "google/gemma-4-31b-it", "deepseek/deepseek-v3.2", "deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-oss-120b", "moonshotai/kimi-k2.5", "qwen/qwen3-32b"
    ]
    DECOMPOSE_LLM_PRIORITIES: List[str] = []
    OPENROUTER_DECOMPOSE_LLM_PRIORITIES: List[str] = ["deepseek/deepseek-v3.2", "google/gemma-4-31b-it"]
    SELECT_LLM_PRIORITIES: List[str] = []
    OPENROUTER_SELECT_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "deepseek/deepseek-v3.2"]
    CLASSIFY_LLM_PRIORITIES: List[str] = []
    OPENROUTER_CLASSIFY_LLM_PRIORITIES: List[str] = ["google/gemma-4-31b-it", "deepseek/deepseek-v3.2"]
    @staticmethod
    def active_inference_provider() -> str:
        raw = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
        return raw if raw in ("chutes", "openrouter") else "chutes"
    @staticmethod
    def default_llm_priorities() -> List[str]:
        if LGT_ProxyUtil.active_inference_provider() == "openrouter":
            return LGT_ProxyUtil.OPENROUTER_LLM_PRIORITIES
        return LGT_ProxyUtil.LLM_PRIORITIES
    @staticmethod
    def get_llm_priorities(prefers: List[str]) -> List[str]:
        priorities = prefers.copy()
        for llm in LGT_ProxyUtil.default_llm_priorities():
            if llm not in priorities:
                priorities.append(llm)
        return priorities
    @staticmethod
    def get_role_llm_priorities(role: str) -> List[str]:


        openrouter = LGT_ProxyUtil.active_inference_provider() == "openrouter"
        if role == "decompose":
            preferred = LGT_ProxyUtil.OPENROUTER_DECOMPOSE_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.DECOMPOSE_LLM_PRIORITIES
        elif role == "select":
            preferred = LGT_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.SELECT_LLM_PRIORITIES
        elif role == "filter":
            preferred = LGT_ProxyUtil.OPENROUTER_SELECT_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.SELECT_LLM_PRIORITIES
        elif role == "classify":
            preferred = LGT_ProxyUtil.OPENROUTER_CLASSIFY_LLM_PRIORITIES if openrouter else LGT_ProxyUtil.CLASSIFY_LLM_PRIORITIES
        else:
            preferred = []
        return LGT_ProxyUtil.get_llm_priorities(preferred)
    LLM_PROXY = ProxyClient(timeout = 180, max_retries = 5)
    @staticmethod
    def arm_deadline() -> None:
        LGT_ProxyUtil.DEADLINE = time.monotonic() + LGT_SearchEngine.TIMEOUT_SECS
    @staticmethod
    def time_left() -> float:
        if not LGT_ProxyUtil.DEADLINE:
            return math.inf
        return LGT_ProxyUtil.DEADLINE - time.monotonic()
    @staticmethod
    def out_of_time() -> bool:
        return bool(LGT_ProxyUtil.DEADLINE) and time.monotonic() > LGT_ProxyUtil.DEADLINE
    @staticmethod
    def __empty_for__(expect: str) -> Tuple[Dict[str, Any] | List[Any], Any]:
        if expect == "list":
            return [], None
        return {}, None
    @staticmethod
    def _llm_payload(model: str, prompt: str, thinking: bool, t: float) -> Dict[str, Any]:
        return {"model": model, "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": thinking, "thinking": thinking}, "temperature": t, "stream": False, "max_completion_tokens": 8192, "max_tokens": 8192}
    @staticmethod
    def _llm_content(resp: Dict) -> str:
        choice: Dict[str, Dict] = resp.get("choices")[0]; content = choice.get("message").get("content")
        if not content: raise Exception("invalid response")
        return content
    @staticmethod
    def _llm_parse_expect(content: str, expect: str, t: float) -> Tuple[Any, float]:
        js = LGT_MiscUtil.parse_json(content)
        if js is None: raise Exception(f"failed to parse json\n*******\n{content}\n******")
        if (expect == "dict" and not isinstance(js, dict)) or (expect == "list" and not isinstance(js, list)): raise Exception(f"expected {expect} but got {type(js)}")
        return js, t
    @staticmethod
    def trigger_llm(prompt: str, thinking: bool = False, preferred_llms: List[str] = [], temperature: float = 0, expect: str = "any") -> Tuple[Dict[str, Any] | List[Any], Any]:
        llm_priorities = LGT_ProxyUtil.get_llm_priorities(preferred_llms); trigger_tick = time.monotonic(); t = temperature
        for _pass in range(2):
            for model in llm_priorities:
                if LGT_ProxyUtil.out_of_time():
                    print("# trigger_llm: global deadline reached, returning safe empty value"); return LGT_ProxyUtil.__empty_for__(expect)
                try:
                    tick = time.monotonic(); resp = LGT_ProxyUtil.LLM_PROXY.post("/inference/chat/completions", json_data=LGT_ProxyUtil._llm_payload(model, prompt, thinking, t))
                    try: content = LGT_ProxyUtil._llm_content(resp)
                    except Exception: time.sleep(0.5); raise Exception("invalid response")
                    try: js, t = LGT_ProxyUtil._llm_parse_expect(content, expect, t)
                    except Exception as exc:
                        t += 0.3; raise Exception(str(exc))
                    throughput = resp.get("usage", {}).get("completion_tokens", 0); delay = time.monotonic() - tick; trigger_delay = time.monotonic() - trigger_tick
                    print(f"* llm triggered: model = '{model}' | throughput = {throughput} | delay = {delay:.2f}s | total_delay = {trigger_delay:.2f}s"); return js, resp
                except Exception as exc:
                    print(f"# llm trigger error with '{model}': {exc}"); continue
        print("# trigger_llm: exhausted all models, returning safe empty value"); return LGT_ProxyUtil.__empty_for__(expect)
    ORO_PROXY = ProxyClient(timeout = 30, max_retries = 5)
    @staticmethod
    def _empty_find_res(so: LGT_SearchOption, page: int) -> ToolCallResult:
        return {"name": "find_product", "parameters": {"q": so.q, "page": page}, "tool_call_id": "", "result": []}
    @staticmethod
    def _hydrate_find_hits(js_prods: List[Dict]) -> List[LGT_Product]:
        products = []
        for j in js_prods:
            price = LGT_MiscUtil.extract_number(str(j.get("price") or 0))
            products.append(LGT_Product(id=str(j.get("product_id")), title=j.get("title", ""), price=price, shop_id=j.get("shop_id", ""), service=j.get("service", []) or [], attributes={}, sku={}, is_viewed=False))
        return products
    @staticmethod
    def find_products(so: LGT_SearchOption, sort: Optional[str] = None, page: int = 1) -> Tuple[List[LGT_Product], ToolCallResult]:
        empty_res = LGT_ProxyUtil._empty_find_res(so, page)
        for _attempt in range(2):
            if LGT_ProxyUtil.out_of_time():
                print("# find_products: global deadline reached, returning empty results"); return [], empty_res
            try:
                tool_res = execute_tool_call("find_product", {"q": so.q, "page": page, "shop_id": so.shop_id, "price": so.price, "sort": sort, "service": ','.join(so.services or [])})
                return LGT_ProxyUtil._hydrate_find_hits(tool_res["result"]), tool_res
            except Exception as exc:
                print(f"# error in `find_products`: {exc}"); time.sleep(0.5)
        print("# find_products: attempts exhausted, returning empty results"); return [], empty_res
    @staticmethod
    def view_products(products: List[LGT_Product]) -> ToolCallResult:
        if not products: return
        pid_str = ','.join([p.id for p in products])
        while True:
            try:
                tool_res = execute_tool_call("view_product_information", {"product_ids": pid_str})
                details = tool_res["result"]
                if not details: return tool_res
                details_map: Dict[str, Dict[str, Any]] = {d["product_id"]: d for d in details}
                for prod in products:
                    d = details_map.get(prod.id, {})
                    prod.sku = d.get("sku_options", {})
                    prod.attributes = d.get("attributes", {})
                    prod.is_viewed = True
                return tool_res
            except Exception as exc:
                print(f"# error in `view_products`: {exc}")
                time.sleep(0.5)
    @staticmethod
    def submit(products: List[LGT_Product]) -> ToolCallResult:
        pid_str = ','.join([p.id for p in products])
        return execute_tool_call("recommend_product", {"product_ids": pid_str})
    @staticmethod
    def terminate(status: str = "success") -> ToolCallResult:
        return execute_tool_call("terminate", {"status": status})
    @staticmethod
    def emit_calculate_voucher(
        sess: "LGT_SearchSession", products: List["LGT_Product"], vopt: "LGT_VoucherOption"
    ) -> Optional[ToolCallResult]:
        if not products or vopt is None: return None
        prices_str = ",".join(str(p.price) for p in products)
        vtype = "fixed" if vopt.is_fixed else "percentage"
        threshold = float(vopt.threshold or 0)
        cap = float(vopt.cap or 0)
        tool_res = execute_tool_call(
            "calculate_voucher",
            {
                "product_prices": prices_str,
                "voucher_type": vtype,
                "discount_value": vopt.discount,
                "threshold": threshold,
                "budget": vopt.budget,
                "cap": cap,
            }
        )
        sess.add_tool_result(tool_res)
        result = tool_res.get("result") or {}
        think = (
            f"Confirming the cart with calculate_voucher: prices=[{prices_str}], "
            f"threshold={threshold:.0f}, budget={vopt.budget:.0f}"
        )
        if cap > 0:
            think += f", cap={cap:.0f}"
        think += (
            f". Result: total_before={result.get('total_before')}, "
            f"discount={result.get('discount_amount')}, "
            f"total_after={result.get('total_after')}, "
            f"within_budget={result.get('within_budget')}."
        )
        sess.step(think, "Voucher calculation confirmed.")
        return tool_res

class LGT_SegTree(object):


    def __init__(self, values, k: int = 30) -> None:
        self.k, n = k, len(values)
        self.size = 1
        while self.size < n:
            self.size <<= 1
        self.seg = [[] for _ in range(2 * self.size)]
        for i, v in enumerate(values):
            self.seg[self.size + i] = [(v, i)]
        for i in range(self.size - 1, 0, -1):
            self.seg[i] = self.__merge__(self.seg[2 * i], self.seg[2 * i + 1])
    def __merge__(self, a: List, b: List) -> List:
        out, ia, ib = [], 0, 0
        while len(out) < self.k:
            if ia == len(a):
                out.extend(b[ib:ib + self.k - len(out)])
                break
            if ib == len(b):
                out.extend(a[ia:ia + self.k - len(out)])
                break
            if a[ia][0] >= b[ib][0]:
                out.append(a[ia])
                ia += 1
            else:
                out.append(b[ib])
                ib += 1
        return out
    def query(self, l: int, r: int) -> List:
        if l > r: return []
        left, right = [], []
        l += self.size
        r += self.size
        while l <= r:
            if l & 1:
                left = self.__merge__(left, self.seg[l])
                l += 1
            if not (r & 1):
                right = self.__merge__(self.seg[r], right)
                r -= 1
            l >>= 1
            r >>= 1
        return self.__merge__(left, right)

class LGT_Debugger(object):
    def __init__(
        self,
        suite: Optional[str] = None,
        frozen: Optional[List[str]] = None,
        stop_after: Optional[str] = None,
        show_dlg: bool = False,
        quiet: bool = False
    ) -> None:
        self.memory: Dict[str, List] = defaultdict(lambda: [])
        self.quiet = quiet
        self.frozen = frozen or []
        self.stop_after = stop_after
        self.suite = suite
        self.show_dlg = show_dlg
    def add(self, stage: str, data: Any) -> None:
        if self.suite: self.memory[stage].append(data)
    def find(self, stage: str, kv: Dict) -> Any:
        for js in self.memory[stage]:
            for key, val in kv.items():
                if js[key] != val: break
            else:
                return js
        raise Exception(
            f"# cannot find the freezed {stage} instance for `{json.dumps(kv, ensure_ascii = False)}`"
        )
    def is_stop_in(self, episode: str) -> bool:
        if self.stop_after:
            if episode == "AGENT" and self.stop_after.startswith("a-"): return True
            if episode == "BASE" and self.stop_after.startswith("b-"): return True
            if episode == "PRODUCT" and self.stop_after.startswith("p-"): return True
            if episode == "SHOP" and self.stop_after.startswith("s-"): return True
            if episode == "VOUCHER" and self.stop_after.startswith("v-"): return True
        return False
    def is_stop(self, stage: str) -> bool:
        return self.stop_after == stage
    def is_frozen(self, stage: str) -> bool:
        return stage in self.frozen
    def dump(self) -> None:
        pass
    def print(self, obj: Any) -> None:
        if not self.quiet:
            if isinstance(obj, dict) or isinstance(obj, list):
                LGT_MiscUtil.print_json(obj)
            else:
                print(obj)

class LGT_FitScoreKit:
    @staticmethod
    def syn_expand(v_o: str, synonyms: List[Tuple[str, str]]) -> List[str]:
        out = [v_o]
        for a, b in synonyms:
            if a == v_o and b not in out: out.append(b)
            if b == v_o and a not in out: out.append(a)
        return out
    @staticmethod
    def attr_key_bonus(ja: str, ks: List[str], atoms: List[str], synonyms: List[Tuple[str, str]]) -> float:
        header = ja.split('🎈')[0]
        if header in ("variation", "color_family"): return 0.05
        for k in ks:
            kn = LGT_MiscUtil.normalize_str(k.replace('_', ' '))
            for ak in kn.split():
                ak = ak.strip()
                if not ak or ak in atoms: continue
                if LGT_EngineUtil.is_kw_in(ak, ja, False, synonyms):
                    return 0.1 + (0.05 if kn == header else 0.0)
        return 0.0
    @staticmethod
    def attr_line_ok(ja: str, query: str, synonyms: List[Tuple[str, str]]) -> bool:
        jv = LGT_MiscUtil.normalize_str(ja.split('🎈')[1]); qn = LGT_MiscUtil.normalize_str(query)
        for s in jv.split():
            if not s: continue
            if s.isnumeric() or s in LGT_Vocab.COLOR_WORDS:
                if f".{s}" in ja or f".{s}" in query: continue
                cands = {s}
                for a, b in synonyms:
                    if a == s: cands.add(b)
                    if b == s: cands.add(a)
                if not any(re.search(rf"\b{re.escape(k)}\b", qn, re.I) for k in cands): return False
        return True
    @staticmethod
    def dry_scan(sq: LGT_SubQuery, title: str, joint_attr: List[str], synonyms: List[Tuple[str, str]]):
        used_attr, used_headers = [], []; len_pool, ti_pool, ai_pool, num_pool, match_pool = {}, {}, {}, {}, {}
        for kw in sq.kw:
            v_cands, ks = LGT_FitScoreKit.syn_expand(kw[-1], synonyms), kw[:-1]
            for v in v_cands:
                atoms = [a.strip() for a in LGT_MiscUtil.normalize_str(v).split() if a.strip()]; hash_key = tuple(kw + [v]); len_pool[hash_key] = len(atoms); as_is = re.search(r"^(\d+)$", v)
                ti_pool[hash_key] = {i for i, a in enumerate(atoms) if LGT_EngineUtil.is_kw_in(a, title, as_is, synonyms)}; num_pool[hash_key] = [i for i, a in enumerate(atoms) if a.isnumeric()]
                for ja in joint_attr:
                    ai_pool[(hash_key, ja)] = [i for i, tok in enumerate(atoms) if LGT_EngineUtil.is_kw_in(tok, ja, as_is, synonyms)]; match_pool[(hash_key, ja)] = (False, 0)
                    if len(ai_pool[(hash_key, ja)]) == len(atoms) and any(LGT_EngineUtil.is_kw_in(a, ja.split('🎈')[1], as_is, synonyms) for a in atoms): match_pool[(hash_key, ja)] = (True, LGT_FitScoreKit.attr_key_bonus(ja, ks, atoms, synonyms))
                    if ja in used_attr: continue
                    h, b = ja.split('🎈')
                    if any(LGT_EngineUtil.is_kw_in(tok, b, as_is, synonyms) for tok in atoms): used_attr.append(ja)
                    if h in used_headers: continue
                    if any(LGT_EngineUtil.is_kw_in(tok, h, as_is, synonyms) for tok in atoms): used_headers.append(h)
        return used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool
    @staticmethod
    def prep_attr_combs(prod: LGT_Product, sq: LGT_SubQuery, title: str, synonyms: List[Tuple[str, str]]):
        all_attr = prod.extract_attr(sq.query, '🎈', True); joint_attr = [l for _, g in all_attr.items() for lines in g for l in lines]
        used_attr, used_headers, len_pool, ti_pool, ai_pool, num_pool, match_pool = LGT_FitScoreKit.dry_scan(sq, title, joint_attr, synonyms); attr = {}
        for idx, g in all_attr.items():
            new_g = []
            for lines in g:
                new_lines, headers_seen = [], []
                for l in lines:
                    if l in used_attr: new_lines.append(l)
                    else:
                        h = l.split('🎈')[0]
                        if h not in headers_seen and h in used_headers: headers_seen.append(h); new_lines.append(l)
                new_lines = [l for l in new_lines if LGT_FitScoreKit.attr_line_ok(l, sq.query, synonyms)]
                if new_lines: new_g.append(new_lines)
            if new_g: attr[idx] = new_g
        return attr, LGT_MiscUtil.get_choice_combs(attr, True), len_pool, ti_pool, ai_pool, num_pool, match_pool
    @staticmethod
    def score_kw_against_attrs(kw, v_cands, len_pool, ti_pool, ai_pool, num_pool, match_pool, joint_attr) -> float:
        max_score = -math.inf; ks = kw[:-1]
        for v in v_cands:
            hash_key = tuple(kw + [v]); atom_len = len_pool[hash_key]; attr_includes = set(); score = 0; title_includes = ti_pool[hash_key]; found_in_attr, bonuses = False, []
            for ja in joint_attr:
                attr_includes.update(ai_pool[(hash_key, ja)])
                if match_pool[(hash_key, ja)][0]: bonuses.append(match_pool[(hash_key, ja)][1]); found_in_attr = True
            if not found_in_attr:
                if len(title_includes) < atom_len:
                    frags = title_includes | attr_includes
                    if all(i in frags for i in num_pool[hash_key]): score += len(frags) / atom_len
                else: score += 1.0
            else:
                score += 1.0 if ks[-1] != "product_type" else 0.95; score += max(bonuses)
            if score > max_score: max_score = score
        return max_score
    @staticmethod
    def best_weight(prod: LGT_Product, sq: LGT_SubQuery, rank: int, synonyms: List[Tuple[str, str]]) -> LGT_Weight:
        title = LGT_MiscUtil.normalize_str(prod.title); attr, combs, len_pool, ti_pool, ai_pool, num_pool, match_pool = LGT_FitScoreKit.prep_attr_combs(prod, sq, title, synonyms); candidates: List[LGT_Weight] = []
        for group in combs:
            joint_attr: List[str] = [a for ak, idx in group.items() for a in (attr[ak][idx] if idx >= 0 else [])]; weight = LGT_Weight(group=group, rank=rank)
            for i_kw, kw in enumerate(sq.kw):
                max_score = LGT_FitScoreKit.score_kw_against_attrs(kw, LGT_FitScoreKit.syn_expand(kw[-1], synonyms), len_pool, ti_pool, ai_pool, num_pool, match_pool, joint_attr); wk = '_'.join(kw[:-1])
                if wk in weight.detail: wk += f"_{i_kw + 1}"
                weight.detail[wk] = max_score; weight.score += max_score
            weight.normalize(); candidates.append(weight)
        candidates.sort(key=lambda w: -w.score_n); return candidates[0]

class LGT_PatchConfig:
    RELAX_ON_ZERO_RECALL = True
    ALWAYS_SUBMIT = True
    HARD_CONSTRAINT_GATE = True
    MAX_GATE_SWAPS = 2
    GATE_MIN_TIME_LEFT = 45.0
    RELAX_SCORE_GAP = 0.6
    RESCUE_SCORE_FLOOR = 0.55
    PRODUCT_JUDGE = True
    PRODUCT_VSF = True
    VERIFY_PICK = True
    VOUCHER_COMBO_MATRIX = True
    VOUCHER_MATRIX_TOP = 8

def _lgt_product_matches_sq(prod: LGT_Product, sq: LGT_SubQuery) -> bool:
    if sq.price:
        if prod.price is None:
            return False
        low, high = LGT_EngineUtil.numerify_price(sq.price)
        if not (low <= prod.price <= high):
            return False
    for svc in sq.services or []:
        prod_svcs = {str(s).lower() for s in (prod.service or [])}
        if str(svc).lower() not in prod_svcs:
            return False
    return True

def _lgt_constraint_coverage(detail: Dict[str, float], min_score: float = 0.8) -> float:
    if not detail:
        return 0.0
    matched = sum(1 for v in detail.values() if v >= min_score)
    return matched / len(detail)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entrypoint: one EnvPack task session.

    Reads environment.binding and environment.policy_view (query, tools,
    max_steps, max_calls_per_turn). Stops when observation done=true.
    Local packs ≠ Backend qualifying suite; paid_reward is verifier-owned;
    completed != correct.
    """
    desk = Desk.from_problem(problem_data)
    agent = NewstartAgent(desk)
    return agent.run()


__all__ = ["agent_main", "NewstartAgent", "Weight", "Vocab", "Desk", "SchemaRouter"]
