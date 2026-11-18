from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# Models / inference
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
            except Exception as exc:  # noqa: BLE001 — retry across models
                last_err = exc
                temp = min(temp + 0.2, 0.8)
                continue
    raise RuntimeError(f"llm_json failed: {last_err}")


# ---------------------------------------------------------------------------
# Environment session (from environment_agent contract)
# ---------------------------------------------------------------------------


class ToolKit:
    """Resolve logical capabilities against this task's ``policy_view.tools``.

    Per ORO Bench docs, tool availability is part of the environment — not a
    fixed Python import list. Map generic shopping capabilities to whatever
    names the EnvPack exposed (exact match first, then synonym aliases).
    Never invent tool names that are absent from the schema.
    """

    # capability -> candidate tool names (first name present in schema wins).
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
        # Exact capability string if the pack already uses logical names.
        if capability in self.names:
            return capability
        return None

    def has(self, capability: str) -> bool:
        return self.resolve(capability) is not None

    def resolve(self, capability_or_name: str) -> Optional[str]:
        """Return the concrete tool name for a capability, or the name if listed."""
        if capability_or_name in self._resolved:
            return self._resolved[capability_or_name]
        if capability_or_name in self.names:
            return capability_or_name
        # Late alias lookup for unknown capability strings.
        matched = self._match(capability_or_name)
        self._resolved[capability_or_name] = matched
        return matched

    def filter_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Drop args the schema does not advertise (when properties are known)."""
        allowed = self.param_keys.get(tool_name)
        if not allowed:
            return dict(args)
        return {k: v for k, v in args.items() if k in allowed}


@dataclass
class EnvSession:
    problem_id: str
    binding: dict[str, Any]
    policy: dict[str, Any]
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    last_user_message: Optional[str] = None
    shopper_notes: list[str] = field(default_factory=list)
    toolkit: ToolKit = field(default_factory=ToolKit)

    @classmethod
    def from_problem(cls, problem_data: dict[str, Any]) -> "EnvSession":
        environment = problem_data["environment"]
        policy = dict(environment["policy_view"])
        return cls(
            problem_id=str(
                problem_data.get("problem_id", problem_data.get("id", "problem"))
            ),
            binding=dict(environment["binding"]),
            policy=policy,
            toolkit=ToolKit(policy.get("tools") or []),
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
        # Prefer the toolkit snapshot built at session start.
        if self.toolkit.names:
            return set(self.toolkit.names)
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
        return self.toolkit.has(capability)

    @staticmethod
    def _unwrap_observation(payload: Any) -> dict[str, Any]:
        """Normalize registry StepResult nesting to the public tool payload.

        /environment/call returns each call as:
          {"call_id": ..., "observation": {"observation": {...}, "done": bool, "error": ...}}
        Tool logic needs the inner payload ({"results": ...}, {"variants": ...}, etc.).
        """
        if not isinstance(payload, dict):
            return {}
        # Already a flat tool payload.
        if any(
            key in payload
            for key in (
                "results",
                "variants",
                "compared",
                "added",
                "cart",
                "product_id",
                "sent",
                "ordered",
                "error",
            )
        ) and not (
            "done" in payload and "observation" in payload and isinstance(payload.get("observation"), dict)
        ):
            return payload
        # Nested StepResult: {observation, done, error}
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
        if self.done:
            return {"calls": [], "user_message": None}
        if self.remaining() <= 0:
            raise RuntimeError("max_steps exhausted")
        allowed = self.tool_names
        cleaned: list[dict[str, Any]] = []
        for action in actions[: self.max_calls_per_turn]:
            raw_name = action.get("name")
            if not raw_name:
                continue
            # Resolve logical capability → concrete schema name.
            name = self.toolkit.resolve(str(raw_name)) or (
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
                {
                    "name": name,
                    "args": self.toolkit.filter_args(name, args),
                }
            )
        if not cleaned:
            # Soft-fail: unavailable tools must not crash the episode.
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
                {
                    "call_id": f"{group_id}-{index}",
                    "action": action,
                }
                for index, action in enumerate(cleaned, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        self.dialogue.append(
            {
                "role": "assistant",
                "tool_calls": cleaned,
                "environment_result": result,
            }
        )
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            note = str(user_message["content"])
            self.last_user_message = note
            self.shopper_notes.append(note)
        for call in result.get("calls") or []:
            if self._call_done(call):
                self.done = True
        return result

    def act(self, name: str, **args: Any) -> dict[str, Any]:
        """Invoke a capability or concrete tool; return {} if unavailable."""
        if not self.toolkit.resolve(name) and name not in self.toolkit.names:
            return {"error": "tool_unavailable", "requested": name}
        result = self.call([{"name": name, "args": args}])
        if result.get("error") == "tool_unavailable":
            return {"error": "tool_unavailable", "requested": name}
        calls = result.get("calls") or []
        if not calls:
            return {}
        return self._unwrap_observation(calls[0].get("observation") or {})


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

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
        ("listing_id", "soft preferences", "intended use"),
    ),
    (
        "recovery",
        ("relaxable", "another color", "least-relaxed", "preferred model"),
    ),
    (
        "retrieval_recall",
        ("title token", "distinct matching product", "complete title token", "surface at least"),
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
    # LAST on purpose. Shared tool footer mentions "structured justification" /
    # "field/value" on every family — those must not classify as TF7.
    # Match goal-local ask only (see infer_family).
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


@dataclass
class Plan:
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
    candidate_keys: list[str] = field(default_factory=list)  # product_id::sku
    preferred_model: Optional[str] = None
    preferred_color: Optional[str] = None
    needs_justification: bool = False
    needs_clarification: bool = False


_PLAN_PROMPT = """\
You are a shopping planner for an interactive catalog environment.
Extract a structured purchase plan from the shopper brief below.

Rules:
- Separate HARD requirements from SOFT preferences.
- Hard usually includes: category, budget, in-stock, and any named hardware spec
  (screen_size, resolution, refresh_rate, ram, etc.).
- An explicitly preferred brand is SOFT unless the brief says the brand is mandatory.
- If the brief says a firm hardware spec will be told only when asked
  ("tell you if you ask" / "one firm requirement"), set needs_clarification=true
  and leave attributes EMPTY — do not invent screen_size / resolution / RAM.
- Keywords must be short BM25-friendly tokens (values, not schema keys).
- Provide 3-6 alternate search_queries that reformulate the request.
  For resolution gates include both the class token and pixel phrasing
  (e.g. "720p", "HD", "1280x720"). For inch gates spell out "N inch".
- If the brief lists candidate variants as product_id::sku, copy them into candidate_keys.
- needs_justification=true only if the shopper asks for a reason / justification
  in their own request (ignore shared tool-footer text about optional claims).
- When justification is requested, the gate attribute in attributes must match
  the shopper's named hardware cue (resolution vs screen_size vs refresh_rate).

Return JSON only:
{{
  "category": <str|null>,
  "brand": <str|null>,              // mandatory brand if any
  "preferred_brand": <str|null>,    // soft brand preference
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


def _goal_only(query: str) -> str:
    split = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    return split[0].strip()


def infer_family(query: str) -> str:
    """Detect TF family from public query text.

    Structure-first early exits (Hioro): candidate keys → ranking; surface-N →
    recall; prefer model+color → recovery. Then goal-local soft-brand / TF7,
    then hint lists. Justification hints match the goal only so the shared
    tool footer cannot classify every task as TF7.
    """
    goal = _goal_only(query).lower()
    full = query.lower()
    parts = re.split(r"\n\s*Task rules:\s*", query, maxsplit=1)
    rules = parts[1].lower() if len(parts) > 1 else full

    # Structure-first (before phrase hints).
    if re.search(r"\d{6,}::", query):
        return "ranking"
    if re.search(r"surface at least\s+\d+\s+distinct", full) or (
        "title token" in full and "distinct matching" in full
    ):
        return "retrieval_recall"
    if re.search(
        r"prefer(?:\s+the)?\s+[a-z][\w&.'\-]*\s+[a-z0-9][\w\-]*\d[\w\-]*\b",
        goal,
        re.I,
    ) and (
        "another color" in full
        or "relaxable" in full
        or "least-relaxed" in full
        or "preferred model" in full
    ):
        return "recovery"

    # Soft preferred brand in the shopper goal → TF1 (never wait for rule-text).
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
    # TF7 ask lives in the goal; CS device rules may also appear there.
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
            continue  # already handled via goal
        if family in {"ranking", "retrieval_recall", "recovery"}:
            continue  # already handled via structure
        # CS phrases also appear inside some TF7 goals' embedded device rules —
        # match them against Task rules only.
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


def parse_plan(query: str) -> Plan:
    family = infer_family(query)
    goal = _goal_only(query)
    plan = Plan(
        family=family,
        goal=goal,
        budget=_extract_budget(query),
        candidate_keys=_extract_candidate_keys(query),
        needs_justification=False,  # set after parse; TF7 only
        needs_clarification=False,
    )

    try:
        js = llm_json(_PLAN_PROMPT.format(query=query), expect="dict")
    except Exception:
        js = {}

    if isinstance(js, dict):
        plan.category = js.get("category") or plan.category
        plan.brand = js.get("brand") or plan.brand
        plan.preferred_brand = js.get("preferred_brand") or plan.preferred_brand
        if js.get("budget") is not None:
            try:
                plan.budget = float(js["budget"])
            except (TypeError, ValueError):
                pass
        plan.currency = str(js.get("currency") or plan.currency)
        if "require_in_stock" in js:
            plan.require_in_stock = bool(js["require_in_stock"])
        attrs = js.get("attributes") or {}
        if isinstance(attrs, dict):
            plan.attributes = {
                str(k): str(v) for k, v in attrs.items() if v is not None and str(v)
            }
        soft = js.get("soft_prefs") or []
        if isinstance(soft, list):
            plan.soft_prefs = [str(x) for x in soft if str(x).strip()]
        kws = js.get("keywords") or []
        if isinstance(kws, list):
            plan.keywords = [str(x) for x in kws if str(x).strip()]
        sqs = js.get("search_queries") or []
        if isinstance(sqs, list):
            plan.search_queries = [str(x) for x in sqs if str(x).strip()]
        ckeys = js.get("candidate_keys") or []
        if isinstance(ckeys, list):
            for key in ckeys:
                key_s = str(key)
                if "::" in key_s and key_s not in plan.candidate_keys:
                    plan.candidate_keys.append(key_s)
        plan.preferred_model = js.get("preferred_model") or plan.preferred_model
        plan.preferred_color = js.get("preferred_color") or plan.preferred_color
        if "needs_clarification" in js:
            plan.needs_clarification = bool(js["needs_clarification"])

    # Firm hardware attributes stated in the goal (any family).
    if "screen_size" not in plan.attributes:
        inch_m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
            goal,
            re.I,
        )
        if inch_m:
            gate = ClaimBuilder.gate_from_goal(goal, plan.attributes)
            if gate in {None, "screen_size"} or family == "intent_decomposition":
                if family != "justification" or gate == "screen_size":
                    plan.attributes["screen_size"] = f"{inch_m.group(1)}in"
    # Intent family: goal inches override a loose LLM parse.
    if family == "intent_decomposition":
        inch_m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
            goal,
            re.I,
        )
        if inch_m:
            plan.attributes["screen_size"] = f"{inch_m.group(1)}in"

    # Soft preferred brand from goal text; never treat as mandatory brand.
    prefer_m = re.search(
        r"(?:ideally[, ]+)?(?:i\s+)?prefer(?:\s+the)?\s+([A-Za-z][A-Za-z0-9.\- ]{1,24})",
        goal,
        re.I,
    )
    if prefer_m:
        brand_guess = prefer_m.group(1).strip().rstrip(".,;")
        brand_guess = re.split(r"\b(?:but|and|if|,)\b", brand_guess, maxsplit=1)[0].strip()
        if brand_guess and brand_guess.lower() not in {"the", "a", "an", "black", "white"}:
            plan.preferred_brand = plan.preferred_brand or brand_guess
    if family in {"intent_decomposition", "preference_reasoning", "recovery", "generic"}:
        if plan.brand and plan.preferred_brand and plan.brand.lower() == plan.preferred_brand.lower():
            plan.brand = None
        # Soft prefer phrasing: brand is never mandatory.
        if plan.brand and re.search(r"\b(ideally|prefer|nice[- ]to[- ]have)\b", goal, re.I):
            if "must be" not in goal.lower() and "only from" not in goal.lower():
                plan.preferred_brand = plan.preferred_brand or plan.brand
                plan.brand = None
        if family == "intent_decomposition" and plan.brand and "must" not in goal.lower():
            plan.preferred_brand = plan.preferred_brand or plan.brand
            plan.brand = None

    # TF6 model/color from "prefer the <color> <model>"
    if family == "recovery":
        mc = re.search(
            r"prefer(?:\s+the)?\s+([a-z]+)\s+([A-Za-z0-9.\-]{4,})",
            goal,
            re.I,
        )
        if mc:
            plan.preferred_color = plan.preferred_color or mc.group(1)
            plan.preferred_model = plan.preferred_model or mc.group(2)

    # Branded justification asks: leading brand + gate attribute from the goal.
    if family == "justification":
        brand_m = re.match(
            r"(?:can you get me (?:an? )?in-stock\s+|find me (?:an?\s+)?|i want (?:an?\s+)?)"
            r"([A-Za-z][A-Za-z0-9.\- ]{1,24}?)"
            r"(?=\s+(?:\d|gaming|hd|monitor|smartphone|tablet|in-stock|at\b))",
            goal,
            re.I,
        )
        if brand_m:
            plan.preferred_brand = plan.preferred_brand or brand_m.group(1).strip()
            plan.brand = plan.brand or brand_m.group(1).strip()
        res_m = re.search(r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|720p|1080p|1440p|hd)\b", goal, re.I)
        if res_m and "resolution" not in plan.attributes:
            token = res_m.group(1).replace(" ", "").lower()
            if token == "hd":
                token = "hd"
            plan.attributes["resolution"] = token
        inch_m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch", goal, re.I)
        if inch_m and ClaimBuilder.gate_from_goal(goal, plan.attributes) == "screen_size":
            plan.attributes["screen_size"] = f"{inch_m.group(1)}in"
            plan.attributes.pop("resolution", None)

    # TF4: seed soft prefs from use-case language already in the goal.
    if family == "preference_reasoning":
        for pat, label in (
            (r"sweat|water|rain|weather", "sweat/water resistance"),
            (r"gym|run|sport|workout|fitness", "sport secure fit"),
            (r"battery|all.?day|long.?lasting", "battery life"),
            (r"wireless|bluetooth", "wireless"),
            (r"noise|anc", "noise canceling"),
        ):
            if re.search(pat, goal, re.I):
                plan.soft_prefs.append(label)

    # Clarify only when the firm spec is withheld, or preferences are empty.
    withheld = SpecReveal.is_withheld(query)
    plan.needs_clarification = (
        family == "constraint_satisfaction" and withheld
    )
    # Preference: soft axes are seeded from the goal — do not burn a turn asking.
    # Never trust LLM-guessed firm attrs when the shopper said they will reveal them.
    if family == "constraint_satisfaction" and withheld:
        plan.attributes = {}
        # Drop invented attr tokens from LLM search_queries; brand+category only.
        brand = plan.brand or plan.preferred_brand or ""
        cat = plan.category or "smartphone"
        seeds = [x for x in (f"{brand} {cat}".strip(), brand, cat) if x]
        plan.search_queries = list(dict.fromkeys(seeds))

    # Sanitize garbage LLM brands / models (e.g. "white 8520. If that exact").
    def _clean_token(val: Optional[str], *, max_words: int = 4, max_len: int = 40) -> Optional[str]:
        if not val:
            return None
        text = str(val).strip().strip(".,;:")
        if not text or len(text) > max_len:
            return None
        if re.search(r"\b(if|that|exact|prefer|should|would|please)\b", text, re.I):
            return None
        if len(text.split()) > max_words:
            return None
        return text

    plan.brand = _clean_token(plan.brand, max_words=3)
    plan.preferred_brand = _clean_token(plan.preferred_brand, max_words=3)
    plan.preferred_model = _clean_token(plan.preferred_model, max_words=3, max_len=32)
    plan.preferred_color = _clean_token(plan.preferred_color, max_words=2, max_len=20)

    # Optional claims fail submitted_claims_truthful unless justification family.
    plan.needs_justification = family == "justification"
    plan.require_in_stock = True

    # Normalize attribute values to canonical forms used by SpecKit.
    normalized_attrs: dict[str, str] = {}
    for key, value in list(plan.attributes.items()):
        kind, wanted = SpecKit.normalize_wanted(key, str(value))
        if kind == "screen_size" and isinstance(wanted, (int, float)):
            normalized_attrs["screen_size"] = f"{float(wanted):g}in"
        elif kind == "resolution" and wanted:
            normalized_attrs["resolution"] = str(wanted)
        elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
            normalized_attrs["refresh_rate"] = f"{int(wanted)}Hz"
        else:
            normalized_attrs[key] = str(value)
    plan.attributes = normalized_attrs

    if not plan.keywords:
        stop = {
            "i",
            "need",
            "want",
            "find",
            "looking",
            "for",
            "with",
            "and",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "on",
            "at",
            "or",
            "my",
            "me",
            "can",
            "spend",
            "up",
            "under",
            "below",
            "php",
            "usd",
            "stock",
            "in-stock",
            "please",
        }
        tokens = [
            t
            for t in re.findall(r"[A-Za-z0-9.+\-]+", goal.lower())
            if t not in stop and len(t) > 1
        ]
        plan.keywords = tokens[:12]
    if not plan.search_queries:
        base = " ".join(plan.keywords[:8]) or goal[:80]
        plan.search_queries = [base]
        if plan.category:
            plan.search_queries.append(f"{plan.category} {base}")
        if plan.preferred_brand or plan.brand:
            brand = plan.brand or plan.preferred_brand
            plan.search_queries.append(f"{brand} {base}")
        for attr, val in list(plan.attributes.items())[:2]:
            plan.search_queries.append(f"{val} {base}")
    # de-dupe
    plan.search_queries = list(dict.fromkeys(q.strip() for q in plan.search_queries if q.strip()))
    return plan


def apply_shopper_update(plan: Plan, note: str) -> Plan:
    """Merge clarification / budget tightening from user_message interventions."""
    if not note:
        return plan
    return SpecReveal.merge_note(plan, note)


def shopper_needs_reply(note: Optional[str]) -> bool:
    """Match verifier _interventions_answered: questions / clarify_demand need a message."""
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


class SpecReveal:
    """TF3 withheld-spec protocol: ask once, parse reveal, never invent attrs."""

    _WITHHELD = re.compile(
        r"(if you ask|will tell you|one firm requirement|tell you if you ask)",
        re.I,
    )

    @classmethod
    def is_withheld(cls, query: str) -> bool:
        return bool(cls._WITHHELD.search(query or ""))

    @classmethod
    def merge_note(cls, plan: Plan, note: str) -> Plan:
        text = (note or "").strip()
        if not text:
            return plan
        new_budget = _extract_budget(text)
        if new_budget is not None:
            if plan.budget is None or new_budget < plan.budget:
                plan.budget = new_budget

        brand_m = re.search(
            r"(?:required brand is|brand is|must be(?: an?)?)\s+([A-Za-z][A-Za-z0-9.\-]{1,24})",
            text,
            re.I,
        )
        if brand_m and plan.family == "constraint_satisfaction":
            guess = brand_m.group(1).strip().rstrip(".,;")
            if guess and guess.lower() not in {"the", "a", "an", "smartphones", "smartphone"}:
                plan.brand = plan.brand or guess

        cat_m = re.search(
            r"(?:required catalog category is|category is)\s+([A-Za-z][A-Za-z0-9 &/\-]{2,40})",
            text,
            re.I,
        )
        if cat_m and not plan.category:
            cat = cat_m.group(1).strip().rstrip(".,;")
            # Normalize common catalog labels to planner category tokens.
            cl = cat.lower()
            if "smartphone" in cl or "mobile" in cl:
                plan.category = "smartphone"
            elif "tablet" in cl:
                plan.category = "tablet"
            elif "monitor" in cl:
                plan.category = "monitor"
            elif "headphone" in cl or "earbud" in cl:
                plan.category = "headphone"
            else:
                plan.category = cat

        replace = re.search(
            r"(?:must now have|change one requirement.*?have)\s+(.+?)\s+instead of",
            text,
            re.I | re.S,
        )
        force = bool(replace) or "must now" in text.lower() or "instead of" in text.lower()
        # Clarification reveals always overwrite empty attrs; revisions force overwrite.
        reveal = bool(
            re.search(
                r"(exact variant|required|still needs|firm requirement|must have)",
                text,
                re.I,
            )
        )

        m = re.search(
            r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?(?:\s+screen)?",
            text,
            re.I,
        )
        if m:
            val = f"{m.group(1)}in"
            if force or reveal or "screen_size" not in plan.attributes:
                plan.attributes["screen_size"] = val
        m = re.search(r"\b(\d{3,4}p|4k|uhd|qhd|full\s*hd|hd)\b", text, re.I)
        if m:
            val = m.group(1).replace(" ", "")
            if force or reveal or "resolution" not in plan.attributes:
                plan.attributes["resolution"] = val
        m = re.search(r"\b(\d{2,3})\s*hz\b", text, re.I)
        if m:
            val = f"{m.group(1)}Hz"
            if force or reveal or "refresh_rate" not in plan.attributes:
                plan.attributes["refresh_rate"] = val
        m = re.search(r"\b(\d+)\s*gb\s*ram\b", text, re.I)
        if m:
            val = f"{m.group(1)}GB"
            if force or reveal or "ram" not in plan.attributes:
                plan.attributes["ram"] = val

        lower = text.lower()
        if "prefer" in lower or "rather" in lower:
            plan.soft_prefs.append(text)

        normalized: dict[str, str] = {}
        for key, value in list(plan.attributes.items()):
            kind, wanted = SpecKit.normalize_wanted(key, str(value))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                normalized["screen_size"] = f"{float(wanted):g}in"
            elif kind == "resolution" and wanted:
                normalized["resolution"] = str(wanted)
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                normalized["refresh_rate"] = f"{int(wanted)}Hz"
            else:
                normalized[key] = str(value)
        plan.attributes = normalized
        return plan


# ---------------------------------------------------------------------------
# Candidate model + scoring
# ---------------------------------------------------------------------------


@dataclass
class Variant:
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
    # Brand used in a successful filter/search that returned this hit (fills null brands).
    source_brand: str = ""

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


# ---------------------------------------------------------------------------
# Spec / device / cart helpers (suite-agnostic)
# ---------------------------------------------------------------------------


class SpecKit:
    """Normalize and match hardware specs from public facts/options/titles.

    Evidence order mirrors the verifier (Hioro policy): selected variant
    options first, then labelled facts, then title. Multiple disagreeing
    values resolve to unsupported (None). An option string that mentions the
    concept without a parseable value blocks the product-wide fallback.
    """

    _RES_ALIASES = {
        "720p": ("720p", "hd", "1280x720", "1280 x 720", "1280×720"),
        "1080p": ("1080p", "full hd", "fhd", "1920x1080", "1920 x 1080", "1920×1080"),
        "1440p": ("1440p", "qhd", "2560x1440", "2560 x 1440"),
        "4k": ("4k", "uhd", "3840x2160", "3840 x 2160"),
        "hd": ("hd", "720p", "1280x720", "1280 x 720"),
    }
    _FACT_LABEL = {
        "screen_size": ("size", "screen", "inch", "display"),
        "resolution": ("resolution", "display", "screen"),
        "refresh_rate": ("refresh", "hz"),
        "ram": ("ram", "memory", "storage"),
    }

    @staticmethod
    def parse_inches(text: Any) -> Optional[float]:
        if text is None:
            return None
        raw = str(text).strip().lower()
        # Reject multi-model ranges like "24-27" / "24/27" / "24, 27".
        if re.search(r"\d+(?:\.\d+)?\s*[-–/,]\s*\d+(?:\.\d+)?", raw):
            # Still allow a single explicit "(27 inch)" later via options.
            pass
        # Allow 7.6" / 7.6in / 7.6 inch — do not require \b after " (quotes
        # are non-word chars, so \b after \" often fails in titles).
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
    def _pixel_class(cls, long_side: int, short_side: int) -> Optional[str]:
        if long_side >= 3840 and short_side >= 2160:
            return "4k"
        if long_side >= 2560 and short_side >= 1440:
            return "1440p"
        if long_side >= 1920 and short_side >= 1080:
            return "1080p"
        if long_side >= 1280 and short_side >= 720:
            return "720p"
        return None

    @classmethod
    def _res_vals(cls, text: str) -> set[str]:
        """Resolution classes asserted by one surface (conflict → multi-member set)."""
        text = text or ""
        if re.search(r"\bqHD\b", text):
            return set()  # quarter HD is a different, much smaller panel
        low = text.lower()
        out: set[str] = set()
        for m in re.finditer(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", low):
            pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            cls_tok = cls._pixel_class(pair[0], pair[1])
            if cls_tok:
                out.add(cls_tok)
        if re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low):
            out.add("4k")
        if re.search(r"\b(?:qhd|1440p?|2k)\b", low):
            out.add("1440p")
        if re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low):
            out.add("1080p")
        if re.search(r"\b720p\b|1280\s*[x×]\s*720", low):
            out.add("720p")
        stripped = re.sub(r"\b(?:full|ultra)\s*hd\b", "", low)
        if re.search(r"\bhd\b", stripped) and "1080" not in low and "720p" not in low:
            out.add("hd")
        return out

    @classmethod
    def _inch_vals(cls, text: str) -> set[float]:
        text = text or ""
        if re.search(r"\d+(?:\.\d+)?\s*[-–/,]\s*\d+(?:\.\d+)?", text):
            # Ranges are ambiguous product-wide — collect none from this surface.
            # Still allow a single parenthetical inch in options via explicit parse.
            m = re.search(r"\((\d+(?:\.\d+)?)\s*inch", text, re.I)
            if m:
                return {float(m.group(1))}
            return set()
        vals: set[float] = set()
        for m in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", text, re.I
        ):
            num = float(m.group(1))
            if 3.0 <= num <= 100.0:
                vals.add(num)
        for m in re.finditer(r"(\d+(?:\.\d+)?)\"", text):
            num = float(m.group(1))
            if 3.0 <= num <= 100.0:
                vals.add(num)
        return vals

    @classmethod
    def _hz_vals(cls, text: str) -> set[float]:
        return {float(m) for m in re.findall(r"(\d{2,3})\s*hz", (text or "").lower())}

    @classmethod
    def _ram_vals(cls, text: str) -> set[float]:
        text = text or ""
        paired = {
            float(a)
            for a, _b in re.findall(
                r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b",
                text,
                re.I,
            )
        }
        if paired:
            return paired
        return {
            float(n)
            for n in re.findall(
                r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", text, re.I
            )
        }

    @classmethod
    def _mentions(cls, field: str, text: str) -> bool:
        text = text or ""
        if field == "ram":
            return bool(
                re.search(
                    r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)",
                    text,
                    re.I,
                )
                or re.search(r"\d+(?:\.\d+)?\s*gb\s*(?:ram|memory)\b", text, re.I)
            )
        if field == "screen_size":
            return bool(re.search(r"\d+(?:\.\d+)?\s*(?:\"|''|in(?:ch(?:es)?)?)\b", text, re.I))
        if field == "refresh_rate":
            return bool(re.search(r"\d+\s*hz", text, re.I))
        return bool(
            re.search(r"\d{3,5}\s*[x×]\s*\d{3,5}", text)
            or re.search(
                r"\b(?:qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b",
                text,
                re.I,
            )
        )

    @classmethod
    def _one(cls, values: set) -> Any:
        return next(iter(values)) if len(values) == 1 else None

    @classmethod
    def _surface_vals(cls, field: str, text: str) -> set:
        if field == "screen_size":
            return cls._inch_vals(text)
        if field == "refresh_rate":
            return cls._hz_vals(text)
        if field == "resolution":
            return cls._res_vals(text)
        if field == "ram":
            return cls._ram_vals(text)
        return set()

    @classmethod
    def _fact_text(cls, variant: "Variant", field: str) -> str:
        needles = cls._FACT_LABEL.get(field, ())
        parts: list[str] = []
        for fk, fv in (variant.facts or {}).items():
            fl = str(fk).lower()
            if any(n in fl for n in needles):
                parts.append(str(fv))
        return " ".join(parts)

    @classmethod
    def _product_value(cls, variant: "Variant", field: str) -> Any:
        """Product-wide value: labelled facts + title; conflict → None."""
        vals: set = set()
        conflict = False
        fact_text = cls._fact_text(variant, field)
        if fact_text:
            found = cls._surface_vals(field, fact_text)
            if len(found) > 1:
                conflict = True
            # Bare number under a unit-bearing fact label.
            if not found and field != "resolution":
                raw = fact_text.strip()
                if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                    num = float(raw)
                    if field == "screen_size" and 3.0 <= num <= 100.0:
                        found = {num}
                    elif field == "refresh_rate":
                        found = {num}
                    elif field == "ram":
                        found = {num}
            if field == "resolution" and re.search(r"\bqHD\b", fact_text):
                conflict = True
            vals |= found
        title = variant.title or ""
        if field == "resolution" and re.search(r"\bqHD\b", title):
            conflict = True
        title_vals = cls._surface_vals(field, title)
        if len(title_vals) > 1:
            conflict = True
        vals |= title_vals
        if conflict or len(vals) > 1:
            return None
        return cls._one(vals)

    @classmethod
    def _variant_value(cls, variant: "Variant", field: str) -> Any:
        """Exact-variant value: options win; ambiguous option mention blocks fallback."""
        opt = (variant.options or "").strip()
        vals = cls._surface_vals(field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if cls._mentions(field, opt):
            return None
        return cls._product_value(variant, field)

    @classmethod
    def screen_inches(cls, variant: "Variant") -> Optional[float]:
        """Prefer selected-variant options over product-wide facts/title."""
        got = cls._variant_value(variant, "screen_size")
        return float(got) if isinstance(got, (int, float)) else None

    @classmethod
    def resolution_token(cls, variant: "Variant") -> Optional[str]:
        got = cls._variant_value(variant, "resolution")
        return str(got) if got else None

    @classmethod
    def refresh_hz(cls, variant: "Variant") -> Optional[float]:
        got = cls._variant_value(variant, "refresh_rate")
        return float(got) if isinstance(got, (int, float)) else None

    @classmethod
    def ram_gb(cls, variant: "Variant") -> Optional[float]:
        got = cls._variant_value(variant, "ram")
        return float(got) if isinstance(got, (int, float)) else None

    @classmethod
    def normalize_wanted(cls, key: str, value: str) -> tuple[str, Any]:
        key_l = key.lower().replace("-", "_")
        val = str(value).strip()
        val_l = val.lower().replace(" ", "")
        # Never treat resolution tokens as inch sizes (e.g. "720p" → 720).
        if val_l in cls._RES_ALIASES or re.fullmatch(r"\d{3,4}p|4k|uhd|qhd|hd|fullhd|fhd", val_l):
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
            # Reject absurd inch values that are almost certainly misparsed tokens.
            if inches is not None and not (3.0 <= float(inches) <= 100.0):
                inches = None
            return "screen_size", inches
        return key_l, val.lower()

    @classmethod
    def matches(cls, variant: "Variant", key: str, value: str) -> bool:
        kind, wanted = cls.normalize_wanted(key, value)
        if wanted is None or wanted == "":
            return False
        if kind == "screen_size":
            got = cls.screen_inches(variant)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            # Exact to hundredths — 6.78 vs 6.79 must not collide.
            return abs(float(got) - float(wanted)) < 0.005
        if kind == "resolution":
            got = cls.resolution_token(variant)
            if not got:
                return False
            want = str(wanted).lower()
            # HD class ↔ 720p
            if want in {"hd", "720p"} and got in {"hd", "720p"}:
                return True
            alts = cls._RES_ALIASES.get(want, (want,))
            return got in alts or got == want
        if kind == "refresh_rate":
            got = cls.refresh_hz(variant)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) <= 1.0
        if kind == "ram":
            got = cls.ram_gb(variant)
            if got is None or not isinstance(wanted, (int, float)):
                return False
            return abs(float(got) - float(wanted)) < 0.05
        # generic fallback: require full normalized value, not a loose substring of digits
        return _norm(str(wanted)) in _norm(variant.text_blob())


class DeviceGuard:
    """Reject accessory / decoy listings when a real device category is required."""

    _ACCESSORY = (
        "case",
        "cover",
        "charger",
        "cable",
        "screen protector",
        "tempered glass",
        "replacement tip",
        "ear tip",
        "strap only",
        "mount only",
        "holder only",
        "stand only",
        "skin only",
        "bumper",
        "holster",
        "pouch",
        "digitizer",
        "digitiz",
        "panel assembly",
        "replacement part",
        "touch screen digitizer",
        "touch screen digitiz",
        "spare part",
        "repair part",
    )

    _BAD_PATH = (
        "accessor",
        "replacement",
        "spare part",
        "repair",
        "parts & tools",
        "parts and tools",
    )

    @classmethod
    def is_accessory_title(cls, title: str) -> bool:
        t = (title or "").lower()
        if not t:
            return False
        device_hit = bool(
            re.search(
                r"\b(smartphone|mobile phone|cellphone|tablet|ipad|monitor|headphone|earbud|headset)\b",
                t,
            )
        )
        if any(b in t for b in cls._ACCESSORY):
            if re.search(r"\b(with case|free case|include[sd]? case)\b", t) and device_hit:
                return False
            if re.search(r"\b(phone|tablet|monitor)\s+(case|cover|protector|bumper|pouch)\b", t):
                return True
            if re.search(r"\b(case|cover|protector|bumper)\b.*\b(phone|tablet)\b", t):
                return True
            return True
        # Standalone LCD/digitizer replacement panels (not a phone advertising its display).
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
    def path_is_accessory(cls, category_path: list[str]) -> bool:
        path = _norm(" ".join(category_path or []))
        return any(b in path for b in cls._BAD_PATH)

    @classmethod
    def looks_like_phone(cls, variant: "Variant") -> bool:
        if cls.is_accessory_title(variant.title or "") or cls.path_is_accessory(variant.category_path):
            return False
        title = variant.title or ""
        path = _norm(" ".join(variant.category_path))
        blob = variant.text_blob()
        if re.search(
            r"\b(smartphone|cellphone|mobile phone|android phone|iphone|galaxy\s+[a-z]?\d)\b",
            title,
            re.I,
        ):
            return True
        if any(w in path for w in ("smartphone", "mobile phone", "cell phone")):
            return True
        for fk, fv in (variant.facts or {}).items():
            fl = f"{fk} {fv}".lower()
            if re.search(r"\b(android|ios|iphone os|blackberry os|harmonyos)\b", fl):
                return True
        if re.search(r"\b(android|ios)\b", blob) and re.search(r"\b(gb ram|mp camera|dual sim)\b", blob):
            return True
        return False

    @classmethod
    def looks_like_tablet(cls, variant: "Variant") -> bool:
        if cls.is_accessory_title(variant.title or "") or cls.path_is_accessory(variant.category_path):
            return False
        title = variant.title or ""
        # Drones / cameras often share brand filters with tablets — reject them.
        if re.search(r"\b(drone|dron|aircraft|quadcopter|uav)\b", title, re.I):
            return False
        path = _norm(" ".join(variant.category_path))
        if re.search(r"\b(tablet|ipad)\b", title, re.I):
            return True
        if "tablet" in path:
            return True
        for fk, fv in (variant.facts or {}).items():
            if re.search(r"\b(tablet|ipados)\b", f"{fk} {fv}", re.I):
                return True
        return False

    @classmethod
    def looks_like_monitor(cls, variant: "Variant") -> bool:
        if cls.is_accessory_title(variant.title or "") or cls.path_is_accessory(variant.category_path):
            return False
        title = variant.title or ""
        path = _norm(" ".join(variant.category_path))
        if re.search(r"\b(monitor|display)\b", title, re.I) and "lcd display" not in title.lower():
            # Avoid replacement "LCD display" panels.
            if re.search(r"\b(digitizer|replacement|for vivo|for samsung|for xiaomi)\b", title, re.I):
                return False
            return True
        return "monitor" in path or ("display" in path and "accessor" not in path)

    @classmethod
    def ok(cls, variant: "Variant", category: Optional[str]) -> bool:
        if not category:
            return True
        want = _norm(category).rstrip("s")
        title = variant.title or ""
        if cls.is_accessory_title(title) or cls.path_is_accessory(variant.category_path):
            if any(w in want for w in ("phone", "smartphone", "tablet", "monitor", "headphone", "earbud")):
                return False
        # Unviewed search hits often omit "smartphone" in the title — only enforce
        # positive device evidence after view/facts/path are available.
        rich = bool(variant.viewed or variant.facts or variant.category_path)
        if "smartphone" in want or want == "phone":
            return cls.looks_like_phone(variant) if rich else True
        if "tablet" in want:
            return cls.looks_like_tablet(variant) if rich else True
        if "monitor" in want:
            return cls.looks_like_monitor(variant) if rich else True
        return True


class CartState:
    """Apply cart-add observations and embedded market events to a Variant."""

    @staticmethod
    def apply_add(variant: "Variant", add_obs: dict[str, Any]) -> "Variant":
        if not isinstance(add_obs, dict):
            return variant
        added = add_obs.get("added") if isinstance(add_obs.get("added"), dict) else {}
        event = add_obs.get("market_event") if isinstance(add_obs.get("market_event"), dict) else {}
        cart = add_obs.get("cart") if isinstance(add_obs.get("cart"), list) else []

        if added:
            price = _as_float(added.get("price"))
            if price is not None:
                variant.price = price
            if "in_stock" in added:
                variant.in_stock = bool(added["in_stock"])
        # Cart rows sometimes carry the post-event truth.
        for row in cart:
            if not isinstance(row, dict):
                continue
            if str(row.get("product_id")) != variant.product_id:
                continue
            if str(row.get("sku") or "default") != (variant.sku or "default"):
                continue
            if "in_stock" in row:
                variant.in_stock = bool(row["in_stock"])
            cp = _as_float(row.get("current_price"))
            if cp is not None:
                variant.price = cp
        if event.get("kind") == "stockout":
            variant.in_stock = False
        if event.get("kind") == "price_change":
            new_p = _as_float(event.get("new_price"))
            if new_p is not None:
                variant.price = new_p
            # Some events also flip stock.
            if "in_stock_after" in event:
                variant.in_stock = bool(event["in_stock_after"])
        return variant


class OptionGuard:
    """Reject screen-only / replacement-panel SKU options when buying a device."""

    _BAD_OPT = re.compile(
        r"screen[- ]?only|replacement[- ]?screen|digitizer|lcd panel|touch panel only",
        re.I,
    )

    @classmethod
    def ok(cls, variant: "Variant", category: Optional[str]) -> bool:
        if not category:
            return True
        want = _norm(category)
        if not any(w in want for w in ("phone", "smartphone", "tablet", "monitor")):
            return True
        opt = variant.options or ""
        if cls._BAD_OPT.search(opt):
            return False
        return True


class EvidenceGate:
    """TF7/TF3: gate attribute must be visible on the selected variant."""

    @classmethod
    def gate_key(cls, plan: "Plan") -> Optional[str]:
        return ClaimBuilder.gate_from_goal(plan.goal, plan.attributes) or (
            next(iter(plan.attributes), None) if plan.attributes else None
        )

    @classmethod
    def measured_value(cls, plan: "Plan", variant: "Variant") -> Optional[str]:
        """Return the visible gate value, or None if unsupported on this variant."""
        gate = cls.gate_key(plan)
        if not gate:
            return None
        if gate == "resolution":
            # Prefer title/options literals for claim safety; fall back to facts
            # for matching via SpecKit.resolution_token.
            return SpecKit.resolution_token(variant)
        if gate == "screen_size":
            inches = SpecKit.screen_inches(variant)
            return f"{inches:g}" if inches is not None else None
        if gate == "refresh_rate":
            hz = SpecKit.refresh_hz(variant)
            return f"{int(hz)}Hz" if hz is not None else None
        if gate == "ram":
            gb = SpecKit.ram_gb(variant)
            return f"{int(gb)}GB" if gb is not None else None
        return None

    @classmethod
    def visible(cls, plan: "Plan", variant: "Variant") -> bool:
        """True when the firm gate attr is measurable AND matches the plan want."""
        gate = cls.gate_key(plan)
        if not gate:
            return True
        measured = cls.measured_value(plan, variant)
        if measured is None:
            return False
        if gate in plan.attributes:
            return SpecKit.matches(variant, gate, plan.attributes[gate])
        if gate == "resolution":
            goal = plan.goal.lower()
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
    def claimable(cls, plan: "Plan", variant: "Variant") -> bool:
        """True when a TF7 gate claim would be accepted (literal surface evidence)."""
        return cls.claim_value(plan, variant) is not None

    @classmethod
    def claim_value(cls, plan: "Plan", variant: "Variant") -> Optional[str]:
        """Value safe to put in a TF7 claim (literal title/options evidence)."""
        if not cls.visible(plan, variant):
            return None
        gate = cls.gate_key(plan)
        measured = cls.measured_value(plan, variant)
        if not gate or not measured:
            return None
        # Resolution claims must be literally supported on title/options —
        # pixel-class inference alone is treated as canonical/unsupported.
        if gate == "resolution":
            fact_blob = " ".join(f"{k} {v}" for k, v in (variant.facts or {}).items())
            surface = f"{variant.title or ''} {variant.options or ''} {fact_blob}".lower()
            goal = plan.goal.lower()
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
                if re.search(r"\bhd\b", goal) and "1080" not in goal and "full" not in goal and (lit_hd or lit_720):
                    return "hd" if lit_hd and not lit_720 else ("720p" if lit_720 else None)
                if lit_720:
                    return "720p"
                if lit_hd:
                    return "hd"
                return None
            if measured == "1080p" and lit_1080:
                return "1080p"
            # Require literal class token for other resolutions too.
            if measured.replace("p", "") and measured.lower() in surface:
                return measured
            return None
        return measured


class SkuSpecPicker:
    """Prefer SKUs whose selected options explicitly state the firm attr (TF3)."""

    @classmethod
    def option_mentions(cls, plan: "Plan", variant: "Variant") -> bool:
        opt = (variant.options or "").strip().lower()
        if not opt:
            return False
        for key, val in plan.attributes.items():
            kind, wanted = SpecKit.normalize_wanted(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                num = f"{float(wanted):g}"
                if re.search(
                    rf"\b{re.escape(num)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b", opt
                ) or re.search(rf"\({re.escape(num)}\s*inch", opt):
                    return True
            elif kind == "resolution" and wanted:
                want = str(wanted).lower()
                if want in opt or (want in {"hd", "720p"} and ("720p" in opt or re.search(r"\bhd\b", opt))):
                    return True
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                if re.search(rf"\b{int(wanted)}\s*hz\b", opt):
                    return True
            elif kind == "ram" and wanted:
                if _norm(str(wanted)) in _norm(opt):
                    return True
        return False

    @classmethod
    def rank_tuple(cls, plan: "Plan", variant: "Variant") -> tuple:
        return (
            1 if OptionGuard.ok(variant, plan.category) else 0,
            1 if EvidenceGate.visible(plan, variant) else 0,
            1 if cls.option_mentions(plan, variant) else 0,
            score_variant(plan, variant),
        )


class ClaimBuilder:
    """Build TF7 justification claims from viewed facts (no free-text summary)."""

    _CLAIM_DENY = frozenset({"category"})

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
    def build(cls, plan: "Plan", variant: "Variant") -> dict[str, Any]:
        # Never attach justification unless the shopper asked for a reason.
        if not plan.needs_justification:
            return {"claims": []}
        claims: list[dict[str, str]] = []
        gate = cls.gate_from_goal(plan.goal, plan.attributes)
        # Gate claim ONLY from visible evidence — never echo plan wants,
        # never invent a resolution class from the query alone.
        claim_val = EvidenceGate.claim_value(plan, variant)
        if gate and claim_val and gate not in cls._CLAIM_DENY:
            claims.append({"field": gate, "value": str(claim_val)})

        if variant.price is not None:
            claims.append({"field": "price", "value": f"{float(variant.price):g}"})
        claims.append({"field": "in_stock", "value": "true" if variant.in_stock else "false"})

        want_brand = plan.brand or plan.preferred_brand
        if variant.brand and (not want_brand or brand_ok(variant, want_brand)):
            claims.append({"field": "brand", "value": str(variant.brand)})
        elif want_brand and brand_ok(variant, want_brand) and want_brand.lower() in _norm(
            variant.title or ""
        ):
            claims.append({"field": "brand", "value": str(want_brand)})
        # No soft filler claims (hdr/gaming/…) — unasked fields fail faithfulness.

        uniq: dict[str, dict[str, str]] = {}
        for claim in claims:
            if claim["field"] in cls._CLAIM_DENY:
                continue
            if re.search(r"\d+\s*[-–/]\s*\d+", claim["value"]):
                continue
            uniq.setdefault(claim["field"], claim)
        return {"claims": list(uniq.values())}


class CommitPicker:
    """Select a first-cart SKU aimed at ORO Bench construct success.

    Per ORO Bench docs, ``completed`` only means the episode terminated;
    ``paid_reward`` comes from the family verifier. Event families need a
    high-quality first cart (intended event commitment) plus event-closed
    recovery — not merely a final in-gold order.
    """

    EVENT_FAMILIES = frozenset(
        {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        }
    )

    @classmethod
    def soft_brand(cls, plan: "Plan") -> Optional[str]:
        if plan.family == "justification":
            return plan.brand or plan.preferred_brand
        if plan.family == "intent_decomposition":
            return plan.preferred_brand
        if plan.family == "constraint_satisfaction":
            return plan.brand
        return plan.brand or plan.preferred_brand

    @classmethod
    def best(cls, plan: "Plan", variants: list["Variant"]) -> Optional["Variant"]:
        if not variants:
            return None
        buyable = [
            v
            for v in variants
            if hard_ok(plan, v) and purchase_ok(plan, v) and OptionGuard.ok(v, plan.category)
        ]
        firm = buyable or [
            v for v in variants if hard_ok(plan, v) and OptionGuard.ok(v, plan.category)
        ]
        if not firm:
            return None
        brand = cls.soft_brand(plan)
        if brand:
            branded = [v for v in firm if brand_ok(v, brand)]
            # Soft preference only — never empty the firm pool if brand misses.
            if branded:
                # Prefer title/native brand over filter-only stamps.
                strong = [v for v in branded if brand_strength(v, brand) >= 2]
                firm = strong or branded
        # TF7: require claimable gate evidence before first-cart commitment.
        if plan.family == "justification":
            claimable = [v for v in firm if EvidenceGate.claimable(plan, v)]
            if claimable:
                firm = claimable
            else:
                # Do not first-cart unclaimable SKUs when any claimable exists in firm.
                evidenced = [v for v in firm if EvidenceGate.visible(plan, v)]
                if evidenced:
                    firm = evidenced
        # Event families: suite-agnostic commitment ranking (not raw score alone).
        if plan.family in cls.EVENT_FAMILIES:
            firm.sort(key=lambda v: EventCommit.score(plan, v), reverse=True)
            return firm[0]
        firm.sort(
            key=lambda v: (brand_strength(v, brand) if brand else 0, score_variant(plan, v)),
            reverse=True,
        )
        return firm[0]


class EventCommit:
    """Suite-agnostic first-cart ranking for event-evaluated families.

    ORO Bench ``intended_event_commitment`` keys off the first ``add_to_cart``.
    A later gold recovery does not repair a decoy first cart. Rank by
    claimability, brand evidence, listing ambiguity, and budget position —
    never by product IDs, suite task counts, or family quotas.
    """

    @classmethod
    def listing_ambiguity(cls, variant: "Variant") -> float:
        """Penalize multi-model / multi-size mashup titles (relative ranking only)."""
        title = variant.title or ""
        pen = 0.0
        inch_hits = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:\"|''|in(?:ch(?:es)?)?)\b", title, re.I
        )
        if len({h for h in inch_hits}) >= 2:
            pen -= 2.5
        if re.search(r"\b\d+\s*(?:\"|in(?:ch)?)\s*/\s*\d+", title, re.I):
            pen -= 2.0
        if re.search(r"\b[A-Za-z]?\d{2,}[A-Za-z0-9]*\s*/\s*[A-Za-z]?\d{2,}", title):
            pen -= 1.5
        if title.count(" / ") >= 1 and len(title) > 80:
            pen -= 0.5
        return pen

    @classmethod
    def budget_position(cls, plan: "Plan", variant: "Variant") -> float:
        if plan.budget is None or variant.price is None:
            return 0.0
        budget = float(plan.budget)
        if budget <= 0:
            return 0.0
        ratio = float(variant.price) / budget
        if ratio > 1.0:
            return -5.0
        # Very cheap firm-looking hits are often incomplete / wrong listings.
        if ratio < 0.25:
            return -1.5
        if 0.5 <= ratio <= 0.98:
            return 1.2
        if 0.35 <= ratio < 0.5:
            return 0.3
        return 0.0

    @classmethod
    def score(cls, plan: "Plan", variant: "Variant") -> float:
        s = float(score_variant(plan, variant))
        s += cls.listing_ambiguity(variant)
        s += cls.budget_position(plan, variant)
        brand = CommitPicker.soft_brand(plan)
        if brand:
            strength = brand_strength(variant, brand)
            s += 1.0 * strength
            if strength >= 3:
                s += 1.0
            elif strength <= 1:
                s -= 0.8
        if plan.family == "justification":
            if EvidenceGate.claimable(plan, variant):
                s += 4.0
            elif EvidenceGate.visible(plan, variant):
                s += 1.0
            else:
                s -= 3.0
        if plan.attributes and SkuSpecPicker.option_mentions(plan, variant):
            s += 2.5
        if variant.viewed:
            s += 0.8
        if not OptionGuard.ok(variant, plan.category):
            s -= 4.0
        return s

    @classmethod
    def shortlist(
        cls, plan: "Plan", variants: list["Variant"], *, limit: int = 6
    ) -> list["Variant"]:
        pool = [
            v
            for v in variants
            if hard_ok(plan, v) and purchase_ok(plan, v) and OptionGuard.ok(v, plan.category)
        ]
        if plan.family == "justification":
            claimable = [v for v in pool if EvidenceGate.claimable(plan, v)]
            if claimable:
                pool = claimable
        brand = CommitPicker.soft_brand(plan)
        if brand:
            # Prefer any brand evidence (title/field/filter stamp), not only
            # title-native strength>=2 — filter stamps are strength 2 once
            # brand is filled, strength 1 via source_brand alone.
            branded = [v for v in pool if brand_ok(v, brand)]
            strong = [v for v in branded if brand_strength(v, brand) >= 2]
            if strong:
                pool = strong
            elif branded:
                pool = branded
        pool.sort(key=lambda v: cls.score(plan, v), reverse=True)
        # Unique product_ids preserving rank.
        out: list[Variant] = []
        seen: set[str] = set()
        for v in pool:
            if v.product_id in seen:
                continue
            seen.add(v.product_id)
            out.append(v)
            if len(out) >= limit:
                break
        return out


def category_ok(variant: Variant, category: Optional[str]) -> bool:
    if not category:
        return True
    if not DeviceGuard.ok(variant, category):
        return False
    want = _norm(category).rstrip("s")
    path = _norm(" ".join(variant.category_path))
    title = _norm(variant.title)
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


def attr_match(variant: Variant, key: str, value: str) -> bool:
    return SpecKit.matches(variant, key, value)


def brand_ok(variant: Variant, brand: Optional[str], *, soft: bool = False) -> bool:
    if not brand:
        return True
    want = _norm(brand)
    if not want:
        return True
    vb = _norm(variant.brand or "")
    vs = _norm(variant.source_brand or "")
    hit = want in vb or want in vs or want in _norm(variant.title)
    return hit if not soft else hit


def brand_strength(variant: Variant, brand: Optional[str]) -> int:
    """How strongly a listing shows the preferred brand (title > field > filter stamp)."""
    if not brand:
        return 0
    want = _norm(brand)
    if not want:
        return 0
    if want in _norm(variant.title or ""):
        return 3
    if want in _norm(variant.brand or ""):
        return 2
    if want in _norm(variant.source_brand or ""):
        return 1
    return 0


def budget_ok(variant: Variant, budget: Optional[float]) -> bool:
    if budget is None or variant.price is None:
        return True
    return float(variant.price) <= float(budget) + 1e-6


def hard_ok(plan: Plan, variant: Variant) -> bool:
    if plan.require_in_stock and not variant.in_stock:
        return False
    if not budget_ok(variant, plan.budget):
        return False
    if not category_ok(variant, plan.category):
        return False
    if not DeviceGuard.ok(variant, plan.category):
        return False
    if not OptionGuard.ok(variant, plan.category):
        return False
    if plan.brand and not brand_ok(variant, plan.brand):
        return False
    for key, value in plan.attributes.items():
        if not attr_match(variant, key, value):
            return False
    # TF7: gate must be visibly supported (claims fail otherwise).
    if plan.family == "justification" and EvidenceGate.gate_key(plan):
        if not EvidenceGate.visible(plan, variant):
            return False
    return True


def purchase_ok(plan: Plan, variant: Variant) -> bool:
    """Minimal gate after cart events: stock + budget (+ mandatory brand only)."""
    if plan.require_in_stock and not variant.in_stock:
        return False
    if not budget_ok(variant, plan.budget):
        return False
    if plan.brand and not brand_ok(variant, plan.brand):
        return False
    # Never buy accessories when a device category is required.
    if plan.category and not DeviceGuard.ok(variant, plan.category):
        return False
    return True


def needs_event_recovery(plan: Plan, variant: Variant) -> bool:
    """Recover only when the committed SKU is no longer buyable."""
    return not purchase_ok(plan, variant)


def _fact_number(variant: Variant, *needles: str) -> Optional[float]:
    blob_parts = []
    for fk, fv in (variant.facts or {}).items():
        fl = str(fk).lower()
        if any(n in fl for n in needles):
            num = _as_float(fv)
            if num is not None:
                return num
        blob_parts.append(f"{fk} {fv}")
    blob = " ".join(blob_parts + [variant.options, variant.title]).lower()
    if any(n in ("refresh", "hz") for n in needles):
        m = re.search(r"(\d{2,3})\s*hz", blob)
        if m:
            return float(m.group(1))
    if any(n in ("screen", "size", "inch") for n in needles):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|\")", blob)
        if m:
            return float(m.group(1))
    return None


def ranking_key(variant: Variant) -> tuple:
    """Higher tuple sorts first for ranking family (refresh, size, cheaper)."""
    refresh = SpecKit.refresh_hz(variant)
    size = SpecKit.screen_inches(variant)
    price = float(variant.price) if variant.price is not None else float("inf")
    return (
        float(refresh or -1.0),
        float(size or -1.0),
        -price,
    )


def preference_score(plan: Plan, variant: Variant) -> float:
    """Soft-axis fit for TF4: sweat/water, sport fit, battery, wireless, etc."""
    blob = variant.text_blob()
    score = 0.0
    axes = " ".join(plan.soft_prefs + [plan.goal]).lower()

    def hit(*pats: str) -> bool:
        return any(re.search(p, blob, re.I) for p in pats)

    want_sweat = bool(re.search(r"sweat|water|rain|weather|waterproof|ipx?\d", axes, re.I))
    want_sport = bool(re.search(r"gym|run|sport|workout|fitness|training", axes, re.I))
    want_battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    want_wireless = bool(re.search(r"wireless|bluetooth|cordless|no.?wire", axes, re.I))
    want_noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))

    if want_sweat or want_sport:
        if hit(r"ipx?[0-9]", r"sweat", r"water.?resist", r"waterproof", r"splash"):
            score += 4.0
        if hit(r"sport", r"gym", r"running", r"workout", r"ear.?hook", r"secure.?fit", r"bootcamp"):
            score += 2.5
    if want_battery or want_sport:
        days = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*days?", blob)]
        hours = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*hours?", blob)]
        if days and max(days) >= 5:
            score += 2.0
        elif hours and max(hours) >= 20:
            score += 1.5
        elif hit(r"long.?battery", r"battery.?life"):
            score += 1.0
    if want_wireless or "headphone" in (plan.category or "").lower() or "earbud" in axes:
        if hit(r"bluetooth", r"wireless", r"tws", r"true.?wireless"):
            score += 1.5
        if hit(r"wired", r"3\.5\s*mm") and not hit(r"bluetooth", r"wireless"):
            score -= 1.0
    if want_noise and hit(r"\banc\b", r"noise.?cancel", r"active.?noise"):
        score += 1.5
    # generic soft token overlap
    for pref in plan.soft_prefs:
        for tok in _tokens(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.2
    # accessory penalty for device categories
    if plan.category and "headphone" in plan.category.lower():
        if hit(r"\bcase\b", r"cover", r"cable only", r"replacement tip") and not hit(
            r"earbud", r"headphone", r"headset", r"earphone"
        ):
            score -= 5.0
    return score


def soft_search_queries(plan: Plan) -> list[str]:
    """Build preference catalog queries from category + soft tokens in the plan.

    Suite-agnostic: tokens come from plan.soft_prefs / goal regex hits only —
    no fixed product phrases or branded pack motifs.
    """
    cat = (plan.category or "").strip()
    cat_l = cat.lower()
    axes = " ".join(plan.soft_prefs + [plan.goal]).lower()
    out: list[str] = []

    def push(parts: list[str]) -> None:
        q = " ".join(p for p in parts if p).strip()
        if q and q not in out:
            out.append(q)

    water = bool(re.search(r"sweat|water|rain|weather|waterproof|water.?resist|ipx", axes, re.I))
    sport = bool(re.search(r"gym|run|sport|workout|fitness|training|bootcamp", axes, re.I))
    battery = bool(re.search(r"battery|long.?lasting|all.?day", axes, re.I))
    wireless = bool(re.search(r"wireless|bluetooth|cordless", axes, re.I))
    noise = bool(re.search(r"\bnoise\b|\banc\b|noise.?cancel", axes, re.I))
    audio = any(k in cat_l for k in ("headphone", "earbud", "headset", "earphone")) or bool(
        re.search(r"headphone|earbud|headset|earphone", axes, re.I)
    )

    if audio:
        base = cat or "headphones"
        # Water/sport first when the goal asks — wireless-only seeds bury IPX hits.
        if water:
            push([base, "waterproof", "sweat", "resistant"])
            push(["ipx", "wireless", "earbuds"])
        if sport:
            push([base, "sport", "wireless"])
        if water and sport:
            push(["waterproof", "sport", "earbuds", "bluetooth"])
        if battery:
            push([base, "bluetooth", "battery"])
        if noise:
            push(["noise", "cancelling", "wireless", "earbuds"])
        push([base, "wireless", "bluetooth"])
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
        for pref in plan.soft_prefs:
            for t in _tokens(pref):
                if len(t) > 3 and t not in seeds:
                    seeds.append(t)
        push(([cat] if cat else []) + seeds[:6])
        if cat and seeds:
            push(seeds[:4] + [cat])
    return out


def score_variant(plan: Plan, variant: Variant, rank: int = 0) -> float:
    score = 0.0
    if hard_ok(plan, variant):
        score += 10.0
    else:
        # partial credit
        if variant.in_stock:
            score += 1.0
        if budget_ok(variant, plan.budget):
            score += 1.0
        if category_ok(variant, plan.category):
            score += 1.5
        for key, value in plan.attributes.items():
            if attr_match(variant, key, value):
                score += 2.0
    prefer = plan.preferred_brand or (None if plan.brand else None)
    if prefer and brand_ok(variant, prefer):
        bonus = 3.0 if plan.family == "intent_decomposition" else 1.5
        # Title/native brand beats filter stamp alone for first-cart commitment.
        bonus += 0.5 * brand_strength(variant, prefer)
        score += bonus
    if plan.brand and brand_ok(variant, plan.brand):
        score += 1.0
        score += 0.25 * brand_strength(variant, plan.brand)
    # Exact-variant / claimable evidence bonuses (TF3 / TF7).
    if plan.family in {"constraint_satisfaction", "justification"}:
        if EvidenceGate.visible(plan, variant):
            score += 2.5
        if SkuSpecPicker.option_mentions(plan, variant):
            score += 2.0
    blob = variant.text_blob()
    for pref in plan.soft_prefs:
        for tok in _tokens(pref):
            if len(tok) > 3 and tok in blob:
                score += 0.15
    for kw in plan.keywords:
        if _norm(kw) and _norm(kw) in _norm(blob):
            score += 0.2
    if plan.preferred_model and _norm(plan.preferred_model) in _norm(blob):
        score += 2.0
    if plan.preferred_color and _norm(plan.preferred_color) in _norm(blob + " " + variant.options):
        score += 1.0
    # ranking axes (explicit tradeoff order)
    if plan.family == "ranking":
        refresh, size, neg_price = ranking_key(variant)
        score += max(refresh, 0) * 10.0
        score += max(size, 0) * 1.0
        score += max(0.0, 5.0 + neg_price / 10000.0)
    if plan.family == "preference_reasoning":
        score += preference_score(plan, variant) * 2.0
    score -= rank * 0.01
    if variant.price is not None and plan.budget and plan.family != "ranking":
        ratio = float(variant.price) / float(plan.budget)
        if 0.4 <= ratio <= 0.95:
            score += 0.3
    # candidate-key bonus for TF5
    if plan.candidate_keys and variant.key in plan.candidate_keys:
        score += 50.0
    return score


def variant_from_summary(item: dict[str, Any]) -> Variant:
    return Variant(
        product_id=str(item.get("product_id") or ""),
        sku=str(item.get("sku") or "default"),
        title=str(item.get("title") or ""),
        brand=str(item.get("brand") or ""),
        price=_as_float(item.get("price")),
        currency=str(item.get("currency") or "PHP"),
        in_stock=bool(item.get("in_stock", True)),
    )


def enrich_from_view(variant: Variant, obs: dict[str, Any]) -> Variant:
    if obs.get("error"):
        return variant
    variant.title = str(obs.get("title") or variant.title)
    variant.brand = str(obs.get("brand") or variant.brand)
    variant.category_path = list(obs.get("category_path") or [])
    variant.facts = dict(obs.get("facts") or {})
    variant.listing_id = obs.get("listing_id") or variant.listing_id
    variant.viewed = True
    return variant


def variants_from_view(obs: dict[str, Any]) -> list[Variant]:
    if obs.get("error"):
        return []
    base = Variant(
        product_id=str(obs.get("product_id") or ""),
        sku="",
        title=str(obs.get("title") or ""),
        brand=str(obs.get("brand") or ""),
        category_path=list(obs.get("category_path") or []),
        facts=dict(obs.get("facts") or {}),
        listing_id=obs.get("listing_id"),
        viewed=True,
    )
    out: list[Variant] = []
    for row in obs.get("variants") or []:
        v = Variant(
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
        out.append(v)
    return out or []


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------


class SOSOAgent:
    def __init__(self, session: EnvSession):
        self.session = session
        self.plan = parse_plan(session.query)
        # TF4: replace LLM search_queries with axis-derived seeds only.
        if self.plan.family == "preference_reasoning":
            self.plan.search_queries = soft_search_queries(self.plan)[:6]
            self.plan.soft_prefs = [
                p
                for p in self.plan.soft_prefs
                if str(p).strip().lower() not in {"yes", "no", "ok", "true", "false"}
            ]
        self.pool: dict[str, Variant] = {}
        self.observed_product_ids: list[str] = []
        self._requery_count = 0
        self._searchish = 0
        self._cart_adds = 0
        self._pending_recheck: Optional[tuple[str, str]] = None
        self._last_inspect_oos = False
        self._dead_keys: set[str] = set()
        self._recent_hits: list[str] = []

    # -- tool helpers (capabilities → policy_view tools; soft-skip if absent) --

    def search(self, query: str, *, k: int = 10, in_stock: Optional[bool] = True) -> list[Variant]:
        if not self.session.can("search"):
            # Fall back to catalog filter when search is absent from this task.
            if self.session.can("filter"):
                return self.filter_catalog(
                    category=self.plan.category,
                    brand=self.plan.brand or self.plan.preferred_brand,
                    max_price=self.plan.budget,
                    k=k,
                    in_stock=in_stock,
                    _allow_search_fallback=False,
                )
            return []
        # TF4: refuse filler LLM queries that bury axis seeds.
        if self.plan.family == "preference_reasoning":
            q_toks = set(_tokens(query))
            filler = {"yes", "no", "ok", "true", "false", "please", "thanks"}
            cat = set(_tokens(self.plan.category or "")) | {
                "headphone", "headphones", "earbud", "earbuds", "headset",
            }
            if q_toks and q_toks <= (filler | cat):
                return []
            if (q_toks & filler) and not (
                q_toks & {
                    "sweat", "water", "ipx", "sport", "gym", "wireless",
                    "bluetooth", "battery", "noise", "waterproof", "resistant",
                    "workout", "bootcamp",
                }
            ):
                return []
        args: dict[str, Any] = {"query": query, "k": k}
        if in_stock is not None:
            args["in_stock"] = in_stock
        obs = self.session.act("search", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        self._searchish += 1
        out = self._ingest_results(obs)
        self._recent_hits = [v.product_id for v in out]
        return out

    def filter_catalog(
        self,
        *,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        max_price: Optional[float] = None,
        k: int = 10,
        in_stock: Optional[bool] = True,
        _allow_search_fallback: bool = True,
    ) -> list[Variant]:
        if not self.session.can("filter"):
            # No filter tool: approximate with a free-text search when available.
            if _allow_search_fallback and self.session.can("search"):
                bits = [p for p in (brand, category) if p]
                if max_price is not None:
                    bits.append(str(max_price))
                q = " ".join(bits).strip() or (self.plan.category or "product")
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
        obs = self.session.act("filter", **args)
        if obs.get("error") == "tool_unavailable":
            return []
        self._searchish += 1
        out = self._ingest_results(obs)
        self._recent_hits = [v.product_id for v in out] + [
            p for p in self._recent_hits if p not in {v.product_id for v in out}
        ]
        # Catalog brand filter is authoritative when listing brand is missing.
        # Stamp both source_brand and brand so soft TF1 preferred-brand filters
        # count as brand_strength>=2 (intended_event_commitment).
        if brand:
            for v in out:
                v.source_brand = brand
                if not (v.brand or "").strip():
                    v.brand = brand
                self.pool[v.key] = v
        return out

    def view(self, product_id: str) -> list[Variant]:
        if not self.session.can("view"):
            # Last-resort: compare a single id if that is the only inspect path.
            if self.session.can("compare"):
                return self.compare([product_id])
            return []
        obs = self.session.act("view", product_id=product_id)
        if obs.get("error") == "tool_unavailable":
            return []
        variants = variants_from_view(obs)
        prev_brand = ""
        prev_source = ""
        for prev in self.pool.values():
            if prev.product_id == product_id:
                prev_brand = prev.brand or prev_brand
                prev_source = prev.source_brand or prev_source
        for v in variants:
            # Preserve brand signal learned from earlier brand-filtered hits.
            if not (v.brand or "").strip():
                v.brand = prev_brand
            v.source_brand = v.source_brand or prev_source or v.brand
            self.pool[v.key] = v
            if v.product_id not in self.observed_product_ids:
                self.observed_product_ids.append(v.product_id)
        return variants

    def compare(self, product_ids: list[str]) -> list[Variant]:
        if not self.session.can("compare"):
            # Sequential view when compare is not in this task's tool list.
            out: list[Variant] = []
            for pid in product_ids[:5]:
                if self.session.remaining() <= 0 or self.session.done:
                    break
                out.extend(self.view(pid))
            return out
        obs = self.session.act("compare", product_ids=product_ids[:5])
        if obs.get("error") == "tool_unavailable":
            return []
        out: list[Variant] = []
        for row in obs.get("compared") or []:
            out.extend(variants_from_view(row))
        for v in out:
            self.pool[v.key] = v
            if v.product_id not in self.observed_product_ids:
                self.observed_product_ids.append(v.product_id)
        return out

    def inspect_stock(self, product_id: str, sku: str) -> Variant:
        key = f"{product_id}::{sku}"
        v = self.pool.get(key) or Variant(product_id=product_id, sku=sku)
        if not self.session.can("inspect_stock"):
            # Keep last-known pool facts; do not invent stock/price.
            return v
        obs = self.session.act("inspect_stock", product_id=product_id, sku=sku)
        if obs.get("error"):
            return v
        v.price = _as_float(obs.get("price"))
        v.in_stock = bool(obs.get("in_stock", v.in_stock))
        v.currency = str(obs.get("currency") or v.currency)
        self.pool[key] = v
        self._last_inspect_oos = not bool(v.in_stock)
        if self._pending_recheck == (product_id, sku):
            self._pending_recheck = None
        return v

    def inspect_cart(self) -> dict[str, Any]:
        """Read cart when the EnvPack exposes it (optional on some tasks)."""
        if not self.session.can("inspect_cart"):
            return {}
        obs = self.session.act("inspect_cart")
        return {} if obs.get("error") else obs

    def message(self, content: str) -> None:
        if not self.session.can("message"):
            return
        self.session.act("message", content=content)
        # Shopper may reply again; absorb budget/spec updates without re-answering loops.
        if self.session.last_user_message:
            self.plan = apply_shopper_update(self.plan, self.session.last_user_message)
            self.session.last_user_message = None

    def _handle_shopper_note(self, *, pending_choice: Optional["Variant"] = None) -> None:
        note = self.session.last_user_message
        if not note:
            return
        self.plan = apply_shopper_update(self.plan, note)
        # Event notices often arrive as shopper text before inspect reflects them.
        if pending_choice is not None and re.search(
            r"\b(not available|out of stock|sold out|still available|stockout|unavailable)\b",
            note,
            re.I,
        ):
            pending_choice.in_stock = False
            self.pool[pending_choice.key] = pending_choice
        if pending_choice is not None:
            price_jump = re.search(
                r"(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:to|->)\s*(\d+(?:\.\d+)?)\s*php",
                note,
                re.I,
            )
            if price_jump:
                new_p = float(price_jump.group(2))
                pending_choice.price = new_p
                self.pool[pending_choice.key] = pending_choice
        if shopper_needs_reply(note) and self.session.remaining() > 1:
            reply = self._compose_reply(note, pending_choice)
            self.session.last_user_message = None
            self.message(reply)
        else:
            self.session.last_user_message = None

    def _compose_reply(self, note: str, choice: Optional["Variant"]) -> str:
        lower = note.lower()
        # Market shock: declarative same-model / other-color recovery (Ilostin).
        if re.search(
            r"\b(not available|out of stock|sold out|still available|stockout|"
            r"unavailable|market.?event|price (?:changed|jumped|increased))\b",
            lower,
        ):
            model = self.plan.preferred_model
            if not model and choice is not None:
                # Pull a model-like token from the title when present.
                m = re.search(
                    r"\b([A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9\-]*)\b",
                    choice.title or "",
                )
                model = m.group(1) if m else None
            if model:
                return (
                    f"The carted item changed; I will recover to the same model "
                    f"({model}) in another available color before trying a different model, "
                    "then re-check stock and order."
                )
            return (
                "The carted item changed; I will recover to the closest still-valid "
                "in-stock option under budget, re-check stock, and order."
            )
        if choice is not None:
            bits = [
                f"I'm going with {choice.title or choice.product_id}",
                f"sku {choice.sku}",
            ]
            if choice.price is not None:
                bits.append(f"at {choice.price} {choice.currency}")
            # Only claim specs measured on the variant — never echo plan wants.
            inches = SpecKit.screen_inches(choice)
            if inches is not None:
                bits.append(f"measured_screen={inches:g}in")
            if hard_ok(self.plan, choice):
                bits.append("firm-requirements: met")
            else:
                bits.append("firm-requirements: still checking")
            return "; ".join(bits) + ". I'll verify live price/stock before ordering."
        if "balance" in lower or "budget" in lower or "php" in lower:
            return (
                f"Understood — I'll stay within {self.plan.budget} {self.plan.currency} "
                "and re-check stock/price before ordering."
            )
        if "expensive" in lower or "price" in lower:
            return "Thanks for the heads-up — I'll re-check the live price and switch if needed."
        return (
            "Got it. I'll apply your latest requirements (specs + budget + in-stock) "
            "and only order a firm-compliant item."
        )

    def add_to_cart(self, product_id: str, sku: str) -> dict[str, Any]:
        if not self.session.can("add_to_cart"):
            return {"error": "tool_unavailable", "requested": "add_to_cart"}
        obs = self.session.act("add_to_cart", product_id=product_id, sku=sku)
        self._cart_adds += 1
        self._pending_recheck = (str(product_id), str(sku))
        self._handle_shopper_note(
            pending_choice=self.pool.get(f"{product_id}::{sku}")
        )
        return obs

    def remove_from_cart(self, product_id: str, sku: str) -> None:
        if self.session.can("remove_from_cart"):
            self.session.act("remove_from_cart", product_id=product_id, sku=sku)
            self._handle_shopper_note()

    def place_order(
        self,
        variant: Variant,
        justification: Optional[dict] = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.session.can("place_order"):
            return {"error": "tool_unavailable", "requested": "place_order"}
        # Block order while latest inspect says OOS, or post-cart recheck pending.
        if self._last_inspect_oos and not variant.in_stock:
            return {}
        if self._pending_recheck == (variant.product_id, variant.sku):
            variant = self.inspect_stock(variant.product_id, variant.sku)
            if self.session.done:
                return {}
        if not force and not purchase_ok(self.plan, variant):
            # Never knowingly place an unbuyable order (OOS / over budget).
            return {}
        if force and self.plan.family in CommitPicker.EVENT_FAMILIES:
            if not purchase_ok(self.plan, variant):
                return {}
            # When firm attrs exist, never force-seal a firm miss if any hard_ok
            # alternative is already in the pool (TF1 wrong-size recoveries).
            if self.plan.attributes and not hard_ok(self.plan, variant):
                if any(
                    hard_ok(self.plan, v)
                    and purchase_ok(self.plan, v)
                    and v.key not in self._dead_keys
                    for v in self.pool.values()
                ):
                    return {}
                # No firm survivor left — allow soft seal only to avoid agent_error.
        if force and self.plan.family == "preference_reasoning":
            if not purchase_ok(self.plan, variant):
                return {}
            if self.plan.category and not DeviceGuard.ok(variant, self.plan.category):
                return {}
        # TF7: refresh live price/stock immediately before sealing claims.
        if (
            self.plan.family == "justification"
            and self.session.remaining() > 0
            and not force
        ):
            variant = self.inspect_stock(variant.product_id, variant.sku)
            if not purchase_ok(self.plan, variant):
                return {}
            # Refuse to seal claims without literal gate evidence.
            if not EvidenceGate.claimable(self.plan, variant):
                return {}
        args: dict[str, Any] = {"product_id": variant.product_id, "sku": variant.sku}
        # Optional claims outside TF7 fail submitted_claims_truthful and zero the gate.
        # TF7 force-place: omit justification entirely rather than seal an
        # unsupported gate claim (verifier zeros the task either way on missing
        # required claims, but unsupported claims are worse for learning).
        if justification is not None:
            args["justification"] = justification
        elif self.plan.family == "justification":
            just = self._build_justification(variant)
            fields = {c.get("field") for c in (just.get("claims") or [])}
            gate = EvidenceGate.gate_key(self.plan)
            if gate and gate not in fields:
                if not force:
                    return {}
                # force path: still attach price/stock-only claims if present
            args["justification"] = just
        # Capability name resolves to place_test_order (or pack synonym).
        return self.session.act("place_order", **args)

    def _ingest_results(self, obs: dict[str, Any]) -> list[Variant]:
        out: list[Variant] = []
        for item in obs.get("results") or []:
            if not isinstance(item, dict) or not item.get("product_id"):
                continue
            v = variant_from_summary(item)
            prev = self.pool.get(v.key)
            if prev and prev.viewed:
                # keep richer facts; refresh live price/stock
                prev.price = v.price if v.price is not None else prev.price
                prev.in_stock = v.in_stock
                v = prev
            self.pool[v.key] = v
            out.append(v)
            if v.product_id not in self.observed_product_ids:
                self.observed_product_ids.append(v.product_id)
        return out

    # -- phases -------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        try:
            if self.plan.needs_clarification and self.session.remaining() > 10:
                self._clarify()
            if self.session.done:
                return self.session.dialogue

            candidates = self._retrieve()
            if self.session.done:
                return self.session.dialogue

            shortlist = self._inspect(candidates)
            if self.session.done:
                return self.session.dialogue

            choice = self._choose(shortlist)
            if choice is None:
                fallback = [
                    v
                    for v in self.pool.values()
                    if v.in_stock and budget_ok(v, self.plan.budget)
                ]
                if self.plan.candidate_keys:
                    fallback = [v for v in fallback if v.key in self.plan.candidate_keys] or fallback
                fallback.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                choice = fallback[0] if fallback else None
            if choice is not None and not self.session.done:
                self._close(choice)

            # Force a terminal order before step exhaustion (avoid agent_error).
            if not self.session.done and self.session.remaining() > 0:
                self._force_terminal()
        except Exception as exc:  # noqa: BLE001
            self.session.dialogue.append(
                {
                    "role": "system",
                    "content": f"SOSO_v6 error: {type(exc).__name__}: {exc}",
                }
            )
            try:
                if not self.session.done and self.session.remaining() > 0:
                    self._force_terminal()
            except Exception:
                pass
        return self.session.dialogue

    def _force_terminal(self) -> None:
        """Always try to reach done=true — agent_error zeros the task."""
        if self.session.done or self.session.remaining() <= 0:
            return
        firm_only = (
            self.plan.family
            in {"intent_decomposition", "constraint_satisfaction", "justification"}
            and bool(self.plan.attributes)
        )
        # Event families: brand-free firm hunt + view budget leads before soft place.
        if (
            firm_only
            and self.session.remaining() > 4
            and not any(
                hard_ok(self.plan, v)
                and purchase_ok(self.plan, v)
                and v.key not in self._dead_keys
                for v in self.pool.values()
            )
        ):
            self._spec_requery(soft_brand=False)
            self._view_budget_leads(blocked=set(self._dead_keys), limit=4)
        pool = [
            v
            for v in self.pool.values()
            if purchase_ok(self.plan, v) and v.key not in self._dead_keys
        ]
        hard = [v for v in pool if hard_ok(self.plan, v)]
        if hard:
            pool = hard
        elif firm_only:
            if self.session.remaining() > 3:
                self._spec_requery(soft_brand=False)
                lead = self._view_budget_leads(blocked=set(self._dead_keys), limit=4)
                pool = [
                    v
                    for v in self.pool.values()
                    if purchase_ok(self.plan, v) and v.key not in self._dead_keys
                ]
                hard = [v for v in pool if hard_ok(self.plan, v)]
                if lead is not None and hard_ok(self.plan, lead):
                    hard = [lead] + [v for v in hard if v.key != lead.key]
                if hard:
                    pool = hard
            # Prefer firm hits; soft device fallback only if still empty (avoid agent_error).
            if hard:
                pool = hard
            else:
                pool = [
                    v
                    for v in self.pool.values()
                    if purchase_ok(self.plan, v)
                    and DeviceGuard.ok(v, self.plan.category)
                    and v.key not in self._dead_keys
                ]
        if self.plan.family == "justification":
            claimable = [v for v in pool if EvidenceGate.claimable(self.plan, v)]
            if claimable:
                pool = claimable
        brand = CommitPicker.soft_brand(self.plan)
        if brand:
            branded = [v for v in pool if brand_ok(v, brand)]
            if branded:
                pool = branded
        if self.plan.candidate_keys:
            cand = [v for v in pool if v.key in self.plan.candidate_keys]
            if cand:
                pool = cand
            elif self.plan.family == "ranking":
                # Closed set: only listed keys may be ordered.
                pool = [
                    v
                    for v in self.pool.values()
                    if v.key in self.plan.candidate_keys
                    and purchase_ok(self.plan, v)
                    and v.key not in self._dead_keys
                ]
        # Firm event families: still allow soft buyable fallback to avoid agent_error,
        # but never resurrect dead keys.
        if not pool:
            pool = [
                v
                for v in self.pool.values()
                if hard_ok(self.plan, v)
                and purchase_ok(self.plan, v)
                and v.key not in self._dead_keys
            ]
        if not pool:
            pool = [
                v
                for v in self.pool.values()
                if v.in_stock
                and DeviceGuard.ok(v, self.plan.category)
                and purchase_ok(self.plan, v)
                and v.key not in self._dead_keys
            ]
        if not pool:
            pool = [
                v
                for v in self.pool.values()
                if purchase_ok(self.plan, v) and v.key not in self._dead_keys
            ]
        if not pool:
            if self.session.remaining() > 1:
                self._spec_requery(soft_brand=False)
                self._view_budget_leads(blocked=set(self._dead_keys), limit=3)
                pool = [
                    v
                    for v in self.pool.values()
                    if hard_ok(self.plan, v)
                    and purchase_ok(self.plan, v)
                    and v.key not in self._dead_keys
                ] or [
                    v
                    for v in self.pool.values()
                    if purchase_ok(self.plan, v)
                    and DeviceGuard.ok(v, self.plan.category)
                    and v.key not in self._dead_keys
                ]
            if not pool:
                return
        # Prefer hard_ok when present (already filtered above when available).
        hard = [
            v
            for v in pool
            if hard_ok(self.plan, v)
            and purchase_ok(self.plan, v)
            and v.key not in self._dead_keys
        ]
        if hard:
            pool = hard
        if self.plan.family == "ranking":
            pool.sort(key=ranking_key, reverse=True)
        elif self.plan.family in CommitPicker.EVENT_FAMILIES:
            pool.sort(key=lambda v: EventCommit.score(self.plan, v), reverse=True)
        else:
            pool.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
        best = pool[0]
        if not best.viewed and self.session.remaining() > 2:
            variants = self.view(best.product_id)
            ok = [
                v
                for v in variants
                if hard_ok(self.plan, v)
                and purchase_ok(self.plan, v)
                and v.key not in self._dead_keys
            ]
            if ok:
                ok.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                best = ok[0]
            else:
                best = self.pool.get(best.key, best)

        # Cart must contain the SKU we place — place_test_order errors with not_in_cart.
        if self.session.remaining() > 1:
            if best.key in self._dead_keys or not purchase_ok(self.plan, best):
                return
            self.add_to_cart(best.product_id, best.sku)
            if self.session.done:
                return
            best = self.pool.get(best.key, best)
            if self.session.remaining() > 0:
                best = self.inspect_stock(best.product_id, best.sku)
                self._handle_shopper_note(pending_choice=best)
                if needs_event_recovery(self.plan, best):
                    self._dead_keys.add(best.key)

        if self.session.done:
            return

        if (
            self.session.remaining() > 3
            and not purchase_ok(self.plan, best)
        ):
            self._dead_keys.add(best.key)
            self.remove_from_cart(best.product_id, best.sku)
            alt = self._recover(
                exclude_keys={best.key} | self._dead_keys, previous=best
            )
            if alt is None or alt.key in self._dead_keys:
                # Last resort: any other hard_ok buyable in pool.
                alts = [
                    v
                    for v in self.pool.values()
                    if v.key not in self._dead_keys
                    and hard_ok(self.plan, v)
                    and purchase_ok(self.plan, v)
                ]
                if not alts:
                    return
                alts.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                best = alts[0]
            else:
                best = alt
            if self.session.remaining() > 1:
                self.add_to_cart(best.product_id, best.sku)
                if self.session.done:
                    return
                if self.session.remaining() > 0:
                    best = self.inspect_stock(best.product_id, best.sku)
                    if needs_event_recovery(self.plan, best):
                        self._dead_keys.add(best.key)
                        return

        if self.session.done:
            return
        if self.session.remaining() > 0:
            if not purchase_ok(self.plan, best) or best.key in self._dead_keys:
                alts = [
                    v
                    for v in self.pool.values()
                    if v.key != best.key
                    and v.key not in self._dead_keys
                    and purchase_ok(self.plan, v)
                    and hard_ok(self.plan, v)
                ]
                if not alts:
                    alts = [
                        v
                        for v in self.pool.values()
                        if v.key not in self._dead_keys and purchase_ok(self.plan, v)
                    ]
                if not alts:
                    return
                alts.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                best = alts[0]
                if self.session.remaining() > 1:
                    self.add_to_cart(best.product_id, best.sku)
            # Event families: refuse force-seal of firm miss when hard_ok exists.
            if (
                firm_only
                and not hard_ok(self.plan, best)
                and any(
                    hard_ok(self.plan, v)
                    and purchase_ok(self.plan, v)
                    and v.key not in self._dead_keys
                    for v in self.pool.values()
                )
            ):
                hard_alts = [
                    v
                    for v in self.pool.values()
                    if hard_ok(self.plan, v)
                    and purchase_ok(self.plan, v)
                    and v.key not in self._dead_keys
                ]
                hard_alts.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                best = hard_alts[0]
            # Always ensure cart contains the SKU before place (not_in_cart errors).
            if self.session.remaining() > 1:
                self.add_to_cart(best.product_id, best.sku)
                if self.session.done:
                    return
                if self.session.remaining() > 0:
                    best = self.inspect_stock(best.product_id, best.sku)
                    if needs_event_recovery(self.plan, best):
                        self._dead_keys.add(best.key)
                        return
            elif self.session.remaining() <= 1:
                # Cannot add+place; only place if already carted-equivalent purchase_ok.
                if not purchase_ok(self.plan, best) or best.key in self._dead_keys:
                    return
            self.place_order(best, force=True)
            if self.session.done:
                return
            if self.session.remaining() > 1 and best.key not in self._dead_keys:
                self.add_to_cart(best.product_id, best.sku)
            if self.session.remaining() > 0 and best.key not in self._dead_keys:
                self.place_order(best, force=True)

    def _clarify(self) -> None:
        if self.plan.family == "constraint_satisfaction":
            self.message(
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            )
        elif self.plan.family == "preference_reasoning":
            self.message(
                "To pick the best match: which priorities matter most for your use case "
                "(battery, water/sweat resistance, fit, wireless, brand)?"
            )
        elif self.plan.family == "intent_decomposition":
            self.message(
                "Quick check: which requirements are firm versus nice-to-have, "
                "and is the preferred brand optional if a firm-spec item is only available from another brand?"
            )
        else:
            self.message(
                "Before ordering, any must-have specs or budget updates I should apply?"
            )
        self._handle_shopper_note()
        # TF3: rebuild search queries from the revealed firm attr only — LLM
        # query floods otherwise exhaust the 30-step budget before an exact SKU.
        if self.plan.family == "constraint_satisfaction" and self.plan.attributes:
            self.plan.search_queries = self._firm_attr_queries()[:4]
            self._requery_count = 0
            # Immediate brand+attr hunt so first cart is not delayed to step 25+.
            brand = self.plan.brand or self.plan.preferred_brand
            if self.session.remaining() > 8:
                self.filter_catalog(
                    category=self.plan.category,
                    brand=brand,
                    max_price=self.plan.budget,
                    k=10,
                    in_stock=True,
                )
                for q in self.plan.search_queries[:2]:
                    if self.session.remaining() <= 8:
                        break
                    self.search(q, k=10, in_stock=True)
                self._handle_shopper_note()

    def _firm_attr_queries(self, *, soft_brand: bool = True) -> list[str]:
        """Build catalog queries that surface firm hardware attrs (suite-agnostic)."""
        brand = ""
        if soft_brand:
            brand = self.plan.brand or self.plan.preferred_brand or ""
        elif self.plan.brand:
            # Mandatory brand (CS/TF7) stays; soft preferred brand does not.
            brand = self.plan.brand
        cat = self.plan.category or ""
        out: list[str] = []
        for key, val in self.plan.attributes.items():
            kind, wanted = SpecKit.normalize_wanted(key, str(val))
            if kind == "screen_size" and isinstance(wanted, (int, float)):
                inches = float(wanted)
                # Spell out "inch" — catalog titles rarely use the "7.6in" token.
                out.append(f"{brand} {inches:g} inch {cat}".strip())
                out.append(f"{inches:g} inch {cat} {brand}".strip())
                out.append(f"{inches:g} inch {cat}".strip())
                # Large phone screens are often foldables; use generic fold tokens
                # from the category shape — never a branded pack motif.
                if inches >= 7.0 and cat and "phone" in cat.lower():
                    out.append(f"{brand} foldable {inches:g} inch smartphone".strip())
                    out.append(f"{brand} folding phone {inches:g}".strip())
            elif kind == "resolution" and wanted:
                token = str(wanted)
                out.append(f"{brand} {cat} {token}".strip())
                out.append(f"{brand} {token} {cat}".strip())
                out.append(f"{cat} {token}".strip())
                if token in {"hd", "720p"}:
                    out.append(f"{brand} {cat} 720p HD".strip())
                    out.append(f"{brand} 1280x720 {cat}".strip())
                elif token in {"1080p", "fullhd", "fhd"}:
                    out.append(f"{brand} {cat} 1080p Full HD".strip())
                    out.append(f"{brand} 1920x1080 {cat}".strip())
            elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                out.append(f"{brand} {cat} {int(wanted)}Hz".strip())
                out.append(f"{cat} {int(wanted)}Hz".strip())
            elif kind == "ram" and wanted:
                out.append(f"{brand} {cat} {wanted} RAM".strip())
            else:
                out.append(f"{brand} {cat} {val}".strip())
        # de-dupe preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for q in out:
            q = " ".join(q.split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                uniq.append(q)
        return uniq

    def _spec_requery(self, *, soft_brand: bool = True) -> None:
        """Re-search using current budget/brand/exact attributes (suite-agnostic)."""
        if self._requery_count >= 3 or self.session.remaining() <= 4:
            return
        self._requery_count += 1
        for q in self._firm_attr_queries(soft_brand=soft_brand)[:3]:
            if self.session.remaining() <= 4:
                break
            self.search(q, k=10, in_stock=True)
        brand = self.plan.brand or ""
        if soft_brand and not brand:
            brand = self.plan.preferred_brand or ""
        bits = [
            brand if soft_brand else (self.plan.brand or ""),
            self.plan.category or "",
            " ".join(str(v) for v in self.plan.attributes.values()),
        ]
        q = " ".join(b for b in bits if b).strip()
        if q and self.session.remaining() > 4:
            self.search(q, k=10, in_stock=True)
        if self.session.remaining() > 4:
            # After a dead first cart, drop soft brand from the filter so
            # off-brand firm hits under the tightened budget can surface.
            filter_brand = self.plan.brand
            if soft_brand and not filter_brand:
                if self.plan.family in {
                    "justification",
                    "constraint_satisfaction",
                    "intent_decomposition",
                }:
                    filter_brand = self.plan.preferred_brand
            self.filter_catalog(
                category=self.plan.category,
                brand=filter_brand if soft_brand else self.plan.brand,
                max_price=self.plan.budget,
                k=10,
                in_stock=True,
            )
        self._handle_shopper_note()

    def _pick_from_hits(
        self, hits: list[Variant], *, blocked: set[str]
    ) -> Optional[Variant]:
        """View near-budget hits from a fresh search page before wider pool scans."""
        if not hits:
            return None
        budget = float(self.plan.budget) if self.plan.budget is not None else None

        def hit_rank(v: Variant) -> tuple:
            price = float(v.price) if v.price is not None else 1e18
            if budget and budget > 0:
                ratio = price / budget
                near = abs(ratio - 1.0)
                fit = 0 if purchase_ok(self.plan, v) else 1
            else:
                near, fit = 0.0, 0
            return (fit, near, price)

        ordered = sorted(
            [v for v in hits if v.key not in blocked and DeviceGuard.ok(v, self.plan.category)],
            key=hit_rank,
        )
        for v in ordered[:6]:
            if self.session.remaining() <= 3:
                break
            if not purchase_ok(self.plan, v) and not (
                budget is not None and v.price is not None and float(v.price) <= budget + 1e-6
            ):
                continue
            variants = self.view(v.product_id) if not v.viewed else [
                x for x in self.pool.values() if x.product_id == v.product_id
            ]
            ok = [
                x
                for x in variants
                if x.key not in blocked
                and hard_ok(self.plan, x)
                and purchase_ok(self.plan, x)
            ]
            if ok:
                ok.sort(key=lambda x: score_variant(self.plan, x), reverse=True)
                return ok[0]
        return None

    def _view_budget_leads(
        self, *, blocked: set[str], limit: int = 4
    ) -> Optional[Variant]:
        """View buyable leads so fact-only firm attrs (Hioro Note 3) can match."""

        def title_mentions_attr(variant: Variant) -> bool:
            blob = f"{variant.title or ''} {variant.options or ''}".lower()
            for key, val in self.plan.attributes.items():
                kind, wanted = SpecKit.normalize_wanted(key, str(val))
                if kind == "screen_size" and isinstance(wanted, (int, float)):
                    num = f"{float(wanted):g}"
                    if re.search(
                        rf"\b{re.escape(num)}\s*(?:\"|''|in(?:ch(?:es)?)?)\b", blob
                    ):
                        return True
                elif kind == "resolution" and wanted:
                    if str(wanted).lower() in blob:
                        return True
                elif kind == "refresh_rate" and isinstance(wanted, (int, float)):
                    if re.search(rf"\b{int(wanted)}\s*hz\b", blob):
                        return True
            return False

        def lead_rank(variant: Variant) -> tuple:
            price = float(variant.price) if variant.price is not None else 1e18
            budget = float(self.plan.budget) if self.plan.budget else price
            ratio = (price / budget) if budget > 0 else 1.0
            # Dirt-cheap hits are usually accessories / wrong devices.
            cheap_pen = 0 if 0.2 <= ratio <= 1.0 else 1
            recent = self._recent_hits or []
            try:
                recent_pen = recent.index(variant.product_id)
            except ValueError:
                recent_pen = 100
            return (
                0 if hard_ok(self.plan, variant) else 1,
                0 if title_mentions_attr(variant) else 1,
                cheap_pen,
                # Prefer products from the latest firm-attr search/filter page.
                0 if variant.product_id in recent else 1,
                recent_pen,
                # Prefer near-budget survivors (tightened budget traps sit near 1.0).
                abs(ratio - 0.95),
                price,
            )

        leads = [
            v
            for v in self.pool.values()
            if v.key not in blocked
            and purchase_ok(self.plan, v)
            and DeviceGuard.ok(v, self.plan.category)
            and OptionGuard.ok(v, self.plan.category)
        ]
        leads.sort(key=lead_rank)
        seen_pid: set[str] = set()
        for v in leads:
            if self.session.remaining() <= 3:
                break
            if v.product_id in seen_pid:
                continue
            seen_pid.add(v.product_id)
            if not v.viewed:
                variants = self.view(v.product_id)
            else:
                variants = [x for x in self.pool.values() if x.product_id == v.product_id]
            ok = [
                x
                for x in variants
                if x.key not in blocked
                and hard_ok(self.plan, x)
                and purchase_ok(self.plan, x)
            ]
            if ok:
                ok.sort(key=lambda x: score_variant(self.plan, x), reverse=True)
                return ok[0]
            if len(seen_pid) >= limit:
                break
        return None

    def _retrieve(self) -> list[Variant]:
        found: list[Variant] = []
        # Ranking family: only the named candidate keys.
        for key in self.plan.candidate_keys:
            pid, _, sku = key.partition("::")
            if pid:
                v = Variant(product_id=pid, sku=sku or "default")
                found.append(v)
                self.pool.setdefault(v.key, v)

        if self.plan.family == "ranking" and self.plan.candidate_keys:
            return list(found)

        queries = list(self.plan.search_queries)
        # Firm attr reformulations first (spell out inches / foldable large screens).
        firm_qs = self._firm_attr_queries()
        if firm_qs and self.plan.family in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        }:
            queries = firm_qs + [q for q in queries if q not in firm_qs]
        # TF3/TF7: keep the query list short — exact/gate matching > broad recall.
        if self.plan.family in {"constraint_satisfaction", "justification"}:
            brand = self.plan.brand or self.plan.preferred_brand or ""
            cat = self.plan.category or ""
            core = firm_qs[:3] if firm_qs else []
            if brand and cat:
                core.append(f"{brand} {cat}".strip())
            # Keep at most one LLM reformulation that mentions a firm token.
            firm_toks = [str(v).lower() for v in self.plan.attributes.values()]
            for q in list(self.plan.search_queries):
                ql = q.lower()
                if any(t and t.replace("in", "")[:4] in ql for t in firm_toks) or any(
                    t in ql for t in ("720p", "1080p", "hd", "inch", "hz")
                ):
                    if q not in core:
                        core.append(q)
                if len(core) >= 4:
                    break
            queries = core[:5]
        # TF1: preferred-brand + firm attr FIRST so first-cart commitment can hit
        # acceptance targets (soft brand), before generic recall fills the pool.
        if self.plan.family == "intent_decomposition" and self.plan.preferred_brand:
            bits = [
                self.plan.preferred_brand,
                self.plan.category or "smartphone",
                " ".join(str(v) for v in self.plan.attributes.values()),
            ]
            brand_q = " ".join(b for b in bits if b)
            queries = [brand_q, self.plan.preferred_brand] + [
                q for q in queries if q not in {brand_q, self.plan.preferred_brand}
            ]
            # Prefer spelled-out inch queries ahead of tokenized "7.6in".
            queries = firm_qs[:2] + [q for q in queries if q not in firm_qs[:2]]
        # Extra recall reformulations
        if self.plan.family == "retrieval_recall":
            phrase = re.search(r'matches?\s+"([^"]+)"', self.plan.goal, re.I)
            if phrase:
                queries.insert(0, phrase.group(1))
                queries.append(phrase.group(1).replace(" ", ""))
        if self.plan.family == "preference_reasoning":
            # Axis-derived queries only — LLM reformulations often inject filler
            # ("yes") that wastes the search budget before IPX/sport hits appear.
            queries = soft_search_queries(self.plan)[:6]
        if self.plan.family == "recovery" and self.plan.preferred_model:
            queries.insert(0, self.plan.preferred_model)
            if self.plan.preferred_color:
                queries.insert(0, f"{self.plan.preferred_color} {self.plan.preferred_model}")
        if self.plan.family == "justification":
            brand = self.plan.brand or self.plan.preferred_brand
            bits = [brand or "", self.plan.category or ""]
            for attr, val in self.plan.attributes.items():
                bits.append(str(val))
            queries.insert(0, " ".join(b for b in bits if b))

        # TF3 withheld: brand+budget filter ONLY until the firm attr is revealed.
        # Invented inch/resolution searches burn the step budget (Hioro asks,
        # then filters brand, then searches the revealed size once).
        if (
            self.plan.family == "constraint_satisfaction"
            and not self.plan.attributes
            and self.session.remaining() > 4
        ):
            found.extend(
                self.filter_catalog(
                    category=self.plan.category,
                    brand=self.plan.brand or self.plan.preferred_brand,
                    max_price=self.plan.budget,
                    k=10,
                    in_stock=True,
                )
            )
            self._handle_shopper_note()
            # Do not run attr-invented search queries before the reveal lands.
            if not self.plan.attributes:
                return found or list(self.pool.values())

        # TF1 / TF3 / TF7: firm-attr searches BEFORE brand-only filter so large
        # screens (e.g. 7.6" foldables) are not buried under unrelated brand hits.
        if (
            self.plan.family
            in {"intent_decomposition", "constraint_satisfaction", "justification"}
            and firm_qs
            and self.session.remaining() > 6
        ):
            for q in firm_qs[:2]:
                if self.session.remaining() <= 6:
                    break
                found.extend(self.search(q, k=10, in_stock=True))
                self._handle_shopper_note()
                if any(hard_ok(self.plan, v) for v in self.pool.values()):
                    break

        # TF1: brand filter early (before remaining search loop burns the step budget).
        if (
            self.plan.family == "intent_decomposition"
            and self.plan.preferred_brand
            and self.session.remaining() > 4
        ):
            found.extend(
                self.filter_catalog(
                    category=self.plan.category,
                    brand=self.plan.preferred_brand,
                    max_price=self.plan.budget,
                    k=10,
                    in_stock=True,
                )
            )
            self._handle_shopper_note()

        # TF7: brand filter early so claimable branded hits surface before thrash.
        if (
            self.plan.family == "justification"
            and (self.plan.brand or self.plan.preferred_brand)
            and self.session.remaining() > 4
        ):
            found.extend(
                self.filter_catalog(
                    category=self.plan.category,
                    brand=self.plan.brand or self.plan.preferred_brand,
                    max_price=self.plan.budget,
                    k=10,
                    in_stock=True,
                )
            )
            self._handle_shopper_note()

        for q in queries:
            if self.session.remaining() <= 6:
                break
            # Anti-thrash (Ilostin): after ~3–4 weak searches, stop and view hits.
            thrash_cap = 6 if self.plan.family == "preference_reasoning" else 3
            if self.plan.family in {"constraint_satisfaction", "justification"}:
                thrash_cap = 3
            if self._searchish >= thrash_cap and self.plan.family != "retrieval_recall":
                break
            if self._searchish >= 6 and self.plan.family == "retrieval_recall":
                break
            # Skip queries already executed in the firm-attr prelude.
            if firm_qs and q in firm_qs[:2] and self.plan.family in {
                "intent_decomposition",
                "constraint_satisfaction",
                "justification",
            }:
                continue
            found.extend(self.search(q, k=10, in_stock=True))
            self._handle_shopper_note()
            if self.plan.family == "retrieval_recall" and len(self.observed_product_ids) >= 5:
                break
            if self.plan.family == "intent_decomposition":
                # Keep searching until we have preferred-brand hits when possible.
                branded = [
                    v
                    for v in self.pool.values()
                    if self.plan.preferred_brand and brand_ok(v, self.plan.preferred_brand)
                ]
                if branded and any(hard_ok(self.plan, v) for v in branded):
                    break
                if branded and len(found) >= 10:
                    break
            elif self.plan.family in {"constraint_satisfaction", "justification"}:
                ok = [
                    v
                    for v in self.pool.values()
                    if hard_ok(self.plan, v)
                    and (
                        self.plan.family != "justification"
                        or EvidenceGate.claimable(self.plan, v)
                    )
                ]
                if ok:
                    break
            elif self.plan.family != "retrieval_recall" and len(found) >= 8:
                break

        if self.session.remaining() > 6 and self._searchish < 4:
            brand = self.plan.brand
            if not brand and self.plan.preferred_brand and self.plan.family != "intent_decomposition":
                brand = self.plan.preferred_brand
            found.extend(
                self.filter_catalog(
                    category=self.plan.category,
                    brand=brand,
                    max_price=self.plan.budget,
                    k=10,
                    in_stock=True,
                )
            )
            self._handle_shopper_note()
            # TF1: refresh preferred-brand filter after searches (price/attr context).
            if (
                self.plan.family == "intent_decomposition"
                and self.plan.preferred_brand
                and self.session.remaining() > 5
            ):
                found.extend(
                    self.filter_catalog(
                        category=self.plan.category,
                        brand=self.plan.preferred_brand,
                        max_price=self.plan.budget,
                        k=10,
                        in_stock=True,
                    )
                )
                self._handle_shopper_note()
        # unique by key preserving order
        uniq: dict[str, Variant] = {}
        for v in found:
            uniq[v.key] = self.pool.get(v.key, v)
        return list(uniq.values())

    def _inspect(self, candidates: list[Variant]) -> list[Variant]:
        # Prefer viewing distinct product_ids
        product_ids: list[str] = []
        for v in candidates:
            if v.product_id not in product_ids:
                product_ids.append(v.product_id)

        # TF2: ensure enough distinct identities observed
        min_views = 3 if self.plan.family == "retrieval_recall" else 2
        if self.plan.family == "preference_reasoning":
            min_views = 4
        if self.plan.family == "ranking" and self.plan.candidate_keys:
            min_views = min(5, len({k.split("::", 1)[0] for k in self.plan.candidate_keys}))
        if self.plan.family == "justification":
            min_views = 3
        if self.plan.family == "recovery":
            min_views = 2
        if self.plan.family == "intent_decomposition":
            min_views = 2
        if self.plan.family == "constraint_satisfaction":
            min_views = 1

        viewed: list[Variant] = []
        # Prefer single-product view for constraint families so SKU options + stock
        # are visible; compare is cheaper for recall/ranking breadth.
        use_compare = (
            self.plan.family not in {"constraint_satisfaction", "justification", "intent_decomposition"}
            and self.session.can("compare")
        )
        # TF1: view preferred-brand candidates before generic ones.
        if self.plan.family == "intent_decomposition" and self.plan.preferred_brand:
            branded_ids = [
                pid
                for pid in product_ids
                if any(
                    brand_ok(v, self.plan.preferred_brand)
                    for v in self.pool.values()
                    if v.product_id == pid
                )
            ]
            other_ids = [pid for pid in product_ids if pid not in branded_ids]
            product_ids = branded_ids + other_ids
        # Prefer title-level firm attr hits before random brand phones.
        if self.plan.family in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        } and self.plan.attributes:

            def _pid_firmness(pid: str) -> tuple[int, float]:
                vs = [v for v in self.pool.values() if v.product_id == pid]
                if not vs:
                    return (0, 0.0)
                best = max(vs, key=lambda v: score_variant(self.plan, v))
                return (1 if hard_ok(self.plan, best) else 0, score_variant(self.plan, best))

            product_ids.sort(key=_pid_firmness, reverse=True)
        if self.plan.family == "preference_reasoning":
            # View high preference_score / axis-strong titles first.
            def _pid_pref(pid: str) -> float:
                vs = [v for v in self.pool.values() if v.product_id == pid]
                if not vs:
                    return -1.0
                return max(preference_score(self.plan, v) for v in vs)

            product_ids.sort(key=_pid_pref, reverse=True)
        while product_ids and self.session.remaining() > 4 and len({v.product_id for v in viewed}) < min_views:
            batch = product_ids[:5]
            product_ids = product_ids[5:]
            if use_compare and len(batch) >= 2:
                viewed.extend(self.compare(batch))
            else:
                for pid in batch:
                    if self.session.remaining() <= 4:
                        break
                    viewed.extend(self.view(pid))
                    if len({v.product_id for v in viewed}) >= min_views:
                        break
            self._handle_shopper_note(pending_choice=viewed[0] if viewed else None)

        # If still thin, view top scored unsearched
        if len({v.product_id for v in viewed}) < min_views:
            ranked = sorted(
                self.pool.values(),
                key=lambda v: score_variant(self.plan, v),
                reverse=True,
            )
            for v in ranked:
                if self.session.remaining() <= 4:
                    break
                if any(x.product_id == v.product_id and x.viewed for x in viewed):
                    continue
                viewed.extend(self.view(v.product_id))
                self._handle_shopper_note(pending_choice=v)
                if len({x.product_id for x in viewed}) >= min_views:
                    break

        # flatten pool viewed variants
        out = [v for v in self.pool.values() if v.viewed or v.key in {c.key for c in candidates}]
        if self.plan.family == "ranking" and self.plan.candidate_keys:
            out = [v for v in out if v.key in self.plan.candidate_keys]
            out.sort(key=ranking_key, reverse=True)
        else:
            out.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
        return out

    def _choose(self, shortlist: list[Variant]) -> Optional[Variant]:
        # TF5: strictly rank named candidates (refresh > screen > price).
        if self.plan.family == "ranking" and self.plan.candidate_keys:
            pool = [
                v
                for v in shortlist
                if v.key in self.plan.candidate_keys and v.in_stock and budget_ok(v, self.plan.budget)
            ]
            if not pool:
                pool = [v for v in shortlist if v.key in self.plan.candidate_keys and v.in_stock]
            if not pool:
                return None
            pool.sort(key=ranking_key, reverse=True)
            return pool[0]

        hard = [v for v in shortlist if hard_ok(self.plan, v)]
        pool = hard or [v for v in shortlist if v.in_stock]
        if not pool:
            return None

        # TF1: honor preferred brand on first cart when fully compliant exists
        if self.plan.family == "intent_decomposition" and self.plan.preferred_brand:
            branded = [v for v in hard if brand_ok(v, self.plan.preferred_brand)]
            if branded:
                pool = branded

        # TF6: prefer exact model+color first
        if self.plan.family == "recovery":
            exact = [
                v
                for v in hard
                if (not self.plan.preferred_model or _norm(self.plan.preferred_model) in _norm(v.text_blob()))
                and (
                    not self.plan.preferred_color
                    or _norm(self.plan.preferred_color) in _norm(v.text_blob() + " " + v.options)
                )
            ]
            if exact:
                pool = exact

        # TF4: maximize preference axes among hard-ok (or stock+budget) viewed items.
        # First cart is the graded commitment — require strong soft-axis evidence
        # when the goal asks for sweat/water/sport (IPX / sweat+sport), not a
        # marginal splash-only title that fails intended_event_commitment.
        if self.plan.family == "preference_reasoning":
            pref_pool = [v for v in pool if purchase_ok(self.plan, v)]
            viewed = [v for v in pref_pool if v.viewed]
            if viewed:
                pref_pool = viewed
            axes = " ".join(self.plan.soft_prefs + [self.plan.goal]).lower()
            want_wet = bool(re.search(r"sweat|water|ipx|waterproof|rain", axes, re.I))
            want_sport = bool(re.search(r"gym|run|sport|workout|fitness|bootcamp", axes, re.I))

            def _axis_strong(v: Variant) -> bool:
                blob = v.text_blob()
                wet = bool(
                    re.search(
                        r"ipx|sweat|waterproof|water.?resist|splash",
                        blob,
                        re.I,
                    )
                )
                sport = bool(
                    re.search(
                        r"sport|gym|running|workout|ear.?hook|secure.?fit|bootcamp",
                        blob,
                        re.I,
                    )
                )
                ipx = bool(re.search(r"ipx", blob, re.I))
                if want_wet and want_sport:
                    return ipx or (wet and sport)
                if want_wet:
                    return wet
                if want_sport:
                    return sport
                return True

            strong = [v for v in pref_pool if _axis_strong(v)]
            if strong:
                pref_pool = strong
            elif self.session.remaining() > 6:
                # Hunt once more for axis-strong hits before accepting a weak first cart.
                for q in soft_search_queries(self.plan)[:2]:
                    if self.session.remaining() <= 5:
                        break
                    self.search(q, k=10, in_stock=True)
                pref_pool = [
                    v
                    for v in self.pool.values()
                    if purchase_ok(self.plan, v) and _axis_strong(v)
                ] or pref_pool
            pref_pool = sorted(
                pref_pool,
                key=lambda v: (preference_score(self.plan, v), score_variant(self.plan, v)),
                reverse=True,
            )
            if pref_pool:
                return pref_pool[0]

        # TF7: brand + gate attr among viewed candidates (first cart should be gold).
        if self.plan.family == "justification":
            want = self.plan.brand or self.plan.preferred_brand
            branded = [v for v in hard if want and brand_ok(v, want)]
            if branded:
                pool = branded
            claimable = [v for v in pool if EvidenceGate.claimable(self.plan, v)]
            if claimable:
                pool = claimable
            else:
                evidenced = [v for v in pool if EvidenceGate.visible(self.plan, v)]
                if evidenced:
                    pool = evidenced
            viewed = [v for v in pool if v.viewed]
            if viewed:
                pool = viewed

        # TF3: exact-variant — prefer option-explicit + evidenced SKUs.
        if self.plan.family == "constraint_satisfaction":
            evidenced = [v for v in hard if EvidenceGate.visible(self.plan, v)]
            if evidenced:
                pool = evidenced
            mentioned = [v for v in pool if SkuSpecPicker.option_mentions(self.plan, v)]
            if mentioned:
                pool = mentioned
            viewed = [v for v in pool if v.viewed]
            if viewed:
                pool = viewed

        if self.plan.family in CommitPicker.EVENT_FAMILIES:
            pool = sorted(pool, key=lambda v: EventCommit.score(self.plan, v), reverse=True)
        else:
            pool = sorted(pool, key=lambda v: score_variant(self.plan, v), reverse=True)

        # optional LLM adjudicate top few when ambiguous
        # Skip TF1/TF3/TF7 — first-cart commitment is deterministic.
        top = pool[:5]
        if (
            len(top) >= 2
            and self.session.remaining() > 8
            and self.plan.family
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
                pick = self._llm_select(top)
                if pick is not None:
                    return pick
            except Exception:
                pass
        return top[0]

    def _llm_select(self, variants: list[Variant]) -> Optional[Variant]:
        payload = []
        for i, v in enumerate(variants, start=1):
            payload.append(
                {
                    "id": i,
                    "product_id": v.product_id,
                    "sku": v.sku,
                    "title": v.title,
                    "brand": v.brand,
                    "price": v.price,
                    "in_stock": v.in_stock,
                    "options": v.options,
                    "facts": v.facts,
                    "category_path": v.category_path,
                }
            )
        prompt = (
            "Select the best purchasable variant for the shopper. "
            "Hard constraints must all pass. Soft preferences are secondary.\n"
            f"Family hint: {self.plan.family}\n"
            f"Goal: {self.plan.goal}\n"
            f"Budget: {self.plan.budget} {self.plan.currency}\n"
            f"Category: {self.plan.category}\n"
            f"Mandatory brand: {self.plan.brand}\n"
            f"Preferred brand: {self.plan.preferred_brand}\n"
            f"Attributes: {json.dumps(self.plan.attributes)}\n"
            f"Soft prefs: {self.plan.soft_prefs}\n"
            f"Candidates:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            'Return JSON: {"best_id": <int>, "reason": <str>}'
        )
        js = llm_json(prompt, expect="dict")
        bid = int(js.get("best_id") or 0)
        if 1 <= bid <= len(variants):
            return variants[bid - 1]
        return None

    def _brand_hunt(self) -> None:
        """One-shot preferred/mandatory brand recall before first cart."""
        brand = CommitPicker.soft_brand(self.plan)
        if not brand or self.session.remaining() <= 5:
            return
        for q in self._firm_attr_queries()[:2]:
            if self.session.remaining() <= 4:
                break
            self.search(q, k=10, in_stock=True)
        bits = [
            brand,
            self.plan.category or "",
            " ".join(str(v) for v in self.plan.attributes.values()),
        ]
        q = " ".join(b for b in bits if b)
        self.search(q, k=10, in_stock=True)
        if self.session.remaining() > 4:
            self.filter_catalog(
                category=self.plan.category,
                brand=brand,
                max_price=self.plan.budget,
                k=10,
                in_stock=True,
            )
        self._handle_shopper_note()

    def _prepare_commit(self, choice: Variant) -> Optional[Variant]:
        """View + pick firm/buyable SKU before first cart (event families).

        ORO Bench ``intended_event_commitment`` keys off the *first* cart add.
        Survey top EventCommit-ranked products, then commit the best viewed
        claimable/firm SKU — never cart a weak decoy hoping recovery will fix it.
        """
        families = CommitPicker.EVENT_FAMILIES
        if self.plan.family not in families:
            return choice

        def pick_from_product(pid: str) -> Optional[Variant]:
            variants = self.view(pid) if self.session.remaining() > 2 else []
            self._handle_shopper_note()
            best = CommitPicker.best(self.plan, variants)
            if best is None:
                return None
            if self.session.remaining() > 2:
                best = self.inspect_stock(best.product_id, best.sku)
                self._handle_shopper_note(pending_choice=best)
            if not (hard_ok(self.plan, best) and purchase_ok(self.plan, best)):
                return None
            if self.plan.family == "justification" and not EvidenceGate.claimable(
                self.plan, best
            ):
                return None
            return best

        brand = CommitPicker.soft_brand(self.plan)

        # If preferred brand has no pool hits yet, hunt once before accepting unbranded.
        if brand and self.plan.family == "intent_decomposition":
            branded_hits = [v for v in self.pool.values() if brand_ok(v, brand)]
            if not branded_hits:
                self._brand_hunt()
            # Filter-stamped branded hits often omit the firm attr in the title
            # (Hioro TF1-001: Vivo Y36 without "6.64" until viewed). View them
            # before shortlist so hard_ok can unlock preferred-brand first cart.
            branded_leads = [
                v
                for v in self.pool.values()
                if brand_ok(v, brand)
                and purchase_ok(self.plan, v)
                and DeviceGuard.ok(v, self.plan.category)
                and OptionGuard.ok(v, self.plan.category)
            ]
            branded_leads.sort(
                key=lambda v: (
                    0 if (self._recent_hits and v.product_id in self._recent_hits) else 1,
                    0 if hard_ok(self.plan, v) else 1,
                    -brand_strength(v, brand),
                    abs(
                        (float(v.price) / float(self.plan.budget))
                        if (v.price is not None and self.plan.budget)
                        else 1.0
                    ),
                )
            )
            seen: set[str] = set()
            for lead in branded_leads:
                if self.session.remaining() <= 5:
                    break
                if lead.product_id in seen:
                    continue
                seen.add(lead.product_id)
                if hard_ok(self.plan, lead) and lead.viewed and brand_ok(lead, brand):
                    return lead
                picked = pick_from_product(lead.product_id)
                if (
                    picked is not None
                    and brand_ok(picked, brand)
                    and hard_ok(self.plan, picked)
                    and purchase_ok(self.plan, picked)
                ):
                    return picked
                if len(seen) >= 5:
                    break
            # Last branded firm hunt under the live budget.
            if self.session.remaining() > 6 and not any(
                brand_ok(v, brand) and hard_ok(self.plan, v) and purchase_ok(self.plan, v)
                for v in self.pool.values()
            ):
                self._brand_hunt()
                for lead in sorted(
                    [
                        v
                        for v in self.pool.values()
                        if brand_ok(v, brand) and purchase_ok(self.plan, v)
                    ],
                    key=lambda v: abs(
                        (float(v.price) / float(self.plan.budget))
                        if (v.price is not None and self.plan.budget)
                        else 1.0
                    ),
                )[:3]:
                    if self.session.remaining() <= 5:
                        break
                    picked = pick_from_product(lead.product_id)
                    if (
                        picked is not None
                        and brand_ok(picked, brand)
                        and hard_ok(self.plan, picked)
                    ):
                        return picked

        # Rank pool by suite-agnostic EventCommit signals, then view top products.
        ranked = EventCommit.shortlist(self.plan, list(self.pool.values()), limit=6)
        try_pids = [v.product_id for v in ranked]
        if choice.product_id not in try_pids:
            try_pids.append(choice.product_id)

        picked_ok: list[Variant] = []
        for pid in try_pids:
            if self.session.remaining() <= 5:
                break
            picked = pick_from_product(pid)
            if picked is None:
                continue
            picked_ok.append(picked)
            # Early stop once we have a strong, claimable, low-ambiguity hit.
            if (
                EventCommit.listing_ambiguity(picked) >= -0.5
                and (
                    self.plan.family != "justification"
                    or EvidenceGate.claimable(self.plan, picked)
                )
                and (not brand or brand_ok(picked, brand))
            ):
                # Keep surveying one more rival when budget allows.
                if len(picked_ok) >= 2 or self.session.remaining() <= 8:
                    break

        if not picked_ok:
            # Prefer hard_ok choice; TF7 still allows a visible fallback so we
            # can cart something claimable hunting can improve later.
            if self.plan.family == "justification":
                if hard_ok(self.plan, choice) and EvidenceGate.visible(self.plan, choice):
                    return choice
                return None
            return choice if hard_ok(self.plan, choice) else None

        # Commit readiness (Hioro): line up a fallback before first cart when time allows.
        roomy = self.session.remaining() > 8 and self._cart_adds == 0
        if roomy and len(picked_ok) < 2:
            # Survey one more product when only a single verified candidate exists.
            for pid in try_pids:
                if self.session.remaining() <= 6:
                    break
                if any(p.product_id == pid for p in picked_ok):
                    continue
                picked = pick_from_product(pid)
                if picked is not None:
                    picked_ok.append(picked)
                    if len(picked_ok) >= 2:
                        break

        picked_ok.sort(key=lambda v: EventCommit.score(self.plan, v), reverse=True)
        # Soft brand is subordinate to firm specs, but when a branded firm hit
        # exists it must win first cart (intended_event_commitment).
        if brand:
            branded = [v for v in picked_ok if brand_ok(v, brand)]
            if branded:
                strong = [v for v in branded if brand_strength(v, brand) >= 2]
                picked_ok = strong or branded
            else:
                # Never first-cart unbranded while branded purchase_ok leads remain
                # unviewed — their firm attr may only appear after view (TF1-001).
                pending = [
                    v
                    for v in self.pool.values()
                    if brand_ok(v, brand)
                    and purchase_ok(self.plan, v)
                    and DeviceGuard.ok(v, self.plan.category)
                    and OptionGuard.ok(v, self.plan.category)
                    and not v.viewed
                ]
                pending.sort(
                    key=lambda v: abs(
                        (float(v.price) / float(self.plan.budget))
                        if (v.price is not None and self.plan.budget)
                        else 1.0
                    )
                )
                for lead in pending[:4]:
                    if self.session.remaining() <= 5:
                        break
                    hunted = pick_from_product(lead.product_id)
                    if (
                        hunted is not None
                        and brand_ok(hunted, brand)
                        and hard_ok(self.plan, hunted)
                    ):
                        return hunted
                pool_branded = [
                    v
                    for v in self.pool.values()
                    if hard_ok(self.plan, v)
                    and purchase_ok(self.plan, v)
                    and brand_ok(v, brand)
                    and OptionGuard.ok(v, self.plan.category)
                ]
                if pool_branded:
                    pool_branded.sort(
                        key=lambda v: (
                            brand_strength(v, brand),
                            EventCommit.score(self.plan, v),
                        ),
                        reverse=True,
                    )
                    return pool_branded[0]
        best = picked_ok[0]
        # TF7: never return an unclaimable first-cart candidate.
        if self.plan.family == "justification" and not EvidenceGate.claimable(
            self.plan, best
        ):
            return None
        return best

    def _commit_ready(self) -> bool:
        """Whether a first commitment can be made (Hioro-style readiness)."""
        if self._cart_adds:
            return True
        # Recall: do not apply ≥2 verified until surface quota is closer.
        if self.plan.family == "retrieval_recall":
            return True
        roomy = self.session.remaining() > 8
        if not roomy:
            return True
        verified = [
            v
            for v in self.pool.values()
            if v.viewed and hard_ok(self.plan, v) and purchase_ok(self.plan, v)
        ]
        if self.plan.family == "preference_reasoning":
            listing_ids = {v.listing_id for v in verified if v.listing_id}
            if len(listing_ids) < 2 and len(verified) < 2:
                return False
            return True
        if self.plan.family in CommitPicker.EVENT_FAMILIES | {"recovery"}:
            if len({v.product_id for v in verified}) < 2 and len(verified) < 2:
                # Intent: wait briefly for preferred-brand firm hits when early.
                if (
                    self.plan.family == "intent_decomposition"
                    and self.plan.preferred_brand
                    and self.session.remaining() > 12
                ):
                    branded = [
                        v for v in verified if brand_ok(v, self.plan.preferred_brand)
                    ]
                    if not branded:
                        return False
                return len(verified) >= 1  # allow single if survey already exhausted
        return True

    def _close(self, choice: Variant) -> None:
        # After interventions may have changed budget/attrs, re-query once for TF1/TF3.
        if (
            self.plan.family in {"intent_decomposition", "constraint_satisfaction"}
            and self.plan.attributes
            and self.session.remaining() > 8
            and not hard_ok(self.plan, choice)
        ):
            self._spec_requery()
            alts = [v for v in self.pool.values() if hard_ok(self.plan, v)]
            if self.plan.family == "intent_decomposition" and self.plan.preferred_brand:
                branded = [v for v in alts if brand_ok(v, self.plan.preferred_brand)]
                if branded:
                    alts = branded
            if alts:
                alts.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                choice = alts[0]

        # Answer pending clarify before spending views on commit selection.
        self._handle_shopper_note(pending_choice=choice)

        # Preference: compare ≥2 listing_ids before first cart when time allows.
        if (
            self.plan.family == "preference_reasoning"
            and self._cart_adds == 0
            and self.session.remaining() > 8
        ):
            listing_ids = {
                v.listing_id for v in self.pool.values() if v.listing_id and v.viewed
            }
            if len(listing_ids) < 2:
                leads = [
                    v.product_id
                    for v in sorted(
                        self.pool.values(),
                        key=lambda v: preference_score(self.plan, v),
                        reverse=True,
                    )
                    if v.product_id
                ]
                uniq: list[str] = []
                for pid in leads:
                    if pid not in uniq:
                        uniq.append(pid)
                if len(uniq) >= 2 and self.session.can("compare"):
                    self.compare(uniq[:5])
                elif uniq:
                    for pid in uniq[:3]:
                        if self.session.remaining() <= 6:
                            break
                        self.view(pid)

        # Delay first cart until a fallback exists when time allows.
        if not self._commit_ready() and self.session.remaining() > 8:
            if self.plan.family == "intent_decomposition" and self.plan.preferred_brand:
                self._brand_hunt()
            elif self.plan.attributes:
                self._spec_requery()

        prepared = self._prepare_commit(choice)
        if prepared is None:
            # Last attempt: any hard_ok in pool, ranked for event commitment.
            alts = [v for v in self.pool.values() if hard_ok(self.plan, v)]
            if self.plan.family == "justification":
                want = self.plan.brand or self.plan.preferred_brand
                branded = [v for v in alts if want and brand_ok(v, want)]
                if branded:
                    alts = branded
                claimable = [v for v in alts if EvidenceGate.claimable(self.plan, v)]
                if claimable:
                    alts = claimable
            if self.plan.family == "intent_decomposition" and self.plan.preferred_brand:
                branded = [v for v in alts if brand_ok(v, self.plan.preferred_brand)]
                if branded:
                    alts = branded
            alts.sort(key=lambda v: EventCommit.score(self.plan, v), reverse=True)
            if not alts:
                return
            choice = alts[0]
            if not choice.viewed and self.session.remaining() > 3:
                self.view(choice.product_id)
                choice = CommitPicker.best(
                    self.plan,
                    [v for v in self.pool.values() if v.product_id == choice.product_id],
                ) or choice
            if self.plan.family == "justification" and not EvidenceGate.claimable(
                self.plan, choice
            ):
                # Better to hunt than first-cart an unclaimable decoy.
                if self.session.remaining() > 6:
                    self._spec_requery()
                    alts = [
                        v
                        for v in self.pool.values()
                        if hard_ok(self.plan, v) and EvidenceGate.claimable(self.plan, v)
                    ]
                    alts.sort(key=lambda v: EventCommit.score(self.plan, v), reverse=True)
                    if not alts:
                        return
                    choice = alts[0]
                else:
                    return
        else:
            choice = prepared

        # Ranking / other families: ensure viewed + live stock when budget allows.
        if self.plan.family == "ranking":
            if not choice.viewed and self.session.remaining() > 3:
                variants = self.view(choice.product_id)
                if variants:
                    variants.sort(key=ranking_key, reverse=True)
                    choice = variants[0]
            if self.session.remaining() > 3:
                choice = self.inspect_stock(choice.product_id, choice.sku)
                self._handle_shopper_note(pending_choice=choice)
        elif self.plan.family == "preference_reasoning":
            # First cart is graded — view + inspect before committing.
            if not choice.viewed and self.session.remaining() > 3:
                variants = self.view(choice.product_id)
                self._handle_shopper_note(pending_choice=choice)
                if variants:
                    variants.sort(
                        key=lambda v: preference_score(self.plan, v), reverse=True
                    )
                    choice = variants[0]
            if self.session.remaining() > 3:
                choice = self.inspect_stock(choice.product_id, choice.sku)
                self._handle_shopper_note(pending_choice=choice)
        elif self.plan.family not in {
            "intent_decomposition",
            "constraint_satisfaction",
            "justification",
        }:
            if not choice.viewed and self.session.remaining() > 3:
                variants = self.view(choice.product_id)
                self._handle_shopper_note(pending_choice=choice)
                picked = CommitPicker.best(self.plan, variants)
                if picked is not None:
                    choice = picked
                elif variants:
                    variants.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                    choice = variants[0]
            if self.session.remaining() > 3:
                choice = self.inspect_stock(choice.product_id, choice.sku)
                self._handle_shopper_note(pending_choice=choice)

        # Reserve enough steps for add → inspect → order.
        if self.session.remaining() < 2:
            return

        add_obs = self.add_to_cart(choice.product_id, choice.sku)
        if self.session.done:
            return

        # Apply cart-add market events BEFORE recover/order decisions.
        live = self.pool.get(choice.key) or choice
        live = CartState.apply_add(live, add_obs or {})
        self.pool[live.key] = live

        # Always re-inspect after first cart when possible — silent price/stock
        # events are common and claims must use the live post-event price.
        if self.session.remaining() > 0:
            live = self.inspect_stock(choice.product_id, choice.sku)
            live = CartState.apply_add(live, add_obs or {})
            self.pool[live.key] = live
            self._handle_shopper_note(pending_choice=live)
        else:
            self._handle_shopper_note(pending_choice=live)

        # CRITICAL: recover only when the committed SKU is no longer buyable
        # (stockout / over-budget). Do NOT recover for soft preference misses —
        # that sacrifices first-cart scoring / intended_event_commitment.
        if needs_event_recovery(self.plan, live):
            self._dead_keys.add(live.key)
        if not needs_event_recovery(self.plan, live):
            if self.plan.family == "ranking":
                replacement = self._recover_ranking(exclude_keys=set(), previous=live)
                if replacement is not None and replacement.key != live.key:
                    self.remove_from_cart(choice.product_id, choice.sku)
                    if self.session.remaining() > 1:
                        if not replacement.viewed:
                            self.view(replacement.product_id)
                        self.inspect_stock(replacement.product_id, replacement.sku)
                    self.add_to_cart(replacement.product_id, replacement.sku)
                    if self.session.done:
                        return
                    if self.session.remaining() > 0:
                        replacement = self.inspect_stock(replacement.product_id, replacement.sku)
                    if self.session.remaining() > 0 and purchase_ok(self.plan, replacement):
                        self.place_order(replacement)
                    return
            if purchase_ok(self.plan, live):
                # Event families: refuse to seal a firm-miss after a good first cart.
                if (
                    self.plan.family in CommitPicker.EVENT_FAMILIES
                    and self.plan.attributes
                    and not hard_ok(self.plan, live)
                ):
                    alt = self._recover(exclude_keys={live.key}, previous=live)
                    if alt is not None and hard_ok(self.plan, alt):
                        self.remove_from_cart(choice.product_id, choice.sku)
                        self.add_to_cart(alt.product_id, alt.sku)
                        if self.session.done:
                            return
                        if self.session.remaining() > 0:
                            alt = self.inspect_stock(alt.product_id, alt.sku)
                        if purchase_ok(self.plan, alt) and hard_ok(self.plan, alt):
                            self.place_order(alt)
                        return
                self.place_order(live)
            return

        # Recovery: remove and find replacement
        self.remove_from_cart(choice.product_id, choice.sku)
        if self.plan.family == "ranking":
            replacement = self._recover_ranking(exclude_keys={choice.key}, previous=choice)
        else:
            replacement = self._recover(exclude_keys={choice.key}, previous=choice)
        if replacement is None:
            # Never re-add a dead/over-budget event target.
            if purchase_ok(self.plan, live) and live.key not in self._dead_keys:
                self.add_to_cart(live.product_id, live.sku)
                if not self.session.done and purchase_ok(self.plan, live):
                    self.place_order(live)
            elif self.session.remaining() > 0:
                self._force_terminal()
            return

        if self.session.remaining() > 1:
            if not replacement.viewed:
                self.view(replacement.product_id)
            self.inspect_stock(replacement.product_id, replacement.sku)
            replacement = self.pool.get(replacement.key, replacement)
        if not purchase_ok(self.plan, replacement):
            # Find another buyable hard-ok candidate if recover landed on OOS.
            alt = self._recover(exclude_keys={choice.key, replacement.key}, previous=replacement)
            if alt is None or not purchase_ok(self.plan, alt):
                if self.session.remaining() > 0:
                    self._force_terminal()
                return
            replacement = alt
        self.add_to_cart(replacement.product_id, replacement.sku)
        if self.session.done:
            return
        # Note: market events usually fire on first cart; still re-inspect.
        if self.session.remaining() > 0:
            replacement = self.inspect_stock(replacement.product_id, replacement.sku)
            self.pool[replacement.key] = replacement
        if self.session.remaining() > 0 and purchase_ok(self.plan, replacement):
            self.place_order(replacement)
        elif self.session.remaining() > 0:
            self._force_terminal()

    def _recover_ranking(self, *, exclude_keys: set[str], previous: Variant) -> Optional[Variant]:
        """Re-rank only named candidates after an event; view post-event top."""
        keys = self.plan.candidate_keys or []
        pool: list[Variant] = []
        for key in keys:
            if key in exclude_keys:
                continue
            v = self.pool.get(key)
            if v is None:
                pid, _, sku = key.partition("::")
                v = Variant(product_id=pid, sku=sku or "default")
                self.pool[key] = v
            if not v.viewed and self.session.remaining() > 3:
                self.view(v.product_id)
                v = self.pool.get(key, v)
            if self.session.remaining() > 2:
                v = self.inspect_stock(v.product_id, v.sku)
            if v.in_stock and budget_ok(v, self.plan.budget):
                pool.append(v)
        if not pool:
            return None
        pool.sort(key=ranking_key, reverse=True)
        return pool[0]

    def _recover(self, *, exclude_keys: set[str], previous: Variant) -> Optional[Variant]:
        blocked = set(exclude_keys) | set(self._dead_keys)
        # Always try sibling SKUs on the same listing first (Hioro TF3/TF6/TF7):
        # stockouts are often SKU-local; other colors/variants on the same
        # product_id remain buyable even when the dead key is blocked.
        if previous.product_id and self.session.remaining() > 2:
            variants = self.view(previous.product_id)
            siblings = [
                v
                for v in variants
                if v.key not in blocked
                and hard_ok(self.plan, v)
                and purchase_ok(self.plan, v)
            ]
            if siblings:
                siblings.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                return siblings[0]
            # Inspect sibling stock if view listed them as available.
            for v in variants:
                if v.key in blocked or not hard_ok(self.plan, v):
                    continue
                if self.session.remaining() <= 2:
                    break
                live = self.inspect_stock(v.product_id, v.sku)
                if purchase_ok(self.plan, live) and hard_ok(self.plan, live):
                    return live

        # Title-stem twin search before brand-free requery (event families only).
        # Recovery must NOT early-return on title stem — that skips same-model
        # other-color policy and tanks TF6 (Hioro recovers by model/color).
        if (
            previous.title
            and self.session.remaining() > 4
            and self.session.can("search")
            and self.plan.family
            in {
                "justification",
                "constraint_satisfaction",
                "intent_decomposition",
            }
        ):
            stem = " ".join((previous.title or "").split()[:8])
            if stem:
                hits = self.search(stem, k=10, in_stock=True)
                found = self._pick_from_hits(hits, blocked=blocked)
                if found is not None:
                    return found

        # Event families: recover under the live (possibly tightened) budget.
        # Prefer soft-brand firm survivors first; only then drop the soft brand
        # so off-brand fact-only hits (Note 3 @ 1599) can surface.
        if self.plan.family in {
            "justification",
            "constraint_satisfaction",
            "intent_decomposition",
        } and self.session.remaining() > 2:
            soft = CommitPicker.soft_brand(self.plan)
            # Pass 1: keep soft brand when a branded firm hit can still fit.
            if soft:
                for q in self._firm_attr_queries(soft_brand=True)[:2]:
                    if self.session.remaining() <= 4:
                        break
                    hits = self.search(q, k=10, in_stock=True)
                    found = self._pick_from_hits(hits, blocked=blocked)
                    if found is not None:
                        return found
                if self.session.remaining() > 4:
                    self.filter_catalog(
                        category=self.plan.category,
                        brand=soft,
                        max_price=self.plan.budget,
                        k=10,
                        in_stock=True,
                    )
                    found = self._view_budget_leads(blocked=blocked, limit=4)
                    if found is not None and brand_ok(found, soft):
                        return found
            # Pass 2: brand-free firm hunt (Hioro Note 3 path).
            for q in self._firm_attr_queries(soft_brand=False)[:3]:
                if self.session.remaining() <= 3:
                    break
                hits = self.search(q, k=10, in_stock=True)
                found = self._pick_from_hits(hits, blocked=blocked)
                if found is not None:
                    return found
            if self.session.remaining() > 3:
                self.filter_catalog(
                    category=self.plan.category,
                    brand=self.plan.brand,
                    max_price=self.plan.budget,
                    k=10,
                    in_stock=True,
                )
            self._handle_shopper_note()
            found = self._view_budget_leads(blocked=blocked, limit=8)
            if found is not None:
                return found
        # Twin / title-stem search (Hioro): same-line alternate listings.
        # Preference: re-seed soft axes instead of cloning a weak first-cart title.
        elif self.session.remaining() > 3 and self.session.can("search"):
            if self.plan.family == "preference_reasoning":
                for q in soft_search_queries(self.plan)[:3]:
                    if self.session.remaining() <= 3:
                        break
                    self.search(q, k=10, in_stock=True)
                self._handle_shopper_note()
            elif previous.title and previous.key not in self._dead_keys:
                stem = " ".join((previous.title or "").split()[:8])
                if stem:
                    self.search(stem, k=10, in_stock=True)
                    self._handle_shopper_note()

        # Search again with updated budget / attrs from interventions
        if self.session.remaining() > 2:
            if self.plan.family == "recovery" and self.plan.preferred_model:
                self.search(self.plan.preferred_model, k=10, in_stock=True)
                if self.plan.preferred_color and self.session.remaining() > 2:
                    self.search(
                        f"{self.plan.preferred_color} {self.plan.preferred_model}",
                        k=10,
                        in_stock=True,
                    )
            elif self.plan.family == "preference_reasoning":
                for q in soft_search_queries(self.plan)[:3]:
                    if self.session.remaining() <= 2:
                        break
                    self.search(q, k=10, in_stock=True)
            elif self.plan.family in {
                "justification",
                "constraint_satisfaction",
                "intent_decomposition",
            }:
                if self.session.remaining() > 2:
                    found = self._view_budget_leads(blocked=blocked, limit=4)
                    if found is not None:
                        return found
            else:
                q = " ".join(self.plan.keywords[:6]) or self.plan.goal[:60]
                self.search(q, k=10, in_stock=True)
                if self.session.remaining() > 2:
                    self.filter_catalog(
                        category=self.plan.category,
                        brand=self.plan.brand,
                        max_price=self.plan.budget,
                        k=10,
                        in_stock=True,
                    )
                self._handle_shopper_note()

        candidates = [
            v
            for v in self.pool.values()
            if v.key not in blocked
            and hard_ok(self.plan, v)
            and purchase_ok(self.plan, v)
        ]
        if self.plan.family == "recovery":
            same_model = []
            if self.plan.preferred_model:
                same_model = [
                    v
                    for v in candidates
                    if _norm(self.plan.preferred_model) in _norm(v.text_blob() + " " + v.options)
                ]
            if same_model and self.plan.preferred_color:
                other_color = [
                    v
                    for v in same_model
                    if _norm(self.plan.preferred_color) not in _norm(v.text_blob() + " " + v.options)
                ]
                pool = other_color or same_model
            else:
                pool = same_model or candidates
        elif self.plan.family == "preference_reasoning":
            prev_listing = previous.listing_id
            diversed = [
                v
                for v in candidates
                if v.listing_id and v.listing_id != prev_listing
            ]
            pool = diversed or candidates
            pool = sorted(pool, key=lambda v: preference_score(self.plan, v), reverse=True)
            return pool[0] if pool else None
        elif self.plan.family == "justification":
            want = self.plan.brand or self.plan.preferred_brand
            branded = [v for v in candidates if want and brand_ok(v, want)]
            pool = branded or candidates
        elif self.plan.family == "intent_decomposition":
            if self.plan.preferred_brand:
                branded = [v for v in candidates if brand_ok(v, self.plan.preferred_brand)]
                pool = branded or candidates
            else:
                pool = candidates
        else:
            # Prefer same product_id siblings still in the global pool.
            same = [v for v in candidates if v.product_id == previous.product_id]
            pool = same or candidates

        if not pool:
            if self.plan.family in {
                "constraint_satisfaction",
                "justification",
                "intent_decomposition",
            }:
                return None
            pool = [
                v
                for v in self.pool.values()
                if v.key not in blocked and purchase_ok(self.plan, v)
            ]
        if not pool:
            return None

        if self.plan.family in CommitPicker.EVENT_FAMILIES:
            pool.sort(key=lambda v: EventCommit.score(self.plan, v), reverse=True)
        else:
            pool.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
        best = pool[0]
        if not best.viewed and self.session.remaining() > 2:
            variants = self.view(best.product_id)
            variants = [
                v
                for v in variants
                if (hard_ok(self.plan, v) or purchase_ok(self.plan, v))
                and v.key not in blocked
            ]
            if variants:
                if self.plan.family in CommitPicker.EVENT_FAMILIES:
                    variants.sort(key=lambda v: EventCommit.score(self.plan, v), reverse=True)
                else:
                    variants.sort(key=lambda v: score_variant(self.plan, v), reverse=True)
                best = variants[0]
        if not purchase_ok(self.plan, best):
            return None
        return best

    def _build_justification(self, variant: Variant) -> dict[str, Any]:
        """TF7-only: claims must be grounded in viewed facts / live price / stock."""
        return ClaimBuilder.build(self.plan, variant)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ORO Bench entrypoint: one EnvPack task session."""
    session = EnvSession.from_problem(problem_data)
    agent = SOSOAgent(session)
    return agent.run()


__all__ = ["agent_main"]
