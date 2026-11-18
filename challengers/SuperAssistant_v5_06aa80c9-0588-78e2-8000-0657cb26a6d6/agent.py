from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient


_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)

_TF_FAMILIES = ("TF1", "TF2", "TF3", "TF4", "TF5", "TF6", "TF7")

# Only used when scores are essentially equal (within 0.02).
_TF_TIE_BREAK = ("TF5", "TF7", "TF1", "TF6", "TF3", "TF4", "TF2")

_JSON_SYSTEM = "Return ONLY one JSON object. No markdown. No commentary."

_TF_DETECT_PROMPT = """Pick the SINGLE dominant ORO Bench task family from the PUBLIC shopper goal and tools.
You never see a sealed TF label. Ignore shared boilerplate ("do not wait for permission",
generic stock/budget reminders) — score the DISTINCTIVE objective of this episode.

Return ONLY one JSON object:
{{
  "scores": {{"TF1":0.0,"TF2":0.0,"TF3":0.0,"TF4":0.0,"TF5":0.0,"TF6":0.0,"TF7":0.0}},
  "primary": "TF1",
  "reason": "short"
}}

Score each family in [0,1]. primary MUST be the family with the highest score.

Distinctive SIGNATURES (choose one primary):
- TF1 Intent Decomposition: shopper must TELL FIRM VS SOFT APART; "mix of firm requirements and nice-to-haves"; initial cart choice and later recovery assessed SEPARATELY; soft preference subordinate to firm requirements.
- TF2 Retrieval Recall: survey/shortlist several distinct products; reformulate searches; avoid one lucky hit; retrieval coverage is the main goal.
- TF3 Constraint Satisfaction: EVERY firm requirement must hold on the EXACT variant/SKU; brand/spec/budget/stock gates; NO firm-vs-soft curriculum and NO closed candidate ranking list.
- TF4 Preference Reasoning: soft tradeoffs / intended use dominate ("rather pay more", best match vs cheapest valid).
- TF5 Ranking: "Compare only these candidate variants" and/or explicit product_id::sku list with priority order / among ties.
- TF6 Recovery: preferred exact item may disappear; try another color/model; least-relaxed recovery after stock/price change is the MAIN goal.
- TF7 Justification: must explain/walk through why; terminal tool requires structured justification claims.

Hard rules:
- If product_id::sku candidates / "compare only these" appear → primary TF5.
- If terminal justification is required or goal says "walk me through why" → primary TF7 (unless TF5 list dominates).
- Do NOT pick TF1 just because budget/stock/permission boilerplate appears.
- Do NOT pick TF3 just because a budget appears if TF1/TF4/TF5/TF6 signatures are stronger.

Shopper goal:
{query}

Available tool names:
{tool_names}

Terminal/schema hints:
{schema_hints}
"""

_BASE = (
    "You are a shopping agent in a sealed evaluation environment. "
    "Use only the tools supplied in this task's policy_view. "
    "Continue until an observation reports done=true or the step budget ends. "
    "Never invent tools or product facts. "
    "Decision order: firm requirements first, then soft preferences, then "
    "ranking/selection. Verify constraints against viewed attributes and "
    "SKU/option facts — not title keywords alone. "
    "If a shopper user_message arrives and a dialogue tool is available, "
    "answer it before other actions, then resume catalog work. Do not ask "
    "the shopper to relax firm constraints. Once a firm-valid option is "
    "verified, finish with the terminal tool instead of waiting for permission."
)

_RETRY_NO_TOOL = (
    "The environment has not reported done=true. "
    "Choose one of the supplied tools to continue."
)

# Family-aware episode plan — transfers valid.py decompose / attribute /
# keyword / select ideas into ORO without suite-specific catalogs.
_EPISODE_PLAN_PROMPT = """Build a compact working plan for an ORO shopping episode.
Return ONLY one JSON object. No markdown. No product/brand invention beyond the goal text.

Family for this episode: {family}
Family focus: {family_focus}

Transferable planning rules (from high-performing shopping agents):
1) DECOMPOSE: separate firm/non-negotiable requirements from soft preferences.
   Soft never overrides firm. Scope each constraint to the nearest product unless
   the goal explicitly shares it ("all"/"both").
2) STRUCTURE: rewrite firm requirements as short attribute-like cues derived from
   the goal's own wording (mutually exclusive aspects). Do not invent specs.
3) KEYWORDS: propose 1-3 search keyword strings using VALUE-side tokens from the
   goal (product type + distinctive values). Avoid bare metadata key names as
   lone keywords. Optionally offer one slightly broader alt for recall.
4) PRECISION: every firm requirement must eventually hold on the exact variant.
5) EVIDENCE: final claims must come from viewed attributes/SKU facts, not titles.

Return JSON shape:
{{
  "firm_requirements": ["..."],
  "soft_preferences": ["..."],
  "attribute_cues": {{"product_type": "..."}},
  "search_keywords": ["primary keywords", "optional alt"],
  "steps": ["ordered action 1", "action 2"],
  "priority_order": ["priority cue from goal or empty"],
  "listed_variants": ["product_id::sku", "..."],
  "spec_claim_field": null,
  "spec_claim_value": null,
  "focus": "one short sentence"
}}

Rules:
- Extract ONLY from the public goal and tool schemas. Never invent products, brands,
  SKUs, numeric specs, or attributes that are not present in the goal text.
- firm_requirements / soft_preferences: quote or paraphrase ONLY goal wording.
- attribute_cues: keys/values taken from goal wording; include product_type when clear.
- search_keywords: short space-separated strings; first is primary; later are recall alts
  using value-side tokens from the goal (not bare metadata key names alone).
- steps: 1-6 actions using only supplied tool capabilities (search/view/compare/cart/order/etc.).
- listed_variants: copy any product_id::sku tokens from the goal (else []).
- priority_order: explicit ranking/tie-break cues from the goal (else []).
- spec_claim_field / spec_claim_value: primary firm specification attribute implied by
  the goal for structured claims (snake_case field from goal wording), or null.
- Never invent tool names.

Shopper goal:
{query}

Available tool names:
{tool_names}

Terminal/schema hints:
{schema_hints}
"""


# ---------------------------------------------------------------------------
# Model / inference
# ---------------------------------------------------------------------------


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "model": _model(),
        "messages": messages,
        "temperature": temperature,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    inference = _proxy.post("/inference/chat/completions", json_data=payload)
    if inference is None:
        return None
    try:
        if inference["choices"][0]["message"] is not None:
            return inference
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _llm_json(
    prompt: str,
    *,
    temperature: float = 0,
    attempts: int = 2,
) -> dict[str, Any] | None:
    temp = temperature
    for _ in range(attempts):
        inference = _chat_completion(
            [
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
        )
        if inference is None:
            temp = min(temp + 0.3, 0.9)
            continue
        try:
            content = inference["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            temp = min(temp + 0.3, 0.9)
            continue
        parsed = _extract_json_object(content)
        if parsed is not None:
            return parsed
        temp = min(temp + 0.3, 0.9)
    return None


# ---------------------------------------------------------------------------
# Tool / observation helpers
# ---------------------------------------------------------------------------


def _tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {}
    function = tool.get("function")
    if isinstance(function, dict):
        return function
    return tool


def _tool_entries(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for tool in tools:
        function = _tool_function(tool)
        name = function.get("name")
        if isinstance(name, str) and name:
            entries.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters")
                    if isinstance(function.get("parameters"), dict)
                    else {},
                }
            )
    return entries


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    return [entry["name"] for entry in _tool_entries(tools)]


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    return str(function.get("name") or "")


def _schema_blob(parameters: dict[str, Any]) -> str:
    try:
        return json.dumps(parameters, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        return str(parameters).lower()


def _find_dialogue_tool(tools: list[dict[str, Any]]) -> str | None:
    for entry in _tool_entries(tools):
        props = (entry["parameters"].get("properties") or {}) if entry["parameters"] else {}
        required = entry["parameters"].get("required") or []
        desc = entry["description"].lower()
        prop_names = {str(key).lower() for key in props.keys()}
        if prop_names == {"content"} or (
            "content" in prop_names and list(required) == ["content"]
        ):
            if any(
                token in desc
                for token in ("message", "shopper", "user", "send", "reply", "ask")
            ) or not desc:
                return entry["name"]
        if any(
            token in desc for token in ("send a short message", "message to the shopper")
        ):
            return entry["name"]
    return None


def _find_terminal_tool(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    entries = _tool_entries(tools)
    with_justification: list[dict[str, Any]] = []
    terminal_like: list[dict[str, Any]] = []
    for entry in entries:
        blob = _schema_blob(entry["parameters"]) + " " + entry["description"].lower()
        has_justification = "justification" in blob or "claims" in blob
        looks_terminal = any(
            token in blob
            for token in ("place", "order", "submit", "finish", "checkout", "done")
        )
        if has_justification:
            with_justification.append(entry)
        elif looks_terminal:
            terminal_like.append(entry)
    if with_justification:
        return with_justification[0]
    if terminal_like:
        return terminal_like[0]
    return None


def _terminal_requires_justification(terminal_tool: dict[str, Any] | None) -> bool:
    if not terminal_tool:
        return False
    parameters = terminal_tool.get("parameters") or {}
    required = parameters.get("required") or []
    return isinstance(required, list) and "justification" in required


def _justification_field_schema(terminal_tool: dict[str, Any] | None) -> dict[str, Any]:
    if not terminal_tool:
        return {}
    parameters = terminal_tool.get("parameters") or {}
    props = parameters.get("properties") or {}
    justification = props.get("justification") or {}
    if not isinstance(justification, dict):
        return {}
    jprops = justification.get("properties") or {}
    claims = jprops.get("claims") or {}
    if not isinstance(claims, dict):
        return {}
    items = claims.get("items") or {}
    if not isinstance(items, dict):
        return {}
    field_schema = (items.get("properties") or {}).get("field") or {}
    return field_schema if isinstance(field_schema, dict) else {}


def _justification_field_enum(terminal_tool: dict[str, Any] | None) -> list[str]:
    field_schema = _justification_field_schema(terminal_tool)
    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list):
        return [str(value) for value in enum_values if isinstance(value, str)]
    return []


def _justification_field_description(terminal_tool: dict[str, Any] | None) -> str:
    field_schema = _justification_field_schema(terminal_tool)
    return str(field_schema.get("description") or "")


def _user_message_text(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    user_message = result.get("user_message")
    if isinstance(user_message, dict):
        return str(user_message.get("content") or "").strip()
    if isinstance(user_message, str):
        return user_message.strip()
    return ""


def _observation_blob(observation: Any) -> str:
    try:
        return json.dumps(observation, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        return str(observation).lower()


def _looks_failed(observation: Any) -> bool:
    blob = _observation_blob(observation)
    return any(
        token in blob
        for token in (
            '"ok": false',
            '"success": false',
            "error",
            "failed",
            "failure",
            "invalid",
            "rejected",
            "not allowed",
        )
    )


def _looks_unavailable(observation: Any) -> bool:
    blob = _observation_blob(observation)
    return any(
        token in blob
        for token in (
            "out of stock",
            "oos",
            "unavailable",
            "not available",
            "sold out",
            '"in_stock": false',
            '"in_stock":false',
        )
    )


def _observation_has_results(observation: Any) -> bool | None:
    if not isinstance(observation, dict):
        return None
    for key in ("products", "items", "results", "candidates"):
        value = observation.get(key)
        if isinstance(value, list):
            return len(value) > 0
    return None


def _observation_done(observation: Any) -> bool:
    return isinstance(observation, dict) and bool(observation.get("done"))


def _split_query_steps(query: str) -> list[str]:
    text = " ".join(query.split()).strip()
    if not text:
        return []
    numbered = re.findall(
        r"(?:^|\s)(?:\d+[.)]|first|second|third|then|next|finally|lastly|also)[,:]?\s+([^.;]+)",
        text,
        flags=re.I,
    )
    if len(numbered) >= 2:
        return [" ".join(part.split()) for part in numbered[:6]]
    parts = [
        p.strip()
        for p in re.split(r"[.;]|(?:\band then\b)|(?:\bthen\b)", text, flags=re.I)
        if p.strip()
    ]
    if len(parts) >= 2:
        return parts[:6]
    return [text]


def _extract_listed_variants(query: str) -> list[tuple[str, str]]:
    if not query:
        return []
    return re.findall(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9_\-]*)", query)


def _extract_hard_constraints(query: str) -> list[str]:
    """Light linguistic extraction of firm limits from the public goal."""

    q = " ".join(query.split())
    if not q:
        return []
    cues = []
    for match in re.finditer(
        r"(?i)\b(?:must|only|under|less than|at least|exactly|between|no more than|"
        r"budget(?:\s+of)?|priced?\s+(?:under|below|at)|in stock|available)\b[^.;]{0,80}",
        q,
    ):
        snippet = " ".join(match.group(0).split())
        if snippet and snippet not in cues:
            cues.append(snippet[:120])
        if len(cues) >= 6:
            break
    return cues


def _extract_soft_preferences(query: str) -> list[str]:
    q = " ".join(query.split())
    if not q:
        return []
    prefs = []
    for match in re.finditer(
        r"(?i)\b(?:prefer|preferably|rather|ideally|if possible|would like|"
        r"nice to have|soft preference)\b[^.;]{0,80}",
        q,
    ):
        snippet = " ".join(match.group(0).split())
        if snippet and snippet not in prefs:
            prefs.append(snippet[:120])
        if len(prefs) >= 6:
            break
    return prefs


_PROSE_CLAIM_FIELDS = frozenset(
    {
        "plus",
        "also",
        "reason",
        "reasons",
        "feature",
        "features",
        "why",
        "because",
        "good",
        "pick",
        "summary",
        "evidence",
        "note",
        "comment",
        "extra",
        "other",
        "detail",
        "details",
    }
)


def _fact_pairs_from_observation(observation: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(key: Any, value: Any) -> None:
        if not isinstance(key, str) or key.startswith("_"):
            return
        key_s = key.strip()
        if not key_s or key_s.lower() in _PROSE_CLAIM_FIELDS:
            return
        if isinstance(value, (dict, list)):
            return
        if value is None:
            return
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value).strip()
        if not text or len(text) > 120:
            return
        marker = f"{key_s.lower()}::{text.lower()}"
        if marker in seen:
            return
        seen.add(marker)
        pairs.append((key_s, text))

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 5 or node is None:
            return
        if isinstance(node, dict):
            facts = node.get("facts")
            if isinstance(facts, dict):
                for key, value in facts.items():
                    add(key, value)
            for key, value in node.items():
                if key in {"facts", "variants", "results", "title", "products", "items"}:
                    if key == "variants" and isinstance(value, list):
                        for variant in value[:4]:
                            walk(variant, depth + 1)
                    elif key in {"products", "items", "results"} and isinstance(
                        value, list
                    ):
                        for item in value[:8]:
                            walk(item, depth + 1)
                    continue
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)
                else:
                    add(key, value)
        elif isinstance(node, list):
            for item in node[:12]:
                walk(item, depth + 1)

    walk(observation)
    return pairs


def _merge_fact_context(
    fact_context: dict[str, str], observation: Any
) -> dict[str, str]:
    updated = dict(fact_context)
    for key, value in _fact_pairs_from_observation(observation):
        updated[key] = value
    return updated


def _normalize_price_claim_value(value: str) -> str:
    match = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
    if not match:
        return value.strip()
    return match.group(0)


def _normalize_stock_claim_value(value: str) -> str:
    token = value.strip().lower()
    if token in {"true", "yes", "y", "1", "in stock", "instock", "available"}:
        return "true"
    if token in {"false", "no", "n", "0", "out of stock", "oos", "unavailable"}:
        return "false"
    return value.strip()


def _soft_reason_fields_from_schema(terminal_tool: dict[str, Any] | None) -> list[str]:
    desc = _justification_field_description(terminal_tool)
    if not desc:
        return []
    match = re.search(r"reason(?:s)?\s*:\s*([^\.]+)", desc, flags=re.I)
    if not match:
        return []
    names = re.findall(r"\b[a-z][a-z0-9_]*\b", match.group(1).lower())
    stop = {
        "use",
        "only",
        "the",
        "specification",
        "field",
        "in",
        "shopper",
        "request",
        "also",
        "and",
        "or",
        "one",
        "supported",
        "reason",
        "reasons",
        "feature",
        "features",
        "must",
        "be",
        "a",
        "an",
        "to",
        "for",
        "with",
        "from",
        "value",
        "values",
        "plus",
        "further",
        "grounded",
        "product",
        "facts",
        "fact",
        "at",
        "least",
        "including",
        "include",
        "such",
        "as",
    }
    ordered: list[str] = []
    for name in names:
        if name in stop or name in ordered or name in {"price", "in_stock", "stock"}:
            continue
        ordered.append(name)
    return ordered


def _schema_requires_extra_reason(terminal_tool: dict[str, Any] | None) -> bool:
    desc = _justification_field_description(terminal_tool).lower()
    if not desc:
        return False
    cues = (
        "further supported reason",
        "at least one further",
        "at least one supported reason",
        "additional reason",
        "one further supported",
        "further reason",
    )
    return any(cue in desc for cue in cues)


def _normalize_to_allowed_field(field: Any, allowed: list[str]) -> str | None:
    if not isinstance(field, str) or not allowed:
        return None
    raw = " ".join(field.strip().lower().replace("-", " ").replace("_", " ").split())
    if not raw:
        return None
    normalized_allowed = {
        " ".join(value.lower().replace("-", " ").replace("_", " ").split()): value
        for value in allowed
    }
    if raw in normalized_allowed:
        return normalized_allowed[raw]
    underscored = raw.replace(" ", "_")
    for value in allowed:
        if value.lower() == underscored or value.lower() == raw:
            return value
    hits = [
        value
        for key, value in normalized_allowed.items()
        if key and (key in raw or raw in key)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _build_claim_allowlist(
    terminal_tool: dict[str, Any] | None,
    *,
    spec_claim_field: str | None = None,
) -> list[str]:
    enum_fields = _justification_field_enum(terminal_tool)
    if enum_fields:
        allowed = list(enum_fields)
    else:
        allowed = []
        desc = _justification_field_description(terminal_tool).lower()
        if "price" in desc or not desc:
            allowed.append("price")
        if "in_stock" in desc or "stock" in desc or not desc:
            allowed.append("in_stock")
    if isinstance(spec_claim_field, str) and spec_claim_field.strip():
        field = spec_claim_field.strip()
        if field not in allowed:
            allowed.append(field)
    for name in _soft_reason_fields_from_schema(terminal_tool):
        if name not in allowed:
            allowed.append(name)
    for required in ("price", "in_stock"):
        if required not in allowed:
            allowed.append(required)
    return [name for name in allowed if name and name not in _PROSE_CLAIM_FIELDS]


def _pick_extra_reason_claim(
    fact_context: dict[str, str],
    *,
    reserved: set[str],
) -> dict[str, str] | None:
    skip = reserved | {
        "title",
        "query",
        "options",
        "sku",
        "product_id",
        "currency",
        "price",
        "in_stock",
        "stock",
    }
    candidates: list[tuple[int, str, str]] = []
    for key, value in fact_context.items():
        key_l = key.lower()
        if key_l in skip or key_l in _PROSE_CLAIM_FIELDS:
            continue
        if len(value) > 80:
            continue
        score = len(value) + (0 if 3 <= len(value) <= 40 else 20)
        candidates.append((score, key, value))
    if not candidates:
        return None
    candidates.sort()
    _, key, value = candidates[0]
    return {"field": key, "value": value}


def _sanitize_terminal_args(
    args: dict[str, Any],
    *,
    terminal_tool: dict[str, Any] | None,
    fact_context: dict[str, str],
    spec_claim_field: str | None = None,
    omit_optional_justification: bool = False,
) -> dict[str, Any]:
    """Repair justification claims using schema allowlist + observed facts."""

    cleaned = dict(args)
    if not terminal_tool:
        return cleaned

    if omit_optional_justification and not _terminal_requires_justification(
        terminal_tool
    ):
        cleaned.pop("justification", None)
        return cleaned

    if "justification" not in cleaned:
        return cleaned
    if not _terminal_requires_justification(terminal_tool):
        # Optional justification: drop rather than risk untruthful claims.
        cleaned.pop("justification", None)
        return cleaned

    justification = cleaned.get("justification")
    if not isinstance(justification, dict):
        return cleaned
    claims = justification.get("claims")
    if not isinstance(claims, list):
        return cleaned

    allowed = _build_claim_allowlist(
        terminal_tool, spec_claim_field=spec_claim_field
    )
    allowed_lower = {name.lower(): name for name in allowed}
    soft = {name.lower() for name in _soft_reason_fields_from_schema(terminal_tool)}
    repaired: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    soft_kept = False

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        raw_field = str(claim.get("field") or "").strip()
        raw_value = claim.get("value")
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not raw_field or not value:
            continue
        if raw_field.lower() in _PROSE_CLAIM_FIELDS:
            continue
        mapped = _normalize_to_allowed_field(raw_field, allowed)
        if mapped is None:
            mapped = allowed_lower.get(raw_field.lower())
        if mapped is None:
            hits = [
                name
                for key, name in allowed_lower.items()
                if key and (key in raw_field.lower() or raw_field.lower() in key)
            ]
            mapped = hits[0] if len(hits) == 1 else None
        if mapped is None:
            continue
        if mapped.lower() == "price":
            value = _normalize_price_claim_value(value)
            if "price" in fact_context:
                value = _normalize_price_claim_value(fact_context["price"])
        elif mapped.lower() in {"in_stock", "stock"}:
            value = _normalize_stock_claim_value(value)
            if "in_stock" in fact_context:
                value = _normalize_stock_claim_value(fact_context["in_stock"])
        elif mapped in fact_context:
            value = fact_context[mapped]
        elif mapped.lower() in {k.lower(): k for k in fact_context}:
            for key, observed in fact_context.items():
                if key.lower() == mapped.lower():
                    value = observed
                    break
        key = mapped.lower()
        if key in seen_fields:
            continue
        is_soft = key in soft
        if is_soft:
            if soft_kept:
                continue
            soft_kept = True
        seen_fields.add(key)
        repaired.append({"field": mapped, "value": value})

    if "price" not in seen_fields and "price" in fact_context:
        repaired.append(
            {
                "field": allowed_lower.get("price", "price"),
                "value": _normalize_price_claim_value(fact_context["price"]),
            }
        )
        seen_fields.add("price")
    if "in_stock" not in seen_fields and "in_stock" in fact_context:
        repaired.append(
            {
                "field": allowed_lower.get("in_stock", "in_stock"),
                "value": _normalize_stock_claim_value(fact_context["in_stock"]),
            }
        )
        seen_fields.add("in_stock")

    if (
        isinstance(spec_claim_field, str)
        and spec_claim_field
        and spec_claim_field.lower() not in seen_fields
    ):
        for key, observed in fact_context.items():
            if key.lower() == spec_claim_field.lower():
                repaired.append({"field": spec_claim_field, "value": observed})
                seen_fields.add(spec_claim_field.lower())
                break

    if _schema_requires_extra_reason(terminal_tool) and not soft_kept:
        extra = _pick_extra_reason_claim(fact_context, reserved=seen_fields)
        if extra is not None:
            mapped = _normalize_to_allowed_field(extra["field"], allowed)
            if mapped is None and soft:
                # Prefer a schema soft/reason field name when available.
                mapped = next(iter(soft))
                mapped = allowed_lower.get(mapped, mapped)
            if mapped is not None and mapped.lower() not in seen_fields:
                repaired.append({"field": mapped, "value": extra["value"]})

    cleaned["justification"] = {**justification, "claims": repaired}
    return cleaned


# ---------------------------------------------------------------------------
# Episode context
# ---------------------------------------------------------------------------


@dataclass
class EpisodeContext:
    problem_id: str
    binding: dict[str, Any]
    query: str
    tools: list[dict[str, Any]]
    max_steps: int
    max_calls: int
    dialogue_tool: str | None = None
    terminal_tool: dict[str, Any] | None = None
    family: str = "TF2"
    scores: dict[str, float] = field(default_factory=dict)
    detect_reason: str = ""
    detect_source: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TFDetectionResult:
    primary: str
    scores: dict[str, float]
    reason: str
    source: str


def _episode_context(problem_data: dict[str, Any]) -> EpisodeContext:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    tools = list(policy.get("tools") or [])
    return EpisodeContext(
        problem_id=str(problem_data.get("problem_id", problem_data.get("id", "problem"))),
        binding=binding,
        query=str(policy.get("query", "")),
        tools=tools,
        max_steps=int(policy["max_steps"]),
        max_calls=int(policy.get("max_calls_per_turn", 1)),
        dialogue_tool=_find_dialogue_tool(tools),
        terminal_tool=_find_terminal_tool(tools),
    )


def _environment_call(
    ctx: EpisodeContext,
    turn: int,
    tool_calls: list[dict[str, Any]],
    *,
    arg_transform: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    group_id = f"{ctx.problem_id}-turn-{turn}"
    calls_payload = []
    for index, call in enumerate(tool_calls, start=1):
        name = call["function"]["name"]
        args = _arguments(call)
        if arg_transform is not None:
            args = arg_transform(name, args)
        calls_payload.append(
            {
                "call_id": f"{group_id}-{index}",
                "action": {"name": name, "args": args},
            }
        )
    envelope = {
        **ctx.binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": calls_payload,
    }
    result = _proxy.post("/environment/call", json_data=envelope)
    if result is None:
        raise RuntimeError("environment call failed")
    return result


def _infer_tool_calls(
    messages: list[dict[str, Any]],
    ctx: EpisodeContext,
    *,
    attempts: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    assistant_content = ""
    tool_calls: list[dict[str, Any]] = []
    for _ in range(attempts):
        inference = _chat_completion(
            messages,
            tools=ctx.tools,
            tool_choice="required",
            temperature=0,
        )
        if inference is None:
            raise RuntimeError("inference request failed")
        assistant = inference["choices"][0]["message"]
        assistant_content = assistant.get("content") or ""
        tool_calls = (assistant.get("tool_calls") or [])[: ctx.max_calls]
        if tool_calls:
            return assistant_content, tool_calls
        messages.extend(
            [
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": _RETRY_NO_TOOL},
            ]
        )
    raise RuntimeError("model returned no tool call before completion")


# ---------------------------------------------------------------------------
# TF detection (scored primary route only)
# ---------------------------------------------------------------------------


def _schema_hints(tools: list[dict[str, Any]]) -> str:
    terminal = _find_terminal_tool(tools)
    parts = [f"tool_names={', '.join(_tool_names(tools)) or '(none)'}"]
    if terminal:
        parts.append(f"terminal={terminal['name']}")
        parts.append(
            "justification_required="
            + str(_terminal_requires_justification(terminal)).lower()
        )
        enum_fields = _justification_field_enum(terminal)
        if enum_fields:
            parts.append("claim_fields=" + ",".join(enum_fields[:12]))
    else:
        parts.append("terminal=none")
    return "; ".join(parts)


def _empty_scores() -> dict[str, float]:
    return {family: 0.0 for family in _TF_FAMILIES}


def _clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _normalize_scores(raw: Any) -> dict[str, float]:
    scores = _empty_scores()
    if not isinstance(raw, dict):
        return scores
    for family in _TF_FAMILIES:
        if family in raw:
            scores[family] = _clamp_score(raw.get(family))
        else:
            scores[family] = _clamp_score(raw.get(family.lower()))
    return scores


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


_TF1_SIGNATURES = (
    "tell them apart",
    "mix of firm requirements and nice-to-haves",
    "mix of firm",
    "subordinate to every firm",
    "initial cart choice",
    "initial choice and your later recovery",
    "initial choice and later recovery",
    "assessed separately",
    "soft preference, subordinate",
    "nice-to-haves, and you must tell",
)

_TF2_SIGNATURES = (
    "survey the realistic options",
    "several distinct products",
    "not one lucky hit",
    "reformulate queries",
    "good shortlist covers",
)

_TF3_SIGNATURES = (
    "every requirement must hold",
    "exact variant you order",
    "check the specific sku",
    "firm, non-negotiable requirements",
    "there is one firm requirement on the spec",
)

_TF4_SIGNATURES = (
    "rather pay",
    "soft preferences matter",
    "best match",
    "intended use",
    "use-specific features",
    "loose hard request plus soft",
)

_TF5_SIGNATURES = (
    "compare only these",
    "among ties",
    "among remaining ties",
    "candidate variants",
    "best to worst",
)

_TF6_SIGNATURES = (
    "if that exact",
    "disappears",
    "try another",
    "least-relaxed",
    "preferred exact",
    "becomes unavailable",
)

_TF7_SIGNATURES = (
    "walk me through why",
    "explain why",
    "with evidence",
    "must justify",
    "provide justification",
    "structured justification is required",
)


def _apply_signal_boosts(
    scores: dict[str, float], query: str, tools: list[dict[str, Any]]
) -> dict[str, float]:
    """Boost scores from distinctive public cues (not shared boilerplate)."""

    boosted = dict(scores)
    q = query.lower()
    listed = _extract_listed_variants(query)
    terminal = _find_terminal_tool(tools)
    names = {name.lower() for name in _tool_names(tools)}

    sig = {
        "TF1": _has_any(q, _TF1_SIGNATURES),
        "TF2": _has_any(q, _TF2_SIGNATURES),
        "TF3": _has_any(q, _TF3_SIGNATURES),
        "TF4": _has_any(q, _TF4_SIGNATURES),
        "TF5": bool(listed) or _has_any(q, _TF5_SIGNATURES),
        "TF6": _has_any(q, _TF6_SIGNATURES),
        "TF7": _terminal_requires_justification(terminal)
        or (
            _has_any(q, _TF7_SIGNATURES)
            and "justification is optional" not in q
            and "unless the shopper asks for a reason" not in q
        ),
    }

    if sig["TF5"]:
        boosted["TF5"] = max(boosted["TF5"], 0.97)
    elif any("rank" in n for n in names):
        boosted["TF5"] = max(boosted["TF5"], 0.7)

    if sig["TF7"]:
        boosted["TF7"] = max(
            boosted["TF7"],
            0.97
            if _terminal_requires_justification(terminal)
            or "walk me through why" in q
            else 0.93,
        )

    if sig["TF1"]:
        boosted["TF1"] = max(boosted["TF1"], 0.96)

    if sig["TF2"]:
        boosted["TF2"] = max(boosted["TF2"], 0.94)

    if sig["TF6"]:
        boosted["TF6"] = max(boosted["TF6"], 0.95)

    if sig["TF4"]:
        boosted["TF4"] = max(boosted["TF4"], 0.93)

    if sig["TF3"] and not (sig["TF1"] or sig["TF5"] or sig["TF6"] or sig["TF7"]):
        boosted["TF3"] = max(boosted["TF3"], 0.92)

    strong = {fam for fam, hit in sig.items() if hit}
    if strong:
        for family in _TF_FAMILIES:
            if family not in strong and boosted[family] > 0.55:
                boosted[family] = min(boosted[family], 0.55)

    return boosted


def _pick_primary(scores: dict[str, float]) -> str:
    """Strict argmax; family order only for near-identical scores (<= 0.02)."""

    best = max(scores.values()) if scores else 0.0
    if best <= 0:
        return "TF2"
    contenders = [
        family for family, value in scores.items() if value >= best - 0.02
    ]
    if len(contenders) == 1:
        return contenders[0]
    for family in _TF_TIE_BREAK:
        if family in contenders:
            return family
    return contenders[0]


def _heuristic_tf_detection(query: str, tools: list[dict[str, Any]]) -> TFDetectionResult:
    scores = _empty_scores()
    q = query.lower()
    listed = _extract_listed_variants(query)
    terminal = _find_terminal_tool(tools)

    if _has_any(q, _TF1_SIGNATURES):
        scores["TF1"] = 0.9

    if listed or _has_any(q, _TF5_SIGNATURES):
        scores["TF5"] = 0.95

    if _terminal_requires_justification(terminal):
        scores["TF7"] = 0.95
    elif (
        _has_any(q, _TF7_SIGNATURES)
        and "justification is optional" not in q
        and "unless the shopper asks for a reason" not in q
    ):
        scores["TF7"] = 0.9

    if _has_any(q, _TF2_SIGNATURES):
        scores["TF2"] = 0.88

    if _has_any(q, _TF6_SIGNATURES):
        scores["TF6"] = 0.45 if scores["TF1"] >= 0.8 else 0.9

    if _has_any(q, _TF4_SIGNATURES):
        scores["TF4"] = 0.5 if scores["TF1"] >= 0.8 else 0.85

    if _has_any(q, _TF3_SIGNATURES):
        if scores["TF1"] >= 0.8 or scores["TF5"] >= 0.8 or scores["TF6"] >= 0.8:
            scores["TF3"] = 0.45
        else:
            scores["TF3"] = 0.88
    elif any(
        k in q
        for k in (
            "must",
            "at or under",
            "budget",
            "in stock",
            "non-negotiable",
            "exactly",
        )
    ) and max(scores.values()) < 0.6:
        scores["TF3"] = 0.7

    if any(k in q for k in ("find", "search", "options", "candidates", "look for")):
        scores["TF2"] = max(scores["TF2"], 0.4)

    scores = _apply_signal_boosts(scores, query, tools)
    if max(scores.values()) <= 0:
        scores["TF2"] = 0.5

    return TFDetectionResult(
        primary=_pick_primary(scores),
        scores=scores,
        reason="heuristic signature scores",
        source="heuristic",
    )


def _detection_from_llm_payload(
    raw: dict[str, Any], query: str, tools: list[dict[str, Any]]
) -> TFDetectionResult | None:
    scores = _normalize_scores(raw.get("scores"))
    legacy = str(raw.get("family") or raw.get("primary") or "").strip().upper()
    if max(scores.values()) <= 0 and legacy in _TF_FAMILIES:
        scores[legacy] = 1.0

    scores = _apply_signal_boosts(scores, query, tools)
    if max(scores.values()) <= 0:
        return None

    primary = _pick_primary(scores)
    reason = str(raw.get("reason") or "").strip() or "llm scored signatures"
    return TFDetectionResult(
        primary=primary,
        scores=scores,
        reason=reason,
        source="llm",
    )


def TF_Detection(query: str, tools: list[dict[str, Any]]) -> TFDetectionResult:
    """Score TF1-TF7 from query+tools; return the single primary route."""

    prompt = _TF_DETECT_PROMPT.format(
        query=query,
        tool_names=", ".join(_tool_names(tools)) or "(none provided)",
        schema_hints=_schema_hints(tools),
    )
    raw = _llm_json(prompt, temperature=0, attempts=2)
    if isinstance(raw, dict):
        parsed = _detection_from_llm_payload(raw, query, tools)
        if parsed is not None:
            return parsed
    return _heuristic_tf_detection(query, tools)


# ---------------------------------------------------------------------------
# Family focus + episode plan (valid.py-style decompose / keywords / attributes)
# ---------------------------------------------------------------------------


_FAMILY_FOCUS = {
    "TF1": (
        "Family focus TF1 Intent Decomposition: separate firm requirements from "
        "soft preferences; honor soft only after firm gates; treat initial choice "
        "and later recovery separately when the goal says so; finish once verified."
    ),
    "TF2": (
        "Family focus TF2 Retrieval Recall: gather candidates broadly enough to "
        "avoid misses, reformulate keywords when results are thin, survey several "
        "distinct products, then refine before finishing."
    ),
    "TF3": (
        "Family focus TF3 Constraint Satisfaction: every firm requirement must hold "
        "on the exact variant/SKU; verify attributes and options, reject near-misses."
    ),
    "TF4": (
        "Family focus TF4 Preference Reasoning: among firm-valid options, prefer the "
        "best soft-preference / intended-use match rather than merely any valid item."
    ),
    "TF5": (
        "Family focus TF5 Ranking: compare only listed candidates when present; "
        "follow stated priority / tie-break order; omit optional justification."
    ),
    "TF6": (
        "Family focus TF6 Recovery: on failure/stock/price change, switch to the "
        "least-relaxed compliant alternative, re-verify, then finish."
    ),
    "TF7": (
        "Family focus TF7 Justification: ground the final outcome in viewed "
        "attribute/SKU facts and schema-valid justification claims."
    ),
}


def _as_str_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = " ".join(str(item).split()).strip()
        if text and text not in out:
            out.append(text[:160])
        if len(out) >= limit:
            break
    return out


def _as_str_dict(value: Any, limit: int = 10) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        text = " ".join(str(raw).split()).strip()
        if not text or isinstance(raw, (dict, list)):
            continue
        out[key.strip()[:64]] = text[:120]
        if len(out) >= limit:
            break
    return out


def _heuristic_plan(ctx: EpisodeContext, family: str) -> dict[str, Any]:
    listed = [f"{pid}::{sku}" for pid, sku in _extract_listed_variants(ctx.query)]
    firm = _extract_hard_constraints(ctx.query)
    soft = _extract_soft_preferences(ctx.query)
    steps = _split_query_steps(ctx.query)
    keywords: list[str] = []
    # Value-side keyword seed: drop common filler; keep distinctive tokens.
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "with",
        "from",
        "that",
        "this",
        "find",
        "looking",
        "need",
        "want",
        "please",
        "shopper",
        "budget",
        "under",
        "must",
        "have",
        "has",
    }
    tokens = [
        tok
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/%]*", ctx.query)
        if tok.lower() not in stop and len(tok) > 1
    ]
    if tokens:
        keywords.append(" ".join(tokens[:8]))
        if len(tokens) > 4:
            keywords.append(" ".join(tokens[:4]))
    return {
        "firm_requirements": firm,
        "soft_preferences": soft,
        "attribute_cues": {},
        "search_keywords": keywords,
        "steps": steps[:6],
        "priority_order": [],
        "listed_variants": listed,
        "spec_claim_field": None,
        "spec_claim_value": None,
        "focus": _FAMILY_FOCUS.get(family, _FAMILY_FOCUS["TF2"]),
        "source": "heuristic",
    }


def _normalize_plan(raw: dict[str, Any] | None, ctx: EpisodeContext, family: str) -> dict[str, Any]:
    base = _heuristic_plan(ctx, family)
    if not isinstance(raw, dict):
        return base
    firm = _as_str_list(raw.get("firm_requirements"))
    soft = _as_str_list(raw.get("soft_preferences"))
    steps = _as_str_list(raw.get("steps"), limit=6)
    keywords = _as_str_list(raw.get("search_keywords"), limit=4)
    priority = _as_str_list(raw.get("priority_order"), limit=6)
    listed = _as_str_list(raw.get("listed_variants"), limit=12)
    if not listed:
        listed = base["listed_variants"]
    cues = _as_str_dict(raw.get("attribute_cues"))
    spec_field = raw.get("spec_claim_field")
    spec_value = raw.get("spec_claim_value")
    if isinstance(spec_field, str):
        spec_field = re.sub(r"[^a-z0-9_]+", "_", spec_field.strip().lower()).strip("_")
        if not spec_field:
            spec_field = None
    else:
        spec_field = None
    if isinstance(spec_value, str):
        spec_value = " ".join(spec_value.split()).strip()[:80] or None
    else:
        spec_value = None
    focus = str(raw.get("focus") or "").strip() or base["focus"]
    return {
        "firm_requirements": firm or base["firm_requirements"],
        "soft_preferences": soft or base["soft_preferences"],
        "attribute_cues": cues or base["attribute_cues"],
        "search_keywords": keywords or base["search_keywords"],
        "steps": steps or base["steps"],
        "priority_order": priority,
        "listed_variants": listed,
        "spec_claim_field": spec_field,
        "spec_claim_value": spec_value,
        "focus": focus,
        "source": "llm",
    }


def _build_episode_plan(ctx: EpisodeContext, family: str) -> dict[str, Any]:
    prompt = _EPISODE_PLAN_PROMPT.format(
        family=family,
        family_focus=_FAMILY_FOCUS.get(family, _FAMILY_FOCUS["TF2"]),
        query=ctx.query,
        tool_names=", ".join(_tool_names(ctx.tools)) or "(none)",
        schema_hints=_schema_hints(ctx.tools),
    )
    raw = _llm_json(prompt, temperature=0, attempts=2)
    plan = _normalize_plan(raw, ctx, family)
    if _terminal_requires_justification(ctx.terminal_tool) and not plan.get(
        "spec_claim_field"
    ):
        # Keep null rather than inventing; claims still repair via facts.
        pass
    return plan


def _plan_system_hint(plan: dict[str, Any], family: str) -> str:
    parts = [f"Working plan ({family}): {plan.get('focus') or ''}"]
    firm = plan.get("firm_requirements") or []
    if firm:
        parts.append("Firm requirements: " + "; ".join(firm))
    soft = plan.get("soft_preferences") or []
    if soft:
        parts.append("Soft preferences (after firm gates): " + "; ".join(soft))
    cues = plan.get("attribute_cues") or {}
    if cues:
        rendered = ", ".join(f"{k}={v}" for k, v in list(cues.items())[:8])
        parts.append(
            "Structured attribute cues (verify on attributes/SKU, not title alone): "
            + rendered
        )
    keywords = plan.get("search_keywords") or []
    if keywords:
        parts.append("Search keyword seeds: " + " | ".join(keywords))
    listed = plan.get("listed_variants") or []
    if listed:
        parts.append("Listed candidate variants only: " + ", ".join(listed))
    priority = plan.get("priority_order") or []
    if priority:
        parts.append("Priority / tie-break order: " + " > ".join(priority))
    steps = plan.get("steps") or []
    if steps:
        checklist = "; ".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
        parts.append("Checklist: " + checklist)
    spec_field = plan.get("spec_claim_field")
    spec_value = plan.get("spec_claim_value")
    if isinstance(spec_field, str) and spec_field:
        if isinstance(spec_value, str) and spec_value:
            parts.append(
                f"Primary spec claim target: `{spec_field}`=`{spec_value}` from viewed facts."
            )
        else:
            parts.append(
                f"Primary spec claim field: `{spec_field}` from viewed facts on the final item."
            )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Shared dialogue bookkeeping helpers used inside each fork
# ---------------------------------------------------------------------------


def _append_tool_observations(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[Any]:
    observations: list[Any] = []
    for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
        observation = call_result["observation"]
        observations.append(observation)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(observation),
            }
        )
    return observations


def _seed_dialogue(ctx: EpisodeContext) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "tf_detection",
            "family": ctx.family,
            "primary": ctx.family,
            "scores": dict(ctx.scores),
            "reason": ctx.detect_reason,
            "source": ctx.detect_source,
            "dialogue_tool": ctx.dialogue_tool,
            "terminal_tool": ctx.terminal_tool["name"] if ctx.terminal_tool else None,
            "plan": dict(ctx.plan),
        }
    ]


def _intervention_prompt(dialogue_tool: str) -> str:
    return (
        f"A shopper message is waiting. Call `{dialogue_tool}` now with a short "
        "answer. Confirm firm requirements if asked; do not offer to relax them. "
        "Do not call any other tool until this dialogue reply is sent."
    )


def _recovery_prompt(ctx: EpisodeContext) -> str:
    terminal_name = (
        ctx.terminal_tool["name"] if ctx.terminal_tool else "the terminal tool"
    )
    dialogue_bit = (
        f" Do not call `{ctx.dialogue_tool}` unless a new shopper user_message is waiting."
        if ctx.dialogue_tool
        else ""
    )
    return (
        "RECOVERY: a prior candidate became invalid (stock/price/failure). "
        "Keep every firm requirement unchanged — never ask the shopper to relax "
        "hard constraints. Prefer the least-relaxed compliant alternative "
        "(another variant that still meets firm gates). Remove the invalid cart "
        "item if present, search/view a DIFFERENT firm-valid product, inspect "
        f"current stock and price, then call `{terminal_name}`."
        f"{dialogue_bit}"
    )


# ---------------------------------------------------------------------------
# Family loops — each TF owns a complete run_tfN_loop implementation
# ---------------------------------------------------------------------------


def _common_status_prefix(
    family: str,
    ctx: EpisodeContext,
    *,
    turn: int,
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
    observations: list[Any],
) -> list[str] | str:
    """Shared dialogue/recovery preamble. Returns a full string when shopper reply
    must happen immediately; otherwise a tip list to extend."""

    remaining = ctx.max_steps - turn
    if pending_shopper and ctx.dialogue_tool:
        return (
            _intervention_prompt(ctx.dialogue_tool)
            + f" Episode status: turn {turn}/{ctx.max_steps}."
        )
    tips = [
        f"{family} status: turn {turn}/{ctx.max_steps} ({remaining} left). "
        "done is false; choose one productive supplied tool."
    ]
    if recovery_active or any(
        _looks_failed(obs) or _looks_unavailable(obs) for obs in observations
    ):
        tips.append(_recovery_prompt(ctx))
    if consecutive_dialogue >= 1 and ctx.dialogue_tool:
        tips.append(
            f"Dialogue pause: do not call `{ctx.dialogue_tool}` again unless a new "
            "user_message arrives. Resume search/view/cart/terminal tools."
        )
    return tips


def _budget_finish_tip(
    ctx: EpisodeContext, *, remaining: int, recovery_active: bool
) -> str | None:
    terminal = ctx.terminal_tool
    if not terminal or not (remaining <= 12 or recovery_active):
        return None
    return (
        f"Step budget is limited: after verifying current stock and price on a "
        f"firm-valid item, call `{terminal['name']}` now instead of more dialogue."
    )


def _choose_tool_calls(
    messages: list[dict[str, Any]],
    ctx: EpisodeContext,
    *,
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
    remaining: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Infer a tool call while enforcing shopper-reply and no-chatter guards."""

    terminal_name = ctx.terminal_tool["name"] if ctx.terminal_tool else None
    for _attempt in range(5):
        assistant_content, tool_calls = _infer_tool_calls(messages, ctx)
        chosen = _tool_call_name(tool_calls[0])
        if pending_shopper and ctx.dialogue_tool and chosen != ctx.dialogue_tool:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        _intervention_prompt(ctx.dialogue_tool)
                        + f" Previous choice `{chosen}` is not allowed yet."
                    ),
                }
            )
            continue
        block_voluntary = bool(
            ctx.dialogue_tool
            and chosen == ctx.dialogue_tool
            and not pending_shopper
            and (recovery_active or consecutive_dialogue >= 1 or remaining <= 12)
        )
        if block_voluntary:
            reason = (
                _recovery_prompt(ctx)
                if recovery_active
                else (
                    f"Do not call `{ctx.dialogue_tool}` now — no new shopper "
                    "user_message is waiting. Firm constraints stay fixed. "
                    f"Continue with catalog/`{terminal_name or 'terminal'}` tools."
                )
            )
            messages.append(
                {
                    "role": "user",
                    "content": reason
                    + f" Previous choice `{chosen}` is not allowed yet.",
                }
            )
            continue
        return assistant_content, tool_calls
    raise RuntimeError("model returned no valid tool call before completion")


def _tf1_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
) -> str:
    prefix = _common_status_prefix(
        "TF1",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    plan = ctx.plan or {}
    firm = plan.get("firm_requirements") or []
    soft = plan.get("soft_preferences") or []
    steps = plan.get("steps") or []
    tips.append(
        "TF1 decompose: keep firm vs soft distinct. Soft preferences are "
        "subordinate to every firm gate — never promote a soft preference into "
        "a hard search filter. If initial choice and later recovery are assessed "
        "separately, do not let soft recovery undo firm compliance."
    )
    if firm:
        tips.append("Firm gates: " + "; ".join(firm))
    if soft:
        tips.append("Soft only after firm: " + "; ".join(soft))
    if steps:
        tips.append(
            "Advance checklist: "
            + "; ".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))
        )
    finish = _budget_finish_tip(
        ctx, remaining=ctx.max_steps - turn, recovery_active=recovery_active
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _tf2_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
    empty_result_streak: int,
) -> str:
    prefix = _common_status_prefix(
        "TF2",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    keywords = (ctx.plan or {}).get("search_keywords") or []
    tips.append(
        "TF2 recall: survey several distinct products before finishing. "
        "Avoid stopping at one lucky hit. Catalog search indexes VALUES not "
        "field names — prefer value-side keywords."
    )
    if keywords:
        tips.append("Keyword seeds / alts: " + " | ".join(keywords))
    if empty_result_streak >= 1 and keywords:
        alt = keywords[min(empty_result_streak, len(keywords) - 1)]
        tips.append(
            "Sparse/empty retrieval: reformulate with a single small change "
            f"(add or drop one token). Try: `{alt}`."
        )
    elif empty_result_streak >= 1:
        tips.append(
            "Sparse/empty retrieval: reformulate the query with one small "
            "keyword change to improve recall without inventing specs."
        )
    if turn <= 3:
        tips.append("Early turns: broaden coverage, then refine.")
    finish = _budget_finish_tip(
        ctx, remaining=ctx.max_steps - turn, recovery_active=recovery_active
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _tf3_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
) -> str:
    prefix = _common_status_prefix(
        "TF3",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    plan = ctx.plan or {}
    firm = plan.get("firm_requirements") or []
    cues = plan.get("attribute_cues") or {}
    tips.append(
        "TF3 precision: a candidate is valid ONLY if every firm requirement "
        "holds on the exact variant/SKU. Reject near-misses. Verify using "
        "attributes and sku options — never title alone."
    )
    if firm:
        tips.append("Required constraints: " + "; ".join(firm))
    if cues:
        rendered = ", ".join(f"{k}={v}" for k, v in list(cues.items())[:8])
        tips.append("Attribute cues to confirm: " + rendered)
    finish = _budget_finish_tip(
        ctx, remaining=ctx.max_steps - turn, recovery_active=recovery_active
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _tf4_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
) -> str:
    prefix = _common_status_prefix(
        "TF4",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    plan = ctx.plan or {}
    firm = plan.get("firm_requirements") or []
    soft = plan.get("soft_preferences") or []
    tips.append(
        "TF4 preference: first filter to firm-valid options, then choose the "
        "best soft-preference / intended-use match (not merely cheapest valid)."
    )
    if firm:
        tips.append("Firm filter: " + "; ".join(firm))
    if soft:
        tips.append("Prefer among valid: " + "; ".join(soft))
    finish = _budget_finish_tip(
        ctx, remaining=ctx.max_steps - turn, recovery_active=recovery_active
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _tf5_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
) -> str:
    prefix = _common_status_prefix(
        "TF5",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    plan = ctx.plan or {}
    listed = plan.get("listed_variants") or []
    priority = plan.get("priority_order") or []
    remaining = ctx.max_steps - turn
    terminal = ctx.terminal_tool
    terminal_name = terminal["name"] if terminal else "the terminal tool"
    tips.append(
        "TF5 ranking: compare ONLY the listed/allowed candidates. Apply the "
        "stated priority and tie-break order. Do not open-ended catalog search "
        "away from the closed set when a candidate list is given."
    )
    if listed:
        tips.append("Closed candidate set: " + ", ".join(listed))
    if priority:
        tips.append("Priority order: " + " > ".join(priority))
    if terminal and not _terminal_requires_justification(terminal):
        tips.append(
            "Justification is optional here — omit it unless the shopper "
            "explicitly asked for a reason."
        )
    if remaining <= 15:
        tips.append(
            f"Ranking finish: verify stock/price on the top pick, then call "
            f"`{terminal_name}` before the step limit."
        )
    finish = _budget_finish_tip(
        ctx, remaining=remaining, recovery_active=recovery_active
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _tf6_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
) -> str:
    prefix = _common_status_prefix(
        "TF6",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    plan = ctx.plan or {}
    firm = plan.get("firm_requirements") or []
    soft = plan.get("soft_preferences") or []
    tips.append(
        "TF6 recovery-first: the preferred exact item may disappear. Switch to "
        "another compliant color/model with the least relaxation of firm rules, "
        "re-check live stock and price, then finish."
    )
    if firm:
        tips.append("Unchanged firm gates: " + "; ".join(firm))
    if soft:
        tips.append(
            "Soft preferences guide the least-relaxed alternative: " + "; ".join(soft)
        )
    # Recovery family: always remind finish path when recovery is active.
    finish = _budget_finish_tip(
        ctx, remaining=ctx.max_steps - turn, recovery_active=True
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _tf7_coach(
    ctx: EpisodeContext,
    *,
    turn: int,
    observations: list[Any],
    pending_shopper: bool,
    recovery_active: bool,
    consecutive_dialogue: int,
) -> str:
    prefix = _common_status_prefix(
        "TF7",
        ctx,
        turn=turn,
        pending_shopper=pending_shopper,
        recovery_active=recovery_active,
        consecutive_dialogue=consecutive_dialogue,
        observations=observations,
    )
    if isinstance(prefix, str):
        return prefix
    tips = list(prefix)
    plan = ctx.plan or {}
    terminal = ctx.terminal_tool
    tips.append(
        "TF7 justification: inspect the FINAL item, then submit structured "
        "claims grounded in viewed attribute/SKU facts. Do not invent field "
        "names or title-only guesses."
    )
    if terminal and _terminal_requires_justification(terminal):
        allowed = _build_claim_allowlist(
            terminal, spec_claim_field=plan.get("spec_claim_field")
        )
        tips.append(
            "Required claims use only allowlisted fields: "
            + ", ".join(allowed)
            + ". Always include price and in_stock from observed facts."
        )
        spec_field = plan.get("spec_claim_field")
        spec_value = plan.get("spec_claim_value")
        if isinstance(spec_field, str) and spec_field:
            if isinstance(spec_value, str) and spec_value:
                tips.append(
                    f"Spec claim `{spec_field}`=`{spec_value}` must be supported "
                    "by the final item's viewed facts; otherwise keep searching."
                )
            else:
                tips.append(
                    f"Include spec claim field `{spec_field}` from viewed facts."
                )
        if _schema_requires_extra_reason(terminal):
            tips.append("Schema also needs one further fact-grounded reason claim.")
    finish = _budget_finish_tip(
        ctx, remaining=ctx.max_steps - turn, recovery_active=recovery_active
    )
    if finish:
        tips.append(finish)
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def _run_family_loop(
    ctx: EpisodeContext,
    family: str,
    *,
    coach: Callable[..., str],
    track_empty_results: bool = False,
    omit_optional_justification: bool = False,
) -> list[dict[str, Any]]:
    """Shared environment turn loop; family policy lives in plan + coach."""

    ctx.plan = _build_episode_plan(ctx, family)
    system = f"{_BASE} {_FAMILY_FOCUS[family]} {_plan_system_hint(ctx.plan, family)}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": ctx.query},
    ]
    dialogue = _seed_dialogue(ctx)
    pending_shopper = False
    recovery_active = False
    consecutive_dialogue = 0
    empty_result_streak = 0
    fact_context: dict[str, str] = {}
    terminal_name = ctx.terminal_tool["name"] if ctx.terminal_tool else None

    def _transform(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if terminal_name and name == terminal_name:
            return _sanitize_terminal_args(
                args,
                terminal_tool=ctx.terminal_tool,
                fact_context=fact_context,
                spec_claim_field=(ctx.plan or {}).get("spec_claim_field"),
                omit_optional_justification=omit_optional_justification,
            )
        return args

    for turn in range(1, ctx.max_steps + 1):
        remaining = ctx.max_steps - turn + 1
        assistant_content, tool_calls = _choose_tool_calls(
            messages,
            ctx,
            pending_shopper=pending_shopper,
            recovery_active=recovery_active,
            consecutive_dialogue=consecutive_dialogue,
            remaining=remaining,
        )
        used_dialogue = bool(
            ctx.dialogue_tool
            and any(_tool_call_name(c) == ctx.dialogue_tool for c in tool_calls)
        )
        if used_dialogue:
            pending_shopper = False
            consecutive_dialogue += 1
        else:
            consecutive_dialogue = 0

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        result = _environment_call(ctx, turn, tool_calls, arg_transform=_transform)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )
        observations = _append_tool_observations(messages, tool_calls, result)
        for obs in observations:
            fact_context = _merge_fact_context(fact_context, obs)
        if any(_looks_failed(obs) or _looks_unavailable(obs) for obs in observations):
            recovery_active = True

        if track_empty_results:
            has_results = [
                flag
                for flag in (_observation_has_results(obs) for obs in observations)
                if flag is not None
            ]
            if has_results and not any(has_results):
                empty_result_streak += 1
            elif has_results and any(has_results):
                empty_result_streak = 0

        shopper = _user_message_text(result)
        if shopper:
            pending_shopper = bool(ctx.dialogue_tool) and not used_dialogue
            messages.append({"role": "user", "content": shopper})
        if any(_observation_done(obs) for obs in observations):
            break

        coach_kwargs: dict[str, Any] = {
            "turn": turn,
            "observations": observations,
            "pending_shopper": pending_shopper,
            "recovery_active": recovery_active,
            "consecutive_dialogue": consecutive_dialogue,
        }
        if track_empty_results:
            coach_kwargs["empty_result_streak"] = empty_result_streak
        messages.append({"role": "user", "content": coach(ctx, **coach_kwargs)})
    return dialogue


def run_tf1_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF1 Intent Decomposition: firm vs soft, scoped constraints, checklist."""

    return _run_family_loop(ctx, "TF1", coach=_tf1_coach)


def run_tf2_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF2 Retrieval Recall: broad survey + keyword reformulation on empty hits."""

    return _run_family_loop(
        ctx, "TF2", coach=_tf2_coach, track_empty_results=True
    )


def run_tf3_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF3 Constraint Satisfaction: every firm requirement on exact SKU/attrs."""

    return _run_family_loop(ctx, "TF3", coach=_tf3_coach)


def run_tf4_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF4 Preference Reasoning: firm filter, then best soft/intended-use match."""

    return _run_family_loop(ctx, "TF4", coach=_tf4_coach)


def run_tf5_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF5 Ranking: closed candidate set + stated priority / tie-break."""

    return _run_family_loop(
        ctx,
        "TF5",
        coach=_tf5_coach,
        omit_optional_justification=True,
    )


def run_tf6_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF6 Recovery: least-relaxed alternate after stock/price/failure."""

    return _run_family_loop(ctx, "TF6", coach=_tf6_coach)


def run_tf7_loop(ctx: EpisodeContext) -> list[dict[str, Any]]:
    """TF7 Justification: schema-valid claims grounded in viewed facts."""

    return _run_family_loop(ctx, "TF7", coach=_tf7_coach)




# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect primary TF family from query+tools, then run that family loop."""

    ctx = _episode_context(problem_data)
    detection = TF_Detection(ctx.query, ctx.tools)
    ctx.family = detection.primary
    ctx.scores = dict(detection.scores)
    ctx.detect_reason = detection.reason
    ctx.detect_source = detection.source

    runners = {
        "TF1": run_tf1_loop,
        "TF2": run_tf2_loop,
        "TF3": run_tf3_loop,
        "TF4": run_tf4_loop,
        "TF5": run_tf5_loop,
        "TF6": run_tf6_loop,
        "TF7": run_tf7_loop,
    }
    return runners.get(detection.primary, run_tf7_loop)(ctx)


__all__ = ["agent_main", "TF_Detection", "TFDetectionResult"]
