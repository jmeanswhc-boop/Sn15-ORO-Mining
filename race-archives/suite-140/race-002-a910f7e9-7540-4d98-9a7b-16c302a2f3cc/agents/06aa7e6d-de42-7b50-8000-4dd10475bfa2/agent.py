from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)

_CAPABILITY_KEYS = (
    "decompose",
    "retrieve",
    "constrain",
    "prefer",
    "rank",
    "recover",
    "justify",
)

_CAPABILITY_INSTRUCTIONS = {
    "decompose": (
        "Break the shopper goal into ordered decisions and execute them in "
        "sequence. Do not skip earlier unmet steps."
    ),
    "retrieve": (
        "Gather candidates broadly enough to avoid misses, then refine."
    ),
    "constrain": (
        "Treat explicit hard limits as mandatory. Reject near-misses. Never "
        "trade away a hard constraint for a soft preference. Never ask the "
        "shopper to raise budget, change the gated spec, or relax firm rules."
    ),
    "prefer": (
        "Apply soft preferences only among candidates that already satisfy "
        "every hard constraint."
    ),
    "rank": (
        "When the goal needs ordering, rank valid candidates instead of "
        "stopping at one arbitrary pick."
    ),
    "recover": (
        "On failure, empty results, stock-out, or price change, switch to a "
        "different compliant product, re-check current stock and price, then "
        "finish with the terminal tool. Do not stop after one bad attempt and "
        "do not spend the step budget on negotiation dialogue."
    ),
    "justify": (
        "When a terminal tool requires structured justification, use only "
        "claim fields allowed by that tool's schema and values observed for "
        "the final item. Include every schema-required fact (typically price, "
        "in_stock, and the goal's primary specification field) plus one further "
        "reason copied from a viewed product fact key/value when the schema "
        "asks for it. Never invent field names that were not observed or "
        "allowed by the schema."
    ),
}

_BASE_SYSTEM = (
    "You are a shopping agent in a sealed evaluation environment. "
    "Use only the tools supplied in this task's policy_view. "
    "Continue until an observation reports done=true or the step budget ends. "
    "Never invent tools or product facts. "
    "Decision order: hard constraints, then soft preferences, then "
    "ranking/selection. "
    "Retrieve broadly, then narrow. Recover after failures. "
    "If the environment returns a non-empty shopper user_message and a "
    "dialogue tool is available in the supplied tools, answer that message "
    "with the dialogue tool before any other action. "
    "After answering, resume catalog actions; do not enter a dialogue-only "
    "loop and do not ask to relax firm constraints (budget, brand, gated "
    "spec, stock). "
    "If cart/stock/price becomes invalid, remove the bad item if needed, "
    "search for a different firm-valid product, inspect current stock and "
    "price, then place the order. Ending with no order fails the task. "
    "Once a firm-valid option is verified, finish with the supplied terminal "
    "tool instead of waiting for permission. "
    "Any structured justification must follow the terminal tool schema and "
    "observed facts only: claim field names must be allowlisted for this task, "
    "must include the goal's primary specification attribute, and must not "
    "invent prose labels. Prefer raw product facts over title keywords when "
    "checking the gated specification after a recovery switch."
)

_BRIEF_PROMPT = """Prepare a compact working brief for a shopping agent.
Return ONLY one JSON object. No markdown.

Rules:
- Not an exclusive family classifier. Multi-label capabilities are expected.
- Tool names below are authoritative; never invent tools.
- Extract hard constraints vs soft preferences from the public goal only.
- steps: 1-6 ordered actions the agent can take with the supplied tools.

Capabilities (0 or 1 each):
- decompose, retrieve, constrain, prefer, rank, recover, justify

Also include:
- steps, hard_constraints, soft_preferences, focus
- spec_claim_field: the single primary specification attribute field name
  implied by the shopper goal for structured claims (or null if none). Use a
  short snake_case name taken from the goal's own wording for the firm
  hardware/spec requirement — not a product identity, not a compound
  paraphrase, and not a marketing label.
- spec_claim_value: the firm target value for that attribute as stated in the
  goal (short string), or null if none.

JSON shape:
{{
  "decompose": 0,
  "retrieve": 1,
  "constrain": 1,
  "prefer": 0,
  "rank": 0,
  "recover": 1,
  "justify": 0,
  "steps": ["step 1", "step 2"],
  "hard_constraints": ["..."],
  "soft_preferences": ["..."],
  "spec_claim_field": null,
  "spec_claim_value": null,
  "focus": "one short sentence"
}}

Shopper goal:
{query}

Available tool names:
{tool_names}
"""

_JSON_SYSTEM = (
    "Return valid JSON only. No markdown. Multi-label fields are allowed. "
    "Never invent tool names that were not provided."
)

_RETRY_NO_TOOL_MESSAGE = (
    "The environment has not reported done=true yet. Choose one of the "
    "supplied tools to continue. Hard constraints first, then preferences, "
    "then ranking/selection."
)


def _provider() -> str:
    return getenv("INFERENCE_PROVIDER", "chutes")


def _primary_model() -> str:
    """Always use the provider's default DeepSeek model."""
    return _DEFAULT_MODELS.get(_provider(), _DEFAULT_MODELS["chutes"])


def _model_chain() -> list[str]:
    return [_primary_model()]


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


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
    """Discover a shopper-dialogue tool from supplied schemas only."""

    for entry in _tool_entries(tools):
        props = (entry["parameters"].get("properties") or {}) if entry["parameters"] else {}
        required = entry["parameters"].get("required") or []
        desc = entry["description"].lower()
        prop_names = {str(key).lower() for key in props.keys()}
        # Generic shape: send text content to the shopper/user.
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
    """Discover a finishing tool, preferring one with justification schema."""

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
    field = (items.get("properties") or {}).get("field") or {}
    return field if isinstance(field, dict) else {}


def _justification_field_enum(terminal_tool: dict[str, Any] | None) -> list[str]:
    """Read allowed claim fields from the terminal tool schema, if present."""

    field = _justification_field_schema(terminal_tool)
    enum_values = field.get("enum")
    if isinstance(enum_values, list):
        return [str(value) for value in enum_values if isinstance(value, str)]
    return []


def _justification_field_description(terminal_tool: dict[str, Any] | None) -> str:
    field = _justification_field_schema(terminal_tool)
    return str(field.get("description") or "")


def _soft_reason_fields_from_schema(terminal_tool: dict[str, Any] | None) -> list[str]:
    """Parse optional reason-field names from an explicit schema reason list.

    Only reads identifiers after a `reason(s):` cue. Does not tokenize free
    prose (which otherwise invents fake field names from English words).
    """

    desc = _justification_field_description(terminal_tool)
    if not desc:
        return []
    match = re.search(
        r"reason(?:s)?\s*:\s*([^\.]+)",
        desc,
        flags=re.I,
    )
    if not match:
        return []
    chunk = match.group(1)
    names = re.findall(r"\b[a-z][a-z0-9_]*\b", chunk.lower())
    # English filler from schema prose — not product attributes.
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
        "explicitly",
        "positive",
        "capabilities",
        "true",
        "not",
        "unknown",
        "absent",
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
        "products",
        "own",
        "facts",
        "fact",
        "at",
        "least",
        "including",
        "include",
        "such",
        "as",
        "eg",
        "etc",
    }
    ordered: list[str] = []
    for name in names:
        if name in stop or name in ordered:
            continue
        if name in {"price", "in_stock", "stock"}:
            continue
        ordered.append(name)
    return ordered


# Observation keys that are session/meta noise, not claimable product facts.
_FACT_META_KEYS = frozenset(
    {
        "title",
        "query",
        "options",
        "sku",
        "product_id",
        "currency",
        "error",
        "done",
        "observation",
        "message",
        "content",
        "results",
        "products",
        "items",
        "candidates",
        "variants",
        "facts",
        "ok",
        "success",
        "status",
        "call_id",
        "session_id",
        "user_message",
    }
)


def _terminal_requires_justification(terminal_tool: dict[str, Any] | None) -> bool:
    if not terminal_tool:
        return False
    parameters = terminal_tool.get("parameters") or {}
    required = parameters.get("required") or []
    return isinstance(required, list) and "justification" in required


def _user_message_text(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    user_message = result.get("user_message")
    if isinstance(user_message, dict):
        return str(user_message.get("content") or "").strip()
    if isinstance(user_message, str):
        return user_message.strip()
    return ""


def _normalize_to_allowed_field(field: Any, allowed: list[str]) -> str | None:
    """Map a claim field onto an allowlist without suite-specific aliases."""

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


def _schema_requires_extra_reason(terminal_tool: dict[str, Any] | None) -> bool:
    """True when the schema asks for a further fact-grounded reason claim."""

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


def _normalize_price_claim_value(value: str) -> str:
    """Keep a bare numeric price; strip currency words/symbols."""

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


def _fact_pairs_from_observation(observation: Any) -> list[tuple[str, str]]:
    """Collect (field, value) pairs from tool observations (facts-first)."""

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(field: Any, value: Any) -> None:
        if not isinstance(field, str):
            return
        key = field.strip()
        key_l = key.lower()
        if not key or key_l in seen or key_l in _FACT_META_KEYS:
            return
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (int, float)):
            text = str(value)
        elif isinstance(value, str):
            text = value.strip()
        else:
            return
        if not text or len(text) > 120:
            return
        seen.add(key_l)
        pairs.append((key, text))

    def _walk(node: Any, *, depth: int = 0) -> None:
        if depth > 5 or node is None:
            return
        if isinstance(node, dict):
            facts = node.get("facts")
            if isinstance(facts, dict):
                for key, value in facts.items():
                    _add(key, value)
            elif isinstance(facts, str):
                try:
                    parsed = json.loads(facts)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        _add(key, value)
            # Current-state scalars that appear beside facts on commerce tools.
            for key, value in node.items():
                if key in {"facts", "variants", "results", "title", "products", "items"}:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    _add(key, value)
                elif isinstance(value, (dict, list)):
                    _walk(value, depth=depth + 1)
            variants = node.get("variants")
            if isinstance(variants, list):
                for variant in variants[:3]:
                    if isinstance(variant, dict):
                        for key, value in variant.items():
                            if isinstance(value, (str, int, float, bool)):
                                _add(key, value)
        elif isinstance(node, list):
            for item in node[:8]:
                _walk(item, depth=depth + 1)

    _walk(observation)
    return pairs


def _merge_fact_context(
    current: dict[str, str], observation: Any
) -> dict[str, str]:
    merged = dict(current)
    for key, value in _fact_pairs_from_observation(observation):
        merged[key] = value
    return merged


def _align_value_to_facts(value: str, fact_context: dict[str, str]) -> str:
    needle = value.strip().lower()
    if not needle:
        return value
    for observed in fact_context.values():
        if observed.strip().lower() == needle:
            return observed.strip()
    return value


def _fact_value_for_field(fact_context: dict[str, str], field: str) -> str | None:
    """Resolve an observed fact value for a claim field (exact, then fuzzy)."""

    if field in fact_context:
        return fact_context[field]
    field_l = field.lower().replace(" ", "_")
    for key, value in fact_context.items():
        if key.lower().replace(" ", "_") == field_l:
            return value
    return None


def _match_fact_key(raw_field: str, fact_context: dict[str, str]) -> str | None:
    """Map a claim field name onto an observed fact key when uniquely possible."""

    raw_key = " ".join(
        raw_field.strip().lower().replace("-", " ").replace("_", " ").split()
    ).replace(" ", "_")
    if not raw_key:
        return None
    exact = {
        key.lower().replace(" ", "_"): key for key in fact_context if isinstance(key, str)
    }
    if raw_key in exact:
        return exact[raw_key]
    hits = [
        key
        for norm, key in exact.items()
        if norm and (norm in raw_key or raw_key in norm)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _pick_extra_reason_claim(
    fact_context: dict[str, str],
    *,
    reserved: set[str],
    occupied_values: set[str] | None = None,
) -> dict[str, str] | None:
    """Choose one compact observed fact as an extra schema-required reason."""

    occupied = {v.strip().lower() for v in (occupied_values or set()) if v}
    reserved_l = {r.lower().replace(" ", "_") for r in reserved}
    candidates: list[tuple[int, str, str]] = []
    for key, value in fact_context.items():
        key_l = key.lower().replace(" ", "_")
        if key_l in reserved_l or key_l in _FACT_META_KEYS:
            continue
        if len(value) > 80:
            continue
        if value.strip().lower() in occupied:
            continue
        # Prefer compact attribute-like values; no product-attribute name list.
        score = len(value) + (0 if 3 <= len(value) <= 40 else 20)
        candidates.append((score, key, value))
    if not candidates:
        return None
    candidates.sort()
    _, key, value = candidates[0]
    return {"field": key, "value": value}


def _build_claim_allowlist(
    terminal_tool: dict[str, Any] | None,
    *,
    spec_claim_field: str | None,
) -> list[str]:
    enum_fields = _justification_field_enum(terminal_tool)
    if enum_fields:
        return enum_fields

    allowed: list[str] = []
    desc = _justification_field_description(terminal_tool).lower()
    if "price" in desc or not desc:
        allowed.append("price")
    if "in_stock" in desc or "stock" in desc or not desc:
        allowed.append("in_stock")
    if isinstance(spec_claim_field, str) and spec_claim_field.strip():
        allowed.append(spec_claim_field.strip())
    for name in _soft_reason_fields_from_schema(terminal_tool):
        if name not in allowed:
            allowed.append(name)
    out: list[str] = []
    for name in allowed:
        if name and name not in out:
            out.append(name)
    return out


def _sanitize_terminal_args(
    args: dict[str, Any],
    *,
    terminal_tool: dict[str, Any] | None,
    spec_claim_field: str | None = None,
    fact_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Sanitize terminal-tool args using schema + observed product facts only."""

    if not isinstance(args, dict) or not terminal_tool:
        return args

    # Ranking-style tasks: justification is optional. Omitting it avoids
    # failing the hard gate on untruthful optional claims.
    if not _terminal_requires_justification(terminal_tool):
        if "justification" in args:
            return {key: value for key, value in args.items() if key != "justification"}
        return args

    justification = args.get("justification")
    if not isinstance(justification, dict):
        return args
    claims = justification.get("claims")
    if not isinstance(claims, list):
        return args

    allowed = _build_claim_allowlist(
        terminal_tool, spec_claim_field=spec_claim_field
    )
    if not allowed:
        return args

    facts = {
        str(key): str(value)
        for key, value in (fact_context or {}).items()
        if isinstance(key, str) and isinstance(value, str) and key and value
    }
    soft = set(_soft_reason_fields_from_schema(terminal_tool))
    needs_extra = _schema_requires_extra_reason(terminal_tool)
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    soft_kept = False
    pending_spec: dict[str, str] | None = None

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        raw_field = claim.get("field")
        value = claim.get("value")
        if value is None or not isinstance(raw_field, str):
            continue
        value_s = str(value).strip()
        if not value_s:
            continue
        raw_norm = " ".join(
            raw_field.strip().lower().replace("-", " ").replace("_", " ").split()
        )
        raw_key = raw_norm.replace(" ", "_")

        field = _normalize_to_allowed_field(raw_field, allowed)
        if (
            field is None
            and isinstance(spec_claim_field, str)
            and spec_claim_field in allowed
            and raw_key not in {"price", "in_stock"}
            and raw_key not in soft
        ):
            # Model used an alternate label for the brief's primary spec field.
            field = spec_claim_field
        # Extra reason: accept only when the field is an observed fact key.
        if field is None and needs_extra:
            field = _match_fact_key(raw_field, facts)

        if field is None:
            continue
        if field in seen:
            continue

        if field == "price":
            observed_price = _fact_value_for_field(facts, "price")
            value_s = _normalize_price_claim_value(
                observed_price if observed_price is not None else value_s
            )
        elif field == "in_stock":
            observed_stock = _fact_value_for_field(facts, "in_stock")
            value_s = _normalize_stock_claim_value(
                observed_stock if observed_stock is not None else value_s
            )
        else:
            observed = _fact_value_for_field(facts, field)
            if observed is not None:
                value_s = observed
            else:
                # Spec/extra claims must be grounded; drop ungrounded guesses.
                if field not in {"price", "in_stock"} and field not in allowed:
                    continue
                value_s = _align_value_to_facts(value_s, facts)
                if (
                    needs_extra
                    and field not in allowed
                    and value_s.strip().lower()
                    not in {v.strip().lower() for v in facts.values()}
                ):
                    continue

        is_soft = field in soft or (
            needs_extra
            and field not in {"price", "in_stock"}
            and field != spec_claim_field
        )
        if is_soft:
            if soft_kept:
                continue
            soft_kept = True

        if (
            isinstance(spec_claim_field, str)
            and field == spec_claim_field
            and pending_spec is None
            and field not in {"price", "in_stock"}
        ):
            pending_spec = {"field": field, "value": value_s}
            seen.add(field)
            continue
        cleaned.append({"field": field, "value": value_s})
        seen.add(field)

    if pending_spec and pending_spec["field"] not in {c["field"] for c in cleaned}:
        cleaned.append(pending_spec)

    # Backfill schema-typical state claims from observations when omitted.
    for required_field in ("price", "in_stock"):
        if required_field in seen or required_field not in allowed:
            continue
        observed = _fact_value_for_field(facts, required_field)
        if observed is None:
            continue
        if required_field == "price":
            cleaned.append(
                {"field": "price", "value": _normalize_price_claim_value(observed)}
            )
        else:
            cleaned.append(
                {
                    "field": "in_stock",
                    "value": _normalize_stock_claim_value(observed),
                }
            )
        seen.add(required_field)

    if (
        isinstance(spec_claim_field, str)
        and spec_claim_field
        and spec_claim_field in allowed
        and spec_claim_field not in seen
    ):
        observed_spec = _fact_value_for_field(facts, spec_claim_field)
        if observed_spec is not None:
            cleaned.append({"field": spec_claim_field, "value": observed_spec})
            seen.add(spec_claim_field)

    if needs_extra and not soft_kept:
        reserved = set(seen) | {"price", "in_stock"}
        if isinstance(spec_claim_field, str) and spec_claim_field:
            reserved.add(spec_claim_field)
        extra = _pick_extra_reason_claim(
            facts,
            reserved=reserved,
            occupied_values={c["value"] for c in cleaned},
        )
        if extra:
            cleaned.append(extra)

    priority = {"price": 0, "in_stock": 1}
    if isinstance(spec_claim_field, str) and spec_claim_field:
        priority[spec_claim_field] = 2

    def _sort_key(claim: dict[str, str]) -> tuple[int, str]:
        return (priority.get(claim["field"], 10), claim["field"])

    cleaned = sorted(cleaned, key=_sort_key)

    if not cleaned:
        return args
    return {
        **args,
        "justification": {
            **justification,
            "claims": cleaned,
        },
    }


def _intervention_prompt(dialogue_tool: str) -> str:
    return (
        "REQUIRED NEXT ACTION: the environment returned a non-empty shopper "
        f"user_message. Call the `{dialogue_tool}` tool now with a short direct "
        "answer to that shopper message. Confirm any hard requirements they "
        "asked about using observed facts. Acknowledge updated limits if they "
        "changed constraints. Do not call any other tool until this dialogue "
        f"tool (`{dialogue_tool}`) has answered them."
    )


def _contract_hint(
    *,
    dialogue_tool: str | None,
    terminal_tool: dict[str, Any] | None,
    spec_claim_field: str | None = None,
) -> str:
    parts = [
        "Contract reminders derived from this task's tool schemas:",
        "Use only supplied tools.",
    ]
    if dialogue_tool:
        parts.append(
            f"Dialogue tool available: `{dialogue_tool}`. When a non-empty "
            f"shopper user_message arrives after a non-dialogue action, answer "
            f"once with `{dialogue_tool}`, then resume task progress. Do not "
            "enter a dialogue-only loop. Do not use dialogue to ask the "
            "shopper to relax firm constraints."
        )
    if terminal_tool:
        name = terminal_tool["name"]
        parts.append(
            f"Terminal tool available: `{name}`. Use it to finish when the goal "
            "is satisfied; do not wait for permission after verification."
        )
        if _terminal_requires_justification(terminal_tool):
            allowed = _build_claim_allowlist(
                terminal_tool, spec_claim_field=spec_claim_field
            )
            field_desc = _justification_field_description(terminal_tool)
            extra = ""
            if _schema_requires_extra_reason(terminal_tool):
                extra = (
                    " Also include exactly one further reason claim whose "
                    "field and value are copied from a viewed product fact "
                    "key/value. Never invent field names that were not observed."
                )
            parts.append(
                f"`{name}` requires structured justification claims. "
                f"Schema field guidance: {field_desc} "
                f"Prefer only these claim fields: {', '.join(allowed)}. "
                "Include price, in_stock, and the goal's primary specification "
                "field with values observed for the FINAL item. Price claims "
                "must be numeric only; stock claims must be true/false."
                + extra
            )
        else:
            parts.append(
                f"Justification on `{name}` is optional for this task. Prefer "
                "omitting justification unless the shopper explicitly asked for "
                "a reason; untruthful optional claims fail the episode."
            )
    return " ".join(parts)


def _as_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 0


def _as_str_list(value: Any, limit: int = 6) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [" ".join(value.split())][:limit]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(" ".join(item.split()))
        elif isinstance(item, (int, float)):
            items.append(str(item))
        if len(items) >= limit:
            break
    return items


def _empty_brief() -> dict[str, Any]:
    brief = {key: 0 for key in _CAPABILITY_KEYS}
    brief.update(
        {
            "focus": "",
            "steps": [],
            "hard_constraints": [],
            "soft_preferences": [],
            "spec_claim_field": None,
            "spec_claim_value": None,
            "source": "empty",
        }
    )
    return brief


def _normalize_spec_claim_field(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if not cleaned or cleaned in {"null", "none"}:
        return None
    # Reject state fields and single English filler tokens — not attribute names.
    if cleaned in {"price", "in_stock", "stock", "plus", "reason", "feature", "why"}:
        return None
    if "_" not in cleaned and cleaned in {
        "also",
        "extra",
        "other",
        "detail",
        "details",
        "summary",
        "evidence",
        "note",
        "comment",
    }:
        return None
    return cleaned


def _normalize_spec_claim_value(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    if not cleaned or cleaned.lower() in {"null", "none"}:
        return None
    return cleaned[:80]


def _normalize_brief(raw: dict[str, Any] | None, source: str) -> dict[str, Any]:
    brief = _empty_brief()
    brief["source"] = source
    if not isinstance(raw, dict):
        return brief
    for key in _CAPABILITY_KEYS:
        brief[key] = _as_flag(raw.get(key))
    focus = raw.get("focus")
    if isinstance(focus, str):
        brief["focus"] = " ".join(focus.split())
    brief["steps"] = _as_str_list(raw.get("steps"))
    brief["hard_constraints"] = _as_str_list(raw.get("hard_constraints"))
    brief["soft_preferences"] = _as_str_list(raw.get("soft_preferences"))
    brief["spec_claim_field"] = _normalize_spec_claim_field(raw.get("spec_claim_field"))
    brief["spec_claim_value"] = _normalize_spec_claim_value(raw.get("spec_claim_value"))
    return brief


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
    payload_base: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
    }
    if tools is not None:
        payload_base["tools"] = tools
    if tool_choice is not None:
        payload_base["tool_choice"] = tool_choice

    for model in _model_chain():
        payload = {**payload_base, "model": model}
        inference = _proxy.post("/inference/chat/completions", json_data=payload)
        if inference is None:
            continue
        try:
            if inference["choices"][0]["message"] is not None:
                return inference
        except (KeyError, IndexError, TypeError):
            continue
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


def _heuristic_brief(query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Generic linguistic fallback when the LLM brief call fails."""

    brief = _empty_brief()
    brief["source"] = "heuristic"
    q = query.lower()
    names = {name.lower() for name in _tool_names(tools)}

    steps = _split_query_steps(query)
    if len(steps) >= 2:
        brief["decompose"] = 1
        brief["steps"] = steps
    else:
        brief["steps"] = steps or [
            query.strip() or "Satisfy the shopper goal with the available tools."
        ]

    if any(k in q for k in ("find", "search", "options", "candidates", "look for")):
        brief["retrieve"] = 1
    if any(
        k in q
        for k in ("must", "only", "under", "less than", "at least", "exactly", "between")
    ):
        brief["constrain"] = 1
    if any(k in q for k in ("prefer", "rather", "ideally", "if possible", "would like")):
        brief["prefer"] = 1
    if any(k in q for k in ("rank", "order", "sort", "top ", "best to worst")) or any(
        "rank" in n for n in names
    ):
        brief["rank"] = 1
    if any(k in q for k in ("again", "instead", "unavailable", "failed", "try another")):
        brief["recover"] = 1
    if any(k in q for k in ("why", "explain", "justify", "evidence", "walk me through")):
        brief["justify"] = 1

    brief["recover"] = 1
    if not any(brief[key] for key in _CAPABILITY_KEYS if key != "recover"):
        brief["retrieve"] = 1
        brief["constrain"] = 1

    active = [key for key in _CAPABILITY_KEYS if brief[key]]
    brief["focus"] = "Heuristic brief focusing on: " + ", ".join(active)
    # Keep constraints/preferences/spec empty here; avoid suite-specific extractors.
    brief["hard_constraints"] = []
    brief["soft_preferences"] = []
    return brief


def _build_task_brief(query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = _BRIEF_PROMPT.format(
        query=query,
        tool_names=", ".join(_tool_names(tools)) or "(none provided)",
    )
    raw = _llm_json(prompt, temperature=0, attempts=2)
    if raw is None:
        return _heuristic_brief(query, tools)

    brief = _normalize_brief(raw, source="llm")
    if not any(brief[key] for key in _CAPABILITY_KEYS):
        return _heuristic_brief(query, tools)
    if not brief["steps"]:
        brief["steps"] = _heuristic_brief(query, tools)["steps"]
    brief["recover"] = 1
    return brief


def _brief_to_system_hint(brief: dict[str, Any]) -> str:
    parts = [
        "Episode operating rules:",
        "1) Use only supplied tools.",
        "2) Hard constraints before soft preferences before ranking/selection.",
        "3) Retrieve broadly, then refine.",
        "4) Recover on failure/empty/change.",
        "5) Ground outcomes in observations.",
        "6) Continue until done=true or the step budget ends.",
    ]
    active = [key for key in _CAPABILITY_KEYS if brief.get(key)]
    if active:
        parts.append("Active capabilities:")
        for key in active:
            parts.append(f"- {key}: {_CAPABILITY_INSTRUCTIONS[key]}")
    steps = brief.get("steps") or []
    if steps:
        parts.append("Ordered checklist:")
        for index, step in enumerate(steps, start=1):
            parts.append(f"{index}. {step}")
    hard = brief.get("hard_constraints") or []
    if hard:
        parts.append("Hard constraints: " + "; ".join(hard))
    soft = brief.get("soft_preferences") or []
    if soft:
        parts.append("Soft preferences: " + "; ".join(soft))
    spec_field = brief.get("spec_claim_field")
    spec_value = brief.get("spec_claim_value")
    if isinstance(spec_field, str) and spec_field:
        if isinstance(spec_value, str) and spec_value:
            parts.append(
                f"Primary specification claim: field `{spec_field}` with firm "
                f"target `{spec_value}` (must be supported by viewed raw facts "
                "on the FINAL item, not title keywords alone)."
            )
        else:
            parts.append(
                f"Primary specification claim field: `{spec_field}` (must be "
                "supported by viewed raw facts on the FINAL item)."
            )
    focus = brief.get("focus")
    if isinstance(focus, str) and focus.strip():
        parts.append(f"Working focus: {focus.strip()}")
    return " ".join(parts)


def _observation_blob(observation: Any) -> str:
    try:
        return json.dumps(observation, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        return str(observation).lower()


def _looks_failed(observation: Any) -> bool:
    text = _observation_blob(observation)
    failure_tokens = (
        "error",
        "failed",
        "failure",
        "unavailable",
        "not found",
        "no results",
        "empty",
        "invalid",
        "timeout",
        "try again",
        "retry",
    )
    return any(token in text for token in failure_tokens)


def _looks_unavailable(observation: Any) -> bool:
    """Detect stock-out / unavailable cart or variant observations."""

    text = _observation_blob(observation)
    tokens = (
        "out of stock",
        "sold out",
        "not available",
        "unavailable",
        '"in_stock": false',
        '"in_stock":false',
        '"in_stock": 0',
        '"in_stock":0',
    )
    return any(token in text for token in tokens)


def _recovery_prompt(
    *,
    dialogue_tool: str | None,
    terminal_tool: dict[str, Any] | None,
) -> str:
    terminal_name = terminal_tool["name"] if terminal_tool else "the terminal tool"
    dialogue_bit = (
        f" Do not call `{dialogue_tool}` unless a new shopper user_message is waiting."
        if dialogue_tool
        else ""
    )
    return (
        "RECOVERY REQUIRED: a prior candidate became invalid (stock/price/"
        "failure). Keep every firm constraint unchanged — never ask to raise "
        "budget, change the gated spec, switch required brand, or otherwise "
        "relax hard rules. Remove the invalid cart item if still present, "
        "search/filter for a DIFFERENT product that still satisfies all firm "
        "constraints, view it, inspect current stock and price, then call "
        f"`{terminal_name}` to finish. Ending with no order fails."
        f"{dialogue_bit}"
    )


def _coaching_message(
    brief: dict[str, Any],
    *,
    turn: int,
    max_steps: int,
    observations: list[Any],
    done: bool,
    pending_shopper_reply: bool,
    dialogue_tool: str | None,
    terminal_tool: dict[str, Any] | None,
    recovery_active: bool = False,
    consecutive_dialogue: int = 0,
) -> str | None:
    if done:
        return None

    remaining = max_steps - turn
    if pending_shopper_reply and dialogue_tool:
        return (
            _intervention_prompt(dialogue_tool)
            + f" Episode status: turn {turn}/{max_steps} ({remaining} left)."
        )

    tips = [
        f"Episode status: turn {turn}/{max_steps} ({remaining} left). "
        "done is false; choose a supplied tool. Use the latest observation."
    ]
    failed_now = any(
        _looks_failed(obs) or _looks_unavailable(obs) for obs in observations
    )
    if recovery_active or failed_now or brief.get("recover"):
        tips.append(
            _recovery_prompt(
                dialogue_tool=dialogue_tool, terminal_tool=terminal_tool
            )
        )
    if consecutive_dialogue >= 1 and dialogue_tool:
        tips.append(
            f"Dialogue pause: you already messaged the shopper. Do not call "
            f"`{dialogue_tool}` again unless a new user_message arrives. Use "
            "search/filter/view/inspect/cart/terminal tools to make progress."
        )
    if brief.get("retrieve") and turn <= 2:
        tips.append("Retrieval: gather candidates before finalizing.")
    if brief.get("constrain") or brief.get("hard_constraints"):
        hard = brief.get("hard_constraints") or []
        detail = f" Hard constraints: {'; '.join(hard)}." if hard else ""
        tips.append(
            "Constraints: keep only fully compliant candidates. Firm limits "
            "are non-negotiable; do not ask the shopper to relax them."
            + detail
        )
    if brief.get("prefer") or brief.get("soft_preferences"):
        soft = brief.get("soft_preferences") or []
        detail = f" Soft preferences: {'; '.join(soft)}." if soft else ""
        tips.append(
            "Preferences: apply only among hard-valid candidates." + detail
        )
    if brief.get("rank"):
        tips.append(
            "Ranking: compare the allowed candidates with the supplied compare/"
            "view tools, apply the stated priority order, then finish. Do not "
            "add optional justification unless the shopper asked for a reason."
        )
    if brief.get("decompose") and brief.get("steps"):
        checklist = "; ".join(
            f"{index}. {step}" for index, step in enumerate(brief["steps"], start=1)
        )
        tips.append(f"Decomposition: advance the checklist: {checklist}.")
    if terminal_tool and _terminal_requires_justification(terminal_tool):
        allowed = _build_claim_allowlist(
            terminal_tool, spec_claim_field=brief.get("spec_claim_field")
        )
        spec_field = brief.get("spec_claim_field")
        spec_value = brief.get("spec_claim_value")
        spec_bit = ""
        if isinstance(spec_field, str) and spec_field:
            if isinstance(spec_value, str) and spec_value:
                spec_bit = (
                    f" Required spec claim: `{spec_field}`=`{spec_value}` from "
                    "viewed raw facts on the FINAL item (not the title alone). "
                    "If the latest viewed item does not support that firm spec, "
                    "do not order it — keep searching."
                )
            else:
                spec_bit = (
                    f" Required spec claim field: `{spec_field}` with a value "
                    "taken from viewed raw facts on the FINAL item."
                )
        extra_bit = ""
        if _schema_requires_extra_reason(terminal_tool):
            extra_bit = (
                " Schema also requires one further reason claim grounded in a "
                "viewed product fact field/value. Never invent field names that "
                "were not observed. Price values must be numeric only."
            )
        tips.append(
            "Justification required: inspect the FINAL item, then submit "
            f"claims using only [{', '.join(allowed)}] plus one observed-fact "
            "reason when the schema asks for it. Include price, in_stock, and "
            "the primary specification field. Never invent unsupported fields."
            + spec_bit
            + extra_bit
        )
    elif terminal_tool:
        tips.append(
            f"Justification is optional on `{terminal_tool['name']}`. Prefer "
            "omitting it unless a reason was explicitly requested."
        )
    if terminal_tool and (remaining <= 12 or recovery_active):
        tips.append(
            f"Budget is limited: after verifying current stock and price on a "
            f"firm-valid item, call `{terminal_tool['name']}` now instead of "
            "more dialogue or permission-seeking."
        )
    tips.append("Next: one productive supplied tool call.")
    return " ".join(tips)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one ORO Bench episode using dynamic tools for any TF1-TF7 task."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    query = str(policy.get("query", ""))
    tools = list(policy.get("tools") or [])

    dialogue_tool = _find_dialogue_tool(tools)
    terminal_tool = _find_terminal_tool(tools)

    brief = _build_task_brief(query, tools)
    if terminal_tool and _terminal_requires_justification(terminal_tool):
        brief["justify"] = 1
    elif terminal_tool:
        brief["justify"] = 0
    system_prompt = (
        f"{_BASE_SYSTEM} {_brief_to_system_hint(brief)} "
        f"{_contract_hint(dialogue_tool=dialogue_tool, terminal_tool=terminal_tool, spec_claim_field=brief.get('spec_claim_field'))}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "task_brief",
            "task_brief": brief,
            "discovered_tools": {
                "dialogue_tool": dialogue_tool,
                "terminal_tool": terminal_tool["name"] if terminal_tool else None,
                "justification_required": bool(
                    terminal_tool and _terminal_requires_justification(terminal_tool)
                ),
                "justification_fields": _build_claim_allowlist(
                    terminal_tool, spec_claim_field=brief.get("spec_claim_field")
                ),
                "spec_claim_field": brief.get("spec_claim_field"),
                "spec_claim_value": brief.get("spec_claim_value"),
            },
        }
    ]
    pending_shopper_reply = False
    recovery_active = False
    consecutive_dialogue = 0
    fact_context: dict[str, str] = {}

    for turn in range(1, max_steps + 1):
        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        remaining = max_steps - turn + 1
        for _attempt in range(5):
            inference = _chat_completion(
                messages,
                tools=tools,
                tool_choice="required",
                temperature=0,
            )
            if inference is None:
                raise RuntimeError("inference request failed")
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                dialogue.append({"role": "assistant", "content": assistant_content})
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": _RETRY_NO_TOOL_MESSAGE},
                    ]
                )
                continue

            chosen = _tool_call_name(tool_calls[0])
            if (
                pending_shopper_reply
                and dialogue_tool
                and chosen != dialogue_tool
            ):
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "rejected_tool": chosen,
                        "rejected_for": "unanswered_shopper_message",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            _intervention_prompt(dialogue_tool)
                            + f" Previous choice `{chosen}` is not allowed yet."
                        ),
                    }
                )
                tool_calls = []
                continue
            # After stock/price failure, or after an answered dialogue turn,
            # block voluntary messaging that burns the step budget.
            block_voluntary_dialogue = bool(
                dialogue_tool
                and chosen == dialogue_tool
                and not pending_shopper_reply
                and (
                    recovery_active
                    or consecutive_dialogue >= 1
                    or remaining <= 12
                )
            )
            if block_voluntary_dialogue:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "rejected_tool": chosen,
                        "rejected_for": "voluntary_dialogue_blocked",
                    }
                )
                block_reason = _recovery_prompt(
                    dialogue_tool=dialogue_tool, terminal_tool=terminal_tool
                )
                if not recovery_active:
                    terminal_name = (
                        terminal_tool["name"] if terminal_tool else "the terminal tool"
                    )
                    block_reason = (
                        f"Do not call `{dialogue_tool}` now — no new shopper "
                        "user_message is waiting. Firm constraints stay fixed; "
                        "do not ask to relax them. Continue with search/filter/"
                        f"view/inspect/cart/`{terminal_name}`."
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            block_reason
                            + f" Previous choice `{chosen}` is not allowed yet."
                        ),
                    }
                )
                tool_calls = []
                continue
            break
        else:
            raise RuntimeError("model returned no valid tool call before completion")

        used_dialogue = bool(
            dialogue_tool
            and any(_tool_call_name(call) == dialogue_tool for call in tool_calls)
        )
        if used_dialogue:
            pending_shopper_reply = False
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
        group_id = f"{problem_id}-turn-{turn}"
        calls_payload = []
        terminal_name = terminal_tool["name"] if terminal_tool else None
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _arguments(call)
            if terminal_name and name == terminal_name:
                args = _sanitize_terminal_args(
                    args,
                    terminal_tool=terminal_tool,
                    spec_claim_field=brief.get("spec_claim_field"),
                    fact_context=fact_context,
                )
            calls_payload.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": calls_payload,
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        observations: list[Any] = []
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result["observation"]
            observations.append(observation)
            fact_context = _merge_fact_context(fact_context, observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(observation),
                }
            )

        if any(
            _looks_failed(obs) or _looks_unavailable(obs) for obs in observations
        ):
            recovery_active = True

        shopper_text = _user_message_text(result)
        if shopper_text:
            # Avoid dialogue loops: only force a reply when the shopper speaks
            # after a non-dialogue action.
            pending_shopper_reply = bool(dialogue_tool) and not used_dialogue
            messages.append({"role": "user", "content": shopper_text})

        done = any(
            isinstance(obs, dict) and obs.get("done") for obs in observations
        )
        if done:
            break

        coach = _coaching_message(
            brief,
            turn=turn,
            max_steps=max_steps,
            observations=observations,
            done=False,
            pending_shopper_reply=pending_shopper_reply,
            dialogue_tool=dialogue_tool,
            terminal_tool=terminal_tool,
            recovery_active=recovery_active,
            consecutive_dialogue=consecutive_dialogue,
        )
        if coach:
            messages.append({"role": "user", "content": coach})

    return dialogue


__all__ = ["agent_main"]
