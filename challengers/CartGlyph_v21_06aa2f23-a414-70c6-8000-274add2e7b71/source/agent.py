"""ORO Bench agent Plan A: working-memory controller with a turn-budgeted phase loop."""

from __future__ import annotations

import json
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_PRIMARY_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_FALLBACK_MODELS = {
    "chutes": "Qwen/Qwen3.6-27B-TEE",
    "openrouter": "qwen/qwen3.6-27b",
}
_MAX_OBSERVATION_CHARS = 14000
_MAX_LIST_ITEMS = 16
_MAX_CONTEXT_MESSAGES = 16
_PRIORITY_KEYS = (
    "done",
    "error",
    "success",
    "ok",
    "status",
    "product_id",
    "sku",
    "id",
    "price",
    "stock",
    "quantity",
    "available",
    "title",
    "name",
    "attributes",
    "sku_options",
    "items",
    "results",
    "products",
    "cart",
    "observation",
)
_ID_KEYS = frozenset(
    {"product_id", "sku", "item_id", "variant_id", "shop_id", "offer_id"}
)
_SEARCH_ARG_KEYS = ("q", "query", "search", "keywords", "text", "term")
_inference_proxy = ProxyClient(timeout=150, max_retries=1)
_environment_proxy = ProxyClient(timeout=75, max_retries=2)

_PHASE_PLAYBOOKS = {
    "SEEK": (
        "Search with value terms taken from the current goal. Start precise; if a "
        "search already missed, change the least important term. Never search attribute "
        "field names. Do not invent identities."
    ),
    "VERIFY": (
        "Inspect published details, stock, or cart for identities already observed. "
        "If several actions are allowed this turn, inspect distinct candidates together."
    ),
    "COMPARE": (
        "Compare at least two observed candidates on hard constraints first, then "
        "soft preferences. Cite concrete observed fields. Eliminate any candidate that "
        "fails a hard constraint."
    ),
    "EXPLAIN": (
        "Use the published message tool. Name the chosen identity and the observed facts "
        "that satisfy the current goal. Do not claim unsupported superiority."
    ),
    "COMMIT": (
        "Take the published completion action using only an observed identity. If a cart "
        "tool exists, the cart must already match the choice."
    ),
}


def _provider() -> str:
    return getenv("INFERENCE_PROVIDER", "chutes")


def _models() -> tuple[str, str]:
    provider = _provider()
    primary = getenv("AGENT_MODEL") or _PRIMARY_MODELS.get(
        provider, _PRIMARY_MODELS["chutes"]
    )
    fallback = getenv("AGENT_FALLBACK_MODEL") or _FALLBACK_MODELS.get(
        provider, _FALLBACK_MODELS["chutes"]
    )
    return primary, fallback


def _tool_entries(tools: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if item.get("type") == "function" else item
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            entries.append(fn)
    return entries


def _tool_names(tools: list[Any]) -> set[str]:
    return {entry["name"] for entry in _tool_entries(tools)}


def _tokens(text: str) -> set[str]:
    token = ""
    found: set[str] = set()
    for char in text.lower():
        if char.isalnum():
            token += char
        elif token:
            found.add(token)
            token = ""
    if token:
        found.add(token)
    return found


def _tool_guide(tools: list[Any]) -> str:
    lines: list[str] = []
    for entry in _tool_entries(tools):
        name = entry["name"]
        description = str(entry.get("description") or "").strip()
        params = entry.get("parameters") if isinstance(entry.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        fields: list[str] = []
        for key, spec in props.items():
            mark = "required" if key in required else "optional"
            hint = ""
            if isinstance(spec, dict):
                hint = str(spec.get("description") or spec.get("type") or "").strip()
            fields.append(f"{key} ({mark}{'; ' + hint if hint else ''})")
        detail = f"{name}: {description}" if description else name
        if fields:
            detail += " | args: " + "; ".join(fields)
        lines.append(f"- {detail}")
    return "\n".join(lines) if lines else "- (no tools published for this task)"


def _tool_roles(tools: list[Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for entry in _tool_entries(tools):
        name = entry["name"]
        blob = f"{name} {entry.get('description') or ''}"
        tokens = _tokens(blob)
        if tokens & {"checkout", "purchase", "finalize", "submit"} or "order" in tokens:
            roles[name] = "terminal"
        elif tokens & {"message", "communicate", "explain", "respond", "reply"}:
            roles[name] = "message"
        elif tokens & {"inspect", "detail", "cart", "sku"} or "view" in tokens:
            roles[name] = "inspect"
        elif tokens & {"search", "find", "browse", "list", "query"}:
            roles[name] = "search"
        else:
            roles[name] = "act"
    return roles


def _system_prompt(policy: dict[str, Any], query: str) -> str:
    max_steps = int(policy.get("max_steps") or 1)
    max_calls = int(policy.get("max_calls_per_turn") or 1)
    return (
        "You are a shopping agent in a generated commerce environment. Use only "
        "published tools and observed facts. Never invent a product ID, SKU, price, "
        "feature, stock state, or shopper preference.\n\n"
        f"Turn budget: {max_steps}. Actions allowed per turn: {max_calls}.\n"
        "Published tools:\n"
        f"{_tool_guide(policy.get('tools') or [])}\n\n"
        "Follow the controller phase in the latest working-memory note. Independent "
        "of phase, keep this method:\n"
        "1. Treat later shopper messages as amendments that override earlier goal text.\n"
        "2. Search values, not field names. After an empty result, reformulate.\n"
        "3. Verify details and stock before committing. Compare candidates when the "
        "goal involves quality, preference, ranking, or multiple viable options.\n"
        "4. Never treat an unverified requirement as satisfied.\n"
        "5. If a message tool exists and enough turns remain, explain with evidence "
        "before the terminal action.\n"
        "6. On error, stock change, or shopper correction, change the relevant "
        "variable; never repeat an identical failed action.\n"
        "The episode ends only when an observation has done=true."
    )


def _shrink(value: Any) -> Any:
    if isinstance(value, dict):
        keys = [key for key in _PRIORITY_KEYS if key in value]
        keys.extend(key for key in value if key not in _PRIORITY_KEYS)
        return {str(key): _shrink(value[key]) for key in keys}
    if isinstance(value, list):
        items = [_shrink(item) for item in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append({"truncated_items": len(value) - _MAX_LIST_ITEMS})
        return items
    if isinstance(value, str) and len(value) > 1800:
        return value[:1800] + "...[truncated]"
    return value


def _compact(value: Any, limit: int = _MAX_OBSERVATION_CHARS) -> str:
    compacted = _shrink(value)
    raw = json.dumps(compacted, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(raw) <= limit:
        return raw
    preview_limit = max(200, limit - 180)
    return json.dumps(
        {
            "truncated": True,
            "instruction": "Use visible evidence and request a narrower view if needed.",
            "preview": raw[:preview_limit],
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _normalize_tool_calls(raw_calls: Any, allowed: set[str], max_calls: int) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    if not isinstance(raw_calls, list):
        return accepted
    for index, call in enumerate(raw_calls):
        if len(accepted) >= max_calls:
            break
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name")
        if name not in allowed:
            continue
        try:
            _arguments(call)
        except (ValueError, json.JSONDecodeError, TypeError, KeyError):
            continue
        if not call.get("id"):
            call = {**call, "id": f"model-call-{index + 1}"}
        accepted.append(call)
    return accepted


def _call_signature(call: dict[str, Any]) -> str:
    function = call["function"]
    return json.dumps(
        {"name": function["name"], "args": _arguments(call)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _search_query(args: dict[str, Any]) -> str:
    for key in _SEARCH_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _observation_failed(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    node = value.get("observation") if isinstance(value.get("observation"), dict) else value
    if not isinstance(node, dict):
        return False
    if node.get("error"):
        return True
    if node.get("success") is False or node.get("ok") is False:
        return True
    status = str(node.get("status") or node.get("outcome") or "").lower()
    return status in {"error", "failed", "failure", "rejected", "unavailable"}


def _observation_empty(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) == 0
    if not isinstance(value, dict):
        return False
    node = value.get("observation") if isinstance(value.get("observation"), dict) else value
    if isinstance(node, list):
        return len(node) == 0
    if not isinstance(node, dict):
        return False
    for key in ("results", "items", "products", "candidates", "hits"):
        found = node.get(key)
        if isinstance(found, list) and len(found) == 0:
            return True
    return False


def _observation_done(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("done") is True:
        return True
    inner = value.get("observation")
    return isinstance(inner, dict) and inner.get("done") is True


def _collect_ids(value: Any, found: list[str], limit: int = 24) -> None:
    if len(found) >= limit:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _ID_KEYS or lowered.endswith("_id"):
                if isinstance(item, (str, int)) and str(item) and lowered != "session_id":
                    marker = f"{lowered}={item}"
                    if marker not in found:
                        found.append(marker)
                        if len(found) >= limit:
                            return
            else:
                _collect_ids(item, found, limit)
    elif isinstance(value, list):
        for item in value[:_MAX_LIST_ITEMS]:
            _collect_ids(item, found, limit)


def _message_has_explanation(call: dict[str, Any]) -> bool:
    args = _arguments(call)
    content = next(
        (
            value
            for key, value in args.items()
            if key.lower() in {"content", "message", "text", "response"}
            and isinstance(value, str)
        ),
        "",
    )
    return len(content.split()) >= 8


def _phase(
    search_obs: int,
    inspect_obs: int,
    seen_ids: list[str],
    explanation_sent: bool,
    remaining: int,
    roles: dict[str, str],
) -> str:
    has_search = "search" in roles.values()
    has_inspect = "inspect" in roles.values()
    has_message = "message" in roles.values()
    has_terminal = "terminal" in roles.values()
    if remaining <= 1:
        return "COMMIT" if has_terminal else "VERIFY"
    if remaining <= 2 and has_message and not explanation_sent:
        return "EXPLAIN"
    if has_search and search_obs == 0:
        return "SEEK"
    if has_inspect and inspect_obs == 0:
        return "VERIFY"
    if has_inspect and inspect_obs < 2 and len(seen_ids) >= 2 and remaining > 3:
        return "VERIFY"
    if len(seen_ids) >= 2 and remaining > 3:
        return "COMPARE"
    if has_message and not explanation_sent and remaining <= 4:
        return "EXPLAIN"
    if has_terminal and (inspect_obs > 0 or search_obs > 0) and remaining <= 3:
        return "COMMIT"
    if search_obs > 0 and inspect_obs == 0 and has_inspect:
        return "VERIFY"
    return "COMPARE" if seen_ids else "SEEK"


def _quality_gate(
    tool_calls: list[dict[str, Any]],
    roles: dict[str, str],
    failed_signatures: set[str],
    missed_queries: set[str],
    evidence_observations: int,
    explanation_sent: bool,
    remaining: int,
    phase: str,
) -> str | None:
    for call in tool_calls:
        if _call_signature(call) in failed_signatures:
            return (
                "That exact action already failed. Diagnose the failure and change its "
                "query, candidate, SKU, arguments, or action order."
            )
        role = roles.get(call["function"]["name"], "act")
        if role == "search":
            query = _search_query(_arguments(call))
            if query and query in missed_queries:
                return (
                    "That exact search already returned no usable candidates. Reformulate "
                    "using different value terms from the current goal."
                )

    proposed = {roles.get(call["function"]["name"], "act") for call in tool_calls}
    if remaining <= 1:
        return None
    if "terminal" in proposed and phase in {"SEEK", "VERIFY"}:
        return f"Too early to commit. Current phase is {phase}."
    if (
        "terminal" in proposed
        and evidence_observations == 0
        and any(role in {"search", "inspect"} for role in roles.values())
    ):
        return (
            "Do not commit yet: no successful search or inspection evidence has been "
            "recorded. Gather and verify evidence first."
        )
    if phase == "EXPLAIN" and "message" in roles.values() and "message" not in proposed:
        return (
            "Before the terminal action, use the published message tool to give a "
            "concise claim-evidence explanation based only on observations."
        )
    if (
        "terminal" in proposed
        and "message" in roles.values()
        and not explanation_sent
        and "message" not in proposed
        and remaining > 2
    ):
        return (
            "Before the terminal action, use the published message tool to give a "
            "concise claim-evidence explanation based only on observations."
        )
    return None


def _conversation_view(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(messages) <= _MAX_CONTEXT_MESSAGES + 2:
        return messages
    head = [messages[0], messages[1]]
    shoppers = [
        message
        for message in messages[2:]
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and str(message["content"]).startswith("Shopper update")
    ]
    recent = messages[-_MAX_CONTEXT_MESSAGES:]
    while recent and recent[0].get("role") == "tool":
        recent = recent[1:]
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for message in [*head, *shoppers, *recent]:
        marker = id(message)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(message)
    return merged


def _infer(
    messages: list[dict[str, Any]],
    policy: dict[str, Any],
    model: str,
    memory_note: str,
) -> dict[str, Any] | None:
    view = [*_conversation_view(messages), {"role": "user", "content": memory_note}]
    return _inference_proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": model,
            "messages": view,
            "tools": policy.get("tools") or [],
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 1536,
        },
    )


def _nudge(turn: int, max_steps: int, reason: str) -> dict[str, str]:
    remaining = max_steps - turn + 1
    return {
        "role": "user",
        "content": (
            f"{reason} The environment is not done. "
            f"{remaining} turn(s) remain including this one. "
            "Pick a published tool, use only IDs from observations, "
            "and take the next concrete action."
        ),
    }


def _memory_note(
    turn: int,
    max_steps: int,
    max_calls: int,
    phase: str,
    seen_ids: list[str],
    missed_queries: set[str],
    failed_calls: int,
    shopper_notes: list[str],
    search_obs: int,
    inspect_obs: int,
    explanation_sent: bool,
) -> str:
    remaining = max_steps - turn + 1
    missed = list(missed_queries)[-6:]
    shopper = shopper_notes[-3:]
    return (
        "Internal controller note (not a shopper request):\n"
        f"- turn {turn}/{max_steps}; remaining={remaining}; phase={phase}\n"
        f"- playbook: {_PHASE_PLAYBOOKS[phase]}\n"
        f"- seen identities: {seen_ids or ['none yet']}\n"
        f"- missed searches: {missed or ['none']}\n"
        f"- failed calls={failed_calls}; searches={search_obs}; "
        f"inspections={inspect_obs}; explanation_sent={explanation_sent}\n"
        f"- shopper amendments: {shopper or ['none']}\n"
        f"Issue at most {max_calls} published tool call(s). Prefer packing independent "
        "inspections when the phase is VERIFY and multiple identities are available."
    )


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy.get("query") or problem_data.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_steps = max(1, int(policy.get("max_steps") or 1))
    max_calls = max(1, int(policy.get("max_calls_per_turn") or 1))
    allowed = _tool_names(policy.get("tools") or [])
    roles = _tool_roles(policy.get("tools") or [])
    primary_model, fallback_model = _models()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(policy, query)},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []
    failed_signatures: set[str] = set()
    missed_queries: set[str] = set()
    seen_ids: list[str] = []
    shopper_notes: list[str] = []
    failed_calls = 0
    search_obs = 0
    inspect_obs = 0
    explanation_sent = False

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        phase = _phase(
            search_obs,
            inspect_obs,
            seen_ids,
            explanation_sent,
            remaining,
            roles,
        )
        memory = _memory_note(
            turn,
            max_steps,
            max_calls,
            phase,
            seen_ids,
            missed_queries,
            failed_calls,
            shopper_notes,
            search_obs,
            inspect_obs,
            explanation_sent,
        )
        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        last_reason = "Choose a valid published tool."
        for attempt in range(2):
            model = primary_model if attempt == 0 else fallback_model
            inference = _infer(messages, policy, model, memory)
            if inference is None:
                last_reason = "The inference request failed."
                messages.append(
                    _nudge(
                        turn,
                        max_steps,
                        f"{last_reason} Switch reasoning model and continue safely.",
                    )
                )
                continue
            assistant = (inference.get("choices") or [{}])[0].get("message") or {}
            assistant_content = assistant.get("content") or ""
            tool_calls = _normalize_tool_calls(
                assistant.get("tool_calls"), allowed, max_calls
            )
            if tool_calls:
                gate_reason = _quality_gate(
                    tool_calls,
                    roles,
                    failed_signatures,
                    missed_queries,
                    search_obs + inspect_obs,
                    explanation_sent,
                    remaining,
                    phase,
                )
                if gate_reason is None:
                    break
                tool_calls = []
                last_reason = gate_reason
            else:
                last_reason = (
                    "That reply had no usable tool call or used an unpublished tool name."
                )
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    _nudge(turn, max_steps, last_reason),
                ]
            )
        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {
                        "name": call["function"]["name"],
                        "args": _arguments(call),
                    },
                }
                for index, call in enumerate(tool_calls, start=1)
            ],
        }
        result = _environment_proxy.post("/environment/call", json_data=envelope)
        if result is None:
            failed_calls += 1
            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "environment_error": "environment call failed",
                }
            )
            messages.append(
                _nudge(turn, max_steps, "The environment call failed. Try a different action.")
            )
            continue

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        env_calls = result.get("calls") or []
        finished = False
        for tool_call, call_result in zip(tool_calls, env_calls):
            observation = call_result.get("observation", call_result)
            role = roles.get(tool_call["function"]["name"], "act")
            signature = _call_signature(tool_call)
            if _observation_failed(observation):
                failed_signatures.add(signature)
                failed_calls += 1
            elif role == "search" and _observation_empty(observation):
                query_text = _search_query(_arguments(tool_call))
                if query_text:
                    missed_queries.add(query_text)
                failed_signatures.add(signature)
            else:
                if role == "search":
                    search_obs += 1
                elif role == "inspect":
                    inspect_obs += 1
                if role == "message" and _message_has_explanation(tool_call):
                    explanation_sent = True
                _collect_ids(observation, seen_ids)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or f"{group_id}-tool",
                    "content": _compact(observation),
                }
            )
            if _observation_done(observation):
                finished = True
        if result.get("user_message"):
            shopper = result["user_message"]
            shopper_text = (
                shopper.get("content") if isinstance(shopper, dict) else str(shopper)
            )
            shopper_notes.append(str(shopper_text))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Shopper update: "
                        f"{shopper_text}\n"
                        "This amends the original goal. Later shopper instructions override "
                        "earlier ones. Original request: "
                        f"{query}"
                    ),
                }
            )
        if finished:
            break

    return dialogue


__all__ = ["agent_main"]
