"""Tulip agent for dynamically generated commerce environments."""

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
_RECOVERY_MODELS = {
    "chutes": "openai/gpt-oss-120b-TEE",
    "openrouter": "openai/gpt-oss-120b",
}
_FALLBACK_MODELS = {
    "chutes": "moonshotai/Kimi-K2.6-TEE",
    "openrouter": "moonshotai/kimi-k2.6",
}
_proxy = ProxyClient(timeout=150, max_retries=2)


def _model(query: str) -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes") or "chutes"
    models = (
        _RECOVERY_MODELS
        if _needs_recovery_planning(query) and not _needs_justification(query)
        else _DEFAULT_MODELS
    )
    return models.get(provider, models["chutes"])


def _fallback_model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes") or "chutes"
    return _FALLBACK_MODELS.get(provider, _FALLBACK_MODELS["chutes"])


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _needs_justification(query: str) -> bool:
    shopper_request = query.split("\n\n", 1)[0]
    return bool(
        re.search(
            r"\b(?:why|reason|explain|justify|justification|evidence)\b",
            shopper_request,
            re.IGNORECASE,
        )
    )


def _needs_recovery_planning(query: str) -> bool:
    """Detect an explicit shopper-authored contingency with an ordered fallback."""
    shopper_request = query.split("\n\n", 1)[0]
    has_contingency = bool(
        re.search(
            r"\b(?:if|when)\b[^.?!]*\b(?:unavailable|disappears?|out\s+of\s+stock|changes?)\b",
            shopper_request,
            re.IGNORECASE,
        )
    )
    has_ordered_fallback = bool(
        re.search(r"\b(?:before|first|then)\b", shopper_request, re.IGNORECASE)
    )
    return has_contingency and has_ordered_fallback


def _safe_order_args(
    args: dict[str, Any],
    query: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep optional claims within the order schema supplied for this task."""
    cleaned = {key: value for key, value in args.items() if key != "justification"}
    if not _needs_justification(query):
        return cleaned

    field_schema: dict[str, Any] = {}
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") != "place_test_order":
            continue
        parameters = function.get("parameters", {})
        field_schema = (
            parameters.get("properties", {})
            .get("justification", {})
            .get("properties", {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        break

    allowed = field_schema.get("enum")
    allowed_fields = set(allowed) if isinstance(allowed, list) else None
    supplied = args.get("justification", {})
    raw_claims = supplied.get("claims", []) if isinstance(supplied, dict) else []
    claims: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if isinstance(raw_claims, list):
        for claim in raw_claims:
            if not isinstance(claim, dict):
                continue
            field = claim.get("field")
            value = claim.get("value")
            if not isinstance(field, str) or not isinstance(value, str):
                continue
            field, value = field.strip(), value.strip()
            if not field or not value:
                continue
            if allowed_fields is not None and field not in allowed_fields:
                continue
            key = (field, value)
            if key not in seen:
                seen.add(key)
                claims.append({"field": field, "value": value})
    cleaned["justification"] = {"claims": claims}
    return cleaned


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Solve the disclosed policy with evidence-backed, state-aware tool use."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))

    instructions = """
You are a precise commerce execution agent. The complete task policy follows in
the user message and is authoritative. Finish the task, not merely a plausible
shopping request.

Before acting, silently separate firm constraints, soft preferences, requested
comparisons or explanations, and any required interaction sequence. Preserve
that state throughout the session. Treat every later shopper message as an
authoritative update: answer direct questions with the message tool, replace
only requirements the shopper explicitly changes, and then re-check the choice.

Use structured filters when available, but search alternate formulations when
recall matters. Search snippets are discovery evidence only. Use the supplied
inspection and comparison tools to verify structured facts, and reject
accessories or misleading titles. When ranking or preference matters, inspect
enough distinct valid candidates before choosing. Firm constraints always
outrank price and soft preferences; among equally valid candidates obey the
stated preference and ranking rule.

Keep exploration bounded. Start with one structured filter and at most a few
short, materially different searches; do not repeat a query or re-view unchanged
evidence. Maintain a shortlist of candidates proven valid by observations. A
final product must match the requested catalog category and identity constraints.
If no candidate is yet proven, keep searching rather than committing a
title-only guess.

Before any cart action, verify the exact product and SKU. Before ordering,
inspect current stock and price. After every cart action, read the returned
shopper message before taking another action. If requirements or availability
change, remove an invalid cart item, find a compliant replacement, re-verify it,
and only then order. Never assume that a later correction repairs an incorrect
initial commitment.

Treat direct shopper questions as required protocol steps: answer them with the
message tool before any subsequent cart or order action. After a market event,
re-check category, identity, requested specification, updated budget, stock, and
all unchanged constraints from scratch. Never recover by relaxing a firm
constraint or switching to a conflicting category.

If neither the shopper nor policy asks for an explanation, omit optional
justification. When it is required, follow the current order tool schema exactly
and provide the smallest set of structured claims requested by the public task.
Choose claim names from that schema and copy claim values from inspection or
comparison observations for the final item and its latest state. Do not infer
facts from titles or add unsupported prose. If the schema asks for the
shopper's specification field, use the semantic attribute named by the public
request or policy, normalized as a concise snake_case field label; do not
substitute an internal catalog storage key merely because it holds the evidence.

Use only supplied tools. Continue until an observation reports done=true. Keep
actions efficient, but never skip a policy-required comparison, interaction,
verification, recovery, or justification step.
""".strip()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": policy["query"]},
    ]
    dialogue: list[dict[str, Any]] = []
    max_steps = int(policy["max_steps"])
    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        if remaining == 12:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Execution budget notice: 12 tool steps remain. Stop broad exploration. "
                        "Use the evidence already gathered to select the best fully compliant "
                        "candidate, trigger/read any required cart event, verify current state, "
                        "and complete the order. If none is proven, make one focused search "
                        "for the missing firm constraints; never relax them."
                    ),
                }
            )
        elif remaining == 5:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Final execution budget: 5 tool steps remain. Complete the required "
                        "recovery or clarification now, then inspect the final SKU if needed "
                        "and place the order only if every firm constraint is proven; do not "
                        "perform more broad searches or substitute an invalid category."
                    ),
                }
            )
        for _attempt in range(3):
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": (
                        _model(policy["query"])
                        if _attempt == 0
                        else _fallback_model()
                    ),
                    "messages": messages,
                    "tools": policy["tools"],
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                continue
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "The environment is not finished. Re-read the task policy and "
                            "choose the next supplied tool needed to satisfy it."
                        ),
                    },
                ]
            )
        else:
            raise RuntimeError("inference failed to produce a tool call")

        group_id = f"{problem_id}-turn-{turn}"
        actions = []
        for index, call in enumerate(tool_calls, start=1):
            name = call["function"]["name"]
            args = _arguments(call)
            if name == "place_test_order":
                args = _safe_order_args(args, policy["query"], policy["tools"])
                call["function"]["arguments"] = json.dumps(args)
            actions.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        messages.append(
            {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
        )
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": actions,
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
        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(call_result["observation"]),
                }
            )
        shopper_message = result.get("user_message")
        if shopper_message:
            messages.append({"role": "user", "content": shopper_message["content"]})
            if "?" in shopper_message["content"]:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Protocol checkpoint: the latest shopper message asks a direct "
                            "question. Your next action must answer it with the message tool "
                            "before changing the cart or ordering."
                        ),
                    }
                )
        if any(call["observation"]["done"] for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
