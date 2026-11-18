"""ORO Bench agent 101 — generic generated-environment shopping agent.

No problem-specific lookup tables. Model follows the public shopper goal and
dynamic tool schemas. Sanitizes place_test_order claims using only the live
tool enum so optional/false claims cannot zero an otherwise-correct order.
"""

from __future__ import annotations

import json
import logging
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent101")

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_FALLBACK_MODELS = {
    "chutes": "Qwen/Qwen3-32B-TEE",
    "openrouter": "qwen/qwen3-32b",
}

_proxy = ProxyClient(timeout=180, max_retries=3)
_OBS_CHARS = 7000
_MAX_PARALLEL = 6

_SYSTEM = """You are a careful shopping agent for a live product catalog.

Use ONLY the tools provided in this episode. Never invent product_id or sku —
copy them from tool observations.

The user message is the shopper's goal. Interpret requirements, budget, stock,
candidate lists, ranking priorities, recovery rules, and any request for a
reason from that goal and from observations — no external lookup tables.

Speed matters: you have a hard step budget. Prefer batched tool calls. Do not
wander. After you have inspected a valid in-stock variant within budget, cart
it, handle any shopper/event message, re-inspect, and place_test_order.

Workflow:
1) search/filter (reformulate if needed; use k=10)
2) compare or multi-view the shortlist
3) inspect_stock on the chosen product_id + sku
4) add_to_cart only for a real intended purchase (first cart is graded)
5) on price/stock change messages: recover, inspect again, then order
6) place_test_order promptly once verified

Family habits from the goal text itself:
- Candidate variant list → only those compete; follow the stated priority order;
  compare them early; first cart must be the current best; after an event, re-rank.
- Preferred model/color → find and cart the exact match first; if it vanishes,
  same model other color before changing models; search the model code / title.
- "why" / "reason" / "walk me through" → place_test_order MUST include
  justification.claims. Otherwise OMIT justification entirely (optional false
  claims fail the episode).
- When claims are required: use ONLY schema fields (brand, category, in_stock,
  price, refresh_rate, resolution, screen_size). Values must match viewed facts
  / inspect_stock exactly. Include the gated spec from the goal, price, in_stock,
  and at most one soft brand/category claim. Never invent fields like
  category_path or unsupported specs.

Firm constraints beat soft preferences. Stop when done=true.
"""


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _models() -> list[str]:
    override = getenv("AGENT_MODEL")
    if override:
        return [override]
    primary = _DEFAULT_MODELS.get(_provider(), _DEFAULT_MODELS["chutes"])
    fallback = _FALLBACK_MODELS.get(_provider(), _FALLBACK_MODELS["chutes"])
    return [primary] if primary == fallback else [primary, fallback]


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim_obs(obs: Any) -> str:
    text = json.dumps(obs, ensure_ascii=False, default=str)
    if len(text) <= _OBS_CHARS:
        return text
    return text[:_OBS_CHARS] + f"...[truncated {len(text) - _OBS_CHARS} chars]"


def _needs_justification(query: str) -> bool:
    return bool(
        re.search(
            r"\b(why|reason|justify|justification|explain|walk me through|tell me why)\b",
            query or "",
            re.I,
        )
    )


def _claim_field_enum(tools: list[dict[str, Any]]) -> set[str] | None:
    """Allowed justification.field values from the live tool schema (not a hardcode table)."""
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != "place_test_order":
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return {str(x) for x in enum}
    return None


def _sanitize_action(
    name: str,
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: set[str] | None,
) -> dict[str, Any]:
    """Drop optional/false claims that zero family_gate; keep schema-legal claims only."""
    out = dict(args)
    if name != "place_test_order":
        return out
    if not _needs_justification(query):
        out.pop("justification", None)
        return out
    just = out.get("justification")
    if not isinstance(just, dict):
        return out
    claims = just.get("claims")
    if not isinstance(claims, list):
        out.pop("justification", None)
        return out
    cleaned = []
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "").strip()
        value = claim.get("value")
        if not field or value is None:
            continue
        if allowed_fields is not None and field not in allowed_fields:
            continue
        if field in seen:
            continue
        seen.add(field)
        cleaned.append({"field": field, "value": str(value)})
    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


def _chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return _proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
        },
    )


def _infer(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_calls: int,
) -> tuple[str, list[dict[str, Any]], str]:
    last_content = ""
    for model in _models():
        attempt_messages = messages
        for _ in range(2):
            inference = _chat(model, attempt_messages, tools)
            if not inference or not isinstance(inference, dict):
                logger.warning("inference failed model=%s", model)
                break
            try:
                assistant = inference["choices"][0]["message"]
            except (KeyError, IndexError, TypeError, AttributeError):
                logger.warning("malformed inference model=%s", model)
                break
            content = assistant.get("content") or ""
            last_content = content
            tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                return content, tool_calls, model
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "The environment is not done. Call one of the supplied "
                        "tools now."
                    ),
                },
            ]
    return last_content, [], _models()[0]


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    query = str(policy.get("query") or problem_data.get("query") or "")
    tools = policy["tools"]
    max_steps = int(policy.get("max_steps") or 30)
    max_calls = min(int(policy.get("max_calls_per_turn") or 1), _MAX_PARALLEL)
    allowed_claim_fields = _claim_field_enum(tools)
    want_reason = _needs_justification(query)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"{query}\n\n"
                f"Budget: {max_steps} turns, up to {max_calls} tool calls/turn. "
                "Batch reads. Order once a valid inspected variant is ready. "
                + (
                    "This goal asks for a reason — include truthful schema-only "
                    "justification.claims on place_test_order."
                    if want_reason
                    else "This goal does not ask for a reason — omit justification "
                    "on place_test_order."
                )
            ),
        },
    ]
    dialogue: list[dict[str, Any]] = []
    saw_cart = False
    saw_event = False

    for turn in range(1, max_steps + 1):
        # Urgency nudges so we do not burn the step budget without ordering.
        if turn == max(8, max_steps // 3) and not saw_cart:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Time check: shortlist with compare/view, inspect_stock a "
                        "valid variant, and add_to_cart soon. Do not keep searching."
                    ),
                }
            )
        if turn == max(14, (2 * max_steps) // 3):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Time check: if a valid inspected item is in cart (or ready), "
                        "re-inspect after any event and place_test_order now."
                    ),
                }
            )
        if turn >= max_steps - 4 and turn < max_steps:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{max_steps - turn} turns left. place_test_order on the best "
                        "verified in-stock in-budget variant immediately"
                        + (
                            " with schema-only justification.claims."
                            if want_reason
                            else " with NO justification object."
                        )
                    ),
                }
            )

        content, tool_calls, model_used = _infer(messages, tools, max_calls)
        if not tool_calls:
            dialogue.append(
                {
                    "role": "assistant",
                    "content": content,
                    "error": "no_tool_calls",
                    "model": model_used,
                }
            )
            raise RuntimeError("model returned no tool call before completion")

        actions = []
        for call in tool_calls:
            name = call["function"]["name"]
            args = _sanitize_action(
                name,
                _arguments(call),
                query=query,
                allowed_fields=allowed_claim_fields,
            )
            actions.append((call, name, args))

        messages.append(
            {
                "role": "assistant",
                "content": content,
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
                    "action": {"name": name, "args": args},
                }
                for index, (_call, name, args) in enumerate(actions, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        dialogue.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "model": model_used,
            }
        )

        call_results = result.get("calls") or []
        for (tool_call, name, _args), call_result in zip(actions, call_results):
            obs = (call_result or {}).get("observation")
            if name == "add_to_cart":
                saw_cart = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _trim_obs(obs),
                }
            )

        user_message = result.get("user_message") or {}
        user_text = user_message.get("content") if isinstance(user_message, dict) else None
        if user_text:
            saw_event = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{user_text}\n\n"
                        "Re-check live price/stock with inspect_stock. If the carted "
                        "item failed, recover to the next-best valid option (for a "
                        "named model/color: same model other color before other "
                        "models), then place_test_order."
                    ),
                }
            )

        if any(
            isinstance(c, dict)
            and isinstance(c.get("observation"), dict)
            and c["observation"].get("done")
            for c in call_results
        ):
            break

        if saw_cart and saw_event and turn < max_steps - 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Event already observed. Finish recovery with inspect_stock "
                        "and place_test_order; do not reopen broad search unless the "
                        "replacement is unknown."
                    ),
                }
            )
            saw_event = False  # one reminder

    return dialogue


__all__ = ["agent_main"]
