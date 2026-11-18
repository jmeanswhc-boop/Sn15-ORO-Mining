"""Shopping agent for ORO generated environments (EnvPack / tools v5).

IMPORTANT: oro-new no longer uses the legacy ShoppingBench tool surface
(`find_product` / `recommend_product` / `terminate`). Agents must:
  1. Read `problem_data["environment"]["policy_view"]` for query + tools
  2. Call LLM with those dynamic tools
  3. Execute actions via POST `/environment/call` until observation.done=true

This file is the submit-ready agent. Keep it self-contained.
"""

from __future__ import annotations

import json
import logging
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_SYSTEM_PROMPT = """You are an expert shopping agent in a live product environment.

Goal: find a valid in-stock product matching the shopper's firm requirements,
add the correct variant to the cart, and place a test order. End when
observation.done=true.

Hard rules:
- Prefer firm requirements over soft preferences ("must"/"at or under" beat "ideally").
- Stay within budget. Prefer in_stock=true.
- NEVER invent product_ids, skus, prices, or attributes.
- Avoid the `message` tool unless the observation explicitly requires a shopper reply.
  Messaging triggers the shopper simulator and often wastes the turn.
- Be decisive. Do not loop on search. Typical successful path:
  1) search and/or filter (in_stock=true, max_price if budget known)
  2) view 1-3 candidates to get sku + raw facts
  3) inspect_stock on the chosen product_id+sku
  4) add_to_cart
  5) place_test_order (include justification.claims when the query asks for a reason)
- You may emit multiple tool calls in one turn (up to max_calls_per_turn).
  Batch independent reads (e.g. several view calls) when useful.
- For place_test_order justification.claims, use ONLY these fields when needed:
  brand, category, in_stock, price, refresh_rate, resolution, screen_size.
  Values must match facts from view/compare/inspect_stock for the FINAL product.
- If search/filter results are weak, reformulate once or twice, then commit to the
  best available match that satisfies firm constraints.
"""


def _inference_api_key() -> str | None:
    """Sandbox images historically read CHUTES_ACCESS_TOKEN; validators inject
    INFERENCE_ACCESS_TOKEN. Pass the key explicitly so either naming works."""
    return (
        getenv("INFERENCE_ACCESS_TOKEN")
        or getenv("CHUTES_ACCESS_TOKEN")
        or getenv("OPENROUTER_API_KEY")
        or getenv("CHUTES_API_KEY")
    )


# Stable sandbox ProxyClient still defaults to CHUTES_ACCESS_TOKEN; validators
# inject INFERENCE_ACCESS_TOKEN. Missing auth makes nginx return 429.
_proxy = ProxyClient(
    timeout=180,
    max_retries=5,
    rate_limit_retry_delay=6,
    api_key=_inference_api_key(),
)


def _model() -> str:
    provider = (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
    override = (getenv("SANDBOX_MODEL") or "").strip()
    if override:
        return override
    return _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["chutes"])


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _inference_with_backoff(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    attempts: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        inference = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": _model(),
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if inference is None:
            last_error = RuntimeError("inference request failed")
            time.sleep(min(2**attempt, 12))
            continue
        try:
            return inference["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            last_error = RuntimeError(f"malformed inference response: {exc}")
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(str(last_error) if last_error else "inference failed")


def _drop_message_tools(
    tool_calls: list[dict[str, Any]],
    *,
    allow_message: bool,
) -> list[dict[str, Any]]:
    if allow_message:
        return tool_calls
    filtered = [
        call
        for call in tool_calls
        if call.get("function", {}).get("name") != "message"
    ]
    return filtered or tool_calls


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Drive the environment tools until done=true or max_steps."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy["tools"]
    query = policy["query"]

    logger.info(
        "[agent] problem=%s max_steps=%s max_calls=%s query=%s",
        problem_id,
        max_steps,
        max_calls,
        query[:120],
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []
    ordered = False

    for turn in range(1, max_steps + 1):
        remaining = max_steps - turn + 1
        # Late turns: force ordering rather than more search.
        urgency = ""
        if remaining <= 8 and not ordered:
            urgency = (
                " TIME CRITICAL: few turns remain. Stop searching. "
                "View/inspect one viable candidate, add_to_cart, then place_test_order now."
            )
            messages.append({"role": "user", "content": urgency.strip()})

        assistant: dict[str, Any] | None = None
        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""

        for nudge in range(2):
            assistant = _inference_with_backoff(messages=messages, tools=tools)
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            # Avoid message unless the model has nothing else; messaging trips
            # the shopper simulator which is fragile on some providers.
            tool_calls = _drop_message_tools(tool_calls, allow_message=False)
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "Choose a tool now. Prefer "
                            "search/filter → view → inspect_stock → "
                            "add_to_cart → place_test_order. Do not message."
                        ),
                    },
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )

        action_names = [c["function"]["name"] for c in tool_calls]
        if "place_test_order" in action_names:
            ordered = True

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

        # Do NOT retry the same idempotency_key — session registry treats reuse
        # after a harness error as a hard failure and burns the turn budget.
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError(
                f"environment call failed at turn={turn} actions={action_names}"
            )

        logger.info("[agent] turn=%s actions=%s", turn, action_names)

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result["observation"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(observation),
                }
            )
            if observation.get("error"):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The last tool call returned an error. "
                            "Adjust and continue toward placing a valid test order. "
                            "Do not use the message tool."
                        ),
                    }
                )
        if result.get("user_message"):
            messages.append(
                {"role": "user", "content": result["user_message"]["content"]}
            )
        if any(call["observation"].get("done") for call in result["calls"]):
            logger.info("[agent] done at turn=%s", turn)
            break

    return dialogue


__all__ = ["agent_main"]
