"""ORO Bench miner agent (Phases 1–3).

Same interaction workflow as src/agent/environment_agent.py:
  policy_view → inference (dynamic tools) → /environment/call → observe → done.

Phase 3 adds strategy: tool-aware prompting, turn budget, observation coaching.
"""

from __future__ import annotations

import json
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}


def _inference_api_key() -> str | None:
    # Local runner injects INFERENCE_ACCESS_TOKEN; some images use CHUTES_ACCESS_TOKEN.
    return getenv("INFERENCE_ACCESS_TOKEN") or getenv("CHUTES_ACCESS_TOKEN")


_proxy = ProxyClient(
    timeout=120,
    max_retries=6,
    rate_limit_retry_delay=15,
    api_key=_inference_api_key(),
)


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Parse model tool arguments; never crash the episode on bad JSON."""
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_payload(call_result: dict[str, Any]) -> str:
    """Serialize observation (+ error if present) for the model tool message."""
    payload: dict[str, Any] = {"observation": call_result.get("observation")}
    if call_result.get("error") is not None:
        payload["error"] = call_result["error"]
    return json.dumps(payload, default=str)


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        name = (tool.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _system_prompt(query: str, tools: list[dict[str, Any]], max_steps: int) -> str:
    """Build a strategy prompt from public policy fields only."""
    names = _tool_names(tools)
    name_set = set(names)
    name_list = ", ".join(names) if names else "(none listed)"

    lines = [
        "You are a shopping agent inside a generated environment.",
        "Use ONLY the supplied tools. Never invent tool names.",
        f"Available tools: {name_list}.",
        f"Episode budget: at most {max_steps} turns. Prefer decisive, useful actions.",
        "Goal: satisfy the shopper until an observation reports done=true.",
        "",
        "Working style:",
        "- Read the shopper goal carefully; extract constraints (price, category, stock, attributes).",
        "- Gather evidence with search/filter/view-style tools before committing.",
        "- Verify critical facts (price, stock, attributes) before terminal actions.",
        "- If an observation has an error or empty results, change strategy; do not repeat the same failing call.",
        "- If the environment sends a shopper message, treat it as high priority and adapt.",
        "- Prefer concrete tool actions over chat-only replies when a tool can progress the task.",
        "- When you have enough evidence, take the terminal/finish action promptly; do not burn remaining turns.",
    ]

    # Light heuristics from the public tool set (allowed: branch on problem_data).
    if "place_test_order" in name_set or "place_order" in name_set:
        lines.append(
            "- This task appears order-oriented: locate a valid in-stock item, confirm constraints, then place the order."
        )
    if any(n in name_set for n in ("rank", "reorder", "sort")):
        lines.append(
            "- Ranking tools are available: produce an ordering that matches the stated objective."
        )
    if any(n in name_set for n in ("justify", "explain", "message")):
        lines.append(
            "- When justification or messaging tools exist, support the final outcome with evidence you already observed."
        )
    if any("recover" in n or "retry" in n or "fix" in n for n in name_set):
        lines.append(
            "- Recovery-oriented tools are available: if something fails or changes, re-check and correct course."
        )

    # Query length only — no suite-specific phrases.
    if len(query) > 280:
        lines.append(
            "- The goal is detailed; track every explicit constraint before finishing."
        )

    return "\n".join(lines)


def _turn_nudge(turn: int, max_steps: int) -> str | None:
    """Budget reminder derived only from public step limits."""
    remaining = max_steps - turn + 1
    if remaining <= 3:
        return (
            f"Budget warning: turn {turn}/{max_steps} ({remaining} left including this one). "
            "Prioritize a finishing action that can set done=true if constraints are already met; "
            "otherwise take the single highest-value verification step."
        )
    if turn == 1:
        return None
    if turn % 5 == 0:
        return (
            f"Progress check: turn {turn}/{max_steps}. "
            "Avoid redundant inspect loops; move toward completing the shopper goal."
        )
    return None


def _observation_coach(env_calls: list[dict[str, Any]], turn: int, max_steps: int) -> str | None:
    """Short coach text from public observations only."""
    notes: list[str] = []
    for call in env_calls:
        if call.get("error") is not None:
            notes.append("Last action reported an error — adjust arguments or choose a different tool.")
            break
        obs = call.get("observation") or {}
        if obs.get("error"):
            notes.append("Observation contains an error — do not ignore it; recover or change approach.")
            break
        # Empty-ish result shapes commonly returned by search/filter tools.
        for key in ("results", "items", "candidates", "products"):
            value = obs.get(key)
            if isinstance(value, list) and len(value) == 0:
                notes.append(
                    f"Previous tool returned an empty '{key}' list — broaden, rephrase, or relax filters."
                )
                break

    remaining = max_steps - turn
    if remaining <= 2 and remaining >= 0:
        notes.append(
            f"Only {remaining} turn(s) left after this. Drive toward a terminal tool that can set done=true."
        )

    if not notes:
        return None
    return " ".join(notes)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one ORO Bench episode using the environment's dynamic tool list."""

    # --- Step 1: load public task contract ---
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    tools = policy.get("tools") or []
    query = policy["query"]

    # --- Step 2: start chat (system + shopper goal) ---
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(query, tools, max_steps)},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, max_steps + 1):
        nudge = _turn_nudge(turn, max_steps)
        if nudge:
            messages.append({"role": "user", "content": nudge})

        # --- Step 3: ask allowlisted model for a tool call ---
        for _attempt in range(2):
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
                raise RuntimeError("inference request failed")
            choices = inference.get("choices") or []
            if not choices:
                raise RuntimeError("inference returned no choices")
            assistant = choices[0].get("message") or {}
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "The environment has not reported done=true. "
                            "You must call one of the supplied tools now — "
                            "pick the action that most directly advances the shopper goal."
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

        # --- Step 4: execute tools via /environment/call ---
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

        # --- Step 5: feed observations (+ optional shopper message); stop on done ---
        env_calls = result.get("calls") or []
        for tool_call, call_result in zip(tool_calls, env_calls):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _tool_payload(call_result),
                }
            )
        if result.get("user_message"):
            messages.append(
                {"role": "user", "content": result["user_message"]["content"]}
            )

        if any((call.get("observation") or {}).get("done") for call in env_calls):
            break

        coach = _observation_coach(env_calls, turn, max_steps)
        if coach:
            messages.append({"role": "user", "content": coach})

    return dialogue


__all__ = ["agent_main"]
