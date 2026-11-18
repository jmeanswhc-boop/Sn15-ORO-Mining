"""Knowledge-first planner for validator-owned generated environments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_PROVIDER_DEFAULTS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_PROXY = ProxyClient(timeout=120, max_retries=3)
_MAX_INFERENCE_ATTEMPTS = 3
_LIST_DIGEST_LIMIT = 10
_STRING_DIGEST_LIMIT = 1500


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required: tuple[str, ...]
    properties: tuple[str, ...]


@dataclass
class ToolRegistry:
    tools: tuple[ToolSpec, ...]
    names: frozenset[str]

    @classmethod
    def from_policy(cls, raw_tools: list[dict[str, Any]]) -> ToolRegistry:
        specs: list[ToolSpec] = []
        for item in raw_tools:
            fn = item.get("function") or {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            required = tuple(str(k) for k in params.get("required") or ())
            specs.append(
                ToolSpec(
                    name=name,
                    description=str(fn.get("description") or "").strip(),
                    required=required,
                    properties=tuple(str(k) for k in props),
                )
            )
        return cls(
            tools=tuple(specs),
            names=frozenset(spec.name for spec in specs),
        )

    def playbook(self) -> str:
        if not self.tools:
            return "No tool metadata was supplied; infer usage from each tool schema."
        lines: list[str] = []
        for spec in self.tools:
            line = f"- {spec.name}"
            if spec.description:
                line += f": {spec.description}"
            lines.append(line)
            hints: list[str] = []
            if spec.required:
                hints.append(f"required={', '.join(spec.required)}")
            if spec.properties:
                hints.append(f"optional={', '.join(spec.properties)}")
            if hints:
                lines.append(f"  ({'; '.join(hints)})")
        return "\n".join(lines)

    def capability_hints(self) -> list[str]:
        names = self.names
        hints: list[str] = []
        if any("search" in name for name in names):
            hints.append(
                "Discovery tools are available: craft short keyword queries, "
                "then inspect promising candidates before committing."
            )
        if any("cart" in name for name in names):
            hints.append(
                "Cart tools are available: verify product identifiers and "
                "options from inspection results before adding items."
            )
        if any(name == "message" for name in names):
            hints.append(
                "A messaging tool is available: answer shopper follow-ups "
                "directly instead of ignoring user_message observations."
            )
        if any("order" in name or "checkout" in name or "place" in name for name in names):
            hints.append(
                "Completion tools are available: only finalize after "
                "constraints from the request and observations are satisfied."
            )
        return hints


@dataclass
class EpisodeState:
    tools_invoked: set[str] = field(default_factory=set)
    error_notes: list[str] = field(default_factory=list)
    shopper_replies: int = 0
    cart_touched: bool = False
    search_runs: int = 0
    terminal_signal: bool = False

    def register_observation(self, tool_name: str, observation: dict[str, Any]) -> None:
        self.tools_invoked.add(tool_name)
        if observation.get("done") is True:
            self.terminal_signal = True
        err = observation.get("error")
        if err:
            self.error_notes.append(str(err))
        payload = observation.get("observation")
        if isinstance(payload, dict):
            cart = payload.get("cart")
            if isinstance(cart, list) and cart:
                self.cart_touched = True
            if "search" in tool_name or "results" in payload or "products" in payload:
                self.search_runs += 1

    def status_line(self, turn: int, max_steps: int) -> str:
        parts = [f"Solver turn {turn} of {max_steps}."]
        if self.tools_invoked:
            parts.append(
                "Prior tools: " + ", ".join(sorted(self.tools_invoked)) + "."
            )
        if self.error_notes:
            parts.append(
                "Recent tool errors: "
                + "; ".join(self.error_notes[-2:])
                + ". Adjust arguments or switch tools."
            )
        if self.shopper_replies:
            parts.append(
                f"Shopper follow-ups received: {self.shopper_replies}. "
                "Respond when a messaging tool exists."
            )
        if self.cart_touched:
            parts.append("Cart currently holds items according to observations.")
        remaining = max_steps - turn
        if remaining <= 2:
            parts.append(
                "Step budget is nearly exhausted; favor decisive completion actions."
            )
        elif not self.search_runs and remaining > 2:
            parts.append(
                "No discovery activity yet; explore the catalog before committing."
            )
        return " ".join(parts)


def _resolve_model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _PROVIDER_DEFAULTS.get(
        provider, _PROVIDER_DEFAULTS["chutes"]
    )


def _digest(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "…"
    if isinstance(value, list):
        if len(value) > _LIST_DIGEST_LIMIT:
            trimmed = [_digest(item, depth + 1) for item in value[:_LIST_DIGEST_LIMIT]]
            return {
                "preview": trimmed,
                "omitted_count": len(value) - _LIST_DIGEST_LIMIT,
            }
        return [_digest(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: _digest(item, depth + 1) for key, item in value.items()}
    if isinstance(value, str) and len(value) > _STRING_DIGEST_LIMIT:
        return value[:_STRING_DIGEST_LIMIT] + "…"
    return value


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "{}").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("tool arguments must be a JSON object") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _compose_system_prompt(
    query: str,
    registry: ToolRegistry,
    max_steps: int,
) -> str:
    capability_lines = registry.capability_hints()
    capability_block = (
        "\n".join(f"- {line}" for line in capability_lines)
        if capability_lines
        else "- Follow the tool schemas and observation feedback."
    )
    return (
        "You are a meticulous shopping assistant operating inside a sealed "
        "simulated store. Satisfy the shopper using ONLY the supplied tools. "
        "Stop only after an environment observation reports done=true.\n\n"
        f"Shopper request:\n{query}\n\n"
        f"Solver turn budget: {max_steps}.\n\n"
        "Operating principles:\n"
        "- Derive constraints (product type, brand, price, quantity, shop, "
        "budget, ordering) from the request and from observation payloads.\n"
        "- Prefer concise discovery queries; inspect candidates before cart or "
        "checkout actions.\n"
        "- Treat tool errors as guidance: revise inputs or choose another tool.\n"
        "- When several read-only tools are permitted in one turn, batch them.\n"
        "- Never invent product identifiers; take them from search or inspection "
        "observations.\n\n"
        "Capability notes:\n"
        f"{capability_block}\n\n"
        "Tool reference:\n"
        f"{registry.playbook()}"
    )


def _continuation_prompt(
    state: EpisodeState,
    registry: ToolRegistry,
    *,
    turn: int,
    max_steps: int,
    attempt: int,
) -> str:
    if attempt == 1:
        return (
            "The episode is not finished (done=true has not appeared). "
            "Select one or more supplied tools and continue."
        )
    if attempt == 2:
        return (
            f"{state.status_line(turn, max_steps)} "
            "You must respond with tool calls only."
        )
    suggestions: list[str] = []
    if "message" in registry.names and state.shopper_replies:
        suggestions.append("reply to the shopper with the message tool")
    if state.error_notes:
        suggestions.append("fix the arguments that produced the last tool error")
    if not state.search_runs and any("search" in name for name in registry.names):
        suggestions.append("run a discovery/search tool with focused keywords")
    if state.cart_touched and any(
        "order" in name or "checkout" in name or "place" in name
        for name in registry.names
    ):
        suggestions.append("advance toward checkout if constraints are met")
    if not suggestions:
        suggestions.append("inspect current state, then take the next concrete action")
    return (
        "Tool call required. Suggested focus: "
        + "; ".join(suggestions)
        + ". Use only tools from the supplied schema."
    )


def _chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "model": _resolve_model(),
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0,
    }
    response = _PROXY.post("/inference/chat/completions", json_data=payload)
    if response is None:
        raise RuntimeError("inference request failed")
    choice = response["choices"][0]["message"]
    return {
        "content": choice.get("content") or "",
        "tool_calls": list(choice.get("tool_calls") or []),
    }


def _plan_actions(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    registry: ToolRegistry,
    state: EpisodeState,
    *,
    turn: int,
    max_steps: int,
    max_calls: int,
) -> tuple[str, list[dict[str, Any]]]:
    working_messages = list(messages)
    last_content = ""
    last_calls: list[dict[str, Any]] = []
    for attempt in range(1, _MAX_INFERENCE_ATTEMPTS + 1):
        turn = _chat(working_messages, tools)
        last_content = turn["content"]
        calls = turn["tool_calls"][:max_calls]
        invalid = [
            call["function"]["name"]
            for call in calls
            if call["function"]["name"] not in registry.names
        ]
        if calls and not invalid:
            return last_content, calls
        working_messages.append({"role": "assistant", "content": last_content})
        if invalid:
            hint = (
                "These tool names are not in the schema: "
                + ", ".join(sorted(set(invalid)))
                + ". Choose valid tools only."
            )
        else:
            hint = _continuation_prompt(
                state,
                registry,
                turn=turn,
                max_steps=max_steps,
                attempt=attempt,
            )
        working_messages.append({"role": "user", "content": hint})
        last_calls = calls
    if last_calls:
        valid = [
            call
            for call in last_calls
            if call["function"]["name"] in registry.names
        ][:max_calls]
        if valid:
            return last_content, valid
    raise RuntimeError("model returned no valid tool call before completion")


def _environment_envelope(
    binding: dict[str, Any],
    *,
    problem_id: str,
    turn: int,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    group_id = f"{problem_id}-turn-{turn}"
    return {
        **binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": [
            {
                "call_id": f"{group_id}-{index}",
                "action": {
                    "name": call["function"]["name"],
                    "args": _parse_arguments(call["function"].get("arguments", "{}")),
                },
            }
            for index, call in enumerate(tool_calls, start=1)
        ],
    }


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Plan and execute tool actions until the environment reports completion."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    query = str(policy["query"])
    max_steps = int(policy["max_steps"])
    max_calls = int(policy.get("max_calls_per_turn", 1))
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))

    registry = ToolRegistry.from_policy(list(policy.get("tools") or []))
    state = EpisodeState()
    dialogue: list[dict[str, Any]] = []

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _compose_system_prompt(query, registry, max_steps)},
        {"role": "user", "content": query},
    ]

    for turn in range(1, max_steps + 1):
        briefing = state.status_line(turn, max_steps)
        planner_messages = messages + [{"role": "system", "content": briefing}]

        assistant_text, tool_calls = _plan_actions(
            planner_messages,
            list(policy.get("tools") or []),
            registry,
            state,
            turn=turn,
            max_steps=max_steps,
            max_calls=max_calls,
        )

        messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls,
            }
        )

        envelope = _environment_envelope(
            binding,
            problem_id=problem_id,
            turn=turn,
            tool_calls=tool_calls,
        )
        result = _PROXY.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")

        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            observation = call_result["observation"]
            state.register_observation(tool_call["function"]["name"], observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(_digest(observation)),
                }
            )

        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            state.shopper_replies += 1
            messages.append({"role": "user", "content": user_message["content"]})

        if any(call["observation"].get("done") for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
