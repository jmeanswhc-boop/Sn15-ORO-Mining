"""ORO Bench agent for 117_ACE — generated-environment contract (tool contract v5).

ShoppingBench's fixed tools (find_product / view_product_information / ...) do not
exist in ORO Bench: every task now supplies its own dynamic tool schemas in
``problem_data["environment"]["policy_view"]["tools"]``, and actions are executed by
POSTing to ``/environment/call`` instead of calling a Python tool function directly.
There is therefore no product-search / decomposition-pipeline logic left to port from
117_ACE.py — that logic (Prompt.SV_DECOMPOSE, PV_SELECT's Fibonacci-batch candidate
scan, PSV_BUILD's attribute normalization, ...) lived entirely inside the old fixed
tools and multi-stage LLM pipeline, which the environment now owns outright.

What *is* ported from 117_ACE.py:
  - the multi-model fallback chain (``ProxyUtil.LLM_PRIORITIES``),
  - the same proxy timeout/retry tuning (``LLM_PROXY`` / ``ORO_PROXY``),
  - ACE's ``TEMP_UP`` trick: nudge temperature up on a retry after a malformed/empty
    model response instead of hammering the same deterministic call repeatedly,
  - never letting a single bad turn or unhandled exception forfeit the whole episode.

On top of that baseline this version folds in two rounds of local-test findings from
145_Valid_envpack.py / 140_BILL_envpack.py (both scored well on constraint_satisfaction
but weak on ranking/justification, and recovery varied a lot by model chain): the
system prompt below explicitly distills 117_ACE's PV_SELECT literal-vs-semantic
matching discipline and "structured attributes over title guesses" evidence rule
into generic, family-agnostic guidance — no query phrases, synonym maps, or other
suite-specific content, per the Code Requirements.
"""

from __future__ import annotations

import json
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_LLM_PROXY = ProxyClient(timeout=180, max_retries=5, retry_delay=1.0)
_ENV_PROXY = ProxyClient(timeout=30, max_retries=5, retry_delay=1.0)

MAX_MODEL_PASSES = 2      # full sweeps over the priority chain before giving up a turn
MAX_TOOLCALL_RETRIES = 3  # re-prompts within one turn if no model returns a tool call
TEMP_UP = 0.3             # ACE's trick: nudge temperature up after an empty/bad reply

# Same relative ordering as 117_ACE.py's ProxyUtil.LLM_PRIORITIES, kept in sync with
# the live Backend allowlist (GLM-5-TEE / Kimi-K2.5 / MiniMax-M2.5 removed).
_CHUTES_PRIORITIES = [
    "moonshotai/Kimi-K2.6-TEE",
    "zai-org/GLM-5.1-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
    "deepseek-ai/DeepSeek-R1-0528-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "Qwen/Qwen3-32B-TEE",
    "XiaomiMiMo/MiMo-V2-Flash-TEE",
    "google/gemma-4-31B-turbo-TEE",
]
_OPENROUTER_PRIORITIES = [
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-32b",
    "xiaomi/mimo-v2-flash",
    "google/gemma-4-31b-it",
]

_SYSTEM_PROMPT = (
    "You are a careful shopping agent operating inside a generated environment. "
    "Read the task goal fully before acting and keep every explicit requirement in "
    "mind — price, quantity, attributes, ordering, and any stated preference — for "
    "the rest of the episode; do not silently drop one partway through. Use only "
    "the tools supplied for this specific task; never assume a fixed tool set or "
    "hardcode a tool name from memory.\n"
    "When you evaluate a candidate against a requirement, separate the requirement "
    "into an exact-match part (brand, model, code, size, color, or any other literal "
    "identifier — these must appear explicitly, never guessed or normalized) and a "
    "flexible part (everything else, where reasonable paraphrase and common synonyms "
    "are fine). Trust structured fields (attributes, specs, stock, price) over a "
    "product's title text — a title only proves a word is present, not that every "
    "claim about the product is true. When you must choose among several plausible "
    "candidates, briefly compare the strongest ones against each other using that "
    "structured evidence before committing, and be ready to say why the one you "
    "picked is better than the runner-up.\n"
    "If asked to justify or explain your final choice, only state facts about the "
    "chosen item that a tool observation in this episode actually returned for it. "
    "Cover every fact the task asked you to justify (price, stock, and any named "
    "attribute) using the exact values you observed, but never add an extra "
    "attribute claim (display type, resolution, OS, connectivity, or anything "
    "similar) that you did not verify through a tool call this episode — omit an "
    "unconfirmed detail rather than guess it from the title, brand, or general "
    "knowledge; an unverifiable claim is worse than no claim.\n"
    "State and prices can change mid-episode. If a later observation or a message "
    "shows the price, stock, or ranking of items has changed since you last checked, "
    "explicitly note that change (send a short message about it if a messaging tool "
    "is available) and re-verify the current best option against the same "
    "requirements before acting — do not act on stale information, and do not treat "
    "your earlier read of the market as still valid without re-checking it. If a "
    "tool call fails, times out, or returns nothing useful, change your next call "
    "(different query, filter, or target) instead of repeating the same one or "
    "giving up. Before your final action, double check it is supported by what the "
    "tools returned so far. Continue issuing actions until an observation reports "
    "done=true."
)


def _provider() -> str:
    return getenv("INFERENCE_PROVIDER", "chutes")


def _model_priorities() -> list[str]:
    chain = _OPENROUTER_PRIORITIES if _provider() == "openrouter" else _CHUTES_PRIORITIES
    override = getenv("SANDBOX_MODEL")
    if not override:
        return chain
    return [override] + [m for m in chain if m != override]


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_calls: int):
    """Walk the model priority chain; never let one provider's hiccup end the episode.

    Mirrors 117_ACE.py's ProxyUtil.trigger_llm: bump temperature on a bad/empty
    response instead of retrying the exact same deterministic call.
    """
    temperature = 0.0
    for _pass in range(MAX_MODEL_PASSES):
        for model in _model_priorities():
            try:
                resp = _LLM_PROXY.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "required",
                        "temperature": min(temperature, 1.0),
                    },
                )
            except Exception:
                temperature += TEMP_UP
                continue
            if not resp or not resp.get("choices"):
                temperature += TEMP_UP
                continue
            assistant = resp["choices"][0]["message"]
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                return assistant, tool_calls
            temperature += TEMP_UP
    return None, []


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue: list[dict[str, Any]] = []
    try:
        environment = problem_data["environment"]
        binding = environment["binding"]
        policy = environment["policy_view"]
        problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
        max_steps = max(1, int(policy.get("max_steps", 20)))
        tools = policy.get("tools") or []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": policy["query"]},
        ]

        for turn in range(1, max_steps + 1):
            assistant, tool_calls = None, []
            for _attempt in range(MAX_TOOLCALL_RETRIES):
                assistant, tool_calls = _chat(messages, tools, max_calls)
                if tool_calls:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The environment has not reported done=true. Choose one "
                            "of the supplied tools to continue."
                        ),
                    }
                )
            if not tool_calls:
                dialogue.append(
                    {
                        "role": "assistant",
                        "content": "No model in the priority chain returned a tool call this turn.",
                    }
                )
                break

            assistant_content = (assistant or {}).get("content") or ""
            messages.append(
                {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
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
                            "name": (call.get("function") or {}).get("name", ""),
                            "args": _arguments(call),
                        },
                    }
                    for index, call in enumerate(tool_calls, start=1)
                ],
            }

            result = None
            try:
                result = _ENV_PROXY.post("/environment/call", json_data=envelope)
            except Exception as exc:
                dialogue.append({"role": "assistant", "content": f"environment call failed: {exc}"})

            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                    "environment_result": result,
                }
            )
            if not result or "calls" not in result:
                break

            for tool_call, call_result in zip(tool_calls, result["calls"]):
                observation = call_result.get("observation", {})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": json.dumps(observation),
                    }
                )
            if result.get("user_message"):
                messages.append({"role": "user", "content": result["user_message"]["content"]})
            if any((c.get("observation") or {}).get("done") for c in result["calls"]):
                break
    except Exception as exc:
        # Mirrors 117_ACE.py's agent_main: never propagate — a raised exception
        # forfeits the episode outright, while a returned dialogue can still be
        # finalized (typically at zero reward) by the trusted runtime.
        dialogue.append(
            {"role": "assistant", "content": f"Unhandled agent error: {type(exc).__name__}: {exc}"}
        )
    return dialogue


__all__ = ["agent_main"]
