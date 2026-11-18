"""ORO Bench environment agent ported from ShoppingBench seed heuristics: Tapir."""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_SEED_NAME = 'Tapir'
_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)

_SYSTEM_PROMPT = """You are a shopping agent (Tapir) operating inside ORO Bench generated environments.
Protocol rules (mandatory):
1. Use ONLY tools listed in policy_view.tools. Never invent tool names.
2. Copy the full environment.binding into every /environment/call envelope.
3. Prefer this loop: search and/or filter -> view/compare -> inspect_stock -> add_to_cart -> place_test_order.
4. Stop when any observation has done=true (normally after place_test_order).
5. Keep call_id / idempotency_key unique per turn.
6. When user_message arrives (shopper/simulator), incorporate it before the next action.
7. Do not claim success without a completed place_test_order observation.

Task-family playbook (family id is sealed; infer from the goal + tool schemas):
- intent_decomposition: clarify multi-part goals; search each sub-need; message if ambiguous.
- retrieval_recall: issue focused search queries; broaden only after empty/weak hits; view top hits.
- constraint_satisfaction: treat hard constraints as must-pass; use filter when schema allows; reject mismatches.
- preference_reasoning: soft prefs are ranked, not binary; prefer better matches then explain tradeoffs via message if needed.
- ranking: shortlist with compare; pick the best justified option.
- recovery: if stock/price/event breaks the plan, recover with new search/filter; do not place a broken order.
- justification: before place_test_order, ensure justification.claims (if required) cite concrete observed facts.

Seed-specific strategy:
Be strict on product attributes, brand, price band, and variants. Prefer filter+view before cart. Confirm stock/price with inspect_stock before place_test_order.
Budget discipline: extract numeric budget/threshold/discount language from the goal; keep the cart within feasible spend; prefer inspect_stock price checks before ordering.
Attribute discipline: brand, size, color, material, and price bounds in the goal are hard filters. Drop candidates that fail any hard attribute after view.
Multi-intent discipline: decompose the goal into item slots; search each slot; only add_to_cart SKUs that satisfy their slot; use compare when choosing among near-ties.
Search retry: if search returns empty or irrelevant hits, rewrite to shorter keywords (drop adjectives first) and search again before giving up.
Commit gate: call inspect_stock for the chosen product_id+sku immediately before add_to_cart / place_test_order; recover if out of stock or price shifted.
Strategy heritage: ported heuristics inspired by seed 'Tapir' (lane bias product > voucher > shop).
"""

_BUDGET_RE = re.compile(
    r"(?:budget|under|below|max(?:imum)?|no more than|at most|within)\s*(?:of\s*)?"
    r"(?:php|₱|\$)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.I,
)


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


def _parse_budget_hint(query: str) -> float | None:
    hits = _BUDGET_RE.findall(query or "")
    if not hits:
        return None
    try:
        return float(str(hits[-1]).replace(",", ""))
    except ValueError:
        return None


def _keyword_hints(query: str) -> list[str]:
    q = re.sub(r"\s+", " ", (query or "").strip())
    if not q:
        return []
    # Drop very common stop words; keep contentful tokens for broaden-search coaching.
    stop = {
        "a", "an", "the", "and", "or", "for", "with", "from", "that", "this", "these",
        "those", "to", "of", "in", "on", "is", "are", "be", "as", "at", "by", "it",
        "my", "me", "i", "we", "you", "your", "please", "looking", "find", "show",
        "need", "want", "get", "buy", "purchase",
    }
    toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", q.lower()) if t not in stop]
    # Unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:16]


def _strategy_user_preamble(query: str) -> str:
    parts = [query]
    budget = _parse_budget_hint(query)
    if budget is not None:
        parts.append(f"[strategy_hint] Detected spend ceiling-like amount: {budget}")
    kws = _keyword_hints(query)
    if kws:
        parts.append("[strategy_hint] Keyword backbone for search retries: " + ", ".join(kws[:10]))
    parts.append(
        "[strategy_hint] Before place_test_order: inspect_stock the chosen sku; "
        "if required, fill justification.claims with facts from view/compare observations."
    )
    return "\n".join(parts)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    query = policy.get("query") or problem_data.get("query") or ""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _strategy_user_preamble(str(query))},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, int(policy["max_steps"]) + 1):
        for _attempt in range(2):
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model(),
                    "messages": messages,
                    "tools": policy["tools"],
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                raise RuntimeError("inference request failed")
            assistant = inference["choices"][0]["message"]
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
                            "Choose one of the supplied tools to continue. "
                            "If search was weak, broaden keywords; if ready to buy, "
                            "inspect_stock then place_test_order."
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
                "seed": _SEED_NAME,
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
        if result.get("user_message"):
            messages.append(
                {"role": "user", "content": result["user_message"]["content"]}
            )
        if any(call["observation"]["done"] for call in result["calls"]):
            break

    return dialogue


__all__ = ["agent_main"]
