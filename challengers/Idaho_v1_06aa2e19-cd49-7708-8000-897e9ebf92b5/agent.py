"""ORO Bench shopping agent for generated environments (TF1–TF7).

Drives the dynamic tool contract via /environment/call until done=true.
Optimized for the seven EnvPack families: intent decomposition, retrieval
recall, constraint satisfaction, preference reasoning, ranking, recovery,
and justification.
"""

from __future__ import annotations

import json
import logging
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "chutes": "Qwen/Qwen3.5-397B-A17B-TEE",
    "openrouter": "qwen/qwen3.5-397b-a17b",
}
# Fallback if the preferred frontier model is rejected by the allowlist.
_FALLBACK_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

_proxy = ProxyClient(timeout=120, max_retries=2)

_OBS_CHARS = 6000
_MAX_PARALLEL = 6

_SYSTEM = """You are an expert shopping agent for a live product catalog.

## Tools
Use ONLY the supplied tools. Never invent product_id or sku values.
Typical loop: search/filter → view/compare → inspect_stock → (message if needed) → add_to_cart → handle shopper replies / price-stock events → inspect again → place_test_order.

## Hard rules for every task
1. Firm requirements always beat soft preferences (brand, color, niceness, cheapest-first).
2. Order the EXACT variant (product_id + sku). Titles can mix right and wrong specs — always view and inspect_stock that SKU.
3. Prices and stock change after cart actions. After any cart change or shopper message about price/stock, re-inspect before ordering.
4. First successful add_to_cart is scored as your real commitment — never use a sacrificial placeholder.
5. Once the final item is verified in-stock and within budget, place_test_order. Do not wait for permission.
6. When the shopper asks for a reason, place_test_order MUST include justification.claims as [{field, value}, ...] only (no summary/evidence text). Include the requested spec field, price, in_stock, plus at least one useful reason field (brand / feature) grounded in view/compare facts.

## Family playbooks (follow the matching one from the shopper brief)
- Intent mix (firm vs soft): honor preferred brand on the FIRST cart only when a fully compliant preferred-brand item exists; never drop RAM/screen/resolution/refresh, budget, category, or stock for a soft preference. On reprice above budget, switch to any firm-compliant in-stock item.
- Retrieval: reformulate search queries (distinct phrasings) and survey several distinct products before committing; keep matches visible in search/filter results.
- Exact constraints: every gated attr + brand/category/budget/stock must hold on the ordered SKU. Resolution classes: HD≥1280x720, FHD≥1920x1080, QHD≥2560x1440, 4K/UHD≥3840x2160 (longer×shorter). If underspecified, message the shopper to elicit the sealed spec. On stockout, do not swap to a wrong SKU of the same product.
- Soft preferences: before first cart, message to clarify priorities; compare ≥2 listing_ids that meet hard reqs; pick best preference fit on raw facts (battery, materials, connectivity, use features), not cheapest-only. After reprice, recover to a different listing_id.
- Ranking: priority is refresh (incl. advertised overclock) ↓ screen size ↓ price. First cart must be the best live option. After stockout/reprice, re-rank remaining options and inspect the replacement.
- Recovery: never relax category/budget/stock. Prefer exact model+color → same model other color → other model. Commit to the full match first when available (may trigger stockout), then minimize relaxation.
- Justification: view/compare before first cart and again before order; claims must match inspected facts of the final product.

## Efficiency
Batch independent reads (multiple search/view/compare/inspect_stock) in one turn when useful. Keep messages to the shopper short and specific. Prefer k≤10. Stop when an observation reports done=true.
"""


def _provider() -> str:
    return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()


def _model() -> str:
    override = getenv("AGENT_MODEL") or getenv("SANDBOX_MODEL")
    if override:
        return override
    return _DEFAULT_MODELS.get(_provider(), _DEFAULT_MODELS["chutes"])


def _fallback_model() -> str:
    return _FALLBACK_MODELS.get(_provider(), _FALLBACK_MODELS["chutes"])


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        parsed = raw
    else:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _trim_obs(obs: Any) -> str:
    text = json.dumps(obs, ensure_ascii=False, default=str)
    if len(text) <= _OBS_CHARS:
        return text
    return text[:_OBS_CHARS] + f"...[truncated {len(text) - _OBS_CHARS} chars]"


def _family_hint(query: str) -> str:
    q = query.lower()
    hints: list[str] = []
    if "soft preference" in q or "nice-to-have" in q or "preferred brand" in q:
        hints.append(
            "This looks like firm-vs-soft decomposition: protect hard specs/budget/stock; "
            "honor preferred brand on first cart only if fully compliant."
        )
    if "reformulate" in q or "survey" in q or "distinct" in q:
        hints.append(
            "This looks like retrieval recall: run several distinct searches and shortlist "
            "multiple products before cart."
        )
    if "exact variant" in q or "specific sku" in q or "resolution names" in q:
        hints.append(
            "This looks like constraint satisfaction: verify the exact SKU options; "
            "message if a gated spec is missing."
        )
    if "soft preferences matter" in q or "preference axes" in q or "listing_id" in q:
        hints.append(
            "This looks like preference reasoning: clarify priorities before first cart; "
            "compare listing_ids on raw facts."
        )
    if "refresh" in q and "largest screen" in q:
        hints.append(
            "This looks like ranking: refresh ↓ size ↓ price; first cart must be best; "
            "re-rank after events."
        )
    if "relax" in q or "another color" in q or "least-relaxed" in q:
        hints.append(
            "This looks like recovery: try exact model+color first, then minimize relaxation."
        )
    if "justification" in q or "trustworthy reason" in q or "structured" in q:
        hints.append(
            "This looks like justification: ground field/value claims in view/compare; "
            "include spec, price, in_stock, and one reason."
        )
    if not hints:
        return (
            "Follow the shopper brief exactly. Search, verify SKU price/stock, handle "
            "events, then place_test_order."
        )
    return " ".join(hints)


def _chat(
    *,
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
    """Return (assistant_content, tool_calls, model_used)."""
    primary = _model()
    models = [primary]
    fallback = _fallback_model()
    if fallback != primary:
        models.append(fallback)

    last_content = ""
    for model in models:
        for _attempt in range(2):
            inference = _chat(model=model, messages=messages, tools=tools)
            if inference is None:
                logger.warning("inference failed model=%s", model)
                break
            # Allowlist / provider rejection → try fallback model.
            if isinstance(inference, dict) and inference.get("error"):
                logger.warning("inference error model=%s err=%s", model, inference.get("error"))
                break
            try:
                assistant = inference["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                logger.warning("malformed inference payload model=%s", model)
                break
            content = assistant.get("content") or ""
            last_content = content
            tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                return content, tool_calls, model
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "The environment is not done. You must call one of the supplied "
                        "tools now (search, filter, view, compare, inspect_stock, "
                        "inspect_cart, add_to_cart, remove_from_cart, message, or "
                        "place_test_order)."
                    ),
                },
            ]
    return last_content, [], primary


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the generated-environment shopping loop for one problem."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    query = str(policy.get("query") or problem_data.get("query") or "")
    tools = policy["tools"]
    max_steps = int(policy.get("max_steps") or 30)
    max_calls = min(int(policy.get("max_calls_per_turn") or 1), _MAX_PARALLEL)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"{query}\n\n"
                f"Strategy note: {_family_hint(query)}\n"
                f"Budget of steps: {max_steps}. You may issue up to {max_calls} "
                "tool calls per turn."
            ),
        },
    ]
    dialogue: list[dict[str, Any]] = []
    carted = False

    for turn in range(1, max_steps + 1):
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
                "content": content,
                "tool_calls": tool_calls,
                "environment_result": result,
                "model": model_used,
            }
        )

        call_results = result.get("calls") or []
        for tool_call, call_result in zip(tool_calls, call_results):
            obs = (call_result or {}).get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _trim_obs(obs),
                }
            )
            action_name = tool_call.get("function", {}).get("name")
            if action_name == "add_to_cart":
                carted = True

        user_message = result.get("user_message") or {}
        user_text = user_message.get("content") if isinstance(user_message, dict) else None
        if user_text:
            follow = str(user_text)
            if carted or re.search(
                r"price|stock|sold out|unavailable|budget|changed",
                follow,
                re.I,
            ):
                follow += (
                    "\n\n[Agent reminder] Re-check live price/stock with inspect_stock "
                    "(and re-rank or recover if the market changed). Do not place an "
                    "order on stale facts. First cart commitment still counts."
                )
            messages.append({"role": "user", "content": follow})

        if any(
            isinstance(c, dict)
            and isinstance(c.get("observation"), dict)
            and c["observation"].get("done")
            for c in call_results
        ):
            break

        # Soft nudge near the step limit if still shopping.
        if turn >= max_steps - 3 and turn < max_steps:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Only {max_steps - turn} steps remain. Verify the best valid "
                        "SKU with inspect_stock and place_test_order (with justification "
                        "claims if the brief requires a reason)."
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
