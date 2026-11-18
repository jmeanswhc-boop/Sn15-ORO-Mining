"""Shopping agent for ORO Bench generated environments.

Reliable environment loop plus observation-grounded order cleanup and finish
pressure so episodes complete with truthful tool arguments.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_proxy = ProxyClient(timeout=120, max_retries=2)

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _order_schema(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    for spec in tools:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if isinstance(fn, dict) and fn.get("name") == "place_test_order":
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            return params
    return None


def _tool_required(tools: list[dict[str, Any]], name: str) -> list[str]:
    for spec in tools:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        return [x for x in params.get("required", []) if isinstance(x, str)]
    return []


def _allowed_claim_fields(tools: list[dict[str, Any]]) -> set[str]:
    """Read allowed justification field names from the live tool schema only."""
    params = _order_schema(tools) or {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
    jprops = just.get("properties") if isinstance(just.get("properties"), dict) else {}
    claims = jprops.get("claims") if isinstance(jprops.get("claims"), dict) else {}
    items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
    iprops = items.get("properties") if isinstance(items.get("properties"), dict) else {}
    field = iprops.get("field") if isinstance(iprops.get("field"), dict) else {}
    allowed: set[str] = set()
    enum = field.get("enum")
    if isinstance(enum, list):
        allowed.update(str(x) for x in enum if isinstance(x, str))
    # Also accept snake_case tokens named by the schema description itself.
    for blob in (field.get("description"), just.get("description"), claims.get("description")):
        if isinstance(blob, str) and blob.strip():
            allowed.update(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", blob))
            allowed.update(
                tok
                for tok in re.findall(r"\b[a-z][a-z0-9]{2,}\b", blob)
                if tok
                not in {
                    "the",
                    "and",
                    "for",
                    "with",
                    "from",
                    "that",
                    "this",
                    "must",
                    "only",
                    "field",
                    "value",
                    "claim",
                    "claims",
                    "true",
                    "false",
                    "string",
                    "object",
                    "array",
                    "required",
                    "optional",
                    "product",
                    "order",
                    "justify",
                    "justification",
                }
            )
    # Catalog truth fields always appear on inspect_stock observations.
    allowed.update({"price", "in_stock"})
    return allowed


def _system_prompt(tools: list[dict[str, Any]], max_calls: int, max_steps: int) -> str:
    needs_justification = "justification" in _tool_required(tools, "place_test_order")
    allowed = sorted(_allowed_claim_fields(tools)) if needs_justification else []

    lines = [
        "You are a shopping assistant in a product catalog.",
        "Help the shopper find a valid in-stock item, confirm it, and complete the purchase with the tools.",
        "Follow the shopper request and every attached instruction. Never invent product_id or sku values.",
        "",
        "Method:",
        "- Search with short keywords (2-5 words). Reformulate or use filter when results are poor.",
        "- Prefer in_stock=true when available.",
        f"- Up to {max_calls} tool calls per turn; batch independent reads when useful.",
        "- Inspect with view/compare before the first add_to_cart. Cart only a real intended choice.",
        "- After cart changes, call inspect_stock. If invalid, remove_from_cart, recover, then order.",
        "- Order path: view/compare -> add_to_cart -> inspect_stock -> place_test_order.",
        "- Answer shopper questions with message before ordering.",
        f"- Budget about {max_steps} turns. When verified, place_test_order immediately.",
        "- Do not spend many turns only messaging or repeatedly removing from cart without ordering.",
    ]
    if needs_justification:
        lines.append(
            "- place_test_order requires justification claims. Use only field names supported by "
            f"the tool schema ({', '.join(allowed)}). Values must match inspected facts for the "
            "final product. Always include truthful price and in_stock from inspect_stock. "
            "Never invent unsupported features."
        )
    else:
        lines.append(
            "- Omit justification on place_test_order unless the tool schema requires it."
        )

    lines.append("")
    lines.append("Tools:")
    for spec in tools:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            lines.append(f"- {fn['name']}: {str(fn.get('description') or '').strip()}")
    lines.append("")
    lines.append("Continue until an observation reports done=true.")
    return "\n".join(lines)


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    if isinstance(raw, dict):
        parsed: Any = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*}", "}", raw.replace("'", '"'))
            fixed = re.sub(r",\s*]", "]", fixed)
            parsed = json.loads(fixed)
    else:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _ground_order_args(
    args: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    stock_by_key: dict[str, dict[str, Any]],
    viewed_facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cleaned = dict(args)
    required = _tool_required(tools, "place_test_order")
    if "justification" in cleaned and "justification" not in required:
        cleaned.pop("justification", None)
        return cleaned
    if "justification" not in required:
        return cleaned

    pid = str(cleaned.get("product_id") or "")
    sku = str(cleaned.get("sku") or "")
    key = f"{pid}::{sku}"
    stock = stock_by_key.get(key) or stock_by_key.get(pid) or {}
    facts = viewed_facts.get(pid) or {}
    allowed = _allowed_claim_fields(tools)

    raw_claims = []
    just = cleaned.get("justification")
    if isinstance(just, dict) and isinstance(just.get("claims"), list):
        raw_claims = [c for c in just["claims"] if isinstance(c, dict)]

    grounded: list[dict[str, str]] = []
    seen_fields: set[str] = set()

    def _lookup(field: str) -> Any:
        for blob in (stock, facts):
            if field in blob and blob.get(field) not in (None, ""):
                return blob.get(field)
            specs = blob.get("specification") if isinstance(blob.get("specification"), dict) else {}
            attrs = blob.get("attributes") if isinstance(blob.get("attributes"), dict) else {}
            for nest in (specs, attrs):
                if field in nest and nest.get(field) not in (None, ""):
                    return nest.get(field)
        return None

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen_fields:
            return
        if value is None or value == "":
            return
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value).strip()
        if not text:
            return
        grounded.append({"field": field, "value": text})
        seen_fields.add(field)

    # Truthful live stock/price first.
    if "price" in stock:
        add("price", stock.get("price"))
    if "in_stock" in stock:
        add("in_stock", stock.get("in_stock"))

    # Keep model claims only when field is schema-allowed; prefer observed values.
    for claim in raw_claims:
        field = str(claim.get("field") or "").strip()
        if field in {"price", "in_stock"}:
            continue
        if field not in allowed:
            continue
        observed = _lookup(field)
        if observed is not None:
            add(field, observed)
            continue
        # Drop unsupported/invented values when we have no observation for the field.
        value = claim.get("value")
        if isinstance(value, bool) or str(value).lower() in {"true", "false"}:
            # Boolean feature claims need observed evidence.
            continue
        add(field, value)

    # Fill remaining observed facts that the schema allows.
    for field in sorted(allowed):
        if field in seen_fields:
            continue
        observed = _lookup(field)
        if observed is not None:
            add(field, observed)

    if not grounded:
        if "price" in stock:
            add("price", stock.get("price"))
        if "in_stock" in stock:
            add("in_stock", stock.get("in_stock"))

    cleaned["justification"] = {"claims": grounded}
    return cleaned


def _clean_args(
    name: str,
    args: dict[str, Any],
    tools: list[dict[str, Any]],
    stock_by_key: dict[str, dict[str, Any]],
    viewed_facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cleaned = dict(args)
    if name == "search" and "in_stock" not in cleaned:
        cleaned["in_stock"] = True
    if name == "place_test_order":
        cleaned = _ground_order_args(
            cleaned,
            tools=tools,
            stock_by_key=stock_by_key,
            viewed_facts=viewed_facts,
        )
    return cleaned


def _remember_observation(
    name: str,
    args: dict[str, Any],
    observation: dict[str, Any],
    stock_by_key: dict[str, dict[str, Any]],
    viewed_facts: dict[str, dict[str, Any]],
) -> None:
    if name == "inspect_stock":
        pid = str(observation.get("product_id") or args.get("product_id") or "")
        sku = str(observation.get("sku") or args.get("sku") or "")
        if pid:
            stock_by_key[f"{pid}::{sku}"] = dict(observation)
            stock_by_key[pid] = dict(observation)
    if name in {"view", "compare"}:
        # view payload usually has product_id at top level; compare has compared rows.
        pid = observation.get("product_id")
        if isinstance(pid, str) and pid:
            viewed_facts[pid] = dict(observation)
        for row in observation.get("compared") or []:
            if isinstance(row, dict) and isinstance(row.get("product_id"), str):
                viewed_facts[row["product_id"]] = dict(row)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Satisfy the shopper with dynamic environment tools until done=true."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    tools = list(policy.get("tools") or [])
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = int(policy["max_steps"])

    stock_by_key: dict[str, dict[str, Any]] = {}
    viewed_facts: dict[str, dict[str, Any]] = {}
    carted = False
    removed = False

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(tools, max_calls, max_steps)},
        {"role": "user", "content": policy["query"]},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, max_steps + 1):
        left = max_steps - turn + 1
        nudge = None
        if left <= 6:
            nudge = (
                f"{left} turns left. If a valid item is inspected, add_to_cart, "
                "inspect_stock, and place_test_order now."
            )
        if left <= 3:
            nudge = (
                f"Only {left} turns left. Place the order for the best verified "
                "in-cart variant immediately."
            )
        if removed and carted is False:
            nudge = (
                (nudge + " " if nudge else "")
                + "The cart was cleared. Select another valid product, cart it, "
                "re-check stock, and order."
            )

        ask = list(messages)
        if nudge:
            ask.append({"role": "user", "content": nudge})

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(2):
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model(),
                    "messages": ask,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                raise RuntimeError("inference request failed")
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            ask.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": (
                            "Continue with a tool call toward inspecting and ordering "
                            "a valid product."
                        ),
                    },
                ]
            )
        else:
            raise RuntimeError("model returned no tool call before completion")

        prepared: list[dict[str, Any]] = []
        action_pairs: list[tuple[str, dict[str, Any]]] = []
        for call in tool_calls:
            try:
                name = call["function"]["name"]
                args = _clean_args(
                    name,
                    _arguments(call),
                    tools,
                    stock_by_key,
                    viewed_facts,
                )
            except (ValueError, json.JSONDecodeError, KeyError, TypeError):
                continue
            action_pairs.append((name, args))
            prepared.append(
                {
                    "id": call.get("id") or f"call-{turn}-{len(prepared)+1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        if not prepared:
            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": "Call a tool with valid JSON arguments.",
                    },
                ]
            )
            continue
        tool_calls = prepared

        for name, _args in action_pairs:
            if name == "add_to_cart":
                carted = True
                removed = False
            if name == "remove_from_cart":
                removed = True
                carted = False

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
                    "action": {"name": name, "args": args},
                }
                for index, (name, args) in enumerate(action_pairs, start=1)
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

        for (name, args), tool_call, call_result in zip(
            action_pairs, tool_calls, result["calls"], strict=True
        ):
            obs = call_result["observation"]
            _remember_observation(name, args, obs, stock_by_key, viewed_facts)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
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
