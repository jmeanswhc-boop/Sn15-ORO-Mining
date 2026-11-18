"""Shopping agent for ORO Bench generated environments.

Environment tool loop with observation-grounded place_test_order claims and
finish pressure so episodes terminate with truthful, schema-valid arguments.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_proxy = ProxyClient(timeout=180, max_retries=3)

_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

# Words that appear in schema prose but are not claim field names.
_SCHEMA_STOPWORDS = {
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
    "fields",
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
    "supported",
    "reason",
    "reasons",
    "useful",
    "such",
    "also",
    "use",
    "shopper",
    "request",
    "specification",  # instruction word, not a claim field
    "feature",
    "features",
    "explicitly",
    "positive",
    "capabilities",
    "unknown",
    "absent",
    "privacy",  # part of privacy_cover in prose
    "cover",
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


def _justification_field_desc(tools: list[dict[str, Any]]) -> str:
    params = _order_schema(tools) or {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
    jprops = just.get("properties") if isinstance(just.get("properties"), dict) else {}
    claims = jprops.get("claims") if isinstance(jprops.get("claims"), dict) else {}
    items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
    iprops = items.get("properties") if isinstance(items.get("properties"), dict) else {}
    field = iprops.get("field") if isinstance(iprops.get("field"), dict) else {}
    parts = [
        field.get("description"),
        just.get("description"),
        claims.get("description"),
    ]
    return " ".join(str(p) for p in parts if isinstance(p, str) and p.strip())


def _soft_reason_fields(tools: list[dict[str, Any]]) -> set[str]:
    """Soft reason tokens named by the live place_test_order schema text."""
    desc = _justification_field_desc(tools).lower()
    found: set[str] = set()
    enum_match = re.search(
        r"(?:reason|reasons)\s*:?\s*([^.]+)",
        desc,
        flags=re.IGNORECASE,
    )
    blob = enum_match.group(1) if enum_match else desc
    found.update(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", blob))
    for tok in re.findall(r"\b[a-z][a-z0-9]{2,}\b", blob):
        if tok not in _SCHEMA_STOPWORDS:
            found.add(tok)
    # privacy cover / HDR often appear as prose in the schema.
    if "privacy" in desc and "cover" in desc:
        found.add("privacy_cover")
    if re.search(r"\bhdr\b", desc):
        found.add("hdr")
    found.difference_update(_SCHEMA_STOPWORDS)
    found.discard("price")
    found.discard("in_stock")
    return found


def _allowed_claim_fields(tools: list[dict[str, Any]], extra: set[str] | None = None) -> set[str]:
    params = _order_schema(tools) or {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
    jprops = just.get("properties") if isinstance(just.get("properties"), dict) else {}
    claims = jprops.get("claims") if isinstance(jprops.get("claims"), dict) else {}
    items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
    iprops = items.get("properties") if isinstance(items.get("properties"), dict) else {}
    field = iprops.get("field") if isinstance(iprops.get("field"), dict) else {}
    allowed: set[str] = {"price", "in_stock"}
    enum = field.get("enum")
    if isinstance(enum, list):
        allowed.update(str(x) for x in enum if isinstance(x, str))
    allowed.update(_soft_reason_fields(tools))
    if extra:
        allowed.update(extra)
    allowed.difference_update({"specification"})
    return allowed


def _shopper_preamble(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


def _infer_spec_claim(query: str, product: dict[str, Any]) -> tuple[str, str] | None:
    """Infer the shopper's required specification claim from the request + facts."""
    text = _shopper_preamble(query)
    q = text.lower()
    facts = product.get("facts") if isinstance(product.get("facts"), dict) else {}
    merged = {**facts, **{k: v for k, v in product.items() if k != "facts"}}

    inch = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*inch(?:es)?", q)
    if inch:
        raw = merged.get("size") or merged.get("screen_size")
        # Prefer the shopper's stated size; catalog size strings vary widely.
        return "screen_size", f"{inch.group(1)}-inch"

    if re.search(r"\b(full\s*hd|fhd|1080p)\b", q):
        disp = merged.get("display") or merged.get("resolution")
        if isinstance(disp, str) and re.search(r"1920|1080", disp):
            return "resolution", disp.strip()
        return "resolution", "fhd"

    if re.search(r"\b(720p|hd)\b", q):
        return "resolution", "hd"

    if re.search(r"\b(4k|2160p|uhd)\b", q):
        disp = merged.get("display") or merged.get("resolution")
        if isinstance(disp, str) and disp.strip():
            return "resolution", disp.strip()
        return "resolution", "4k"

    hz = re.search(r"(\d+)\s*hz\b", q)
    if hz:
        return "refresh_rate", f"{hz.group(1)}hz"

    ram = re.search(r"(\d+)\s*gb(?:\s*ram)?\b", q)
    if ram and "ram" in q:
        return "ram", f"{ram.group(1)}GB"

    return None


def _system_prompt(tools: list[dict[str, Any]], max_calls: int, max_steps: int) -> str:
    needs_justification = "justification" in _tool_required(tools, "place_test_order")
    soft = sorted(_soft_reason_fields(tools)) if needs_justification else []

    lines = [
        "You are a shopping assistant in a product catalog.",
        "Help the shopper find a valid in-stock item, confirm it, and complete the purchase with the tools.",
        "Follow the shopper request and every attached instruction. Never invent product_id or sku values.",
        "",
        "Method:",
        "- Search with short keywords (2-5 words). Reformulate or use filter when results are poor.",
        "- Prefer in_stock=true when available.",
        f"- Up to {max_calls} tool calls per turn; batch independent reads (multi-view) when useful.",
        "- Inspect with view/compare before the first add_to_cart. Cart only a real intended choice.",
        "- Check category_path and raw facts against firm requirements; reject accessories when a device is required.",
        "- After cart changes, call inspect_stock. If invalid, remove_from_cart, recover, then order.",
        "- Order path: view/compare -> add_to_cart -> inspect_stock -> place_test_order.",
        "- Answer shopper questions with message before ordering.",
        f"- Budget about {max_steps} turns. Once verified, place_test_order immediately — do not wait for permission.",
        "- Do not spend many turns only messaging or repeatedly removing from cart without ordering.",
    ]
    if needs_justification:
        lines.append(
            "- place_test_order requires structured justification claims. Include: "
            "(1) the specification field implied by the shopper request "
            "(for example resolution, screen_size, refresh_rate, or ram — not a field named "
            "'specification'), (2) price and in_stock from the latest inspect_stock, and "
            f"(3) at least one supported reason from the tool schema ({', '.join(soft) or 'brand'}). "
            "Ground every claim in viewed raw facts / current stock. Never invent features."
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


def _payload(observation: dict[str, Any]) -> dict[str, Any]:
    """Unwrap environment observation envelopes to the action payload."""
    if not isinstance(observation, dict):
        return {}
    inner = observation.get("observation")
    if isinstance(inner, dict) and (
        any(
            key in inner
            for key in (
                "product_id",
                "price",
                "facts",
                "results",
                "compared",
                "order",
                "status",
            )
        )
        or "error" in observation
    ):
        return inner
    return observation


def _flatten_product(row: dict[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    facts = row.get("facts")
    if isinstance(facts, dict):
        for key, value in facts.items():
            flat.setdefault(key, value)
    return flat


def _lookup(product: dict[str, Any], field: str) -> Any:
    if field in product and product.get(field) not in (None, ""):
        return product.get(field)
    for nest_name in ("facts", "specification", "attributes", "specs"):
        nest = product.get(nest_name)
        if isinstance(nest, dict) and nest.get(field) not in (None, ""):
            return nest.get(field)
    return None


def _ground_order_args(
    args: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    query: str,
    stock_by_key: dict[str, dict[str, Any]],
    viewed_facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cleaned = dict(args)
    required = _tool_required(tools, "place_test_order")
    if "justification" not in required:
        cleaned.pop("justification", None)
        return cleaned

    pid = str(cleaned.get("product_id") or "")
    sku = str(cleaned.get("sku") or "")
    key = f"{pid}::{sku}"
    stock = stock_by_key.get(key) or stock_by_key.get(pid) or {}
    product = viewed_facts.get(pid) or {}

    spec = _infer_spec_claim(query, product)
    extra = {spec[0]} if spec else set()
    soft = _soft_reason_fields(tools)
    allowed = _allowed_claim_fields(tools, extra)

    grounded: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(field: str, value: Any) -> None:
        if field not in allowed or field in seen:
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
        seen.add(field)

    # Required live state claims only — do not attach unrelated catalog fields.
    if "price" in stock:
        add("price", stock.get("price"))
    elif "price" in product:
        add("price", product.get("price"))
    if "in_stock" in stock:
        add("in_stock", stock.get("in_stock"))
    elif "in_stock" in product:
        add("in_stock", product.get("in_stock"))

    # Required specification claim from the shopper request.
    if spec:
        add(spec[0], spec[1])

    # Exactly one soft reason when possible: prefer observed brand.
    brand = _lookup(product, "brand")
    if brand and "brand" in soft:
        add("brand", brand)
    elif soft:
        for field in sorted(soft):
            if field == "brand":
                continue
            observed = _lookup(product, field)
            if observed is True or str(observed).lower() == "true":
                add(field, "true")
                break

    cleaned["justification"] = {"claims": grounded}
    return cleaned


def _clean_args(
    name: str,
    args: dict[str, Any],
    tools: list[dict[str, Any]],
    query: str,
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
            query=query,
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
    payload = _payload(observation)
    if name == "inspect_stock":
        pid = str(payload.get("product_id") or args.get("product_id") or "")
        sku = str(payload.get("sku") or args.get("sku") or "")
        if pid:
            stock_by_key[f"{pid}::{sku}"] = dict(payload)
            stock_by_key[pid] = dict(payload)
    if name == "view":
        pid = payload.get("product_id")
        if isinstance(pid, str) and pid:
            viewed_facts[pid] = _flatten_product(payload)
    if name == "compare":
        rows = payload.get("compared") or payload.get("products") or []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("product_id"), str):
                    viewed_facts[row["product_id"]] = _flatten_product(row)
        pid = payload.get("product_id")
        if isinstance(pid, str) and pid:
            viewed_facts[pid] = _flatten_product(payload)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Satisfy the shopper with dynamic environment tools until done=true."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    tools = list(policy.get("tools") or [])
    query = str(policy.get("query") or "")
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = int(policy["max_steps"])

    stock_by_key: dict[str, dict[str, Any]] = {}
    viewed_facts: dict[str, dict[str, Any]] = {}
    carted = False
    removed = False

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(tools, max_calls, max_steps)},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []

    for turn in range(1, max_steps + 1):
        left = max_steps - turn + 1
        nudge = None
        if left <= 8:
            nudge = (
                f"{left} turns left. If a candidate already satisfies the firm "
                "requirements, finish: add_to_cart (if needed), inspect_stock, "
                "place_test_order."
            )
        if left <= 4:
            nudge = (
                f"Only {left} turns left. Place the order for the best verified "
                "in-cart variant immediately with valid arguments."
            )
        if removed and not carted:
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
                    query,
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
