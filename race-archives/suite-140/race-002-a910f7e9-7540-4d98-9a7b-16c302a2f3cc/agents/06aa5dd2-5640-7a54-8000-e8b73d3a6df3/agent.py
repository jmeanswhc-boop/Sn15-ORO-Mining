from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_ERPING_MODELS = {
    "chutes": ("google/gemma-4-31B-turbo-TEE",),
    "openrouter": ("google/gemma-4-31b-it",),
}
_BMARS_DECISION_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_ERPING_PROXY = ProxyClient(
    timeout=12,
    max_retries=1,
    retry_delay=0.5,
    rate_limit_retry_delay=2,
)
_BMARS_DECISION_PROXY = ProxyClient(
    timeout=28,
    max_retries=1,
    retry_delay=0.5,
    rate_limit_retry_delay=2,
)
_ERPING_POLICY = """You are a rigorous shopping agent in a live store. Complete the public shopper goal with only the supplied tools. Continue until an observation says done=true.

Separate hard requirements from preferences before acting. Category, requested specification, budget, and current stock are hard unless the shopper explicitly revises them. A preferred brand, color, feature, or use-case priority is soft unless the goal says otherwise. Apply every later shopper update and answer every direct question with the message tool before ordering.

Prefer no more than three distinct, precise search formulations, but continue with a genuinely new formulation when the observed catalog has not exposed a valid candidate. Search uses title words, while filter uses structured category, brand, price, and stock. Search with k=10 when surveying. Reformulate rather than repeating. Surface genuinely distinct candidates, then use view or compare on a focused shortlist. Reject accessories, wrong catalog categories, review pages, incompatible variants, and title-only guesses. Base the choice on observed option values for the chosen variant, its latest cost, and its availability.

Inspect the intended listing before modifying the cart for the first time. When the shopper requests preference clarification, ask one concise question first and evaluate two or more separate catalog listings. After the shopper reveals preference priorities, optimize every revealed preference axis among the hard-valid survivors and select the survivor with the least avoidable preference regret. When it gives a lexicographic ranking rule, calculate it exactly from observed variant facts and commit first to the best live candidate. When it asks for broad retrieval, establish the required identity coverage from search results before committing. Begin checkout while at least five turns remain. Once five turns remain, stop expanding the shortlist and use the best fully observed hard-valid survivor.

Cart additions are real commitments, never probes. After every mutation, observe the exact selected SKU again. A market event may be silent in the triggering response, so treat a changed price or stock as authoritative, remove an invalid item, and choose the least-relaxed valid replacement. Preserve category and budget; relax preferences only in their stated order. Build the relaxation ladder only from candidates whose observed category_path, latest observed cost, and availability satisfy every hard requirement. A matching model, color, or title never makes a wrong catalog category acceptable. Re-rank after every changed fact. Keep only the intended final item in the cart.

At order time, use the exact product_id and sku already in the cart. Keep justification field spelling exactly as exposed by the live order-tool schema and take values from the selected product's observations. Claims must contain only permitted field/value pairs and must exactly match viewed facts or the latest stock inspection; never invent a claim or add prose fields. Batch independent read-only calls when their arguments are already known. Do not stop, summarize, or ask for confirmation when a verified valid order can be placed."""


def _erping_provider() -> str:
    value = str(getenv("INFERENCE_PROVIDER", "chutes")).strip().lower()
    return "openrouter" if value == "openrouter" else "chutes"


def _erping_models() -> tuple[str, ...]:
    return _ERPING_MODELS[_erping_provider()]


def _bmars_decision_model() -> str:
    return _BMARS_DECISION_MODELS[_erping_provider()]


def _bmars_turn_limit(policy: dict[str, Any]) -> int:
    return min(30, max(1, int(policy["max_steps"])))


def _erping_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    raw = function.get("arguments", "{}")
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _erping_schemas(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        if name:
            found[name] = function
    return found


def _erping_valid_args(function: dict[str, Any], args: dict[str, Any]) -> bool:
    parameters = function.get("parameters") or {}
    required = parameters.get("required") or []
    if any(key not in args or args[key] in (None, "") for key in required):
        return False
    properties = parameters.get("properties") or {}
    for key, value in args.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            if parameters.get("additionalProperties") is False:
                return False
            continue
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return False
        if expected == "array" and not isinstance(value, list):
            return False
        if expected == "object" and not isinstance(value, dict):
            return False
        if expected == "boolean" and not isinstance(value, bool):
            return False
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return False
        if expected == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return False
        if expected in {"integer", "number"} and isinstance(value, (int, float)):
            if spec.get("minimum") is not None and value < spec["minimum"]:
                return False
            if spec.get("maximum") is not None and value > spec["maximum"]:
                return False
        if expected == "array" and isinstance(value, list):
            if spec.get("minItems") is not None and len(value) < spec["minItems"]:
                return False
            if spec.get("maxItems") is not None and len(value) > spec["maxItems"]:
                return False
    return True


def _erping_calls(
    message: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    maximum: int,
    turn: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        name = str(function.get("name") or "")
        schema = schemas.get(name)
        if schema is None:
            continue
        args = _erping_arguments(raw)
        if not _erping_valid_args(schema, args):
            continue
        marker = name + ":" + json.dumps(args, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        selected.append(
            {
                "id": f"erping-{turn}-{len(selected) + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, separators=(",", ":"), default=str),
                },
            }
        )
        if len(selected) >= maximum:
            break
    return selected


def _erping_action(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(call["function"]["name"]),
        "args": _erping_arguments(call),
    }


def _erping_target(args: dict[str, Any]) -> tuple[str, str]:
    return str(args.get("product_id") or ""), str(args.get("sku") or "")


def _erping_key(target: tuple[str, str]) -> str:
    return target[0] + "::" + target[1]


def _erping_actual(step: Any) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    nested = step.get("observation")
    return nested if isinstance(nested, dict) else {}


def _erping_synthetic(name: str, args: dict[str, Any], turn: int) -> dict[str, Any]:
    return {
        "id": f"erping-{turn}-guard",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, separators=(",", ":"), default=str),
        },
    }


def _erping_cart_targets(state: dict[str, Any]) -> list[tuple[str, str]]:
    rows = state.get("cart")
    if not isinstance(rows, list):
        return []
    return [
        (str(row.get("product_id") or ""), str(row.get("sku") or ""))
        for row in rows
        if isinstance(row, dict) and row.get("product_id") and row.get("sku")
    ]


def _erping_order_target(
    action: dict[str, Any], state: dict[str, Any]
) -> tuple[str, str]:
    target = _erping_target(action["args"])
    if all(target):
        return target
    cart = _erping_cart_targets(state)
    return cart[-1] if cart else ("", "")


def _erping_budget(text: str) -> float | None:
    matches = re.findall(
        r"(?:at\s+most|at\s+or\s+under|up\s+to|only\s+go\s+up\s+to|budget(?:\s+is|\s+of)?|under)\s*(?:PHP\s*)?([0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _erping_named_targets(text: str) -> list[tuple[str, str]]:
    return [
        (product_id, sku)
        for product_id, sku in re.findall(
            r"\b([A-Za-z0-9_-]+)::([A-Za-z0-9_.-]+)\b", text
        )
    ]


def _erping_category_ok(row: dict[str, Any], text: str) -> bool:
    path = row.get("category_path")
    if not isinstance(path, list) or not path:
        return True
    visible = " ".join(str(item) for item in path).casefold()
    explicit = re.findall(
        r"required\s+catalog\s+category\s+is\s+([^\n.,]+)",
        text,
        flags=re.IGNORECASE,
    )
    explicit.extend(
        re.findall(r"in\s+the\s+([^\n.,]+?)\s+category\b", text, flags=re.IGNORECASE)
    )
    if explicit:
        expected = explicit[-1].strip().casefold()
        expected = re.sub(r"\s+category$", "", expected).strip()
        words = [
            word
            for word in re.findall(r"[a-z0-9]+", expected)
            if word not in {"and", "the"}
        ]
        if words and all(word.rstrip("s") in visible for word in words):
            return True
        return False
    negative = re.findall(
        r"(?:not|without|exclude|excluding|reject)\s+(?:a\s+|an\s+|the\s+)?([^\n,.;]+)",
        text,
        flags=re.IGNORECASE,
    )
    visible_words = {word.rstrip("s") for word in re.findall(r"[a-z0-9]+", visible)}
    for phrase in negative:
        denied_words = {
            word.rstrip("s")
            for word in re.findall(r"[a-z0-9]+", phrase.casefold())
            if word not in {"and", "or", "evidence"}
        }
        if denied_words and denied_words.issubset(visible_words):
            return False
    return True


def _bmars_required_specs(text: str) -> list[tuple[str, str]]:
    lowered = text.casefold()
    found: list[tuple[str, str]] = []
    patterns = (
        ("screen", r"(\d+(?:\.\d+)?)\s*(?:-|\s)*(?:inch|inches)\s+screen"),
        ("screen", r"screen(?:\s+size)?[^\d]{0,16}(\d+(?:\.\d+)?)\s*(?:inch|inches)"),
        ("refresh", r"(\d+(?:\.\d+)?)\s*hz"),
        ("memory", r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory)"),
        ("resolution", r"(\d{3,4})\s*p\b"),
    )
    for kind, pattern in patterns:
        values = re.findall(pattern, lowered)
        if values:
            found.append((kind, str(values[-1])))
    return found


def _bmars_spec_ok(row: dict[str, Any], variant: dict[str, Any], text: str) -> bool:
    required = _bmars_required_specs(text)
    if not required:
        return True
    evidence = json.dumps(
        {"facts": row.get("facts"), "options": variant.get("options")},
        separators=(",", ":"),
        default=str,
    ).casefold()
    for kind, value in required:
        normalized = value.rstrip("0").rstrip(".") if "." in value else value
        number = re.escape(normalized)
        if kind == "screen" and not re.search(
            rf"(?<![\d.]){number}(?:\.0+)?\s*(?:-|\s)*(?:inch|inches|in\b)", evidence
        ):
            return False
        if kind == "refresh" and not re.search(
            rf"(?<![\d.]){number}(?:\.0+)?\s*hz", evidence
        ):
            return False
        if kind == "memory" and not re.search(
            rf"(?<![\d.]){number}(?:\.0+)?\s*gb", evidence
        ):
            return False
        if kind == "resolution" and not re.search(
            rf"(?<!\d){number}(?:p|\b)", evidence
        ):
            return False
    return True


def _erping_variant_target(
    state: dict[str, Any], product_id: str, budget: float | None, shopper_text: str
) -> tuple[str, str]:
    if not product_id or product_id in state["excluded"]:
        return "", ""
    row = state["products"].get(product_id)
    if not isinstance(row, dict):
        return "", ""
    if not _erping_category_ok(row, shopper_text):
        return "", ""
    for key, stock in reversed(list(state["stock"].items())):
        prefix, separator, sku = key.partition("::")
        if separator and prefix == product_id and isinstance(stock, dict):
            variant = next(
                (
                    item
                    for item in row.get("variants") or []
                    if isinstance(item, dict) and str(item.get("sku") or "") == sku
                ),
                {},
            )
            if not _bmars_spec_ok(row, variant, shopper_text):
                continue
            if stock.get("in_stock") is False:
                continue
            price = stock.get("price")
            if (
                budget is not None
                and isinstance(price, (int, float))
                and price > budget
            ):
                continue
            return product_id, sku
    variants = row.get("variants")
    if not isinstance(variants, list):
        return "", ""
    for variant in variants:
        if not isinstance(variant, dict) or not variant.get("sku"):
            continue
        if variant.get("in_stock") is False:
            continue
        if not _bmars_spec_ok(row, variant, shopper_text):
            continue
        price = variant.get("price")
        if budget is not None and isinstance(price, (int, float)) and price > budget:
            continue
        return product_id, str(variant["sku"])
    return "", ""


def _erping_finish_target(
    calls: list[dict[str, Any]], state: dict[str, Any], shopper_text: str
) -> tuple[str, str]:
    budget = _erping_budget(shopper_text)
    preferred: list[str] = []
    if state.get("selected_product"):
        target = _erping_variant_target(
            state, str(state["selected_product"]), budget, shopper_text
        )
        if all(target):
            return target
    for call in calls:
        action = _erping_action(call)
        if action["name"] == "compare":
            preferred.extend(
                str(item) for item in action["args"].get("product_ids") or []
            )
    preferred.extend(str(item) for item in state["shortlist"])
    for product_id in reversed(preferred):
        target = _erping_variant_target(state, product_id, budget, shopper_text)
        if all(target):
            return target
    for key in reversed(list(state["stock"])):
        product_id, separator, _ = key.partition("::")
        if separator:
            target = _erping_variant_target(state, product_id, budget, shopper_text)
            if all(target):
                return target
    for product_id in reversed(list(state["products"])):
        target = _erping_variant_target(state, product_id, budget, shopper_text)
        if all(target):
            return target
    return "", ""


def _erping_relaxed_target(state: dict[str, Any], shopper_text: str) -> tuple[str, str]:
    ranked: list[tuple[float, tuple[str, str]]] = []
    for product_id in reversed(list(state["products"])):
        target = _erping_variant_target(state, product_id, None, shopper_text)
        if not all(target):
            continue
        stock = state["stock"].get(_erping_key(target))
        price = stock.get("price") if isinstance(stock, dict) else None
        if not isinstance(price, (int, float)):
            row = state["products"].get(product_id) or {}
            variant = next(
                (
                    item
                    for item in row.get("variants") or []
                    if isinstance(item, dict)
                    and str(item.get("sku") or "") == target[1]
                ),
                {},
            )
            price = variant.get("price") if isinstance(variant, dict) else None
        ranked.append(
            (float(price) if isinstance(price, (int, float)) else float("inf"), target)
        )
    return min(ranked, default=(float("inf"), ("", "")), key=lambda item: item[0])[1]


def _erping_endgame(
    calls: list[dict[str, Any]],
    state: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    shopper_text: str,
    turn: int,
    maximum: int,
) -> list[dict[str, Any]]:
    left = maximum - turn + 1
    first = _erping_action(calls[0]) if calls else {"name": "", "args": {}}
    if (
        first["name"] == "message"
        and state["updates"]
        and "?" not in str(first["args"].get("content") or "")
        and not _erping_cart_targets(state)
        and "place_test_order" in schemas
    ):
        target = _erping_finish_target([], state, shopper_text)
        if (
            all(target)
            and state["checked_at"].get(_erping_key(target), -1) >= state["mutation"]
        ):
            return [
                calls[0],
                _erping_synthetic(
                    "add_to_cart", {"product_id": target[0], "sku": target[1]}, turn
                ),
            ]
    if left > 6 or "place_test_order" not in schemas:
        return calls
    if first["name"] == "message" and left > 3 and not state["messages_sent"]:
        return calls[:1]
    cart = _erping_cart_targets(state)
    if cart:
        target = cart[-1]
        budget = _erping_budget(shopper_text)
        stock = state["stock"].get(_erping_key(target))
        cart_row = next(
            (
                row
                for row in state.get("cart") or []
                if isinstance(row, dict)
                and (str(row.get("product_id") or ""), str(row.get("sku") or ""))
                == target
            ),
            {},
        )
        price = cart_row.get("current_price") if isinstance(cart_row, dict) else None
        invalid = (
            target[0] in state["excluded"]
            or (isinstance(stock, dict) and stock.get("in_stock") is False)
            or (isinstance(cart_row, dict) and cart_row.get("in_stock") is False)
            or (
                not state.get("terminal_fallback")
                and budget is not None
                and isinstance(price, (int, float))
                and price > budget
            )
        )
        if invalid and "remove_from_cart" in schemas:
            return [
                _erping_synthetic(
                    "remove_from_cart",
                    {"product_id": target[0], "sku": target[1]},
                    turn,
                )
            ]
        extra = next((item for item in cart if item != target), None)
        if extra is not None and "remove_from_cart" in schemas:
            return [
                _erping_synthetic(
                    "remove_from_cart",
                    {"product_id": extra[0], "sku": extra[1]},
                    turn,
                )
            ]
        checked = state["checked_at"].get(_erping_key(target), -1)
        if checked < state["mutation"] and "inspect_stock" in schemas:
            stock_call = _erping_synthetic(
                "inspect_stock",
                {"product_id": target[0], "sku": target[1]},
                turn,
            )
            if left == 1:
                return [
                    stock_call,
                    _erping_synthetic(
                        "place_test_order",
                        {"product_id": target[0], "sku": target[1]},
                        turn,
                    ),
                ]
            return [stock_call]
        return [
            _erping_synthetic(
                "place_test_order",
                {"product_id": target[0], "sku": target[1]},
                turn,
            )
        ]
    if first["name"] == "add_to_cart":
        return calls[:1]
    if first["name"] == "inspect_stock" and left >= 4:
        return calls[:1]
    target = _erping_finish_target(calls, state, shopper_text)
    if not all(target):
        if left >= 3 and "view" in schemas:
            for product_id in reversed(list(state["summaries"])):
                if product_id not in state["products"]:
                    return [_erping_synthetic("view", {"product_id": product_id}, turn)]
        if left <= 3:
            target = _erping_relaxed_target(state, shopper_text)
            if all(target):
                state["terminal_fallback"] = True
        if not all(target):
            return calls
    checked = state["stock"].get(_erping_key(target))
    if not isinstance(checked, dict) and left >= 4 and "inspect_stock" in schemas:
        return [
            _erping_synthetic(
                "inspect_stock",
                {"product_id": target[0], "sku": target[1]},
                turn,
            )
        ]
    add_call = _erping_synthetic(
        "add_to_cart",
        {"product_id": target[0], "sku": target[1]},
        turn,
    )
    if left == 1 and {"inspect_stock", "place_test_order"}.issubset(schemas):
        return [
            add_call,
            _erping_synthetic(
                "inspect_stock",
                {"product_id": target[0], "sku": target[1]},
                turn,
            ),
            _erping_synthetic(
                "place_test_order",
                {"product_id": target[0], "sku": target[1]},
                turn,
            ),
        ]
    return [add_call]


def _erping_guard(
    calls: list[dict[str, Any]], state: dict[str, Any], turn: int
) -> list[dict[str, Any]]:
    if not calls:
        return []
    actions = [_erping_action(call) for call in calls]
    for index, action in enumerate(actions):
        name = action["name"]
        if name == "message":
            return [calls[index]] if index == 0 else calls[:index]
        if name not in {"add_to_cart", "remove_from_cart", "place_test_order"}:
            continue
        if index:
            return calls[:index]
        if name == "remove_from_cart":
            return [calls[index]]
        if name == "add_to_cart":
            target = _erping_target(action["args"])
            if not all(target):
                return []
            locked_product = str(state.get("selected_product") or "")
            if (
                state.get("choice_locked")
                and locked_product
                and target[0] != locked_product
            ):
                preferred = _erping_variant_target(
                    state,
                    locked_product,
                    _erping_budget(str(state.get("shopper_text") or "")),
                    str(state.get("shopper_text") or ""),
                )
                if all(preferred):
                    if (
                        state["checked_at"].get(_erping_key(preferred), -1)
                        < state["mutation"]
                    ):
                        return [
                            _erping_synthetic(
                                "inspect_stock",
                                {"product_id": preferred[0], "sku": preferred[1]},
                                turn,
                            )
                        ]
                    return [
                        _erping_synthetic(
                            "add_to_cart",
                            {"product_id": preferred[0], "sku": preferred[1]},
                            turn,
                        )
                    ]
            if target[0] not in state["products"]:
                return [_erping_synthetic("view", {"product_id": target[0]}, turn)]
            if not _erping_category_ok(
                state["products"][target[0]], str(state.get("shopper_text") or "")
            ):
                state["excluded"].add(target[0])
                return []
            state["selected_product"] = target[0]
            if target in _erping_cart_targets(state):
                return [
                    _erping_synthetic(
                        "inspect_stock",
                        {"product_id": target[0], "sku": target[1]},
                        turn,
                    )
                ]
            return [calls[index]]
        target = _erping_order_target(action, state)
        if not all(target):
            return [_erping_synthetic("inspect_cart", {}, turn)]
        if target[0] not in state["products"]:
            return [_erping_synthetic("view", {"product_id": target[0]}, turn)]
        cart = _erping_cart_targets(state)
        if not _erping_category_ok(
            state["products"][target[0]], str(state.get("shopper_text") or "")
        ):
            state["excluded"].add(target[0])
            if target in cart:
                return [
                    _erping_synthetic(
                        "remove_from_cart",
                        {"product_id": target[0], "sku": target[1]},
                        turn,
                    )
                ]
            return []
        if target not in cart:
            return [
                _erping_synthetic(
                    "add_to_cart",
                    {"product_id": target[0], "sku": target[1]},
                    turn,
                )
            ]
        extra = next((item for item in cart if item != target), None)
        if extra is not None:
            return [
                _erping_synthetic(
                    "remove_from_cart",
                    {"product_id": extra[0], "sku": extra[1]},
                    turn,
                )
            ]
        checked = state["checked_at"].get(_erping_key(target), -1)
        if checked < state["mutation"]:
            return [
                _erping_synthetic(
                    "inspect_stock",
                    {"product_id": target[0], "sku": target[1]},
                    turn,
                )
            ]
        return [calls[index]]
    chosen: list[dict[str, Any]] = []
    new_queries: set[str] = set()
    for call, action in zip(calls, actions, strict=True):
        if action["name"] == "view":
            product_id = str(action["args"].get("product_id") or "")
            if product_id and product_id in state["products"]:
                target = _erping_variant_target(
                    state,
                    product_id,
                    _erping_budget(str(state.get("shopper_text") or "")),
                    str(state.get("shopper_text") or ""),
                )
                if all(target):
                    state["selected_product"] = product_id
                    if (
                        state["checked_at"].get(_erping_key(target), -1)
                        >= state["mutation"]
                    ):
                        continue
                    return [
                        _erping_synthetic(
                            "inspect_stock",
                            {"product_id": target[0], "sku": target[1]},
                            turn,
                        )
                    ]
                continue
        if action["name"] == "compare":
            product_ids = [
                str(item) for item in action["args"].get("product_ids") or []
            ]
            if product_ids and all(item in state["products"] for item in product_ids):
                continue
        if action["name"] == "search":
            query = " ".join(str(action["args"].get("query") or "").casefold().split())
            if not query or query in state["queries"] or query in new_queries:
                continue
            if len(state["queries"] | new_queries) >= 3:
                continue
            new_queries.add(query)
        if action["name"] == "inspect_stock":
            target = _erping_target(action["args"])
            if (
                all(target)
                and state["checked_at"].get(_erping_key(target), -1)
                >= state["mutation"]
            ):
                continue
        chosen.append(call)
    return chosen[:6]


def _erping_rescue(
    rejected: list[dict[str, Any]],
    state: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    turn: int,
) -> list[dict[str, Any]]:
    safe_reads = {
        "search",
        "filter",
        "view",
        "compare",
        "inspect_stock",
        "inspect_cart",
        "message",
    }
    for call in rejected:
        action = _erping_action(call)
        if action["name"] == "search":
            query = " ".join(str(action["args"].get("query") or "").casefold().split())
            if not query or query in state["queries"] or len(state["queries"]) >= 3:
                continue
        if action["name"] == "view":
            product_id = str(action["args"].get("product_id") or "")
            if product_id and product_id in state["products"]:
                continue
        if action["name"] == "compare":
            product_ids = [
                str(item) for item in action["args"].get("product_ids") or []
            ]
            if product_ids and all(item in state["products"] for item in product_ids):
                continue
        if action["name"] == "inspect_stock":
            target = _erping_target(action["args"])
            if (
                all(target)
                and state["checked_at"].get(_erping_key(target), -1)
                >= state["mutation"]
            ):
                continue
        if action["name"] in safe_reads:
            return [call]
    cart = _erping_cart_targets(state)
    if cart and "inspect_stock" in schemas:
        target = cart[-1]
        if state["checked_at"].get(_erping_key(target), -1) < state["mutation"]:
            return [
                _erping_synthetic(
                    "inspect_stock",
                    {"product_id": target[0], "sku": target[1]},
                    turn,
                )
            ]
    target = _erping_finish_target([], state, str(state.get("shopper_text") or ""))
    if all(target):
        if _erping_key(target) not in state["stock"] and "inspect_stock" in schemas:
            return [
                _erping_synthetic(
                    "inspect_stock",
                    {"product_id": target[0], "sku": target[1]},
                    turn,
                )
            ]
    if "view" in schemas:
        for product_id in reversed(list(state["summaries"])):
            if product_id not in state["products"]:
                return [_erping_synthetic("view", {"product_id": product_id}, turn)]
        if (
            all(target)
            and target not in _erping_cart_targets(state)
            and "add_to_cart" in schemas
        ):
            return [
                _erping_synthetic(
                    "add_to_cart",
                    {"product_id": target[0], "sku": target[1]},
                    turn,
                )
            ]
    if not state["queries"] and "search" in schemas:
        goal = str(state.get("shopper_text") or "").split("\n\nTask rules:", 1)[0]
        query = " ".join(goal.split())[:240]
        if query:
            return [
                _erping_synthetic(
                    "search",
                    {"query": query, "k": 10},
                    turn,
                )
            ]
    if "inspect_cart" in schemas:
        return [_erping_synthetic("inspect_cart", {}, turn)]
    return []


def _bmars_claim_schema(order_schema: dict[str, Any]) -> tuple[set[str], bool]:
    parameters = order_schema.get("parameters") or {}
    required = "justification" in (parameters.get("required") or [])
    justification = (parameters.get("properties") or {}).get("justification") or {}
    claims = (justification.get("properties") or {}).get("claims") or {}
    field = ((claims.get("items") or {}).get("properties") or {}).get("field") or {}
    enum = field.get("enum")
    allowed = {str(item) for item in enum} if isinstance(enum, list) else set()
    return allowed, required


def _bmars_property_field(
    shopper_text: str,
    draft_field: str,
    shopper_value: str,
    observed_value: str,
    field_description: str,
) -> str:
    schema = {
        "name": "name_requested_property",
        "description": "Name the abstract product property constrained by the shopper.",
        "parameters": {
            "type": "object",
            "properties": {
                "property_field": {
                    "type": "string",
                    "description": "Shortest standard snake_case name for the measurable property itself, not a device part, evidence carrier, unit, or requested value.",
                },
                "carrier_or_source": {
                    "type": "string",
                    "description": "The component, raw fact label, or evidence source that carries the observed value.",
                },
            },
            "required": ["property_field", "carrier_or_source"],
            "additionalProperties": False,
        },
    }
    try:
        response = _BMARS_DECISION_PROXY.post(
            "/inference/chat/completions",
            json_data={
                "model": _bmars_decision_model(),
                "messages": [
                    {
                        "role": "system",
                        "content": "Canonicalize one requested product property. Separate the abstract characteristic being constrained from the component, source label, unit, and value used as evidence. The property name must be a concise conventional e-commerce attribute in snake_case. Do not copy a value token as the property name. Return only the supplied tool.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "shopper_request": shopper_text,
                                "draft_field": draft_field,
                                "shopper_value": shopper_value,
                                "observed_value": observed_value,
                                "order_field_instruction": field_description,
                            },
                            separators=(",", ":"),
                            default=str,
                        ),
                    },
                ],
                "tools": [{"type": "function", "function": schema}],
                "tool_choice": "required",
                "temperature": 0,
                "max_tokens": 256,
            },
        )
    except Exception:
        return draft_field
    choices = response.get("choices") if isinstance(response, dict) else None
    message = (
        choices[0].get("message")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else None
    )
    if not isinstance(message, dict):
        return draft_field
    for call in message.get("tool_calls") or []:
        action = (
            _erping_action(call) if isinstance(call, dict) else {"name": "", "args": {}}
        )
        if action["name"] != "name_requested_property":
            continue
        field = str(action["args"].get("property_field") or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", field):
            return ""
        field_words = set(re.findall(r"[a-z0-9]+", field.casefold()))
        value_words = set(
            re.findall(
                r"[a-z0-9]+",
                f"{shopper_value} {observed_value}".casefold(),
            )
        )
        if field_words and field_words.issubset(value_words):
            return ""
        return field
    return draft_field


def _bmars_observed_values(
    row: dict[str, Any], stock: dict[str, Any]
) -> dict[str, str]:
    observed: dict[str, str] = {}
    facts = row.get("facts")
    if isinstance(facts, dict):
        for key, value in facts.items():
            if value not in (None, ""):
                observed[str(key)] = str(value)
    if row.get("brand") not in (None, ""):
        observed["brand"] = str(row["brand"])
    path = row.get("category_path")
    if isinstance(path, list) and path:
        observed["category"] = str(path[-1])
    if stock.get("price") is not None:
        observed["price"] = str(stock["price"])
    if stock.get("in_stock") is not None:
        observed["in_stock"] = "true" if bool(stock["in_stock"]) else "false"
    return observed


def _erping_claims(
    state: dict[str, Any],
    target: tuple[str, str],
    proposed: Any,
    order_schema: dict[str, Any],
    shopper_text: str = "",
) -> list[dict[str, str]]:
    row = state["products"].get(target[0])
    if not isinstance(row, dict):
        return []
    stock = state["stock"].get(_erping_key(target))
    if not isinstance(stock, dict):
        stock = {}
    allowed, required = _bmars_claim_schema(order_schema)
    observed = _bmars_observed_values(row, stock)
    observed_values: dict[str, list[str]] = {}
    for field, value in observed.items():
        observed_values.setdefault(value, []).append(field)
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    support = proposed.get("support") if isinstance(proposed, dict) else None
    support_field = str(support.get("field") or "") if isinstance(support, dict) else ""
    support_value = (
        str(support.get("observed_value") or "") if isinstance(support, dict) else ""
    )
    request_words = " ".join(re.findall(r"[a-z0-9]+", shopper_text.casefold()))
    rows = proposed.get("claims") if isinstance(proposed, dict) else []
    for claim in rows if isinstance(rows, list) else []:
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "")
        value = str(claim.get("value") or "")
        if not field or not value or field in seen:
            continue
        if allowed and field not in allowed:
            continue
        sources = observed_values.get(value, [])
        value_words = " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
        if (
            not sources
            and support_field == field
            and support_value in observed_values
            and value_words
            and value_words in request_words
        ):
            sources = observed_values[support_value]
        if not sources:
            continue
        normalized = set(re.findall(r"[a-z0-9]+", field.casefold()))
        related = any(
            normalized & set(re.findall(r"[a-z0-9]+", source.casefold()))
            for source in sources
        )
        if value.casefold() in {"true", "false"} and not related:
            continue
        claims.append({"field": field, "value": value})
        seen.add(field)
    for field, value in observed.items():
        if field in seen or field not in {"brand", "price", "in_stock"}:
            continue
        if allowed and field not in allowed:
            continue
        claims.append({"field": field, "value": value})
        seen.add(field)
    if required and not claims:
        return []
    return claims


def _bmars_refine_justification(
    state: dict[str, Any],
    target: tuple[str, str],
    shopper_text: str,
    order_schema: dict[str, Any],
    proposed: Any,
) -> Any:
    row = state["products"].get(target[0])
    stock = state["stock"].get(_erping_key(target))
    if not isinstance(row, dict) or not isinstance(stock, dict):
        return proposed
    facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
    evidence_values = list(
        dict.fromkeys(str(value) for value in facts.values() if value not in (None, ""))
    )
    if not evidence_values:
        return proposed
    submitted_claims = proposed.get("claims") if isinstance(proposed, dict) else None
    routine_fields = {"brand", "category", "price", "in_stock"}
    allowed_fields, _ = _bmars_claim_schema(order_schema)
    technical_fields = sorted(allowed_fields - routine_fields)
    parameters = order_schema.get("parameters") or {}
    justification_schema = (parameters.get("properties") or {}).get(
        "justification"
    ) or {}
    claim_schema = (justification_schema.get("properties") or {}).get("claims") or {}
    field_schema = ((claim_schema.get("items") or {}).get("properties") or {}).get(
        "field"
    ) or {}
    field_description = str(field_schema.get("description") or "")
    handled_fields = sorted(
        {
            str(claim.get("field") or "").strip().casefold()
            for claim in (submitted_claims or [])
            if isinstance(claim, dict)
            and str(claim.get("field") or "").strip().casefold() in routine_fields
        }
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Select the single requested technical specification supported by the observed product values. Routine identity, cost, and availability claims are handled elsewhere and must not be repeated. Name the requested physical or technical property using the shortest widely standardized e-commerce attribute in snake_case. For the value, prefer the shopper's exact wording when an observed value semantically establishes the same specification; otherwise use an exact observed value. Do not name a product component, interface, data container, category, brand, price, or stock state. Use a compound name only when necessary to distinguish the property. Return only the supplied proposal tool.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "shopper_request": shopper_text,
                    "product_title": row.get("title"),
                    "observed_specification_values": evidence_values,
                    "already_handled_claim_fields": handled_fields,
                    "permitted_claim_fields": technical_fields,
                    "order_requirements": (
                        (order_schema.get("parameters") or {}).get("properties") or {}
                    ).get("justification"),
                },
                separators=(",", ":"),
                default=str,
            ),
        },
    ]
    proposal_schema = {
        "name": "propose_grounded_specification",
        "description": "Propose one semantically named specification claim from observed evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "Shortest standardized snake_case name of the physical or technical property measured by the requested value.",
                    **({"enum": technical_fields} if technical_fields else {}),
                },
                "value": {
                    "type": "string",
                    "description": "Exact shopper wording for a requested specification supported by an observed value, or an exact observed value.",
                },
                "observed_value": {
                    "type": "string",
                    "description": "Exact value copied from observed_specification_values that semantically supports the proposed shopper-facing value.",
                },
            },
            "required": ["field", "value", "observed_value"],
            "additionalProperties": False,
        },
    }
    for attempt in range(2):
        try:
            response = _BMARS_DECISION_PROXY.post(
                "/inference/chat/completions",
                json_data={
                    "model": _bmars_decision_model(),
                    "messages": messages,
                    "tools": [{"type": "function", "function": proposal_schema}],
                    "tool_choice": "required",
                    "temperature": 0,
                    "max_tokens": 256,
                },
            )
        except Exception:
            break
        choices = response.get("choices") if isinstance(response, dict) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        candidate = None
        if isinstance(message, dict):
            for call in message.get("tool_calls") or []:
                action = (
                    _erping_action(call)
                    if isinstance(call, dict)
                    else {"name": "", "args": {}}
                )
                if action["name"] == "propose_grounded_specification":
                    candidate = action["args"]
                    break
        field = str((candidate or {}).get("field") or "").strip()
        value = str((candidate or {}).get("value") or "").strip()
        observed_value = str((candidate or {}).get("observed_value") or "").strip()
        if not observed_value and value in evidence_values:
            observed_value = value
        if not observed_value and len(evidence_values) == 1:
            observed_value = evidence_values[0]
        value_words = re.findall(r"[a-z0-9]+", value.casefold())
        request_words = re.findall(r"[a-z0-9]+", shopper_text.casefold())
        request_grounded = bool(value_words) and " ".join(value_words) in " ".join(
            request_words
        )
        field_key = field.casefold()
        if not allowed_fields and field_description and field:
            field = _bmars_property_field(
                shopper_text,
                field,
                value,
                observed_value,
                field_description,
            )
            field_key = field.casefold()
        canonical = not allowed_fields or field in allowed_fields
        if (
            field
            and field_key not in routine_fields
            and canonical
            and observed_value in evidence_values
            and request_grounded
        ):
            result: dict[str, Any] = {"claims": [{"field": field, "value": value}]}
            if value not in evidence_values:
                result["support"] = {
                    "field": field,
                    "observed_value": observed_value,
                }
            return result
        if attempt == 0:
            messages.append(
                {
                    "role": "system",
                    "content": "The proposal was not a permitted, grounded answer to the requested technical property. Choose field only from the permitted claim fields, copy observed_value exactly from the supplied observed values, and use the shopper-facing value only when that observation supports it.",
                }
            )
    return proposed


def _erping_prepare(
    call: dict[str, Any],
    state: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    shopper_text: str,
) -> dict[str, Any]:
    action = _erping_action(call)
    if action["name"] != "place_test_order":
        return call
    target = _erping_order_target(action, state)
    args: dict[str, Any] = {}
    if target[0]:
        args["product_id"] = target[0]
    if target[1]:
        args["sku"] = target[1]
    properties = (
        schemas.get("place_test_order", {}).get("parameters", {}).get("properties", {})
    )
    proposed = action["args"].get("justification")
    order_schema = schemas.get("place_test_order", {})
    _, justification_required = _bmars_claim_schema(order_schema)
    if justification_required:
        proposed = _bmars_refine_justification(
            state, target, shopper_text, order_schema, proposed
        )
    claims = _erping_claims(state, target, proposed, order_schema, shopper_text)
    if "justification" in properties and claims:
        args["justification"] = {"claims": claims}
    return {
        "id": call["id"],
        "type": "function",
        "function": {
            "name": "place_test_order",
            "arguments": json.dumps(args, separators=(",", ":"), default=str),
        },
    }


def _erping_remember(
    state: dict[str, Any], action: dict[str, Any], step: dict[str, Any]
) -> None:
    name = action["name"]
    args = action["args"]
    actual = _erping_actual(step)
    error = step.get("error") or actual.get("error")
    if name in {"search", "filter"}:
        if name == "search" and not error:
            query = " ".join(str(args.get("query") or "").casefold().split())
            if query:
                state["queries"].add(query)
        for row in actual.get("results") or []:
            if isinstance(row, dict) and row.get("product_id") is not None:
                state["summaries"][str(row["product_id"])] = row
    if name == "view" and actual.get("product_id") is not None:
        state["products"][str(actual["product_id"])] = actual
    if name == "compare":
        for row in actual.get("compared") or []:
            if isinstance(row, dict) and row.get("product_id") is not None:
                state["products"][str(row["product_id"])] = row
    if name in {"add_to_cart", "remove_from_cart"} and not error:
        state["mutation"] += 1
        if isinstance(actual.get("cart"), list):
            state["cart"] = actual["cart"]
        if name == "remove_from_cart":
            target = _erping_target(args)
            if target[0]:
                state["excluded"].add(target[0])
    if name == "inspect_cart" and isinstance(actual.get("cart"), list):
        state["cart"] = actual["cart"]
    if name == "inspect_stock" and not error:
        target = _erping_target(args)
        key = _erping_key(target)
        state["stock"][key] = actual
        state["checked_at"][key] = state["mutation"]
    if name == "compare":
        state["shortlist"] = [
            str(item) for item in args.get("product_ids") or [] if item is not None
        ][-5:]
        named_products = {item[0] for item in state["named_targets"]}
        compared_products = set(state["shortlist"])
        state["decision_pending"] = (
            len(named_products) >= 3 and len(named_products & compared_products) >= 3
        )
    if name == "message" and not error:
        state["messages_sent"] += 1
    event = actual.get("market_event") or actual.get("event")
    if event:
        state["events"].append(event)
        state["events"] = state["events"][-8:]
        state["queries"].clear()
        state["selected_product"] = ""
        state["choice_locked"] = False
        named_products = {item[0] for item in state["named_targets"]}
        if len(named_products & set(state["shortlist"])) >= 3:
            state["decision_pending"] = True
    for name in ("summaries", "products", "stock", "checked_at"):
        if len(state[name]) > 32:
            for key in list(state[name])[: len(state[name]) - 32]:
                del state[name][key]


def _erping_memory(state: dict[str, Any]) -> dict[str, Any]:
    products: list[dict[str, Any]] = []
    for product_id, row in state["products"].items():
        if not isinstance(row, dict):
            continue
        products.append(
            {
                "product_id": product_id,
                "listing_id": row.get("listing_id"),
                "title": row.get("title"),
                "brand": row.get("brand"),
                "category_path": row.get("category_path"),
                "facts": row.get("facts"),
                "variants": row.get("variants"),
            }
        )
    return {
        "viewed_products": products[-18:],
        "latest_cart": state.get("cart"),
        "latest_stock": dict(list(state["stock"].items())[-10:]),
        "market_events": state["events"][-6:],
        "shopper_updates": state["updates"][-6:],
        "mutation_epoch": state["mutation"],
    }


def _erping_checkpoint(state: dict[str, Any], turn: int, maximum: int) -> str:
    cart = _erping_cart_targets(state)
    left = max(0, maximum - turn + 1)
    urgency = (
        " FINALIZATION WINDOW: stop expanding the shortlist, commit the best fully observed hard-valid survivor now, and reserve actions for add_to_cart, a fresh inspect_stock, and place_test_order."
        if left <= 5
        else ""
    )
    return (
        "Execution checkpoint: "
        + json.dumps(
            {
                "turn": turn,
                "turns_left": left,
                "distinct_search_results": len(state["summaries"]),
                "distinct_search_queries": len(state["queries"]),
                "viewed_products": len(state["products"]),
                "cart_targets": [list(item) for item in cart],
                "messages_sent": state["messages_sent"],
                "events_seen": len(state["events"]),
                "latest_update": state["updates"][-1] if state["updates"] else None,
            },
            separators=(",", ":"),
            default=str,
        )
        + ". Return the next supplied tool call immediately without narration or extended deliberation. Do not repeat an unchanged read."
        + urgency
    )


def _erping_context(
    base: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    state: dict[str, Any],
    turn: int,
    maximum: int,
) -> list[dict[str, Any]]:
    checkpoint = {"role": "system", "content": _erping_checkpoint(state, turn, maximum)}
    full = base + [item for group in groups for item in group] + [checkpoint]
    if len(json.dumps(full, default=str)) <= 28000:
        return full
    memory = {
        "role": "system",
        "content": "Verified earlier observations:\n"
        + json.dumps(_erping_memory(state), separators=(",", ":"), default=str),
    }
    return (
        base
        + [memory]
        + [item for group in groups[-4:] for item in group]
        + [checkpoint]
    )


def _erping_infer(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    deliberate: bool = False,
) -> dict[str, Any] | None:
    proxy = _BMARS_DECISION_PROXY if deliberate else _ERPING_PROXY
    model = _bmars_decision_model() if deliberate else _erping_models()[0]
    if deliberate:
        messages = list(messages) + [
            {
                "role": "system",
                "content": "A focused comparison is now available. Select from observed candidates only. Apply the shopper's criteria in their stated order. Candidate-specific variant options override broad listing-level text when they conflict. Return the next tool action without prose.",
            }
        ]
    response = proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 640 if deliberate else 512,
        },
    )
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if isinstance(message, dict) and message.get("tool_calls"):
        return message
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = dict(environment["binding"])
    policy = environment["policy_view"]
    tools = list(policy["tools"])
    schemas = _erping_schemas(tools)
    maximum_calls = max(1, min(int(policy.get("max_calls_per_turn", 1)), 16))
    maximum_turns = _bmars_turn_limit(policy)
    problem_id = str(problem_data.get("problem_id") or problem_data.get("id") or "task")
    base = [
        {"role": "system", "content": _ERPING_POLICY},
        {"role": "user", "content": str(policy["query"])},
    ]
    initial_goal = str(policy["query"]).split("\n\nTask rules:", 1)[0]
    state: dict[str, Any] = {
        "summaries": {},
        "queries": set(),
        "products": {},
        "stock": {},
        "checked_at": {},
        "cart": [],
        "events": [],
        "updates": [],
        "shortlist": [],
        "selected_product": "",
        "named_targets": _erping_named_targets(initial_goal),
        "choice_locked": False,
        "terminal_fallback": False,
        "excluded": set(),
        "shopper_text": initial_goal,
        "mutation": 0,
        "messages_sent": 0,
        "decision_pending": False,
    }
    groups: list[list[dict[str, Any]]] = []
    dialogue: list[dict[str, Any]] = []
    for turn in range(1, maximum_turns + 1):
        shopper_text = initial_goal + "\n" + "\n".join(state["updates"])
        state["shopper_text"] = shopper_text
        messages = _erping_context(base, groups, state, turn, maximum_turns)
        assistant = None
        calls: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for attempt in range(1):
            request_messages = list(messages)
            if attempt:
                request_messages.append(
                    {
                        "role": "user",
                        "content": "The environment is not done. Return one valid supplied tool call now.",
                    }
                )
            deliberate = bool(state["decision_pending"])
            candidate = _erping_infer(request_messages, tools, deliberate=deliberate)
            if candidate is None:
                continue
            proposed = _erping_calls(candidate, schemas, maximum_calls, turn)
            selected = _erping_guard(proposed, state, turn)
            rejected = proposed
            if selected:
                if deliberate:
                    selected_action = _erping_action(selected[0])
                    selected_target = _erping_target(selected_action["args"])
                    named_products = {item[0] for item in state["named_targets"]}
                    if selected_target[0] in named_products:
                        state["selected_product"] = selected_target[0]
                        state["choice_locked"] = True
                    state["decision_pending"] = False
                assistant = candidate
                calls = selected
                break
        if assistant is None or not calls:
            calls = _erping_rescue(rejected, state, schemas, turn)
            if not calls:
                raise RuntimeError("environment exposes no recoverable action")
            assistant = {"content": "", "tool_calls": calls}
        calls = _erping_endgame(
            calls,
            state,
            schemas,
            shopper_text,
            turn,
            maximum_turns,
        )
        calls = [_erping_prepare(call, state, schemas, shopper_text) for call in calls]
        content = str(assistant.get("content") or "")
        assistant_message = {
            "role": "assistant",
            "content": content,
            "tool_calls": calls,
        }
        group_id = f"{problem_id}-erping-{turn}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {
                    "call_id": f"{group_id}-{index}",
                    "action": _erping_action(call),
                }
                for index, call in enumerate(calls, start=1)
            ],
        }
        result = _ERPING_PROXY.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict):
            raise RuntimeError("environment call failed")
        results = result.get("calls")
        if not isinstance(results, list) or len(results) != len(calls):
            raise RuntimeError("environment returned an incomplete action group")
        group = [assistant_message]
        for call, outcome in zip(calls, results, strict=True):
            step = outcome.get("observation") if isinstance(outcome, dict) else {}
            if not isinstance(step, dict):
                step = {}
            _erping_remember(state, _erping_action(call), step)
            group.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(step, separators=(",", ":"), default=str),
                }
            )
        user_message = result.get("user_message")
        if isinstance(user_message, dict) and user_message.get("content"):
            update = str(user_message["content"])
            state["updates"].append(update)
            state["queries"].clear()
            group.append({"role": "user", "content": update})
        groups.append(group)
        dialogue.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": calls,
                "environment_result": result,
            }
        )
        if any(
            isinstance(outcome, dict)
            and isinstance(outcome.get("observation"), dict)
            and outcome["observation"].get("done") is True
            for outcome in results
        ):
            break
    return dialogue


__all__ = ["agent_main"]
