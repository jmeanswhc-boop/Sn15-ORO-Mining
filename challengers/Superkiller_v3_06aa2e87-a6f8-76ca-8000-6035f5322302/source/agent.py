from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_ERPING_MODELS = {
    "chutes": (
        "Qwen/Qwen3.5-397B-A17B-TEE",
        "deepseek-ai/DeepSeek-V3.2-TEE",
    ),
    "openrouter": (
        "qwen/qwen3.5-397b-a17b",
        "deepseek/deepseek-v3.2",
    ),
}
_ERPING_PROXY = ProxyClient(
    timeout=45,
    max_retries=1,
    retry_delay=0.5,
    rate_limit_retry_delay=2,
)
_ERPING_POLICY = """You are a rigorous shopping agent in a live store. Complete the public shopper goal with only the supplied tools. Continue until an observation says done=true.

Separate hard requirements from preferences before acting. Category, requested specification, budget, and current stock are hard unless the shopper explicitly revises them. A preferred brand, color, feature, or use-case priority is soft unless the goal says otherwise. Apply every later shopper update and answer every direct question with the message tool before ordering.

Prefer no more than three distinct, precise search formulations, but continue with a genuinely new formulation when the observed catalog has not exposed a valid candidate. Search uses title words, while filter uses structured category, brand, price, and stock. Search with k=10 when surveying. Reformulate rather than repeating. Surface genuinely distinct candidates, then use view or compare on a focused shortlist. Reject accessories, wrong catalog categories, review pages, incompatible variants, and title-only guesses. Base the choice on observed option values for the chosen variant, its latest cost, and its availability.

Inspect the intended listing before modifying the cart for the first time. When the shopper requests preference clarification, ask one concise question first and evaluate two or more separate catalog listings. After the shopper reveals preference priorities, optimize every revealed preference axis among the hard-valid survivors and select the survivor with the least avoidable preference regret. When it gives a lexicographic ranking rule, calculate it exactly from observed variant facts and commit first to the best live candidate. When it asks for broad retrieval, establish the required identity coverage from search results before committing. Begin checkout while at least five turns remain. Once five turns remain, stop expanding the shortlist and use the best fully observed hard-valid survivor.

Cart additions are real commitments, never probes. After every mutation, observe the exact selected SKU again. A market event may be silent in the triggering response, so treat a changed price or stock as authoritative, remove an invalid item, and choose the least-relaxed valid replacement. Preserve category and budget; relax preferences only in their stated order. Build the relaxation ladder only from candidates whose observed category_path, latest observed cost, and availability satisfy every hard requirement. A matching model, color, or title never makes a wrong catalog category acceptable. Re-rank after every changed fact. Keep only the intended final item in the cart.

At order time, use the exact product_id and sku already in the cart. Include justification only when the tool schema or shopper requires it. Use the schema's canonical claim names: screen_size for an observed size fact, resolution for an observed display fact, refresh_rate for an observed refresh fact, plus current price and in_stock. Claims must contain only permitted field/value pairs and must exactly match viewed facts or the latest stock inspection; never invent a claim or add prose fields. Batch independent read-only calls when their arguments are already known. Do not stop, summarize, or ask for confirmation when a verified valid order can be placed."""


def _erping_provider() -> str:
    value = str(getenv("INFERENCE_PROVIDER", "chutes")).strip().lower()
    return "openrouter" if value == "openrouter" else "chutes"


def _erping_models() -> tuple[str, ...]:
    override = str(getenv("SANDBOX_MODEL", "")).strip()
    defaults = _ERPING_MODELS[_erping_provider()]
    if not override:
        return defaults
    return (override,) + tuple(item for item in defaults if item != override)


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
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
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


def _erping_synthetic(
    name: str, args: dict[str, Any], turn: int
) -> dict[str, Any]:
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
    aliases = (
        (("smartphone", "smart phone"), ("smartphone",)),
        (("headphone", "earphone", "earbud"), ("headphone", "earbud", "earphone")),
        (("monitor",), ("monitor",)),
        (("tablet",), ("tablet",)),
    )
    lowered = text.casefold()
    for request_words, path_words in aliases:
        if any(word in lowered for word in request_words):
            return any(word in visible for word in path_words)
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
            if stock.get("in_stock") is False:
                continue
            price = stock.get("price")
            if budget is not None and isinstance(price, (int, float)) and price > budget:
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
    for call in calls:
        action = _erping_action(call)
        if action["name"] == "compare":
            preferred.extend(str(item) for item in action["args"].get("product_ids") or [])
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


def _erping_endgame(
    calls: list[dict[str, Any]],
    state: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
    shopper_text: str,
    turn: int,
    maximum: int,
) -> list[dict[str, Any]]:
    left = maximum - turn + 1
    if left > 5 or "place_test_order" not in schemas:
        return calls
    first = _erping_action(calls[0]) if calls else {"name": "", "args": {}}
    if first["name"] == "message":
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
                budget is not None
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
    return [
        _erping_synthetic(
            "add_to_cart",
            {"product_id": target[0], "sku": target[1]},
            turn,
        )
    ]


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
            if target[0] not in state["products"]:
                return [_erping_synthetic("view", {"product_id": target[0]}, turn)]
            if not _erping_category_ok(
                state["products"][target[0]], str(state.get("shopper_text") or "")
            ):
                state["excluded"].add(target[0])
                return []
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
        if action["name"] == "search":
            query = " ".join(str(action["args"].get("query") or "").casefold().split())
            if not query or query in state["queries"] or query in new_queries:
                continue
            if len(state["queries"] | new_queries) >= 3:
                continue
            new_queries.add(query)
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
        if _erping_action(call)["name"] in safe_reads:
            return [call]
    cart = _erping_cart_targets(state)
    if cart and "inspect_stock" in schemas:
        target = cart[-1]
        return [
            _erping_synthetic(
                "inspect_stock",
                {"product_id": target[0], "sku": target[1]},
                turn,
            )
        ]
    if "view" in schemas:
        for product_id in state["summaries"]:
            if product_id not in state["products"]:
                return [
                    _erping_synthetic("view", {"product_id": product_id}, turn)
                ]
    if "inspect_cart" in schemas:
        return [_erping_synthetic("inspect_cart", {}, turn)]
    return []


def _erping_fact(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    facts = row.get("facts")
    if not isinstance(facts, dict):
        facts = {}
    for name in names:
        value = facts.get(name)
        if value not in (None, ""):
            return value
    return None


def _erping_claims(
    state: dict[str, Any], target: tuple[str, str], shopper_text: str
) -> list[dict[str, str]]:
    row = state["products"].get(target[0])
    if not isinstance(row, dict):
        return []
    text = shopper_text.casefold()
    claims: list[dict[str, str]] = []
    size = _erping_fact(row, ("screen_size", "size"))
    resolution = _erping_fact(row, ("resolution", "display"))
    refresh = _erping_fact(row, ("refresh_rate", "refresh"))
    if size not in (None, "") and any(
        marker in text for marker in ("screen", "inch", "-in", '"')
    ):
        claims.append({"field": "screen_size", "value": str(size)})
    if resolution not in (None, "") and any(
        marker in text
        for marker in ("resolution", "720p", "1080p", "1440p", "fhd", "uhd", "4k")
    ):
        claims.append({"field": "resolution", "value": str(resolution)})
    if refresh not in (None, "") and any(
        marker in text for marker in ("refresh", " hz", "hz ")
    ):
        claims.append({"field": "refresh_rate", "value": str(refresh)})
    if row.get("brand") not in (None, ""):
        claims.append({"field": "brand", "value": str(row["brand"])})
    stock = state["stock"].get(_erping_key(target))
    if isinstance(stock, dict):
        if stock.get("price") is not None:
            claims.append({"field": "price", "value": str(stock["price"])})
        if stock.get("in_stock") is not None:
            claims.append(
                {
                    "field": "in_stock",
                    "value": "true" if bool(stock["in_stock"]) else "false",
                }
            )
    return claims


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
    claims = _erping_claims(state, target, shopper_text)
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
    if name == "message" and not error:
        state["messages_sent"] += 1
    event = actual.get("market_event") or actual.get("event")
    if event:
        state["events"].append(event)
        state["events"] = state["events"][-8:]
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
    if len(json.dumps(full, default=str)) <= 76000:
        return full
    memory = {
        "role": "system",
        "content": "Verified earlier observations:\n"
        + json.dumps(_erping_memory(state), separators=(",", ":"), default=str),
    }
    return base + [memory] + [item for group in groups[-7:] for item in group] + [checkpoint]


def _erping_infer(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for model in _erping_models():
        response = _ERPING_PROXY.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
                "max_tokens": 4096,
            },
        )
        if not isinstance(response, dict):
            continue
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
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
    maximum_turns = max(1, int(policy["max_steps"]))
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
        "excluded": set(),
        "shopper_text": initial_goal,
        "mutation": 0,
        "messages_sent": 0,
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
        for attempt in range(2):
            request_messages = list(messages)
            if attempt:
                request_messages.append(
                    {
                        "role": "user",
                        "content": "The environment is not done. Return one valid supplied tool call now.",
                    }
                )
            candidate = _erping_infer(request_messages, tools)
            if candidate is None:
                continue
            proposed = _erping_calls(candidate, schemas, maximum_calls, turn)
            selected = _erping_guard(proposed, state, turn)
            rejected = proposed
            if selected:
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
        calls = [
            _erping_prepare(call, state, schemas, shopper_text) for call in calls
        ]
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
