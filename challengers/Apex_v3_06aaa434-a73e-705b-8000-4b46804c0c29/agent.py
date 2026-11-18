from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient



orbit_bag = ProxyClient(timeout=120, max_retries=2)

seedling_bag = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}

def unpack_glade() -> str:
    try:
        foam_node = getenv("INFERENCE_PROVIDER", "chutes")
        return getenv("SANDBOX_MODEL") or seedling_bag.get(
            foam_node, seedling_bag["chutes"]
        )
    except Exception:
        return seedling_bag["chutes"]


def unpack_bog(dewdrop_pool: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        for cascade_pool in dewdrop_pool:
            horizon_cache = cascade_pool.get("function") if isinstance(cascade_pool, dict) else None
            if isinstance(horizon_cache, dict) and horizon_cache.get("name") == "place_test_order":
                grove_pool = horizon_cache.get("parameters") if isinstance(horizon_cache.get("parameters"), dict) else {}
                return grove_pool
        return None
    except Exception:
        return None


def spin_blossom(dewdrop_pool: list[dict[str, Any]], cove_bag: str) -> list[str]:
    try:
        for cascade_pool in dewdrop_pool:
            horizon_cache = cascade_pool.get("function") if isinstance(cascade_pool, dict) else None
            if not isinstance(horizon_cache, dict) or horizon_cache.get("name") != cove_bag:
                continue
            grove_pool = horizon_cache.get("parameters") if isinstance(horizon_cache.get("parameters"), dict) else {}
            return [slate_stack_87_bag for slate_stack_87_bag in grove_pool.get("required", []) if isinstance(slate_stack_87_bag, str)]
        return []
    except Exception:
        return []


def clamp_brook(dewdrop_pool: list[dict[str, Any]]) -> set[str]:
    """Read allowed justification field names from the live tool schema only."""
    try:
        grove_pool = unpack_bog(dewdrop_pool) or {}
        nebula_slot = grove_pool.get("properties") if isinstance(grove_pool.get("properties"), dict) else {}
        lagoon_bag = nebula_slot.get("justification") if isinstance(nebula_slot.get("justification"), dict) else {}
        cedar_cache = lagoon_bag.get("properties") if isinstance(lagoon_bag.get("properties"), dict) else {}
        clover_node = cedar_cache.get("claims") if isinstance(cedar_cache.get("claims"), dict) else {}
        thaw_node = clover_node.get("items") if isinstance(clover_node.get("items"), dict) else {}
        maple_cache = thaw_node.get("properties") if isinstance(thaw_node.get("properties"), dict) else {}
        summit_slot = maple_cache.get("field") if isinstance(maple_cache.get("field"), dict) else {}
        comet_ref: set[str] = set()
        dusk_ref = summit_slot.get("enum")
        if isinstance(dusk_ref, list):
            comet_ref.update(str(slate_stack_87_bag) for slate_stack_87_bag in dusk_ref if isinstance(slate_stack_87_bag, str))
        # Also accept snake_case tokens named by the schema description itself.
        for rivulet_slot in (summit_slot.get("description"), lagoon_bag.get("description"), clover_node.get("description")):
            if isinstance(rivulet_slot, str) and rivulet_slot.strip():
                comet_ref.update(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", rivulet_slot))
                comet_ref.update(
                    pearl_cache
                    for pearl_cache in re.findall(r"\b[a-z][a-z0-9]{2,}\b", rivulet_slot)
                    if pearl_cache
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
        comet_ref.update({"price", "in_stock"})
        return comet_ref
    except Exception:
        return {"price", "in_stock"}


def loose_juniper(dewdrop_pool: list[dict[str, Any]], ivy_bag: int, quartz_slot: int) -> str:
    try:
        reed_ref = "justification" in spin_blossom(dewdrop_pool, "place_test_order")
        comet_ref = sorted(clamp_brook(dewdrop_pool)) if reed_ref else []

        elm_pool = [
            "You are a shopping assistant in a product catalog.",
            "Help the shopper find a valid in-stock item, confirm it, and complete the purchase with the tools.",
            "Follow the shopper request and every attached instruction. Never invent product_id or sku values.",
            "",
            "Method:",
            "- Search with short keywords (2-5 words). Reformulate or use filter when results are poor.",
            "- Prefer in_stock=true when available.",
            f"- Up to {ivy_bag} tool calls per turn; batch independent reads when useful.",
            "- Inspect with view/compare before the first add_to_cart. Cart only a real intended choice.",
            "- After cart changes, call inspect_stock. If invalid, remove_from_cart, recover, then order.",
            "- Order path: view/compare -> add_to_cart -> inspect_stock -> place_test_order.",
            "- Answer shopper questions with message before ordering.",
            f"- Budget about {quartz_slot} turns. When verified, place_test_order immediately.",
            "- Do not spend many turns only messaging or repeatedly removing from cart without ordering.",
        ]
        if reed_ref:
            elm_pool.append(
                "- place_test_order requires justification claims. Use only field names supported by "
                f"the tool schema ({', '.join(comet_ref)}). Values must match inspected facts for the "
                "final product. Always include truthful price and in_stock from inspect_stock. "
                "Never invent unsupported features."
            )
        else:
            elm_pool.append(
                "- Omit justification on place_test_order unless the tool schema requires it."
            )

        elm_pool.append("")
        elm_pool.append("Tools:")
        for cascade_pool in dewdrop_pool:
            horizon_cache = cascade_pool.get("function") if isinstance(cascade_pool, dict) else None
            if isinstance(horizon_cache, dict) and isinstance(horizon_cache.get("name"), str):
                elm_pool.append(f"- {horizon_cache['name']}: {str(horizon_cache.get('description') or '').strip()}")
        elm_pool.append("")
        elm_pool.append("Continue until an observation reports done=true.")
        return "\n".join(elm_pool)
    except Exception:
        return "You are a shopping assistant. Use tools to find and order a valid in-stock product."


def scan_spore(spray_bag: dict[str, Any]) -> dict[str, Any]:
    try:
        acorn_ref = spray_bag["function"].get("arguments", "{}")
        if isinstance(acorn_ref, dict):
            drizzle_ref: Any = acorn_ref
        elif isinstance(acorn_ref, str):
            try:
                drizzle_ref = json.loads(acorn_ref) if acorn_ref.strip() else {}
            except json.JSONDecodeError:
                try:
                    hollow_slot = re.sub(r",\s*}", "}", acorn_ref.replace("'", '"'))
                    hollow_slot = re.sub(r",\s*]", "]", hollow_slot)
                    drizzle_ref = json.loads(hollow_slot)
                except Exception:
                    drizzle_ref = {}
        else:
            drizzle_ref = {}
        if not isinstance(drizzle_ref, dict):
            return {}
        return drizzle_ref
    except Exception:
        return {}


def cast_eddy(
    thicket_bag: dict[str, Any],
    *,
    dewdrop_pool: list[dict[str, Any]],
    canopy_node: dict[str, dict[str, Any]],
    crimson_gadget_87_ref: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        ember_bag = dict(thicket_bag)
        vine_pool = spin_blossom(dewdrop_pool, "place_test_order")
        if "justification" in ember_bag and "justification" not in vine_pool:
            ember_bag.pop("justification", None)
            return ember_bag
        if "justification" not in vine_pool:
            return ember_bag

        squall_bag = str(ember_bag.get("product_id") or "")
        rootlet_cache = str(ember_bag.get("sku") or "")
        ripple_bag = f"{squall_bag}::{rootlet_cache}"
        ridge_pool = canopy_node.get(ripple_bag) or canopy_node.get(squall_bag) or {}
        meadow_ref = crimson_gadget_87_ref.get(squall_bag) or {}
        comet_ref = clamp_brook(dewdrop_pool)

        cinder_slot = []
        lagoon_bag = ember_bag.get("justification")
        if isinstance(lagoon_bag, dict) and isinstance(lagoon_bag.get("claims"), list):
            cinder_slot = [inlet_ref for inlet_ref in lagoon_bag["claims"] if isinstance(inlet_ref, dict)]

        sapling_buf: list[dict[str, str]] = []
        pebble_bag: set[str] = set()

        def bind_gust(summit_slot: str) -> Any:
            try:
                for rivulet_slot in (ridge_pool, meadow_ref):
                    if summit_slot in rivulet_slot and rivulet_slot.get(summit_slot) not in (None, ""):
                        return rivulet_slot.get(summit_slot)
                    twilight_cache = rivulet_slot.get("specification") if isinstance(rivulet_slot.get("specification"), dict) else {}
                    valley_slot = rivulet_slot.get("attributes") if isinstance(rivulet_slot.get("attributes"), dict) else {}
                    for flint_bag in (twilight_cache, valley_slot):
                        if summit_slot in flint_bag and flint_bag.get(summit_slot) not in (None, ""):
                            return flint_bag.get(summit_slot)
                return None
            except Exception:
                return None

        def trace_lotus(summit_slot: str, scarlet_rack_87_cache: Any) -> None:
            try:
                if summit_slot not in comet_ref or summit_slot in pebble_bag:
                    return
                if scarlet_rack_87_cache is None or scarlet_rack_87_cache == "":
                    return
                if isinstance(scarlet_rack_87_cache, bool):
                    petal_slot = "true" if scarlet_rack_87_cache else "false"
                else:
                    petal_slot = str(scarlet_rack_87_cache).strip()
                if not petal_slot:
                    return
                sapling_buf.append({"field": summit_slot, "value": petal_slot})
                pebble_bag.add(summit_slot)
            except Exception:
                return None

        # Truthful live stock/price first.
        if "price" in ridge_pool:
            trace_lotus("price", ridge_pool.get("price"))
        if "in_stock" in ridge_pool:
            trace_lotus("in_stock", ridge_pool.get("in_stock"))

        # Keep model claims only when field is schema-allowed; prefer observed values.
        for marsh_ref in cinder_slot:
            summit_slot = str(marsh_ref.get("field") or "").strip()
            if summit_slot in {"price", "in_stock"}:
                continue
            if summit_slot not in comet_ref:
                continue
            tide_pool = bind_gust(summit_slot)
            if tide_pool is not None:
                trace_lotus(summit_slot, tide_pool)
                continue
            # Drop unsupported/invented values when we have no observation for the field.
            scarlet_rack_87_cache = marsh_ref.get("value")
            if isinstance(scarlet_rack_87_cache, bool) or str(scarlet_rack_87_cache).lower() in {"true", "false"}:
                # Boolean feature claims need observed evidence.
                continue
            trace_lotus(summit_slot, scarlet_rack_87_cache)

        # Fill remaining observed facts that the schema allows.
        for summit_slot in sorted(comet_ref):
            if summit_slot in pebble_bag:
                continue
            tide_pool = bind_gust(summit_slot)
            if tide_pool is not None:
                trace_lotus(summit_slot, tide_pool)

        if not sapling_buf:
            if "price" in ridge_pool:
                trace_lotus("price", ridge_pool.get("price"))
            if "in_stock" in ridge_pool:
                trace_lotus("in_stock", ridge_pool.get("in_stock"))

        ember_bag["justification"] = {"claims": sapling_buf}
        return ember_bag
    except Exception:
        try:
            return dict(thicket_bag)
        except Exception:
            return {}


def draw_heather(
    cove_bag: str,
    thicket_bag: dict[str, Any],
    dewdrop_pool: list[dict[str, Any]],
    canopy_node: dict[str, dict[str, Any]],
    crimson_gadget_87_ref: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        ember_bag = dict(thicket_bag)
        if cove_bag == "search" and "in_stock" not in ember_bag:
            ember_bag["in_stock"] = True
        if cove_bag == "place_test_order":
            ember_bag = cast_eddy(
                ember_bag,
                dewdrop_pool=dewdrop_pool,
                canopy_node=canopy_node,
                crimson_gadget_87_ref=crimson_gadget_87_ref,
            )
        return ember_bag
    except Exception:
        try:
            return dict(thicket_bag)
        except Exception:
            return {}


def trim_solstice(
    cove_bag: str,
    thicket_bag: dict[str, Any],
    amber_buf: dict[str, Any],
    canopy_node: dict[str, dict[str, Any]],
    crimson_gadget_87_ref: dict[str, dict[str, Any]],
) -> None:
    try:
        if cove_bag == "inspect_stock":
            squall_bag = str(amber_buf.get("product_id") or thicket_bag.get("product_id") or "")
            rootlet_cache = str(amber_buf.get("sku") or thicket_bag.get("sku") or "")
            if squall_bag:
                canopy_node[f"{squall_bag}::{rootlet_cache}"] = dict(amber_buf)
                canopy_node[squall_bag] = dict(amber_buf)
        if cove_bag in {"view", "compare"}:
            # view payload usually has product_id at top level; compare has compared rows.
            squall_bag = amber_buf.get("product_id")
            if isinstance(squall_bag, str) and squall_bag:
                crimson_gadget_87_ref[squall_bag] = dict(amber_buf)
            for fen_node in amber_buf.get("compared") or []:
                if isinstance(fen_node, dict) and isinstance(fen_node.get("product_id"), str):
                    crimson_gadget_87_ref[fen_node["product_id"]] = dict(fen_node)
    except Exception:
        return None


def agent_main(birch_buf: dict[str, Any]) -> list[dict[str, Any]]:
    """Satisfy the shopper with dynamic environment tools until done=true."""
    try:
        aspen_ref = birch_buf["environment"]
        breeze_ref = aspen_ref["binding"]
        rapids_slot = aspen_ref["policy_view"]
        dewdrop_pool = list(rapids_slot.get("tools") or [])
        zephyr_node = str(birch_buf.get("problem_id", birch_buf.get("id", "problem")))
        ivy_bag = max(1, int(rapids_slot.get("max_calls_per_turn", 1)))
        quartz_slot = int(rapids_slot["max_steps"])

        canopy_node: dict[str, dict[str, Any]] = {}
        crimson_gadget_87_ref: dict[str, dict[str, Any]] = {}
        lichen_node = False
        frost_buf = False

        pollen_node: list[dict[str, Any]] = [
            {"role": "system", "content": loose_juniper(dewdrop_pool, ivy_bag, quartz_slot)},
            {"role": "user", "content": rapids_slot["query"]},
        ]
        equinox_node: list[dict[str, Any]] = []

        for spruce_pool in range(1, quartz_slot + 1):
            try:
                delta_cache = quartz_slot - spruce_pool + 1
                noon_bag = None
                if delta_cache <= 6:
                    noon_bag = (
                        f"{delta_cache} turns left. If a valid item is inspected, add_to_cart, "
                        "inspect_stock, and place_test_order now."
                    )
                if delta_cache <= 3:
                    noon_bag = (
                        f"Only {delta_cache} turns left. Place the order for the best verified "
                        "in-cart variant immediately."
                    )
                if frost_buf and lichen_node is False:
                    noon_bag = (
                        (noon_bag + " " if noon_bag else "")
                        + "The cart was cleared. Select another valid product, cart it, "
                        "re-check stock, and order."
                    )

                timber_slot = list(pollen_node)
                if noon_bag:
                    timber_slot.append({"role": "user", "content": noon_bag})

                moss_bag = ""
                bramble_buf: list[dict[str, Any]] = []
                for dawn_slot in range(2):
                    try:
                        aurora_ref = orbit_bag.post(
                            "/inference/chat/completions",
                            json_data={
                                "model": unpack_glade(),
                                "messages": timber_slot,
                                "tools": dewdrop_pool,
                                "tool_choice": "required",
                                "temperature": 0,
                            },
                        )
                        if aurora_ref is None:
                            continue
                        fern_node = aurora_ref["choices"][0]["message"]
                        moss_bag = fern_node.get("content") or ""
                        bramble_buf = list(fern_node.get("tool_calls") or [])[:ivy_bag]
                        if bramble_buf:
                            break
                        timber_slot.extend(
                            [
                                {"role": "assistant", "content": moss_bag},
                                {
                                    "role": "user",
                                    "content": (
                                        "Continue with a tool call toward inspecting and ordering "
                                        "a valid product."
                                    ),
                                },
                            ]
                        )
                    except Exception:
                        continue
                if not bramble_buf:
                    break

                mist_node: list[dict[str, Any]] = []
                shell_node: list[tuple[str, dict[str, Any]]] = []
                for nectar_ref in bramble_buf:
                    try:
                        cove_bag = nectar_ref["function"]["name"]
                        thicket_bag = draw_heather(
                            cove_bag,
                            scan_spore(nectar_ref),
                            dewdrop_pool,
                            canopy_node,
                            crimson_gadget_87_ref,
                        )
                    except Exception:
                        continue
                    shell_node.append((cove_bag, thicket_bag))
                    mist_node.append(
                        {
                            "id": nectar_ref.get("id") or f"call-{spruce_pool}-{len(mist_node)+1}",
                            "type": "function",
                            "function": {
                                "name": cove_bag,
                                "arguments": json.dumps(thicket_bag, ensure_ascii=False),
                            },
                        }
                    )
                if not mist_node:
                    pollen_node.extend(
                        [
                            {"role": "assistant", "content": moss_bag},
                            {
                                "role": "user",
                                "content": "Call a tool with valid JSON arguments.",
                            },
                        ]
                    )
                    continue
                bramble_buf = mist_node

                for cove_bag, sedge_cache in shell_node:
                    try:
                        if cove_bag == "add_to_cart":
                            lichen_node = True
                            frost_buf = False
                        if cove_bag == "remove_from_cart":
                            frost_buf = True
                            lichen_node = False
                    except Exception:
                        continue

                pollen_node.append(
                    {
                        "role": "assistant",
                        "content": moss_bag,
                        "tool_calls": bramble_buf,
                    }
                )
                harbor_node = f"{zephyr_node}-turn-{spruce_pool}"
                orchid_cache = {
                    **breeze_ref,
                    "call_id": harbor_node,
                    "idempotency_key": harbor_node,
                    "turn": spruce_pool,
                    "calls": [
                        {
                            "call_id": f"{harbor_node}-{haze_bag}",
                            "action": {"name": cove_bag, "args": thicket_bag},
                        }
                        for haze_bag, (cove_bag, thicket_bag) in enumerate(shell_node, start=1)
                    ],
                }
                dune_bag = orbit_bag.post("/environment/call", json_data=orchid_cache)
                if dune_bag is None:
                    continue

                equinox_node.append(
                    {
                        "role": "assistant",
                        "content": moss_bag,
                        "tool_calls": bramble_buf,
                        "environment_result": dune_bag,
                    }
                )

                for (cove_bag, thicket_bag), spray_bag, willow_slot in zip(
                    shell_node, bramble_buf, dune_bag["calls"], strict=True
                ):
                    try:
                        coral_buf = willow_slot["observation"]
                        trim_solstice(cove_bag, thicket_bag, coral_buf, canopy_node, crimson_gadget_87_ref)
                        pollen_node.append(
                            {
                                "role": "tool",
                                "tool_call_id": spray_bag["id"],
                                "content": json.dumps(coral_buf),
                            }
                        )
                    except Exception:
                        continue
                if dune_bag.get("user_message"):
                    try:
                        pollen_node.append(
                            {"role": "user", "content": dune_bag["user_message"]["content"]}
                        )
                    except Exception:
                        pass
                try:
                    if any(nectar_ref["observation"]["done"] for nectar_ref in dune_bag["calls"]):
                        break
                except Exception:
                    continue
            except Exception:
                continue

        return equinox_node
    except Exception:
        return []


__all__ = ["agent_main"]
