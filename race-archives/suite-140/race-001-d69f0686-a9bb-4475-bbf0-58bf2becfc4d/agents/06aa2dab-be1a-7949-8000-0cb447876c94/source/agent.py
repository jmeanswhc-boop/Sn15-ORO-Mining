"""Hotkey 05 agent: nothing is bought until it has been proven.

Workflow is the ORO Bench generated-environment contract, unchanged:

    policy_view -> inference over the dynamic tools -> /environment/call
                -> observation -> repeat until done=true

The strategy is depth-first scepticism. Each variant carries a small set of
proof obligations, each one a predicate over observed values only. The agent
concentrates on a single focus candidate until its obligations are discharged
or it is disqualified, and a disqualified variant is remembered so it is never
proposed again. The model is told, by name, exactly which obligations remain
outstanding -- steering by information rather than by restriction.

Written as plain functions over one mutable state mapping, with the obligation
predicates held in a lookup table.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.agent.proxy_client import ProxyClient


RUNGS = {
    "chutes": ("zai-org/GLM-5.2-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"),
    "openrouter": ("z-ai/glm-5.2", "deepseek/deepseek-v3.2"),
}

MARKER = "Task rules:"
ENDGAME = 4

# Every episode must reach a terminal state well inside this wall clock. Past
# the soft share the obligations stop being worth chasing and the cart gets
# settled; past the hard share the only remaining goal is to finish at all.
TIME_BUDGET = 300.0
SOFT_SHARE = 0.60
HARD_SHARE = 0.86

# Discovery calls that reveal no new variant are stagnant. Enough in a row and
# searching has stopped paying, so the turns are better spent committing.
DISCOVERY = frozenset({"search", "filter"})
STAGNANT_LIMIT = 6
RECENT_MESSAGES = 14
PROBE_WORDS = 9

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]{1,}")
AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*[A-Z]{3}\b")
CAP_HINT = re.compile(
    r"(under|below|up to|at most|no more than|within|max|maximum|budget|spend)\W{0,24}$",
    re.IGNORECASE,
)
LOOKING = ("search", "filter", "view", "compare", "inspect_stock", "inspect_cart")


def say(text: str) -> None:
    """Instrumentation only; the returned dialogue is untouched."""
    try:
        print(f"[hk05] {text}", flush=True)
    except Exception:
        pass


def mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def listing(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def decimal(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def counted(value: Any, spare: int) -> int:
    try:
        number = int(value)
        return number if number > 0 else spare
    except (TypeError, ValueError):
        return spare


def tokens_of(text: str) -> List[str]:
    found: List[str] = []
    marked = set()
    for raw in TOKEN.findall(text or ""):
        word = raw.strip(".-").lower()
        if len(word) < 2 or word in marked:
            continue
        marked.add(word)
        found.append(word)
    return found


# --------------------------------------------------------------- obligations


def _met_examined(record: Dict[str, Any], world: Dict[str, Any]) -> bool:
    """Raw facts for the listing have actually been retrieved."""
    return bool(record.get("examined"))


def current(record: Dict[str, Any], world: Dict[str, Any]) -> bool:
    """A reading taken since the cart last changed. Adding to the cart is not
    a reading: the catalogue may reprice or sell out between add and order."""
    return record.get("fresh_at") == world.get("epoch")


def _met_priced(record: Dict[str, Any], world: Dict[str, Any]) -> bool:
    """A current price for this exact variant has been read back."""
    return current(record, world) and record.get("price") is not None


def _met_stocked(record: Dict[str, Any], world: Dict[str, Any]) -> bool:
    return record.get("in_stock") is True


def _met_affordable(record: Dict[str, Any], world: Dict[str, Any]) -> bool:
    cap = world.get("ceiling")
    price = record.get("price")
    if cap is None or price is None:
        return True
    return price <= cap


CHECKS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]] = {
    "examined": _met_examined,
    "priced": _met_priced,
    "stocked": _met_stocked,
    "affordable": _met_affordable,
}

# How an outstanding obligation is discharged, given the tools on offer.
REMEDY = {
    "examined": ("view", "compare"),
    "priced": ("inspect_stock",),
    "stocked": ("inspect_stock",),
    "affordable": ("inspect_stock",),
}


def outstanding(record: Dict[str, Any], world: Dict[str, Any]) -> Tuple[str, ...]:
    unmet: List[str] = []
    for name, predicate in CHECKS.items():
        try:
            if not predicate(record, world):
                unmet.append(name)
        except Exception:
            unmet.append(name)
    return tuple(unmet)


def disqualify(record: Dict[str, Any], world: Dict[str, Any]) -> Optional[str]:
    """A settled reason this variant can never be bought, or None."""
    try:
        if not current(record, world):
            return None
        if record.get("in_stock") is False:
            return "out of stock on a current reading"
        cap = world.get("ceiling")
        price = record.get("price")
        if cap is not None and price is not None and price > cap:
            return f"current price {price} exceeds the ceiling {cap}"
        return None
    except Exception:
        return None


# ---------------------------------------------------------------- goal input


def ceiling_from(head: str) -> Optional[float]:
    try:
        anchored: List[float] = []
        every: List[float] = []
        for hit in AMOUNT.finditer(head or ""):
            amount = decimal(hit.group(1))
            if amount is None:
                continue
            every.append(amount)
            if CAP_HINT.search(head[max(0, hit.start() - 40) : hit.start()]):
                anchored.append(amount)
        if anchored:
            return min(anchored)
        if len(every) == 1:
            return every[0]
        return min(every) if every else None
    except Exception:
        return None


def finisher_from(tools: Sequence[Any]) -> Tuple[str, List[str]]:
    """Session-ending tool and its claim vocabulary, taken from the schema."""
    guess = ""
    try:
        for tool in tools:
            function = mapping(mapping(tool).get("function"))
            name = str(function.get("name") or "")
            enum = listing(
                mapping(
                    mapping(
                        mapping(
                            mapping(
                                mapping(
                                    mapping(function.get("parameters")).get("properties")
                                ).get("justification")
                            ).get("properties")
                        ).get("claims")
                    ).get("items")
                )
                .get("properties", {})
                .get("field", {})
                .get("enum")
            )
            if enum:
                return name, [str(item) for item in enum]
            if not guess and "order" in name.lower():
                guess = name
    except Exception:
        pass
    return guess, []


def new_world(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    source = mapping(problem_data)
    environment = mapping(source.get("environment"))
    policy = mapping(environment.get("policy_view"))
    goal = str(policy.get("query") or source.get("query") or "")
    head, _, rules = goal.partition(MARKER)
    tools = listing(policy.get("tools"))
    finisher, claims = finisher_from(tools)

    provider = getenv("INFERENCE_PROVIDER", "chutes")
    ladder = list(RUNGS.get(provider, RUNGS["chutes"]))
    spare = getenv("SANDBOX_MODEL")
    if spare and spare not in ladder:
        ladder.append(spare)

    return {
        "binding": mapping(environment.get("binding")),
        "tools": tools,
        "names": [
            str(mapping(mapping(tool).get("function")).get("name") or "")
            for tool in tools
        ],
        "goal": goal,
        "head": head,
        "rules": rules.strip(),
        "ceiling": ceiling_from(head),
        "words": tokens_of(head),
        "finisher": finisher,
        "claims": claims,
        "limit": counted(policy.get("max_steps"), 20),
        "per_turn": counted(policy.get("max_calls_per_turn"), 1),
        "task": str(source.get("problem_id") or source.get("id") or "task"),
        "proxy": ProxyClient(
            timeout=40,
            max_retries=2,
            retry_delay=2,
            rate_limit_retry_delay=8,
            api_key=getenv("INFERENCE_ACCESS_TOKEN") or getenv("CHUTES_ACCESS_TOKEN"),
        ),
        "ladder": ladder,
        "rung": 0,
        "epoch": 0,
        "started": time.monotonic(),
        "stagnant": 0,
        "misses": 0,
        "muted": False,
        "records": {},
        "rejects": {},
        "cart": [],
        "focus": None,
        "ordered": False,
        "messages": [],
        "transcript": [],
        "spent": 0,
        "quiet": 0,
        "probed": 0,
    }


# ------------------------------------------------------------------ transport


def pace(world: Dict[str, Any]) -> None:
    """Size the timeout and retry count to the time actually left.

    The client applies its timeout per attempt, so a stalled endpoint can
    otherwise burn retries x timeout seconds inside one call and take the whole
    episode with it.
    """
    try:
        left = max(0.0, TIME_BUDGET * HARD_SHARE - elapsed(world))
        window = max(10.0, min(40.0, left / 5.0))
        world["proxy"].timeout = int(window)
        world["proxy"].max_retries = 2 if left > 3 * window else 1
    except Exception:
        pass


def missed(world: Dict[str, Any], answered: bool) -> None:
    """Give up on inference once it has stalled twice in a row.

    An endpoint that has already failed twice will not recover inside this
    budget, and the obligation table still drives the episode to a terminal
    state on its own.
    """
    if answered:
        world["misses"] = 0
        return
    world["misses"] = world.get("misses", 0) + 1
    if world["misses"] >= 2 and not world.get("muted"):
        world["muted"] = True
        say("inference stalled twice; running on the obligation ladder alone")


def current_model(world: Dict[str, Any]) -> str:
    ladder = world["ladder"]
    return ladder[min(world["rung"], len(ladder) - 1)]


def consult(
    world: Dict[str, Any], messages: Sequence[Any]
) -> Optional[Dict[str, Any]]:
    """One tool-choosing completion, descending the model ladder on failure."""
    payload = {
        "model": current_model(world),
        "messages": list(messages),
        "tools": world["tools"],
        "tool_choice": "required",
        "temperature": 0,
    }
    if world.get("muted"):
        return None
    for _attempt in range(2):
        pace(world)
        try:
            answer = world["proxy"].post(
                "/inference/chat/completions", json_data=payload
            )
        except Exception as blip:
            say(f"inference raised {type(blip).__name__}: {blip}")
            answer = None
        choices = listing(mapping(answer).get("choices"))
        missed(world, bool(choices))
        if choices:
            return mapping(mapping(choices[0]).get("message"))
        if world["rung"] + 1 >= len(world["ladder"]):
            return None
        world["rung"] += 1
        payload["model"] = current_model(world)
        say(f"model ladder now on {payload['model']}")
    return None


def commit(world: Dict[str, Any], envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pace(world)
    try:
        answer = world["proxy"].post("/environment/call", json_data=envelope)
        return answer if isinstance(answer, dict) else None
    except Exception as blip:
        say(f"environment call raised {type(blip).__name__}: {blip}")
        return None


# -------------------------------------------------------------- bookkeeping


def touch(
    world: Dict[str, Any],
    row: Any,
    parent: str = "",
    examined: bool = False,
    fresh: bool = False,
) -> Optional[Tuple[str, str]]:
    entry = mapping(row)
    product = str(entry.get("product_id") or parent or "")
    sku = str(entry.get("sku") or "")
    if not product or not sku:
        return None
    key = (product, sku)
    record = world["records"].setdefault(
        key, {"product_id": product, "sku": sku, "examined": False, "fresh_at": -1}
    )
    for column in ("title", "brand", "currency", "category_path"):
        if entry.get(column) is not None:
            record[column] = entry[column]
    for column in ("price", "price_at_add", "price_at_order"):
        amount = decimal(entry.get(column))
        if amount is not None:
            record["price"] = amount
    if isinstance(entry.get("in_stock"), bool):
        record["in_stock"] = entry["in_stock"]
    if examined:
        record["examined"] = True
    if fresh:
        record["fresh_at"] = world["epoch"]
    return key


def absorb(world: Dict[str, Any], action: str, body: Dict[str, Any]) -> None:
    """Fold one observation into the record table and the rejection log."""
    try:
        deep = action in ("view", "compare")
        live = action == "inspect_stock"
        if action in ("add_to_cart", "remove_from_cart"):
            world["epoch"] += 1
        for row in listing(body.get("results")):
            touch(world, row)
        for row in listing(body.get("products")):
            touch(world, row, examined=True)
        parent = str(body.get("product_id") or "")
        for row in listing(body.get("variants")):
            touch(world, row, parent=parent, examined=deep)
        touch(world, body, examined=deep, fresh=live)
        added = body.get("added")
        if isinstance(added, dict):
            touch(world, added)
        if isinstance(body.get("cart"), list):
            world["cart"] = []
            for row in listing(body.get("cart")):
                key = touch(world, row)
                if key:
                    world["cart"].append(key)
        if body.get("order"):
            world["ordered"] = True
        review_rejects(world)
    except Exception:
        pass


def review_rejects(world: Dict[str, Any]) -> None:
    for key, record in world["records"].items():
        if key in world["rejects"]:
            continue
        reason = disqualify(record, world)
        if reason:
            world["rejects"][key] = reason
            say(f"disqualified {key[0]}::{key[1]} -- {reason}")
            if world["focus"] == key:
                world["focus"] = None


def elapsed(world: Dict[str, Any]) -> float:
    try:
        return time.monotonic() - float(world.get("started") or 0.0)
    except Exception:
        return 0.0


def pressed(world: Dict[str, Any]) -> bool:
    """Turns, wall clock or fruitless searching say stop investigating."""
    return (
        (world["limit"] - world["spent"]) <= ENDGAME
        or elapsed(world) >= TIME_BUDGET * SOFT_SHARE
        or world["stagnant"] >= STAGNANT_LIMIT
    )


def spent_out(world: Dict[str, Any]) -> bool:
    return (world["limit"] - world["spent"]) <= 2 or elapsed(world) >= (
        TIME_BUDGET * HARD_SHARE
    )


def any_focus(world: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Cheapest variant known, obligations and ceiling ignored.

    A last resort only. An episode that never reaches a terminal state scores
    nothing, so an imperfect purchase is worth more than an unfinished one.
    """
    try:
        pool = []
        for key, record in world["records"].items():
            if key in world["cart"]:
                continue
            price = record.get("price")
            pool.append(
                (
                    0 if record.get("in_stock") is not False else 1,
                    price if isinstance(price, float) else float("inf"),
                    key,
                )
            )
        if not pool:
            return None
        pool.sort(key=lambda row: (row[0], row[1]))
        return pool[0][2]
    except Exception:
        return None


def last_ditch(world: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One envelope that reaches a terminal state, proven or not.

    The add and the order travel together because a single round trip is all
    the remaining budget can pay for.
    """
    try:
        held = world["cart"][0] if world["cart"] else None
        if held is not None and world["finisher"]:
            return [order_move(world, world["records"].get(held, {}))]
        key = choose_focus(world) or any_focus(world)
        if key is None or "add_to_cart" not in world["names"]:
            return []
        filling = made_move(
            "add_to_cart", {"product_id": key[0], "sku": key[1]}, None
        )
        if not world["finisher"]:
            return [filling]
        return [filling, order_move(world, world["records"].get(key, {}))]
    except Exception:
        return []


def choose_focus(world: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The live candidate closest to being provable."""
    try:
        current = world["focus"]
        if current and current not in world["rejects"] and current in world["records"]:
            return current
        best: Optional[Tuple[Tuple[int, float], Tuple[str, str]]] = None
        for key, record in world["records"].items():
            if key in world["rejects"] or record.get("in_stock") is False:
                continue
            price = record.get("price")
            rank = (
                len(outstanding(record, world)),
                price if isinstance(price, float) else float("inf"),
            )
            if best is None or rank < best[0]:
                best = (rank, key)
        world["focus"] = best[1] if best else None
        return world["focus"]
    except Exception:
        return None


# ------------------------------------------------------------------ briefing


def charter(world: Dict[str, Any]) -> str:
    lines = [
        "You are a shopping agent in a generated catalogue environment.",
        "Only the listed tools exist; never invent a name or an argument.",
        f"Tools: {', '.join(name for name in world['names'] if name) or 'none'}.",
        f"Turn allowance: {world['limit']}.",
        "",
        "This session is deliberately sceptical. A candidate counts as proven"
        " only once four things hold, each from something the environment"
        " returned rather than from a title:",
        "  examined   - its raw facts and catalogue classification were retrieved",
        "  priced     - a current price for that exact variant was read back",
        "  stocked    - that reading says it is in stock",
        "  affordable - that price sits inside the stated budget",
        "",
        "Each turn you are told which obligations are still outstanding for the"
        " candidate under investigation, and which variants have already been"
        " disqualified. Work the outstanding obligation. Do not revisit a"
        " disqualified variant.",
        "- Firm requirements outrank preferences; never trade one away.",
        "- One title often covers a compliant and a non-compliant variant."
        " Judge the specific SKU.",
        "- Stock and prices move as you shop, so a reading taken before a cart"
        " action is stale afterwards.",
        "- An error or an empty result means change approach, not repeat.",
        "- A shopper message overrides your earlier assumptions.",
        "- The first item you cart is judged on its own merits.",
    ]
    if world["ceiling"] is not None:
        lines.append(f"- Stated ceiling: {world['ceiling']}. Treat it as firm.")
    if world["finisher"]:
        lines.append(
            f"- {world['finisher']} ends the session. Call it as soon as the"
            " candidate is proven; do not ask for confirmation."
        )
    if world["claims"]:
        lines.append(
            "- If the shopper wants reasons, attach claims using only these"
            f" fields: {', '.join(world['claims'])}. Each must be true against an"
            " observed value, as field and value, with no prose."
        )
    if world["rules"]:
        lines.extend(["", "Rules for this task:", world["rules"]])
    return "\n".join(lines)


def status_note(world: Dict[str, Any]) -> str:
    left = world["limit"] - world["spent"]
    key = choose_focus(world)
    lines: List[str] = []
    if key is None:
        lines.append(
            "No live candidate is on the table. Search or filter for listings"
            " that could satisfy the firm requirements."
        )
    else:
        record = world["records"].get(key, {})
        unmet = outstanding(record, world)
        title = str(record.get("title") or "")[:70]
        lines.append(f"Candidate under investigation: {key[0]}::{key[1]} {title}")
        if record.get("price") is not None:
            lines.append(f"  last seen price: {record['price']}")
        if unmet:
            lines.append(f"  outstanding obligations: {', '.join(unmet)}")
            hints = [
                tool
                for name in unmet
                for tool in REMEDY.get(name, ())
                if tool in world["names"]
            ]
            if hints:
                lines.append(f"  discharge them with: {', '.join(dict.fromkeys(hints))}")
        else:
            lines.append("  every obligation is satisfied; it is ready to buy")
    if world["rejects"]:
        recent = list(world["rejects"].items())[-4:]
        lines.append("Disqualified already, do not revisit:")
        lines.extend(f"  {a}::{b} -- {why}" for (a, b), why in recent)
    if world["cart"]:
        lines.append(f"In the cart: {', '.join(f'{a}::{b}' for a, b in world['cart'])}")
    lines.append(f"{left} turns remain.")
    return "\n".join(lines)


# -------------------------------------------------------------- move choosing


def envelope_for(
    world: Dict[str, Any], turn: int, moves: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    tag = f"{world['task']}-turn-{turn}"
    return {
        **world["binding"],
        "call_id": tag,
        "idempotency_key": tag,
        "turn": turn,
        "calls": [
            {
                "call_id": f"{tag}-{index}",
                "action": {"name": move["name"], "args": move["args"]},
            }
            for index, move in enumerate(moves, start=1)
        ],
    }


def made_move(name: str, args: Dict[str, Any], answers: Optional[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "args": {key: value for key, value in args.items() if value is not None},
        "answers": answers,
    }


def move_from_call(call: Any) -> Optional[Dict[str, Any]]:
    entry = mapping(call)
    function = mapping(entry.get("function"))
    name = str(function.get("name") or "")
    if not name:
        return None
    raw = function.get("arguments")
    args: Dict[str, Any] = {}
    if isinstance(raw, dict):
        args = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            args = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            args = {}
    identity = entry.get("id")
    return made_move(name, args, identity if isinstance(identity, str) else None)


def order_move(world: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "product_id": record.get("product_id"),
        "sku": record.get("sku"),
    }
    allowed = set(world["claims"])
    proven: List[Dict[str, str]] = []
    if "price" in allowed and record.get("price") is not None and record.get(
        "fresh_at", -1
    ) >= 0:
        proven.append({"field": "price", "value": f"{record['price']}"})
    if "in_stock" in allowed and record.get("in_stock") is True:
        proven.append({"field": "in_stock", "value": "true"})
    if "brand" in allowed and record.get("brand"):
        proven.append({"field": "brand", "value": str(record["brand"])})
    if proven:
        args["justification"] = {"claims": proven}
    return made_move(world["finisher"], args, None)


def obligation_move(world: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Discharge the focus candidate's next obligation, without the model."""
    try:
        held = world["cart"][0] if world["cart"] else None
        if held is not None:
            record = world["records"].get(held, {})
            if not outstanding(record, world) and world["finisher"]:
                return order_move(world, record)
        key = choose_focus(world)
        if key is None:
            return discovery_move(world)
        record = world["records"].get(key, {})
        unmet = outstanding(record, world)
        for name in unmet:
            for tool in REMEDY.get(name, ()):
                if tool not in world["names"]:
                    continue
                if tool == "view":
                    return made_move("view", {"product_id": key[0]}, None)
                if tool == "compare":
                    return made_move("compare", {"product_ids": [key[0]]}, None)
                return made_move(
                    tool, {"product_id": key[0], "sku": key[1]}, None
                )
        if held is None and "add_to_cart" in world["names"]:
            return made_move(
                "add_to_cart", {"product_id": key[0], "sku": key[1]}, None
            )
        if world["finisher"]:
            return order_move(world, record)
        return None
    except Exception:
        return None


def gate(
    world: Dict[str, Any], moves: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Hold back an order while the cart item still owes an obligation.

    The model may commit on a reading taken before the cart changed. Rather
    than forwarding that, the outstanding obligation is discharged first and
    the model decides again from the result.
    """
    try:
        finisher = world["finisher"]
        if not finisher or not any(move["name"] == finisher for move in moves):
            return None
        held = world["cart"][0] if world["cart"] else None
        if held is None:
            return None
        record = world["records"].get(held, {})
        unmet = outstanding(record, world)
        if not unmet:
            return None
        if disqualify(record, world) and "remove_from_cart" in world["names"]:
            return made_move(
                "remove_from_cart", {"product_id": held[0], "sku": held[1]}, None
            )
        for name in unmet:
            for tool in REMEDY.get(name, ()):
                if tool not in world["names"]:
                    continue
                if tool == "view":
                    return made_move("view", {"product_id": held[0]}, None)
                if tool == "compare":
                    return made_move("compare", {"product_ids": [held[0]]}, None)
                return made_move(
                    tool, {"product_id": held[0], "sku": held[1]}, None
                )
        return None
    except Exception:
        return None


def discovery_move(world: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        world["probed"] += 1
        if "search" in world["names"] and world["probed"] <= 3:
            words = [word for word in world["words"] if len(word) >= 3][:PROBE_WORDS]
            return made_move(
                "search", {"query": " ".join(words) or "in stock", "in_stock": True},
                None,
            )
        if "filter" in world["names"]:
            return made_move(
                "filter", {"in_stock": True, "max_price": world["ceiling"]}, None
            )
        return None
    except Exception:
        return None


def plan_turn(world: Dict[str, Any], turn: int) -> List[Dict[str, Any]]:
    if spent_out(world):
        final = last_ditch(world)
        if final:
            say(
                f"turn {turn}: {elapsed(world):.0f}s gone, finishing with "
                f"{', '.join(move['name'] for move in final)}"
            )
            return final

    if pressed(world):
        forced = endgame_move(world)
        if forced is not None:
            say(
                f"turn {turn}: closing out -> {forced['name']} "
                f"({elapsed(world):.0f}s, stagnant={world['stagnant']})"
            )
            return [forced]

    world["messages"].append({"role": "user", "content": status_note(world)})
    thread = [
        {"role": "system", "content": charter(world)},
        {"role": "user", "content": world["goal"]},
    ] + world["messages"]

    reply = consult(world, thread)
    if reply is None:
        say(f"turn {turn}: retrying on a shorter thread")
        reply = consult(world, thread[:2] + world["messages"][-RECENT_MESSAGES:])
    if reply is None:
        world["quiet"] += 1
        fallback = obligation_move(world)
        if fallback is None:
            return []
        say(f"turn {turn}: no inference, discharging via {fallback['name']}")
        return [fallback]

    spoken = str(reply.get("content") or "")
    raw = listing(reply.get("tool_calls"))
    picked: List[Dict[str, Any]] = []
    room = max(1, min(world["per_turn"], 3))
    for call in raw:
        move = move_from_call(call)
        if move is None:
            continue
        if move["name"] not in LOOKING:
            picked = picked or [move]
            break
        picked.append(move)
        if len(picked) >= room:
            break

    if not picked:
        world["quiet"] += 1
        world["messages"].append({"role": "assistant", "content": spoken})
        world["messages"].append(
            {
                "role": "user",
                "content": (
                    "That reply carried no tool call. Discharge the outstanding"
                    " obligation with a single tool call now."
                ),
            }
        )
        world["transcript"].append({"role": "assistant", "content": spoken})
        fallback = obligation_move(world) if world["quiet"] >= 2 else None
        return [fallback] if fallback is not None else []

    held_back = gate(world, picked)
    if held_back is not None:
        owed = outstanding(world["records"].get(world["cart"][0], {}), world)
        say(f"order held back over {', '.join(owed)}; {held_back['name']} runs first")
        world["messages"].append({"role": "assistant", "content": spoken})
        world["messages"].append(
            {
                "role": "user",
                "content": (
                    "That order was not sent. The cart item still owes"
                    f" {', '.join(owed)}, so {held_back['name']} is running first."
                    " Decide again from its result."
                ),
            }
        )
        return [held_back]

    answered = {move["answers"] for move in picked if move["answers"]}
    world["messages"].append(
        {
            "role": "assistant",
            "content": spoken,
            "tool_calls": [
                call for call in raw if mapping(call).get("id") in answered
            ],
        }
    )
    return picked


def endgame_move(world: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Reach a terminal state, preferring a proven item but taking any."""
    try:
        if world["cart"]:
            head = world["cart"][0]
            record = world["records"].get(head, {})
            spare = choose_focus(world)
            if disqualify(record, world) and spare is not None and (
                "remove_from_cart" in world["names"]
            ):
                return made_move(
                    "remove_from_cart",
                    {"product_id": head[0], "sku": head[1]},
                    None,
                )
            # With nothing better to swap to, churning the cart only burns
            # turns, so take what is already in it.
            if world["finisher"] and record.get("in_stock") is not False:
                return order_move(world, record)
        key = choose_focus(world) or any_focus(world)
        if key is not None and "add_to_cart" in world["names"]:
            return made_move(
                "add_to_cart", {"product_id": key[0], "sku": key[1]}, None
            )
        return discovery_move(world)
    except Exception:
        return None


# ------------------------------------------------------------------ digestion


def digest(
    world: Dict[str, Any], moves: Sequence[Dict[str, Any]], reply: Dict[str, Any]
) -> bool:
    results = listing(reply.get("calls"))
    known = len(world["records"])
    finished = False
    for index, move in enumerate(moves):
        slot = results[index] if index < len(results) else {}
        outer = mapping(mapping(slot).get("observation"))
        body = mapping(outer.get("observation"))
        done = bool(outer.get("done"))
        fault = outer.get("error")
        absorb(world, move["name"], body)
        finished = finished or done
        rendered = json.dumps(
            {"result": body, "error": fault, "done": done}, default=str
        )
        if move["answers"]:
            world["messages"].append(
                {"role": "tool", "tool_call_id": move["answers"], "content": rendered}
            )
        else:
            world["messages"].append(
                {"role": "user", "content": f"{move['name']} returned {rendered}"}
            )
        if fault:
            world["messages"].append(
                {
                    "role": "user",
                    "content": (
                        f"{move['name']} reported an error. Adjust the arguments or"
                        " pick another tool rather than repeating it."
                    ),
                }
            )
    if any(move["name"] in DISCOVERY for move in moves):
        if len(world["records"]) > known:
            world["stagnant"] = 0
        else:
            world["stagnant"] += 1
            say(f"discovery revealed nothing new ({world['stagnant']} in a row)")

    spoken = mapping(reply.get("user_message")).get("content")
    if spoken:
        world["messages"].append({"role": "user", "content": str(spoken)})
        world["stagnant"] = 0
    return finished


def charge(
    world: Dict[str, Any], moves: Sequence[Dict[str, Any]], reply: Dict[str, Any]
) -> None:
    try:
        before = world["spent"]
        claimed = [
            counted(reply.get("action_count"), 0),
            counted(reply.get("solver_turn_count"), 0),
        ]
        world["spent"] = max([before + 1] + claimed)
        if len(moves) > 1 and world["spent"] - before >= len(moves):
            say("budget is charged per action; looking one at a time")
            world["per_turn"] = 1
    except Exception:
        world["spent"] += 1


def record_turn(
    world: Dict[str, Any],
    turn: int,
    moves: Sequence[Dict[str, Any]],
    reply: Dict[str, Any],
) -> None:
    world["transcript"].append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": move["answers"] or f"{world['task']}-{turn}-{index}",
                    "type": "function",
                    "function": {
                        "name": move["name"],
                        "arguments": json.dumps(move["args"], default=str),
                    },
                }
                for index, move in enumerate(moves, start=1)
            ],
            "environment_result": reply,
        }
    )


# ----------------------------------------------------------------- main loop


def run(world: Dict[str, Any]) -> List[Dict[str, Any]]:
    say(
        f"task {world['task']}: limit={world['limit']} per_turn={world['per_turn']} "
        f"ceiling={world['ceiling']} finisher={world['finisher'] or 'unknown'}"
    )
    for turn in range(1, world["limit"] + 1):
        if world["spent"] >= world["limit"] or world["ordered"]:
            break

        moves = plan_turn(world, turn)
        if not moves:
            world["quiet"] += 1
            if world["quiet"] >= 3:
                say(f"turn {turn}: nothing left to try; stopping")
                break
            continue

        reply = commit(world, envelope_for(world, turn, moves))
        if reply is None and len(moves) > 1:
            say(f"turn {turn}: batch refused; resending a single action")
            moves = moves[:1]
            world["per_turn"] = 1
            reply = commit(world, envelope_for(world, turn, moves))
        if reply is None:
            say(f"turn {turn}: environment unreachable; stopping")
            break

        record_turn(world, turn, moves, reply)
        finished = digest(world, moves, reply)
        charge(world, moves, reply)
        focus = world["focus"]
        say(
            f"turn {turn}: {', '.join(move['name'] for move in moves)} "
            f"focus={focus[0] + '::' + focus[1] if focus else 'none'} "
            f"rejects={len(world['rejects'])} done={finished} "
            f"spent={world['spent']}/{world['limit']}"
        )
        if finished:
            break
    return world["transcript"]


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point: run one generated-environment episode."""
    world: Optional[Dict[str, Any]] = None
    try:
        world = new_world(problem_data)
        return run(world)
    except Exception as blip:
        say(f"episode ended by {type(blip).__name__}: {blip}")
        try:
            return listing(mapping(world).get("transcript"))
        except Exception:
            return []


__all__ = ["agent_main"]
