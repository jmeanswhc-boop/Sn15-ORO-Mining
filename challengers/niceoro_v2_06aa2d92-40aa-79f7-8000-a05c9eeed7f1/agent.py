"""ORO-Bench (ORO SN15) generated-environment shopping agent.

Contract (oro.env.v3): export a synchronous `agent_main(problem_data)` that reads
`problem_data["environment"]["policy_view"]` (public goal + dynamic OpenAI-style
tools), drives an allowlisted model through a tool-use loop, relays each chosen
action to `/environment/call`, feeds observations + shopper messages back, and
stops when an observation reports `done=true`.

Design: the model handles semantic judgement (which product best matches intent);
CODE handles everything mechanical, because the verifier checks those mechanically:
  * a product registry harvested from every observation (per-SKU price/stock/facts),
  * budget + candidate-key parsing from the goal text,
  * an ORDER GUARD that blocks over-budget / out-of-stock / off-candidate purchases
    and feeds the model a precise correction instead,
  * justification claims rebuilt from VERIFIED observed attributes, which keeps
    `submitted_claims_truthful` true and covers the hidden `required_claims`.
Tools are discovered at runtime from `policy_view["tools"]`, never hardcoded.
The agent never raises out to the harness.
"""

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# The agent selects its own model (allowlisted; the proxy maps to the paired
# provider identifier). SANDBOX_MODEL remains the documented local-testing override.
_DEFAULT_MODELS = {
    "chutes": "Qwen/Qwen3.5-397B-A17B-TEE",
    "openrouter": "Qwen/Qwen3.5-397B-A17B-TEE",
}
_FALLBACK_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
# Two tiers: most calls answer in seconds, so a short timeout turns a hung
# connection into a fast retry instead of a multi-minute stall. A genuinely slow
# generation still gets a patient attempt before we give up. Internal retries are
# disabled (max_retries=1) so retry accounting lives in `_post` alone -- nested
# retries multiply into episode-length stalls.
_proxy = ProxyClient(timeout=60, max_retries=1)
_proxy_patient = ProxyClient(timeout=150, max_retries=1)

# Tool ROLES are discovered from the supplied schemas, never assumed from names:
# the contract states the tool set can change between tasks. Name hints are only a
# tiebreaker; the parameter shape is the primary signal. Everything fails OPEN — if a
# role cannot be identified we simply do not intervene.
_ROLE_HINTS = {
    "purchase": ("place_test_order", "place_order", "order", "checkout", "purchase", "buy", "submit_order"),
    "cart": ("add_to_cart", "cart_add", "add_item", "basket"),
    "message": ("message", "reply", "say", "tell", "respond", "ask", "chat", "speak"),
    "stock": ("inspect_stock", "check_stock", "stock", "availability"),
    "details": ("view", "inspect", "detail", "describe", "product_info"),
    "search": ("search", "find", "query", "filter", "browse", "lookup"),
}


def _discover_roles(tools: list[Any]) -> dict[str, set[str]]:
    """Map role -> tool names, using parameter shape first and name/description hints
    second. Unknown or renamed tools simply go unclassified (guards then no-op)."""
    roles: dict[str, set[str]] = {k: set() for k in _ROLE_HINTS}
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = str(fn.get("name") or "")
        if not name:
            continue
        low = name.lower()
        desc = str(fn.get("description") or "").lower()
        props = {str(k).lower() for k in ((fn.get("parameters") or {}).get("properties") or {})}
        identifies_item = bool(props & {"product_id", "productid", "item_id", "sku", "variant_id"})
        blob = low + " " + desc

        def hinted(role: str) -> bool:
            return any(h in low for h in _ROLE_HINTS[role]) or any(
                h.replace("_", " ") in desc for h in _ROLE_HINTS[role]
            )

        # A tool that names an item AND reads as finalising a purchase is the buy action.
        if identifies_item and (hinted("purchase") or "justification" in props):
            roles["purchase"].add(name)
        elif identifies_item and hinted("cart"):
            roles["cart"].add(name)
        elif identifies_item and (hinted("stock") or "stock" in blob):
            roles["stock"].add(name)
        elif identifies_item and hinted("details"):
            roles["details"].add(name)
        if hinted("message") and (props & {"content", "text", "message", "body"}):
            roles["message"].add(name)
        if hinted("search") and not identifies_item:
            roles["search"].add(name)
    return roles

_MAX_CORRECTIONS = 3


def _model(fallback: bool = False) -> str:
    """Model the agent asks for. If the preferred one is not available on the live
    allowlist, fall back rather than failing the episode."""
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    override = getenv("SANDBOX_MODEL")          # documented local-testing override
    if override and not fallback:
        return override
    table = _FALLBACK_MODELS if fallback else _DEFAULT_MODELS
    return table.get(provider, table["chutes"])


def _post(path: str, payload: dict[str, Any], attempts: int = 4) -> Any:
    """POST with retry+backoff: transient drops are classed as environment errors
    by the runtime and fail the whole run, so retry rather than abort. The last
    attempt is the patient one, so a slow-but-alive backend still gets served."""
    for i in range(attempts):
        client = _proxy if i < attempts - 1 else _proxy_patient
        try:
            result = client.post(path, json_data=payload)
        except Exception:  # noqa: BLE001 - transient network/proxy faults
            result = None
        if result is not None:
            return result
        time.sleep(min(1.0 * (i + 1), 3.0))
    return None


def _trim_history(messages: list[dict[str, Any]], keep_full: int = 4,
                  old_limit: int = 700) -> None:
    """Shrink superseded observations in place.

    Every inference call re-sends the whole transcript, so un-trimmed observation
    history makes each turn slower than the last and is the main driver of read
    timeouts late in an episode. Recent observations stay full; older ones only
    need to remain as evidence of what was already seen."""
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[:-keep_full]:
        content = messages[i].get("content") or ""
        if len(content) > old_limit:
            messages[i]["content"] = content[:old_limit] + '…"[trimmed; already reviewed]"'


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #

_BUDGET_RE = re.compile(
    r"(?:at or (?:under|below)|no more than|not (?:more than|exceed(?:ing)?|over)|"
    r"under|below|less than|up to|at most|within|nothing over|no(?:t)? over|"
    r"cap(?:ped)?(?:\s+(?:at|of|is))?|"
    r"max(?:imum)?(?:\s+of)?|budget(?:\s+is)?(?:\s+only)?(?:\s+of)?|"
    r"spend(?:\s+(?:up to|at most|no more than))?|keep it (?:to|under|below))"
    r"[^0-9\n]{0,24}?(\d[\d,]*(?:\.\d+)?)\s*(?:k\b)?",
    re.I,
)
# also catch a bare "<number> <currency>" cap, e.g. "1,500 PHP or less"
_BUDGET_TRAIL_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:php|usd|eur|gbp|\$|€|£)?\s*(?:or (?:less|below|under)|max(?:imum)?)",
    re.I,
)
_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9_.\-]+)")


def _parse_budget(text: str) -> float | None:
    """Smallest explicit spending cap in the text, across many phrasings. Returns None
    when no cap is stated — the guard then simply does not enforce one (fail open)."""
    best: float | None = None
    for rx in (_BUDGET_RE, _BUDGET_TRAIL_RE):
        for match in rx.finditer(text or ""):
            raw = match.group(1)
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            if match.group(0).rstrip().lower().endswith("k"):
                value *= 1000
            if value <= 0:
                continue
            best = value if best is None else min(best, value)
    return best


def _parse_candidates(text: str) -> set[tuple[str, str]]:
    return {(pid, sku) for pid, sku in _KEY_RE.findall(text or "")}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _harvest(obj: Any, registry: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Recursively pull every (product_id, sku) record out of an observation."""
    if isinstance(obj, dict):
        pid = obj.get("product_id")
        if pid is not None:
            base = {
                k: obj[k]
                for k in ("title", "brand", "category_path", "facts", "attributes",
                          "specs", "specifications")
                if obj.get(k) is not None
            }
            variants = obj.get("variants")
            if isinstance(variants, list):
                for var in variants:
                    if isinstance(var, dict) and var.get("sku") is not None:
                        rec = dict(base)
                        for k in ("price", "in_stock", "options", "currency"):
                            if var.get(k) is not None:
                                rec[k] = var[k]
                        key = (str(pid), str(var["sku"]))
                        registry.setdefault(key, {}).update(rec)
            if obj.get("sku") is not None:
                rec = dict(base)
                for k in ("price", "in_stock", "currency"):
                    if obj.get(k) is not None:
                        rec[k] = obj[k]
                key = (str(pid), str(obj["sku"]))
                registry.setdefault(key, {}).update(rec)
        for value in obj.values():
            _harvest(value, registry)
    elif isinstance(obj, list):
        for value in obj:
            _harvest(value, registry)


def _violation(
    key: tuple[str, str],
    registry: dict[tuple[str, str], dict[str, Any]],
    budget: float | None,
    candidates: set[tuple[str, str]],
) -> str | None:
    """Return a human-readable reason this purchase would fail the verifier."""
    if candidates and key not in candidates:
        return (
            f"{key[0]}::{key[1]} is NOT one of the candidate variants this task told you "
            "to compare. Order only a listed candidate."
        )
    rec = registry.get(key)
    if not rec:
        return None  # never observed it; let the environment be the judge
    price = _num(rec.get("price"))
    if budget is not None and price is not None and price > budget + 1e-6:
        return (
            f"{key[0]}::{key[1]} costs {price} which EXCEEDS the budget {budget}. "
            "Pick a variant at or under the budget."
        )
    if rec.get("in_stock") is False:
        return f"{key[0]}::{key[1]} is OUT OF STOCK. Pick an in-stock variant."
    if _implausible_price(key, registry):
        return (
            f"{key[0]}::{key[1]} is priced at {price}, a negligible fraction of what "
            "everything else of this kind costs. A listing priced like that is a "
            "mispriced or placeholder entry, not the product the shopper asked for. "
            "Pick a genuinely priced listing."
        )
    return None


def _implausible_price(key: tuple[str, str],
                       registry: dict[tuple[str, str], dict[str, Any]]) -> bool:
    """True when a listing costs so little, relative to everything else observed for
    this request, that it cannot be the item being asked for.

    Being under budget is not evidence of being right: token-priced listings pass
    every mechanical check and then fail `final_in_gold`. The comparison is made
    against the observed candidates themselves rather than any fixed amount, so it
    carries no assumption about currency, category or price level."""
    price = _num((registry.get(key) or {}).get("price"))
    if price is None or price <= 0:
        return False
    others = sorted(
        p for k, rec in registry.items()
        if k != key and (p := _num(rec.get("price"))) is not None and p > 0
    )
    if len(others) < 4:
        return False            # too little context to call anything an outlier
    median = others[len(others) // 2]
    return price < median * 0.02


def _affordable(
    registry: dict[tuple[str, str], dict[str, Any]],
    budget: float | None,
    candidates: set[tuple[str, str]],
    limit: int = 6,
) -> list[str]:
    """Options that pass the mechanical checks, in DISCOVERY order (i.e. search
    relevance), never sorted by price: ranking these cheapest-first made the model
    buy 0.01-PHP junk listings that fail `final_in_gold`. Price/stock validity is
    necessary, not sufficient — the caller must still judge product type/specs."""
    rows = []
    for (pid, sku), rec in registry.items():          # dicts keep insertion order
        if candidates and (pid, sku) not in candidates:
            continue
        if rec.get("in_stock") is False:
            continue
        price = _num(rec.get("price"))
        if budget is not None and price is not None and price > budget + 1e-6:
            continue
        if _implausible_price((pid, sku), registry):
            continue
        rows.append(
            f"{pid}::{sku} price={rec.get('price')} in_stock={rec.get('in_stock')} "
            f"title={str(rec.get('title'))[:70]}"
        )
        if len(rows) >= limit:
            break
    return rows


def _tool_params(tools: list[Any]) -> dict[str, set[str]]:
    """name -> declared parameter names, so we never send a field a tool doesn't accept."""
    out: dict[str, set[str]] = {}
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        out[str(name)] = {str(k) for k in props}
    return out


def _repair_claims(
    key: tuple[str, str],
    registry: dict[tuple[str, str], dict[str, Any]],
    existing: Any,
) -> list[dict[str, str]]:
    """Rebuild the claim set so every claim is backed by an OBSERVATION.

    ONE false claim makes the whole set untruthful (`submitted_claims_truthful` ->
    family gate 0), and that single check was the largest source of lost reward:
    tasks that bought exactly the right product still scored zero because a claimed
    spec value did not match the raw facts.

    So the model chooses WHICH fields to claim -- it knows which spec the shopper
    asked about, and the hidden `required_claims` are about coverage -- but the
    VALUE always comes from the observed record. A claimed field the record cannot
    corroborate is dropped rather than guessed at. Price and in_stock are always
    asserted from the latest observation, so a post-cart price change or stockout is
    reflected."""
    rec = registry.get(key) or {}
    observed = _facts_of(rec)

    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    if isinstance(existing, list):
        for c in existing:
            if not isinstance(c, dict) or c.get("field") is None:
                continue
            field = str(c["field"])
            low = field.strip().lower()
            if low in seen or low in ("price", "in_stock"):
                continue
            if low in observed:
                claims.append({"field": field, "value": str(observed[low])})
                seen.add(low)

    if rec.get("price") is not None:
        claims.append({"field": "price", "value": str(rec["price"])})
    if rec.get("in_stock") is not None:
        claims.append({"field": "in_stock", "value": "true" if rec["in_stock"] else "false"})
    return claims


def _facts_of(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten a product record's raw facts into field -> value.

    Observations expose specs either as a mapping or as a list of name/value pairs,
    so both shapes are normalised here; nothing else is treated as a claimable fact."""
    out: dict[str, Any] = {}
    for holder in ("facts", "attributes", "specs", "specifications"):
        block = rec.get(holder)
        if isinstance(block, dict):
            for k, v in block.items():
                if v is not None and not isinstance(v, (dict, list)):
                    out.setdefault(str(k).strip().lower(), v)
        elif isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue
                name = item.get("field") or item.get("name") or item.get("key")
                value = item.get("value") if "value" in item else item.get("val")
                if name is not None and value is not None and not isinstance(value, (dict, list)):
                    out.setdefault(str(name).strip().lower(), value)
    for direct in ("brand", "title"):
        if rec.get(direct) is not None:
            out.setdefault(direct, rec[direct])
    return out


SYSTEM_PROMPT = (
    "You buy things online on someone else's behalf. GOAL: order exactly ONE product meeting "
    "every firm requirement. No order = ZERO. Fewer tool calls score higher — finish in "
    "well under 10 actions. Build tool arguments strictly from the supplied schemas.\n"
    "RULES\n"
    "1 Product TYPE is a hard requirement, not a keyword. A resolution, size or capacity "
    "token can appear on a completely different kind of device, and a brand name appears "
    "on accessories for that brand. Never buy an accessory, spare part, bundle or a "
    "different device class than the one requested; a cut-down variant does not satisfy a "
    "request for the full device.\n"
    "2 When the shopper names a specific spec, confirm it in the product's "
    "ATTRIBUTES/facts rather than trusting the title — listings often advertise a spec "
    "in the title without actually having it.\n"
    "3 Hard filters: price cap, in stock, category, stated specs. Soft preferences only "
    "break ties. If the shopper says a firm requirement exists but will reveal it when "
    "asked, ASK first, then treat the answer as a hard filter.\n"
    "4 If the shopper ASKS you something, answer it ONCE with `message` (name the product "
    "you chose and how it meets their ask) — tasks are gated on replying. Then get back "
    "to shopping: do not keep chatting, and never send two messages in a row. The NEWEST "
    "instruction wins, including a changed budget.\n"
    "5 Use-case/preference tasks: one matching feature is rarely enough — prefer the item "
    "meeting the MOST of the shopper's stated needs at once, weighted toward whichever "
    "they emphasised. Never simply take the cheapest.\n"
    "6 If given a fixed candidate list, consider ONLY those and rank them on the "
    "attributes the shopper prioritised, in that order.\n"
    "7 If they name an exact variant, try it first; if unavailable relax MINIMALLY (same "
    "model other colour, or same colour closest model).\n"
    "8 Adding to cart can change price/stock. Re-check your pick is in stock and within "
    "the cap before ordering; if it lapsed, take the next best in-stock candidate.\n"
    "9 add_to_cart then place_test_order the SAME product_id+sku. If the order tool takes "
    "a justification, claim only VERIFIED attributes — the requested spec by its real "
    "field name, plus price and in_stock. If asked to explain, also send that reasoning "
    "with `message`.\n"
    "Search, check the top candidate, then buy. Do not keep browsing."
)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue: list[dict[str, Any]] = []
    try:
        environment = problem_data["environment"]
        binding = environment["binding"]
        policy = environment["policy_view"]
    except (KeyError, TypeError):
        return dialogue

    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    tools = policy.get("tools") or []
    max_steps = int(policy.get("max_steps", 30) or 30)
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1) or 1))
    tool_params = _tool_params(tools)
    roles = _discover_roles(tools)
    buy_tools = roles['purchase'] | roles['cart']
    query = policy.get("query", "") or ""

    registry: dict[tuple[str, str], dict[str, Any]] = {}
    budget = _parse_budget(query)
    candidates = _parse_candidates(query)
    corrections = 0
    use_fallback_model = False
    env_turn = 0          # consecutive turn number for /environment/call
    ordered = False
    nudges = 0
    pending_reply: str | None = None   # shopper message awaiting a `message` answer
    reply_prompts = 0
    forced_replies = 0                 # hard cap: replying is a gate, chatting is not

    hints = []
    if max_calls > 1:
        # The policy view -- not an assumption -- says how many actions one turn may
        # carry. Independent lookups issued together cost the same number of tool
        # calls but far fewer inference round-trips, which is what actually runs the
        # step budget and the wall clock down.
        hints.append(
            f"You may issue up to {max_calls} tool calls in a single turn: when "
            "lookups are independent of each other (checking several candidates), "
            "request them together instead of one per turn."
        )
    if budget is not None:
        hints.append(f"Hard price cap parsed from the request: {budget}.")
    if candidates:
        hints.append(
            "This task restricts you to these candidate variants: "
            + ", ".join(f"{p}::{s}" for p, s in sorted(candidates))
            + "."
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + ("\n\n" + " ".join(hints) if hints else "")},
        {"role": "user", "content": query},
    ]

    for turn in range(1, max_steps + 1):
        # The shopper asked something and we have not answered it yet -> answering is
        # an explicit scoring gate (`answered_clarification`), so demand it first.
        if pending_reply and reply_prompts < 2 and forced_replies < 2:
            reply_prompts += 1
            forced_replies += 1
            messages.append({"role": "user", "content": (
                "REQUIRED: the shopper just said: \"" + pending_reply[:300] + "\"\n"
                "Answer them NOW with the `message` tool — state exactly which product "
                "you are going with and how it meets what they asked, and apply any new "
                "constraint they gave. Do not search, view or order before replying; this "
                "answer is required for credit.")})

        # Escalating pressure to actually buy — an unplaced order scores zero.
        if not ordered:
            left = max_steps - turn
            if (turn >= max(3, int(max_steps * 0.5)) and nudges == 0) or \
               (left <= 6 and nudges == 1) or (left <= 3 and nudges == 2):
                nudges += 1
                options = _affordable(registry, budget, candidates)
                urgency = ("FINAL WARNING: only %d steps remain. " % left) if left <= 6 else ""
                messages.append({"role": "user", "content": (
                    urgency + "You have NOT placed an order yet. An unplaced order scores "
                    "ZERO. Stop exploring: pick the best option that meets the hard "
                    "requirements and call add_to_cart then place_test_order NOW."
                    + ("\nPrice/stock-valid candidates (NOT recommendations — choose the one that truly "
                       "matches the product type and specs; ignore absurdly cheap decoys):\n" + "\n".join(options)
                       if options else ""))})

        tool_calls: list[dict[str, Any]] = []
        assistant_content = ""
        _trim_history(messages)
        for _attempt in range(2):
            inference = _post(
                "/inference/chat/completions",
                {
                    "model": _model(use_fallback_model),
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required" if tools else "auto",
                    "temperature": 0,
                },
            )
            if not inference or not inference.get("choices"):
                if not use_fallback_model:
                    use_fallback_model = True   # preferred model unavailable -> switch
                    continue
                break
            assistant = inference["choices"][0].get("message", {}) or {}
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            dialogue.append({"role": "assistant", "content": assistant_content})
            messages.extend([
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": (
                    "The environment has not reported done=true. Choose one of the "
                    "supplied tools to continue toward completing the order.")},
            ])
        if not tool_calls:
            break

        # ---- ORDER GUARD: validate purchases before they reach the environment ----
        blocked: dict[str, str] = {}
        if corrections < _MAX_CORRECTIONS:
            for call in tool_calls:
                fn = call.get("function", {}) or {}
                if fn.get("name") not in buy_tools:
                    continue
                args = _args(call)
                pid, sku = args.get("product_id"), args.get("sku")
                if pid is None or sku is None:
                    continue
                reason = _violation((str(pid), str(sku)), registry, budget, candidates)
                if reason:
                    blocked[call.get("id", "")] = reason

        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        })

        if blocked:
            corrections += 1
            options = _affordable(registry, budget, candidates)
            for call in tool_calls:
                cid = call.get("id", "")
                reason = blocked.get(cid)
                content = (
                    {"rejected": True, "reason": reason,
                     "valid_options": options} if reason else
                    {"skipped": "not sent; fix the rejected purchase first"}
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": json.dumps(content),
                })
            dialogue.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "guard_rejected": list(blocked.values()),
            })
            continue  # do not spend an environment call on an invalid purchase

        # ---- enrich purchases with verified claims, then send ----
        actions = []
        for index, call in enumerate(tool_calls, start=1):
            fn = call.get("function", {}) or {}
            args = _args(call)
            if fn.get("name") in roles["message"]:
                pending_reply = None  # the shopper has been answered
            name = fn.get("name", "")
            if name in buy_tools:
                pid, sku = args.get("product_id"), args.get("sku")
                # Only attach a justification to a tool whose schema declares one
                # (place_test_order does; add_to_cart does NOT — sending it there
                # fails schema validation and the environment rejects the call).
                if pid is not None and sku is not None and "justification" in tool_params.get(name, set()):
                    just = args.get("justification")
                    existing = just.get("claims") if isinstance(just, dict) else None
                    merged = _repair_claims((str(pid), str(sku)), registry, existing)
                    if merged:
                        args["justification"] = ({**just, "claims": merged}
                                                 if isinstance(just, dict) else {"claims": merged})
                if name in roles['purchase'] or not roles['purchase']:
                    ordered = True
            actions.append({
                "call_id": f"{problem_id}-turn-{turn}-{index}",
                "action": {"name": fn.get("name", ""), "args": args},
            })

        env_turn += 1
        group_id = f"{problem_id}-turn-{env_turn}"
        for _i, _a in enumerate(actions, start=1):
            _a["call_id"] = f"{group_id}-{_i}"
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": env_turn,
            "calls": actions,
        }
        result = _post("/environment/call", envelope)
        if not result or "calls" not in result:
            break

        dialogue.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
            "environment_result": result,
        })

        calls_out = result.get("calls") or []
        for index, call in enumerate(tool_calls):
            observation = calls_out[index]["observation"] if index < len(calls_out) else {}
            _harvest(observation, registry)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", f"{group_id}-{index + 1}"),
                "content": _compact(observation),
            })
        user_message = result.get("user_message")
        if user_message and user_message.get("content"):
            content = str(user_message["content"])
            messages.append({"role": "user", "content": content})
            pending_reply = content
            reply_prompts = 0
            # The shopper can TIGHTEN the budget mid-task (add_constraint); the family
            # gate is `budget_after_intervention`, so the guard must track the newest
            # cap, not the one parsed from the opening request.
            # Only demand a reply when the shopper actually asks something. Replying
            # makes the shopper reply again, so without this (and the cap below) the
            # agent ping-pongs in conversation and never shops.
            asks = ("?" in content) or any(w in content.lower() for w in (
                "which", "what", "confirm", "tell me", "let me know", "are you",
                "did you", "can you", "please", "make sure", "instead", "actually"))
            if not asks or forced_replies >= 2:
                pending_reply = None
            new_budget = _parse_budget(content)
            if new_budget is not None and (budget is None or new_budget != budget):
                budget = new_budget
                messages.append({"role": "user", "content": (
                    f"CONSTRAINT UPDATE: the binding price cap is now {budget}. Any order "
                    "you place must be at or under it. If your current pick exceeds it, "
                    "switch to a cheaper option that still meets every other requirement.")})

        if any((c.get("observation") or {}).get("done") for c in calls_out):
            break

    return dialogue


def _args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function", {}) or {}).get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _compact(obj: Any, limit: int = 6000) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(obj)
    if len(text) > limit:
        text = text[:limit] + '…"[truncated]"'
    return text


__all__ = ["agent_main"]
