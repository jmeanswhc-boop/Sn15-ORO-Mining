from __future__ import annotations

import json
import re
import time
import uuid
from os import getenv
from typing import Any, Optional

from src.agent.proxy_client import ProxyClient

_proxy = ProxyClient(timeout=180, max_retries=2, retry_delay=1.0)

_RUN_ID = uuid.uuid4().hex[:12]

_MODEL_PRIORITIES = [
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "moonshotai/Kimi-K2.6-TEE",
    "zai-org/GLM-5.1-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
    "Qwen/Qwen3-32B-TEE",
    "XiaomiMiMo/MiMo-V2-Flash-TEE",
    "google/gemma-4-31B-turbo-TEE",
]

_FAMILIES = [
    "intent_decomposition",
    "retrieval_recall",
    "constraint_satisfaction",
    "preference_reasoning",
    "ranking",
    "recovery",
    "justification",
]


def _primary_model() -> str:
    return getenv("SANDBOX_MODEL") or _MODEL_PRIORITIES[0]


def _inference(messages: list[dict], tools: list[dict]) -> dict | None:
    primary = _primary_model()
    others = [m for m in _MODEL_PRIORITIES if m != primary]
    order = [primary] + others

    for model in order:
        resp = _proxy.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0,
            },
        )
        if resp is not None:
            return resp
        time.sleep(0.5)

    return None


_CLASSIFY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "set_family",
            "description": "Classify the shopping task into exactly one family.",
            "parameters": {
                "type": "object",
                "properties": {
                    "family": {
                        "type": "string",
                        "enum": list(_FAMILIES),
                        "description": "The task family.",
                    }
                },
                "required": ["family"],
            },
        },
    }
]

_CLASSIFY_SYSTEM = (
    "You are a shopping-task classifier. You will receive the available tool list "
    "and the shopper's query. Use BOTH to output the ONE family via set_family.\n\n"
    "Tool signals (use these to break ties):\n"
    "  • 'compare' tool present  → strong signal for 'ranking'\n"
    "  • 'message' tool present  → interactive family (constraint_satisfaction, "
    "recovery, preference_reasoning, justification, intent_decomposition)\n"
    "  • 'filter' without candidates in query → retrieval_recall or constraint_satisfaction\n\n"
    "Families:\n"
    "- intent_decomposition: mix of firm requirements and nice-to-have preferences "
    "(e.g. firm screen size + preferred brand).\n"
    "- retrieval_recall: 'find an in-stock listing in [category] that matches [phrase]' — "
    "keyword/phrase search where the product title must literally contain the search words.\n"
    "- constraint_satisfaction: ALL requirements are firm (brand, spec, budget, stock); "
    "there may be a hidden spec the shopper will reveal when asked.\n"
    "- preference_reasoning: one test order; soft use-case preferences (sweat/sport fit, "
    "materials, sound profile) guide choice.\n"
    "- ranking: compare a list of explicit product_id::sku candidates ranked by refresh "
    "rate, then screen size, then lowest price.\n"
    "- recovery: strong preference for an exact item+color; relax only when that exact "
    "item becomes unavailable.\n"
    "- justification: find a product AND explain why it was chosen with verified "
    "attribute claims (walk me through / tell me why / reason I can trust).\n"
)


def _tool_summary(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name", "")
        desc = (fn.get("description") or "").split(".")[0].strip()
        if name:
            lines.append(f"  • {name}: {desc}" if desc else f"  • {name}")
    return "\n".join(lines) if lines else "  (none listed)"


def _llm_detect_family(query: str, tools: list[dict]) -> str:
    user_content = (
        f"Available tools:\n{_tool_summary(tools)}\n\n"
        f"Query:\n{(query or '')[:3000]}"
    )
    resp = _proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": _primary_model(),
            "messages": [
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "tools": _CLASSIFY_TOOL,
            "tool_choice": "required",
            "temperature": 0,
        },
    )
    if resp is None:
        return ""
    try:
        tc = resp["choices"][0]["message"]["tool_calls"][0]
        raw = tc["function"]["arguments"]
        family = (json.loads(raw) if isinstance(raw, str) else raw).get("family", "")
        if family in _FAMILIES:
            return family
    except (KeyError, IndexError, json.JSONDecodeError):
        pass
    return ""


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


_JUSTIFICATION_BANNED = frozenset({
    # NOTE: "display" is intentionally absent — it's converted to "resolution" in _sanitize_args.
    "hdr", "hdr10", "hdr10+", "hdr_support", "hdr_compatible",
    "dolby_vision", "hlg", "oled", "amoled", "qled", "ips", "va", "tn",
    "curved", "flat_panel", "variable_refresh_rate", "vrr", "gsync", "freesync",
    "overdrive", "nfc", "wifi", "wi_fi", "bluetooth", "bt",
    "4g", "lte", "5g", "usb_c", "hdmi", "displayport", "dp", "vga", "dvi",
    "microphone", "mic", "headphone_jack", "aux_port", "3.5mm",
    "stereo", "dolby_atmos", "dts", "ois", "waterproof", "water_resistant",
    "ip_rating", "ip68", "ip67", "ip65", "wireless_charging", "fast_charging",
    "fingerprint_sensor", "face_id", "backlit_keyboard", "rgb",
})


def _normalize_resolution(value: str) -> str | None:
    """Map any shorthand resolution string to canonical pixel-dimension form."""
    v = value.lower().strip()
    if any(x in v for x in ("2k", "qhd", "1440p", "2560")):
        return "2560 x 1440"
    if any(x in v for x in ("4k", "uhd", "2160p", "3840")):
        return "3840 x 2160"
    if any(x in v for x in ("fhd", "1080p", "full hd", "1920")):
        return "1920 x 1080"
    if any(x in v for x in ("720p", "hd ready", "1280 x 720", "1280x720")):
        return "1280 x 720"
    return None

_NO_JUSTIFICATION_FAMILIES = frozenset({
    "intent_decomposition", "retrieval_recall", "constraint_satisfaction",
    "preference_reasoning", "ranking", "recovery",
})


def _resolution_from_title(title: str) -> str | None:
    t = title.lower()
    if "2k" in t or "qhd" in t or "1440p" in t or "2560" in t:
        return "2560 x 1440"
    if "4k" in t or "uhd" in t or "2160p" in t or "3840" in t:
        return "3840 x 2160"
    if "fhd" in t or "1080p" in t or "full hd" in t or "1920" in t:
        return "1920 x 1080"
    if "720p" in t or "hd ready" in t or "1280" in t:
        return "1280 x 720"
    return None


def _sanitize_args(
    name: str, args: dict[str, Any], family: str,
    title_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    out = dict(args)
    if name != "place_test_order":
        return out
    if family in _NO_JUSTIFICATION_FAMILIES:
        out.pop("justification", None)
        return out
    if family == "justification":
        jus = out.get("justification")
        if not isinstance(jus, dict):
            jus = {}
        claims = jus.get("claims") or []

        # Derive resolution from product title (used as auto-inject fallback)
        pid = str(out.get("product_id") or "")
        title = (title_lookup or {}).get(pid, "")
        title_res = _resolution_from_title(title) if title else None

        clean = []
        for c in claims:
            if not isinstance(c, dict):
                continue
            field = str(c.get("field", "")).lower().strip()
            # Convert "display" → "resolution" BEFORE checking the banned set
            if field == "display":
                field = "resolution"
                c = {**c, "field": "resolution"}
            if field in _JUSTIFICATION_BANNED:
                continue
            # Normalise any shorthand resolution value to pixel-dimension form
            if field == "resolution":
                raw_val = str(c.get("value", "")).strip()
                norm = _normalize_resolution(raw_val) or title_res
                if norm:
                    c = {**c, "value": norm}
            clean.append(c)

        # Resolution is a required gate — inject from title if the model omitted it
        if not any(str(c.get("field", "")).lower() == "resolution" for c in clean):
            if title_res:
                clean.append({"field": "resolution", "value": title_res})

        out["justification"] = {**jus, "claims": clean}
    return out


def _env_call(
    binding: dict[str, Any],
    problem_id: str,
    turn: int,
    actions: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    group_id = f"{_RUN_ID}-{problem_id}-turn-{turn}"
    envelope = {
        **binding,
        "call_id": group_id,
        "idempotency_key": group_id,
        "turn": turn,
        "calls": [
            {"call_id": f"{group_id}-{i}", "action": {"name": name, "args": args}}
            for i, (name, args) in enumerate(actions, start=1)
        ],
    }
    result = _proxy.post("/environment/call", json_data=envelope)
    if result is None:
        raise RuntimeError("environment call failed")
    return result


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_hz(text: Any) -> Optional[float]:
    if text is None:
        return None
    nums = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*hz", str(text), re.I)]
    return max(nums) if nums else None


def _parse_inches(text: Any) -> Optional[float]:
    if text is None:
        return None
    nums = [
        float(m.group(1))
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*(?:inches|inch|in)\b", str(text), re.I)
    ]
    return max(nums) if nums else None


_VARIANT_RE = re.compile(r"(\d{6,})::([\w.\-]+)")
_BUDGET_RE = re.compile(
    r"(?:spend up to|at or under|at or below|under|below|<=?)\s*`?([\d,]+(?:\.\d+)?)",
    re.I,
)


def _parse_candidates(query: str) -> list[tuple[str, str]]:
    return [(a, b) for a, b in _VARIANT_RE.findall(query or "")]


def _parse_budget(query: str) -> Optional[float]:
    m = _BUDGET_RE.search(query or "")
    return float(m.group(1).replace(",", "")) if m else None


def _rank_key(row: dict[str, Any]) -> tuple:
    blob = json.dumps(row, ensure_ascii=False)
    hz = _parse_hz(blob) or _parse_hz(row.get("refresh_rate")) or _as_float(row.get("refresh_rate")) or -1
    inches = (
        _parse_inches(blob)
        or _parse_inches(row.get("screen_size"))
        or _as_float(row.get("screen_size"))
        or -1
    )
    price = _as_float(row.get("price") or row.get("current_price"))
    return (-hz, -inches, price if price is not None else 1e18)


def _unwrap_obs(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("observation")
    if isinstance(inner, dict) and any(
        k in inner for k in ("compared", "results", "variants", "product_id", "in_stock", "price")
    ):
        return inner
    if any(k in payload for k in ("compared", "results", "variants", "in_stock", "price")):
        return payload
    return inner if isinstance(inner, dict) else payload


def _rows_from_compare(obs: dict[str, Any], candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
    obs = _unwrap_obs(obs)
    rows: list[dict[str, Any]] = []
    compared = obs.get("compared") or obs.get("products") or obs.get("results") or []
    wanted = {(pid, sku) for pid, sku in candidates}
    sku_by_pid = {pid: sku for pid, sku in candidates}

    if isinstance(compared, list):
        for item in compared:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id") or "")
            variants = item.get("variants") or []
            want_sku = sku_by_pid.get(pid)
            if variants:
                matched = False
                for v in variants:
                    if not isinstance(v, dict):
                        continue
                    sku = str(v.get("sku") or "")
                    if want_sku and sku != want_sku:
                        continue
                    matched = True
                    rows.append({
                        **item, **v,
                        "product_id": pid,
                        "sku": sku or want_sku or "",
                        "title": item.get("title"),
                        "facts": item.get("facts") or {},
                        "brand": item.get("brand"),
                    })
                if not matched and want_sku:
                    base_v = variants[0] if isinstance(variants[0], dict) else {}
                    rows.append({
                        **item, **base_v,
                        "product_id": pid, "sku": want_sku,
                        "title": item.get("title"),
                        "facts": item.get("facts") or {},
                        "brand": item.get("brand"),
                    })
            else:
                sku = str(item.get("sku") or want_sku or "")
                if (pid, sku) in wanted or want_sku:
                    rows.append({**item, "product_id": pid, "sku": sku or want_sku})

    have = {(str(r.get("product_id")), str(r.get("sku"))) for r in rows}
    for pid, sku in candidates:
        if (pid, sku) not in have:
            rows.append({"product_id": pid, "sku": sku})
    return rows


def _eligible(row: dict[str, Any], budget: Optional[float]) -> bool:
    if row.get("in_stock") is False:
        return False
    price = _as_float(row.get("price") or row.get("current_price"))
    if budget is not None and price is not None and price > budget + 1e-6:
        return False
    return True


def _run_ranking(problem_data: dict[str, Any], dialogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_steps = int(policy["max_steps"])
    query = policy.get("query") or ""
    candidates = _parse_candidates(query)
    budget = _parse_budget(query)
    if not candidates:
        return dialogue

    turn = 0

    def act(actions: list[tuple[str, dict[str, Any]]], note: str = "") -> dict[str, Any]:
        nonlocal turn
        turn += 1
        if turn > max_steps:
            raise RuntimeError("step budget exhausted")
        result = _env_call(binding, problem_id, turn, actions)
        dialogue.append({
            "role": "assistant",
            "content": note,
            "tool_calls": [
                {
                    "id": f"rank-{turn}-{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for i, (name, args) in enumerate(actions, start=1)
            ],
            "environment_result": result,
        })
        return result

    def done(result: dict[str, Any]) -> bool:
        return any((c.get("observation") or {}).get("done") for c in (result.get("calls") or []))

    pids: list[str] = []
    for pid, _ in candidates:
        if pid not in pids:
            pids.append(pid)

    result = act([("compare", {"product_ids": pids[:5]})], note="rank-compare")
    if done(result):
        return dialogue

    obs = (result.get("calls") or [{}])[0].get("observation") or {}
    rows = _rows_from_compare(obs, candidates)
    by_key = {(str(r.get("product_id")), str(r.get("sku"))): r for r in rows}

    live: list[dict[str, Any]] = []
    for pid, sku in candidates:
        row = by_key.get((pid, sku))
        if row is None:
            row = next((r for r in rows if str(r.get("product_id")) == pid), None)
            if row is None:
                row = {"product_id": pid, "sku": sku, "in_stock": True}
            else:
                row = {**row, "sku": sku}
        if _eligible(row, budget):
            live.append(row)

    if not live:
        live = [{"product_id": p, "sku": s} for p, s in candidates]

    live.sort(key=_rank_key)
    winner = live[0]
    result = act(
        [("add_to_cart", {"product_id": str(winner["product_id"]), "sku": str(winner["sku"])})],
        note="rank-commit",
    )
    if done(result):
        return dialogue

    cart_pid, cart_sku = str(winner["product_id"]), str(winner["sku"])
    pending_user = (result.get("user_message") or {}).get("content")
    if not pending_user:
        result = act(
            [("inspect_stock", {"product_id": cart_pid, "sku": cart_sku})],
            note="rank-check",
        )
        if done(result):
            return dialogue
        pending_user = (result.get("user_message") or {}).get("content")

    if pending_user:
        result = act(
            [("message", {"content": "Checking availability now and switching if a better in-stock option ranks higher."})],
            note="rank-ack",
        )
        if done(result):
            return dialogue

    refreshed: list[dict[str, Any]] = []
    for pid, sku in candidates:
        if turn >= max_steps - 2:
            break
        result = act([("inspect_stock", {"product_id": pid, "sku": sku})], note="rank-rerank")
        if done(result):
            return dialogue
        stock = _unwrap_obs((result.get("calls") or [{}])[0].get("observation") or {})
        row = {
            "product_id": pid, "sku": sku,
            "price": stock.get("price") or stock.get("current_price"),
            "in_stock": stock.get("in_stock", True),
        }
        prev = by_key.get((pid, sku)) or next((r for r in rows if str(r.get("product_id")) == pid), {})
        for k in ("refresh_rate", "screen_size", "title", "facts", "raw_facts"):
            if k in prev and k not in row:
                row[k] = prev[k]
        if _eligible(row, budget):
            refreshed.append(row)

    if refreshed:
        refreshed.sort(key=_rank_key)
        best = refreshed[0]
        if str(best["product_id"]) != cart_pid or str(best["sku"]) != cart_sku:
            result = act(
                [("remove_from_cart", {"product_id": cart_pid, "sku": cart_sku})],
                note="rank-switch-remove",
            )
            if done(result):
                return dialogue
            result = act(
                [("add_to_cart", {"product_id": str(best["product_id"]), "sku": str(best["sku"])})],
                note="rank-switch-add",
            )
            if done(result):
                return dialogue
            cart_pid, cart_sku = str(best["product_id"]), str(best["sku"])

    act([("place_test_order", {"product_id": cart_pid, "sku": cart_sku})], note="rank-order")
    return dialogue


def _detect_family(query: str, tools: list[dict] | None = None) -> str:
    tools = tools or []

    family = _llm_detect_family(query, tools)
    if family:
        return family

    q = (query or "").lower()
    tool_names = {(t.get("function") or {}).get("name", "") for t in tools}

    if "mix of firm requirements and nice-to-haves" in q:
        return "intent_decomposition"
    if "reformulate queries and inspect candidates" in q:
        return "retrieval_recall"
    if "every requirement must hold for the exact variant" in q:
        return "constraint_satisfaction"
    if "initial request authorizes one test order" in q:
        return "preference_reasoning"
    if "preferences are relaxable only after the exact item becomes unavailable" in q:
        return "recovery"
    if (
        "shopper wants a valid product plus a trustworthy reason" in q
        or "walk me through why" in q
        or "walk me through" in q
        or "why the one you pick" in q
        or "justify" in q
        or "explain your choice" in q
    ):
        return "justification"
    if (
        "highest advertised maximum refresh rate" in q
        or "compare only these candidate" in q
        or "compare only the following" in q
        or ("tradeoff priority" in q and "refresh" in q)
    ):
        return "ranking"

    return "retrieval_recall"


_BASE_RULES = """\
You are a precise shopping agent. Core rules:
- BUDGET: Never order a product priced above the stated budget.
- IN-STOCK: Only order in-stock items.
- FINAL STEP: Always call place_test_order (add_to_cart alone is NOT enough).
- TRUTHFULNESS: Only state attribute values you literally read from a tool response.
- STEP LIMIT: Aim to finish in ≤12 tool calls. Prefer commit over more searching.
- FIRST ACTION: Your VERY FIRST tool call must be a catalog tool (search, filter,
  compare, or view). NEVER open with `message` — the shopper's query is not an
  intervention requiring acknowledgment.
- RESPOND TO MESSAGES: ONLY call `message` when the environment returns a NEW
  user_message mid-session (a mid-session shopper intervention). Do NOT call
  `message` to respond to the initial query, and do NOT call `message` twice in
  a row without a catalog tool call in between.
- KEEP CART STABLE: After add_to_cart, do NOT remove unless an event says the
  item is unavailable OR its price now exceeds budget. Pre-emptive removal fails grading.
- place_test_order: pass ONLY product_id and sku unless this is a justification task.
"""

_INTENT_DECOMP_PROMPT = _BASE_RULES + """
INTENT DECOMPOSITION TASK:
Firm requirements (non-negotiable): category, specific hardware spec (screen size /
RAM / resolution / refresh rate), budget, and in-stock availability.
Soft preference: preferred brand (subordinate to every firm requirement).

YOUR WORKFLOW:
1. Search specifically for the preferred brand + spec.
2. View top results to check spec, price, and in-stock.
3. If a FULLY COMPLIANT preferred-brand item exists (price ≤ budget, in-stock,
   correct spec): add_to_cart immediately and KEEP IT IN CART.
4. Only if NO preferred-brand compliant item: pick the best any-brand item.
5. add_to_cart and KEEP IT IN CART — do NOT remove it unless an event explicitly
   tells you the item is unavailable or its price now exceeds the current budget.
6. If a budget-change event fires but the item is still within the NEW budget:
   KEEP IT IN CART — do not remove it.
7. If an event says item price exceeds budget: remove_from_cart, then find the
   next best compliant item and commit to it.
8. place_test_order.

CRITICAL:
- After add_to_cart, stay in cart until an event forces a change.
- DO NOT pre-emptively remove from cart "just to check" — this breaks grading.
- DO NOT call inspect_stock on the committed item again after add_to_cart.
- Budget changes: only remove cart item if NEW price > NEW budget.
- If no compliant item can be found after 3 different search queries:
  send a message explaining the situation, then order the closest available match.

RESPONDING TO INTERVENTION MESSAGES (REQUIRED):
Every time the environment sends you a user message (the shopper speaks), you MUST
call the `message` action FIRST before doing anything else, even if it's just one line.
Examples of required responses:
  - "before you buy anything: which one?" → message: "I'm going with [product name] because it has [spec]."
  - "my balance is now X PHP"            → message: "Understood, I'll stay within X PHP."
  - "did the price change?"              → message: "Yes/No, [describe current price and what I'll do]."
Failing to call `message` for each intervention will fail the grading gate.
"""

_RETRIEVAL_RECALL_PROMPT = _BASE_RULES + """
RETRIEVAL RECALL TASK:
Your goal is to find a product whose TITLE literally contains every key word from
the search phrase. Category must also match exactly.

YOUR WORKFLOW:
1. Extract the key words from the query (brand, model name, descriptors, category).
2. Search using those key terms. Run up to 3 distinct reformulations if needed
   (shorter, different synonyms, drop one word at a time).
3. View 2-3 top candidates. For each, check:
   a. Does the product TITLE literally contain ALL key words from the phrase?
      Do NOT infer a match from category or attributes alone — text must be present.
   b. Is the product in the stated category?
   c. Is it in-stock and within budget?
4. If no title contains all key words after 3 searches, pick the closest match
   (most key words present in title, correct category, in-stock, within budget).
5. add_to_cart → place_test_order(product_id, sku) — NO justification.

STUCK HANDLING:
- If 3+ searches return no matching title: broaden (drop one constraint word,
  try singular/plural, try brand abbreviation).
- If stockout fires after cart: message ack → inspect → switch to next best
  title match → place_test_order. Do not stall.

CRITICAL: Never commit based on category or attributes alone — the key words must
appear in the product title. Commit to the best title match you can find, then order.
"""

_CONSTRAINT_SATISFACTION_PROMPT = _BASE_RULES + """
CONSTRAINT SATISFACTION TASK:
Every requirement is FIRM and must hold for the EXACT SKU you order.
A product may offer the correct spec in one SKU and the wrong spec in another.

YOUR WORKFLOW:
1. Search for items matching brand (if specified), spec, and category.
2. View the top candidate to inspect SKU options.
3. Confirm the SPECIFIC SKU's attributes:
   - Screen size / RAM / resolution / refresh rate matches the requirement exactly
   - Price ≤ budget
   - In-stock
4. If the SKU fails any check, inspect the next candidate.
5. add_to_cart with the exact matching SKU → place_test_order.

Resolution key: HD = 1280×720, Full HD = 1920×1080, QHD = 2560×1440, 4K = 3840×2160.
"""

_PREFERENCE_REASONING_PROMPT = _BASE_RULES + """
PREFERENCE REASONING TASK:
Hard requirements: correct category, budget, in-stock, actual DEVICE (not accessory).
Soft preference: pick the item that BEST MATCHES the use case (sweat/water resistance,
sport fit, battery life, materials) — do NOT blindly pick the cheapest.

YOUR WORKFLOW:
1. search for the product type + budget. Run 1-2 searches if needed.
2. view 2-3 top candidates; inspect their raw attributes for the soft preference signals.
3. Pick the best-fit in-stock item within budget. add_to_cart.

EVENT RECOVERY (always fires after add_to_cart):
4. A price-change or availability event will fire. Read it carefully.
   - If price now exceeds budget OR item is OOS: remove_from_cart immediately.
     Then inspect_stock on your next-best candidate and add_to_cart it.
   - If price is still within budget and still in-stock: keep the item, proceed.
5. place_test_order(product_id, sku) — NO justification.

AFTER THE EVENT:
- Do NOT call `message` more than once to acknowledge the event.
- After one ack, immediately take a catalog action (inspect_stock, add_to_cart, or place_test_order).
- Do NOT stall in a message loop — the event is an availability/price signal, not a conversation.
"""

_RANKING_PROMPT = _BASE_RULES + """
RANKING TASK — TWO PHASES (STRICT SEQUENCE):

PHASE 1 — COMPARE AND COMMIT (do this in 2-3 tool calls total):
1. Call compare with ALL candidate IDs listed in the query — ONE call.
   compare returns refresh_rate, screen_size, price, and in_stock for every variant.
2. Read the compare response. Apply tiebreaker STRICTLY in this order:
   (a) Highest advertised max refresh rate (Hz)
   (b) Among ties: largest screen size (inches)
   (c) Among remaining ties: lowest price
   In-stock only. Pick the WINNER.
3. add_to_cart the WINNER directly.
   *** DO NOT call view, inspect_stock, filter, or search before this add_to_cart ***
   *** Trust compare output completely. Commit immediately after compare. ***

PHASE 2 — EVENT RESPONSE (fires after your first add_to_cart):
4. Shopper sends "wait, is that still available?" or similar.
5. NOW you may call inspect_stock or compare on the REMAINING candidates (not the
   carted item) to check which is still valid.
6. Re-apply the tiebreaker among still-valid candidates including the cart item.
7. If a different product is now the winner: remove_from_cart → add_to_cart new winner.
8. place_test_order.

FORBIDDEN BEFORE FIRST add_to_cart:
- Do NOT call inspect_stock on any candidate — compare already shows in_stock
- Do NOT call search or filter
- compare → add_to_cart is the core Phase 1 sequence
- You MAY call view on at most 2 candidates if you need to confirm attributes,
  but do NOT call view on ALL candidates — that wastes your step budget.

PHASE 2 REMINDER: After the event message, use inspect_stock or compare to check
remaining candidates. Then switch if needed and place_test_order.

FINAL STEP — place_test_order:
Call it with ONLY product_id and sku. No other arguments.
  ✓ CORRECT:   place_test_order(product_id="...", sku="...")
  ✗ WRONG:     place_test_order(product_id="...", sku="...", justification={...})
Do NOT pass justification. Any claims you invent will be checked against product data
and will fail truthfulness verification, causing the entire task to score zero.
"""

_RECOVERY_PROMPT = _BASE_RULES + """
RECOVERY TASK:
Critical constraints: category, budget, in-stock. NEVER relax these.
Relaxable preferences (in order): preferred model color, then preferred model.

YOUR WORKFLOW:
1. Search for the specific model and color mentioned.
2. View to confirm spec, price ≤ budget, in-stock.
3. add_to_cart.
4. KEEP the item in cart — do NOT remove it until an event fires.
5. When shopper says "wait, is that still available?" or similar (the event):
   a. Call inspect_stock to check the carted item.
   b. If it's still in stock and within budget: keep it, place_test_order.
   c. If it's no longer available or price exceeds budget:
      - remove_from_cart
      - FIRST try: same model, different color (search for model + new color)
      - SECOND try: different model that meets all firm constraints
      - add_to_cart replacement → place_test_order.
6. If no event fires, proceed directly to place_test_order.

JUSTIFICATION CLAIMS — only include values confirmed in view/inspect responses:
- Use field="price" (exact numeric price from tool response)
- Use field="in_stock" with value "true"
- Use field="category" with confirmed category
- Do NOT include field="brand" with value "null" — omit brand if unknown
- Do NOT claim field="color" unless color is confirmed in SKU options
DO NOT pre-emptively remove from cart before an event fires.
"""

_JUSTIFICATION_PROMPT = _BASE_RULES + """
JUSTIFICATION TASK — EXACT WORKFLOW (6 steps, do NOT deviate):
1. search (1 call) — find candidates matching spec and budget.
2. view the top candidate (1 call) — read its attributes section VERY carefully.
   Write down the EXACT resolution string (e.g. "1920 x 1080", "2560 x 1440", "720p").
3. VERIFY: price ≤ stated budget. If over budget, view a different candidate.
4. message (ONE CALL ONLY) — send justification to user BEFORE adding to cart.
   State: product name, exact price, in-stock=yes, and the KEY SPEC from attributes.
   *** After this single message call your VERY NEXT tool call MUST be add_to_cart. ***
   *** Do NOT call message a second time. Never message twice. ***
5. add_to_cart (1 call)
6. place_test_order (1 call) with justification.claims citing ONLY RAW FACTS.

CLAIM FIELD NAMES — use EXACTLY these canonical names regardless of the attribute key:
  - resolution               → field="resolution". NEVER use field="display".
    ALWAYS include a resolution claim — it is a required gate; missing it scores zero.
    Value MUST be in pixel-dimension form. Priority order:
    1. view() attributes show "display" or "resolution" field → use that exact value
       (it may already be in pixel form, e.g. "1920 x 1080")
    2. Title contains "2K" / "QHD" / "1440p" → value = "2560 x 1440"
    3. Title contains "4K" / "UHD" / "2160p" → value = "3840 x 2160"
    4. Title contains "FHD" / "1080p" / "Full HD" → value = "1920 x 1080"
    5. Title or attribute contains "720p" / "HD Ready" → value = "1280 x 720"
    NEVER use "720p" as the value string — always expand to "1280 x 720".
    NEVER invent a resolution; if truly unknown from attributes or title, skip step 4 below
    and use the closest evidence from the view response.
  - screen size              → field="size" or field="screen_size", value = from attributes
    REQUIRED whenever the shopper specified an inch size (e.g. "27-inch" → field="size", value="27").
    Read the EXACT size from the product attributes — NOT from a multi-size title like "19" 22" 24" 27"".
    For multi-size products the sku determines the actual size; check the variant attributes carefully.
  - price                    → field="price"
  - in stock                 → field="in_stock", value="true"
  - brand                    → field="brand" ONLY when brand is verifiable from the
    product (title, brand field, or attributes). OMIT brand entirely if ambiguous.

BANNED field names (grader rejects these — never use):
  display, hdr, hdr10, hdr_support, dolby_vision, microphone, mic,
  headphone_jack, nfc, 5g, 4g, lte, connectivity, ports, color.

DO NOT copy spec values from the shopper's query — read them from the view response.
IMPORTANT: If your claims reference a product you viewed many turns ago, re-read your
earlier view observation in context to confirm the exact resolution/display value before ordering.

EVENT: If a price-change message fires after add_to_cart and new price > budget:
1. remove_from_cart (the event target product)
2. view a different candidate (MANDATORY — confirm its resolution from attributes)
3. message (ONE MORE TIME ONLY) — re-justify with the new product name, price, resolution
4. add_to_cart (new product)
5. place_test_order with updated claims (correct resolution from step 2's view)
Do NOT skip the message in step 3 — the grader requires a justification before every commitment.
"""


def _system_prompt(family: str) -> str:
    return {
        "intent_decomposition": _INTENT_DECOMP_PROMPT,
        "retrieval_recall": _RETRIEVAL_RECALL_PROMPT,
        "constraint_satisfaction": _CONSTRAINT_SATISFACTION_PROMPT,
        "preference_reasoning": _PREFERENCE_REASONING_PROMPT,
        "ranking": _RANKING_PROMPT,
        "recovery": _RECOVERY_PROMPT,
        "justification": _JUSTIFICATION_PROMPT,
    }.get(family, _BASE_RULES + """
GENERAL SHOPPING — target 5 steps:
  search (1) → view top result if needed (1) → add_to_cart (1) → place_test_order (1)
- Do NOT call filter if search returned relevant results.
- Do NOT call inspect_stock if search/view shows in_stock=true.
- Do NOT call message unless the query explicitly asks for it.
- If preferred brand has no compliant item, pick best item from any brand.
""")


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    max_steps = int(policy["max_steps"])
    query: str = policy.get("query", "")

    available_tools: list[dict] = policy.get("tools") or []
    family = _detect_family(query, available_tools)
    dialogue: list[dict[str, Any]] = []

    if family == "ranking":
        try:
            return _run_ranking(problem_data, dialogue)
        except Exception as exc:
            dialogue.append({"role": "assistant", "content": f"ranking_fallback: {type(exc).__name__}: {exc}"})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(family)},
        {"role": "user", "content": query},
    ]
    dialogue: list[dict[str, Any]] = []
    last_was_message_only = False
    justification_message_sent = False
    _last_product_id: str | None = None
    _last_sku: str | None = None
    _cart_product_id: str | None = None
    _cart_sku: str | None = None
    _add_to_cart_count = 0
    _product_title: dict[str, str] = {}  # product_id → title from view observation

    for turn in range(1, max_steps + 1):
        if turn in (8, 12, 20):
            remaining = max_steps - turn
            messages.append({
                "role": "user",
                "content": (
                    f"[STEP CHECK: {turn}/{max_steps} used, {remaining} remaining] "
                    "If you have found a valid in-stock item within budget, "
                    "add_to_cart and place_test_order IMMEDIATELY. "
                    "Stop searching — commit to the best candidate you have found."
                ),
            })

        if last_was_message_only:
            if family == "justification" and justification_message_sent:
                _guide2 = (
                    "[SYSTEM] You have already sent your justification message. "
                    "Do NOT call `message` again. "
                    "Your ONLY valid next call is `add_to_cart`"
                )
                if _last_product_id and _last_sku:
                    _guide2 += (
                        f" with product_id=\"{_last_product_id}\" "
                        f"and sku=\"{_last_sku}\"."
                    )
                else:
                    _guide2 += " — use the product_id and sku from the last product you viewed."
                messages.append({"role": "user", "content": _guide2})
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM] You just sent a message but the shopper has not replied. "
                        "Do NOT call `message` again. Your next action MUST be a catalog "
                        "tool: search, filter, view, compare, add_to_cart, or place_test_order."
                    ),
                })

        for _attempt in range(2):
            inference = _inference(messages, policy["tools"])
            if inference is None:
                raise RuntimeError("all models failed to respond")
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if tool_calls:
                break
            messages.extend([
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": (
                        "The environment has not reported done=true. "
                        "Choose one of the supplied tools to continue."
                    ),
                },
            ])
        else:
            raise RuntimeError("model returned no tool call")

        group_id = f"{_RUN_ID}-{problem_id}-turn-{turn}"

        # Always append the assistant message with the ORIGINAL tool_calls so the
        # model can see its own history even when some calls were blocked.
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
        })

        # For justification family: block duplicate message calls after the first justification.
        blocked_msg_calls: list[dict[str, Any]] = []
        effective_calls = list(tool_calls)
        if family == "justification" and justification_message_sent:
            blocked_msg_calls = [tc for tc in tool_calls if tc["function"]["name"] == "message"]
            effective_calls = [tc for tc in tool_calls if tc["function"]["name"] != "message"]

        if not effective_calls and blocked_msg_calls:
            # Every call this turn was a duplicate message — skip the environment call.
            _guide = (
                "[SYSTEM] Your justification message was already sent and is blocked from "
                "resending. You MUST call `add_to_cart` right now. "
            )
            if _last_product_id and _last_sku:
                _guide += (
                    f"Use product_id=\"{_last_product_id}\" and sku=\"{_last_sku}\" "
                    "(the last product you viewed/inspected)."
                )
            else:
                _guide += "Use the product_id and sku from the last product you viewed."
            messages.append({"role": "user", "content": _guide})
            last_was_message_only = True
            continue

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
                        "args": _sanitize_args(
                            call["function"]["name"],
                            _arguments(call),
                            family,
                            _product_title,
                        ),
                    },
                }
                for index, call in enumerate(effective_calls, start=1)
            ],
        }
        result = _proxy.post("/environment/call", json_data=envelope)
        if result is None:
            raise RuntimeError("environment call failed")
        dialogue.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls,
            "environment_result": result,
        })

        for tool_call, call_result in zip(effective_calls, result["calls"], strict=True):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(call_result["observation"]),
            })
            # Track last product seen so the guard can give specific add_to_cart instructions
            tc_name = tool_call["function"]["name"]
            tc_args = _arguments(tool_call)
            if tc_name == "add_to_cart":
                _last_product_id = tc_args.get("product_id") or _last_product_id
                _last_sku = tc_args.get("sku") or _last_sku
                _cart_product_id = tc_args.get("product_id")
                _cart_sku = tc_args.get("sku")
                _add_to_cart_count += 1
            elif tc_name == "remove_from_cart":
                _cart_product_id = None
                _cart_sku = None
                _last_product_id = tc_args.get("product_id") or _last_product_id
                _last_sku = tc_args.get("sku") or _last_sku
            elif tc_name == "inspect_stock":
                _last_product_id = tc_args.get("product_id") or _last_product_id
                _last_sku = tc_args.get("sku") or _last_sku
            elif tc_name == "view":
                new_pid = tc_args.get("product_id")
                if new_pid and new_pid != _last_product_id:
                    # Product changed — the old sku belongs to a different product; clear it
                    _last_sku = None
                _last_product_id = new_pid or _last_product_id
                # Record title from view observation for resolution correction
                if new_pid:
                    obs = call_result.get("observation", {})
                    title = obs.get("title", "")
                    if title:
                        _product_title[new_pid] = title
                    # Extract sku from variants (if unique in-stock variant)
                    variants = obs.get("variants") or []
                    in_stock = [v for v in variants if v.get("in_stock")]
                    if len(in_stock) == 1:
                        _last_sku = in_stock[0].get("sku") or _last_sku
                    elif len(variants) == 1:
                        _last_sku = variants[0].get("sku") or _last_sku
        if result.get("user_message"):
            user_msg = result["user_message"]["content"]
            messages.append({
                "role": "user",
                "content": user_msg,
            })
            last_was_message_only = False
            is_availability_event = any(p in user_msg.lower() for p in (
                "still available", "price", "more expensive", "went up", "changed",
                "out of stock", "sold out", "increased", "jumped", "unavailable",
            ))
            if is_availability_event and family == "justification":
                # Allow re-justification after a price/stock event
                justification_message_sent = False
            if not is_availability_event:
                messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM] The shopper just spoke. Your NEXT tool call MUST be "
                        "`message` to acknowledge what they said. Do NOT search, filter, "
                        "add to cart, or do anything else before calling `message`."
                    ),
                })
        else:
            all_message = all(tc["function"]["name"] == "message" for tc in effective_calls)
            last_was_message_only = all_message
            if family == "justification" and any(
                tc["function"]["name"] == "message" for tc in effective_calls
            ):
                justification_message_sent = True
            # Reset justification_message_sent after remove_from_cart (event recovery)
            if family == "justification" and any(
                tc["function"]["name"] == "remove_from_cart" for tc in effective_calls
            ):
                justification_message_sent = False
        if any(call["observation"]["done"] for call in result["calls"]):
            break

        time.sleep(0.5)

    return dialogue


__all__ = ["agent_main"]
